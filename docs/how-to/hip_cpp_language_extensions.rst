.. meta::
  :description: This chapter describes the built-in variables and functions that
                are accessible from HIP kernels and HIP's C++ support. It's
                intended for users who are familiar with CUDA kernel syntax and
                want to learn how HIP differs from CUDA.
  :keywords: AMD, ROCm, HIP, CUDA, c++ language extensions, HIP functions

################################################################################
HIP C++ language extensions
################################################################################

HIP extends the C++ language with additional features designed for programming
heterogeneous applications. These extensions mostly relate to the kernel
language, but some can also be applied to host functionality.

********************************************************************************
HIP qualifiers
********************************************************************************

Function-type qualifiers
================================================================================

HIP introduces three different function qualifiers to mark functions for
execution on the device or the host, and also adds new qualifiers to control
inlining of functions.

.. _host_attr:

__host__
--------------------------------------------------------------------------------

The ``__host__`` qualifier is used to specify functions for execution
on the host. This qualifier is implicitly defined for any function where no
``__host__``, ``__device__`` or ``__global__`` qualifier is added, in order to
not break compatibility with existing C++ functions.

You can't combine ``__host__`` with ``__global__``.

__device__
--------------------------------------------------------------------------------

The ``__device__`` qualifier is used to specify functions for execution on the
device. They can only be called from other ``__device__`` functions or from
``__global__`` functions.

You can combine it with the ``__host__`` qualifier and mark functions
``__host__ __device__``. In this case, the function is compiled for the host and
the device. Note that these functions can't use the HIP built-ins (e.g.,
:ref:`threadIdx.x <thread_and_block_idx>` or :ref:`warpSize <warp_size>`), as
they are not available on the host. If you need to use HIP grid coordinate
functions, you can pass the necessary coordinate information as an argument.

__global__
--------------------------------------------------------------------------------

Functions marked ``__global__`` are executed on the device and are referred to
as kernels. Their return type must be ``void``. Kernels have a special launch
mechanism, and have to be launched from the host.

There are some restrictions on the parameters of kernels. Kernels can't:

* have a parameter of type ``std::initializer_list`` or ``va_list``
* have a variable number of arguments
* use references as parameters
* use parameters having different sizes in host and device code, e.g. long double arguments, or structs containing long double members.
* use struct-type arguments which have different layouts in host and device code.

Kernels can have variadic template parameters, but only one parameter pack,
which must be the last item in the template parameter list.

.. note::
    Unlike CUDA, HIP does not support dynamic parallelism, meaning that kernels
    can not be called from the device.

Calling __global__ functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The launch mechanism for kernels differs from standard function calls, as they
need an additional configuration, that specifies the grid and block dimensions
(i.e. the amount of threads to be launched), as well as specifying the amount of
shared memory per block and which stream to execute the kernel on.

Kernels are called using the triple chevron ``<<<>>>`` syntax known from CUDA,
but HIP also supports the ``hipLaunchKernelGGL`` macro.

When using ``hipLaunchKernelGGL``, the first five configuration parameters must
be:

* ``symbol kernelName``: The name of the kernel you want to launch. To support
  template kernels that contain several template parameters separated by use the
  ``HIP_KERNEL_NAME`` macro to wrap the template instantiation
  (:doc:`HIPIFY <hipify:index>` inserts this automatically).
* ``dim3 gridDim``: 3D-grid dimensions that specifies the number of blocks to
  launch.
* ``dim3 blockDim``: 3D-block dimensions that specifies the number of threads in
  each block.
* ``size_t dynamicShared``: The amount of additional shared dynamic memory to
  allocate per block.
* ``hipStream_t``: The stream on which to run the kernel. A value of ``0``
  corresponds to the default stream.

The kernel arguments are listed after the configuration parameters.

.. code-block:: cpp

  #include <hip/hip_runtime.h>
  #include <iostream>

  #define HIP_CHECK(expression)                                \
  {                                                            \
      const hipError_t err = expression;                       \
      if(err != hipSuccess){                                   \
          std::cerr << "HIP error: " << hipGetErrorString(err) \
              << " at " << __LINE__ << "\n";                   \
      }                                                        \
  }

  // Performs a simple initialization of an array with the thread's index variables.
  // This function is only available in device code.
  __device__ void init_array(float * const a, const unsigned int arraySize){
    // globalIdx uniquely identifies a thread in a 1D launch configuration.
    const int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread initializes a single element of the array.
    if(globalIdx < arraySize){
      a[globalIdx] = globalIdx;
    }
  }

  // Rounds a value up to the next multiple.
  // This function is available in host and device code.
  __host__ __device__ constexpr int round_up_to_nearest_multiple(int number, int multiple){
    return (number + multiple - 1)/multiple;
  }

  __global__ void example_kernel(float * const a, const unsigned int N)
  {
    // Initialize array.
    init_array(a, N);
    // Perform additional work:
    // - work with the array
    // - use the array in a different kernel
    // - ...
  }

  int main()
  {
    constexpr int N = 100000000; // problem size
    constexpr int blockSize = 256; //configurable block size

    //needed number of blocks for the given problem size
    constexpr int gridSize = round_up_to_nearest_multiple(N, blockSize);

    float *a;
    // allocate memory on the GPU
    HIP_CHECK(hipMalloc(&a, sizeof(*a) * N));

    std::cout << "Launching kernel." << std::endl;
    example_kernel<<<dim3(gridSize), dim3(blockSize), 0/*example doesn't use shared memory*/, 0/*default stream*/>>>(a, N);
    // make sure kernel execution is finished by synchronizing. The CPU can also
    // execute other instructions during that time
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "Kernel execution finished." << std::endl;

    HIP_CHECK(hipFree(a));
  }

Inline qualifiers
--------------------------------------------------------------------------------

HIP adds the ``__noinline__`` and ``__forceinline__`` function qualifiers.

``__noinline__`` is a hint to the compiler to not inline the function, whereas
``__forceinline__`` forces the compiler to inline the function. These qualifiers
can be applied to both ``__host__`` and ``__device__`` functions.

``__noinline__`` and ``__forceinline__`` can not be used in combination.

__launch_bounds__
--------------------------------------------------------------------------------

GPU multiprocessors have a fixed pool of resources (primarily registers and
shared memory) which are shared by the actively running warps. Using more
resources per thread can increase executed instructions per cycle but reduces
the resources available for other warps and may therefore limit the occupancy,
i.e. the number of warps that can be executed simultaneously. Thus GPUs have to
balance resource usage between instruction- and thread-level parallelism.

``__launch_bounds__`` allows the application to provide hints that influence the
resource (primarily registers) usage of the generated code. It is a function
attribute that must be attached to a __global__ function:

.. code-block:: cpp

  __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)
  kernel_name(/*args*/);

The  ``__launch_bounds__`` parameters are explained in the following sections:

MAX_THREADS_PER_BLOCK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This parameter is a guarantee from the programmer, that kernel will not be
launched with more threads than ``MAX_THREADS_PER_BLOCK``.

If no ``__launch_bounds__`` are specified, ``MAX_THREADS_PER_BLOCK`` is
the maximum block size supported by the device (see
:doc:`../reference/hardware_features`). Reducing ``MAX_THREADS_PER_BLOCK``
allows the compiler to use more resources per thread than an unconstrained
compilation. This might however reduce the amount of blocks that can run
concurrently on a CU, thereby reducing occupancy and trading thread-level
parallelism for instruction-level parallelism.

``MAX_THREADS_PER_BLOCK`` is particularly useful in cases, where the compiler is
constrained by register usage in order to meet requirements of large block sizes
that are never used at launch time.

The compiler can only use the hints to manage register usage, and does not
automatically reduce shared memory usage. The compilation fails, if the compiler
can not generate code that satisfies the launch bounds.

On NVCC this parameter maps to the ``.maxntid`` PTX directive.

When launching kernels HIP will validate the launch configuration to make sure
the requested block size is not larger than ``MAX_THREADS_PER_BLOCK`` and
return an error if it is exceeded.

If :doc:`AMD_LOG_LEVEL <./logging>` is set, detailed information will be shown
in the error log message, including the launch configuration of the kernel and
the specified ``__launch_bounds__``.

MIN_WARPS_PER_EXECUTION_UNIT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This parameter specifies the minimum number of warps that must be able to run
concurrently on an execution unit.
``MIN_WARPS_PER_EXECUTION_UNIT`` is optional and defaults to 1 if not specified.
Since active warps compete for the same fixed pool of resources, the compiler
must constrain the resource usage of the warps. This option gives a lower
bound to the occupancy of the kernel.

From this parameter, the compiler derives a maximum number of registers that can
be used in the kernel. The amount of registers that can be used at most is
:math:`\frac{\text{available registers}}{\text{MIN_WARPS_PER_EXECUTION_UNIT}}`,
but it might also have other, architecture specific, restrictions.

The available registers per Compute Unit are listed in
:doc:`rocm:reference/gpu-arch-specs`. Beware that these values are per Compute
Unit, not per Execution Unit. On AMD GPUs a Compute Unit consists of 4 Execution
Units, also known as SIMDs, each with their own register file. For more
information see :doc:`../understand/hardware_implementation`.
:cpp:struct:`hipDeviceProp_t` also has a field ``executionUnitsPerMultiprocessor``.

Memory space qualifiers
================================================================================

HIP adds qualifiers to specify the memory space in which the variables are
located.

Generally, variables allocated in host memory are not directly accessible within
device code, while variables allocated in device memory are not directly
accessible from the host code. More details on this can be found in
:ref:`unified_memory`.

__device__
--------------------------------------------------------------------------------

Variables marked with ``__device__`` reside in device memory. It can be
combined together with one of the following qualifiers, however these qualifiers
also imply the ``__device__`` qualifier.

By default it can only be accessed from the threads on the device. In order to
access it from the host, its address and size need to be queried using
:cpp:func:`hipGetSymbolAddress` and :cpp:func:`hipGetSymbolSize` and copied with
:cpp:func:`hipMemcpyToSymbol` or :cpp:func:`hipMemcpyFromSymbol`.

__constant__
--------------------------------------------------------------------------------

Variables marked with ``__constant__`` reside in device memory. Variables in
that address space are routed through the constant cache, but that address space
has a limited logical size.
This memory space is read-only from within kernels and can only be set by the
host before kernel execution.

To get the best performance benefit, these variables need a special access
pattern to benefit from the constant cache - the access has to be uniform within
a warp, otherwise the accesses are serialized.

The constant cache reduces the pressure on the other caches and may enable
higher throughput and lower latency accesses.

To set the ``__constant__`` variables the host must copy the data to the device
using :cpp:func:`hipMemcpyToSymbol`, for example:

.. code-block:: cpp

    __constant__ int const_array[8];

    void set_constant_memory(){
      int host_data[8] {1,2,3,4,5,6,7,8};

      hipMemcpyToSymbol(const_array, host_data, sizeof(int) * 8);

      // call kernel that accesses const_array
    }

__shared__
--------------------------------------------------------------------------------

Variables marked with ``__shared__`` are only accessible by threads within the
same block and have the lifetime of that block. It is usually backed by on-chip
shared memory, providing fast access to all threads within a block, which makes
it perfectly suited for sharing variables.

Shared memory can be allocated statically within the kernel, but the size
of it has to be known at compile time.

In order to dynamically allocate shared memory during runtime, but before the
kernel is launched, the variable has to be declared  ``extern``, and the kernel
launch has to specify the needed amount of ``extern`` shared memory in the launch
configuration. The statically allocated shared memory is allocated without this
parameter.

.. code-block:: cpp

  #include <hip/hip_runtime.h>

  extern __shared__ int shared_array[];

  __global__ void kernel(){
    // initialize shared memory
    shared_array[threadIdx.x] = threadIdx.x;
    // use shared memory - synchronize to make sure, that all threads of the
    // block see all changes to shared memory
    __syncthreads();
  }

  int main(){
    //shared memory in this case depends on the configurable block size
    constexpr int blockSize = 256;
    constexpr int sharedMemSize = blockSize * sizeof(int);
    constexpr int gridSize = 2;

    kernel<<<dim3(gridSize), dim3(blockSize), sharedMemSize, 0>>>();
  }

__managed__
--------------------------------------------------------------------------------

Managed memory is a special qualifier, that makes the marked memory available on
the device and on the host. For more details see :ref:`unified_memory`.

__restrict__
--------------------------------------------------------------------------------

The ``__restrict__`` keyword tells the compiler that the associated memory
pointer does not alias with any other pointer in the function. This can help the
compiler perform better optimizations. For best results, every pointer passed to
a function should use this keyword.

********************************************************************************
Built-in constants
********************************************************************************

HIP defines some special built-in constants for use in device code.

These built-ins are not implicitly defined by the compiler, the
``hip_runtime.h`` header has to be included instead.

Index built-ins
================================================================================

Kernel code can use these identifiers to distinguish between the different
threads and blocks within a kernel.

These built-ins are of type dim3, and are constant for each thread, but differ
between the threads or blocks, and are initialized at kernel launch.

blockDim and gridDim
--------------------------------------------------------------------------------

``blockDim`` and ``gridDim`` contain the sizes specified at kernel launch.
``blockDim`` contains the amount of threads in the x-, y- and z-dimensions of
the block of threads. Similarly ``gridDim`` contains the amount of blocks in the
grid.

.. _thread_and_block_idx:

threadIdx and blockIdx
--------------------------------------------------------------------------------

``threadIdx`` and ``blockIdx`` can be used to identify the threads and blocks
within the kernel.

``threadIdx`` identifies the thread within a block, meaning its values are
within ``0`` and ``blockDim.{x,y,z} - 1``. Likewise ``blockIdx`` identifies the
block within the grid, and the values are within ``0`` and ``gridDim.{} - 1``.

A global unique identifier of a three-dimensional grid can be calculated using
the following code:

.. code-block:: cpp

  (threadIdx.x + blockIdx.x * blockDim.x) +
  (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x +
  (threadIdx.z + blockIdx.z * blockDim.z) * blockDim.x * blockDim.y

.. _warp_size:

warpSize
================================================================================

The ``warpSize`` constant contains the number of threads per warp for the given
target device. On AMD hardware, this is referred to as ``wavefront size``, which 
may vary depending on the architecture. For more details, see the
:doc:`hardware features <../reference/hardware_features>`.

Since ``warpSize`` can differ between devices, it can not be assumed to be a
compile-time constant on the host. It has to be queried using
:cpp:func:`hipDeviceGetAttribute` or :cpp:func:`hipDeviceGetProperties`, e.g.:

.. code-block:: cpp

    int warpSizeHost;
    hipDeviceGetAttribute(&warpSizeHost, hipDeviceAttributeWarpSize, deviceId);

.. note::

  ``warpSize`` should not be assumed to be a specific value in portable HIP
  applications. NVIDIA devices return 32 for this variable; AMD devices return
  64 for gfx9 and 32 for gfx10 and above. HIP doesn't support ``warpSize`` of
  64 on gfx10 and above. While code that assumes a ``warpSize``
  of 32 can run on devices with a ``warpSize`` of 64, it only utilizes half of
  the compute resources.

Prior to ROCm 7.0, the warpSize parameter was a compile-time constant. Starting
with ROCm 7.0, it is early folded by the compiler, allowing it to be used in
loop bounds and enabling loop unrolling in a manner similar to a compile-time
constant warp size.

If compile time warp size is required, for example to select the correct mask
type or code path at compile time, the recommended approach is to determine the
warp size of the GPU on host side and setup the kernel accordingly, as shown in
the following block reduce example.

The ``block_reduce`` kernel has a template parameter for warp size and performs
a reduction operation in two main phases:

- Shared memory reduction: Reduction is performed iteratively, halving the
  number of active threads each step until only a warp remains
  (32 or 64 threads, depending on the device).

- Warp-level reduction: Once the shared memory reduction completes, the
  remaining threads use warp-level shuffling to sum the remaining values. This
  is done efficiently with the ``__shfl_down`` intrinsic, which allows threads within
  the warp to exchange values without explicit synchronization.

.. tab-set::

    .. tab-item:: WarpSize template parameter
       :sync: template-warpsize

       .. literalinclude:: ../tools/example_codes/template_warp_size_reduction.hip
          :start-after: // [Sphinx template warp size block reduction kernel start]
          :end-before: // [Sphinx template warp size block reduction kernel end]
          :language: cpp


    .. tab-item:: HIP warpSize
       :sync: hip-warpsize

       .. literalinclude:: ../tools/example_codes/warp_size_reduction.hip
          :start-after: // [Sphinx HIP warp size block reduction kernel start]
          :end-before: // [Sphinx HIP warp size block reduction kernel end]
          :language: cpp

The host code with the main function:

- Retrieves the warp size of the GPU (``warpSizeHost``) to determine the optimal
  kernel configuration.

- Allocates device memory (``d_data`` for input, ``d_results`` for block-wise
  output) and initializes the input vector to 1.

- Generates the mask variables for every warp and copies them to the device.

  .. tab-set::

      .. tab-item:: WarpSize template parameter
         :sync: template-warpsize

         .. literalinclude:: ../tools/example_codes/template_warp_size_reduction.hip
            :start-after: // [Sphinx template warp size mask generation start]
            :end-before: // [Sphinx template warp size mask generation end]
            :language: cpp


      .. tab-item:: HIP warpSize
         :sync: hip-warpsize

         .. literalinclude:: ../tools/example_codes/warp_size_reduction.hip
            :start-after:  // [Sphinx HIP warp size mask generation start]
            :end-before:  // [Sphinx HIP warp size mask generation end]
            :language: cpp

- Selects the appropriate kernel specialization based on the warp
  size (either 32 or 64) and launches the kernel.

  .. tab-set::

      .. tab-item:: WarpSize template parameter
         :sync: template-warpsize

         .. literalinclude:: ../tools/example_codes/template_warp_size_reduction.hip
            :start-after: // [Sphinx template warp size select kernel start]
            :end-before: // [Sphinx template warp size select kernel end]
            :language: cpp


      .. tab-item:: HIP warpSize
         :sync: hip-warpsize

         .. literalinclude:: ../tools/example_codes/warp_size_reduction.hip
            :start-after: // [Sphinx HIP warp size select kernel start]
            :end-before: // [Sphinx HIP warp size select kernel end]
            :language: cpp

- Synchronizes the device and copies the results back to the host.

- Checks that each block's sum is equal with the expected mask bit count, 
  verifying the reduction's correctness.

- Frees the device memory to prevent memory leaks.

.. note::

  The ``warpSize`` runtime example code is also provided for comparison purposes
  and the full example codes are located in the `tools folder <https://github.com/ROCm/hip/tree/docs/develop/docs/tools/example_codes>`_.

  The variable ``warpSize`` can be used for loop bounds and supports 
  loop unrolling similarly to the template parameter ``WarpSize``.

For users who still require a compile-time constant warp size as a macro on the
device side, it can be defined manually based on the target device architecture,
as shown in the following example.

.. code-block:: cpp

  #if defined(__GFX8__) || defined(__GFX9__)
    #define WarpSize 64
  #else
    #define WarpSize 32
  #endif

.. note:: 

  ``mwavefrontsize64`` compiler option is not supported by HIP runtime, that's
  why the architecture based compile time selector is an acceptable approach.

********************************************************************************
Vector types
********************************************************************************

These types are not automatically provided by the compiler. The
``hip_vector_types.h`` header, which is also included by ``hip_runtime.h`` has
to be included to use these types.

Fundamental vector types
================================================================================

Fundamental vector types derive from the `fundamental C++ integral and
floating-point types <https://en.cppreference.com/w/cpp/language/types>`_. These
types are defined in ``hip_vector_types.h``, which is included by
``hip_runtime.h``.

All vector types can be created with ``1``, ``2``, ``3`` or ``4`` elements, the
corresponding type is ``<fundamental_type>i``, where ``i`` is the number of
elements.

All vector types support a constructor function of the form
``make_<type_name>()``. For example,
``float3 make_float3(float x, float y, float z)`` creates a vector of type
``float3`` with value ``(x,y,z)``.
The elements of the vectors can be accessed using their members ``x``, ``y``,
``z``, and ``w``.

.. code-block:: cpp

  double2 d2_vec = make_double2(2.0, 4.0);
  double first_elem = d2_vec.x;

HIP supports vectors created from the following fundamental types:

.. list-table::
  :widths: 50 50

  *
    - **Integral Types**
    -
  *
    - ``char``
    - ``uchar``
  *
    - ``short``
    - ``ushort``
  *
    - ``int``
    - ``uint``
  *
    - ``long``
    - ``ulong``
  *
    - ``longlong``
    - ``ulonglong``
  *
    - **Floating-Point Types**
    -
  *
    - ``float``
    -
  *
    - ``double``
    -

.. _dim3:

dim3
================================================================================

``dim3`` is a special three-dimensional unsigned integer vector type that is
commonly used to specify grid and group dimensions for kernel launch
configurations.

Its constructor accepts up to three arguments. The unspecified dimensions are
initialized to 1.

********************************************************************************
Built-in device functions
********************************************************************************

.. _memory_fence_instructions:

Memory fence instructions
================================================================================

HIP does not enforce strict ordering on memory operations, meaning, that the
order in which memory accesses are executed, is not necessarily the order in
which other threads observe these changes. So it can not be assumed, that data
written by one thread is visible by another thread without synchronization.

Memory fences are a way to enforce a sequentially consistent order on the memory
operations. This means, that all writes to memory made before a memory fence are
observed by all threads after the fence. The scope of these fences depends on
what specific memory fence is called.

HIP supports ``__threadfence()``, ``__threadfence_block()`` and
``__threadfence_system()``:

* ``__threadfence_block()`` orders memory accesses for all threads within a thread block.
* ``__threadfence()`` orders memory accesses for all threads on a device.
* ``__threadfence_system()`` orders memory accesses for all threads in the system, making writes to memory visible to other devices and the host

.. _synchronization_functions:

Synchronization functions
================================================================================

Synchronization functions cause all threads in a group to wait at this
synchronization point until all threads reached it. These functions implicitly
include a :ref:`threadfence <memory_fence_instructions>`, thereby ensuring
visibility of memory accesses for the threads in the group.

The ``__syncthreads()`` function comes in different versions.

``void __syncthreads()`` simply synchronizes the threads of a block. The other
versions additionally evaluate a predicate:

``int __syncthreads_count(int predicate)`` returns the number of threads for
which the predicate evaluates to non-zero.

``int __syncthreads_and(int predicate)`` returns non-zero if the predicate
evaluates to non-zero for all threads.

``int __syncthreads_or(int predicate)`` returns non-zero if any of the
predicates evaluates to non-zero.

The Cooperative Groups API offers options to synchronize threads on a developer
defined set of thread groups. For further information, check the
:ref:`Cooperative Groups API reference <cooperative_groups_reference>` or the
:ref:`Cooperative Groups section in the programming guide
<cooperative_groups_how-to>`.

Math functions
================================================================================

HIP-Clang supports a set of math operations that are callable from the device.
HIP supports most of the device functions supported by CUDA. These are described
on :ref:`Math API page <math_api_reference>`.

Texture functions
================================================================================

The supported texture functions are listed in ``texture_fetch_functions.h`` and
``texture_indirect_functions.h`` header files in the
`HIP-AMD backend repository <https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail>`_.

Texture functions are not supported on some devices. To determine if texture functions are supported
on your device, use ``Macro __HIP_NO_IMAGE_SUPPORT == 1``. You can query the attribute
``hipDeviceAttributeImageSupport`` to check if texture functions are supported in the host runtime
code.

Surface functions
================================================================================

The supported surface functions are located on :ref:`Surface object reference
page <surface_object_reference>`.

Timer functions
================================================================================

HIP provides device functions to read a high-resolution timer from within the
kernel.

The following functions count the cycles on the device, where the rate varies
with the actual frequency.

.. code-block:: cpp

  clock_t clock()
  long long int clock64()

.. note::

  ``clock()`` and ``clock64()`` do not work properly on AMD RDNA3 (GFX11) graphic processors.

The difference between the returned values represents the cycles used.

.. code-block:: cpp

  __global void kernel(){
    long long int start = clock64();
    // kernel code
    long long int stop = clock64();
    long long int cycles = stop - start;
  }

``long long int wall_clock64()`` returns the wall clock time on the device, with a constant, fixed frequency.
The frequency is device dependent and can be queried using:

.. code-block:: cpp

  int wallClkRate = 0; //in kilohertz
  hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, deviceId);

.. _atomic functions:

Atomic functions
================================================================================

Atomic functions are read-modify-write (RMW) operations, whose result is visible
to all other threads on the scope of the atomic operation, once the operation
completes.

If multiple instructions from different devices or threads target the same
memory location, the instructions are serialized in an undefined order.

Atomic operations in kernels can operate on block scope (i.e. shared memory),
device scope (global memory), or system scope (system memory), depending on
:doc:`hardware support <../reference/hardware_features>`.

The listed functions are also available with the ``_system`` (e.g.
``atomicAdd_system``) suffix, operating on system scope, which includes host
memory and other GPUs' memory. The functions without suffix operate on shared
or global memory on the executing device, depending on the memory space of the
variable.

HIP supports the following atomic operations, where ``TYPE`` is one of ``int``,
``unsigned int``, ``unsigned long``, ``unsigned long long``, ``float`` or
``double``, while ``INTEGER`` is ``int``, ``unsigned int``, ``unsigned long``,
``unsigned long long``:

.. list-table:: Atomic operations

    * - ``TYPE atomicAdd(TYPE* address, TYPE val)``

    * - ``TYPE atomicSub(TYPE* address, TYPE val)``

    * - ``TYPE atomicMin(TYPE* address, TYPE val)``
    * - ``long long atomicMin(long long* address, long long val)``

    * - ``TYPE atomicMax(TYPE* address, TYPE val)``
    * - ``long long atomicMax(long long* address, long long val)``

    * - ``TYPE atomicExch(TYPE* address, TYPE val)``

    * - ``TYPE atomicCAS(TYPE* address, TYPE compare, TYPE val)``

    * - ``INTEGER atomicAnd(INTEGER* address, INTEGER val)``

    * - ``INTEGER atomicOr(INTEGER* address, INTEGER val)``

    * - ``INTEGER atomicXor(INTEGER* address, INTEGER val)``

    * - ``unsigned int atomicInc(unsigned int* address)``

    * - ``unsigned int atomicDec(unsigned int* address)``

Unsafe floating-point atomic operations
--------------------------------------------------------------------------------

Some HIP devices support fast atomic operations on floating-point values. For
example, ``atomicAdd`` on single- or double-precision floating-point values may
generate a hardware instruction that is faster than emulating the atomic
operation using an atomic compare-and-swap (CAS) loop.

On some devices, fast atomic instructions can produce results that differ from
the version implemented with atomic CAS loops. For example, some devices
will use different rounding or denormal modes, and some devices produce
incorrect answers if fast floating-point atomic instructions target fine-grained
memory allocations.

The HIP-Clang compiler offers compile-time options to control the generation of
unsafe atomic instructions. By default the compiler does not generate unsafe
instructions. This is the same behaviour as with the ``-mno-unsafe-fp-atomics``
compilation flag. The ``-munsafe-fp-atomics`` flag indicates to the compiler
that all floating-point atomic function calls are allowed to use an unsafe
version, if one exists. For example, on some devices, this flag indicates to the
compiler that no floating-point ``atomicAdd`` function can target fine-grained
memory. These options are applied globally for the entire compilation.

HIP provides special functions that override the global compiler option for safe
or unsafe atomic functions.

The ``safe`` prefix always generates safe atomic operations, even when
``-munsafe-fp-atomics`` is used, whereas ``unsafe`` always generates fast atomic
instructions, even when ``-mno-unsafe-fp-atomics``. The following table lists
the safe and unsafe atomic functions, where ``FLOAT_TYPE`` is either ``float``
or ``double``.

.. list-table:: AMD specific atomic operations

    * - ``FLOAT_TYPE unsafeAtomicAdd(FLOAT_TYPE* address, FLOAT_TYPE val)``

    * - ``FLOAT_TYPE safeAtomicAdd(FLOAT_TYPE* address, FLOAT_TYPE val)``

.. _warp-cross-lane:

Warp cross-lane functions
================================================================================

Threads in a warp are referred to as ``lanes`` and are numbered from ``0`` to
``warpSize - 1``. Warp cross-lane functions cooperate across all lanes in a
warp. AMD GPUs guarantee, that all warp lanes are executed in lockstep, whereas
NVIDIA GPUs that support Independent Thread Scheduling might require additional
synchronization, or the use of the ``__sync`` variants.

Note that different devices can have different warp sizes. You should query the
:ref:`warpSize <warp_size>` in portable code and not assume a fixed warp size.

All mask values returned or accepted by these built-ins are 64-bit unsigned
integer values, even when compiled for a device with 32 threads per warp. On
such devices the higher bits are unused. CUDA code ported to HIP requires
changes to ensure that the correct type is used.

Note that the ``__sync`` variants are available in ROCm 7.0 (and enabled by
default, unlike in previous versions). They can be disabled by setting the
preprocessor macro ``HIP_DISABLE_WARP_SYNC_BUILTINS``.

The ``_sync`` variants require a 64-bit unsigned integer mask argument that
specifies the lanes of the warp that will participate. Each participating thread
must have its own bit set in its mask argument, and all active threads specified
in any mask argument must execute the same call with the same mask, otherwise
the result is undefined. The implementation includes a static assert to check
that the program source uses the correct type for the mask.

.. _warp_vote_functions:

Warp vote and ballot functions
--------------------------------------------------------------------------------

.. code-block:: cpp

  int __all(int predicate)
  int __any(int predicate)
  unsigned long long __ballot(int predicate)
  unsigned long long __activemask()

  int __all_sync(unsigned long long mask, int predicate)
  int __any_sync(unsigned long long mask, int predicate)
  unsigned long long __ballot_sync(unsigned long long mask, int predicate)

You can use ``__any`` and ``__all`` to get a summary view of the predicates evaluated by the
participating lanes.

* ``__any()``: Returns 1 if the predicate is non-zero for any participating lane, otherwise it returns 0.

* ``__all()``: Returns 1 if the predicate is non-zero for all participating lanes, otherwise it returns 0.

To determine if the target platform supports the any/all instruction, you can
query the ``hasWarpVote`` device property on the host or use the
``HIP_ARCH_HAS_WARP_VOTE`` compiler definition in device code.

``__ballot`` returns a bit mask containing the 1-bit predicate value from each
lane. The nth bit of the result contains the bit contributed by the nth lane.

``__activemask()`` returns a bit mask of currently active warp lanes. The nth
bit of the result is 1 if the nth lane is active.

Note that the ``__ballot`` and ``__activemask`` built-ins in HIP have a 64-bit return
value (unlike the 32-bit value returned by the CUDA built-ins). Code ported from
CUDA should be adapted to support the larger warp sizes that the HIP version
requires.

Applications can test whether the target platform supports the ``__ballot`` or
``__activemask`` instructions using the ``hasWarpBallot`` device property in host
code or the ``HIP_ARCH_HAS_WARP_BALLOT`` macro defined by the compiler for device
code.

Warp match functions
--------------------------------------------------------------------------------

.. code-block:: cpp

  unsigned long long __match_any(T value)
  unsigned long long __match_all(T value, int *pred)

  unsigned long long __match_any_sync(unsigned long long mask, T value)
  unsigned long long __match_all_sync(unsigned long long mask, T value, int *pred)

``T`` can be a 32-bit integer type, 64-bit integer type or a single precision or
double precision floating point type.

``__match_any`` returns a bit mask where the n-th bit is set to 1 if the n-th
lane has the same ``value`` as the current lane, and 0 otherwise.

``__match_all`` returns a bit mask with the bits of the participating lanes are
set to 1 if all lanes have the same ``value``, and 0 otherwise.
The predicate ``pred`` is set to true if all participating threads have the same
``value``, and false otherwise.

Warp shuffle functions
--------------------------------------------------------------------------------

.. code-block:: cpp

  T __shfl      (T var, int srcLane, int width=warpSize);
  T __shfl_up   (T var, unsigned int delta, int width=warpSize);
  T __shfl_down (T var, unsigned int delta, int width=warpSize);
  T __shfl_xor  (T var, int laneMask, int width=warpSize);

  T __shfl_sync      (unsigned long long mask, T var, int srcLane, int width=warpSize);
  T __shfl_up_sync   (unsigned long long mask, T var, unsigned int delta, int width=warpSize);
  T __shfl_down_sync (unsigned long long mask, T var, unsigned int delta, int width=warpSize);
  T __shfl_xor_sync  (unsigned long long mask, T var, int laneMask, int width=warpSize);

``T`` can be a 32-bit integer type, 64-bit integer type or a single precision or
double precision floating point type.

The warp shuffle functions exchange values between threads within a warp.

The optional ``width`` argument specifies subgroups, in which the warp can be
divided to share the variables.
It has to be a power of two smaller than or equal to ``warpSize``. If it is
smaller than ``warpSize``, the warp is grouped into separate groups, that are each
indexed from 0 to width as if it was its own entity, and only the lanes within
that subgroup participate in the shuffle. The lane indices in the subgroup are
given by ``laneIdx % width``.

The different shuffle functions behave as following:

``__shfl``
  The thread reads the value from the lane specified in ``srcLane``.

``__shfl_up``
  The thread reads ``var`` from lane ``laneIdx - delta``, thereby "shuffling"
  the values of the lanes of the warp "up". If the resulting source lane is out
  of range, the thread returns its own ``var``.

``__shfl_down``
  The thread reads ``var`` from lane ``laneIdx + delta``, thereby "shuffling"
  the values of the lanes of the warp "down". If the resulting source lane is
  out of range, the thread returns its own ``var``.

``__shfl_xor``
  The thread reads ``var`` from lane ``laneIdx xor lane_mask``. If ``width`` is
  smaller than ``warpSize``, the threads can read values from subgroups before
  the current subgroup. If it tries to read values from later subgroups, the
  function returns the ``var`` of the calling thread.

Warp reduction functions
-------------------------------------------------------------------------------------------------------------
Arithmetic reduces:

.. code-block:: cpp

  T __reduce_add_sync (unsigned long long mask, T var);
  T __reduce_min_sync (unsigned long long mask, T var);
  T __reduce_max_sync (unsigned long long mask, T var);

``T`` can be:

* On Nvidia platform: ``int`` or ``unsigned int``

* On AMD platform: ``int`` or ``unsigned int``; if the user defines the macro ``HIP_ENABLE_EXTRA_WARP_SYNC_TYPES``, then: ``unsigned long long``, ``long long``, ``half``/``single``/``double`` precision floating
point types are also be supported.

Returns the aggregated result of the arithmetic operation, where each of the participating threads
(i.e. the ones mentioned on the mask) contribute ``var``.

NOTE: for type ``half``, these intrinsics are not available in environments where the arithmetic operators are not available for
that type.

Logical reduces:

.. code-block:: cpp

  T __reduce_and_sync (unsigned long long mask, T var);
  T __reduce_or_sync  (unsigned long long mask, T var);
  T __reduce_xor_sync (unsigned long long mask, T var);

``T`` can be:

* On Nvidia platform: ``unsigned int``

* On AMD platform: ``unsigned int``, and if the user defines the macro ``HIP_ENABLE_EXTRA_WARP_SYNC_TYPES``, then ``int``, ``unsigned long long`` or ``long long`` are also supported

Returns the result of the aggregated logical AND/OR/XOR operation where each of the participating threads
(i.e. the ones mentioned on the mask) contribute ``var``.

The mask argument is a 64-bit unsigned integer that specifies the lanes in the warp that 
participate in cross-lane communication with the calling lane. Each participating thread must have its own
bit set in its mask argument, and all active threads specified in any mask argument must execute the same
call with the same mask, otherwise the result is undefined.

Informational note: On the AMD platform, **masks that start from lane zero and have no "holes" use faster cross-lane operations and
exhibit better performance** than masks with "holes" (example of mask with no holes: 0xFF and with holes: 0xFB;
the reduction with 0xFF is faster).

These functions do not provide a memory barrier on any platform.

Warp matrix functions
--------------------------------------------------------------------------------

Warp matrix functions allow a warp to cooperatively operate on small matrices
that have elements spread over lanes in an unspecified manner.

HIP does not support warp matrix types or functions.

Cooperative groups functions
================================================================================

You can use cooperative groups to synchronize groups of threads across thread
blocks. It also provide a way of communicating between these groups.

For further information, check the :ref:`Cooperative Groups API reference
<cooperative_groups_reference>` or the :ref:`Cooperative Groups programming
guide <cooperative_groups_how-to>`.
