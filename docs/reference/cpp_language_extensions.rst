.. meta::
  :description: This chapter describes the built-in variables and functions that are accessible from the
                HIP kernel. It's intended for users who are familiar with CUDA kernel syntax and want to
                learn how HIP differs from CUDA.
  :keywords: AMD, ROCm, HIP, CUDA, c++ language extensions, HIP functions

################################################################################
C++ language extensions
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

``__host__``
--------------------------------------------------------------------------------

The ``__host__`` qualifier is used to specify functions for execution
on the host. This qualifier is implicitly defined for any function where no
host, device or global qualifier is added, in order to not break compatibility
with existing C++ functions.

You can't combine ``__host__`` with ``__global__``.

``__device__``
--------------------------------------------------------------------------------

The ``__device__`` qualifier is used to specify functions for execution on the
device. They can only be called from other ``__device__`` functions or from
``__global__`` functions.

You can combine it with the ``__host__`` qualifier and mark functions
``__host__ __device__``. In this case, the function is compiled for the host and
the device. Note that these functions can't use the HIP built-ins (e.g.,
``threadIdx.x`` or ``warpSize``), as they are not available on the host. If you
need to use HIP grid coordinate functions, you can pass the necessary coordinate
information as an argument.

``__global__``
--------------------------------------------------------------------------------

Functions marked ``__global__`` are executed on the device and are referred to
as kernels. Their return type must be void. Kernels have a special launch
mechanism, and have to be launched from the host.

Unlike CUDA, HIP does not support dynamic parallelism, meaning that kernels can
not be called from the device.

Calling ``__global__`` functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The launch mechanism for kernels differs from standard function calls, as they
need an additional configuration, that specifies the grid and block dimensions
(i.e. the amount of threads to be launched), as well as specifying the amount of
shared memory per block and which stream to execute the kernel on.

Kernels are called using the ``<<<>>>`` syntax known from CUDA, but also
supports the ``hipLaunchKernelGGL`` macro.

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

  __global__ void example_kernel(float * const a, const unsigned int N)
  {
    // Index variables. Determined by the launch configuration.
    // The following uniquely identifies a thread in a 1D configuration.
    const int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    // simple initialization of the array
    if(globalIdx < N){
      a[globalIdx] = globalIdx;
    }
  }

  int main()
  {
    constexpr int N = 1000000; // problem size
    constexpr int blockSize = 256; //configurable block size
    constexpr int gridSize = (N + blockSize - 1)/blockSize; //needed number of blocks for the given problem size

    float *a;
    hipMalloc(&a, sizeof(*a) * N);

    example_kernel<<<dim3(gridSize), dim3(blockSize), 0/*example doesn't use shared memory*/, 0/*default stream*/>>>(a, N);
  }

Inline qualifiers
--------------------------------------------------------------------------------

HIP adds the ``__noinline__`` and ``__forceinline__`` function qualifiers.

``__noinline__`` is a hint to the compiler to not inline the function, whereas 
``__forceinline__`` forces the compiler to inline the function. These qualifiers
can be applied to both ``__host__`` and ``__device__`` functions.

``__noinline__`` and ``__forceinline__`` can not be used in combination.

``__launch_bounds__``
--------------------------------------------------------------------------------

GPU multiprocessors have a fixed pool of resources (primarily registers and
shared memory) which are shared by the actively running warps. Using more
resources can increase IPC of the kernel but reduces the resources available for
other warps and limits the number of warps that can be simultaneously running.
Thus GPUs have to balance resource usage between instruction- and thread-level
parallelism.

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
:doc:`reference/hardware_features`). Reducing ``MAX_THREADS_PER_BLOCK`` allows
the compiler to use more resources per thread than an unconstrained
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

If :doc:`AMD_LOG_LEVEL <how-to/logging>` is set, detailed information will be
shown in the error log message, including the launch configuration of the
kernel and the specified ``__launch_bounds__``.

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
:math:`\frac{\text{available registers}}{\text{MIN_WARPS_PER_EXECUTION_UNIT}}`, but it might
also have other, architecture specific, restrictions.

The available registers per Compute Unit are listed in
:doc:`rocm:reference/gpu-arch-specs`. Beware that these values are per Compute
Unit, not per Execution Unit. On AMD GPUs a Compute Unit consists of 4 Execution
Units, also known as SIMDs, each with their own register file. For more
information see :doc:`understand/hardware_implementation`.
:cpp:struct:`hipDeviceProp_t` also has a field ``executionUnitsPerMultiprocessor``.

Porting from CUDA ``__launch_bounds``
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

CUDA defines the ``__launch_bounds`` qualifier which works similar to
``__launch_bounds__``:

.. code-block:: cpp

  __launch_bounds(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)

The first parameter is the same as HIPs' implementation. The second parameter of
``__launch_bounds`` must  be converted to the format used ``__launch_bounds__``,
which uses warps and execution units rather than blocks and multiprocessors.
This conversion is performed automatically by :doc:`HIPIFY <hipify:index>`.

.. code-block:: cpp

  MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK) / warpSize

Directly controlling the warps per execution unit makes it easier to reason
about the occupancy, unlike with blocks, where the occupancy depends on the
block size.

The use of execution units rather than multiprocessors also provides support for
architectures with multiple execution units per multiprocessor. For example, the
AMD GCN architecture has 4 execution units per multiprocessor.

``maxregcount``
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Unlike ``nvcc``, ``amdclang++`` does not support the ``--maxregcount`` option.
Instead, users are encouraged to use the ``__launch_bounds__`` directive since
the parameters are more intuitive and portable than micro-architecture details
like registers. The directive allows per-kernel control.
``__launch_bounds__`` works on both AMD and NVIDIA platforms.

Memory space qualifiers
================================================================================

HIP adds qualifiers to specify the memory space in which the variables are
located.

``__device__``
--------------------------------------------------------------------------------

Variables marked with ``__device__`` reside in device memory. It can be
combined together with one of the following qualifiers, however these qualifiers
also imply the ``__device__`` qualifier.

By default it can only be accessed from the threads on the device. In order to
access it from the host, its address and size need to be queried using
:cpp:func:`hipGetSymbolAddress` and :cpp:func:`hipGetSymbolSize` and copied with
 :cpp:func:`hipMemcpyToSymbol` or :cpp:func:`hipMemcpyFromSymbol`.

``__constant__``
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

    void use_constant_memory(){
      int host_data[8] {1,2,3,4,5,6,7,8};

      hipMemcpyToSymbol(const_array, host_data, sizeof(int) * 8);

      // call kernel that accesses const_array
    }

``__shared__``
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
    // use shared memory
  }

  int main(){
    //shared memory in this case depends on the configurable block size
    constexpr int blockSize = 256;
    constexpr int sharedMemSize = blockSize * sizeof(int);
    constexpr int gridSize = 2;

    kernel<<<dim3(gridSize), dim3(blockSize), sharedMemSize, 0>>>();
  }

``__managed__``
--------------------------------------------------------------------------------

Managed memory is a special qualifier, that makes the marked memory available on
the device and on the host. For more details see :ref:`unified_memory`.

``__restrict__``
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

``warpSize``
================================================================================

The ``warpSize`` constant contains the number of threads per warp for the given
target device. It can differ between different architectures, and on RDNA
architectures it can even differ between kernel launches, depending on whether
they run in CU or WGP mode. See the
:doc:`hardware features <rocm:/reference/hardware_features>` for more
information.

On the host side it can be queried using:

.. code-block:: cpp

    int val;
    hipDeviceGetAttribute(&val, hipDeviceAttributeWarpSize, deviceId);

.. note::

  ``warpSize`` should not be assumed to be a specific value in portable HIP
  applications. NVIDIA devices return 32 for this variable; AMD devices return
  64 for gfx9 and 32 for gfx10 and above.

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
  *
    - :cspan:`1` Integral Types
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
    - :cspan:`1` Floating-Point Types
  *
    - :cspan:`1` ``float``
  *
    - :cspan:`1` ``double``

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

HIP supports ``__threadfence()``, ``__threadfence_block()`` and
``__threadfence_system()``.

On AMD devices, ``__threadfence_system()``, has restrictions and therefore needs
the following workaround:

#. Build HIP with the ``HIP_COHERENT_HOST_ALLOC`` environment variable enabled.
#. Modify kernels that use ``__threadfence_system()`` as follows:

  * Ensure the kernel operates only on fine-grained system memory, which should be allocated with
    ``hipHostMalloc()``.
  * Remove ``memcpy`` for all allocated fine-grained system memory regions.

.. _synchronization_functions:

Synchronization functions
================================================================================

Synchronization functions causes all threads in the group to wait at this synchronization point, and for all shared and global memory accesses by the threads to complete, before running synchronization. This guarantees the visibility of accessed data for all threads in the group.

The ``__syncthreads()`` built-in function is supported in HIP. The ``__syncthreads_count(int)``,
``__syncthreads_and(int)``, and ``__syncthreads_or(int)`` functions are under development.

The Cooperative Groups API offer options to do synchronization on a developer defined set of thread groups. For further information, check :ref:`Cooperative Groups API <cooperative_groups_reference>` or :ref:`Cooperative Groups how to <cooperative_groups_how-to>`.

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
hardware support. 

The listed functions are also available with the ``_system`` suffix, operating
on system scope, which includes host memory and other GPUs' memory. The
functions without suffix operate on shared or global memory on the executing
device, depending on the memory space of the variable.

For hardware support see :doc:`hardware features <rocm:/reference/hardware_features>` for more information.

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


Differences in HIP and CUDA atomic support
--------------------------------------------------------------------------------

The following table lists differences in atomic support between HIP and CUDA.

.. list-table:: Atomic operations

    * - **Function**
      - **Supported in HIP**
      - **Supported in CUDA**

    * - ``float unsafeAtomicAdd(float* address, float val)``
      - ✓
      - ✗

    * - ``float safeAtomicAdd(float* address, float val)``
      - ✓
      - ✗

    * - ``double unsafeAtomicAdd(double* address, double val)``
      - ✓
      - ✗

    * - ``double safeAtomicAdd(double* address, double val)``
      - ✓
      - ✗


    * - ``unsigned int atomicInc(unsigned int* address)``
      - ✗
      - ✓

    * - ``unsigned int atomicDec(unsigned int* address)``
      - ✗
      - ✓


Unsafe floating-point atomic operations
--------------------------------------------------------------------------------
Some HIP devices support fast atomic RMW operations on floating-point values. For example,
``atomicAdd`` on single- or double-precision floating-point values may generate a hardware RMW
instruction that is faster than emulating the atomic operation using an atomic compare-and-swap
(CAS) loop.

On some devices, fast atomic RMW instructions can produce results that differ from the same
functions implemented with atomic CAS loops. For example, some devices will use different rounding
or denormal modes, and some devices produce incorrect answers if fast floating-point atomic RMW
instructions target fine-grained memory allocations.

The HIP-Clang compiler offers a compile-time option, so you can choose fast - but potentially
unsafe - atomic instructions for your code. On devices that support these instructions, you can include
the ``-munsafe-fp-atomics`` option. This flag indicates to the compiler that all floating-point atomic
function calls are allowed to use an unsafe version, if one exists. For example, on some devices, this
flag indicates to the compiler that no floating-point ``atomicAdd`` function can target fine-grained
memory.

If you want to avoid using unsafe use a floating-point atomic RMW operations, you can use the
``-mno-unsafe-fp-atomics`` option. Note that the compiler default is to not produce unsafe
floating-point atomic RMW instructions, so the ``-mno-unsafe-fp-atomics`` option is not necessarily
required. However, passing this option to the compiler is good practice.

When you pass ``-munsafe-fp-atomics`` or ``-mno-unsafe-fp-atomics`` to the compiler's command line,
the option is applied globally for the entire compilation. Note that if some of the atomic RMW function
calls cannot safely use the faster floating-point atomic RMW instructions, you must use
``-mno-unsafe-fp-atomics`` in order to ensure that your atomic RMW function calls produce correct
results.

HIP has four extra functions that you can use to more precisely control which floating-point atomic
RMW functions produce unsafe atomic RMW instructions:

* ``float unsafeAtomicAdd(float* address, float val)``
* ``double unsafeAtomicAdd(double* address, double val)`` (Always produces fast atomic RMW
  instructions on devices that have them, even when ``-mno-unsafe-fp-atomics`` is used)
* `float safeAtomicAdd(float* address, float val)`
* ``double safeAtomicAdd(double* address, double val)`` (Always produces safe atomic RMW
  operations, even when ``-munsafe-fp-atomics`` is used)

.. _warp-cross-lane:

Warp cross-lane functions
================================================================================

Threads in a warp are referred to as ``lanes`` and are numbered from ``0`` to
``warpSize - 1``. Warp cross-lane functions cooperate across all lanes in a
warp. The hardware guarantees that all warp lanes will execute in lockstep, so
additional synchronization is unnecessary, and the instructions use no shared
memory.

Note that NVIDIA and AMD devices have different warp sizes. You should use the
``warpSize`` built-in in portable code to query the warp size.

.. tip::
  Be sure to review HIP code ported from CUDA to ensure that it doesn't assume a
  ``warpSize`` of 32. Code that assumes a ``warpSize`` of 32 can run on a device
  with 64 threads per warp, but it only utilizes half of the machine's resources.

Since ``warpSize`` can differ between devices, it can not be assumed to be a
compile-time constant on the host. It has to be queried using
:cpp:func:`hipDeviceGetAttribute` or :cpp:func:`hipDeviceGetProperties`.

.. code-block:: cpp

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceID);
  int w = props.warpSize;
    // implement portable algorithm based on w (rather than assume 32 or 64)

Note that assembly kernels may be built for a warp size that is different from the default.
All mask values either returned or accepted by these built-ins are 64-bit
unsigned integer values, even when compiled for a wave-32 device, where all the
higher bits are unused. CUDA code ported to HIP requires changes to ensure that
the correct type is used.

Note that the ``__sync`` variants are made available in ROCm 6.2, but disabled by
default to help with the transition to 64-bit masks. They can be enabled by
setting the preprocessor macro ``HIP_ENABLE_WARP_SYNC_BUILTINS``. These built-ins
will be enabled unconditionally in the next ROCm release. Wherever possible, the
implementation includes a static assert to check that the program source uses
the correct type for the mask.

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

To determine if the target platform supports the any/all instruction, you can use the ``hasWarpVote``
device property or the ``HIP_ARCH_HAS_WARP_VOTE`` compiler definition.

``__ballot`` returns a bit mask containing the 1-bit predicate value from each
lane. The nth bit of the result contains the 1 bit contributed by the nth warp
lane.

``__activemask()`` returns a bit mask of currently active warp lanes. The nth bit
of the result is 1 if the nth warp lane is active.

Note that the ``__ballot`` and ``__activemask`` built-ins in HIP have a 64-bit return
value (unlike the 32-bit value returned by the CUDA built-ins). Code ported from
CUDA should be adapted to support the larger warp sizes that the HIP version
requires.

Applications can test whether the target platform supports the ``__ballot`` or
``__activemask`` instructions using the ``hasWarpBallot`` device property in host
code or the ``HIP_ARCH_HAS_WARP_BALLOT`` macro defined by the compiler for device
code.

The ``_sync`` variants require a 64-bit unsigned integer mask argument that
specifies the lanes in the warp that will participate in cross-lane
communication with the calling lane. Each participating thread must have its own
bit set in its mask argument, and all active threads specified in any mask
argument must execute the same call with the same mask, otherwise the result is
undefined.

Warp match functions
--------------------------------------------------------------------------------

.. code-block:: cpp

  unsigned long long __match_any(T value)
  unsigned long long __match_all(T value, int *pred)

  unsigned long long __match_any_sync(unsigned long long mask, T value)
  unsigned long long __match_all_sync(unsigned long long mask, T value, int *pred)

``T`` can be a 32-bit integer type, 64-bit integer type or a single precision or
double precision floating point type.

``__match_any`` returns a bit mask containing a 1-bit for every participating lane
if and only if that lane has the same value in ``value`` as the current lane, and
a 0-bit for all other lanes.

``__match_all`` returns a bit mask containing a 1-bit for every participating lane
if and only if they all have the same value in ``value`` as the current lane, and
a 0-bit for all other lanes. The predicate ``pred`` is set to true if and only if
all participating threads have the same value in ``value``.

The ``_sync`` variants require a 64-bit unsigned integer mask argument that
specifies the lanes in the warp that will participate in cross-lane
communication with the calling lane. Each participating thread must have its own
bit set in its mask argument, and all active threads specified in any mask
argument must execute the same call with the same mask, otherwise the result is
undefined.

Warp shuffle functions
--------------------------------------------------------------------------------

The default width is ``warpSize`` (see :ref:`warp-cross-lane`). Half-float shuffles are not supported.

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

The ``_sync`` variants require a 64-bit unsigned integer mask argument that
specifies the lanes in the warp that will participate in cross-lane
communication with the calling lane. Each participating thread must have its own
bit set in its mask argument, and all active threads specified in any mask
argument must execute the same call with the same mask, otherwise the result is
undefined.

Cooperative groups functions
================================================================================

You can use cooperative groups to synchronize groups of threads. Cooperative
groups also provide a way of communicating between groups of threads.

For further information, check :ref:`Cooperative Groups API
<cooperative_groups_reference>` or :ref:`Cooperative Groups how to
<cooperative_groups_how-to>`.

Warp matrix functions
================================================================================

Warp matrix functions allow a warp to cooperatively operate on small matrices that have elements
spread over lanes in an unspecified manner.

HIP does not support kernel language warp matrix types or functions.

.. list-table:: Warp matrix functions

    * - **Function**
      - **Supported in HIP**
      - **Supported in CUDA**

    * - ``void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned lda)``
      - ✗
      - ✓

    * - ``void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned lda, layout_t layout)``
      - ✗
      - ✓

    * - ``void store_matrix_sync(T* mptr, fragment<...> &a, unsigned lda, layout_t layout)``
      - ✗
      - ✓

    * - ``void fill_fragment(fragment<...> &a, const T &value)``
      - ✗
      - ✓

    * - ``void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c , bool sat)``
      - ✗
      - ✓
