.. meta::
  :description: This chapter describes the built-in variables and functions that are accessible from the
                HIP kernel. It's intended for users who are familiar with CUDA kernel syntax and want to
                learn how HIP differs from CUDA.
  :keywords: AMD, ROCm, HIP, CUDA, c++ language extensions, HIP functions

********************************************************************************
C++ Language Extensions
********************************************************************************

HIP provides a C++ syntax that is suitable for compiling most code that commonly appears in
compute kernels (classes, namespaces, operator overloading, and templates). HIP also defines other
language features that are designed to target accelerators, such as:

* A kernel-launch syntax that uses standard C++ (this resembles a function call and is portable to all
  HIP targets)
* Short-vector headers that can serve on a host or device
* Math functions that resemble those in ``math.h``, which is included with standard C++ compilers
* Built-in functions for accessing specific GPU hardware capabilities

.. note::

  This chapter describes the built-in variables and functions that are accessible from the HIP kernel. It's
  intended for users who are familiar with CUDA kernel syntax and want to learn how HIP differs from
  CUDA.

Features are labeled with one of the following keywords:

* **Supported**: HIP supports the feature with a CUDA-equivalent function
* **Not supported**: HIP does not support the feature
* **Under development**: The feature is under development and not yet available

Function-type qualifiers
========================================================

``__device__``
-----------------------------------------------------------------------

Supported  ``__device__`` functions are:

  * Run on the device
  * Called from the device only

You can combine ``__device__`` with the host keyword (:ref:`host_attr`).

``__global__``
-----------------------------------------------------------------------

Supported ``__global__`` functions are:

  * Run on the device
  * Called (launched) from the host

HIP ``__global__`` functions must have a ``void`` return type. The first parameter in a HIP ``__global__``
function must have the type ``hipLaunchParm``. Refer to :ref:`kernel-launch-example` to see usage.

HIP doesn't support dynamic-parallelism, which means that you can't call ``__global__`` functions from
the device.

.. _host_attr:

``__host__``
-----------------------------------------------------------------------

Supported ``__host__`` functions are:

  * Run on the host
  * Called from the host

You can combine ``__host__`` with ``__device__``; in this case, the function compiles for the host and the
device. Note that these functions can't use the HIP grid coordinate functions (e.g., ``threadIdx.x``). If
you need to use HIP grid coordinate functions, you can pass the necessary coordinate information as
an argument.

You can't combine ``__host__`` with ``__global__``.

HIP parses the ``__noinline__`` and ``__forceinline__`` keywords and converts them into the appropriate
Clang attributes.

Calling ``__global__`` functions
=============================================================

`__global__` functions are often referred to as *kernels*. When you call a global function, you're
*launching a kernel*. When launching a kernel, you must specify a run configuration that includes the
grid and block dimensions. The run configuration can also include other information for the launch,
such as the amount of additional shared memory to allocate and the stream where you want to run the
kernel.

HIP introduces a standard C++ calling convention (``hipLaunchKernelGGL``) to pass the run
configuration to the kernel. However, you can also use the CUDA ``<<< >>>`` syntax.

When using ``hipLaunchKernelGGL``, your first five parameters must be:

  * **symbol kernelName**: The name of the kernel you want to launch. To support template kernels
    that contain ``","``, use the ``HIP_KERNEL_NAME`` macro (HIPIFY tools insert this automatically).
  * **dim3 gridDim**: 3D-grid dimensions that specify the number of blocks to launch.
  * **dim3 blockDim**: 3D-block dimensions that specify the number of threads in each block.
  * **size_t dynamicShared**: The amount of additional shared memory that you want to allocate
    when launching the kernel (see :ref:`shared-variable-type`).
  * **hipStream_t**: The stream where you want to run the kernel. A value of ``0`` corresponds to the
    NULL stream (see :ref:`synchronization-functions`).

You can include your kernel arguments after these parameters.

.. code:: cpp

  // Example hipLaunchKernelGGL pseudocode:
  __global__ MyKernel(hipLaunchParm lp, float *A, float *B, float *C, size_t N)
  {
  ...
  }

  MyKernel<<<dim3(gridDim), dim3(groupDim), 0, 0>>> (a,b,c,n);

  // Alternatively, you can launch the kernel using:
  // hipLaunchKernelGGL(MyKernel, dim3(gridDim), dim3(groupDim), 0/*dynamicShared*/, 0/*stream), a, b, c, n);

You can use HIPIFY tools to convert CUDA launch syntax to ``hipLaunchKernelGGL``. This includes the
conversion of optional ``<<< >>>`` arguments into the five required ``hipLaunchKernelGGL``
parameters.

.. note::

  HIP doesn't support dimension sizes of :math:`gridDim * blockDim \ge 2^{32}` when launching a kernel.

.. _kernel-launch-example:

Kernel launch example
==========================================================

.. code:: cpp

  // Example showing device function, __device__ __host__
  // <- compile for both device and host
  float PlusOne(float x)
  {
    return x + 1.0;
  }

  __global__
  void
  MyKernel (hipLaunchParm lp, /*lp parm for execution configuration */
            const float *a, const float *b, float *c, unsigned N)
  {
    unsigned gid = threadIdx.x; // <- coordinate index function
    if (gid < N) {
      c[gid] = a[gid] + PlusOne(b[gid]);
    }
  }
  void callMyKernel()
  {
    float *a, *b, *c; // initialization not shown...
    unsigned N = 1000000;
    const unsigned blockSize = 256;

    MyKernel<<<dim3(gridDim), dim3(groupDim), 0, 0>>> (a,b,c,n);
    // Alternatively, kernel can be launched by
    // hipLaunchKernelGGL(MyKernel, dim3(N/blockSize), dim3(blockSize), 0, 0,  a,b,c,N);
  }

Variable type qualifiers
========================================================

``__constant__``
-----------------------------------------------------------------------------

The host writes constant memory before launching the kernel. This memory is read-only from the GPU
while the kernel is running. The functions for accessing constant memory are:

* ``hipGetSymbolAddress()``
* ``hipGetSymbolSize()``
* ``hipMemcpyToSymbol()``
* ``hipMemcpyToSymbolAsync()``
* ``hipMemcpyFromSymbol()``
* ``hipMemcpyFromSymbolAsync()``

.. _shared-variable-type:

``__shared__``
-----------------------------------------------------------------------------

To allow the host to dynamically allocate shared memory, you can specify ``extern __shared__`` as a
launch parameter.

.. note::
  Prior to the HIP-Clang compiler, dynamic shared memory had to be declared using the
  ``HIP_DYNAMIC_SHARED`` macro in order to ensure accuracy. This is because using static shared
  memory in the same kernel could've resulted in overlapping memory ranges and data-races. The
  HIP-Clang compiler provides support for ``extern`` shared declarations, so ``HIP_DYNAMIC_SHARED``
  is no longer required.

``__managed__``
-----------------------------------------------------------------------------

Managed memory, including the `__managed__` keyword, is supported in HIP combined host/device
compilation.

``__restrict__``
-----------------------------------------------------------------------------

``__restrict__`` tells the compiler that the associated memory pointer not to alias with any other pointer
in the kernel or function. This can help the compiler generate better code. In most use cases, every
pointer argument should use this keyword in order to achieve the benefit.

Built-in variables
====================================================

Coordinate built-ins
-----------------------------------------------------------------------------

The kernel uses coordinate built-ins (``thread*``, ``block*``, ``grid*``) to determine the coordinate index
and bounds for the active work item.

Built-ins are defined in ``amd_hip_runtime.h``, rather than being implicitly defined by the compiler.

Coordinate variable definitions for built-ins are the same for HIP and CUDA. For example: ``threadIdx.x``,
``blockIdx.y``, and ``gridDim.y``. The products ``gridDim.x * blockDim.x``, ``gridDim.y * blockDim.y``, and
``gridDim.z * blockDim.z`` are always less than ``2^32``.

Coordinate built-ins are implemented as structures for improved performance. When used with
``printf``, they must be explicitly cast to integer types.

warpSize
-----------------------------------------------------------------------------
The ``warpSize`` variable type is ``int``. It contains the warp size (in threads) for the target device.
``warpSize`` should only be used in device functions that develop portable wave-aware code.

.. note::
  NVIDIA devices return 32 for this variable; AMD devices return 64 for gfx9 and 32 for gfx10
  and above.

Vector types
====================================================

The following vector types are defined in ``hip_runtime.h``. They are not automatically provided by the
compiler.

Short vector types
--------------------------------------------------------------------------------------------

Short vector types derive from basic integer and floating-point types. These structures are defined in
``hip_vector_types.h``. The first, second, third, and fourth components of the vector are defined by the
``x``, ``y``, ``z``, and ``w`` fields, respectively. All short vector types support a constructor function of the
form ``make_<type_name>()``. For example, ``float4 make_float4(float x, float y, float z, float w)`` creates
a vector with type ``float4`` and value ``(x,y,z,w)``.

HIP supports the following short vector formats:

* Signed Integers:

  * ``char1``, ``char2``, ``char3``, ``char4``
  * ``short1``, ``short2``, ``short3``, ``short4``
  * ``int1``, ``int2``, ``int3``, ``int4``
  * ``long1``, ``long2``, ``long3``, ``long4``
  * ``longlong1``, ``longlong2``, ``longlong3``, ``longlong4``

* Unsigned Integers:

  * ``uchar1``, ``uchar2``, ``uchar3``, ``uchar4``
  * ``ushort1``, ``ushort2``, ``ushort3``, ``ushort4``
  * ``uint1``, ``uint2``, ``uint3``, ``uint4``
  * ``ulong1``, ``ulong2``, ``ulong3``, ``ulong4``
  * ``ulonglong1``, ``ulonglong2``, ``ulonglong3``, ``ulonglong4``

* Floating Points:

  * ``float1``, ``float2``, ``float3``, ``float4``
  * ``double1``, ``double2``, ``double3``, ``double4``

.. _dim3:

dim3
--------------------------------------------------------------------------------------------

``dim3`` is a three-dimensional integer vector type that is commonly used to specify grid and group
dimensions.

The dim3 constructor accepts between zero and three arguments. By default, it initializes unspecified
dimensions to 1.

.. code:: cpp

  typedef struct dim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    dim3(uint32_t _x=1, uint32_t _y=1, uint32_t _z=1) : x(_x), y(_y), z(_z) {};
  };


Memory fence instructions
====================================================

HIP supports ``__threadfence()`` and ``__threadfence_block()``. If you're using ``threadfence_system()`` in
the HIP-Clang path, you can use the following workaround:

#. Build HIP with the ``HIP_COHERENT_HOST_ALLOC`` environment variable enabled.
#. Modify kernels that use ``__threadfence_system()`` as follows:

  * Ensure the kernel operates only on fine-grained system memory, which should be allocated with
    ``hipHostMalloc()``.
  * Remove ``memcpy`` for all allocated fine-grained system memory regions.

.. _synchronization-functions:

Synchronization functions
====================================================
The ``__syncthreads()`` built-in function is supported in HIP. The ``__syncthreads_count(int)``,
``__syncthreads_and(int)``, and ``__syncthreads_or(int)`` functions are under development.

Math functions
====================================================
HIP-Clang supports a set of math operations that are callable from the device. These are described in
the following sections.

.. doxygengroup:: Math
   :inner:

Texture functions
===============================================

The supported texture functions are listed in ``texture_fetch_functions.h`` and
``texture_indirect_functions.h`` header files in the
`HIP-AMD backend repository <https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail>`_.

Texture functions are not supported on some devices. To determine if texture functions are supported
on your device, use ``Macro __HIP_NO_IMAGE_SUPPORT == 1``. You can query the attribute
``hipDeviceAttributeImageSupport`` to check if texture functions are supported in the host runtime
code.

Surface functions
===============================================

Surface functions are not supported.

Timer functions
===============================================

To read a high-resolution timer from the device, HIP provides the following built-in functions:

* Returning the incremental counter value for every clock cycle on a device:

  .. code:: cpp

    clock_t clock()
    long long int clock64()

  The difference between the values that are returned represents the cycles used.

* Returning the wall clock count at a constant frequency on the device:

  .. code:: cpp

    long long int wall_clock64()

  This can be queried using the HIP API with the ``hipDeviceAttributeWallClockRate`` attribute of the
  device in HIP application code. For example:

  .. code:: cpp

    int wallClkRate = 0; //in kilohertz
    HIPCHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, deviceId));

  Where ``hipDeviceAttributeWallClockRate`` is a device attribute. Note that wall clock frequency is a
  per-device attribute.

Atomic functions
===============================================

Atomic functions are run as read-modify-write (RMW) operations that reside in global or shared
memory. No other device or thread can observe or modify the memory location during an atomic
operation. If multiple instructions from different devices or threads target the same memory location,
the instructions are serialized in an undefined order.

To support system scope atomic operations, you can use the HIP APIs that contain the ``_system`` suffix.
For example:
* ``atomicAnd``: This function is atomic and coherent within the GPU device running the function
* ``atomicAnd_system``: This function extends the atomic operation from the GPU device to other CPUs
and GPU devices in the system

HIP supports the following atomic operations.

.. list-table::
    * - **Function**
    - **Supported in HIP**
    - **Supported in CUDA**

    * - int atomicAdd(int* address, int val)
      - ✓
      - ✓

    * - int atomicAdd_system(int* address, int val)
      - ✓
      - ✓

    * - unsigned int atomicAdd(unsigned int* address,unsigned int val)
      - ✓
      - ✓

    * - unsigned int atomicAdd_system(unsigned int* address, unsigned int val)
      - ✓
      - ✓

    * - unsigned long long atomicAdd(unsigned long long* address,unsigned long long val)
      - ✓
      - ✓

    * - unsigned long long atomicAdd_system(unsigned long long* address, unsigned long long val)
      - ✓
      - ✓

    * - float atomicAdd(float* address, float val)
      - ✓
      - ✓

    * - float atomicAdd_system(float* address, float val)
      - ✓
      - ✓

    * - double atomicAdd(double* address, double val)
      - ✓
      - ✓

    * - double atomicAdd_system(double* address, double val)
      - ✓
      - ✓

    * - float unsafeAtomicAdd(float* address, float val)
      - ✓
      - ✗

    * - float safeAtomicAdd(float* address, float val)
      - ✓
      - ✗

    * - double unsafeAtomicAdd(double* address, double val)
      - ✓
      - ✗

    * - double safeAtomicAdd(double* address, double val)
      - ✓
      - ✗

    * - int atomicSub(int* address, int val)
      - ✓
      - ✓

    * - int atomicSub_system(int* address, int val)
      - ✓
      - ✓

    * - unsigned int atomicSub(unsigned int* address,unsigned int val)
      - ✓
      - ✓

    * - unsigned int atomicSub_system(unsigned int* address, unsigned int val)
      - ✓
      - ✓

    * - int atomicExch(int* address, int val)
      - ✓
      - ✓

    * - int atomicExch_system(int* address, int val)
      - ✓
      - ✓

    * - unsigned int atomicExch(unsigned int* address,unsigned int val)
      - ✓
      - ✓

    * - unsigned int atomicExch_system(unsigned int* address, unsigned int val)
      - ✓
      - ✓

    * - unsigned long long atomicExch(unsigned long long int* address,unsigned long long int val)
      - ✓
      - ✓

    * - unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val)
      - ✓
      - ✓

    * - unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val)
      - ✓
      - ✓

    * - float atomicExch(float* address, float val)
      - ✓
      - ✓

    * - int atomicMin(int* address, int val)
      - ✓
      - ✓

    * - int atomicMin_system(int* address, int val)
      - ✓
      - ✓

    * - unsigned int atomicMin(unsigned int* address,unsigned int val)
      - ✓
      - ✓

    * - unsigned int atomicMin_system(unsigned int* address, unsigned int val)
      - ✓
      - ✓

    * - unsigned long long atomicMin(unsigned long long* address,unsigned long long val)
      - ✓
      - ✓

    * - int atomicMax(int* address, int val)
      - ✓
      - ✓

    * - int atomicMax_system(int* address, int val)
      - ✓
      - ✓

    * - unsigned int atomicMax(unsigned int* address,unsigned int val)
      - ✓
      - ✓

    * - unsigned int atomicMax_system(unsigned int* address, unsigned int val)
      - ✓
      - ✓

    * - unsigned long long atomicMax(unsigned long long* address,unsigned long long val)
      - ✓
      - ✓

    * - unsigned int atomicInc(unsigned int* address)
      - ✗
      - ✓

    * - unsigned int atomicDec(unsigned int* address)
      - ✗
      - ✓

    * - int atomicCAS(int* address, int compare, int val)
      - ✓
      - ✓

    * - int atomicCAS_system(int* address, int compare, int val)
      - ✓
      - ✓

    * - unsigned int atomicCAS(unsigned int* address,unsigned int compare,unsigned int val)
      - ✓
      - ✓

    * - unsigned int atomicCAS_system(unsigned int* address, unsigned int compare, unsigned int val)
      - ✓
      - ✓

    * - unsigned long long atomicCAS(unsigned long long* address,unsigned long long compare,unsigned long long val)
      - ✓
      - ✓

    * - unsigned long long atomicCAS_system(unsigned long long* address, unsigned long long compare, unsigned long long val)
      - ✓
      - ✓

    * - int atomicAnd(int* address, int val)
      - ✓
      - ✓

    * - int atomicAnd_system(int* address, int val)
      - ✓
      - ✓

    * - unsigned int atomicAnd(unsigned int* address,unsigned int val)
      - ✓
      - ✓

    * - unsigned int atomicAnd_system(unsigned int* address, unsigned int val)
      - ✓
      - ✓

    * - unsigned long long atomicAnd(unsigned long long* address,unsigned long long val)
      - ✓
      - ✓

    * - unsigned long long atomicAnd_system(unsigned long long* address, unsigned long long val)
      - ✓
      - ✓

    * - int atomicOr(int* address, int val)
      - ✓
      - ✓

    * - int atomicOr_system(int* address, int val)
      - ✓
      - ✓

    * - unsigned int atomicOr(unsigned int* address,unsigned int val)
      - ✓
      - ✓

    * - unsigned int atomicOr_system(unsigned int* address, unsigned int val)
      - ✓
      - ✓

    * - unsigned int atomicOr_system(unsigned int* address, unsigned int val)
      - ✓
      - ✓

    * - unsigned long long atomicOr(unsigned long long int* address,unsigned long long val)
      - ✓
      - ✓

    * - unsigned long long atomicOr_system(unsigned long long* address, unsigned long long val)
      - ✓
      - ✓

    * - int atomicXor(int* address, int val)
      - ✓
      - ✓

    * - int atomicXor_system(int* address, int val)
      - ✓
      - ✓

    * - unsigned int atomicXor(unsigned int* address,unsigned int val)
      - ✓
      - ✓

    * - unsigned int atomicXor_system(unsigned int* address, unsigned int val)
      - ✓
      - ✓

    * - unsigned long long atomicXor(unsigned long long* address,unsigned long long val)
      - ✓
      - ✓

    * - unsigned long long atomicXor_system(unsigned long long* address, unsigned long long val)
      - ✓
      - ✓

Unsafe floating-point atomic RMW operations
----------------------------------------------------------------------------------------------------------------
Some HIP devices support fast atomic RMW operations on floating-point values. For example,
``atomicAdd`` on single- or double-precision floating-point values may generate a hardware RMW
instruction that is faster than emulating the atomic operation using an atomic compare-and-swap
(CAS) loop.

On some devices, fast atomic RMW instructions can produce results that differ from the same
functions implemented with atomic CAS loops. For example, some devices will use different rounding
or denormal modes, and some devices produce incorrect answers if fast floating-point atomic RMW
instructions target fine-grained memory allocations.

The HIP-Clang compiler offers a compile-time option, so you can choose fast--but potentially
unsafe--atomic instructions for your code. On devices that support these instructions, you can include
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
========================================================

Warp cross-lane functions operate across all lanes in a warp. The hardware guarantees that all warp
lanes are run in lockstep, meaning that additional synchronization is unnecessary. The instructions
don't use shared memory.

Note that NVIDIA and AMD devices have different warp sizes. You can use ``warpSize`` built-ins in you
portable code to query the warp size.

.. tip::
  Be sure to review HIP code generated from the CUDA path to ensure that it doesn't assume a
  ``waveSize`` of 32. "Wave-aware" code that assumes a ``waveSize`` of 32 can run on a wave-64
  machine, but it only utilizes half of the machine's resources.

To get the default warp size of a GPU device, use ``hipGetDeviceProperties`` in you host functions.

.. code:: cpp

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceID);
    int w = props.warpSize;
    // implement portable algorithm based on w (rather than assume 32 or 64)

Only use ``warpSize`` built-ins in device functions, and don't assume ``warpSize`` to be a compile-time
constant.

Note that assembly kernels may be built for a warp size that is different from the default.

Warp vote and ballot functions
-------------------------------------------------------------------------------------------------------------

.. code:: cpp

  int __all(int predicate)
  int __any(int predicate)
  uint64_t __ballot(int predicate)

Threads in a warp are referred to as *lanes* and are numbered from 0 to :math:` warpSize - 1`. Each
warp lane contributes 1 minus the bit value (the predicate), which is efficiently broadcast to all lanes in
the warp.

The 32-bit int predicate from each lane reduces to a 1-bit value of 0 ``(predicate = 0)`` or 1
``(predicate != 0)``. To get a summary view of the predicates that are contributed by other warp lanes, you
can use:

* ``__any()``: Returns 1 if any warp lane contributes a nonzero predicate, otherwise it returns 0
* ``__all()``: Returns 1 if all other warp lanes contribute nonzero predicates, otherwise it returns 0

To determine if the target platform supports the any/all instruction, you can use the ``hasWarpVote``
device property or the ``HIP_ARCH_HAS_WARP_VOTE`` compiler definition.

HIP's ``__ballot`` function provides a bit mask that contains the 1-bit predicate value from each lane.
The nth bit of this result contains the 1 bit contributed by the nth warp lane. Note that ``__ballot``
supports a 64-bit return value (versus CUDA's 32 bits). Code ported from CUDA should support these
larger warp sizes.

To determine if the target platform supports the ballot instruction, you ca use the ``hasWarpBallot``
device property or the ``HIP_ARCH_HAS_WARP_BALLOT`` compiler definition.

Warp shuffle functions
-------------------------------------------------------------------------------------------------------------

The default width is ``warpSize`` (see :ref:`warp-cross-lane`). Half-float shuffles are not supported.

.. code:: cpp

  int   __shfl      (int var,   int srcLane, int width=warpSize);
  float __shfl      (float var, int srcLane, int width=warpSize);
  int   __shfl_up   (int var,   unsigned int delta, int width=warpSize);
  float __shfl_up   (float var, unsigned int delta, int width=warpSize);
  int   __shfl_down (int var,   unsigned int delta, int width=warpSize);
  float __shfl_down (float var, unsigned int delta, int width=warpSize);
  int   __shfl_xor  (int var,   int laneMask, int width=warpSize);
  float __shfl_xor  (float var, int laneMask, int width=warpSize);

Cooperative groups functions
==============================================================

You can use cooperative groups to synchronize groups of threads. Cooperative groups also provide a
way of communicating between groups of threads at a granularity that is different from the block.

HIP supports the following kernel language cooperative groups types and functions:

.. list-table::
    * - **Function**
    - **Supported in HIP**
    - **Supported in CUDA**

    * - void thread_group.sync();
      - ✓
      - ✓

    * - unsigned thread_group.size();
      - ✓
      - ✓

    * - unsigned thread_group.thread_rank()
      - ✓
      - ✓

    * - bool thread_group.is_valid();
      - ✓
      - ✓

    * - grid_group this_grid()
      - ✓
      - ✓

    * - void grid_group.sync()
      - ✓
      - ✓

    * - unsigned grid_group.size()
      - ✓
      - ✓

    * - unsigned grid_group.thread_rank()
      - ✓
      - ✓

    * - bool grid_group.is_valid()
      - ✓
      - ✓

    * - multi_grid_group this_multi_grid()
      - ✓
      - ✓

    * - void multi_grid_group.sync()
      - ✓
      - ✓

    * - unsigned multi_grid_group.size()
      - ✓
      - ✓

    * - unsigned multi_grid_group.thread_rank()
      - ✓
      - ✓

    * - bool multi_grid_group.is_valid()
      - ✓
      - ✓

    * - unsigned multi_grid_group.num_grids()
      - ✓
      - ✓

    * - unsigned multi_grid_group.grid_rank()
      - ✓
      - ✓

    * - thread_block this_thread_block()
      - ✓
      - ✓

    * - multi_grid_group this_multi_grid()
      - ✓
      - ✓

    * - void multi_grid_group.sync()
      - ✓
      - ✓

    * - void thread_block.sync()
      - ✓
      - ✓

    * - unsigned thread_block.size()
      - ✓
      - ✓

    * - unsigned thread_block.thread_rank()
      - ✓
      - ✓

    * - bool thread_block.is_valid()
      - ✓
      - ✓

    * - dim3 thread_block.group_index()
      - ✓
      - ✓

    * - dim3 thread_block.thread_index()
      - ✓
      - ✓

Warp matrix functions
============================================================

Warp matrix functions allow a warp to cooperatively operate on small matrices that have elements
spread over lanes in an unspecified manner.

HIP does not support kernel language warp matrix types or functions.

.. list-table::
    * - **Function**
    - **Supported in HIP**
    - **Supported in CUDA**

    * - void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned lda)
      - ✗
      - ✓

    * - void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned lda, layout_t layout)
      - ✗
      - ✓

    * - void store_matrix_sync(T* mptr, fragment<...> &a,  unsigned lda, layout_t layout)
      - ✗
      - ✓

    * - void fill_fragment(fragment<...> &a, const T &value)
      - ✗
      - ✓

    * - void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c , bool sat)
      - ✗
      - ✓

Independent thread scheduling
============================================================

Certain architectures that support CUDA allow threads to progress independently of each other. This
independent thread scheduling makes intra-warp synchronization possible.

HIP does not support this type of scheduling.

Profiler Counter Function
============================================================

The CUDA `__prof_trigger()` instruction is not supported.

Assert
============================================================

The assert function is supported in HIP.
Assert function is used for debugging purpose, when the input expression equals to zero, the execution will be stopped.
.. code:: cpp

  void assert(int input)

There are two kinds of implementations for assert functions depending on the use sceneries,
- One is for the host version of assert, which is defined in ``assert.h``,
- Another is the device version of assert, which is implemented in ``hip/hip_runtime.h``.
Users need to include ``assert.h`` to use ``assert``. For assert to work in both device and host functions, users need to include ``"hip/hip_runtime.h"``.

HIP provides the function ``abort()`` which can be used to terminate the application when terminal failures are detected.  It is implemented using the ``__builtin_trap()`` function.

This function produces a similar effect of using ``asm("trap")`` in the CUDA code.

Printf
============================================================

Printf function is supported in HIP.
The following is a simple example to print information in the kernel.

.. code:: cpp

  #include <hip/hip_runtime.h>
  
  __global__ void run_printf() { printf("Hello World\n"); }
  
  int main() {
    run_printf<<<dim3(1), dim3(1), 0, 0>>>();
  }


Device-Side Dynamic Global Memory Allocation
============================================================

Device-side dynamic global memory allocation is under development.  HIP now includes a preliminary
implementation of malloc and free that can be called from device functions.

`__launch_bounds__`
============================================================

GPU multiprocessors have a fixed pool of resources (primarily registers and shared memory) which are shared by the actively running warps. Using more resources can increase IPC of the kernel but reduces the resources available for other warps and limits the number of warps that can be simulaneously running. Thus GPUs have a complex relationship between resource usage and performance.

__launch_bounds__ allows the application to provide usage hints that influence the resources (primarily registers) used by the generated code.  It is a function attribute that must be attached to a __global__ function:

.. code:: cpp

  __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)
  MyKernel(hipGridLaunch lp, ...)
  ...

__launch_bounds__ supports two parameters:
- MAX_THREADS_PER_BLOCK - The programmers guarantees that kernel will be launched with threads less than MAX_THREADS_PER_BLOCK. (On NVCC this maps to the .maxntid PTX directive). If no launch_bounds is specified, MAX_THREADS_PER_BLOCK is the maximum block size supported by the device (typically 1024 or larger). Specifying MAX_THREADS_PER_BLOCK less than the maximum effectively allows the compiler to use more resources than a default unconstrained compilation that supports all possible block sizes at launch time.
The threads-per-block is the product of (blockDim.x * blockDim.y * blockDim.z).
- MIN_WARPS_PER_EXECUTION_UNIT - directs the compiler to minimize resource usage so that the requested number of warps can be simultaneously active on a multi-processor. Since active warps compete for the same fixed pool of resources, the compiler must reduce resources required by each warp(primarily registers). MIN_WARPS_PER_EXECUTION_UNIT is optional and defaults to 1 if not specified. Specifying a MIN_WARPS_PER_EXECUTION_UNIT greater than the default 1 effectively constrains the compiler's resource usage.

When launch kernel with HIP APIs, for example, hipModuleLaunchKernel(), HIP will do validation to make sure input kernel dimension size is not larger than specified launch_bounds.
In case exceeded, HIP would return launch failure, if AMD_LOG_LEVEL is set with proper value (for details, please refer to docs/markdown/hip_logging.md), detail information will be shown in the error log message, including
launch parameters of kernel dim size, launch bounds, and the name of the faulting kernel. It's helpful to figure out which is the faulting kernel, besides, the kernel dim size and launch bounds values will also assist in debugging such failures.

Compiler Impact
--------------------------------------------------------------------------------------------

The compiler uses these parameters as follows:
- The compiler uses the hints only to manage register usage, and does not automatically reduce shared memory or other resources.
- Compilation fails if compiler cannot generate a kernel which meets the requirements of the specified launch bounds.
- From MAX_THREADS_PER_BLOCK, the compiler derives the maximum number of warps/block that can be used at launch time.
Values of MAX_THREADS_PER_BLOCK less than the default allows the compiler to use a larger pool of registers : each warp uses registers, and this hint constains the launch to a warps/block size which is less than maximum.
- From MIN_WARPS_PER_EXECUTION_UNIT, the compiler derives a maximum number of registers that can be used by the kernel (to meet the required #simultaneous active blocks).
If MIN_WARPS_PER_EXECUTION_UNIT is 1, then the kernel can use all registers supported by the multiprocessor.
- The compiler ensures that the registers used in the kernel is less than both allowed maximums, typically by spilling registers (to shared or global memory), or by using more instructions.
- The compiler may use hueristics to increase register usage, or may simply be able to avoid spilling. The MAX_THREADS_PER_BLOCK is particularly useful in this cases, since it allows the compiler to use more registers and avoid situations where the compiler constrains the register usage (potentially spilling) to meet the requirements of a large block size that is never used at launch time.

CU and EU Definitions
--------------------------------------------------------------------------------------------

A compute unit (CU) is responsible for executing the waves of a work-group. It is composed of one or more execution units (EU) which are responsible for executing waves. An EU can have enough resources to maintain the state of more than one executing wave. This allows an EU to hide latency by switching between waves in a similar way to symmetric multithreading on a CPU. In order to allow the state for multiple waves to fit on an EU, the resources used by a single wave have to be limited. Limiting such resources can allow greater latency hiding, but can result in having to spill some register state to memory. This attribute allows an advanced developer to tune the number of waves that are capable of fitting within the resources of an EU. It can be used to ensure at least a certain number will fit to help hide latency, and can also be used to ensure no more than a certain number will fit to limit cache thrashing.

Porting from CUDA `__launch_bounds`
--------------------------------------------------------------------------------------------

CUDA defines a __launch_bounds which is also designed to control occupancy:

.. code:: cpp

  __launch_bounds(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)

- The second parameter __launch_bounds parameters must be converted to the format used __hip_launch_bounds, which uses warps and execution-units rather than blocks and multi-processors (this conversion is performed automatically by HIPIFY tools).

.. code:: cpp

  MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK) / 32

The key differences in the interface are:
- Warps (rather than blocks):
The developer is trying to tell the compiler to control resource utilization to guarantee some amount of active Warps/EU for latency hiding.  Specifying active warps in terms of blocks appears to hide the micro-architectural details of the warp size, but makes the interface more confusing since the developer ultimately needs to compute the number of warps to obtain the desired level of control.
- Execution Units  (rather than multiProcessor):
The use of execution units rather than multiprocessors provides support for architectures with multiple execution units/multi-processor. For example, the AMD GCN architecture has 4 execution units per multiProcessor.  The hipDeviceProps has a field executionUnitsPerMultiprocessor.
Platform-specific coding techniques such as #ifdef can be used to specify different launch_bounds for NVCC and HIP-Clang platforms, if desired.

maxregcount
--------------------------------------------------------------------------------------------

Unlike nvcc, HIP-Clang does not support the "--maxregcount" option.  Instead, users are encouraged to use the hip_launch_bounds directive since the parameters are more intuitive and portable than
micro-architecture details like registers, and also the directive allows per-kernel control rather than an entire file.  hip_launch_bounds works on both HIP-Clang and nvcc targets.

Asynchronous Functions
============================================================

Memory stream
--------------------------------------------------------------------------------------------

.. doxygengroup:: Stream
   :content-only:

.. doxygengroup:: StreamO
   :content-only:

Peer to peer
--------------------------------------------------------------------------------------------

.. doxygengroup:: PeerToPeer
   :content-only:

Memory management
--------------------------------------------------------------------------------------------

.. doxygengroup:: Memory
   :content-only:

External Resource Interoperability
--------------------------------------------------------------------------------------------

.. doxygengroup:: External
   :content-only:

Register Keyword
============================================================

The register keyword is deprecated in C++, and is silently ignored by both nvcc and HIP-Clang.  You can pass the option `-Wdeprecated-register` the compiler warning message.

Pragma Unroll
============================================================

Unroll with a bounds that is known at compile-time is supported.  For example:

.. code:: cpp

  #pragma unroll 16 /* hint to compiler to unroll next loop by 16 */
  for (int i=0; i<16; i++) ...

.. code:: cpp

  #pragma unroll 1  /* tell compiler to never unroll the loop */
  for (int i=0; i<16; i++) ...

.. code:: cpp

  #pragma unroll /* hint to compiler to completely unroll next loop. */
  for (int i=0; i<16; i++) ...

In-Line Assembly
============================================================

GCN ISA In-line assembly, is supported. For example:

.. code:: cpp

  asm volatile ("v_mac_f32_e32 %0, %2, %3" : "=v" (out[i]) : "0"(out[i]), "v" (a), "v" (in[i]));

We insert the GCN isa into the kernel using `asm()` Assembler statement.
`volatile` keyword is used so that the optimizers must not change the number of volatile operations or change their order of execution relative to other volatile operations.
`v_mac_f32_e32` is the GCN instruction, for more information please refer - [AMD GCN3 ISA architecture manual](http://gpuopen.com/compute-product/amd-gcn3-isa-architecture-manual/)
Index for the respective operand in the ordered fashion is provided by `%` followed by position in the list of operands
`"v"` is the constraint code (for target-specific AMDGPU) for 32-bit VGPR register, for more info please refer - [Supported Constraint Code List for AMDGPU](https://llvm.org/docs/LangRef.html#supported-constraint-code-list)
Output Constraints are specified by an `"="` prefix as shown above ("=v"). This indicate that assemby will write to this operand, and the operand will then be made available as a return value of the asm expression. Input constraints do not have a prefix - just the constraint code. The constraint string of `"0"` says to use the assigned register for output as an input as well (it being the 0'th constraint).

## C++ Support
The following C++ features are not supported:
- Run-time-type information (RTTI)
- Try/catch
- Virtual functions
Virtual functions are not supported if objects containing virtual function tables are passed between GPU's of different offload arch's, e.g. between gfx906 and gfx1030. Otherwise virtual functions are supported.

Kernel Compilation
============================================================
hipcc now supports compiling C++/HIP kernels to binary code objects.
The file format for binary is `.co` which means Code Object. The following command builds the code object using `hipcc`.

.. code:: bash

  hipcc --genco --offload-arch=[TARGET GPU] [INPUT FILE] -o [OUTPUT FILE]

  [TARGET GPU] = GPU architecture
  [INPUT FILE] = Name of the file containing kernels
  [OUTPUT FILE] = Name of the generated code object file

.. note::
  When using binary code objects is that the number of arguments to the kernel is different on HIP-Clang and NVCC path. Refer to the sample in samples/0_Intro/module_api for differences in the arguments to be passed to the kernel.

gfx-arch-specific-kernel
============================================================
Clang defined '__gfx*__' macros can be used to execute gfx arch specific codes inside the kernel. Refer to the sample ``14_gpu_arch`` in ``samples/2_Cookbook``.
