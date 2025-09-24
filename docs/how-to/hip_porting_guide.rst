.. meta::
  :description: This chapter presents how to port CUDA source code to HIP.
  :keywords: AMD, ROCm, HIP, CUDA, porting, port

********************************************************************************
HIP porting guide
********************************************************************************

HIP is designed to ease the porting of existing CUDA code into the HIP
environment. This page describes the available tools and provides practical
suggestions on how to port CUDA code and work through common issues.

Porting a CUDA Project
================================================================================

Mixing HIP and CUDA code results in valid CUDA code. This enables users to
incrementally port CUDA to HIP, and still compile and test the code during the
transition.

The only notable exception is ``hipError_t``, which is not just an alias to
``cudaError_t``. In these cases HIP provides functions to convert between the
error code spaces:

* :cpp:func:`hipErrorToCudaError`
* :cpp:func:`hipErrorToCUResult`
* :cpp:func:`hipCUDAErrorTohipError`
* :cpp:func:`hipCUResultTohipError`

General Tips
--------------------------------------------------------------------------------

* ``hipDeviceptr_t`` is a ``void*`` and treated like a raw pointer, while ``CUdevicptr``
  is an ``unsigned int`` and treated as a device memory handle. 
* Starting to port on an NVIDIA machine is often the easiest approach, as the
  code can be tested for functionality and performance even if not fully ported
  to HIP.
* Once the CUDA code is ported to HIP and is running on the CUDA machine,
  compile the HIP code for an AMD machine.
* You can handle platform-specific features through conditional compilation or
  by adding them to the open-source HIP infrastructure.
* Use the `HIPIFY <https://github.com/ROCm/HIPIFY>`_ tools to automatically
  convert CUDA code to HIP, as described in the following section.

HIPIFY
--------------------------------------------------------------------------------

:doc:`HIPIFY <hipify:index>` is a collection of tools that automatically
translate CUDA to HIP code. There are two flavours available, ``hipfiy-clang``
and ``hipify-perl``.

:doc:`hipify-clang <hipify:how-to/hipify-clang>` is, as the name implies, a Clang-based
tool, and actually parses the code, translates it into an Abstract Syntax Tree,
from which it then generates the HIP source. For this, ``hipify-clang`` needs to
be able to actually compile the code, so the CUDA code needs to be correct, and
a CUDA install with all necessary headers must be provided.

:doc:`hipify-perl <hipify:how-to/hipify-perl>` uses pattern matching, to translate the
CUDA code to HIP. It does not require a working CUDA installation, and can also
convert CUDA code, that is not syntactically correct. It is therefore easier to
set up and use, but is not as powerful as ``hipify-clang``.

Scanning existing CUDA code to scope the porting effort
--------------------------------------------------------------------------------

The ``--examine`` option, supported by the clang and perl version, tells hipify
to do a test-run, without changing the files, but instead scan CUDA code to
determine which files contain CUDA code and how much of that code can
automatically be hipified.

There also are ``hipexamine-perl.sh`` or ``hipexamine.sh`` (for
``hipify-clang``) scripts to automatically scan directories.

For example, the following is a scan of one of the
`cuda-samples <https://github.com/NVIDIA/cuda-samples>`_:

.. code-block:: shell

  > cd Samples/2_Concepts_and_Techniques/convolutionSeparable/
  > hipexamine-perl.sh
  [HIPIFY] info: file './convolutionSeparable.cu' statistics:
    CONVERTED refs count: 2
    TOTAL lines of code: 214
    WARNINGS: 0
  [HIPIFY] info: CONVERTED refs by names:
    cooperative_groups.h => hip/hip_cooperative_groups.h: 1
    cudaMemcpyToSymbol => hipMemcpyToSymbol: 1
  
  [HIPIFY] info: file './main.cpp' statistics:
    CONVERTED refs count: 13
    TOTAL lines of code: 174
    WARNINGS: 0
  [HIPIFY] info: CONVERTED refs by names:
    cudaDeviceSynchronize => hipDeviceSynchronize: 2
    cudaFree => hipFree: 3
    cudaMalloc => hipMalloc: 3
    cudaMemcpy => hipMemcpy: 2
    cudaMemcpyDeviceToHost => hipMemcpyDeviceToHost: 1
    cudaMemcpyHostToDevice => hipMemcpyHostToDevice: 1
    cuda_runtime.h => hip/hip_runtime.h: 1
  
  [HIPIFY] info: file 'GLOBAL' statistics:
    CONVERTED refs count: 15
    TOTAL lines of code: 512
    WARNINGS: 0
  [HIPIFY] info: CONVERTED refs by names:
    cooperative_groups.h => hip/hip_cooperative_groups.h: 1
    cudaDeviceSynchronize => hipDeviceSynchronize: 2
    cudaFree => hipFree: 3
    cudaMalloc => hipMalloc: 3
    cudaMemcpy => hipMemcpy: 2
    cudaMemcpyDeviceToHost => hipMemcpyDeviceToHost: 1
    cudaMemcpyHostToDevice => hipMemcpyHostToDevice: 1
    cudaMemcpyToSymbol => hipMemcpyToSymbol: 1
    cuda_runtime.h => hip/hip_runtime.h: 1

``hipexamine-perl.sh`` reports how many CUDA calls are going to be converted to
HIP (e.g. ``CONVERTED refs count: 2``), and lists them by name together with
their corresponding HIP-version (see the lines following ``[HIPIFY] info:
CONVERTED refs by names:``). It also lists the total lines of code for the file
and potential warnings. In the end it prints a summary for all files.

Automatically converting a CUDA project
--------------------------------------------------------------------------------

To directly replace the files, the ``--inplace`` option of ``hipify-perl`` or
``hipify-clang`` can be used. This creates a backup of the original files in a
``<filename>.prehip`` file and overwrites the existing files, keeping their file
endings. If the ``--inplace`` option is not given, the scripts print the
hipified code to ``stdout``.

``hipconvertinplace.sh``or  ``hipconvertinplace-perl.sh`` operate on whole
directories.

Library Equivalents
--------------------------------------------------------------------------------

ROCm provides libraries to ease porting of code relying on CUDA libraries.
Most CUDA libraries have a corresponding HIP library.

There are two flavours of libraries provided by ROCm, ones prefixed with ``hip``
and ones prefixed with ``roc``. While both are written using HIP, in general
only the ``hip``-libraries are portable. The libraries with the ``roc``-prefix
might also run on CUDA-capable GPUs, however they have been optimized for AMD
GPUs and might use assembly code or a different API, to achieve the best
performance.

.. note::

  If the application is only required to run on AMD GPUs, it is recommended to
  use the ``roc``-libraries.

In the case where a library provides a ``roc``- and a ``hip``- version, the
``hip`` version is a marshalling library, which is just a thin layer that is
redirecting the function calls to either the ``roc``-library or the
corresponding CUDA library, depending on the platform, to provide compatibility.

.. list-table::
  :header-rows: 1

  *
   - CUDA Library
   - ``hip`` Library
   - ``roc`` Library
   - Comment
  *
   - cuBLAS
   - `hipBLAS <https://github.com/ROCm/hipBLAS>`_
   - `rocBLAS <https://github.com/ROCm/rocBLAS>`_
   - Basic Linear Algebra Subroutines
  *
   - cuBLASLt
   - `hipBLASLt <https://github.com/ROCm/hipBLASLt>`_
   -
   - Linear Algebra Subroutines, lightweight and new flexible API
  *
   - cuFFT
   - `hipFFT <https://github.com/ROCm/hipFFT>`_
   - `rocFFT <https://github.com/ROCm/rocfft>`_
   - Fast Fourier Transfer Library
  *
   - cuSPARSE
   - `hipSPARSE <https://github.com/ROCm/hipSPARSE>`_
   - `rocSPARSE <https://github.com/ROCm/rocSPARSE>`_
   - Sparse BLAS + SPMV
  *
   - cuSOLVER
   - `hipSOLVER <https://github.com/ROCm/hipsolver>`_
   - `rocSOLVER <https://github.com/ROCm/rocsolver>`_
   - Lapack library
  *
   - AmgX
   -
   - `rocALUTION <https://github.com/ROCm/rocalution>`_
   - Sparse iterative solvers and preconditioners with algebraic multigrid
  *
   - Thrust
   -
   - `rocThrust <https://github.com/ROCm/rocThrust>`_
   - C++ parallel algorithms library
  *
   - CUB
   - `hipCUB <https://github.com/ROCm/hipcub>`_
   - `rocPRIM <https://github.com/ROCm/rocPRIM>`_
   - Low Level Optimized Parallel Primitives
  *
   - cuDNN
   -
   - `MIOpen <https://github.com/ROCm/MIOpen>`_
   - Deep learning Solver Library
  *
   - cuRAND
   - `hipRAND <https://github.com/ROCm/hiprand>`_
   - `rocRAND <https://github.com/ROCm/rocrand>`_
   - Random Number Generator Library
  *
   - NCCL
   -
   - `RCCL <https://github.com/ROCm/rccl>`_
   - Communications Primitives Library based on the MPI equivalents
     RCCL is a drop-in replacement for NCCL

Distinguishing compilers and platforms
================================================================================

Identifying the HIP Target Platform
--------------------------------------------------------------------------------

HIP projects can target either the AMD or NVIDIA platform. The platform affects
which backend-headers are included and which libraries are used for linking. The
created binaries are not portable between AMD and NVIDIA platforms.

To write code that is specific to a platform the C++-macros specified in the
following section can be used.

Compiler Defines: Summary
--------------------------------------------------------------------------------

This section lists macros that are defined by compilers and the HIP/CUDA APIs,
and what compiler/platform combinations they are defined for.

The following table lists the macros that can be used when compiling HIP. Most
of these macros are not directly defined by the compilers, but in
``hip_common.h``, which is included by ``hip_runtime.h``.

.. list-table:: HIP-related defines
  :header-rows: 1

  *
   - Macro
   - ``amdclang++``
   - ``nvcc`` when used as backend for ``hipcc``
   - Other (GCC, ICC, Clang, etc.)
  *
   - ``__HIP_PLATFORM_AMD__``
   - Defined
   - Undefined
   - Undefined, needs to be set explicitly
  *
   - ``__HIP_PLATFORM_NVIDIA__``
   - Undefined
   - Defined
   - Undefined, needs to be set explicitly
  *
   - ``__HIPCC__``
   - Defined when compiling ``.hip`` files or specifying ``-x hip``
   - Defined when compiling ``.hip`` files or specifying ``-x hip``
   - Undefined
  *
   - ``__HIP_DEVICE_COMPILE__``
   - 1 if compiling for device
     undefined if compiling for host
   - 1 if compiling for device
     undefined if compiling for host
   - Undefined
  *
   - ``__HIP_ARCH_<FEATURE>__``
   - 0 or 1 depending on feature support of targeted hardware (see :ref:`identifying_device_architecture_features`)
   - 0 or 1 depending on feature support of targeted hardware
   - 0
  *
   - ``__HIP__``
   - Defined when compiling ``.hip`` files or specifying ``-x hip``
   - Undefined
   - Undefined

The following table lists macros related to ``nvcc`` and CUDA as HIP backend.

.. list-table:: NVCC-related defines
  :header-rows: 1

  *
   - Macro
   - ``amdclang++``
   - ``nvcc`` when used as backend for ``hipcc``
   - Other (GCC, ICC, Clang, etc.)
  *
   - ``__CUDACC__``
   - Undefined
   - Defined
   - Undefined
     (Clang defines this when explicitly compiling CUDA code)
  *
   - ``__NVCC__``
   - Undefined
   - Defined
   - Undefined
  *
   - ``__CUDA_ARCH__``  [#cuda_arch]_
   - Undefined
   - Defined in device code
     Integer representing compute capability
     Must not be used in host code
   - Undefined

.. [#cuda_arch] the use of ``__CUDA_ARCH__`` to check for hardware features is
   discouraged, as this is not portable. Use the ``__HIP_ARCH_HAS_<FEATURE>``
   macros instead.

Identifying the compilation target platform
--------------------------------------------------------------------------------

Despite HIP's portability, it can be necessary to tailor code to a specific
platform, in order to provide platform-specific code, or aid in
platform-specific performance improvements.

For this, the ``__HIP_PLATFORM_AMD__`` and ``__HIP_PLATFORM_NVIDIA__`` macros
can be used, e.g.:

.. code-block:: cpp

  #ifdef __HIP_PLATFORM_AMD__
    // This code path is compiled when amdclang++ is used for compilation
  #endif

.. code-block:: cpp

  #ifdef __HIP_PLATFORM_NVIDIA__
    // This code path is compiled when nvcc is used for compilation
    //  Could be compiling with CUDA language extensions enabled (for example, a ".cu file)
    //  Could be in pass-through mode to an underlying host compiler (for example, a .cpp file)
  #endif

When using ``hipcc``, the environment variable ``HIP_PLATFORM`` specifies the
runtime to use. When an AMD graphics driver and an AMD GPU is detected,
``HIP_PLATFORM`` is set to ``amd``. If both runtimes are installed, and a
specific one should be used, or ``hipcc`` can't detect the runtime, the
environment variable has to be set manually.

To explicitly use the CUDA compilation path, use:

.. code-block:: bash

  export HIP_PLATFORM=nvidia
  hipcc main.cpp

Identifying Host or Device Compilation Pass
--------------------------------------------------------------------------------

``amdclang++`` makes multiple passes over the code: one for the host code, and
one each for the device code for every GPU architecture to be compiled for.
``nvcc`` makes two passes over the code: one for host code and one for device
code. 

The ``__HIP_DEVICE_COMPILE__``-macro is defined when the compiler is compiling
for the device.


``__HIP_DEVICE_COMPILE__`` is a portable check that can replace the
``__CUDA_ARCH__``.

.. code-block:: cpp

  #include "hip/hip_runtime.h"
  #include <iostream>

  __host__ __device__ void call_func(){
    #ifdef __HIP_DEVICE_COMPILE__
      printf("device\n");
    #else
      std::cout << "host" << std::endl;
    #endif
  }

  __global__ void test_kernel(){
    call_func();
  }

  int main(int argc, char** argv) {
    test_kernel<<<1, 1, 0, 0>>>();

    call_func();
  }

.. _identifying_device_architecture_features:

Identifying Device Architecture Features
================================================================================

GPUs of different generations and architectures do not all provide the same
level of :doc:`hardware feature support <../reference/hardware_features>`. To
guard device-code using these architecture dependent features, the
``__HIP_ARCH_<FEATURE>__`` C++-macros can be used.

Device Code Feature Identification
--------------------------------------------------------------------------------

Some CUDA code tests ``__CUDA_ARCH__`` for a specific value to determine whether
the GPU supports a certain architectural feature, depending on its compute
capability. This requires knowledge about what ``__CUDA_ARCH__`` supports what
feature set.

HIP simplifies this, by replacing these macros with feature-specific macros, not
architecture specific.

For instance,

.. code-block:: cpp

  //#if __CUDA_ARCH__ >= 130 // does not properly specify, what feature is required, not portable
  #if __HIP_ARCH_HAS_DOUBLES__ == 1 // explicitly specifies, what feature is required, portable between AMD and NVIDIA GPUs
    // device code
  #endif

For host code, the ``__HIP_ARCH_<FEATURE>__`` defines are set to 0, if
``hip_runtime.h`` is included, and undefined otherwise. It should not be relied
upon in host code.

Host Code Feature Identification
--------------------------------------------------------------------------------

Host code must not rely on the ``__HIP_ARCH_<FEATURE>__`` macros, as the GPUs
available to a system can not be known during compile time, and their
architectural features differ.

Host code can query architecture feature flags during runtime, by using
:cpp:func:`hipGetDeviceProperties` or :cpp:func:`hipDeviceGetAttribute`.

.. code-block:: cpp

  #include <hip/hip_runtime.h>
  #include <cstdlib>
  #include <iostream>

  #define HIP_CHECK(expression) {                           \
    const hipError_t err = expression;                      \
    if (err != hipSuccess){                                 \
      std::cout << "HIP Error: " << hipGetErrorString(err)) \
                << " at line " << __LINE__ << std::endl;    \
      std::exit(EXIT_FAILURE);                              \
    }                                                       \
  }

  int main(){
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    int device = 0; // Query first available GPU. Can be replaced with any
                    // integer up to, not including, deviceCount
    hipDeviceProp_t deviceProp;
    HIP_CHECK(hipGetDeviceProperties(&deviceProp, device));

    std::cout << "The queried device ";
    if (deviceProp.arch.hasSharedInt32Atomics) // portable HIP feature query
      std::cout << "supports";
    else
      std::cout << "does not support";
    std::cout << " shared int32 atomic operations" << std::endl;
  }

Table of Architecture Properties
--------------------------------------------------------------------------------

The table below shows the full set of architectural properties that HIP
supports, together with the corresponding macros and device properties.

.. list-table::
  :header-rows: 1

  *
   - Macro (for device code)
   - Device Property (host runtime query)
   - Comment
  *
   - ``__HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__``
   - ``hasGlobalInt32Atomics``
   - 32-bit integer atomics for global memory
  *
   - ``__HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__``
   - ``hasGlobalFloatAtomicExch``
   - 32-bit float atomic exchange for global memory
  *
   - ``__HIP_ARCH_HAS_SHARED_INT32_ATOMICS__``
   - ``hasSharedInt32Atomics``
   - 32-bit integer atomics for shared memory
  *
   - ``__HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__``
   - ``hasSharedFloatAtomicExch``
   - 32-bit float atomic exchange for shared memory
  *
   - ``__HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__``
   - ``hasFloatAtomicAdd``
   - 32-bit float atomic add in global and shared memory
  *
   - ``__HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__``
   - ``hasGlobalInt64Atomics``
   - 64-bit integer atomics for global memory
  *
   - ``__HIP_ARCH_HAS_SHARED_INT64_ATOMICS__``
   - ``hasSharedInt64Atomics``
   - 64-bit integer atomics for shared memory
  *
   - ``__HIP_ARCH_HAS_DOUBLES__``
   - ``hasDoubles``
   - Double-precision floating-point operations
  *
   - ``__HIP_ARCH_HAS_WARP_VOTE__``
   - ``hasWarpVote``
   - Warp vote instructions (``any``, ``all``)
  *
   - ``__HIP_ARCH_HAS_WARP_BALLOT__``
   - ``hasWarpBallot``
   - Warp ballot instructions
  *
   - ``__HIP_ARCH_HAS_WARP_SHUFFLE__``
   - ``hasWarpShuffle``
   - Warp shuffle operations (``shfl_*``)
  *
   - ``__HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__``
   - ``hasFunnelShift``
   - Funnel shift two input words into one
  *
   - ``__HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__``
   - ``hasThreadFenceSystem``
   - :cpp:func:`threadfence_system`
  *
   - ``__HIP_ARCH_HAS_SYNC_THREAD_EXT__``
   - ``hasSyncThreadsExt``
   - :cpp:func:`syncthreads_count`, :cpp:func:`syncthreads_and`, :cpp:func:`syncthreads_or`
  *
   - ``__HIP_ARCH_HAS_SURFACE_FUNCS__``
   - ``hasSurfaceFuncs``
   - Supports :ref:`surface functions <surface_object_reference>`.
  *
   - ``__HIP_ARCH_HAS_3DGRID__``
   - ``has3dGrid``
   - Grids and groups are 3D
  *
   - ``__HIP_ARCH_HAS_DYNAMIC_PARALLEL__``
   - ``hasDynamicParallelism``
   - Ability to launch a kernel from within a kernel

Compilation
================================================================================

``hipcc`` is a portable compiler driver that calls ``nvcc`` or ``amdclang++``
and forwards the appropriate options. It passes options through
to the target compiler. Tools that call ``hipcc`` must ensure the compiler
options are appropriate for the target compiler.

``hipconfig`` is a helpful tool in identifying the current systems platform,
compiler and runtime. It can also help set options appropriately.

As an example, it can provide a path to HIP, in Makefiles for example:

.. code-block:: shell

  HIP_PATH ?= $(shell hipconfig --path)

HIP Headers
--------------------------------------------------------------------------------

The ``hip_runtime.h`` headers define all the necessary types, functions, macros,
etc., needed to compile a HIP program, this includes host as well as device
code. ``hip_runtime_api.h`` is a subset of ``hip_runtime.h``.

CUDA has slightly different contents for these two files. In some cases you may
need to convert hipified code to include the richer ``hip_runtime.h`` instead of
``hip_runtime_api.h``.

Using a Standard C++ Compiler
--------------------------------------------------------------------------------

You can compile ``hip_runtime_api.h`` using a standard C or C++ compiler
(e.g., ``gcc`` or ``icc``).
A source file that is only calling HIP APIs but neither defines nor launches any
kernels can be compiled with a standard host compiler (e.g. ``gcc`` or ``icc``)
even when ``hip_runtime_api.h`` or ``hip_runtime.h`` are included.

The HIP include paths and platform macros (``__HIP_PLATFORM_AMD__`` or
``__HIP_PLATFORM_NVIDIA__``) must be passed to the compiler.

``hipconfig`` can help in finding the necessary options, for example on an AMD
platform:

.. code-block:: bash

  hipconfig --cpp_config
   -D__HIP_PLATFORM_AMD__= -I/opt/rocm/include

``nvcc`` includes some headers by default. ``hipcc`` does not include
default headers, and instead all required files must be explicitly included.

The ``hipify`` tool automatically converts ``cuda_runtime.h`` to
``hip_runtime.h``, and it converts ``cuda_runtime_api.h`` to
``hip_runtime_api.h``, but it may miss nested headers or macros.

warpSize
================================================================================

Code should not assume a warp size of 32 or 64, as that is not portable between
platforms and architectures. The ``warpSize`` built-in should be used in device
code, while the host can query it during runtime via the device properties. See
the :ref:`HIP language extension for warpSize <warp_size>` for information on
how to write portable wave-aware code.

Lane masks bit-shift
================================================================================

A thread in a warp is also called a lane, and a lane mask is a bitmask where
each bit corresponds to a thread in a warp. A bit is 1 if the thread is active,
0 if it's inactive. Bit-shift operations are typically used to create lane masks
and on AMD GPUs the ``warpSize`` can differ between different architectures,
that's why it's essential to use correct bitmask type, when porting code.

Example:

.. code-block:: cpp

  // Get the thread's position in the warp
  unsigned int laneId = threadIdx.x % warpSize;

  // Use lane ID for bit-shift
  val & ((1 << (threadIdx.x % warpSize) )-1 );

  // Shift 32 bit integer with val variable
  WarpReduce::sum( (val < warpSize) ? (1 << val) : 0);

Lane masks are 32-bit integer types as this is the integer precision that C 
assigns to such constants by default. GCN/CDNA architectures have a warp size of
64, :code:`threadIdx.x % warpSize` and :code:`val` in the example may obtain 
values greater than 31. Consequently, shifting by such values would clear the 
32-bit register to which the shift operation is applied. For AMD
architectures, a straightforward fix could look as follows:

.. code-block:: cpp
  
  // Get the thread's position in the warp
  unsigned int laneId = threadIdx.x % warpSize;

  // Use lane ID for bit-shift
  val & ((1ull << (threadIdx.x % warpSize) )-1 );

  // Shift 64 bit integer with val variable
  WarpReduce::sum( (val < warpSize) ? (1ull << val) : 0);

For portability reasons, it is better to introduce appropriately
typed placeholders as shown below:

.. code-block:: cpp

  #if defined(__GFX8__) || defined(__GFX9__)
  typedef uint64_t lane_mask_t;
  #else
  typedef uint32_t lane_mask_t;
  #endif

The use of :code:`lane_mask_t` with the previous example:

.. code-block:: cpp

  // Get the thread's position in the warp
  unsigned int laneId = threadIdx.x % warpSize;

  // Use lane ID for bit-shift
  val & ((lane_mask_t{1} << (threadIdx.x % warpSize) )-1 );

  // Shift 32 or 64 bit integer with val variable
  WarpReduce::sum( (val < warpSize) ? (lane_mask_t{1} << val) : 0);

Porting from CUDA __launch_bounds__
================================================================================

CUDA also defines a ``__launch_bounds__`` qualifier which works similar to HIP's
implementation, however it uses different parameters:

.. code-block:: cpp

  __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)

The first parameter is the same as HIP's implementation, but
``MIN_BLOCKS_PER_MULTIPROCESSOR`` must  be converted to
``MIN_WARPS_PER_EXECUTION_UNIT``, which uses warps and execution units rather than
blocks and multiprocessors. This conversion can be done manually with the equation
considering the mode GPU works:

* In Compute Unit (CU) mode,

.. code-block:: cpp

  MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK) / (warpSize * 2)

* In Workgroup Processor (WGP) mode,

.. code-block:: cpp

  MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK) / (warpSize * 4)

Directly controlling the warps per execution unit makes it easier to reason
about the occupancy, unlike with blocks, where the occupancy depends on the
block size.

The use of execution units rather than multiprocessors also provides support for
architectures with multiple execution units per multiprocessor. For example, the
AMD GCN architecture has 4 execution units per multiprocessor.

maxregcount
--------------------------------------------------------------------------------

Unlike ``nvcc``, ``amdclang++`` does not support the ``--maxregcount`` option.
Instead, users are encouraged to use the ``__launch_bounds__`` directive since
the parameters are more intuitive and portable than micro-architecture details
like registers. The directive allows per-kernel control.
