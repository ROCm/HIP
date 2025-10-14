.. meta::
  :description: This chapter presents how to port the CUDA source code to HIP
  :keywords: AMD, ROCm, HIP, CUDA, driver API, porting, port

.. _porting_cuda_code:

*******************************************************************************
Porting NVIDIA CUDA code to HIP
*******************************************************************************

HIP eases the porting of existing NVIDIA CUDA code into the HIP
environment, enabling you to run your application on AMD GPUs. This topic describes
the available tools and provides practical suggestions for porting your CUDA
code and working through common issues.

CUDA provides separate driver and runtime APIs, while HIP mostly uses a single API.
The two CUDA APIs generally provide similar functionality and are mostly interchangeable.
However, the CUDA driver API provides fine-grained control over kernel-level
initialization, contexts, and module management, while the runtime API automatically
manages contexts and modules. The driver API is suitable for applications that require
tight integration with other systems or advanced control over GPU resources.

* Driver API calls begin with the prefix ``cu``, while runtime API calls begin
  with the prefix ``cuda``. For example, the driver API contains
  ``cuEventCreate``, while the runtime API contains ``cudaEventCreate``, which
  has similar functionality.

* The driver API offers two additional low-level functionalities not exposed by
  the runtime API: module management ``cuModule*`` and context management
  ``cuCtx*`` APIs.

The HIP runtime API includes corresponding functions for both the CUDA driver and
the CUDA runtime API. The module and context functionality are available with the
``hipModule`` and ``hipCtx`` prefixes, and driver API functions are usually
prefixed with ``hipDrv``.

Porting a CUDA project
======================

HIP projects can target either AMD or NVIDIA platforms. HIP is a marshalling language
that provides a thin-layer mapping to functions in the AMD ROCm language, or to CUDA
functions. To compile the HIP code, you can use ``amdclang++``, also called HIP-Clang,
or you can use ``hipcc`` to enable compilation by ``nvcc`` to produce CUDA executables,
as described in :ref:`compilation_platform`. 

Because HIP is a marshalling language that can be compiled by ``nvcc``, mixing HIP code
with CUDA code results in valid application code. This enables users to incrementally port
a CUDA project to HIP, and still compile and test the code during the transition.

The only notable exception is ``hipError_t``, which is not just an alias to
``cudaError_t``. In these cases, HIP provides functions to convert between the
error code spaces:

* :cpp:func:`hipErrorToCudaError`
* :cpp:func:`hipErrorToCUResult`
* :cpp:func:`hipCUDAErrorTohipError`
* :cpp:func:`hipCUResultTohipError`

General Tips
------------

* Starting to port on an NVIDIA machine is often the easiest approach, as the
  code can be tested for functionality and performance even if not fully ported
  to HIP.
* Once the CUDA code is ported to HIP and is running on the CUDA machine,
  compile the HIP code for an AMD machine.
* You can handle platform-specific features through conditional compilation as described
  in :ref:`compilation_platform`.
* Use the `HIPIFY <https://github.com/ROCm/HIPIFY>`_ tools to automatically
  convert CUDA code to HIP, as described in the following section.

Using HIPIFY
============

:doc:`HIPIFY <hipify:index>` is a collection of tools that automatically
translate CUDA code to HIP code. For example, ``cuEventCreate`` is translated to
:cpp:func:`hipEventCreate`. HIPIFY tools also convert error codes from the
driver namespace and coding conventions to the equivalent HIP error code.
HIP unifies the APIs for these common functions.

There are two types of HIPIFY available:

* :doc:`hipify-clang <hipify:how-to/hipify-clang>` is a Clang-based tool that parses code,
  translates it into an Abstract Syntax Tree, and generates the HIP source. For this,
  ``hipify-clang`` needs to be able to actually compile the code, so the CUDA code needs
  to be correct, and a CUDA install with all necessary headers must be provided.

* :doc:`hipify-perl <hipify:how-to/hipify-perl>` uses pattern matching, to translate the
  CUDA code to HIP. It does not require a working CUDA installation, and can also
  convert CUDA code, that is not syntactically correct. It is therefore easier to
  set up and use, but is not as powerful as ``hipfiy-clang``.

Memory copy functions
---------------------

When copying memory, the CUDA driver includes the memory direction in the name of
the API (``cuMemcpyHtoD``), while the CUDA runtime API provides a single memory
copy API with a parameter that specifies the direction. It also supports a
default direction where the runtime determines the direction automatically.

HIP provides both versions, for example, :cpp:func:`hipMemcpyHtoD` as well as
:cpp:func:`hipMemcpy`. The first version might be faster in some cases because
it avoids any host overhead to detect the direction of the memory copy.

Address spaces
--------------

HIP-Clang defines a process-wide address space where
the CPU and all devices allocate addresses from a single unified pool.
This means addresses can be shared between contexts. Unlike CUDA, a new context
does not create a new address space for the device.

Context stack behavior differences
----------------------------------

HIP-Clang creates a primary context when the HIP API is called. In CUDA
driver API code, HIP-Clang creates a primary context while HIP/NVCC has an empty
context stack. HIP-Clang pushes the primary context to the context stack when it
is empty. This can lead to subtle differences in applications which mix the
runtime and driver APIs. 

Scanning CUDA source to scope the translation
---------------------------------------------

The ``--examine`` option, tells the hipify tools to do a test-run without changing
the source files, but instead scanning the files to determine which files contain CUDA code and
how much of that code can automatically be hipified.

There also are ``hipexamine-perl.sh`` or ``hipexamine.sh`` (for
``hipify-clang``) scripts to automatically scan directories.

For example, the following is a scan of one of the ``convolutionSeparable`` sample
from `cuda-samples <https://github.com/NVIDIA/cuda-samples>`_:

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
---------------------------------------

To directly replace the files, the ``--inplace`` option of ``hipify-perl`` or
``hipify-clang`` can be used. This creates a backup of the original files in a
``<filename>.prehip`` file and overwrites the existing files, keeping their file
endings. If the ``--inplace`` option is not given, the scripts print the
hipified code to ``stdout``.

``hipconvertinplace.sh`` or ``hipconvertinplace-perl.sh`` operate on whole
directories.

Library and driver equivalents
==============================

ROCm provides libraries to ease porting of code relying on CUDA libraries or the CUDA driver API.
Most CUDA libraries have a corresponding HIP library. For more information,
see either :doc:`ROCm libraries <rocm:reference/api-libraries>` or :doc:`HIPIFY CUDA compatible libraries <hipify:reference/supported_apis>`.

There are two flavours of libraries provided by ROCm, libraries prefixed with ``hip``
and libraries prefixed with ``roc``. While both are written using HIP, in general
only the ``hip``-libraries are portable. The libraries with the ``roc``-prefix
might also run on CUDA-capable GPUs, however they have been optimized for AMD
GPUs and might use assembly code or a different API, to achieve the best
performance.

In the case where a library provides both ``roc`` and ``hip`` versions, such as
``hipSparse`` and ``rocSparse``, the ``hip`` version is a marshalling library,
which is just a thin layer that redirects function calls to either the
``roc`` library or the corresponding CUDA library, depending on the target platform.  

.. note::

  If the application is only required to run on AMD GPUs, it is recommended to use
  the ``roc``-libraries. In hipify tools, this can be accomplished using the ``--roc`` option. 

cuModule and hipModule
----------------------

The ``cuModule`` feature of the driver API provides additional control over how and
when accelerator code objects are loaded. For example, the driver API enables
code objects to be loaded from files or memory pointers. Symbols for kernels or
global data are extracted from the loaded code objects. In contrast, the runtime
API loads automatically and, if necessary, compiles all the kernels from an
executable binary when it runs. In this mode, kernel code must be compiled using
NVCC so that automatic loading can function correctly.

The Module features are useful in an environment that generates the code objects
directly, such as a new accelerator language front end. NVCC is not used here.
Instead, the environment might have a different kernel language or compilation
flow. Other environments have many kernels and don't want all of them to be
loaded automatically. The Module functions load the generated code objects and
launch kernels. 

Like the ``cuModule`` API, the ``hipModule`` API provides additional control
over code object management, including options to load code from files or from
in-memory pointers. 

NVCC and HIP-Clang target different architectures and use different code object
formats. NVCC supports ``cubin`` or ``ptx`` files, while the HIP-Clang path uses
the ``hsaco`` format. The external compilers that generate these code objects are
responsible for generating and loading the correct code object for each platform.
Notably, there is no fat binary format that can contain code for both NVCC and
HIP-Clang platforms. The following table summarizes the formats used on each
platform:

.. list-table:: Module formats
   :header-rows: 1

   * - Format
     - APIs
     - NVCC
     - HIP-CLANG
   * - Code object
     - ``hipModuleLoad``, ``hipModuleLoadData``
     - ``.cubin`` or PTX text
     - ``.hsaco``
   * - Fat binary
     - ``hipModuleLoadFatBin``
     - ``.fatbin``
     - ``.hip_fatbin``


``hipcc`` uses HIP-Clang or NVCC to compile host code. Both of these compilers can
embed code objects into the final executable. These code objects are automatically
loaded when the application starts. The ``hipModule`` API can be used to load
additional code objects. When used this way, it extends the capability of the
automatically loaded code objects. HIP-Clang enables both of these capabilities to
be used together. Of course, it is possible to create a program with no kernels and
no automatic loading.

For ``hipModule`` API reference content, see :ref:`module_management_reference`.

Using hipModuleLaunchKernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both CUDA driver and runtime APIs define a function for launching kernels,
called ``cuLaunchKernel`` or ``cudaLaunchKernel``. The equivalent API in HIP is
:cpp:func:`hipModuleLaunchKernel`. The kernel arguments and the execution
configuration (grid dimensions, group dimensions, dynamic shared memory, and
stream) are passed as arguments to the launch function.

The HIP runtime API additionally supports the triple chevron (``<<< >>>``) syntax for launching
kernels, which resembles a special function call and is easier to use than the
explicit launch API, especially when handling kernel arguments. 

.. _context_driver_api:

cuCtx and hipCtx
----------------

The CUDA driver API defines "Context" and "Devices" as separate entities.
Contexts contain a single device, and a device can theoretically have multiple contexts.
Each context contains a set of streams and events specific to the context.
The ``cuCtx`` API also provide a mechanism to switch between devices, which enables a
single CPU thread to send commands to different GPUs. HIP and recent versions of the
CUDA Runtime provide other mechanisms to accomplish this, such as using streams or ``cudaSetDevice``.

On the other hand, the CUDA runtime API unifies the Context API with the Device API. This simplifies the
APIs and has little loss of functionality because each context can contain a
single device, and the benefits of multiple contexts have been replaced with other interfaces.

HIP provides a Context API as a thin layer over the existing device functions to facilitate
easy porting from existing driver API code. The ``hipCtx`` functions largely provide an
alternate syntax for changing the active device. The ``hipCtx`` API can be used to set the
current context or to query properties of the device associated with the context. The current
context is implicitly used by other APIs, such as ``hipStreamCreate``.

.. note:: 
  The ``hipCtx`` API is **deprecated** and its use is discouraged. Most new applications use
  ``hipSetDevice`` or the ``hipStream`` APIs. For more details on deprecated APIs, see :doc:`../reference/deprecated_api_list`.

.. _compilation_platform:

Compilation and platforms
=========================

HIP projects can target either AMD or NVIDIA platforms. The platform affects
which backend-headers are included and which libraries are used for linking. The
created binaries are not portable between AMD and NVIDIA platforms, and instead
must be separately compiled.

``hipcc`` is a portable compiler driver that calls ``amdclang++`` (on AMD systems)
or ``nvcc`` (on NVIDIA systems), passing the necessary options to the target
compiler. Tools that call ``hipcc`` must ensure the compiler options are appropriate
for the target compiler.

``hipconfig`` is a helpful tool for identifying the current system's platform,
compiler and runtime. It can also help set options appropriately. As an example,
``hipconfig`` can provide a path to HIP, in Makefiles:

.. code-block:: shell

  HIP_PATH ?= $(shell hipconfig --path)

.. note::
  You can use ``amdclang++`` to target NVIDIA systems, but you must manually specify
  the required compiler options. 

HIP Headers
-----------

The ``hip_runtime.h`` headers define all the necessary types, functions, macros,
etc., needed to compile a HIP program, this includes host as well as device
code. ``hip_runtime_api.h`` is a subset of ``hip_runtime.h``.

CUDA has slightly different contents for these two files. In some cases you might
need to convert hipified code to include the richer ``hip_runtime.h`` instead of
``hip_runtime_api.h``.

Using a Standard C++ Compiler
-----------------------------

A source file that is only calling HIP APIs but neither defines nor launches
any kernels can be compiled with a standard C or C++ compiler (GCC or MSVC for example )
even when ``hip_runtime_api.h`` or ``hip_runtime.h`` are included. The HIP include
paths and platform macros (``__HIP_PLATFORM_AMD__`` or ``__HIP_PLATFORM_NVIDIA__``)
must be passed to the compiler.

``hipconfig`` can help define the necessary options, for example on an AMD
platform:

.. code-block:: bash

  hipconfig --cpp_config
   -D__HIP_PLATFORM_AMD__= -I/opt/rocm/include

``nvcc`` includes some headers by default. ``hipcc`` does not include
default headers, and instead you must explicitly include all required files.

.. note::
  The ``hipify`` tool automatically converts ``cuda_runtime.h`` to ``hip_runtime.h``,
  and it converts ``cuda_runtime_api.h`` to ``hip_runtime_api.h``, but it may
  miss nested headers or macros.

Compiler defines for HIP and CUDA
---------------------------------

C++-macros can be used to write code that is specific to a platform. This
section lists macros defined by compilers and the HIP/CUDA APIs,
and the compiler/platform combinations that define them.

The following table lists the macros that can be used when compiling HIP. Most
of these macros are not directly defined by the compilers, but in
``hip_common.h``, which is included by ``hip_runtime.h``.

.. list-table:: HIP-related defines
  :header-rows: 1

  *
   - Macro
   - ``amdclang++``
   - ``nvcc`` when used as backend for ``hipcc``
   - Other (GCC, MSVC, Clang, etc.)
  *
   - ``__HIP_PLATFORM_AMD__``
   - Defined (see :ref:`identifying_compiler_target`)
   - Undefined
   - Undefined, needs to be set explicitly
  *
   - ``__HIP_PLATFORM_NVIDIA__``
   - Undefined
   - Defined (see :ref:`identifying_compiler_target`)
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
   - Other (GCC, MSVC, Clang, etc.)
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

.. _identifying_compiler_target:

Identifying the compilation target platform
-------------------------------------------

With HIP's portability, you might need to provide platform-specific code, or enable
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

Identifying host or device compilation pass
-------------------------------------------

``amdclang++`` makes multiple passes over the code: one pass for the host code, and
for the device code one pass for each GPU architecture to be compiled for.
``nvcc`` only makes two passes over the code: one for the host code and one for the
device code. 

The ``__HIP_DEVICE_COMPILE__`` macro is defined when the compiler is compiling
for the device. This macro is a portable check that can replace the
``__CUDA_ARCH__`` macro.

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

HIP-Clang implementation notes
==============================

HIP-Clang links device code from different translation units together. For each
device target, it generates a code object. ``clang-offload-bundler`` bundles
code objects for different device targets into one fat binary, which is embedded
as the global symbol ``__hip_fatbin`` in the ``.hip_fatbin`` section of the ELF
file of the executable or shared object.

Initialization and termination functions
----------------------------------------

HIP-Clang generates initialization and termination functions for each
translation unit for host code compilation. The initialization functions call
``__hipRegisterFatBinary`` to register the fat binary embedded in the ELF file.
They also call ``__hipRegisterFunction`` and ``__hipRegisterVar`` to register
kernel functions and device-side global variables. The termination functions
call ``__hipUnregisterFatBinary``.

HIP-Clang emits a global variable ``__hip_gpubin_handle`` of type ``void**``
with ``linkonce`` linkage and an initial value of 0 for each host translation
unit. Each initialization function checks ``__hip_gpubin_handle`` and registers
the fat binary only if ``__hip_gpubin_handle`` is 0. It saves the return value
of ``__hip_gpubin_handle`` to ``__hip_gpubin_handle``. This ensures that the fat
binary is registered once. A similar check is performed in the termination
functions.

Kernel launching
----------------

HIP-Clang supports kernel launching using either the triple chevron (``<<<>>>``) syntax,
:cpp:func:`hipLaunchKernel`, or :cpp:func:`hipLaunchKernelGGL`. The last option is a macro that
expands to the ``<<<>>>`` syntax by default. It can also be turned into a template by
defining ``HIP_TEMPLATE_KERNEL_LAUNCH``.

When the executable or shared library is loaded by the dynamic linker, the
initialization functions are called. In the initialization functions, the code
objects containing all kernels are loaded when ``__hipRegisterFatBinary`` is
called. When ``__hipRegisterFunction`` is called, the stub functions are
associated with the corresponding kernels in the code objects.

HIP-Clang implements two sets of APIs for launching kernels.
By default, when HIP-Clang encounters the ``<<<>>>`` statement in the host code,
it first calls :cpp:func:`hipConfigureCall` to set up the threads and grids. It then
calls the stub function with the given arguments. The stub function calls
:cpp:func:`hipSetupArgument` for each kernel argument, then calls :cpp:func:`hipLaunchByPtr`
with a function pointer to the stub function. In ``hipLaunchByPtr``, the actual
kernel associated with the stub function is launched.

NVCC implementation notes
=========================

CUDA applications can mix CUDA code with HIP code (see the
example below). The table shows the equivalent CUDA and HIP types
required to implement this interaction.

.. list-table:: Equivalence table between HIP and CUDA types
   :header-rows: 1

   * - HIP type
     - CU Driver type
     - CUDA Runtime type
   * - :cpp:type:`hipModule_t`
     - ``CUmodule``
     -
   * - :cpp:type:`hipFunction_t`
     - ``CUfunction``
     -
   * - :cpp:type:`hipCtx_t`
     - ``CUcontext``
     -
   * - :cpp:type:`hipDevice_t`
     - ``CUdevice``
     -
   * - :cpp:type:`hipStream_t`
     - ``CUstream``
     - ``cudaStream_t``
   * - :cpp:type:`hipEvent_t`
     - ``CUevent``
     - ``cudaEvent_t``
   * - :cpp:type:`hipArray_t`
     - ``CUarray``
     - ``cudaArray``

Compilation options
-------------------

The :cpp:type:`hipModule_t` interface does not support the ``cuModuleLoadDataEx`` function,
which is used to control PTX compilation options. HIP-Clang does not use PTX, so
it does not support these compilation options. In fact, HIP-Clang code objects contain
fully compiled code for a device-specific instruction set and don't require additional
compilation as a part of the load step. The corresponding HIP function :cpp:func:`hipModuleLoadDataEx`
behaves like :cpp:func:`hipModuleLoadData` on the HIP-Clang path (where compilation options
are not used) and like ``cuModuleLoadDataEx`` on the NVCC path.

For example:

.. tab-set::

    .. tab-item:: HIP

        .. code-block:: cpp

            hipModule_t module;
            void *imagePtr = ...; // Somehow populate data pointer with code object

            const int numOptions = 1;
            hipJitOption options[numOptions];
            void *optionValues[numOptions];

            options[0] = hipJitOptionMaxRegisters;
            unsigned maxRegs = 15;
            optionValues[0] = (void *)(&maxRegs);

            // hipModuleLoadData(module, imagePtr) will be called on HIP-Clang path, JIT
            // options will not be used, and cuModuleLoadDataEx(module, imagePtr,
            // numOptions, options, optionValues) will be called on NVCC path
            hipModuleLoadDataEx(module, imagePtr, numOptions, options, optionValues);

            hipFunction_t k;
            hipModuleGetFunction(&k, module, "myKernel");

    .. tab-item:: CUDA

        .. code-block:: cpp

            CUmodule module;
            void *imagePtr = ...; // Somehow populate data pointer with code object

            const int numOptions = 1;
            CUJit_option options[numOptions];
            void *optionValues[numOptions];

            options[0] = CU_JIT_MAX_REGISTERS;
            unsigned maxRegs = 15;
            optionValues[0] = (void *)(&maxRegs);

            cuModuleLoadDataEx(module, imagePtr, numOptions, options, optionValues);

            CUfunction k;
            cuModuleGetFunction(&k, module, "myKernel");

The sample below shows how to use :cpp:func:``hipModuleGetFunction``.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <hip/hip_runtime_api.h>

    #include <vector>

    int main() {

        size_t elements = 64*1024;
        size_t size_bytes = elements * sizeof(float);

        std::vector<float> A(elements), B(elements);

        // On NVIDIA platforms the driver runtime needs to be initiated
        #ifdef __HIP_PLATFORM_NVIDIA__
        hipInit(0);
        hipDevice_t device;
        hipCtx_t context;
        HIPCHECK(hipDeviceGet(&device, 0));
        HIPCHECK(hipCtxCreate(&context, 0, device));
        #endif

        // Allocate device memory
        hipDeviceptr_t d_A, d_B;
        HIPCHECK(hipMalloc(&d_A, size_bytes));
        HIPCHECK(hipMalloc(&d_B, size_bytes));

        // Copy data to device
        HIPCHECK(hipMemcpyHtoD(d_A, A.data(), size_bytes));
        HIPCHECK(hipMemcpyHtoD(d_B, B.data(), size_bytes));

        // Load module
        hipModule_t Module;
        // For AMD the module file has to contain architecture specific object codee
        // For NVIDIA the module file has to contain PTX, found in e.g. "vcpy_isa.ptx"
        HIPCHECK(hipModuleLoad(&Module, "vcpy_isa.co"));
        // Get kernel function from the module via its name
        hipFunction_t Function;
        HIPCHECK(hipModuleGetFunction(&Function, Module, "hello_world"));

        // Create buffer for kernel arguments
        std::vector<void*> argBuffer{&d_A, &d_B};
        size_t arg_size_bytes = argBuffer.size() * sizeof(void*);

        // Create configuration passed to the kernel as arguments
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argBuffer.data(),
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size_bytes, HIP_LAUNCH_PARAM_END};

        int threads_per_block = 128;
        int blocks = (elements + threads_per_block - 1) / threads_per_block;

        // Actually launch kernel
        HIPCHECK(hipModuleLaunchKernel(Function, blocks, 1, 1, threads_per_block, 1, 1, 0, 0, NULL, config));

        HIPCHECK(hipMemcpyDtoH(A.data(), d_A, elements));
        HIPCHECK(hipMemcpyDtoH(B.data(), d_B, elements));

        #ifdef __HIP_PLATFORM_NVIDIA__
        HIPCHECK(hipCtxDetach(context));
        #endif

        HIPCHECK(hipFree(d_A));
        HIPCHECK(hipFree(d_B));

        return 0;
    }

.. _identifying_device_architecture_features:

Identifying device architecture and features
============================================

GPUs of different generations and architectures do not provide the same
level of :doc:`hardware feature support <../reference/hardware_features>`. To
guard device code that uses architecture-dependent features, the
``__HIP_ARCH_<FEATURE>__`` C++-macros can be used, as described below. 

Device code feature identification
----------------------------------

Some CUDA code tests ``__CUDA_ARCH__`` for a specific value to determine whether
the GPU supports a certain architectural feature, depending on its compute
capability. This requires knowledge about what ``__CUDA_ARCH__`` supports what
feature set.

HIP simplifies this, by replacing these macros with feature-specific macros, not
architecture specific.

For instance,

.. code-block:: cpp

  //#if __CUDA_ARCH__ >= 130 // does not properly specify what feature is required, not portable
  #if __HIP_ARCH_HAS_DOUBLES__ == 1 // explicitly specifies what feature is required, portable between AMD and NVIDIA GPUs
    // device code
  #endif

For host code, the ``__HIP_ARCH_<FEATURE>__`` defines are set to 0, if
``hip_runtime.h`` is included, and undefined otherwise. It should not be relied
upon in host code.

Host code feature identification
--------------------------------

The host code must not rely on the ``__HIP_ARCH_<FEATURE>__`` macros, because the
GPUs available to a system are not known during compile time, and their
architectural features differ. Alternatively, the host code can query architecture
feature flags during runtime by using :cpp:func:`hipGetDeviceProperties`
or :cpp:func:`hipDeviceGetAttribute`.

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

Feature macros and properties
-----------------------------

The following table lists the feature macros that HIP supports,
alongside corresponding device properties that can be queried from the host code.

.. list-table::
  :header-rows: 1

  *
   - Macro (for device code)
   - Device property (for host runtime query)
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

warpSize
========

Code should not assume a warp size of 32 or 64, as that is not portable between
platforms and architectures. The ``warpSize`` built-in should be used in device
code, while the host can query it during runtime via the device properties. See
the :ref:`HIP language extension for warpSize <warp_size>` for information on
how to write portable warpSize-aware code.

Porting from CUDA __launch_bounds__
===================================

CUDA defines a ``__launch_bounds__`` qualifier which works similarly to the HIP
implementation, however, it uses different parameters:

.. code-block:: cpp

  __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)

``MAX_THREADS_PER_BLOCK`` is the same in CUDA and in HIP. However, ``MIN_BLOCKS_PER_MULTIPROCESSOR`` in CUDA
must  be converted to ``MIN_WARPS_PER_EXECUTION_UNIT`` in HIP, which uses warps and execution units
rather than blocks and multiprocessors. This conversion can be done manually with the equation
considering the GPU's configuration mode.

* In Compute Unit (CU) mode, typical of CDNA:

.. code-block:: cpp

  MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK) / (warpSize * 2)

* In Workgroup Processor (WGP) mode, a feature of RDNA:

.. code-block:: cpp

  MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK) / (warpSize * 4)

Directly controlling the warps per execution unit makes it easier to reason about the occupancy,
unlike with blocks, where the occupancy depends on the block size.

The use of execution units rather than multiprocessors also provides support for
architectures with multiple execution units per multiprocessor. For example, the
AMD GCN architecture has 4 execution units per multiprocessor.

maxregcount
-----------

The ``nvcc`` compiler will predict the number of registers per thread based on the launch bounds calculation.
``--maxregcount X`` can be used to override the compiler's decision by enforcing a hard number of registers
(``X``) that the compiler must not exceed. If the compiler is unable to meet this requirement, it will place
additional "registers" into memory instead of using hardware registers. 

Unlike ``nvcc``, ``amdclang++`` does not support the ``--maxregcount`` option. You are encouraged to use
the ``__launch_bounds__`` directive since the parameters are more intuitive and portable than micro-architecture
details like registers. The directive allows per-kernel control.

Driver entry point access
=========================

The HIP runtime provides support for CUDA driver entry point access when using
CUDA 12.0 or later. This feature lets developers interact directly with the
CUDA driver API, providing more control over GPU operations.

Driver entry point access provides several features:

* Retrieving the address of a runtime function
* Requesting the default stream version on a per-thread basis
* Accessing HIP features on older toolkits with a newer driver

For more information on driver entry point access, see :cpp:func:`hipGetProcAddress`.

Address retrieval
-----------------

The :cpp:func:`hipGetProcAddress` function can be used to obtain the address of
a runtime function. This is demonstrated in the following example:

.. code-block:: cpp

  #include <hip/hip_runtime.h>
  #include <hip/hip_runtime_api.h>

  #include <iostream>

  typedef hipError_t (*hipInit_t)(unsigned int);

  int main() {
      // Initialize the HIP runtime
      hipError_t res = hipInit(0);
      if (res != hipSuccess) {
          std::cerr << "Failed to initialize HIP runtime." << std::endl;
          return 1;
      }

      // Get the address of the hipInit function
      hipInit_t hipInitFunc;
      int hipVersion = HIP_VERSION; // Use the HIP version defined in hip_runtime_api.h
      uint64_t flags = 0; // No special flags
      hipDriverProcAddressQueryResult symbolStatus;

      res = hipGetProcAddress("hipInit", (void**)&hipInitFunc, hipVersion, flags, &symbolStatus);
      if (res != hipSuccess) {
          std::cerr << "Failed to get address of hipInit()." << std::endl;
          return 1;
      }

      // Call the hipInit function using the obtained address
      res = hipInitFunc(0);
      if (res == hipSuccess) {
          std::cout << "HIP runtime initialized successfully using hipGetProcAddress()." << std::endl;
      } else {
          std::cerr << "Failed to initialize HIP runtime using hipGetProcAddress()." << std::endl;
      }

      return 0;
  }

Per-thread default stream version request
-----------------------------------------

HIP offers functionality similar to CUDA for managing streams on a per-thread
basis. By using ``hipStreamPerThread``, each thread can independently manage its
default stream, simplifying operations. The following example demonstrates how
this feature enhances performance by reducing contention and improving
efficiency.

.. code-block:: cpp

  #include <hip/hip_runtime.h>

  #include <iostream>

  int main() {
      // Initialize the HIP runtime
      hipError_t res = hipInit(0);
      if (res != hipSuccess) {
          std::cerr << "Failed to initialize HIP runtime." << std::endl;
          return 1;
      }

      // Get the per-thread default stream
      hipStream_t stream = hipStreamPerThread;

      // Use the stream for some operation
      // For example, allocate memory on the device
      void* d_ptr;
      size_t size = 1024;
      res = hipMalloc(&d_ptr, size);
      if (res != hipSuccess) {
          std::cerr << "Failed to allocate memory." << std::endl;
          return 1;
      }

      // Perform some operation using the stream
      // For example, set memory on the device
      res = hipMemsetAsync(d_ptr, 0, size, stream);
      if (res != hipSuccess) {
          std::cerr << "Failed to set memory." << std::endl;
          return 1;
      }

      // Synchronize the stream
      res = hipStreamSynchronize(stream);
      if (res != hipSuccess) {
          std::cerr << "Failed to synchronize stream." << std::endl;
          return 1;
      }

      std::cout << "Operation completed successfully using per-thread default stream." << std::endl;

      // Free the allocated memory
      hipFree(d_ptr);

      return 0;
  }

Accessing HIP features with a newer driver
------------------------------------------

HIP is forward compatible, allowing newer features to be utilized
with older toolkits, provided a compatible driver is present. Feature support
can be verified through runtime API functions and version checks. This approach
ensures that applications can benefit from new features and improvements in the
HIP runtime without requiring recompilation with a newer toolkit. The function
:cpp:func:`hipGetProcAddress` enables dynamic querying and the use of newer
functions offered by the HIP runtime, even if the application was built with an
older toolkit.

.. note::
  :cpp:func:``hipGetProcAddress`` and its CUDA counterpart ``cuGetProcAddress`` are limited
  to HIP/CUDA driver API function calls. For HIP/CUDA runtime API calls,the corresponding
  function is :cpp:func:``hipGetDriverEntryPoint`` / ``cudaGetDriverEntryPoint``. 

An example is provided for a hypothetical ``foo()`` function.

.. code-block:: cpp

  // Get the address of the foo function
  foo_t fooFunc;
  int hipVersion = 60300000; // HIP version number (e.g. 6.3.0)
  uint64_t flags = 0; // No special flags
  hipDriverProcAddressQueryResult symbolStatus;

  res = hipGetProcAddress("foo", (void**)&fooFunc, hipVersion, flags, &symbolStatus);

The HIP version number is defined as an integer:

.. code-block:: cpp

  HIP_VERSION=HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH

CU_POINTER_ATTRIBUTE_MEMORY_TYPE
================================

To return the pointer's memory type in HIP, developers should use :cpp:func:`hipPointerGetAttributes`.
The first parameter of the function is `hipPointerAttribute_t`. Its ``type`` member variable indicates
whether the memory pointed to is allocated on the device or the host. For example:

.. code-block:: cpp

  double * ptr;
  hipMalloc(&ptr, sizeof(double));
  hipPointerAttribute_t attr;
  hipPointerGetAttributes(&attr, ptr); /*attr.type is hipMemoryTypeDevice*/
  if(attr.type == hipMemoryTypeDevice)
    std::cout << "ptr is of type hipMemoryTypeDevice" << std::endl;

  double* ptrHost;
  hipHostMalloc(&ptrHost, sizeof(double));
  hipPointerAttribute_t attr;
  hipPointerGetAttributes(&attr, ptrHost); /*attr.type is hipMemoryTypeHost*/
  if(attr.type == hipMemorTypeHost)
    std::cout << "ptrHost is of type hipMemoryTypeHost" << std::endl;

Note that ``hipMemoryType`` enum values are different from the
``cudaMemoryType`` enum values.

For example, on AMD platform, ``hipMemoryType`` is defined in ``hip_runtime_api.h``:

.. code-block:: cpp

  typedef enum hipMemoryType {
      hipMemoryTypeHost = 0,    ///< Memory is physically located on host
      hipMemoryTypeDevice = 1,  ///< Memory is physically located on device. (see deviceId for specific device)
      hipMemoryTypeArray = 2,   ///< Array memory, physically located on device. (see deviceId for specific device)
      hipMemoryTypeUnified = 3, ///< Not used currently
      hipMemoryTypeManaged = 4  ///< Managed memory, automaticallly managed by the unified memory system
  } hipMemoryType;

In the CUDA toolkit, the ``cudaMemoryType`` is defined as following:

.. code-block:: cpp

  enum cudaMemoryType
  {
    cudaMemoryTypeUnregistered = 0, // Unregistered memory.
    cudaMemoryTypeHost = 1, // Host memory.
    cudaMemoryTypeDevice = 2, // Device memory.
    cudaMemoryTypeManaged = 3, // Managed memory
  }

.. note::
  ``cudaMemoryTypeUnregistered`` is currently not supported as ``hipMemoryType`` enum,
  due to HIP functionality backward compatibility.

The memory type translation for ``hipPointerGetAttributes`` needs to
be handled properly on NVIDIA platform to return the correct memory type in CUDA,
which is done in the file ``nvidia_hip_runtime_api.h``.

In applications that use HIP memory type APIs, you should use ``#ifdef``
to assign the correct enum values depending on NVIDIA or AMD platform.
