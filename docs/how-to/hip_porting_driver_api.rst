.. meta::
  :description: This chapter presents how to port the CUDA driver API and showcases equivalent operations in HIP.
  :keywords: AMD, ROCm, HIP, CUDA, driver API, porting, port

.. _porting_driver_api:

*******************************************************************************
Porting CUDA driver API
*******************************************************************************

CUDA provides separate driver and runtime APIs. The two APIs generally provide
the similar functionality and mostly can be used interchangeably, however the
driver API allows for more fine-grained control over the kernel level
initialization, contexts and module management. This is all taken care of
implicitly by the runtime API.

* Driver API calls begin with the prefix ``cu``, while runtime API calls begin
  with the prefix ``cuda``. For example, the driver API contains
  ``cuEventCreate``, while the runtime API contains ``cudaEventCreate``, which
  has similar functionality.

* The driver API offers two additional low-level functionalities not exposed by
  the runtime API: module management ``cuModule*`` and context management
  ``cuCtx*`` APIs.

HIP does not explicitly provide two different APIs, the corresponding functions
for the CUDA driver API are available in the HIP runtime API, and are usually
prefixed with ``hipDrv``. The module and context functionality is available with
the ``hipModule`` and ``hipCtx`` prefix.

cuModule API
================================================================================

The Module section of the driver API provides additional control over how and
when accelerator code objects are loaded. For example, the driver API enables
code objects to load from files or memory pointers. Symbols for kernels or
global data are extracted from the loaded code objects. In contrast, the runtime
API loads automatically and, if necessary, compiles all the kernels from an
executable binary when it runs. In this mode, kernel code must be compiled using
NVCC so that automatic loading can function correctly.

The Module features are useful in an environment that generates the code objects
directly, such as a new accelerator language front end. NVCC is not used here.
Instead, the environment might have a different kernel language or compilation
flow. Other environments have many kernels and don't want all of them to be
loaded automatically. The Module functions load the generated code objects and
launch kernels. Similar to the cuModule API, HIP defines a hipModule API that
provides similar explicit control over code object management.

.. _context_driver_api:

cuCtx API
================================================================================

The driver API defines "Context" and "Devices" as separate entities.
Contexts contain a single device, and a device can theoretically have multiple contexts.
Each context contains a set of streams and events specific to the context.
Historically, contexts also defined a unique address space for the GPU. This might no longer be the case in unified memory platforms, because the CPU and all the devices in the same process share a single unified address space.
The Context APIs also provide a mechanism to switch between devices, which enables a single CPU thread to send commands to different GPUs.
HIP and recent versions of the CUDA Runtime provide other mechanisms to accomplish this feat, for example, using streams or ``cudaSetDevice``.

The CUDA runtime API unifies the Context API with the Device API. This simplifies the APIs and has little loss of functionality. This is because each context can contain a single device, and the benefits of multiple contexts have been replaced with other interfaces.
HIP provides a Context API to facilitate easy porting from existing Driver code.
In HIP, the ``Ctx`` functions largely provide an alternate syntax for changing the active device.

Most new applications preferentially use ``hipSetDevice`` or the stream APIs. Therefore, HIP has marked the ``hipCtx`` APIs as **deprecated**. Support for these APIs might not be available in future releases. For more details on deprecated APIs, see :doc:`../reference/deprecated_api_list`.

HIP module and Ctx APIs
================================================================================

Rather than present two separate APIs, HIP extends the HIP API with new APIs for
modules and ``Ctx`` control.

hipModule API
--------------------------------------------------------------------------------

Like the CUDA driver API, the Module API provides additional control over how
code is loaded, including options to load code from files or from in-memory
pointers.
NVCC and HIP-Clang target different architectures and use different code object
formats. NVCC supports ``cubin`` or ``ptx`` files, while the HIP-Clang path uses
the ``hsaco`` format.
The external compilers which generate these code objects are responsible for
generating and loading the correct code object for each platform.
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

``hipcc`` uses HIP-Clang or NVCC to compile host code. Both of these compilers can embed code objects into the final executable. These code objects are automatically loaded when the application starts.
The ``hipModule`` API can be used to load additional code objects. When used this way, it extends the capability of the automatically loaded code objects.
HIP-Clang enables both of these capabilities to be used together. Of course, it is possible to create a program with no kernels and no automatic loading.

For module API reference, visit :ref:`module_management_reference`.

hipCtx API
--------------------------------------------------------------------------------

HIP provides a ``Ctx`` API as a thin layer over the existing device functions. The ``Ctx`` API can be used to set the current context or to query properties of the device associated with the context.
The current context is implicitly used by other APIs, such as ``hipStreamCreate``.

For context reference, visit :ref:`context_management_reference`.

HIPIFY translation of CUDA driver API
================================================================================

The HIPIFY tools convert CUDA driver APIs such as streams, events, modules,
devices, memory management, context, and the profiler to the equivalent HIP
calls. For example, ``cuEventCreate`` is translated to :cpp:func:`hipEventCreate`.
HIPIFY tools also convert error codes from the driver namespace and coding
conventions to the equivalent HIP error code. HIP unifies the APIs for these
common functions.

The memory copy API requires additional explanation. The CUDA driver includes
the memory direction in the name of the API (``cuMemcpyHtoD``), while the CUDA
runtime API provides a single memory copy API with a parameter that specifies
the direction. It also supports a "default" direction where the runtime
determines the direction automatically.
HIP provides both versions, for example, :cpp:func:`hipMemcpyHtoD` as well as
:cpp:func:`hipMemcpy`. The first version might be faster in some cases because
it avoids any host overhead to detect the different memory directions.

HIP defines a single error space and uses camel case for all errors (i.e. ``hipErrorInvalidValue``).

For further information, visit the :doc:`hipify:index`.

Address spaces
--------------------------------------------------------------------------------

HIP-Clang defines a process-wide address space where the CPU and all devices
allocate addresses from a single unified pool.
This means addresses can be shared between contexts. Unlike the original CUDA
implementation, a new context does not create a new address space for the device.

Using hipModuleLaunchKernel
--------------------------------------------------------------------------------

Both CUDA driver and runtime APIs define a function for launching kernels,
called ``cuLaunchKernel`` or ``cudaLaunchKernel``. The equivalent API in HIP is
``hipModuleLaunchKernel``.
The kernel arguments and the execution configuration (grid dimensions, group
dimensions, dynamic shared memory, and stream) are passed as arguments to the
launch function.
The runtime API additionally provides the ``<<< >>>`` syntax for launching
kernels, which resembles a special function call and is easier to use than the
explicit launch API, especially when handling kernel arguments.
However, this syntax is not standard C++ and is available only when NVCC is used
to compile the host code.

Additional information
--------------------------------------------------------------------------------

HIP-Clang creates a primary context when the HIP API is called. So, in pure
driver API code, HIP-Clang creates a primary context while HIP/NVCC has an empty
context stack. HIP-Clang pushes the primary context to the context stack when it
is empty. This can lead to subtle differences in applications which mix the
runtime and driver APIs.

HIP-Clang implementation notes
================================================================================

.hip_fatbin
--------------------------------------------------------------------------------

HIP-Clang links device code from different translation units together. For each
device target, it generates a code object. ``clang-offload-bundler`` bundles
code objects for different device targets into one fat binary, which is embedded
as the global symbol ``__hip_fatbin`` in the ``.hip_fatbin`` section of the ELF
file of the executable or shared object.

Initialization and termination functions
--------------------------------------------------------------------------------

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
--------------------------------------------------------------------------------

HIP-Clang supports kernel launching using either the CUDA ``<<<>>>`` syntax,
``hipLaunchKernel``, or ``hipLaunchKernelGGL``. The last option is a macro which
expands to the CUDA ``<<<>>>`` syntax by default. It can also be turned into a
template by defining ``HIP_TEMPLATE_KERNEL_LAUNCH``.

When the executable or shared library is loaded by the dynamic linker, the
initialization functions are called. In the initialization functions, the code
objects containing all kernels are loaded when ``__hipRegisterFatBinary`` is
called. When ``__hipRegisterFunction`` is called, the stub functions are
associated with the corresponding kernels in the code objects.

HIP-Clang implements two sets of APIs for launching kernels.
By default, when HIP-Clang encounters the ``<<<>>>`` statement in the host code,
it first calls ``hipConfigureCall`` to set up the threads and grids. It then
calls the stub function with the given arguments. The stub function calls
``hipSetupArgument`` for each kernel argument, then calls ``hipLaunchByPtr``
with a function pointer to the stub function. In ``hipLaunchByPtr``, the actual
kernel associated with the stub function is launched.

NVCC implementation notes
================================================================================

Interoperation between HIP and CUDA driver
--------------------------------------------------------------------------------

CUDA applications might want to mix CUDA driver code with HIP code (see the
example below). This table shows the equivalence between CUDA and HIP types
required to implement this interaction.

.. list-table:: Equivalence table between HIP and CUDA types
   :header-rows: 1

   * - HIP type
     - CU Driver type
     - CUDA Runtime type
   * - ``hipModule_t``
     - ``CUmodule``
     -
   * - ``hipFunction_t``
     - ``CUfunction``
     -
   * - ``hipCtx_t``
     - ``CUcontext``
     -
   * - ``hipDevice_t``
     - ``CUdevice``
     -
   * - ``hipStream_t``
     - ``CUstream``
     - ``cudaStream_t``
   * - ``hipEvent_t``
     - ``CUevent``
     - ``cudaEvent_t``
   * - ``hipArray``
     - ``CUarray``
     - ``cudaArray``

Compilation options
--------------------------------------------------------------------------------

The ``hipModule_t`` interface does not support the ``cuModuleLoadDataEx`` function, which is used to control PTX compilation options.
HIP-Clang does not use PTX, so it does not support these compilation options.
In fact, HIP-Clang code objects contain fully compiled code for a device-specific instruction set and don't require additional compilation as a part of the load step.
The corresponding HIP function ``hipModuleLoadDataEx`` behaves like ``hipModuleLoadData`` on the HIP-Clang path (where compilation options are not used) and like ``cuModuleLoadDataEx`` on the NVCC path.

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
            // options will not be used, and cupModuleLoadDataEx(module, imagePtr,
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

The sample below shows how to use ``hipModuleGetFunction``.

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

HIP module and texture Driver API
================================================================================

HIP supports texture driver APIs. However, texture references must be declared
within the host scope. The following code demonstrates the use of texture
references for the ``__HIP_PLATFORM_AMD__`` platform.

.. code-block:: cpp

    // Code to generate code object

    #include "hip/hip_runtime.h"
    extern texture<float, 2, hipReadModeElementType> tex;

    __global__ void tex2dKernel(hipLaunchParm lp, float *outputData, int width,
                                int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        outputData[y * width + x] = tex2D(tex, x, y);
    }

.. code-block:: cpp

  // Host code:

  texture<float, 2, hipReadModeElementType> tex;

    void myFunc ()
    {
        // ...

        textureReference* texref;
        hipModuleGetTexRef(&texref, Module1, "tex");
        hipTexRefSetAddressMode(texref, 0, hipAddressModeWrap);
        hipTexRefSetAddressMode(texref, 1, hipAddressModeWrap);
        hipTexRefSetFilterMode(texref, hipFilterModePoint);
        hipTexRefSetFlags(texref, 0);
        hipTexRefSetFormat(texref, HIP_AD_FORMAT_FLOAT, 1);
        hipTexRefSetArray(texref, array, HIP_TRSA_OVERRIDE_FORMAT);

      // ...
    }

Driver entry point access
================================================================================

Starting from HIP version 6.2.0, support for Driver Entry Point Access is
available when using CUDA 12.0 or newer. This feature allows developers to
directly interact with the CUDA driver API, providing more control over GPU
operations.

Driver Entry Point Access provides several features:

* Retrieving the address of a runtime function
* Requesting the default stream version on a per-thread basis
* Accessing new HIP features on older toolkits with a newer driver

For driver entry point access reference, visit :cpp:func:`hipGetProcAddress`.

Address retrieval
--------------------------------------------------------------------------------

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
================================================================================

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

Accessing new HIP features with a newer driver
================================================================================

HIP is designed to be forward compatible, allowing newer features to be utilized
with older toolkits, provided a compatible driver is present. Feature support
can be verified through runtime API functions and version checks. This approach
ensures that applications can benefit from new features and improvements in the
HIP runtime without needing to be recompiled with a newer toolkit. The function
:cpp:func:`hipGetProcAddress` enables dynamic querying and the use of newer
functions offered by the HIP runtime, even if the application was built with an
older toolkit.

An example is provided for a hypothetical ``foo()`` function.

.. code-block:: cpp

  // Get the address of the foo function
  foo_t fooFunc;
  int hipVersion = 60300000; // Use an own HIP version number (e.g. 6.3.0)
  uint64_t flags = 0; // No special flags
  hipDriverProcAddressQueryResult symbolStatus;

  res = hipGetProcAddress("foo", (void**)&fooFunc, hipVersion, flags, &symbolStatus);

The HIP version number is defined as an integer:

.. code-block:: cpp

  HIP_VERSION=HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH

CU_POINTER_ATTRIBUTE_MEMORY_TYPE
================================================================================

To get the pointer's memory type in HIP, developers should use
:cpp:func:`hipPointerGetAttributes`. First parameter of the function is
`hipPointerAttribute_t`. Its ``type`` member variable indicates whether the
memory pointed to is allocated on the device or the host.

For example:

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

For example, on AMD platform, `hipMemoryType` is defined in `hip_runtime_api.h`,

.. code-block:: cpp

  typedef enum hipMemoryType {
      hipMemoryTypeHost = 0,    ///< Memory is physically located on host
      hipMemoryTypeDevice = 1,  ///< Memory is physically located on device. (see deviceId for specific device)
      hipMemoryTypeArray = 2,   ///< Array memory, physically located on device. (see deviceId for specific device)
      hipMemoryTypeUnified = 3, ///< Not used currently
      hipMemoryTypeManaged = 4  ///< Managed memory, automaticallly managed by the unified memory system
  } hipMemoryType;

Looking into CUDA toolkit, it defines `cudaMemoryType` as following,

.. code-block:: cpp

  enum cudaMemoryType
  {
    cudaMemoryTypeUnregistered = 0, // Unregistered memory.
    cudaMemoryTypeHost = 1, // Host memory.
    cudaMemoryTypeDevice = 2, // Device memory.
    cudaMemoryTypeManaged = 3, // Managed memory
  }

In this case, memory type translation for ``hipPointerGetAttributes`` needs to
be handled properly on NVIDIA platform to get the correct memory type in CUDA,
which is done in the file ``nvidia_hip_runtime_api.h``.

So in any HIP applications which use HIP APIs involving memory types, developers
should use ``#ifdef`` in order to assign the correct enum values depending on
NVIDIA or AMD platform.

As an example, please see the code from the `link <https://github.com/ROCm/hip-tests/tree/develop/catch/unit/memory/hipMemcpyParam2D.cc>`_.

With the ``#ifdef`` condition, HIP APIs work as expected on both AMD and NVIDIA
platforms.

Note, ``cudaMemoryTypeUnregistered`` is currently not supported as
``hipMemoryType`` enum, due to HIP functionality backward compatibility.
