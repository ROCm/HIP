# Porting CUDA Driver API

## Introduction to the CUDA Driver and Runtime APIs
CUDA provides a separate CUDA Driver and Runtime APIs. The two APIs have significant overlap in functionality:
- Both APIs support events, streams, memory management, memory copy, and error handling.
- Both APIs deliver similar performance.
- Driver APIs calls begin with the prefix `cu` while Runtime APIs begin with the prefix `cuda`. For example, the Driver API API contains `cuEventCreate` while the Runtime API contains `cudaEventCreate`, with similar functionality.
- The Driver API defines a different but largely overlapping error code space than the Runtime API, and uses a different coding convention. For example, Driver API defines `CUDA_ERROR_INVALID_VALUE` while the Runtime API defines `cudaErrorInvalidValue`


The Driver API offers two additional pieces of functionality not provided by the Runtime API: cuModule and cuCtx APIs.

### cuModule API
The Module section of the Driver API provides additional control over how and when accelerator code objects are loaded.
For example, the driver API allows code objects to be loaded from files or memory pointers.
Symbols for kernels or global data can be extracted from the loaded code objects.
In contrast, the Runtime API automatically loads and (if necessary) compiles all of the kernels from an executable binary when run.
In this mode, NVCC must be used to compile kernel code so the automatic loading can function correctly.

Both Driver and Runtime APIs define a function for launching kernels (called `cuLaunchKernel` or `cudaLaunchKernel`.
The kernel arguments and the execution configuration (grid dimensions, group dimensions, dynamic shared memory, and stream) are passed as arguments to the launch function.
The Runtime additionally provides the `<<< >>>` syntax for launching kernels, which resembles a special function call and is easier to use than explicit launch API (in particular with respect to handling of kernel arguments).
However, this syntax is not standard C++ and is available only when NVCC is used to compile the host code.

The Module features are useful in an environment which generates the code objects directly, such as a new accelerator language front-end.
Here, NVCC is not used. Instead, the environment may have a different kernel language or different compilation flow.
Other environments have many kernels and do not want them to be all loaded automatically.
The Module functions can be used to load the generated code objects and launch kernels.
As we will see below, HIP defines a Module API which provides similar explicit control over code object management.

### cuCtx API
The Driver API defines "Context" and "Devices" as separate entities.
Contexts contain a single device, and a device can theoretically have multiple contexts.
Each context contains a set of streams and events specific to the context.
Historically contexts also defined a unique address space for the GPU, though this may no longer be the case in Unified Memory platforms (since the CPU and all the devices in the same process share a single unified address space).
The Context APIs also provide a mechanism to switch between devices, which allowed a single CPU thread to send commands to different GPUs.
HIP as well as a recent versions of CUDA Runtime provide other mechanisms to accomplish this feat - for example using streams or `cudaSetDevice`.

The CUDA Runtime API unifies the Context API with the Device API. This simplifies the APIs and has little loss of functionality since each Context can contain a single device, and the benefits of multiple contexts has been replaced with other interfaces.
HIP provides a context API to facilitate easy porting from existing Driver codes.
In HIP, the Ctx functions largely provide an alternate syntax for changing the active device.

Most new applications will prefer to use `hipSetDevice` or the stream APIs , therefore HIP has marked hipCtx APIs as **deprecated**. Support for these APIs may not be available in future releases. For more details on deprecated APIs please refer [HIP deprecated APIs](https://github.com/ROCm-Developer-Tools/HIP/tree/master/docs/markdown/hip_deprecated_api_list.md).

## HIP Module and Ctx APIs

Rather than present two separate APIs, HIP extends the HIP API with new APIs for Modules and Ctx control.

### hipModule API

Like the CUDA Driver API, the Module API provides additional control over how code is loaded, including options to load code from files or from in-memory pointers.
NVCC and HCC target different architectures and use different code object formats: NVCC is `cubin` or `ptx` files, while the HCC path is the `hsaco` format.
The external compilers which generate these code objects are responsible for generating and loading the correct code object for each platform.
Notably, there is not a fat binary format that can contain code for both NVCC and HCC platforms. The following table summarizes the formats used on each platform:

| Format        | APIs                             | NVCC               | HCC               | HIP-CLANG    |
| ---           | ---                              | ---                | ---               | ---
| Code Object   | hipModuleLoad, hipModuleLoadData | .cubin or PTX text | .hsaco            | .hsaco       |
| Fat Binary    | hipModuleLoadFatBin              | .fatbin            | Under Development | .hip_fatbin  |

`hipcc` uses NVCC and HCC to compile host codes. Both of these may embed code objects into the final executable, and these code objects will be automatically loaded when the application starts.
The hipModule API can be used to load additional code objects, and in this way provides an extended capability to the automatically loaded code objects.
HCC allows both of these capabilities to be used together, if desired. Of course it is possible to create a program with no kernels and thus no automatic loading.


### hipCtx API
HIP provides a `Ctx` API as a thin layer over the existing Device functions. This Ctx API can be used to set the current context, or to query properties of the device associated with the context.
The current context is implicitly used by other APIs such as `hipStreamCreate`.

### hipify translation of CUDA Driver API
The hipify tool converts CUDA Driver APIs for streams, events, modules, devices, memory management, context, profiler to the equivalent HIP driver calls. For example, `cuEventCreate` will be translated to `hipEventCreate`.
Hipify also converts error code from the Driver namespace and coding convention to the equivalent HIP error code. Thus, HIP unifies the APIs for these common functions.

The memory copy API requires additional explanation. The CUDA driver includes the memory direction in the name of the API (ie `cuMemcpyH2D`) while the CUDA driver API provides a single memory copy API with a parameter that specifies the direction and additionally supports a "default" direction where the runtime determines the direction automatically.
HIP provides APIs with both styles: for example, `hipMemcpyH2D` as well as `hipMemcpy`.
The first flavor may be faster in some cases since they avoid host overhead to detect the different memory directions.

HIP defines a single error space, and uses camel-case for all errors (i.e. `hipErrorInvalidValue`).

### HCC Implementation Notes
#### .hsaco
The .hsaco format used by HCC is described in more detail [here](https://github.com/RadeonOpenCompute/ROCm-ComputeABI-Doc).
An example and blog that show how to use the format is [here](http://gpuopen.com/rocm-with-harmony-combining-opencl-hcc-hsa-in-a-single-program). hsaco can be generated by hcc + extractkernel tool, cloc, the GCN assembler, or other tools.

#### Address Spaces
HCC defines a process-wide address space where the CPU and all devices allocate addresses from a single unified pool.
Thus addresses may be shared between contexts, and unlike the original CUDA definition a new context does not create a new address space for the device.

#### Using hipModuleLaunchKernel
`hipModuleLaunchKernel` is `cuLaunchKernel` in HIP world. It takes the same arguments as `cuLaunchKernel`. The argument `kernelParams` is not fully implemented for HCC. The workaround for it is, to use platform specific macros for each target. Or, `extra` argument can be used which works on both the platforms.

#### Additional Information
- HCC allocates staging buffers (used for unpinned copies) on a per-device basis.
- HCC creates a primary context when the HIP API is called. So in a pure driver API code, HIP/HCC will create a primary context while HIP/NVCC will have empty context stack.
HIP/HCC will push primary context to context stack when it is empty. This can have subtle differences on applications which mix the runtime and driver APIs.

### hip-clang Implementation Notes
#### .hip_fatbin
hip-clang links device code from different translation units together. For each device target, a code object is generated. Code objects for different device targets are bundled by clang-offload-bundler as one fatbinary, which is embeded as a global symbol `__hip_fatbin` in the .hip_fatbin section of the ELF file of the executable or shared object.

#### Initialization and Termination Functions
hip-clang generates initializatiion and termination functions for each translation unit for host code compilation. The initialization functions call `__hipRegisterFatBinary` to register the fatbinary embeded in the ELF file. They also call `__hipRegisterFunction` and `__hipRegisterVar` to register kernel functions and device side global variables. The termination functions call `__hipUnregisterFatBinary`.
hip-clang emits a global variable `__hip_gpubin_handle` of void** type with linkonce linkage and inital value 0 for each host translation unit. Each initialization function checks `__hip_gpubin_handle` and register the fatbinary only if `__hip_gpubin_handle` is 0 and saves the return value of `__hip_gpubin_handle` to `__hip_gpubin_handle`. This is to guarantee that the fatbinary is only registered once. Similar check is done in the termination functions.

#### Kernel Launching
hip-clang supports kernel launching by CUDA `<<<>>>` syntax, hipLaunchKernel, and hipLaunchKernelGGL. The latter two are macros which expand to CUDA `<<<>>>` syntax.

In host code, hip-clang emits a stub function with the same name and arguments as the kernel. In the body of this function, hipSetupArgument is called for each kernel argument, then hipLaunchByPtr is called with a function pointer to the stub function.

When the executable or shared library is loaded by the dynamic linker, the initilization functions are called. In the initialization functions, when `__hipRegisterFatBinary` is called, the code objects containing all kernels are loaded; when `__hipRegisterFunction` is called, the stub functions are associated with the corresponding kernels in code objects.

In the host code, for the `<<<>>>` statement, hip-clang first emits call of hipConfigureCall to set up the threads and grids, then emits call of the stub function with the given arguments. In the stub function, when the runtime host API function hipLaunchByPtr is called, the real kernel associated with the stub function is launched.

### NVCC Implementation Notes

#### Interoperation between HIP and CUDA Driver
CUDA applications may want to mix CUDA driver code with HIP code (see example below). This table shows the type equivalence to enable this interaction.

|**HIP Type**   |**CU Driver Type**|**CUDA Runtime Type**|
| ----          | ----             | ----                |
| hipModule_t   |  CUmodule        |                     |
| hipFunction_t |  CUfunction      |                     |
| hipCtx_t      |  CUcontext       |                     |
| hipDevice_t   |  CUdevice        |                     |
| hipStream_t   |  CUstream        | cudaStream_t        |
| hipEvent_t    |  CUevent         | cudaEvent_t         |
| hipArray      |  CUarray         | cudaArray           |

#### Compilation Options
The `hipModule_t` interface does not support `cuModuleLoadDataEx` function, which is used to control PTX compilation options.
HCC does not use PTX and does not support these compilation options.
In fact, HCC code objects always contain fully compiled ISA and do not require additional compilation as a part of the load step.
The corresponding HIP function `hipModuleLoadDataEx` behaves as `hipModuleLoadData` on HCC path (compilation options are not used) and as `cuModuleLoadDataEx` on NVCC path.
For example (CUDA):
```
CUmodule module;
void *imagePtr = ...;  // Somehow populate data pointer with code object

const int numOptions = 1;
CUJit_option options[numOptions];
void * optionValues[numOptions];

options[0] = CU_JIT_MAX_REGISTERS;
unsigned maxRegs = 15;
optionValues[0] = (void*)(&maxRegs);

cuModuleLoadDataEx(module, imagePtr, numOptions, options, optionValues);

CUfunction k;
cuModuleGetFunction(&k, module, "myKernel");
```
HIP:
```
hipModule_t module;
void *imagePtr = ...;  // Somehow populate data pointer with code object

const int numOptions = 1;
hipJitOption options[numOptions];
void * optionValues[numOptions];

options[0] = hipJitOptionMaxRegisters;
unsigned maxRegs = 15;
optionValues[0] = (void*)(&maxRegs);

// hipModuleLoadData(module, imagePtr) will be called on HCC path, JIT options will not be used, and
// cupModuleLoadDataEx(module, imagePtr, numOptions, options, optionValues) will be called on NVCC path
hipModuleLoadDataEx(module, imagePtr, numOptions, options, optionValues);

hipFunction_t k;
hipModuleGetFunction(&k, module, "myKernel");
```

The below sample shows how to use `hipModuleGetFunction`.

```
#include<hip_runtime.h>
#include<hip_runtime_api.h>
#include<iostream>
#include<fstream>
#include<vector>

#define LEN 64
#define SIZE LEN<<2

#ifdef __HIP_PLATFORM_HCC__
#define fileName "vcpy_isa.co"
#endif

#ifdef __HIP_PLATFORM_NVCC__
#define fileName "vcpy_isa.ptx"
#endif

#define kernel_name "hello_world"

int main(){
    float *A, *B;
    hipDeviceptr_t Ad, Bd;
    A = new float[LEN];
    B = new float[LEN];

    for(uint32_t i=0;i<LEN;i++){
        A[i] = i*1.0f;
        B[i] = 0.0f;
        std::cout<<A[i] << " "<<B[i]<<std::endl;
    }


#ifdef __HIP_PLATFORM_NVCC__
          hipInit(0);
          hipDevice_t device;
          hipCtx_t context;
          hipDeviceGet(&device, 0);
          hipCtxCreate(&context, 0, device);
#endif

    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);

    hipMemcpyHtoD(Ad, A, SIZE);
    hipMemcpyHtoD(Bd, B, SIZE);
    hipModule_t Module;
    hipFunction_t Function;
    hipModuleLoad(&Module, fileName);
    hipModuleGetFunction(&Function, Module, kernel_name);

    std::vector<void*>argBuffer(2);
    memcpy(&argBuffer[0], &Ad, sizeof(void*));
    memcpy(&argBuffer[1], &Bd, sizeof(void*));

    size_t size = argBuffer.size()*sizeof(void*);

    void *config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, &argBuffer[0],
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
      HIP_LAUNCH_PARAM_END
    };

    hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL, (void**)&config);

    hipMemcpyDtoH(B, Bd, SIZE);
    for(uint32_t i=0;i<LEN;i++){
        std::cout<<A[i]<<" - "<<B[i]<<std::endl;
    }

#ifdef __HIP_PLATFORM_NVCC__
          hipCtxDetach(context);
#endif

    return 0;
}
```

## HIP Module and Texture Driver API

HIP supports texture driver APIs however texture reference should be declared in host scope. Following code explains the use of texture reference for __HIP_PLATFORM_HCC__ platform. 

```
// Code to generate code object

#include "hip/hip_runtime.h"
extern texture<float, 2, hipReadModeElementType> tex;

__global__ void tex2dKernel(hipLaunchParm lp, float* outputData,
                             int width,
                             int height)
{
    int x = hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y*hipBlockDim_y + hipThreadIdx_y;
    outputData[y*width + x] = tex2D(tex, x, y);
}

```
```
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
```
