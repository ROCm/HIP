# HIP RTC Programming Guide

## HIP RTC lib
HIP allows you to compile kernels at runtime with its ```hiprtc*``` APIs.
Kernels can be store as a text string and can be passed on to hiprtc APIs alongside options to guide the compilation.

## Example
To use hiprtc functionality, hiprtc header needs to be included first.
```#include <hip/hiprtc.h>```


Kernels can be stored in a string:
```cpp
static constexpr auto kernel {
R"(
    extern "C"
    __global__ void gpu_kernel(...) {
        // Kernel Functionality
    }
)"};
```

Now to compile this kernel, it needs to be associated with hiprtcProgram type, which is done via declaring ```hiprtcProgram prog;``` and associating the string of kernel with this program:

```cpp
hiprtcCreateProgram(&prog,                 // hiprtc program
                    kernel,                // kernel string
                    "gpu_kernel.cu",       // Name of the file
                    num_headers,           // Number of headers
                    &header_sources[0],    // Header sources
                    &header_names[0]);     // Name of header files
```

hiprtcCreateProgram API also allows you to add headers which can be included in your rtc program.
For online compilation, the compiler pre-defines HIP device API functions, HIP specific types and macros for device compilation, but does not include standard C/C++ headers by default. Users can only include header files provided to hiprtcCreateProgram.

After associating the kernel string with hiprtcProgram, you can now compile this program using:
```cpp
hiprtcCompileProgram(prog,     // hiprtcProgram
                    0,         // Number of options
                    options);  // Clang Options [Supported Clang Options](clang_options.md)
```

hiprtcCompileProgram returns a status value which can be converted to string via ```hiprtcGetErrorString```. If compilation is successful, hiprtcCompileProgram will return ```HIPRTC_SUCCESS```.

If the compilation fails, you can look up the logs via:

```cpp
size_t logSize;
hiprtcGetProgramLogSize(prog, &logSize);

if (logSize) {
  string log(logSize, '\0');
  hiprtcGetProgramLog(prog, &log[0]);
  // Corrective action with logs
}
```

If the compilation is successful, you can load the compiled binary in a local variable.
```cpp
size_t codeSize;
hiprtcGetCodeSize(prog, &codeSize);

vector<char> kernel_binary(codeSize);
hiprtcGetCode(kernel_binary, code.data());
```

After loading the binary, hiprtcProgram can be destroyed.
```hiprtcDestroyProgram(&prog);```

The binary present in ```kernel_binary``` can now be loaded via ```hipModuleLoadData``` API.
```cpp
hipModule_t module;
hipFunction_t kernel;

hipModuleLoadData(&module, kernel_binary.data());
hipModuleGetFunction(&kernel, module, "gpu_kernel");
```

And now this kernel can be launched via hipModule APIs.

Please have a look at saxpy.cpp and hiprtcGetLoweredName.cpp files for a detailed example.

## HIPRTC specific options
HIPRTC provides a few hiprtc specific flags
 - ```--gpu-architecture``` : This flag can guide the code object generation for a specific gpu arch. Example: ```--gpu-architecture=gfx906:sramecc+:xnack-```, its equivalent to ```--offload-arch```.

## Deprecation notice
Users will be required to link to libhiprtc.so/libhiprtc.dll in future releases. Currently all symbols are present in libhipamd64.so/libhipamd64.dll and there is a plan in action to separate HIPRTC APIs from HIP APIs.
