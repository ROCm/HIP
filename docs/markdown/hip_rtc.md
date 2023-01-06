# HIP RTC Programming Guide

## HIP RTC lib
HIP allows you to compile kernels at runtime with its ```hiprtc*``` APIs.
Kernels can be store as a text string and can be passed on to hiprtc APIs alongside options to guide the compilation.

NOTE:

  - This library can be used on systems without HIP install nor AMD GPU driver installed at all (offline compilation). Therefore it does not depend on any HIP runtime library.
  - But it does depend on COMGr. We may try to statically link COMGr into hipRTC to avoid any ambiguity.
  - Developers can decide to bundle this library with their application.

## Compile APIs

#### Example
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
hiprtcGetCode(prog, kernel_binary.data());
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

#### HIPRTC specific options
HIPRTC provides a few hiprtc specific flags
 - ```--gpu-architecture``` : This flag can guide the code object generation for a specific gpu arch. Example: ```--gpu-architecture=gfx906:sramecc+:xnack-```, its equivalent to ```--offload-arch```.
    - This option is compulsory if compilation is done on a system without AMD GPUs supported by HIP runtime.
    - Otherwise, hipRTC will load the hip runtime and gather the current device and its architecture info and use it as option.
 - ```-fgpu-rdc``` : This flag when provided during the hiprtcCompileProgram generates the bitcode (HIPRTC doesn't convert this bitcode into ISA and binary). This bitcode can later be fetched using hiprtcGetBitcode and hiprtcGetBitcodeSize APIs.

#### Bitcode
In the usual scenario, the kernel associated with hiprtcProgram is compiled into the binary which can be loaded and run. However, if -fpu-rdc option is provided in the compile options, HIPRTC calls comgr and generates only the LLVM bitcode.  It doesn't convert this bitcode to ISA and generate the final binary.
```cpp
std::string sarg = std::string("-fgpu-rdc");
const char* options[] = {
    sarg.c_str() };
hiprtcCompileProgram(prog, // hiprtcProgram
                     1,    // Number of options
                     options);
```

If the compilation is successful, one can load the bitcode in a local variable using the bitcode APIs provided by HIPRTC.
```cpp
size_t bitCodeSize;
hiprtcGetBitcodeSize(prog, &bitCodeSize);

vector<char> kernel_bitcode(bitCodeSize);
hiprtcGetBitcode(prog, kernel_bitcode.data());
```

## Linker APIs

#### Introduction
The bitcode generated using the HIPRTC Bitcode APIs can be loaded using hipModule APIs and also can be linked with other generated bitcodes with appropriate linker flags using the HIPRTC linker APIs.  This also provides more flexibility and optimizations to the applications who want to generate the binary dynamically according to their needs. The input bitcodes can be generated only for a specific architecture or it can be a bundled bitcode which is generated for multiple architectures.

#### Example
Firstly, hiprtc link instance or a pending linker invocation must be created using hiprtcLinkCreate, with the appropriate linker options provided.
```cpp
hiprtcLinkCreate( num_options,           // number of options
                  options,               // Array of options
                  option_vals,           // Array of option values cast to void*
                  &rtc_link_state );     // hiprtc link state created upon success
```

Following which, the bitcode data can be added to this link instance via hiprtcLinkAddData (if the data is present as a string) or hiprtcLinkAddFile (if the data is present as a file) with the appropriate input type according to the data or the bitcode used.
```cpp
hiprtcLinkAddData(rtc_link_state,        // hiprtc link state
                  input_type,            // type of the input data or bitcode
                  bit_code_ptr,          // input data which is null terminated
                  bit_code_size,         // size of the input data
                  "a",                   // optional name for this input
                  0,                     // size of the options
                  0,                     // Array of options applied to this input
                  0);                    // Array of option values cast to void*
```
```cpp
hiprtcLinkAddFile(rtc_link_state,        // hiprtc link state
                  input_type,            // type of the input data or bitcode
                  bc_file_path.c_str(),  // path to the input file where bitcode is present
                  0,                     // size of the options
                  0,                     // Array of options applied to this input
                  0);                    // Array of option values cast to void*
```

Once the bitcodes for multiple archs are added to the link instance, the linking of the device code must be completed using hiprtcLinkComplete which generates the final binary.
```cpp
hiprtcLinkComplete(rtc_link_state,       // hiprtc link state
                   &binary,              // upon success, points to the output binary
                   &binarySize);         // size of the binary is stored (optional)
```

If the hiprtcLinkComplete returns successfully, the generated binary can be loaded and run using the hipModule* APIs.
```cpp
hipModuleLoadData( &module, binary );
```

#### Note
 - The compiled binary must be loaded before hiprtc link instance is destroyed using the hiprtcLinkDestroy API.
```cpp
hiprtcLinkDestroy( rtc_link_state);
```
 - The correct sequence of calls is : hiprtcLinkCreate, hiprtcLinkAddData or hiprtcLinkAddFile, hiprtcLinkComplete, hiprtcModuleLoadData, hiprtcLinkDestroy.

#### Input Types
HIPRTC provides hiprtcJITInputType enumeration type which defines the input types accepted by the Linker APIs. Here are the enum values of hiprtcJITInputType. However only the input types  HIPRTC_JIT_INPUT_LLVM_BITCODE, HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE and HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE are supported currently.

```cpp
HIPRTC_JIT_INPUT_CUBIN = 0,
HIPRTC_JIT_INPUT_PTX,
HIPRTC_JIT_INPUT_FATBINARY,
HIPRTC_JIT_INPUT_OBJECT,
HIPRTC_JIT_INPUT_LIBRARY,
HIPRTC_JIT_INPUT_NVVM,
HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES,
HIPRTC_JIT_INPUT_LLVM_BITCODE = 100,
HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = 101,
HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = 102,
HIPRTC_JIT_NUM_INPUT_TYPES = (HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES + 3)
```

## Error Handling
HIPRTC defines the hiprtcResult enumeration type and a function hiprtcGetErrorString for API call error handling.  hiprtcResult enum defines the API result codes.  HIPRTC APIs return hiprtcResult to indicate the call result. hiprtcGetErrorString function returns a string describing the given hiprtcResult code, e.g., HIPRTC_SUCCESS to "HIPRTC_SUCCESS". For unrecognized enumeration values, it returns "Invalid HIPRTC error code".

hiprtcResult enum supported values and the hiprtcGetErrorString usage are mentioned below.
```cpp
HIPRTC_SUCCESS = 0,
HIPRTC_ERROR_OUT_OF_MEMORY = 1,
HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
HIPRTC_ERROR_INVALID_INPUT = 3,
HIPRTC_ERROR_INVALID_PROGRAM = 4,
HIPRTC_ERROR_INVALID_OPTION = 5,
HIPRTC_ERROR_COMPILATION = 6,
HIPRTC_ERROR_LINKING = 7,
HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 8,
HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 9,
HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 10,
HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 11,
HIPRTC_ERROR_INTERNAL_ERROR = 12
```
```cpp
hiprtcResult result;
result = hiprtcCompileProgram(prog, 1, opts);
if (result != HIPRTC_SUCCESS) {
std::cout << "hiprtcCompileProgram fails with error " << hiprtcGetErrorString(result);
}
```

## HIPRTC General APIs
HIPRTC provides the following API for querying the version.

hiprtcVersion(int* major, int* minor) -  This sets the output parameters major and minor with the HIP Runtime compilation major version and minor version number respectively.

Currently, it returns hardcoded value.  This should be implemented to return HIP runtime major and minor version in the future releases.

## Lowered Names (Mangled Names)
HIPRTC mangles the ```__global__``` function names and names of ```__device__``` and ```__constant__``` variables. If the generated binary is being loaded using the HIP Runtime API, the kernel function or ```__device__/__constant__``` variable must be looked up by name, but this is very hard when the name has been mangled. To overcome this, HIPRTC provides API functions that map  ```__global__``` function or ```__device__/__constant__``` variable names in the source to the mangled names present in the generated binary.

The two APIs hiprtcAddNameExpression and hiprtcGetLoweredName provide this functionality. First, a 'name expression' string denoting the address for the ```__global__``` function or ```__device__/__constant__``` variable is provided to hiprtcAddNameExpression. Then, the program is compiled with hiprtcCompileProgram. During compilation, HIPRTC will parse the name expression string as a C++ constant expression at the end of the user program. Finally, the function hiprtcGetLoweredName is called with the original name expression and it returns a pointer to the lowered name. The lowered name can be used to refer to the kernel or variable in the HIP Runtime API.

#### Note
 - The identical name expression string must be provided on a subsequent call to hiprtcGetLoweredName to extract the lowered name.
 - The correct sequence of calls is : hiprtcAddNameExpression, hiprtcCompileProgram, hiprtcGetLoweredName, hiprtcDestroyProgram.
 - The lowered names must be fetched using hiprtcGetLoweredName only after the HIPRTC program has been compiled, and before it has been destroyed.

#### Example
kernel containing various definitions ```__global__``` functions/function templates and ```__device__/__constant__``` variables can be stored in a string.

```cpp
static constexpr const char gpu_program[]{
R"(
__device__ int V1; // set from host code
static __global__ void f1(int *result) { *result = V1 + 10; }
namespace N1 {
namespace N2 {
__constant__ int V2; // set from host code
__global__ void f2(int *result) { *result = V2 + 20; }
}
}
template<typename T>
__global__ void f3(int *result) { *result = sizeof(T); }
)"};
```
hiprtcAddNameExpression is called with various name expressions referring to the address of ```__global__``` functions and ```__device__/__constant__``` variables.

```cpp
kernel_name_vec.push_back("&f1");
kernel_name_vec.push_back("N1::N2::f2");
kernel_name_vec.push_back("f3<int>");
for (auto&& x : kernel_name_vec) hiprtcAddNameExpression(prog, x.c_str());
variable_name_vec.push_back("&V1");
variable_name_vec.push_back("&N1::N2::V2");
for (auto&& x : variable_name_vec) hiprtcAddNameExpression(prog, x.c_str());
```

After which, the program is compiled using hiprtcCompileProgram and the generated binary is loaded using hipModuleLoadData.  And the mangled names can be fetched using hirtcGetLoweredName.
```cpp
for (decltype(variable_name_vec.size()) i = 0; i != variable_name_vec.size(); ++i) {
  const char* name;
  hiprtcGetLoweredName(prog, variable_name_vec[i].c_str(), &name);
}
```
```cpp
for (decltype(kernel_name_vec.size()) i = 0; i != kernel_name_vec.size(); ++i) {
  const char* name;
  hiprtcGetLoweredName(prog, kernel_name_vec[i].c_str(), &name);
}
```

The mangled name of the variables are used to look up the variable in the module and update its value.
```
hipDeviceptr_t variable_addr;
size_t bytes{};
hipModuleGetGlobal(&variable_addr, &bytes, module, name);
hipMemcpyHtoD(variable_addr, &initial_value, sizeof(initial_value));
```

Finally, the mangled name of the kernel is used to launch it using the hipModule APIs.

```cpp
hipFunction_t kernel;
hipModuleGetFunction(&kernel, module, name);
hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, config);
```

Please have a look at hiprtcGetLoweredName.cpp for the detailed example.

## Versioning
HIPRTC follows the below versioning.
 - Linux
    - HIPRTC follows the same versioning as HIP runtime library.
    - The soname field for the shared library is set to MAJOR version. eg: For HIP 5.3 the soname is set to 5 (hiprtc.so.5).
 - Windows
    - Currently, the HIPRTC dll doesn't have any version attached. It is just named as hiprtc.dll.
    - In the upcoming releases, HIPRTC dll will be named as hiprtc_XXYY.dll where XX is MAJOR version and YY is MINOR version. eg: For HIP 5.3 the name is hiprtc_0503.dll.

## Deprecation notice
 - Currently HIPRTC APIs are separated from HIP APIs and HIPRTC is available as a separate library libhiprtc.so/libhiprtc.dll. But on Linux, HIPRTC symbols are also present in libhipamd64.so in order to support the existing applications. Gradually, these symbols will be removed from HIP library and applications using HIPRTC will be required to explictly link to HIPRTC library. However, on Windows hiprtc.dll must be used as the hipamd64.dll doesn't contain the HIPRTC symbols.
 - Datatypes such as uint32_t, uint64_t, int32_t, int64_t defined in std namespace in HIPRTC are deprecated and will be removed in the upcoming releases since these can conflict with the standard C++ datatypes. These datatypes are now prefixed with __hip__, e.g. __hip_uint32_t. type_traits templates previously defined in std namespace are moved to __hip_internal namespace as implementation details. Apps previously using std::uint32_t or similar types should use __hip_ prefixed types to avoid conflicts with standard std namespace.
