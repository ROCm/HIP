# Release notes

We have attempted to document known bugs and limitations - in particular the [HIP Kernel Language](docs/markdown/hip_kernel_language.md) document uses the phrase "Under Development", and the [HIP Runtime API bug list](http://rocm-developer-tools.github.io/HIP/bug.html) lists known bugs. 


===================================================================================================


## Revision History:

===================================================================================================
Release: 1.5
Date:
- Support threadIdx, blockIdx, blockDim directly (no need for hipify conversions in kernels.)  HIP
  Kernel syntax is now identical to CUDA kernel syntax - no need for extra parms or conversions.
- Refactor launch syntax.  HIP now extracts kernels from the executable and launches them using the 
  existing module interface.  Kernels dispatch no longer flows through HCC.  Result is faster
  kernel launches and with less resource usage (no signals required). 
- Remove requirement for manual "serializers" previously required when passing complex structures
  into kernels.
- Remove need for manual destructors
- Provide printf in device code
- Support for globals when using module API
- hipify-clang now supports using newer versions of clang
- HIP texture support equivalent to CUDA texture driver APIs
- Updates to hipify-perl, hipify-clang and documentation


===================================================================================================
Release: 1.4
Date: 2017.10.06
- Improvements to HIP event management
- Added new HIP_TRACE_API options
- Enabled device side assert support
- Several bug fixes including hipMallocArray, hipTexture fetch
- Support for RHEL/CentOS 7.4
- Updates to hipify-perl, hipify-clang and documentation


===================================================================================================
Release: 1.3
Date: 2017.08.16
- hipcc now auto-detects amdgcn arch. No need to specify the arch when building for same system.
- HIP texture support (run-time APIs)
- Implemented __threadfence_support
- Improvements in HIP context management logic
- Bug fixes in several APIs including hipDeviceGetPCIBusId, hipEventDestroy, hipMemcpy2DAsync
- Updates to hipify-clang and documentation
- HIP development now fully open and on GitHub. Developers should submit pull requests.


===================================================================================================
Release: 1.2
Date: 2017.06.29
- new APIs: hipMemcpy2DAsync, hipMallocPitch, hipHostMallocCoherent, hipHostMallocNonCoherent
- added support for building hipify-clang using clang 3.9
- hipify-clang updates for CUDA 8.0 runtime+driver support
- renamed hipify to hipify-perl
- initial implementation of hipify-cmakefile
- several documentation updates & bug fixes
- support for abort() function in device code


===================================================================================================
Release: 1.0.17102
Date: 2017.03.07
- Lots of improvements to hipify-clang.
- Added HIP package config for cmake.
- Several bug fixes and documentation updates.


===================================================================================================
Release: 1.0.17066
Date: 2017.02.11
- Improved support for math device functions.
- Added several half math device functions.
- Enabled support for CUDA 8.0 in hipify-clang.
- Lots of bug fixes and documentation updates.


===================================================================================================
Release: 1.0.17015
Date: 2017.01.06
- Several improvements to the hipify-clang infrastructure.
- Refactored module and function APIs.
- HIP now defaults to linking against the shared runtime library.
- Documentation updates.


===================================================================================================
Release: 1.0.16502
Date: 2016.12.13
- Added several fast math and packaged math instrincs
- Improved debug and profiler documentation
- Support for building and linking to HIP shared library
- Several improvements to hipify-clang
- Several bug fixes


===================================================================================================
Release: 1.0.16461
Date: 2016.11.14
- Significant changes to the HIP Profiling APIs. Refer to the documentation for details
- Improvements to P2P support
- New API: hipDeviceGetByPCIBusId
- Several bug fixes in NV path
- hipModuleLaunch now works for multi-dim kernels


===================================================================================================
Release:1.0
Date: 2016.11.8
- Initial implementation for FindHIP.cmake
- HIP library now installs as a static library by default
- Added support for HIP context and HIP module APIs
- Major changes to HIP signal & memory management implementation
- Support for complex data type and math functions
- clang-hipify is now known as hipify-clang
- Added several new HIP samples
- Preliminary support for new APIs: hipMemcpyToSymbol, hipDeviceGetLimit, hipRuntimeGetVersion
- Added support for async memcpy driver API (for example hipMemcpyHtoDAsync)
- Support for memory management device functions: malloc, free, memcpy & memset
- Removed deprecated HIP runtime header locations. Please include "hip/hip_runtime.h" instead of "hip_runtime.h". You can use `find . -type f -exec sed -i 's:#include "hip_runtime.h":#include "hip/hip_runtime.h":g' {} +` to replace all such references


===================================================================================================
Release:0.92.00
Date: 2016.8.14
- hipLaunchKernel supports one-dimensional grid and/or block dims, without explicit cast to dim3 type (actually in 0.90.00)
- fp16 software support
- Support for Hawaii dGPUs using environment variable ROCM_TARGET=hawaii
- Support hipArray
- Improved profiler support
- Documentation updates
- Improvements to clang-hipify


===================================================================================================
Release:0.90.00
Date: 2016.06.29
- Support dynamic shared memory allocations
- Min HCC compiler version is > 16186.
- Expanded math functions (device and host).  Document unsupported functions.
- hipFree with null pointer initializes runtime and returns success.
- Improve error code reporting on nvcc.
- Add hipPeekAtError for nvcc.


===================================================================================================
Release:0.86.00
Date: 2016.06.06
- Add clang-hipify : clang-based hipify tool.  Improved parsing of source code, and automates 
  creation of hipLaunchParm variable.
- Implement memory register / unregister commands (hipHostRegister, hipHostUnregister)
- Add cross-linking support between G++ and HCC, in particular for interfaces that use
  standard C++ libraries (ie std::vectors, std::strings).  HIPCC now uses libstdc++ by default on the HCC
  compilation path.
- More samples including gpu-burn, SHOC, nbody, rtm.  See [HIP-Examples](https://github.com/ROCm-Developer-Tools/HIP-Examples)


===================================================================================================
Release:0.84.01
Date: 2016.04.25
- Refactor HIP make and install system:
    - Move to CMake. Refer to the installation section in README.md for details.
    - Split source into multiple modular .cpp and .h files.
    - Create static library and link.
    - Set HIP_PATH to install.
- Make hipDevice and hipStream thread-safe.
    - Preferred hipStream usage is still to create new streams for each new thread, but it works even if you don;t.
- Improve automated platform detection: If AMD GPU is installed and detected by driver, default HIP_PLATFORM to hcc.
- HIP_TRACE_API now prints arguments to the HIP function (in addition to name of function).
- Deprecate hipDeviceGetProp (Replace with hipGetDeviceProp)
- Deprecate hipMallocHost (Replace with hipHostMalloc)
- Deprecate hipFreeHost (Replace with hipHostFree)
- The mixbench benchmark tool for measuring operational intensity now has a HIP target, in addition to CUDA and OpenCL.  Let the comparisons begin. :)    
See here for more : https://github.com/ekondis/mixbench.


===================================================================================================
Release:0.82.00
Date: 2016.03.07
- Bump minimum required HCC workweek to 16074.
- Bump minimum required ROCK-Kernel-Driver and ROCR-Runtime to Developer Preview 2.
- Enable multi-GPU support.
  * Use hipSetDevice to select a device for subsequent kernel calls and memory allocations.
  * CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICE environment variable selects devices visible to the runtime.
- Support hipStreams – send sequences of copy and kernel commands to a device.
  * Asynchronous copies supported.
- Optimize memory copy operations.
- Support hipPointerGetAttribute – can determine if a pointer is host or device.
- Enable atomics to local memory.
- Support for LC Direct-To-ISA path.
- Improved free memory reporting.
  * hipMemGetInfo (report full memory used in current process).
  * hipDeviceReset (deletes all memory allocated by current process).


===================================================================================================
Release:0.80.01
Date: 2016.02.18
- Improve reporting and support for device-side math functions.
- Update Runtime Documentation.
- Improve implementations of cross-lane operations (_ballot, _any, _all).
- Provide shuffle intrinsics (performance optimization in-progress).
- Support hipDeviceAttribute for querying "one-shot" device attributes, as an alternative to hipGetDeviceProperties.


===================================================================================================
Release:0.80.00
Date: 2016.01.25

Initial release with GPUOpen Launch.



