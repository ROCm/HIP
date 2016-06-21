# Release notes

We have attempted to document known bugs and limitations - in particular the [HIP Kernel Language](docs/markdown/hip_kernel_language.md) document uses the phrase "Under Development", and the [HIP Runtime API bug list](http://gpuopen-professionalcompute-tools.github.io/HIP/bug.html) lists known bugs.    Some of the key items we are working on:
- Tuning built-in functions, including shfl.
- Performance optimization.


Stay tuned - the work for many of these features is already in-flight.

===================================================================================================
Release:0.90.00
Date: 2016.06.xx
-Support dynamic shared memory allocations
-Min HCC compiler version is > 16186.
-Expanded math functions (device and host).  Document unsupported functions.
-Initial support for hipBlas headers.
-hipFree with null pointer initializes runtime and returns success.
-Improve error code reporting on nvcc.
-add hipPeekAtError for nvcc..
===================================================================================================

===================================================================================================
Release:0.86.00
Date: 2016.06.06
- Add clang-hipify : clang-based hipify tool.  Improved parsing of source code, and automates 
  creation of hipLaunchParm variable.
- Implement memory register / unregister commands (hipHostRegister, hipHostUnregister)
- Add cross-linking support between G++ and HCC, in particular for interfaces that use
  standard C++ libraries (ie std::vectors, std::strings).  HIPCC now uses libstdc++ by default on the HCC
  compilation path.
- More samples including gpu-burn, SHOC, nbody, rtm.  See [HIP-Examples](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP-Examples)

===================================================================================================

## Revision History:

===================================================================================================
Release:0.84.01
Date: 2016.04.25
- Refactor HIP make and install system:
    - Move to CMake. Refer to the installation section in README.md for details.
    - Split source into multiple modular .cpp and .h files.
    - Create static library and link.
    - Set HIP_PATH to install.
- Make hipDevice and hipStream thread-safe.
    - Prefered hipStream usage is still to create new streams for each new thread, but it works even if you don;t.
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



