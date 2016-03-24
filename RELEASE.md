# Release notes

Since this is an early access release and we are still in development towards the production ready version Boltzmann Driver and runtime we recommend this release be used for research and early application development.  

We have attempted to document known bugs and limitations - in particular the [HIP Kernel Language](docs/markdown/hip_kernel_language.md) document uses the phrase "Under Development", and the [HIP Runtime API bug list](http://gpuopen-professionalcompute-tools.github.io/HIP/bug.html) lists known bugs.    Some of the key items we are working on:
- Async memory copies.
- hipStream support.
- Multi-GPU
- Shared-scope atomic operations. (due to compiler limitation, shared-scope map atomics map to global)
- Tuning built-in functions, including shfl.
- Performance optimization.


Stay tuned - the work for many of these features is already in-flight.

Next:
- Refactor HIP make and install system:
    - Move to CMake.
    - Split source into multiple modular .cpp and .h files.
    - Create static library and link.
- Deprecate hipDeviceGetProp, replace with hipGetDeviceProp
- Deprecate hipMallocHost (Replace with hipHostMalloc)
- Deprecate hipFreeHost (Replace with hipHostFree). 


## Revision History:

===================================================================================================
Release:0.80.01
Date: 2016.02.18
- Improve reporting and support for device-side math functions.
- Update Runtime Documentation.
- Improve implementations of cross-lane operations (_ballot, _any, _all).
- Provide shuffle intrinsics (performance optimization in-progress).
- Support hipDeviceAttribute for querying "one-shot" device attributes, as an alternative to hipGetDeviceProperties.


===================================================================================================
Release:0.80.00 :
Date: 2016.01.25

Initial release with GPUOpen Launch.



