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


## Revision History:

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
- Support hipDeviceAttribute for querying "one-shot" device attributes, as an alternative to hipDeviceGetProperties.


===================================================================================================
Release:0.80.00 :
Date: 2016.01.25

Initial release with GPUOpen Launch.



