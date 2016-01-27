# Release notes

Since this is an early access release and we are still in development towards the production ready version Boltzmann Driver and runtime we recommend this release be used for research and early application development.  

We have attempted to document known bugs and limitations - in particular the [HIP Kernel Language](docs/markdown/hip_kernel_language.md) document uses the phrase "Under Development", and the [HIP Runtime API bug list](http://gpuopen-professionalcompute-tools.github.io/HIP/bug.html) lists known bugs.    Some of the key items we are working on:
- Async memory copies.
- hipStream support.
- Multi-GPU
- Shared-scope atomic operations. (due to compiler limitation, shared-scope map atomics map to global scope)
- Tuning built-in functions, including shfl.
- Performance optimization.


Stay tuned - the work for many of these features is already in-flight.

