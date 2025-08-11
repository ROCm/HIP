# Glossary of terms

* **host**, **host CPU** : Executes the HIP runtime API and is capable of initiating kernel launches to one or more devices.
* **default device** : Each host thread maintains a default device.
Most HIP runtime APIs (including memory allocation, copy commands, kernel launches) do not accept an explicit device
argument but instead implicitly use the default device.
The default device can be set with `hipSetDevice`.

* **active host thread** - the thread which is running the HIP APIs.

* **HIP-Clang** - Heterogeneous AMDGPU Compiler, with its capability to compile HIP programs on AMD platform (https://github.com/RadeonOpenCompute/llvm-project).

* **clr** - a repository for AMD Compute Language Runtime, contains source codes for AMD's compute languages runtimes: HIP and OpenCL.
clr (https://github.com/ROCm/clr) contains the following three parts,

  * `hipamd`: contains implementation of HIP language on AMD platform.
  * `rocclr`: contains common runtime used in HIP and OpenCL, which provides virtual device interfaces that compute runtimes interact with different backends such as ROCr on Linux or PAL on Windows.
  * `opencl`: contains implementation of OpenCL on AMD platform.

* **hipify tools** - tools to convert CUDA code to portable C++ code (https://github.com/ROCm/HIPIFY).

* **`hipconfig`** - tool to report various configuration properties of the target platform.

* **`nvcc`** - NVIDIA CUDA `nvcc` compiler, do not capitalize.
