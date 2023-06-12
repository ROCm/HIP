# Terms used in HIP Documentation

- host, host cpu : Executes the HIP runtime API and is capable of initiating kernel launches to one or more devices.
- default device : Each host thread maintains a "default device".
Most HIP runtime APIs (including memory allocation, copy commands, kernel launches) do not use accept an explicit device
argument but instead implicitly use the default device.
The default device can be set with hipSetDevice.

- "active host thread" - the thread which is running the HIP APIs.

- HIP-Clang - Heterogeneous AMDGPU Compiler, with its capability to compile HIP programs on AMD platform (https://github.com/RadeonOpenCompute/llvm-project).

- ROCclr - a virtual device interface that compute runtimes interact with different backends such as ROCr on Linux or PAL on Windows.
  The ROCclr (https://github.com/ROCm-Developer-Tools/ROCclr) is an abstraction layer allowing runtimes to work on both OSes without much effort.

- hipify tools - tools to convert CUDA code to portable C++ code (https://github.com/ROCm-Developer-Tools/HIPIFY).

- hipconfig - tool to report various configuration properties of the target platform.

- nvcc = nvcc compiler, do not capitalize.
