## inline asm  ###

This tutorial is about how to use inline GCN asm in kernel. In this tutorial, we'll explain how to by using the simple Matrix Transpose.

## Introduction:

If you want to take advantage of the extra performance benefits of writing in assembly as well as take advantage of special GPU hardware features that were only available through assemby, then this tutorial is for you. In this tutorial we'll be explaining how to start writing inline asm in kernel.

For more insight Please read the following blogs by Ben Sander 
[The Art of AMDGCN Assembly: How to Bend the Machine to Your Will](gpuopen.com/amdgcn-assembly)
[AMD GCN Assembly: Cross-Lane Operations](http://gpuopen.com/amd-gcn-assembly-cross-lane-operations/)

For more information:
[AMD GCN3 ISA Architecture Manual](http://gpuopen.com/compute-product/amd-gcn3-isa-architecture-manual/)
[User Guide for AMDGPU Back-end](llvm.org/docs/AMDGPUUsage.html)

## Requirement:
For hardware requirement and software installation [Installation](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/INSTALL.md) 

## prerequiste knowledge:

Programmers familiar with CUDA, OpenCL will be able to quickly learn and start coding with the HIP API. In case you are not, don't worry. You choose to start with the best one. We'll be explaining everything assuming you are completely new to gpgpu programming.

## Simple Matrix Transpose 

We will be using the Simple Matrix Transpose application from the our very first tutorial.

## asm() Assembler statement

We insert the GCN isa into the kernel using asm() Assembler statement. In the same sourcecode, we used for MatrixTranspose. We'll add the following:

`  asm volatile ("v_mov_b32_e32 %0, %1" : "=v" (out[x*width + y]) : "v" (in[y*width + x]));                    `

## How to build and run:
Use the make command and execute it using ./exe
Use hipcc to build the application, which is using hcc on AMD and nvcc on nvidia.


## More Info:
- [HIP FAQ](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/docs/markdown/hip_faq.md)
- [HIP Kernel Language](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/docs/markdown/hip_kernel_language.md)
- [HIP Runtime API (Doxygen)](http://gpuopen-professionalcompute-tools.github.io/HIP)
- [HIP Porting Guide](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/docs/markdown/hip_porting_guide.md)
- [HIP Terminology](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/docs/markdown/hip_terms.md) (including Rosetta Stone of GPU computing terms across CUDA/HIP/HC/AMP/OpenL)
- [clang-hipify](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/clang-hipify/README.md)
- [Developer/CONTRIBUTING Info](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/CONTRIBUTING.md)
- [Release Notes](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/RELEASE.md)
