.. meta::
  :description: This chapter provides and introduction to the HIP API.
  :keywords: AMD, ROCm, HIP, CUDA, C++ language extensions

.. _intro-to-hip:

*******************************************************************************
Introduction to HIP
*******************************************************************************

The Heterogeneous-computing Interface for Portability (HIP) is a C++ runtime API and kernel language that lets you create portable applications for AMD and NVIDIA GPUs from a single source code. 

* HIP is a thin API with very little performance impact over coding directly in NVIDIA CUDA or AMD ROCm.
* HIP enables coding in a single-source C++ programming language including features such as templates, C++11 lambdas, classes, namespaces, and more.
* The :doc:`HIPify <hipify:index>` tools convert source from CUDA to HIP.
* Developers can specialize for the platform (CUDA or AMD) to tune for performance or handle tricky cases.

HIP includes the runtime API, kernel language, compilers (clang, hipcc), code profilers (rocprof, omnitrace), debugging tools (rocgdb), and libraries to create heterogeneous applications running on both CPUs and GPUs. HIP provides marshalling libraries like :doc:`hipFFT <hipfft:index>` or :doc:`hipBLAS <hipblas:index>` that act as a thin programming layer over either NVIDIA CUDA or AMD ROCm to enable support for either language as a back-end. These libraries offer pointer-based memory interfaces and are easily integrated into your applications.

HIP supports the ability to build and run on either AMD GPUs or NVIDIA GPUs. GPU Programmers familiar with NVIDIA CUDA or OpenCL will find the HIP API familiar and easy to use. Developers no longer need to choose between CUDA and ROCm. You can quickly port your application to run on the available hardware while maintaining a single codebase. The HIPify tools, based on the clang front-end and Perl language can convert CUDA API calls into HIP. However, HIP is not intended to be a drop-in replacement for CUDA, and developers should expect to do some manual coding and performance tuning work to port and existing project as described in `HIP Porting Guide <../how-to/hip_porting_guide.html>`_.  

For the AMD ROCm platform, HIP provides a header and runtime library built on top of HIP-Clang compiler in the repository `Common Language Runtime (CLR) <./amd_clr.html>`_.  The HIP runtime implements HIP streams, events, and memory APIs, and is a object library that is linked with the application.  The source code for all headers and the library implementation is available on GitHub. HIP developers on ROCm can use :doc:`ROCgdb <rocgdb:index>` for debugging and :doc:`ROCProfiler <rocprofiler:index>` for profiling.

For the NVIDIA CUDA platform, HIP provides a header file in the repository `hipother <https://github.com/ROCm/hipother>`_ which translate from the HIP runtime APIs to CUDA runtime APIs.  The header file contains mostly inlined functions and thus has very low overhead. Developers coding in HIP should expect the same performance as coding in native CUDA.  The code is then compiled with ``nvcc``, the standard C++ compiler provided with the CUDA SDK.  Developers can use any tools supported by the CUDA SDK including the CUDA debugger and profiler.

HIP is designed to work seamlessly with the ROCm Runtime (ROCr). HIP provides two types of APIs: those that run on the CPU, or host system, and those that run on GPUs, or accelerators. The host-based code is used to create device buffers, move data between the host application and a device, launch the device code (or kernel), manage streams and events, and perform synchronization. The device or kernel code, is written to run on the GPU and provide significantly increased performance for certain types of functions as described in `Programming Model <./programming_model.html>`_. 

In summary, HIP simplifies cross-platform development, maintains performance, and provides a familiar C++ experience for GPU programming that runs seamlessly on both AMD and NVIDIA GPUs. 
