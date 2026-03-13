.. meta::
  :description: This chapter provides an introduction to the HIP API.
  :keywords: AMD, ROCm, HIP, CUDA, C++ language extensions

.. _intro-to-hip:

*******************************************************************************
What is HIP?
*******************************************************************************

The HIP API is a C++ runtime API
and kernel language that lets developers write performant code for AMD GPUs. 

C++ runtime API
-----------------------------------------------

The HIP runtime implements HIP streams, events, and memory APIs, and is an object library that
is linked with the application. The source code for all headers and the library
implementation is available on GitHub.

For further details, check :ref:`HIP Runtime API Reference <runtime_api_reference>`.

Kernel language
-----------------------------------------------

HIP provides a C++ syntax that is suitable for compiling most code that commonly appears in
compute kernels (classes, namespaces, operator overloading, and templates). HIP also defines other
language features that are designed to target accelerators, such as:

* Short-vector headers that can serve on a host or device
* Math functions that resemble those in ``math.h``, which is included with standard C++ compilers
* Built-in functions for accessing specific GPU hardware capabilities

For further details, check :doc:`HIP C++ language extensions <how-to/hip_cpp_language_extensions>`
and :doc:`Kernel language C++ support <how-to/kernel_language_cpp_support>`.
