.. meta::
  :description: This chapter describes the HIP runtime API and shows
                how to use it.
  :keywords: AMD, ROCm, HIP, CUDA, HIP runtime API How to,

.. _hip_runtime_api_how-to:

********************************************************************************
HIP Runtime API
********************************************************************************

The HIP runtime API provides C and C++ functionality to manage GPUs, like event,
stream and memory management. On AMD platforms the HIP runtime uses the
:doc:`Common Language Runtime (CLR) <hip:understand/amd_clr>`, while on NVIDIA
platforms it is only a thin layer over the CUDA runtime or Driver API.

- **CLR** contains source code for AMD's compute language runtimes: ``HIP`` and
  ``OpenCL™``. CLR includes the implementation of the ``HIP`` on the AMD
  platform `hipamd <https://github.com/ROCm/clr/tree/develop/hipamd>`_ and the
  Radeon Open Compute Common Language Runtime (rocclr). rocclr is a virtual
  device interface, that enables the HIP runtime to interact with different
  backends such as :doc:`ROCr <rocr-runtime:index>` on Linux or PAL on Windows. CLR also include the
  implementation of `OpenCL runtime <https://github.com/ROCm/clr/tree/develop/opencl>`_.
- The **CUDA runtime** is built on top of the CUDA driver API, which is a C API
  with lower-level access to NVIDIA GPUs. For further information about the CUDA
  driver and runtime API and its relation to HIP check the :doc:`CUDA driver API porting guide<hip:how-to/hip_porting_driver_api>`.

The relation between the different runtimes and their backends is presented in
the following figure.

.. figure:: ../data/how-to/hip_runtime_api/runtimes.svg

.. note::

  The CUDA specific headers can be found in the `hipother repository <https://github.com/ROCm/hipother>`_.
