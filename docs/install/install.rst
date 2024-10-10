.. meta::
   :description: This page explains how to install HIP
   :keywords: AMD, ROCm, HIP, install, installation

*******************************************
Install HIP
*******************************************

HIP can be installed on AMD (ROCm with HIP-Clang) and NVIDIA (CUDA with NVCC) platforms.

.. note::
   The version definition for the HIP runtime is different from CUDA. On AMD
   platforms, the :cpp:func:`hipRuntimeGetVersion` function returns the HIP
   runtime version. On NVIDIA platforms, this function returns the CUDA runtime
   version.

.. _install_prerequisites:

Prerequisites
=======================================

.. tab-set::

   .. tab-item:: AMD
      :sync: amd

      Refer to the Prerequisites section in the ROCm install guides:

         * :doc:`rocm-install-on-linux:reference/system-requirements`
         * :doc:`rocm-install-on-windows:reference/system-requirements`

   .. tab-item:: NVIDIA
      :sync: nvidia

      With NVIDIA GPUs, HIP requires unified memory. All CUDA-enabled NVIDIA
      GPUs with compute capability 5.0 or later should be supported. For more
      information, see `NVIDIA's list of CUDA enabled GPUs <https://developer.nvidia.com/cuda-gpus>`_.

Installation
=======================================

.. tab-set::

   .. tab-item:: AMD
      :sync: amd

      HIP is automatically installed during the ROCm installation. If you haven't yet installed ROCm, you
      can find installation instructions here:

         * :doc:`rocm-install-on-linux:index`
         * :doc:`rocm-install-on-windows:index`

      By default, HIP is installed into ``/opt/rocm/hip``.

      .. note::
         There is no autodetection for the HIP installation. If you choose to install it somewhere other than the default location, you must set the ``HIP_PATH`` environment variable as explained in `Build HIP from source <./build.html>`_.

   .. tab-item:: NVIDIA
      :sync: nvidia

      #. Install the NVIDIA driver.

         .. code-block:: shell

            sudo apt-get install ubuntu-drivers-common && sudo ubuntu-drivers autoinstall
            sudo reboot

         Alternatively, you can download the latest
         `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_.

      #. Install the ``hip-runtime-nvidia`` and ``hip-dev`` packages. This installs the CUDA SDK and HIP
         porting layer.

         .. code-block:: shell

            apt-get install hip-runtime-nvidia hip-dev

         The default paths are:
            * CUDA SDK: ``/usr/local/cuda``
            * HIP: ``/opt/rocm/hip``

         You can optionally add ``/opt/rocm/bin`` to your path, which can make it easier to use the tools.

Verify your installation
==========================================================

Run ``hipconfig`` in your installation path.

.. code-block:: shell

   /opt/rocm/bin/hipconfig --full
