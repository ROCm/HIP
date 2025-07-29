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

     * `System requirements (Linux) <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_
     * `System requirements (Windows) <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html>`_

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

     HIP is automatically installed during the ROCm installation. If you haven't
     yet installed ROCm, you can find installation instructions here:

     * `ROCm installation for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html>`_
     * `HIP SDK installation for Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/index.html>`_

     By default, HIP is installed into ``/opt/rocm``.

     .. note::
     
        There is no autodetection for the HIP installation. If you choose to 
        install it somewhere other than the default location, you must set the
        ``HIP_PATH`` environment variable as explained in
        `Build HIP from source <./build.html>`_.

  .. tab-item:: NVIDIA
     :sync: nvidia

     #. Install the NVIDIA toolkit.

        The latest release can be found here:
        `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_.

     #. Setup the radeon repo.

        .. code-block::shell

          # Replace url with appropriate link in the table below
          wget https://repo.radeon.com/amdgpu-install/6.2/distro/version_name/amdgpu-install_6.2.60200-1_all.deb
          sudo apt install ./amdgpu-install_6.2.60200-1_all.deb
          sudo apt update

        .. list-table:: amdgpu-install links
           :widths: 25 100
           :header-rows: 1

           * - Ubuntu version
             - URL
           * - 24.04
             - https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/noble/amdgpu-install_6.2.60204-1_all.deb
           * - 22.04
             - https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb

     #. Install the ``hip-runtime-nvidia`` and ``hip-dev`` packages. This installs the CUDA SDK and HIP
        porting layer.

        .. code-block:: shell

          apt-get install hip-runtime-nvidia hip-dev

        The default paths are:
          * CUDA SDK: ``/usr/local/cuda``
          * HIP: ``/opt/rocm``

     #. Set the HIP_PLATFORM to nvidia.

        .. code-block:: shell

          export HIP_PLATFORM="nvidia"

Verify your installation
==========================================================

Run ``hipconfig`` in your installation path.

.. code-block:: shell

  /opt/rocm/bin/hipconfig --full
