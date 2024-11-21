*******************************************
Install HIP
*******************************************

HIP can be installed on AMD (ROCm with HIP-Clang) and NVIDIA (CUDA with NVCC) platforms.

Note: The version definition for the HIP runtime is different from CUDA. On an AMD platform, the
``hipRuntimeGerVersion`` function returns the HIP runtime version; on an NVIDIA platform, this function
returns the CUDA runtime version.

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

      Check the system requirements in the
      `NVIDIA CUDA Installation Guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>`_.

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
            * HIP: ``/opt/rocm/hip``

      #. Set the HIP_PLATFORM to nvidia.

         .. code-block:: shell

            export HIP_PLATFORM="nvidia"

Verify your installation
==========================================================

Run ``hipconfig`` in your installation path.

.. code-block:: shell

   /opt/rocm/bin/hipconfig --full
