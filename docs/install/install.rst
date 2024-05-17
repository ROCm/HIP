*******************************************
Install HIP
*******************************************

HIP can be installed on AMD (ROCm with HIP-Clang) and NVIDIA (CUDA with nvcc) platforms.

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

   .. tab-item:: NVIDIA
      :sync: nvidia

      #. Install the NVIDIA driver.

         .. code:: shell

            sudo apt-get install ubuntu-drivers-common && sudo ubuntu-drivers autoinstall
            sudo reboot

         Alternatively, you can download the latest
         `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_.

      #. Install the ``hip-runtime-nvidia`` and ``hip-dev`` packages. This installs the CUDA SDK and HIP
         porting layer.

         .. code:: shell

            sudo apt-get install hip-runtime-nvidia hip-dev

         If the above packages are not found, `download the package signing key <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/ubuntu.html#package-signing-key>`_ and add the following repository. 

         .. code:: shell
            CODENAME=$(lsb_release -c | awk '{print $2}')
            echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/debian/ $CODENAME main" \
                | sudo tee --append /etc/apt/sources.list.d/rocm.list
            sudo apt update

         The default paths are:
            * CUDA SDK: ``/usr/local/cuda``
            * HIP: ``/opt/rocm/hip``

         You can optionally add ``/opt/rocm/bin`` to your path, which can make it easier to use the tools.

Verify your installation
==========================================================

Run ``hipconfig`` in your installation path.

.. code:: shell

   /opt/rocm/bin/hipconfig --full
