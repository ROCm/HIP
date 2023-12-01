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

         * :doc:`rocm:/install/linux/install`
         * :doc:`rocm:/install/windows/install`

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

         * :doc:`rocm:/install/linux/install`
         * :doc:`rocm:/install/windows/install`

By default, HIP is installed into /opt/rocm/hip.

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

            apt-get install hip-runtime-nvidia hip-dev

         The default paths are:
            * CUDA SDK: ``/usr/local/cuda``
            * HIP: ``/opt/rocm/hip``

         You can optionally add ``/opt/rocm/bin`` to your path, which can make it easier to use the tools.

Verify your installation
==========================================================

Run ``hipconfig`` in your installation path.

.. code:: shell

   /opt/rocm/bin/hipconfig --full

Build HIP from source
==========================================================

Set the repository branch using the variable: ``ROCM_BRANCH``. For example, for ROCm5.6, use:

.. code:: shell

   export ROCM_BRANCH=rocm-5.6.x

.. tab-set::

   .. tab-item:: AMD
      :sync: amd

      #. Get HIP source code.

         .. code:: shell

            git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/clr.git
            git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/hip.git
            git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/HIPCC.git

      #. Set the environment variables.

         .. code:: shell

            export CLR_DIR="$(readlink -f clr)"
            export HIP_DIR="$(readlink -f hip)"
            export HIPCC_DIR="$(readlink -f hipcc)"

         .. note::
            Starting in ROCM 5.6, CLR is a new repository that includes the former ROCclr, HIPAMD and
            OpenCl repositories. OpenCL provides headers that ROCclr runtime depends on.

      #. Build the HIPCC runtime.

         .. code:: shell

            cd "$HIPCC_DIR"
            mkdir -p build; cd build
            cmake ..
            make -j4

      #. Build HIP.

         .. code:: shell

            cd "$CLR_DIR"
            mkdir -p build; cd build
            cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=amd -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install -DHIPCC_BIN_DIR=$HIPCC_DIR/build -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF ..

            make -j$(nproc)
            sudo make install

         .. note::

            Note, if you don't specify ``CMAKE_INSTALL_PREFIX``, the HIP runtime is installed at
            ``<ROCM_PATH>/hip``. The default version of HIP is the latest release.

         Default paths and environment variables:

            * HIP is installed into ``<ROCM_PATH>/hip``. This can be overridden by setting the ``HIP_PATH``
               environment variable.
            * HSA is in ``<ROCM_PATH>/hsa``. This can be overridden by setting the ``HSA_PATH``
               environment variable.
            * Clang is in ``<ROCM_PATH>/llvm/bin``. This can be overridden by setting the
               ``HIP_CLANG_PATH`` environment variable.
            * The device library is in ``<ROCM_PATH>/lib``. This can be overridden by setting the
               ``DEVICE_LIB_PATH`` environment variable.
            * Optionally, you can add ``<ROCM_PATH>/bin`` to your ``PATH``, which can make it easier to
               use the tools.
            * Optionally, you can set ``HIPCC_VERBOSE=7`` to output the command line for compilation.

         After you run the ``make install`` command, make sure ``HIP_PATH`` points to ``$PWD/install/hip``.

         #. Generate a profiling header after adding/changing a HIP API.

            When you add or change a HIP API, you may need to generate a new ``hip_prof_str.h`` header.
            This header is used by ROCm tools to track HIP APIs, such as``rocprofiler`` and ``roctracer``.

            To generate the header after your change, use the ``hip_prof_gen.py`` tool located in
            ``hipamd/src``.

            Usage:

            .. code:: shell

               `hip_prof_gen.py [-v] <input HIP API .h file> <patched srcs path> <previous output> [<output>]`

            Flags:

               * ``-v``: Verbose messages
               * ``-r``: Process source directory recursively
               * ``-t``: API types matching check
               * ``--priv``: Private API check
               * ``-e``: On error exit mode
               * ``-p``: ``HIP_INIT_API`` macro patching mode

            Example usage:

            .. code:: shell

               hip_prof_gen.py -v -p -t --priv <hip>/include/hip/hip_runtime_api.h \
               <hipamd>/src <hipamd>/include/hip/amd_detail/hip_prof_str.h \
               <hipamd>/include/hip/amd_detail/hip_prof_str.h.new

   .. tab-item:: NVIDIA
      :sync: nvidia

      #. Get the HIP source code.

         .. code:: shell

            git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/hip.git
            git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/clr.git
            git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/HIPCC.git

      #. Set the environment variables.

         .. code:: shell

            export HIP_DIR="$(readlink -f hip)"
            export CLR_DIR="$(readlink -f hipamd)"
            export HIPCC_DIR="$(readlink -f hipcc)"

      #. Build the HIPCC runtime.

         .. code:: shell

            cd "$HIPCC_DIR"
            mkdir -p build; cd build
            cmake ..
            make -j4

      #. Build HIP.

         .. code:: shell

            cd "$CLR_DIR"
            mkdir -p build; cd build
            cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=nvidia -DCMAKE_INSTALL_PREFIX=$PWD/install -DHIPCC_BIN_DIR=$HIPCC_DIR/build -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF ..
            make -j$(nproc)
            sudo make install

Build HIP tests
=================================================

.. tab-set::

   .. tab-item:: AMD
      :sync: amd

      * Build HIP directed tests.

         .. code:: shell

            sudo make install
            make -j$(nproc) build_tests

         By default, all HIP directed tests are built and generated in
         ``$CLR_DIR/build/hipamd/directed_tests``.

         * Run all HIP ``directed_tests``.

            .. code:: shell

               ctest

            or

            .. code:: shell

               make test


         * Build and run a single directed test.

            .. code:: shell

               make directed_tests.texture.hipTexObjPitch
               cd $CLR_DIR/build/hipamd/directed_tests/texture
               ./hipTexObjPitch

         .. note::
            The integrated HIP directed tests will be deprecated in a future release.

      * Build HIP catch tests.

         HIP catch tests are separate from the HIP project and use Catch2.

         * Get HIP tests source code.

            .. code:: shell

               git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/hip-tests.git

         * Build HIP tests from source.

            .. code:: shell

               export HIPTESTS_DIR="$(readlink -f hip-tests)"
               cd "$HIPTESTS_DIR"
               mkdir -p build; cd build
               export HIP_PATH=$CLR_DIR/build/install  # or any path where HIP is installed; for example: ``/opt/rocm``
               cmake ../catch/ -DHIP_PLATFORM=amd
               make -j$(nproc) build_tests
               ctest # run tests

            HIP catch tests are built in ``$HIPTESTS_DIR/build``.

            To run any single catch test, use this example:

            .. code:: shell

               cd $HIPTESTS_DIR/build/catch_tests/unit/texture
               ./TextureTest

         * Build a HIP Catch2 standalone test.

            .. code:: shell

               cd "$HIPTESTS_DIR"
               hipcc $HIPTESTS_DIR/catch/unit/memory/hipPointerGetAttributes.cc \
               -I ./catch/include ./catch/hipTestMain/standalone_main.cc \
               -I ./catch/external/Catch2 -o hipPointerGetAttributes
               ./hipPointerGetAttributes
               ...

               All tests passed

   .. tab-item:: NVIDIA
      :sync: nvidia

      The commands to build HIP tests on an NVIDIA platform are the same as on an AMD platform.
      However, you must first set ``-DHIP_PLATFORM=nvidia``.

      * Run HIP. Compile and run the
      `square sample <https://github.com/ROCm-Developer-Tools/hip-tests/tree/rocm-5.5.x/samples/0_Intro/square>`_.
