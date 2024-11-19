.. meta::
   :description: This page gives instructions on how to build HIP from source.
   :keywords: AMD, ROCm, HIP, build, build instructions, source

*******************************************
Build HIP from source
*******************************************

Prerequisites
=================================================

HIP code can be developed either on AMD ROCm platform using HIP-Clang compiler, or a CUDA platform with ``nvcc`` installed.
Before building and running HIP, make sure drivers and prebuilt packages are installed properly on the platform.

You also need to install Python 3, which includes the ``CppHeaderParser`` package.
Install Python 3 using the following command:

.. code-block:: shell

   apt-get install python3

Check and install ``CppHeaderParser`` package using the command:

.. code-block:: shell

   pip3 install CppHeaderParser

Install ``ROCm LLVM`` package using the command:

.. code-block:: shell

   apt-get install rocm-llvm-dev


.. _Building the HIP runtime:

Building the HIP runtime
==========================================================

Set the repository branch using the variable: ``ROCM_BRANCH``. For example, for ROCm 6.1, use:

.. code-block:: shell

   export ROCM_BRANCH=rocm-6.1.x

.. tab-set::

   .. tab-item:: AMD
      :sync: amd

      #. Get HIP source code.

         .. note::
            Starting in ROCM 5.6, CLR is a new repository that includes the former ROCclr, HIPAMD and
            OpenCl repositories. OpenCL provides headers that ROCclr runtime depends on.

         .. note::
            Starting in ROCM 6.1, a new repository ``hipother`` is added to ROCm, which is branched out from HIP.
            ``hipother`` provides files required to support the HIP back-end implementation on some non-AMD platforms,
            like NVIDIA.

         .. code-block:: shell

            git clone -b "$ROCM_BRANCH" https://github.com/ROCm/clr.git
            git clone -b "$ROCM_BRANCH" https://github.com/ROCm/hip.git

         CLR (Compute Language Runtime) repository includes ROCclr, HIPAMD and OpenCL.

         ROCclr (ROCm Compute Language Runtime) is a virtual device interface which
         is defined on the AMD platform. HIP runtime uses ROCclr to interact with different backends.

         HIPAMD provides implementation specifically for HIP on the AMD platform.

         OpenCL provides headers that ROCclr runtime currently depends on.
         hipother provides headers and implementation specifically for non-AMD HIP platforms, like NVIDIA.

      #. Set the environment variables.

         .. code-block:: shell

            export CLR_DIR="$(readlink -f clr)"
            export HIP_DIR="$(readlink -f hip)"


      #. Build HIP.

         .. code-block:: shell

            cd "$CLR_DIR"
            mkdir -p build; cd build
            cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=amd -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF ..

            make -j$(nproc)
            sudo make install

         .. note::

            Note, if you don't specify ``CMAKE_INSTALL_PREFIX``, the HIP runtime is installed at
            ``<ROCM_PATH>/hip``.

            By default, release version of HIP is built. If need debug version, you can put the option ``CMAKE_BUILD_TYPE=Debug`` in the command line.

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
            This header is used by ROCm tools to track HIP APIs, such as ``rocprofiler`` and ``roctracer``.

            To generate the header after your change, use the ``hip_prof_gen.py`` tool located in
            ``hipamd/src``.

            Usage:

            .. code-block:: shell

               `hip_prof_gen.py [-v] <input HIP API .h file> <patched srcs path> <previous output> [<output>]`

            Flags:

               * ``-v``: Verbose messages
               * ``-r``: Process source directory recursively
               * ``-t``: API types matching check
               * ``--priv``: Private API check
               * ``-e``: On error exit mode
               * ``-p``: ``HIP_INIT_API`` macro patching mode

            Example usage:

            .. code-block:: shell

               hip_prof_gen.py -v -p -t --priv <hip>/include/hip/hip_runtime_api.h \
               <hipamd>/src <hipamd>/include/hip/amd_detail/hip_prof_str.h \
               <hipamd>/include/hip/amd_detail/hip_prof_str.h.new

   .. tab-item:: NVIDIA
      :sync: nvidia

      #. Get the HIP source code.

         .. code-block:: shell

            git clone -b "$ROCM_BRANCH" https://github.com/ROCm/clr.git
            git clone -b "$ROCM_BRANCH" https://github.com/ROCm/hip.git
            git clone -b "$ROCM_BRANCH" https://github.com/ROCm/hipother.git

      #. Set the environment variables.

         .. code-block:: shell

            export CLR_DIR="$(readlink -f clr)"
            export HIP_DIR="$(readlink -f hip)"
            export HIP_OTHER="$(readlink -f hipother)"

      #. Build HIP.

         .. code-block:: shell

            cd "$CLR_DIR"
            mkdir -p build; cd build
            cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=nvidia -DCMAKE_INSTALL_PREFIX=$PWD/install -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIPNV_DIR=$HIP_OTHER/hipnv ..
            make -j$(nproc)
            sudo make install

Build HIP tests
=================================================

.. tab-set::

   .. tab-item:: AMD
      :sync: amd

      * Build HIP catch tests.

         HIP catch tests are separate from the HIP project and use Catch2.

         * Get HIP tests source code.

            .. code-block:: shell

               git clone -b "$ROCM_BRANCH" https://github.com/ROCm/hip-tests.git

         * Build HIP tests from source.

            .. code-block:: shell

               export HIPTESTS_DIR="$(readlink -f hip-tests)"
               cd "$HIPTESTS_DIR"
               mkdir -p build; cd build
               cmake ../catch -DHIP_PLATFORM=amd -DHIP_PATH=$CLR_DIR/build/install  # or any path where HIP is installed; for example: ``/opt/rocm``
               make build_tests
               ctest # run tests

            HIP catch tests are built in ``$HIPTESTS_DIR/build``.

            To run any single catch test, use this example:

            .. code-block:: shell

               cd $HIPTESTS_DIR/build/catch_tests/unit/texture
               ./TextureTest

         * Build a HIP Catch2 standalone test.

            .. code-block:: shell

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


Run HIP
=================================================

After installation and building HIP, you can compile your application and run.
A simple example is `square sample <https://github.com/ROCm/hip-tests/tree/develop/samples/0_Intro/square>`_.
