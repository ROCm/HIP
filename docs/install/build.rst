.. meta::
   :description: This page gives instructions on how to build HIP from source.
   :keywords: AMD, ROCm, HIP, build, build instructions, source

*******************************************
Build HIP from source
*******************************************

Prerequisites
=================================================

HIP code can be developed either on AMD ROCm platform using HIP-Clang compiler,
or a CUDA platform with ``nvcc`` installed. Before building and running HIP,
make sure drivers and prebuilt packages are installed properly on the platform.

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

In the ROCM 7.1 release, HIP is integrated into the core ROCm projects resides in the ``rocm-systems`` monorepository.
In addition, the following components are also part of the monrepository:

* ``clr``, AMD's Compute Language Runtime, includes ROCclr, HIPAMD and OpenCl.
* ``hipother``, provides files required to support the HIP back-end implementation on some non-AMD platforms, like NVIDIA.
* ``hip-tests``, the HIP testing suite. 

Set the repository branch using the variable: ``ROCM_BRANCH``. For example, for ROCM 7.1, use:

.. code-block:: shell

   export ROCM_BRANCH=release/rocm-rel-7.1

.. tab-set::

  .. tab-item:: AMD
     :sync: amd

     #. Get HIP source code.

        .. code-block:: shell

           git clone -b "$ROCM_BRANCH" git@github.com:ROCm/rocm-systems.git   

     #. Set the environment variables.

        .. code-block:: shell

           export CLR_DIR="$(readlink -f rocm-systems/projects/clr)"
           export HIP_DIR="$(readlink -f rocm-systems/projects/hip)"

     #. Build HIP.

        .. code-block:: shell

           cd "$CLR_DIR"
           mkdir -p build; cd build
           cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=amd -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF ..
           make -j$(nproc)
           sudo make install

        .. note::

           If ``CMAKE_INSTALL_PREFIX`` is not explicitly specified, the HIP runtime will be installed at
           ``<ROCM_PATH>``, which is by default at the path ``/opt/rocm``.

           By default, the release version of HIP is built. If you need a debug version, you can put the option ``CMAKE_BUILD_TYPE=Debug`` in the command line.

        Default paths and environment variables:
         
        * HIP is installed into ``<ROCM_PATH>``. This can be overridden by setting the ``INSTALL_PREFIX`` as the command option.

        * HSA is in ``<ROCM_PATH>``. This can be overridden by setting the ``HSA_PATH``
          environment variable.

        * Clang is in ``<ROCM_PATH>/llvm/bin``. This can be overridden by setting the
          ``HIP_CLANG_PATH`` environment variable.

        * The device library is in ``<ROCM_PATH>/lib``. This can be overridden by setting the
          ``DEVICE_LIB_PATH`` environment variable.

        * Optionally, you can add ``<ROCM_PATH>/bin`` to your ``PATH``, which can make it easier to
          use the tools.

        * Optionally, you can set ``HIPCC_VERBOSE=7`` to output the command line for compilation.

        After you run the ``make install`` command, HIP is installed to ``<ROCM_PATH>`` by default, or ``$PWD/install/hip`` while ``INSTALL_PREFIX`` is defined.

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

           git clone -b "$ROCM_BRANCH" git@github.com:ROCm/rocm-systems.git

     #. Set the environment variables.

        .. code-block:: shell

           export CLR_DIR="$(readlink -f rocm-systems/projects/clr)"
           export HIP_DIR="$(readlink -f rocm-systems/projects/hip)"
           export HIP_OTHER="$(readlink -f rocm-systems/projects/hipother)"

     #. Build HIP.

        .. code-block:: shell

           cd "$CLR_DIR"
           mkdir -p build; cd build
           cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=nvidia -DCMAKE_INSTALL_PREFIX=$PWD/install -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIPNV_DIR=$HIP_OTHER/hipnv ..
           make -j$(nproc)
           sudo make install

Build HIP tests
=================================================

.. tab-set::

  .. tab-item:: AMD
     :sync: amd

     **Build HIP catch tests.**

     HIP catch tests utilize the Catch2 testing framework.

     #. Get HIP tests source code.

        .. code-block:: shell

           git clone -b "$ROCM_BRANCH" git@github.com:ROCm/rocm-systems.git
           export HIPTESTS_DIR="$(readlink -f rocm-systems/projects/hip-tests)"

     #. Build HIP tests from source.

        .. code-block:: shell

           cd "$HIPTESTS_DIR"
           mkdir -p build; cd build
           cmake ../catch -DHIP_PLATFORM=amd -DHIP_PATH=$CLR_DIR/build/install  # or any path where HIP is installed; for example: ``/opt/rocm``
           export ROCM_PATH=/opt/rocm
           make build_tests
           ctest # run tests

        HIP catch tests are built in ``$HIPTESTS_DIR/build``.

        To run any single catch test, use this example:

        .. code-block:: shell

           cd $HIPTESTS_DIR/build/catch_tests/unit/texture
           ./TextureTest

     #. Build a HIP Catch2 standalone test.

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
Simple examples can be found in the `ROCm-examples repository <https://github.com/ROCm/rocm-examples>`_.
