.. meta::
   :description: This page gives instructions on how to build HIP from source.
   :keywords: AMD, ROCm, HIP, build, build instructions, source

*******************************************
Build HIP from source
*******************************************

Prerequisites
=================================================

HIP code can be developed on AMD ROCm platform using HIP-Clang compiler.
Before building and running HIP, make sure drivers and prebuilt packages are
installed properly on the platform.

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

HIP is one of the core ROCm projects and resides in the rocm-systems monorepository.
In addition, the monorepository also includes the following components:

* ``clr``, AMD’s Compute Language Runtime, which contains ROCclr, HIPAMD, and OpenCL.
* ``hip-tests``, the HIP testing suite.

Beginning with the TheRock 7.13 release, the rocm-systems codebase is also integrated into the TheRock repository.
Developers may build HIP using one of two methods:

* From ``rocm-systems`` monorepository
* From the ROCm ``TheRock`` repository

This document provides instructions for building HIP from the ``rocm-systems`` monorepository.
For guidance on building HIP using TheRock, see the build documentation included with `TheRock <https://github.com/ROCm/TheRock/blob/main/README.md>`_.

#. Set the repository branch

   Set the branch using the variable: ``ROCM_BRANCH``. For example, for TheRock 7.13, use:

   .. code-block:: shell

      export ROCM_BRANCH=release/therock-7.13

#. Get HIP source code.

   .. code-block:: shell

      git clone -b "$ROCM_BRANCH" git@github.com:ROCm/rocm-systems.git

#. Set the environment variables.

   .. code-block:: shell

      export CLR_DIR="$(readlink -f rocm-systems/projects/clr)"
      export HIP_DIR="$(readlink -f rocm-systems/projects/hip)"
      export ROCM_PATH=/opt/rocm

#. Build HIP.

   .. code-block:: shell

      cd "$CLR_DIR"
      mkdir -p build && cd build
      cmake \
        -DHIP_COMMON_DIR="$HIP_DIR" \
        -DHIP_PLATFORM=amd \
        -DCMAKE_PREFIX_PATH="/opt/rocm/" \
        -DCMAKE_INSTALL_PREFIX="$PWD/install" \
        -DCLR_BUILD_HIP=ON \
        -DCLR_BUILD_OCL=OFF \
        ..
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

   * The device library is in ``<ROCM_PATH>/lib``. This can be overridden by setting the
     ``DEVICE_LIB_PATH`` environment variable.

   * Optionally, you can add ``<ROCM_PATH>/bin`` to your ``PATH``, which can make it easier to
     use the tools.

   After you run the ``make install`` command, HIP is installed to ``<ROCM_PATH>`` by default, or ``$PWD/install/hip`` while ``INSTALL_PREFIX`` is defined.

#. Generate a profiling header after adding/changing a HIP API.

   When you add or change a HIP API, you may need to generate a new ``hip_prof_str.h`` header.
   This header is used by ROCm tools to track HIP APIs, such as ``rocprofiler`` and ``roctracer``.

   To generate the header after your change, use the ``hip_prof_gen.py`` tool located in
   ``hipamd/src``.

   Usage:

   .. code-block:: shell

      hip_prof_gen.py [-v] <input HIP API .h file> <patched srcs path> <previous output> [<output>]

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

Build HIP tests
=================================================

**Build HIP catch tests.**

HIP catch tests are built using AMD's ``amdclang`` compiler.

#. Get HIP tests source code.

   .. code-block:: shell

      git clone -b "$ROCM_BRANCH" git@github.com:ROCm/rocm-systems.git
      export HIPTESTS_DIR="$(readlink -f rocm-systems/projects/hip-tests)"

#. Build HIP tests from source.

   .. code-block:: shell

      cd "$HIPTESTS_DIR"
      mkdir -p build; cd build
      export ROCM_PATH=/opt/rocm
      cmake ../catch \
        -DHIP_PLATFORM=amd \
        -DCMAKE_PREFIX_PATH=$CLR_DIR/build/install \
        -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ \
        -DCMAKE_C_COMPILER=$ROCM_PATH/bin/amdclang \
        -DCMAKE_HIP_COMPILER=$ROCM_PATH/bin/amdclang++ \
        -DOFFLOAD_ARCH_STR="--offload-arch=<selected-gpu-arch>"
      make build_tests
      ctest # run tests

   Note: The value of ``OFFLOAD_ARCH_STR`` should match the GPU architecture present on your system (e.g., gfx1200).
   You can determine the correct architecture by running the ``rocminfo`` command.

   HIP catch tests are built in ``$HIPTESTS_DIR/build``.

   To run any single catch test, use this example:

   .. code-block:: shell

      cd $HIPTESTS_DIR/build/catch_tests/unit/texture
      ./TextureTest

#. Build a HIP Catch standalone test.

   For detailed instructions on building the standalone Catch tests, consult the `hip-tests README.md at <https://github.com/ROCm/rocm-systems/tree/release/therock-7.13/projects/hip-tests/README.md>`_.

Run HIP
=================================================

After installation and building HIP, you can compile your application and run.
Simple examples can be found in the `ROCm-examples repository <https://github.com/ROCm/rocm-examples>`_.
