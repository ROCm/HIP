.. meta::
   :description: This page explains how to install HIP
   :keywords: AMD, ROCm, HIP, install, installation

*******************************************
Install HIP
*******************************************

HIP can be installed on AMD platforms using ROCm with HIP-Clang.

.. _install_prerequisites:

Prerequisites
=======================================

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./build`.

Install the ROCm Core SDK
=======================================

The HIP runtime is included with the ROCm Core SDK on Linux and Windows. For
the most complete installation, we recommend that developers use the
``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. note::

   By default, when installing using your Linux distribution's package manager,
   HIP is installed into ``/opt/rocm``.

   There is no autodetection for the HIP installation. If you choose to
   install it somewhere other than the default location, you must set the
   ``HIP_PATH`` environment variable as explained in
   :doc:`Build HIP from source <./build>`.

Install the ROCm runtime on Linux
=================================

Alternatively, if you want to install the ROCm runtime package without
additional ROCm libraries and tools, install the ``amdrocm-runtime`` package.
This includes the HIP, base ROCm packages, and the compiler ecosystem.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the ROCm runtime package that matches your desired ROCm version.
   Package names use the following format:

   .. code-block:: shell-session

      amdrocm-runtime<rocm_version>

   Where ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
   suffix to install the latest available version.

   For example, to install the latest ROCm runtime package release for
   supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-runtime

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-runtime

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-runtime

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including the HIP
runtime. See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.

Verify your installation
==========================================================

Run ``hipconfig`` in your installation path.

.. code-block:: shell

  /opt/rocm/bin/hipconfig --full
