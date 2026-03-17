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

Refer to the Prerequisites section in the ROCm install guides:

* `System requirements (Linux) <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_
* `System requirements (Windows) <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html>`_

Installation
=======================================

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

Verify your installation
==========================================================

Run ``hipconfig`` in your installation path.

.. code-block:: shell

  /opt/rocm/bin/hipconfig --full
