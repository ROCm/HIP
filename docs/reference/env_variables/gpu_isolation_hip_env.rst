Restricting the access of applications to a subset of GPUs, also known as GPU
isolation, allows users to hide GPU resources from programs. The GPU isolation
environment variables in HIP are collected in the following table.

.. _hip-env-isolation:
.. list-table::
    :header-rows: 1
    :widths: 50,30,20

    * - **Environment variable**
      - **Links**
      - **Value**

    * - | ``ROCR_VISIBLE_DEVICES``
        | A list of device indices or UUIDs that will be exposed to applications.
      - :doc:`GPU isolation <rocm:conceptual/gpu-isolation>`, :doc:`Setting the number of compute units <rocm:how-to/setting-cus>`
      - Example: ``0,GPU-DEADBEEFDEADBEEF``

    * - | ``GPU_DEVICE_ORDINAL``
        | Devices indices exposed to OpenCL and HIP applications.
      - :doc:`GPU isolation <rocm:conceptual/gpu-isolation>`
      - Example: ``0,2``

    * - | ``HIP_VISIBLE_DEVICES`` or ``CUDA_VISIBLE_DEVICES``
        | Device indices exposed to HIP applications.
      - :doc:`GPU isolation <rocm:conceptual/gpu-isolation>`, :doc:`HIP debugging <hip:how-to/debugging>`
      - Example: ``0,2``
