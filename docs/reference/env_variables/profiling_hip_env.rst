The profiling environment variables in HIP are collected in the following table. For
more information, check :doc:`setting the number of CUs page <rocm:how-to/setting-cus>`.

.. _hip-env-prof:
.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HSA_CU_MASK``
        | Sets the mask on a lower level of queue creation in the driver, this mask will also be set for queues being profiled.
      - Example: ``1:0-8``

    * - | ``ROC_GLOBAL_CU_MASK``
        | Sets the mask on queues created by the HIP or the OpenCL runtimes, this mask will also be set for queues being profiled.
      - Example: ``0xf``, enables only 4 CUs

    * - | ``HIP_FORCE_QUEUE_PROFILING``
        | Used to run the app as if it were run in rocprof. Forces command queue profiling on by default.
      - | 0: Disable
        | 1: Enable
