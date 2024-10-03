.. meta::
   :description: Initialization.
   :keywords: AMD, ROCm, HIP, initialization

*************************************************************************
Initialization
*************************************************************************

Initialization involves setting up the environment and resources needed for GPU computation.

Include HIP headers
===================

To use HIP functions, include the HIP runtime header in your source file:

.. code-block:: cpp

    #include <hip/hip_runtime.h>

Initialize the HIP Runtime
==========================

The HIP runtime is initialized automatically when the first HIP API call is made. However, you can explicitly initialize it using ``hipInit``:

.. code-block:: cpp

    hipError_t err = hipInit(0);
    if (err != hipSuccess)
    {
        // Handle error
    }

The initialization includes the following steps:

- Loading the HIP Runtime

  This includes loading necessary libraries and setting up internal data structures.

- Querying GPU Devices

  Identifying and querying the available GPU devices on the system.

- Setting Up Contexts

  Creating contexts for each GPU device, which are essential for managing resources and executing kernels.

Get device properties
=====================

Before using a GPU device, you might want to query its properties:

.. code-block:: cpp

    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i)
    {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
    }

Set device
==========

Select the GPU device to be used for subsequent HIP operations:

.. code-block:: cpp

    int deviceId = 0; // Example: selecting the first device
    hipSetDevice(deviceId);

This function performs several key tasks:

- Context Binding

  Binds the current thread to the context of the specified GPU device. This ensures that all subsequent operations are executed on the selected device.

- Resource Allocation

  Prepares the device for resource allocation, such as memory allocation and stream creation.

- Error Handling

  Checks for errors in device selection and ensures that the specified device is available and capable of executing HIP operations.
