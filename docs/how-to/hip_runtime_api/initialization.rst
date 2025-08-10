.. meta::
   :description: Initialization.
   :keywords: AMD, ROCm, HIP, initialization

.. _initialization:

********************************************************************************
Initialization
********************************************************************************

The initialization involves setting up the environment and resources needed for
using GPUs. The following steps are covered with the initialization:

- Setting up the HIP runtime

  This includes reading the environment variables set during init, setting up
  the active or visible devices, loading necessary libraries, setting up
  internal buffers for memory copies or cooperative launches, initialize the
  compiler as well as HSA runtime and checks any errors due to lack of resources
  or no active devices.

- Querying and setting GPUs

  Identifying and querying the available GPU devices on the system.

- Setting up contexts

  Creating contexts for each GPU device, which are essential for managing
  resources and executing kernels. For further details, check the :ref:`context
  section <context_driver_api>`.

Initialize the HIP runtime
================================================================================

The HIP runtime is initialized automatically when the first HIP API call is
made. However, you can explicitly initialize it using :cpp:func:`hipInit`,
to be able to control the timing of the initialization. The manual
initialization can be useful to ensure that the GPU is initialized and
ready, or to isolate GPU initialization time from other parts of
your program.

.. note::

  You can use :cpp:func:`hipDeviceReset` to delete all streams created, memory
  allocated, kernels running and events created by the current process. Any new
  HIP API call initializes the HIP runtime again.

Querying and setting GPUs
================================================================================

If multiple GPUs are available in the system, you can query and select the
desired GPU(s) to use based on device properties, such as size of global memory,
size shared memory per block, support of cooperative launch and support of
managed memory.

Querying GPUs
--------------------------------------------------------------------------------

The properties of a GPU can be queried using :cpp:func:`hipGetDeviceProperties`,
which returns a struct of :cpp:struct:`hipDeviceProp_t`. The properties in the
struct can be used to identify a device or give an overview of hardware
characteristics, that might make one GPU better suited for the task than others.

The :cpp:func:`hipGetDeviceCount` function returns the number of available GPUs,
which can be used to loop over the available GPUs.

Example code of querying GPUs:

.. code-block:: cpp

  #include <hip/hip_runtime.h>
  #include <iostream>

  int main() {

      int deviceCount;
      if (hipGetDeviceCount(&deviceCount) == hipSuccess){
          for (int i = 0; i < deviceCount; ++i){
              hipDeviceProp_t prop;
              if ( hipGetDeviceProperties(&prop, i) == hipSuccess)
                  std::cout << "Device" << i << prop.name << std::endl;
          }
      }

      return 0;
  }

Setting the GPU
--------------------------------------------------------------------------------

:cpp:func:`hipSetDevice` function select the GPU to be used for subsequent HIP
operations. This function performs several key tasks:

- Context Binding

  Binds the current thread to the context of the specified GPU device. This
  ensures that all subsequent operations are executed on the selected device.

- Resource Allocation

  Prepares the device for resource allocation, such as memory allocation and
  stream creation.

- Check device availability

  Checks for errors in device selection and returns error if the specified
  device is not available or not capable of executing HIP operations.
