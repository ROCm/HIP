.. meta::
    :description: This chapter describes how to use multiple devices on one host.
    :keywords: ROCm, HIP, multi-device, multiple, GPUs, devices

.. _multi-device:

*******************************************************************************
Multi-device management
*******************************************************************************

Device enumeration
==================

Device enumeration involves identifying all the available GPUs connected to the host system. A single host machine can have multiple GPUs, each with its own unique identifier. By listing these devices, you can decide which GPU to use for computation. The host queries the system to count and list all connected GPUs, ensuring that the application can leverage the full computational power available.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <iostream>

    int main()
    {
        int deviceCount;
        hipGetDeviceCount(&deviceCount);
        std::cout << "Number of devices: " << deviceCount << std::endl;

        for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
        {
            hipDeviceProp_t deviceProp;
            hipGetDeviceProperties(&deviceProp, deviceId);
            std::cout << "Device " << deviceId << std::endl << " Properties:" << std::endl;
            std::cout << "  Name: " << deviceProp.name << std::endl;
            std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MiB" << std::endl;
            std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KiB" << std::endl;
            std::cout << "  Registers per Block: " << deviceProp.regsPerBlock << std::endl;
            std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
            std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
            std::cout << "  Max Threads per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
            std::cout << "  Number of Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
            std::cout << "  Max Threads Dimensions: [" 
                    << deviceProp.maxThreadsDim[0] << ", "
                    << deviceProp.maxThreadsDim[1] << ", "
                    << deviceProp.maxThreadsDim[2] << "]" << std::endl;
            std::cout << "  Max Grid Size: [" 
                    << deviceProp.maxGridSize[0] << ", "
                    << deviceProp.maxGridSize[1] << ", "
                    << deviceProp.maxGridSize[2] << "]" << std::endl;
            std::cout << std::endl;
        }

        return 0;
    }

.. _multi-device_selection:

Device selection
================

Once you have enumerated the available GPUs, the next step is to select a specific device for computation. This involves setting the active GPU that will execute subsequent operations. This step is crucial in multi-GPU systems where different GPUs might have different capabilities or workloads. By selecting the appropriate device, you ensure that the computational tasks are directed to the correct GPU, optimizing performance and resource utilization.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <iostream>

    #define HIP_CHECK(expression)                \
    {                                            \
        const hipError_t status = expression;    \
        if (status != hipSuccess) {              \
            std::cerr << "HIP error " << status  \
                    << ": " << hipGetErrorString(status) \
                    << " at " << __FILE__ << ":" \
                    << __LINE__ << std::endl;  \
            exit(status);                        \
        }                                        \
    }

    __global__ void simpleKernel(double *data)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        data[idx] = idx * 2.0;
    }

    int main()
    {
        double* deviceData0;
        double* deviceData1;
        size_t  size = 1024 * sizeof(*deviceData0);

        int deviceId0 = 0;
        int deviceId1 = 1;

        // Set device 0 and perform operations
        HIP_CHECK(hipSetDevice(deviceId0)); // Set device 0 as current
        HIP_CHECK(hipMalloc(&deviceData0, size)); // Allocate memory on device 0
        simpleKernel<<<1000, 128>>>(deviceData0); // Launch kernel on device 0
        HIP_CHECK(hipDeviceSynchronize());

        // Set device 1 and perform operations
        HIP_CHECK(hipSetDevice(deviceId1)); // Set device 1 as current
        HIP_CHECK(hipMalloc(&deviceData1, size)); // Allocate memory on device 1
        simpleKernel<<<1000, 128>>>(deviceData1); // Launch kernel on device 1
        HIP_CHECK(hipDeviceSynchronize());

        // Attempt to use deviceData0 on device 1 (This will not work as deviceData0 is allocated on device 0)
        HIP_CHECK(hipSetDevice(deviceId1));
        hipError_t err = hipMemcpy(deviceData1, deviceData0, size, hipMemcpyDeviceToDevice); // This should fail
        if (err != hipSuccess)
        {
            std::cout << "Error: Cannot access deviceData0 from device 1, deviceData0 is on device 0" << std::endl;
        }

        // Copy result from device 0
        double hostData0[1024];
        HIP_CHECK(hipSetDevice(deviceId0));
        HIP_CHECK(hipMemcpy(hostData0, deviceData0, size, hipMemcpyDeviceToHost));

        // Copy result from device 1
        double hostData1[1024];
        HIP_CHECK(hipSetDevice(deviceId1));
        HIP_CHECK(hipMemcpy(hostData1, deviceData1, size, hipMemcpyDeviceToHost));

        // Display results from both devices
        std::cout << "Device 0 data: " << hostData0[0] << std::endl;
        std::cout << "Device 1 data: " << hostData1[0] << std::endl;

        // Free device memory
        HIP_CHECK(hipFree(deviceData0));
        HIP_CHECK(hipFree(deviceData1));

        return 0;
    }


Stream and event behavior
=========================
Streams are used to manage the execution order of kernels and memory operations on a GPU. Creating streams allows multiple operations to execute concurrently, improving application efficiency. Events, used with streams, track the progress of operations and synchronize different tasks. This coordination ensures that operations complete in the desired order.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <iostream>

    __global__ void simpleKernel(double *data)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        data[idx] = idx * 2.0;
    }

    int main()
    {
        double* deviceData;
        size_t  size = 1024 * sizeof(*deviceData);

        // Create a stream and events
        hipStream_t stream;
        hipEvent_t startEvent, stopEvent;

        hipStreamCreate(&stream);
        hipEventCreate(&startEvent);
        hipEventCreate(&stopEvent);

        // Allocate memory on device
        hipMalloc(&deviceData, size);

        // Record the start event
        hipEventRecord(startEvent, stream);

        // Launch the kernel asynchronously
        simpleKernel<<<1000, 128, 0, stream>>>(deviceData);

        // Record the stop event
        hipEventRecord(stopEvent, stream);

        // Wait for the stop event to complete
        hipEventSynchronize(stopEvent);

        // Calculate elapsed time between the events
        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, startEvent, stopEvent);
        std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

        // Cleanup
        hipEventDestroy(startEvent);
        hipEventDestroy(stopEvent);
        hipStreamSynchronize(stream);
        hipStreamDestroy(stream);
        hipFree(deviceData);

        return 0;
    }

Peer-to-peer memory access
==========================
In multi-GPU systems, peer-to-peer memory access enables one GPU to directly read or write to the memory of another GPU. This capability reduces data transfer times by allowing GPUs to communicate directly without involving the host. Enabling peer-to-peer access can significantly improve the performance of applications that require frequent data exchange between GPUs, as it eliminates the need to transfer data through the host memory.

By adding peer-to-peer access to the example referenced in :ref:`multi_device_selection`, data can be copied between devices:

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <iostream>

    __global__ void simpleKernel(double *data)
    {
        int idx   = blockIdx.x * blockDim.x + threadIdx.x;
        data[idx] = idx * 2.0;
    }

    int main()
    {
        double* deviceData0;
        double* deviceData1;
        size_t  size = 1024 * sizeof(*deviceData0);

        int deviceId0 = 0;
        int deviceId1 = 1;

        // Enable peer access
        hipSetDevice(deviceId0);
        hipDeviceEnablePeerAccess(deviceId1, 0);

        hipSetDevice(deviceId1);
        hipDeviceEnablePeerAccess(deviceId0, 0);

        // Set device 0 and perform operations
        hipSetDevice(deviceId0); // Set device 0 as current
        hipMalloc(&deviceData0, size); // Allocate memory on device 0
        simpleKernel<<<1000, 128>>>(deviceData0); // Launch kernel on device 0

        // Set device 1 and perform operations
        hipSetDevice(deviceId1); // Set device 1 as current
        hipMalloc(&deviceData1, size); // Allocate memory on device 1
        simpleKernel<<<1000, 128>>>(deviceData1); // Launch kernel on device 1

        // Use peer-to-peer access
        hipSetDevice(deviceId0);
        
        // Now device 0 can access memory allocated on device 1
        hipMemcpy(deviceData0, deviceData1, size, hipMemcpyDeviceToDevice);

        // Copy result from device 0
        double hostData0[1024];
        hipSetDevice(deviceId0);
        hipMemcpy(hostData0, deviceData0, size, hipMemcpyDeviceToHost);

        // Copy result from device 1
        double hostData1[1024];
        hipSetDevice(deviceId1);
        hipMemcpy(hostData1, deviceData1, size, hipMemcpyDeviceToHost);

        // Display results from both devices
        std::cout << "Device 0 data: " << hostData0[0] << std::endl;
        std::cout << "Device 1 data: " << hostData1[0] << std::endl;

        // Free device memory
        hipFree(deviceData0);
        hipFree(deviceData1);

        return 0;
    }
