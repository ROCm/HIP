.. meta::
   :description: HIP provides an external resource interoperability API that allows efficient data sharing between HIP's computing power and OpenGL's graphics rendering.
   :keywords: AMD, ROCm, HIP, external, interop, interoperability

********************************************************************************
External resource interoperability
********************************************************************************

This feature allows HIP to work with resources -- like memory and semaphores -- created by other APIs. This means resources can be used from APIs like CUDA, OpenCL and Vulkan within HIP, making it easier to integrate HIP into existing projects.

To use external resources in HIP, you typically follow these steps:

- import the resource: Use functions to import resources from other APIs
- use the resource: the imported resources can be used as if they were created by HIP itself
- destroy the resource: to clean up.

Semaphore Functions
===================

Semaphore functions are essential for synchronization in parallel computing. These functions facilitate communication and coordination between different parts of a program or between different programs. By managing semaphores, tasks are executed in the correct order, and resources are utilized effectively. Semaphore functions ensure smooth operation, preventing conflicts and maintaining the integrity of processes; upholding the integrity and performance of concurrent processes.

Semaphore functions are not supported recently on Linux (see: :doc:`../reference/hip_runtime_api/external_interop`.).

.. code-block::cpp

    #include <hip/hip_runtime.h>
    #include <hip/hip_runtime_api.h>
    #include <iostream>
    #include <fcntl.h>    // For O_* constants
    #include <sys/stat.h> // For mode constants
    #include <sys/mman.h> // For memfd_create
    #include <unistd.h>

    int main() {
        // Create an anonymous file using memfd_create for the semaphore
        int semFd = memfd_create("my_semaphore", MFD_CLOEXEC);
        if (semFd == -1) {
            std::cerr << "Failed to create anonymous file descriptor for semaphore" << std::endl;
            return -1;
        }

        hipExternalSemaphore_t extSem;
        hipExternalSemaphoreHandleDesc semHandleDesc = {};
        semHandleDesc.type = hipExternalSemaphoreHandleTypeOpaqueFd;
        semHandleDesc.handle.fd = semFd;
        semHandleDesc.flags = 0;

        // Import the external semaphore
        hipError_t result = hipImportExternalSemaphore(&extSem, &semHandleDesc);
        if (result != hipSuccess) {
            std::cerr << "Failed to import external semaphore: " << hipGetErrorString(result)<< std::endl;
            close(semFd);
            return -1;
        }

        hipExternalSemaphoreSignalParams signalParams = {};
        signalParams.flags = 0;
        signalParams.params.fence.value = 1;

        // Signal the external semaphore
        result = hipSignalExternalSemaphoresAsync(&extSem, &signalParams, 1, nullptr);
        if (result != hipSuccess) {
            std::cerr << "Failed to signal external semaphore: " << hipGetErrorString(result) << std::endl;
            hipDestroyExternalSemaphore(extSem);
            close(semFd);
            return -1;
        }

        hipExternalSemaphoreWaitParams waitParams = {};
        waitParams.flags = 0;
        waitParams.params.fence.value = 1;

        // Wait on the external semaphore
        result = hipWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, nullptr);
        if (result != hipSuccess) {
            std::cerr << "Failed to wait on external semaphore: " << hipGetErrorString(result) << std::endl;
            hipDestroyExternalSemaphore(extSem);
            close(semFd);
            return -1;
        }

        // Destroy the external semaphore
        result = hipDestroyExternalSemaphore(extSem);
        if (result != hipSuccess) {
            std::cerr << "Failed to destroy external semaphore: " << hipGetErrorString(result) << std::endl;
            close(semFd);
            return -1;
        }

        close(semFd);
        return 0;
    }


Memory Functions
================

Memory functions focus on the efficient sharing and management of memory resources. These functions allow the importation of memory created by external systems, enabling the program to use this memory seamlessly. Memory functions include mapping memory for effective use and ensuring proper cleanup to prevent resource leaks. This is critical for performance, particularly in applications handling large datasets or complex structures such as textures in graphics. Proper memory management ensures stability and efficient resource utilization.

.. code-block::cpp

    #include <hip/hip_runtime.h>
    #include <hip/hip_runtime_api.h>
    #include <iostream>
    #include <fcntl.h>    // For O_* constants
    #include <unistd.h>   // For close()
    #include <sys/stat.h> // For mode constants
    #include <sys/mman.h> // For shm_open

    int main() {
        // Create a shared memory object
        const char* shmName = "my_shared_memory";
        int shmFd = shm_open(shmName, O_CREAT | O_RDWR, 0644);
        if (shmFd == -1) {
            std::cerr << "Failed to create shared memory object" << std::endl;
            return -1;
        }

        // Set the size of the shared memory object
        const size_t memSize = 1024 * 1024; // 1MB
        if (ftruncate(shmFd, memSize) != 0) {
            std::cerr << "Failed to set size of shared memory object" << std::endl;
            close(shmFd);
            return -1;
        }

        hipExternalMemory_t extMem;
        hipExternalMemoryHandleDesc memHandleDesc = {};
        memHandleDesc.type = hipExternalMemoryHandleTypeOpaqueFd;
        memHandleDesc.handle.fd = shmFd;
        memHandleDesc.size = memSize;
        memHandleDesc.flags = 0;

        // Import the external memory
        hipError_t result = hipImportExternalMemory(&extMem, &memHandleDesc);
        if (result != hipSuccess) {
            std::cerr << "Failed to import external memory: " << hipGetErrorString(result) << std::endl;
            close(shmFd);
            return -1;
        }

        void* devPtr;
        hipExternalMemoryBufferDesc bufferDesc = {};
        bufferDesc.offset = 0;
        bufferDesc.size = memSize;
        bufferDesc.flags = 0;

        // Map a buffer onto the imported memory
        result = hipExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufferDesc);
        if (result != hipSuccess) {
            std::cerr << "Failed to map buffer onto external memory: " << hipGetErrorString(result) << std::endl;
            hipDestroyExternalMemory(extMem);
            close(shmFd);
            return -1;
        }

        // Use devPtr in HIP kernels

        // Destroy the external memory
        result = hipDestroyExternalMemory(extMem);
        if (result != hipSuccess) {
            std::cerr << "Failed to destroy external memory: " << hipGetErrorString(result) << std::endl;
            close(shmFd);
            return -1;
        }

        close(shmFd);
        return 0;
    }
