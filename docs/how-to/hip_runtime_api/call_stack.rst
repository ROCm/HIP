.. meta::
    :description: This page describes call stack concept in HIP
    :keywords: AMD, ROCm, HIP, call stack

*******************************************************************************
Call stack
*******************************************************************************

The call stack is a data structure for managing function calls, by saving the
state of the current function. Each time a function is called, a new call frame
is added to the top of the stack, containing information such as local
variables, return addresses and function parameters. When the function
execution completes, the frame is removed from the stack and loaded back into
the corresponding registers. This concept allows the program to return to the
calling function and continue execution from where it left off.

The call stack for each thread must track its function calls, local variables,
and return addresses. However, in GPU programming, the memory required to store
the call stack increases due to the parallelism inherent to the GPUs. NVIDIA
and AMD GPUs use different approaches. NVIDIA GPUs have the independent thread
scheduling feature where each thread has its own call stack and effective
program counter. On AMD GPUs threads are grouped; each warp has its own call
stack and program counter. Warps are described and explained in the
:ref:`inherent_thread_model`

If a thread or warp exceeds its stack size, a stack overflow occurs, causing
kernel failure. This can be detected using debuggers.

Call stack management with HIP
===============================================================================

You can adjust the call stack size as shown in the following example, allowing
fine-tuning based on specific kernel requirements. This helps prevent stack
overflow errors by ensuring sufficient stack memory is allocated.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <iostream>

    #define HIP_CHECK(expression)                \
    {                                            \
        const hipError_t status = expression;    \
        if(status != hipSuccess){                \
                std::cerr << "HIP error "        \
                    << status << ": "            \
                    << hipGetErrorString(status) \
                    << " at " << __FILE__ << ":" \
                    << __LINE__ << std::endl;    \
        }                                        \
    }

    int main()
    {
        size_t stackSize;
        HIP_CHECK(hipDeviceGetLimit(&stackSize, hipLimitStackSize));
        std::cout << "Default stack size: " << stackSize << " bytes" << std::endl;

        // Set a new stack size
        size_t newStackSize = 1024 * 8; // 8 KiB
        HIP_CHECK(hipDeviceSetLimit(hipLimitStackSize, newStackSize));

        HIP_CHECK(hipDeviceGetLimit(&stackSize, hipLimitStackSize));
        std::cout << "Updated stack size: " << stackSize << " bytes" << std::endl;

        return 0;
    }

Depending on the GPU model, at full occupancy, it can consume a significant
amount of memory. For instance, an MI300X with 304 compute units (CU) and up to
2048 threads per CU could use 304 · 2048 · 1024 bytes = 608 MiB for the call
stack by default.

Handling recursion and deep function calls
-------------------------------------------------------------------------------

Similar to CPU programming, recursive functions and deeply nested function
calls are supported. However, developers must ensure that these functions do
not exceed the available stack memory, considering the huge amount of memory
needed for the call stack due to the GPUs inherent parallelism. This can be
achieved by increasing stack size or optimizing code to reduce stack usage. To
detect stack overflow add proper error handling or use debugging tools.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <iostream>

    #define HIP_CHECK(expression)                \
    {                                            \
        const hipError_t status = expression;    \
        if(status != hipSuccess){                \
                std::cerr << "HIP error "        \
                    << status << ": "            \
                    << hipGetErrorString(status) \
                    << " at " << __FILE__ << ":" \
                    << __LINE__ << std::endl;    \
        }                                        \
    }

    __device__ unsigned long long fibonacci(unsigned long long n)
    {
        if (n == 0 || n == 1)
        {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    __global__ void kernel(unsigned long long n)
    {
        unsigned long long result = fibonacci(n);
        const size_t x = threadIdx.x + blockDim.x * blockIdx.x;

        if (x == 0)
            printf("%llu! = %llu \n", n, result);
    }

    int main()
    {
        kernel<<<1, 1>>>(10);
        HIP_CHECK(hipDeviceSynchronize());

        // With -O0 optimization option hit the stack limit
        // kernel<<<1, 256>>>(2048);
        // HIP_CHECK(hipDeviceSynchronize());

        return 0;
    }
