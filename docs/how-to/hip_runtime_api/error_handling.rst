.. meta::
   :description: Error Handling
   :keywords: AMD, ROCm, HIP, error handling, error

*************************************************************************
Error handling
*************************************************************************

Error handling is crucial for several reasons. It enhances the stability of applications by preventing crashes and maintaining a consistent state. It also improves security by protecting against vulnerabilities that could be exploited by malicious actors. Additionally, effective error handling enhances the user experience by providing meaningful feedback and ensuring that applications can recover gracefully from errors. Finally, it aids maintainability by making it easier for developers to diagnose and fix issues.

Strategies
==========

One of the fundamental best practices in error handling is to develop a consistent strategy across the entire application. This involves defining how errors are reported, logged, and managed. This can be achieved by using a centralized error handling mechanism that ensures consistency and reduces redundancy. For instance, using macros to simplify error checking and reduce code duplication is a common practice. A macro like ``HIP_CHECK`` can be defined to check the return value of HIP API calls and handle errors appropriately.

Granular error reporting
------------------------
It involves reporting errors at the appropriate level of detail. Too much detail can overwhelm users, while too little can make debugging difficult. Differentiating between user-facing errors and internal errors is crucial. 

Fail-fast principle
-------------------
It involves detecting and handling errors as early as possible to prevent them from propagating and causing more significant issues. Such as validating inputs and preconditions before performing operations.

Resource management 
-------------------
Ensuring that resources such as memory, file handles, and network connections are properly managed and released in the event of an error is essential.

Integration in Error Handling
=============================
Functions like ``hipGetLastError`` and ``hipPeekAtLastError`` are used to detect errors after HIP API calls. This ensures that any issues are caught early in the execution flow.

For reporting, ``hipGetErrorName`` and ``hipGetErrorString`` provide meaningful error messages that can be logged or displayed to users. This helps in understanding the nature of the error and facilitates debugging.

By checking for errors and providing detailed information, these functions enable developers to implement appropriate error handling strategies, such as retry mechanisms, resource cleanup, or graceful degradation.

Examples
--------

``hipGetLastError`` returns the last error that occurred during a HIP runtime API call and resets the error code to ``hipSuccess``:

    .. code-block:: cpp

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            printf("HIP Error: %s\n", hipGetErrorString(err));
        }

``hipPeekAtLastError`` returns the last error that occurred during a HIP runtime API call **without** resetting the error code:

    .. code-block:: cpp

        hipError_t err = hipPeekAtLastError();
        if (err != hipSuccess)
        {
            printf("HIP Error: %s\n", hipGetErrorString(err));
        }

``hipGetErrorName`` converts a HIP error code to a string representing the error name:

    .. code-block:: cpp

        const char* errName = hipGetErrorName(err);
        printf("Error Name: %s\n", errName);

``hipGetErrorString`` converts a HIP error code to a string describing the error:

    .. code-block:: cpp

        const char* errString = hipGetErrorString(err);
        printf("Error Description: %s\n", errString);

Best Practices
==============

1. Check Errors After Each API Call

   Always check the return value of HIP API calls to catch errors early. For example:

     .. code-block:: cpp

        hipError_t err = hipMalloc(&d_A, size);
        if (err != hipSuccess) {
            printf("hipMalloc failed: %s\n", hipGetErrorString(err));
            return -1;
        }

2. Use Macros for Error Checking

   Define macros to simplify error checking and reduce code duplication. For example:

     .. code-block:: cpp

        #define HIP_CHECK(call) \
        { \
            hipError_t err = call; \
            if (err != hipSuccess) { \
                printf("HIP Error: %s:%d, %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
                exit(err); \
            } \
        }

        // Usage
        HIP_CHECK(hipMalloc(&d_A, size));

3. Handle Errors Gracefully

   Ensure the application can handle errors gracefully, such as by freeing resources or providing meaningful error messages to the user.

Example
-------

A complete example demonstrating error handling:

    .. code-block:: cpp

        #include <stdio.h>
        #include <hip/hip_runtime.h>

        #define HIP_CHECK(call) \
        { \
            hipError_t err = call; \
            if (err != hipSuccess) { \
                printf("HIP Error: %s:%d, %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
                exit(err); \
            } \
        }

        int main()
        {
            constexpr int N    = 100;
            size_t        size = N * sizeof(float);
            float        *d_A;

            // Allocate memory on the device
            HIP_CHECK(hipMalloc(&d_A, size));

            // Perform other operations...

            // Free device memory
            HIP_CHECK(hipFree(d_A));

            return 0;
        }
