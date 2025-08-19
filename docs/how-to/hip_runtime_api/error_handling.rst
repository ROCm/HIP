.. meta::
   :description: Error Handling
   :keywords: AMD, ROCm, HIP, error handling, error

.. _error_handling:

********************************************************************************
Error handling
********************************************************************************

HIP provides functionality to detect, report, and manage errors that occur
during the execution of HIP runtime functions or when launching kernels. Every
HIP runtime function, apart from launching kernels, has :cpp:type:`hipError_t`
as return type. :cpp:func:`hipGetLastError` and :cpp:func:`hipPeekAtLastError`
can be used for catching errors from kernel launches, as kernel launches don't
return an error directly. HIP maintains an internal state, that includes the
last error code. :cpp:func:`hipGetLastError` returns and resets that error to
``hipSuccess``, while :cpp:func:`hipPeekAtLastError` just returns the error
without changing it. To get a human readable version of the errors,
:cpp:func:`hipGetErrorString` and :cpp:func:`hipGetErrorName` can be used.

.. note::

    :cpp:func:`hipGetLastError` returns the last actual HIP API error caught in the current thread
    during the application execution. Prior to ROCm 7.0, ``hipGetLastError`` might also return
    ``hipSuccess`` or ``hipErrorNotReady`` from the last HIP runtime API call, which are not errors.


Best practices of HIP error handling:

1. Check errors after each API call - Avoid error propagation.
2. Use macros for error checking - Check :ref:`hip_check_macros`.
3. Handle errors gracefully - Free resources and provide meaningful error
   messages to the user.

For more details on the error handling functions, see :ref:`error handling
functions reference page <error_handling_reference>`.

For a list of all error codes, see :ref:`HIP error codes <hip_error_codes>`.

.. _hip_check_macros:

HIP check macros
================================================================================

HIP uses check macros to simplify error checking and reduce code duplication.
The ``HIP_CHECK`` macros are mainly used to detect and report errors. It can
also exit from application with ``exit(1);`` function call after the error
print. The ``HIP_CHECK`` macro example:

.. code-block:: cpp

  #define HIP_CHECK(expression)                  \
  {                                              \
      const hipError_t status = expression;      \
      if(status != hipSuccess){                  \
          std::cerr << "HIP error "              \
                    << status << ": "            \
                    << hipGetErrorString(status) \
                    << " at " << __FILE__ << ":" \
                    << __LINE__ << std::endl;    \
      }                                          \
  }

Complete example
================================================================================

A complete example to demonstrate the error handling with a simple addition of
two values kernel:

.. code-block:: cpp

  #include <hip/hip_runtime.h>
  #include <vector>
  #include <iostream>

  #define HIP_CHECK(expression)                  \
  {                                              \
      const hipError_t status = expression;      \
      if(status != hipSuccess){                  \
          std::cerr << "HIP error "              \
                    << status << ": "            \
                    << hipGetErrorString(status) \
                    << " at " << __FILE__ << ":" \
                    << __LINE__ << std::endl;    \
      }                                          \
  }

  // Addition of two values.
  __global__ void add(int *a, int *b, int *c, size_t size) {
      const size_t index = threadIdx.x + blockDim.x * blockIdx.x;
      if(index < size) {
          c[index] += a[index] + b[index];
      }
  }

  int main() {
      constexpr int numOfBlocks = 256;
      constexpr int threadsPerBlock = 256;
      constexpr size_t arraySize = 1U << 16;

      std::vector<int> a(arraySize), b(arraySize), c(arraySize);
      int *d_a, *d_b, *d_c;

      // Setup input values.
      std::fill(a.begin(), a.end(), 1);
      std::fill(b.begin(), b.end(), 2);

      // Allocate device copies of a, b and c.
      HIP_CHECK(hipMalloc(&d_a, arraySize * sizeof(*d_a)));
      HIP_CHECK(hipMalloc(&d_b, arraySize * sizeof(*d_b)));
      HIP_CHECK(hipMalloc(&d_c, arraySize * sizeof(*d_c)));

      // Copy input values to device.
      HIP_CHECK(hipMemcpy(d_a, &a, arraySize * sizeof(*d_a), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(d_b, &b, arraySize * sizeof(*d_b), hipMemcpyHostToDevice));

      // Launch add() kernel on GPU.
      hipLaunchKernelGGL(add, dim3(numOfBlocks), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, arraySize);
      // Check the kernel launch
      HIP_CHECK(hipGetLastError());
      // Check for kernel execution error
      HIP_CHECK(hipDeviceSynchronize());

      // Copy the result back to the host.
      HIP_CHECK(hipMemcpy(&c, d_c, arraySize * sizeof(*d_c), hipMemcpyDeviceToHost));

      // Cleanup allocated memory.
      HIP_CHECK(hipFree(d_a));
      HIP_CHECK(hipFree(d_b));
      HIP_CHECK(hipFree(d_c));

      // Print the result.
      std::cout << a[0] << " + " << b[0] << " = " << c[0] << std::endl;

      return 0;
  }
