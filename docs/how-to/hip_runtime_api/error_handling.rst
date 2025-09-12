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

.. literalinclude:: ../../tools/example_codes/error_handling.hip
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp
