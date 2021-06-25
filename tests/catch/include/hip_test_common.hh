#pragma once
#include "hip_test_context.hh"
#include <catch.hpp>

#define HIP_PRINT_STATUS(status) INFO(hipGetErrorName(status) << " at line: " << __LINE__);

#define HIPCHECK(error)                                                                            \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      INFO("Error: " << hipGetErrorString(localError) << " Code: " << localError << " Str: "       \
                     << #error << " In File: " << __FILE__ << " At line: " << __LINE__);           \
      REQUIRE(false);                                                                              \
    }                                                                                              \
  }

