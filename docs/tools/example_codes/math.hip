// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#define HIP_CHECK(expression)                        \
    {                                                \
        const hipError_t err = expression;           \
        if (err != hipSuccess)                       \
        {                                            \
            std::cerr << "HIP error: "               \
                      << hipGetErrorString(err)      \
                      << " at " << __LINE__ << "\n"; \
            exit(EXIT_FAILURE);                      \
        }                                            \
    }

// Simple ULP difference calculator
int64_t ulp_diff(float a, float b)
{
    if (a == b)
        return 0;
    
    union
    {
        float f;
        int32_t i;
    } ua{a}, ub{b};

    // For negative values, convert to a positive-based representation
    if (ua.i < 0) ua.i = std::numeric_limits<int32_t>::max() - ua.i;
    if (ub.i < 0) ub.i = std::numeric_limits<int32_t>::max() - ub.i;

    return std::abs((int64_t)ua.i - (int64_t)ub.i);
}

// Test kernel
__global__ void test_sin(float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float x = -M_PI + (2.0f * M_PI * i) / (n - 1);
        out[i] = sinf(x);
    }
}

int main()
{
    const int n = 1000000;
    const int blocksize = 256;
    std::vector<float> outputs(n);
    float* d_out;

    HIP_CHECK(hipMalloc(&d_out, n * sizeof(float)));
    dim3 threads(blocksize);
    dim3 blocks((n + blocksize - 1) / blocksize);  // Fixed grid calculation
    test_sin<<<blocks, threads>>>(d_out, n);
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipMemcpy(outputs.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost));

    // Step 1: Find the maximum absolute error
    double max_abs_error = 0.0;
    float max_error_output = 0.0;
    float max_error_expected = 0.0;

    for (int i = 0; i < n; i++)
    {
        float x = -M_PI + (2.0f * M_PI * i) / (n - 1);
        float expected = std::sin(x);
        double abs_error = std::abs(outputs[i] - expected);

        if (abs_error > max_abs_error)
        {
            max_abs_error = abs_error;
            max_error_output = outputs[i];
            max_error_expected = expected;
        }
    }

    // Step 2: Compute ULP difference based on the max absolute error pair
    int64_t max_ulp = ulp_diff(max_error_output, max_error_expected);

    // Output results
    std::cout << "Max Absolute Error: " << max_abs_error << std::endl;
    std::cout << "Max ULP Difference: " << max_ulp << std::endl;
    std::cout << "Max Error Values -> Got: " << max_error_output
              << ", Expected: " << max_error_expected << std::endl;

    HIP_CHECK(hipFree(d_out));
    return 0;
}
// [sphinx-end]
