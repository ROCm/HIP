// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// [sphinx-start]
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(expression)                      \
    {                                              \
        const hipError_t err = expression;         \
        if (err != hipSuccess) {                   \
            std::cerr << "HIP error: "             \
                    << hipGetErrorString(err)      \
                    << " at " << __LINE__ << "\n"; \
            exit(EXIT_FAILURE);                    \
        }                                          \
    }

// Kernel to compute DFT
__global__ void computeDFT(const float* input, hipFloatComplex* output, const int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    hipFloatComplex sum = make_hipFloatComplex(0.0f, 0.0f);

    for (int n = 0; n < N; n++)
    {
        float angle = -2.0f * M_PI * k * n / N;
        hipFloatComplex w = make_hipFloatComplex(cosf(angle), sinf(angle));
        hipFloatComplex x = make_hipFloatComplex(input[n], 0.0f);
        sum = hipCaddf(sum, hipCmulf(x, w));
    }

    output[k] = sum;
}

// CPU implementation of DFT for verification
std::vector<hipFloatComplex> cpuDFT(const std::vector<float>& input)
{
    const int N = input.size();
    std::vector<hipFloatComplex> result(N);

    for (int k = 0; k < N; k++)
    {
        hipFloatComplex sum = make_hipFloatComplex(0.0f, 0.0f);
        for (int n = 0; n < N; n++)
        {
            float angle = -2.0f * M_PI * k * n / N;
            hipFloatComplex w = make_hipFloatComplex(cosf(angle), sinf(angle));
            hipFloatComplex x = make_hipFloatComplex(input[n], 0.0f);
            sum = hipCaddf(sum, hipCmulf(x, w));
        }
        result[k] = sum;
    }
    return result;
}

int main()
{
    const int N = 256;  // Signal length
    const int blockSize = 256;

    // Generate input signal: sum of two sine waves
    std::vector<float> signal(N);
    for (int i = 0; i < N; i++)
    {
        float t = static_cast<float>(i) / N;
        signal[i] = sinf(2.0f * M_PI * 10.0f * t) +  // 10 Hz component
                0.5f * sinf(2.0f * M_PI * 20.0f * t);  // 20 Hz component
    }

    // Compute reference solution on CPU
    std::vector<hipFloatComplex> cpu_output = cpuDFT(signal);

    // Allocate device memory
    float* d_signal;
    hipFloatComplex* d_output;
    HIP_CHECK(hipMalloc(&d_signal, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(hipFloatComplex)));

    // Copy input to device
    HIP_CHECK(hipMemcpy(d_signal, signal.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 grid((N + blockSize - 1) / blockSize);
    dim3 block(blockSize);
    computeDFT<<<grid, block>>>(d_signal, d_output, N);
    HIP_CHECK(hipGetLastError());

    // Get GPU results
    std::vector<hipFloatComplex> gpu_output(N);
    HIP_CHECK(hipMemcpy(gpu_output.data(), d_output, N * sizeof(hipFloatComplex), hipMemcpyDeviceToHost));

    // Verify results
    bool passed = true;
    const float tolerance = 1e-5f;  // Adjust based on precision requirements

    for (int i = 0; i < N; i++)
    {
        float diff_real = std::abs(hipCrealf(gpu_output[i]) - hipCrealf(cpu_output[i]));
        float diff_imag = std::abs(hipCimagf(gpu_output[i]) - hipCimagf(cpu_output[i]));

        if (diff_real > tolerance || diff_imag > tolerance)
        {
            passed = false;
            break;
        }
    }

    std::cout << "DFT Verification: " << (passed ? "PASSED" : "FAILED") << "\n";

    // Cleanup
    HIP_CHECK(hipFree(d_signal));
    HIP_CHECK(hipFree(d_output));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
// [sphinx-end]
