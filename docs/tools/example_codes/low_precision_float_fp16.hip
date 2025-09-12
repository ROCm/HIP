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

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define hip_check(hip_call)                                                    \
{                                                                              \
    auto hip_res = hip_call;                                                   \
    if (hip_res != hipSuccess) {                                               \
        std::cerr << "Failed in HIP call: " << #hip_call                       \
                << " at " << __FILE__ << ":" << __LINE__                       \
                << " with error: " << hipGetErrorString(hip_res) << std::endl; \
        std::abort();                                                          \
    }                                                                          \
}

__global__ void add_half_precision(__half* in1, __half* in2, float* out, size_t size)
{
    int idx = threadIdx.x;
    if (idx < size)
    {
        // Load as half, perform addition in float, store as float
        float sum = __half2float(in1[idx] + in2[idx]);
        out[idx] = sum;
    }
}

int main()
{
    constexpr size_t size = 32;
    constexpr float tolerance = 1e-1f;  // Allowable numerical difference

    // Initialize input vectors as floats
    std::vector<float> in1(size), in2(size);
    for (size_t i = 0; i < size; i++) {
        in1[i] = i + 1.1f;
        in2[i] = i + 2.2f;
    }

    // Compute expected results in full precision on CPU
    std::vector<float> cpu_out(size);
    for (size_t i = 0; i < size; i++) {
        cpu_out[i] = in1[i] + in2[i];  // Direct float addition
    }

    // Allocate device memory (store input as half, output as float)
    __half *d_in1, *d_in2;
    float *d_out;
    hip_check(hipMalloc(&d_in1, sizeof(__half) * size));
    hip_check(hipMalloc(&d_in2, sizeof(__half) * size));
    hip_check(hipMalloc(&d_out, sizeof(float) * size));

    // Convert input to half and copy to device
    std::vector<__half> in1_half(size), in2_half(size);
    for (size_t i = 0; i < size; i++)
    {
        in1_half[i] = __float2half(in1[i]);
        in2_half[i] = __float2half(in2[i]);
    }

    hip_check(hipMemcpy(d_in1, in1_half.data(), sizeof(__half) * size, hipMemcpyHostToDevice));
    hip_check(hipMemcpy(d_in2, in2_half.data(), sizeof(__half) * size, hipMemcpyHostToDevice));

    // Launch kernel
    add_half_precision<<<1, size>>>(d_in1, d_in2, d_out, size);

    // Copy result back to host
    std::vector<float> gpu_out(size, 0.0f);
    hip_check(hipMemcpy(gpu_out.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));

    // Free device memory
    hip_check(hipFree(d_in1));
    hip_check(hipFree(d_in2));
    hip_check(hipFree(d_out));

    // Validation with tolerance
    for (size_t i = 0; i < size; i++)
    {
        if (std::fabs(cpu_out[i] - gpu_out[i]) > tolerance)
        {
            std::cerr << "Mismatch at index " << i
                      << ": CPU result = " << cpu_out[i]
                      << ", GPU result = " << gpu_out[i] << std::endl;
            std::abort();
        }
    }

    std::cout << "Success: CPU and GPU half-precision addition match within tolerance!" << std::endl;
}
