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
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

__global__ void block_reduction(const float* input, float* output, int num_elements)
{
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int global_id = blockDim.x * blockIdx.x + tid;

    if (global_id < num_elements)
    {
        s_data[tid] = input[global_id];
    }
    else
    {
        s_data[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = s_data[0];
    }
}

int main()
{
    int threads = 256;
    const int num_elements = 50000;

    std::vector<float> h_a(num_elements);
    std::vector<float> h_b((num_elements + threads - 1) / threads);

    for (int i = 0; i < num_elements; ++i)
    {
        h_a[i] = rand() / static_cast<float>(RAND_MAX);
    }

    float *d_a, *d_b;
    cudaMalloc(&d_a, h_a.size() * sizeof(float));
    cudaMalloc(&d_b, h_b.size() * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaMemcpyAsync(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaEventRecord(start_event, stream);

    int blocks = (num_elements + threads - 1) / threads;
    block_reduction<<<blocks, threads, threads * sizeof(float), stream>>>(d_a, d_b, num_elements);

    cudaMemcpyAsync(h_b.data(), d_b, h_b.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);

    float milliseconds = 0.f;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    cudaFree(d_a);
    cudaFree(d_b);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaStreamDestroy(stream);

    return 0;
}
// [sphinx-end]
