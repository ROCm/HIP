// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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
