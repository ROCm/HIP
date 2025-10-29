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

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(expression)                \
{                                            \
    const hipError_t status = expression;    \
    if(status != hipSuccess)                 \
    {                                        \
            std::cerr << "HIP error "        \
                << status << ": "            \
                << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;    \
    }                                        \
}

__global__ void kernelA(double* arrayA, std::size_t size)
{
    const std::size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size)
    {
        arrayA[x] *= 2.0;
    }
}

__global__ void kernelB(int* arrayB, std::size_t size)
{
    const std::size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size)
    {
        arrayB[x] = 3;
    }
}

__global__ void kernelC(double* arrayA, const int* arrayB, std::size_t size)
{
    const std::size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size)
    {
        arrayA[x] += arrayB[x];
    }
}

struct set_vector_args
{
    std::vector<double>& h_array;
    double value;
};

void set_vector(void* args)
{
    set_vector_args h_args{*(reinterpret_cast<set_vector_args*>(args))};

    std::vector<double>& vec{h_args.h_array};
    vec.assign(vec.size(), h_args.value);
}

int main()
{
    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr std::size_t arraySize = 1U << 20;

    // This example assumes that kernelA operates on data that needs to be initialized on
    // and copied from the host, while kernelB initializes the array that is passed to it.
    // Both arrays are then used as input to kernelC, where arrayA is also used as
   //  output, that is copied back to the host, while arrayB is only read from and not modified.

    double* d_arrayA;
    int* d_arrayB;
    std::vector<double> h_array(arraySize);
    constexpr double initValue = 2.0;

    hipStream_t captureStream;
    HIP_CHECK(hipStreamCreate(&captureStream));

    // Start capturing the operations assigned to the stream
    HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));

    // hipMallocAsync and hipMemcpyAsync are needed, to be able to assign it to a stream
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&d_arrayA), arraySize*sizeof(double), captureStream));
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&d_arrayB), arraySize*sizeof(int), captureStream));

    // Assign host function to the stream
    // Needs a custom struct to pass the arguments
    set_vector_args args{h_array, initValue};
    HIP_CHECK(hipLaunchHostFunc(captureStream, set_vector, &args));

    HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array.data(), arraySize*sizeof(double), hipMemcpyHostToDevice, captureStream));

    kernelA<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, arraySize);
    kernelB<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayB, arraySize);
    kernelC<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, d_arrayB, arraySize);

    HIP_CHECK(hipMemcpyAsync(h_array.data(), d_arrayA, arraySize*sizeof(*d_arrayA), hipMemcpyDeviceToHost, captureStream));

    HIP_CHECK(hipFreeAsync(d_arrayA, captureStream));
    HIP_CHECK(hipFreeAsync(d_arrayB, captureStream));

    // Stop capturing
    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(captureStream, &graph));

    // Create an executable graph from the captured graph
    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // The graph template can be deleted after the instantiation if it's not needed for later use
    HIP_CHECK(hipGraphDestroy(graph));

    // Actually launch the graph. The stream does not have
    // to be the same as the one used for capturing.
    HIP_CHECK(hipGraphLaunch(graphExec, captureStream));

    HIP_CHECK(hipStreamSynchronize(captureStream));

    // Verify results
    constexpr double expected = initValue * 2.0 + 3;
    bool passed = true;
    for(std::size_t i = 0; i < arraySize; ++i)
    {
        if(h_array[i] != expected)
        {
            passed = false;
            std::cerr << "Validation failed! Expected " << expected << " got " << h_array[0] << std::endl;
            break;
        }
    }

    if(passed)
    {
        std::cerr << "Validation passed." << std::endl;
    }

    // Free graph and stream resources after usage
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipStreamDestroy(captureStream));

    return EXIT_SUCCESS;
}
// [sphinx-end]
