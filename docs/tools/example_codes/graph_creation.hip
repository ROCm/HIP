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
    std::size_t arraySize = 1U << 20;

    // The pointers to the device memory don't need to be declared here,
    // they are contained within the hipMemAllocNodeParams as the dptr member
    std::vector<double> h_array(arraySize);
    constexpr double initValue = 2.0;

    // Create graph an empty graph
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Parameters to allocate arrays
    hipMemAllocNodeParams allocArrayAParams{};
    allocArrayAParams.poolProps.allocType = hipMemAllocationTypePinned;
    allocArrayAParams.poolProps.location.type = hipMemLocationTypeDevice;
    allocArrayAParams.poolProps.location.id = 0; // GPU on which memory resides
    allocArrayAParams.bytesize = arraySize * sizeof(double);

    hipMemAllocNodeParams allocArrayBParams{};
    allocArrayBParams.poolProps.allocType = hipMemAllocationTypePinned;
    allocArrayBParams.poolProps.location.type = hipMemLocationTypeDevice;
    allocArrayBParams.poolProps.location.id = 0; // GPU on which memory resides
    allocArrayBParams.bytesize = arraySize * sizeof(int);

    // Add the allocation nodes to the graph. They don't have any dependencies
    hipGraphNode_t allocNodeA, allocNodeB;
    HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocArrayAParams));
    HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeB, graph, nullptr, 0, &allocArrayBParams));

    // Parameters for the host function
    // Needs custom struct to pass the arguments
    set_vector_args args{h_array, initValue};
    hipHostNodeParams hostParams{};
    hostParams.fn = set_vector;
    hostParams.userData = static_cast<void*>(&args);

    // Add the host node that initializes the host array. It also doesn't have any dependencies
    hipGraphNode_t hostNode;
    HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

    // Add memory copy node, that copies the initialized host array to the device.
    // It has to wait for the host array to be initialized and the device memory to be allocated
    hipGraphNode_t cpyNodeDependencies[] = {allocNodeA, hostNode};
    hipGraphNode_t cpyToDevNode;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&cpyToDevNode, graph, cpyNodeDependencies, 2, allocArrayAParams.dptr, h_array.data(), arraySize * sizeof(double), hipMemcpyHostToDevice));

    // Parameters for kernelA
    hipKernelNodeParams kernelAParams;
    void* kernelAArgs[] = {&allocArrayAParams.dptr, static_cast<void*>(&arraySize)};
    kernelAParams.func = reinterpret_cast<void*>(kernelA);
    kernelAParams.gridDim = numOfBlocks;
    kernelAParams.blockDim = threadsPerBlock;
    kernelAParams.sharedMemBytes = 0;
    kernelAParams.kernelParams = kernelAArgs;
    kernelAParams.extra = nullptr;

    // Add the node for kernelA. It has to wait for the memory copy to finish, as it depends on the values from the host array.
    hipGraphNode_t kernelANode;
    HIP_CHECK(hipGraphAddKernelNode(&kernelANode, graph, &cpyToDevNode, 1, &kernelAParams));

    // Parameters for kernelB
    hipKernelNodeParams kernelBParams;
    void* kernelBArgs[] = {&allocArrayBParams.dptr, static_cast<void*>(&arraySize)};
    kernelBParams.func = reinterpret_cast<void*>(kernelB);
    kernelBParams.gridDim = numOfBlocks;
    kernelBParams.blockDim = threadsPerBlock;
    kernelBParams.sharedMemBytes = 0;
    kernelBParams.kernelParams = kernelBArgs;
    kernelBParams.extra = nullptr;

    //  Add the node for kernelB. It only has to wait for the memory to be allocated, as it initializes the array.
    hipGraphNode_t kernelBNode;
    HIP_CHECK(hipGraphAddKernelNode(&kernelBNode, graph, &allocNodeB, 1, &kernelBParams));

    // Parameters for kernelC
    hipKernelNodeParams kernelCParams;
    void* kernelCArgs[] = {&allocArrayAParams.dptr, &allocArrayBParams.dptr, static_cast<void*>(&arraySize)};
    kernelCParams.func = reinterpret_cast<void*>(kernelC);
    kernelCParams.gridDim = numOfBlocks;
    kernelCParams.blockDim = threadsPerBlock;
    kernelCParams.sharedMemBytes = 0;
    kernelCParams.kernelParams = kernelCArgs;
    kernelCParams.extra = nullptr;

    // Add the node for kernelC. It has to wait on both kernelA and kernelB to finish, as it depends on their results.
    hipGraphNode_t kernelCNode;
    hipGraphNode_t kernelCDependencies[] = {kernelANode, kernelBNode};
    HIP_CHECK(hipGraphAddKernelNode(&kernelCNode, graph, kernelCDependencies, 2, &kernelCParams));

    // Copy the results back to the host. Has to wait for kernelC to finish.
    hipGraphNode_t cpyToHostNode;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&cpyToHostNode, graph, &kernelCNode, 1, h_array.data(), allocArrayAParams.dptr, arraySize * sizeof(double), hipMemcpyDeviceToHost));

    // Free array of allocNodeA. It needs to wait for the copy to finish, as kernelC stores its results in it.
    hipGraphNode_t freeNodeA;
    HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graph, &cpyToHostNode, 1, allocArrayAParams.dptr));
    // Free array of allocNodeB. It only needs to wait for kernelC to finish, as it is not written back to the host.
    hipGraphNode_t freeNodeB;
    HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeB, graph, &kernelCNode, 1, allocArrayBParams.dptr));

    // Instantiate the graph in order to execute it
    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // The graph can be freed after the instantiation if it's not needed for other purposes
    HIP_CHECK(hipGraphDestroy(graph));

    // Actually launch the graph
    hipStream_t graphStream;
    HIP_CHECK(hipStreamCreate(&graphStream));
    HIP_CHECK(hipGraphLaunch(graphExec, graphStream));

    HIP_CHECK(hipStreamSynchronize(graphStream));

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

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipStreamDestroy(graphStream));

    return EXIT_SUCCESS;
}
// [sphinx-end]
