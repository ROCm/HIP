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

#include <hip/hip_runtime.h>
#include <type_traits>
#include <iostream>
#include <vector>
#include <random>

#define HIP_CHECK(expression)                \
{                                            \
    const hipError_t status = expression;    \
    if(status != hipSuccess){                \
            std::cerr << "HIP error "        \
                << status << ": "            \
                << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;    \
    }                                        \
}

// [Sphinx HIP warp size block reduction kernel start]
__global__ void block_reduce(int* input, uint64_t* mask, int* output, size_t size){
  extern __shared__ int shared[];
  // Read of input with bounds check
  auto read_global_safe = [&](const uint32_t i, const uint32_t lane_id, const uint32_t mask_id)
  {
    uint64_t warp_mask = 1ull << lane_id;
    return (i < size) && (mask[mask_id] & warp_mask) ? input[i] : 0;
  };
  const uint32_t tid = threadIdx.x,
                 lid = threadIdx.x % warpSize,
                 wid = threadIdx.x / warpSize,
                 bid = blockIdx.x,
                 gid = bid * blockDim.x + tid;
  // Read input buffer to shared
  shared[tid] = read_global_safe(gid, lid, bid * (blockDim.x / warpSize) + wid);
  __syncthreads();
  // Shared reduction
  for (uint32_t i = blockDim.x / 2; i >= warpSize; i /= 2)
  {
    if (tid < i)
      shared[tid] = shared[tid] + shared[tid + i];
    __syncthreads();
  }

  // Use local variable in warp reduction  
  int result =  shared[tid];
  __syncthreads();

  // This loop would be unrolled the same with the compile-time WarpSize.
  #pragma unroll
  for (uint32_t i = warpSize/2; i >= 1; i /= 2) {
    result = result + __shfl_down(result, i);
  }

  // Write result to output buffer
  if (tid == 0)
    output[bid] = result;
};
// [Sphinx HIP warp size block reduction kernel end]

// [Sphinx HIP warp size mask generation start]
void generate_and_copy_mask(
  uint64_t *d_mask, 
  std::vector<int>& vectorExpected,
  int warpSizeHost,
  int numOfBlocks,
  int numberOfWarp,
  int mask_size,
  int mask_element_size) {
  
  std::random_device rd;
  std::mt19937_64 eng(rd());

  // Host side mask vector
  std::vector<uint64_t> mask(mask_size);
  // Define uniform unsigned int distribution
  std::uniform_int_distribution<uint64_t> distr;
  // Fill up the mask 
  for(int i=0; i < numOfBlocks; i++) {
    int count = 0;
    for(int j=0; j < numberOfWarp; j++) {
      int mask_index = i * numberOfWarp + j;
      mask[mask_index] = distr(eng);
      if(warpSizeHost == 32)
        count += __builtin_popcount(mask[mask_index]);
      else
        count += __builtin_popcountll(mask[mask_index]);
    }
    vectorExpected[i]= count;
  }
  // Copy the mask array
  HIP_CHECK(hipMemcpy(d_mask, mask.data(), mask_size * mask_element_size, hipMemcpyHostToDevice));
}
// [Sphinx HIP warp size mask generation end]

int main() {
  int deviceId = 0;
  int warpSizeHost;
  HIP_CHECK(hipDeviceGetAttribute(&warpSizeHost, hipDeviceAttributeWarpSize, deviceId));
  std::cout << "Warp size: " << warpSizeHost << std::endl;
  constexpr int numOfBlocks = 16;
  constexpr int threadsPerBlock = 1024;
  const int numberOfWarp = threadsPerBlock / warpSizeHost;
  const int mask_element_size = sizeof(uint64_t);
  const int mask_size = numOfBlocks * numberOfWarp;
  constexpr size_t arraySize = numOfBlocks * threadsPerBlock;
  int *d_data, *d_results;
  uint64_t *d_mask;
  int initValue = 1;
  std::vector<int> vectorInput(arraySize, initValue);
  std::vector<int> vectorOutput(numOfBlocks);
  std::vector<int> vectorExpected(numOfBlocks);
  // Allocate device memory
  HIP_CHECK(hipMalloc(&d_data, arraySize * sizeof(*d_data)));
  HIP_CHECK(hipMalloc(&d_mask, mask_size * mask_element_size));
  HIP_CHECK(hipMalloc(&d_results, numOfBlocks * sizeof(*d_results)));
  // Host to Device copy of the input array
  HIP_CHECK(hipMemcpy(d_data, vectorInput.data(), arraySize * sizeof(*d_data), hipMemcpyHostToDevice));
  
  // [Sphinx HIP warp size select kernel start]
  // Generate and copy mask arrays
  generate_and_copy_mask(
    d_mask,
    vectorExpected,
    warpSizeHost,
    numOfBlocks,
    numberOfWarp,
    mask_size,
    mask_element_size);

  // Start the kernel
  block_reduce<<<dim3(numOfBlocks), dim3(threadsPerBlock), threadsPerBlock * sizeof(*d_data)>>>(
    d_data,
    d_mask,
    d_results,
    arraySize);
  // [Sphinx HIP warp size select kernel end]

  // Check the kernel launch
  HIP_CHECK(hipGetLastError());
  // Check for kernel execution error
  HIP_CHECK(hipDeviceSynchronize());
  // Device to Host copy of the result
  HIP_CHECK(hipMemcpy(vectorOutput.data(), d_results, numOfBlocks * sizeof(*d_results), hipMemcpyDeviceToHost));
  // Verify results
  bool passed = true;
  for(size_t i = 0; i < numOfBlocks; ++i) {
    if(vectorOutput[i] != vectorExpected[i]) {
      passed = false;
      std::cerr << "Validation failed! Expected " << vectorExpected[i] << " got " << vectorOutput[i] << " at index: " << i << std::endl;
    }
  }
  if(passed){
    std::cout << "Execution completed successfully." << std::endl;
  }else{
    std::cerr << "Execution failed." << std::endl;
  }
  // Cleanup
  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipFree(d_mask));
  HIP_CHECK(hipFree(d_results));
  return 0;
}