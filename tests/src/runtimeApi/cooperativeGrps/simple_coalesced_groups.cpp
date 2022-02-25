/*
Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Test Description:
/* This test implements sum reduction kernel, first with each threads own rank
   as input and comparing the sum with expected sum output derieved from n(n-1)/2
   formula. The second part, partitions this parent group into child subgroups
   a.k.a tiles using using tiled_partition() collective operation. This can be called
   with a static tile size, passed in templated non-type variable-tiled_partition<tileSz>,
   or in runtime as tiled_partition(thread_group parent, tileSz). This test covers both these
   cases.
   This test tests functionality of cg group partitioning, (static and dynamic) and its respective
   API's size(), thread_rank(), and sync().
*/

#include "test_common.h"
#include <hip/hip_cooperative_groups.h>
#include <stdio.h>
#include <vector>

using namespace cooperative_groups;

#define ASSERT_EQUAL(lhs, rhs) assert(lhs == rhs)

#define NUM_ELEMS 10000000
#define NUM_THREADS_PER_BLOCK 512
#define WAVE_SIZE 32

/* Test coalesced group's functionality.
 *
 */

__device__ int atomicAggInc(int *ptr) {
   coalesced_group g = coalesced_threads();
   int prev;
   // elect the first active thread to perform atomic add
   if (g.thread_rank() == 0) {
     prev = atomicAdd(ptr, g.size());
   }
   // broadcast previous value within the warp
   // and add each active thread’s rank to it
   prev = g.thread_rank() + g.shfl(prev, 0);
   return prev;
}

__global__ void kernel_shfl (int * dPtr, int *dResults, int srcLane, int cg_sizes) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id % cg_sizes == 0) {
    coalesced_group const& g = coalesced_threads();
    int rank = g.thread_rank();
    int val = dPtr[rank];
    dResults[rank] = g.shfl(val, srcLane);
    return;
  }
}

__global__ void kernel_shfl_any_to_any (int *randVal, int *dsrcArr, int *dResults, int cg_sizes) {

 int id = threadIdx.x + blockIdx.x * blockDim.x;

 if (id % cg_sizes == 0) {
    coalesced_group const& g = coalesced_threads();
    int rank = g.thread_rank();
    int val = randVal[rank];
    dResults[rank] = g.shfl(val, dsrcArr[rank]);
    return;
  }

}

__global__ void filter_arr(int *dst, int *nres, const int *src, int n) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  for (int i = id; i < n; i += gridDim.x * blockDim.x) {
    if (src[i] > 0) dst[atomicAggInc(nres)] = src[i];
  }
}

/* Parallel reduce kernel.
 *
 * Step complexity: O(log n)
 * Work complexity: O(n)
 *
 * Note: This kernel works only with power of 2 input arrays.
 */
__device__ int reduction_kernel(coalesced_group g, int* x, int val) {
  int lane = g.thread_rank();
  int sz = g.size();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    // use lds to store the temporary result
    x[lane] = val;
    // Ensure all the stores are completed.
    g.sync();

    if (lane < i) {
      val += x[lane + i];
    }
    // It must work on one tiled thread group at a time,
    // and it must make sure all memory operations are
    // completed before moving to the next stride.
    // sync() here just does that.
    g.sync();
  }

  // Choose the 0'th indexed thread that holds the reduction value to return
  if (g.thread_rank() == 0) {
    return val;
  }
  // Rest of the threads return no useful values
  else {
    return -1;
  }
}

__global__ void kernel_cg_coalesced_group_partition(unsigned int tileSz, int* result,
                                                  bool isGlobalMem, int* globalMem, int cg_sizes) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id % cg_sizes == 0) {
    coalesced_group threadBlockCGTy = coalesced_threads();
    int threadBlockGroupSize = threadBlockCGTy.size();

    int* workspace = NULL;

    if (isGlobalMem) {
      workspace = globalMem;
    } else {
      // Declare a shared memory
      extern __shared__ int sharedMem[];
      workspace = sharedMem;
    }

    int input, outputSum, expectedOutput;

    // input to reduction, for each thread, is its' rank in the group
    input = threadBlockCGTy.thread_rank();

    expectedOutput = (threadBlockGroupSize - 1) * threadBlockGroupSize / 2;

    outputSum = reduction_kernel(threadBlockCGTy, workspace, input);

    if (threadBlockCGTy.thread_rank() == 0) {
      printf(" Sum of all ranks 0..%d in coalesced_group is %d\n\n",
             (int)threadBlockCGTy.size() - 1, outputSum);
      printf(" Creating %d groups, of tile size %d threads:\n\n",
             (int)threadBlockCGTy.size() / tileSz, tileSz);
    }

    threadBlockCGTy.sync();

    coalesced_group tiledPartition = tiled_partition(threadBlockCGTy, tileSz);

    // This offset allows each group to have its own unique area in the workspace array
    int workspaceOffset = threadBlockCGTy.thread_rank() - tiledPartition.thread_rank();

    outputSum = reduction_kernel(tiledPartition, workspace + workspaceOffset, input);

    if (tiledPartition.thread_rank() == 0) {
      printf(
          "   Sum of all ranks 0..%d in this tiledPartition group is %d. Corresponding parent thread "
          "rank: %d\n",
          tiledPartition.size() - 1, outputSum, input);

        result[input / (tileSz)] = outputSum;
    }
    return;
  }
}

__global__ void kernel_coalesced_active_groups() {
  thread_block threadBlockCGTy = this_thread_block();
  int threadBlockGroupSize = threadBlockCGTy.size();

  // input to reduction, for each thread, is its' rank in the group
  int input = threadBlockCGTy.thread_rank();

  if (threadBlockCGTy.thread_rank() == 0) {
    printf(" Creating odd and even set of active thread groups based on branch divergence\n\n");
  }

  threadBlockCGTy.sync();

  // Group all active odd threads
  if (threadBlockCGTy.thread_rank() % 2) {
    coalesced_group activeOdd = coalesced_threads();

    if (activeOdd.thread_rank() == 0) {
      printf(" ODD: Size of odd set of active threads is %d."
             " Corresponding parent thread_rank is %d.\n\n",
               activeOdd.size(), threadBlockCGTy.thread_rank());
    }
  }
  else { // Group all active even threads
    coalesced_group activeEven = coalesced_threads();

    if (activeEven.thread_rank() == 0) {
      printf(" EVEN: Size of even set of active threads is %d."
             " Corresponding parent thread_rank is %d.",
               activeEven.size(), threadBlockCGTy.thread_rank());
    }
  }
  return;
}

void printResults(int* ptr, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << '\n';
}

void compareResults(int* cpu, int* gpu, int size) {
  for (unsigned int i = 0; i < size / sizeof(int); i++) {
    if (cpu[i] != gpu[i]) {
      printf(" results do not match.");
    }
  }
}

static void test_active_threads_grouping() {
  hipError_t err;
  int blockSize = 1;
  int threadsPerBlock = WAVE_SIZE;

  // Launch Kernel
    hipLaunchKernelGGL(kernel_coalesced_active_groups, blockSize, threadsPerBlock, 0, 0);

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }
  printf("\n...PASSED.\n\n");
}

// Search if the sum exists in the expected results array
void verifyResults(int* hPtr, int* dPtr, int size) {
  int i = 0, j = 0;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      if (hPtr[i] == dPtr[j]) {
        break;
      }
    }
    if (j == size) {
      printf(" Result verification failed!");
    }
  }
}


static void test_group_partition(unsigned int tileSz, bool useGlobalMem) {
  hipError_t err;
  int blockSize = 1;
  int threadsPerBlock = WAVE_SIZE;

  std::vector<unsigned int> cg_sizes = {1, 2, 3};
  for (auto i : cg_sizes) {

    int numTiles = ((blockSize * threadsPerBlock) / i) / tileSz;

    // numTiles = 0 when partitioning is possible. The below statement is to avoid
    // out-of-bounds error and still evaluate failure case.
    numTiles = (numTiles == 0) ? 1 : numTiles;

    // Build an array of expected reduction sum output on the host
    // based on the sum of their respective thread ranks to use for verification
    int* expectedSum = new int[numTiles];
    int temp = 0, sum = 0;
    for (int i = 1; i <= numTiles; i++) {
      sum = temp;
      temp = (((tileSz * i) - 1) * (tileSz * i)) / 2;
      expectedSum[i-1] = temp - sum;
    }

    int* dResult = NULL;
    hipMalloc(&dResult, sizeof(int) * numTiles);

    int* globalMem = NULL;
    if (useGlobalMem) {
      hipMalloc((void**)&globalMem, threadsPerBlock * sizeof(int));
    }

    int* hResult = NULL;
    hipHostMalloc(&hResult, numTiles * sizeof(int), hipHostMallocDefault);
    memset(hResult, 0, numTiles * sizeof(int));

    // Launch Kernel
    if (useGlobalMem) {
      hipLaunchKernelGGL(kernel_cg_coalesced_group_partition, blockSize, threadsPerBlock, 0, 0, tileSz,
                         dResult, useGlobalMem, globalMem, i);

      err = hipDeviceSynchronize();
      if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
      }
    } else {
      hipLaunchKernelGGL(kernel_cg_coalesced_group_partition, blockSize, threadsPerBlock,
                         threadsPerBlock * sizeof(int), 0, tileSz, dResult, useGlobalMem, globalMem, i);

      err = hipDeviceSynchronize();
      if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
      }
    }

    hipMemcpy(hResult, dResult, numTiles * sizeof(int), hipMemcpyDeviceToHost);
    verifyResults(expectedSum, hResult, numTiles);
    // Free all allocated memory on host and device
    hipFree(dResult);
    hipFree(hResult);
    if (useGlobalMem) {
      hipFree(globalMem);
    }
    delete[] expectedSum;

    printf("\n...PASSED.\n\n");
  }
}
static void test_shfl_any_to_any() {

  std::vector<unsigned int> cg_sizes = {1, 2, 3};
  for (auto i : cg_sizes) {

    hipError_t err;
    int blockSize = 1;
    int threadsPerBlock = WAVE_SIZE;

    int totalThreads = blockSize * threadsPerBlock;
    int group_size = (totalThreads + i - 1) / i;
    int group_size_in_bytes = group_size * sizeof(int);

    int* hPtr = NULL;
    int* dPtr = NULL;
    int* dsrcArr = NULL;
    int* dResults = NULL;
    int* srcArr = (int*)malloc(group_size_in_bytes);
    int* srcArrCpu = (int*)malloc(group_size_in_bytes);

    std::cout << "Testing coalesced_groups shfl any-to-any\n" <<std::endl;

    int arrSize = blockSize * threadsPerBlock * sizeof(int);

    hipHostMalloc(&hPtr, arrSize);
    // Fill up the array
    for (int i = 0; i < WAVE_SIZE; i++) {
      hPtr[i] = rand() % 1000;
    }

    // Fill up the random array
    for (int i = 0; i < group_size; i++) {
      srcArr[i] = rand() % 1000;
      srcArrCpu[i] = srcArr[i] % group_size;
    }

    /* Fill cpu results array so that we can verify with gpu computation */
    int* cpuResultsArr = (int*)malloc(group_size_in_bytes);
    for(int i = 0; i < group_size; i++) {
      cpuResultsArr[i] = hPtr[srcArrCpu[i]];
    }

    //printf("Array passed to GPU for computation\n");
    //printResults(hPtr, WAVE_SIZE);
    hipMalloc(&dPtr, group_size_in_bytes);
    hipMalloc(&dResults, group_size_in_bytes);

    hipMalloc(&dsrcArr, group_size_in_bytes);
    hipMemcpy(dsrcArr, srcArr, group_size_in_bytes, hipMemcpyHostToDevice);

    hipMemcpy(dPtr, hPtr, group_size_in_bytes, hipMemcpyHostToDevice);
    // Launch Kernel
    hipLaunchKernelGGL(kernel_shfl_any_to_any, blockSize, threadsPerBlock,
                       threadsPerBlock * sizeof(int), 0 , dPtr, dsrcArr, dResults, i);
    hipMemcpy(hPtr, dResults, group_size_in_bytes, hipMemcpyDeviceToHost);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }
    //printf("GPU results: \n");
    //printResults(hPtr, group_size);
    //printf("Printing cpu to be verified array\n");
    //printResults(cpuResultsArr, group_size);
    //printf("Printing srcLane array that was passed\n");
    //printResults(srcArr, group_size);
    //printf("Printing srcLane array on the CPU\n");
    //printResults(srcArrCpu, group_size);
    compareResults(hPtr, cpuResultsArr, group_size_in_bytes);
    std::cout << "Results verified!\n";

    hipFree(hPtr);
    hipFree(dPtr);
    free(srcArr);
    free(srcArrCpu);
    free(cpuResultsArr);
  }
}
static void test_shfl_broadcast() {

  std::vector<unsigned int> cg_sizes = {1, 2, 3};
  for (auto i : cg_sizes) {

    hipError_t err;
    int blockSize = 1;
    int threadsPerBlock = WAVE_SIZE;

    int totalThreads = blockSize * threadsPerBlock;
    int group_size = (totalThreads + i - 1) / i;
    int group_size_in_bytes = group_size * sizeof(int);

    int* hPtr = NULL;
    int* dPtr = NULL;
    int* dResults = NULL;
    int srcLane = rand() % 1000;
    int srcLaneCpu = 0;
    std::cout << "Testing coalesced_groups shfl with srcLane " << srcLane << '\n'
              << " and group size " << i <<std::endl;

    int arrSize = blockSize * threadsPerBlock * sizeof(int);

    hipHostMalloc(&hPtr, arrSize);
    // Fill up the array
    for (int i = 0; i < WAVE_SIZE; i++) {
      hPtr[i] = rand() % 1000;
    }


    /* Fill cpu results array so that we can verify with gpu computation */
    srcLaneCpu = hPtr[srcLane % group_size];

    int* cpuResultsArr = (int*)malloc(sizeof(int) * group_size);
    for (int i = 0; i < group_size; i++) {
      cpuResultsArr[i] = srcLaneCpu;
    }
    printf("Array passed to GPU for computation\n");
    printResults(hPtr, WAVE_SIZE);
    hipMalloc(&dPtr, group_size_in_bytes);
    hipMalloc(&dResults, group_size_in_bytes);

    hipMemcpy(dPtr, hPtr, group_size_in_bytes, hipMemcpyHostToDevice);
    // Launch Kernel
    hipLaunchKernelGGL(kernel_shfl, blockSize, threadsPerBlock,
                       threadsPerBlock * sizeof(int), 0, dPtr, dResults, srcLane, i);
    hipMemcpy(hPtr, dResults, group_size_in_bytes, hipMemcpyDeviceToHost);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }
    printf("GPU results: \n");
    printResults(hPtr, group_size);
    printf("Printing cpu to be verified array\n");
    printResults(cpuResultsArr, group_size);

    compareResults(hPtr, cpuResultsArr, group_size_in_bytes);
    std::cout << "Results verified!\n";

    hipFree(hPtr);
    hipFree(dPtr);
    free(cpuResultsArr);
  }
}

int main() {
  // Use default device for validating the test
  int deviceId;
  ASSERT_EQUAL(hipGetDevice(&deviceId), hipSuccess);
  hipDeviceProp_t deviceProperties;
  ASSERT_EQUAL(hipGetDeviceProperties(&deviceProperties, deviceId), hipSuccess);
  int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

  if (!deviceProperties.cooperativeLaunch) {
    std::cout << "info: Device doesn't support cooperative launch! skipping the test!\n";
    if (hip_skip_tests_enabled()) {
      return hip_skip_retcode();
    } else {
      passed();
    }
  }

  std::cout << "Now testing coalesced_groups" << '\n' << std::endl;

  int *data_to_filter, *filtered_data, nres = 0;
  int *d_data_to_filter, *d_filtered_data, *d_nres;

  int numOfBuckets = 5;

  data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate input data.
  for (int i = 0; i < NUM_ELEMS; i++) {
    data_to_filter[i] = rand() % numOfBuckets;
  }


  hipMalloc(&d_data_to_filter, sizeof(int) * NUM_ELEMS);
  hipMalloc(&d_filtered_data, sizeof(int) * NUM_ELEMS);
  hipMalloc(&d_nres, sizeof(int));

  hipMemcpy(d_data_to_filter, data_to_filter,
                             sizeof(int) * NUM_ELEMS, hipMemcpyHostToDevice);
  hipMemset(d_nres, 0, sizeof(int));

  dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
  dim3 dimGrid((NUM_ELEMS / NUM_THREADS_PER_BLOCK) + 1, 1, 1);

  filter_arr<<<dimGrid, dimBlock>>>(d_filtered_data, d_nres, d_data_to_filter,
                                    NUM_ELEMS);


  hipMemcpy(&nres, d_nres, sizeof(int), hipMemcpyDeviceToHost);

  filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * nres));

  hipMemcpy(filtered_data, d_filtered_data, sizeof(int) * nres,
                             hipMemcpyDeviceToHost);

  int *host_filtered_data =
      reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate host output with host filtering code.
  int host_flt_count = 0;
  for (int i = 0; i < NUM_ELEMS; i++) {
    if (data_to_filter[i] > 0) {
      host_filtered_data[host_flt_count++] = data_to_filter[i];
    }
  }

  printf("\nWarp Aggregated Atomics %s \n",
         (host_flt_count == nres) ? "PASSED" : "FAILED");

  // Now, testing shfl collective
  std::cout << "Now testing shfl collective as a broadcast" << '\n' << std::endl;

  for (int i = 0; i < 100; i++) {
    test_shfl_broadcast();
  }


  // Now, testing shfl collective
  std::cout << "Now testing shfl operations any-to-any member lanes" << '\n' << std::endl;

  for (int i = 0; i < 100; i++) {
    test_shfl_any_to_any();
  }

  // Now, pass a already coalesced_group that was partitioned
  /* Test coalesced group partitioning */
  std::cout << "Now testing coalesced_groups partitioning" << '\n' << std::endl;

  int testNo = 1;
  for (int memTy = 0; memTy < 2; memTy++) {
    std::vector<unsigned int> tileSizes = {2, 4, 8, 16, 32};
    for (auto i : tileSizes) {
      std::cout << "TEST " << testNo << ":" << '\n' << std::endl;
      test_group_partition(i, memTy);
      testNo++;
    }
  }

  std::cout << "Now grouping active threads based on branch divergence" << '\n' << std::endl;
  test_active_threads_grouping();

  passed();
  return 0;
}
