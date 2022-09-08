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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// Test Description:
/*
The general idea of the application is to test how Cooperative Groups kernel
launches to a stream interact with other kernels being launched to different
streams.

For example: the HIP runtime will force cooperative kernel launches to run
serially, even if they are launched to different streams. However,
cooperative kernel launches can run in parallel with regular kernels that
are launched to other streams. This limitation is so that the cooperative
kernels do not conflict with one another for resources and potentially
deadlock the system.

As such, this benchmark tests three situations:

  1. Launching a cooperative kernel by itself to stream[0]
  2. Launching two cooperative kernels in parallel to stream[0] and stream[1]
  3. Launching two cooperative kernels in parallel to stream[0] and stream[1]
     and launching a third non-cooperative kernel to stream[2]

We time how long it takes to run each of these benchmarks and print it as
the output of the benchmark. The kernels themselves are just useless time-
wasting code so that the kernel takes a meaningful amount of time on the
GPU before it exits. We only launch a single wavefront for each kernel, so
any serialization should not be because of GPU occupancy concerns.

If test #2 takes roughly twice as long as #1, that implies that cooperative
kernels are properly serialized with each other by the runtime.

If test #3 takes the same amount of time as test #2, that implies that
regular kernels can properly run in parallel with cooperative kernels.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * TEST_NAMED: %t  cooperative_streams_least_capacity --tests 0x0
 * TEST_NAMED: %t  cooperative_streams_half_capacity --tests 0x1
 * TEST_NAMED: %t  cooperative_streams_full_capacity --tests 0x2
 * HIT_END
 */

#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "test_common.h"

static inline void hipCheckAndFail(hipError_t errval,
        const char *file, int line) {
  hipError_t last_err = hipGetLastError();
  if (errval != hipSuccess) {
    std::cerr << "hip error: " << hipGetErrorString(errval);
    std::cerr << std::endl;
    std::cerr << "Location: " << file << ":" << line << std::endl;
    failed(" ");
  }
  if (last_err != errval) {
    std::cerr << "Error: the return value of a function was not the same ";
    std::cerr << "as the value returned by hipGetLastError()" << std::endl;
    std::cerr << "Location: " << file << ":" << line << std::endl;
    std::cerr << "Function returned: " << hipGetErrorString(errval);
    std::cerr << " (" << errval << ")" << std::endl;
    std::cerr << "hipGetLastError() returned: " << hipGetErrorString(last_err);
    std::cerr << " (" << last_err << ")" << std::endl;
    failed(" ");
  }
}
#define hipCheckErr(errval) \
  do { hipCheckAndFail((errval), __FILE__, __LINE__); } while (0)

__global__ void test_kernel(uint32_t loops, unsigned long long *array, long long totalTicks) {
  cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();
  unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < loops; i++) {
    long long time_diff = 0;
    long long last_clock = clock64();
    do {
      long long cur_clock = clock64();
      if (cur_clock > last_clock) {
        time_diff += (cur_clock - last_clock);
      }
      // If it rolls over, we don't know how much to add to catch up.
      // So just ignore those slipped cycles.
      last_clock = cur_clock;
    } while(time_diff < totalTicks);
    tb.sync();
    array[rank] += clock64();
  }
}

template<typename T>
bool verifyLeastCapacity(T& single_kernel_time, T& double_kernel_time, T& triple_kernel_time)
{
#ifdef __HIP_PLATFORM_AMD__
  // hipLaunchCooperativeKernel() follows serialization policy on AMD devices
  // Test that the two cooperative kernels took roughly twice as long as the one
  if (double_kernel_time < 1.8 * single_kernel_time ||
      double_kernel_time > 2.2 * single_kernel_time ) {
    std::cerr << "ERROR!" << std::endl;
    std::cerr << "Two cooperative kernels launched at the same ";
    std::cerr << "time did not take roughly twice as long as a single ";
    std::cerr << "cooperative kernel." << std::endl;
    std::cerr << "Were they truly serialized?" << std::endl;
    return false;
  }
#else
  // hipLaunchCooperativeKernel() doesn't follow serialization policy on NV devices
  // Test that the two cooperative kernels took roughly as long as the one
  if (double_kernel_time < 0.8 * single_kernel_time ||
      double_kernel_time > 1.2 * single_kernel_time ) {
    std::cerr << "ERROR!" << std::endl;
    std::cerr << "Two cooperative kernels launched at the same ";
    std::cerr << "time did not take roughly as long as a single ";
    std::cerr << "cooperative kernel." << std::endl;
    return false;
  }
#endif

  // Test that the three kernels together took roughly as long as the two
  // cooperative kernels.
  if (triple_kernel_time > 1.1 * double_kernel_time) {
    std::cerr << "ERROR!" << std::endl;
    std::cerr << "Launching a normal kernel in parallel with two ";
    std::cerr << "back-to-back cooperative kernels still ended up taking ";
    std::cerr << "more than 10% longer than the two cooperative kernels ";
    std::cerr << "alone." << std::endl;
    return false;
  }
  return true;
}

template<typename T>
bool verifyHalfCapacity(T& single_kernel_time, T& double_kernel_time, T& triple_kernel_time)
{
  // Test that the two cooperative kernels took roughly twice as long as the one
  if (double_kernel_time < 1.8 * single_kernel_time ||
      double_kernel_time > 2.2 * single_kernel_time ) {
    std::cerr << "ERROR!" << std::endl;
    std::cerr << "Two cooperative kernels launched at the same ";
    std::cerr << "time did not take roughly twice as long as a single ";
    std::cerr << "cooperative kernel." << std::endl;
    return false;
  }

  // Test that the three kernels together took roughly as long as the two
  // cooperative kernels.
  if (triple_kernel_time > 1.1 * double_kernel_time) {
    std::cerr << "ERROR!" << std::endl;
    std::cerr << "Launching a normal kernel in parallel with two ";
    std::cerr << "back-to-back cooperative kernels still ended up taking ";
    std::cerr << "more than 10% longer than the two cooperative kernels ";
    std::cerr << "alone." << std::endl;
    return false;
  }
  return true;
}

template<typename T>
bool verifyFullCapacity(T& single_kernel_time, T& double_kernel_time, T& triple_kernel_time)
{
  // Test that the two cooperative kernels took roughly twice as long as the one
  if (double_kernel_time < 1.8 * single_kernel_time ||
      double_kernel_time > 2.2 * single_kernel_time ) {
    std::cerr << "ERROR!" << std::endl;
    std::cerr << "Two cooperative kernels launched at the same ";
    std::cerr << "time did not take roughly twice as long as a single ";
    std::cerr << "cooperative kernel." << std::endl;
    return false;
  }

  // Test that the three kernels together took roughly 1.6 times as long as the two
  // cooperative kernels. If the first 2 kernels run very fast, the third
  // won't share much time with the second kernel.
  if (triple_kernel_time > 1.7 * double_kernel_time) {
    std::cerr << "ERROR!" << std::endl;
    std::cerr << "Launching a normal kernel in parallel with two ";
    std::cerr << "back-to-back cooperative kernels still ended up taking ";
    std::cerr << "more than 70% longer than the two cooperative kernels ";
    std::cerr << "alone." << std::endl;
    return false;
  }
  return true;
}

template<typename T>
bool verify(int tests, T &single_kernel_time, T &double_kernel_time,
            T &triple_kernel_time) {
  switch (tests) {
    case 0:
      return verifyLeastCapacity(single_kernel_time, double_kernel_time,
                                 triple_kernel_time);
    case 1:
      return verifyHalfCapacity(single_kernel_time, double_kernel_time,
                                triple_kernel_time);
    case 2:
      return verifyFullCapacity(single_kernel_time, double_kernel_time,
                                triple_kernel_time);
    default:
      return false;
  }
}

int main(int argc, char** argv) {
  p_tests = 1; // Default for half capacity
  HipTest::parseStandardArguments(argc, argv, true);

  if (p_tests < 0 || p_tests > 2) {
    printf("--tests 0: test least capacity\n");
    printf("        1: test half capacity\n");
    printf("        2: test full capacity\n");
    failed("Wrong p_tests %d\n", p_tests);
  }
  hipError_t err = hipSuccess;
  /*************************************************************************/
  int device_num = 0, loops = 1000;
  bool FailFlag = false;
  /* Create the streams we will use in this test. **************************/
  hipStream_t streams[3];
  // Alocate the host input buffer, and two device-focused buffers that we
  // will use for our test.
  unsigned long long *dev_array[3];
  HIPCHECK(hipGetDeviceCount(&device_num));
  for (int dev = 0; dev < device_num; ++dev) {
    /*************************************************************************/
    HIPCHECK(hipSetDevice(dev));
    hipDeviceProp_t device_properties;
    HIPCHECK(hipGetDeviceProperties(&device_properties, dev));

    /* Test whether target device supports cooperative groups ****************/
    if (device_properties.cooperativeLaunch == 0) {
      std::cout << "Cooperative group support not available in device " << dev << std::endl;
      continue;
    }

    /* We will launch enough waves to fill up all of the GPU *****************/
    int warp_size = device_properties.warpSize;
    int num_sms = device_properties.multiProcessorCount;
    long long totalTicks = device_properties.clockRate ;
    int max_blocks_per_sm = 0;
    // Calculate the device occupancy to know how many blocks can be run.
    HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                          test_kernel,
                                                          warp_size, 0));
    int max_active_blocks = max_blocks_per_sm * num_sms;
    int coop_blocks = 0;
    int reg_blocks  = 0;

    switch (p_tests) {
      case 0:
        // 1 block
        coop_blocks = 1;
        reg_blocks = 1;
        break;
      case 1:
        // Half capacity
        // To make sure the second kernel launched by hipLaunchCooperativeKernel
        // is invoked after the first kernel finished
        coop_blocks = max_active_blocks / 2 + 1;
        // To make sure the third kernel launched by hipLaunchKernelGGL is invoked
        // concurrently with the second kernel
        reg_blocks  = max_active_blocks - coop_blocks;
        break;
      case 2:
        // Full capacity
        coop_blocks = max_active_blocks;
        reg_blocks = max_active_blocks;
        break;
      default:
        failed("wrong p_tests %d", p_tests);
    }
    std::cout << "p_tests: " << p_tests << std::endl;
    std::cout << "Device: " << dev << std::endl;
    std::cout << "Device name: " << device_properties.name << std::endl;
    std::cout << "clockRate: " << device_properties.clockRate << " khz" <<std::endl;
    std::cout << "warp_size: " << device_properties.warpSize << std::endl;
    std::cout << "num_sms: " << device_properties.multiProcessorCount << std::endl;
    std::cout << "max_blocks_per_sm: " << max_blocks_per_sm << std::endl;
    std::cout << "max_active_blocks: " << max_active_blocks << std::endl;
    std::cout << "coop_blocks: " << coop_blocks << std::endl;
    std::cout << "reg_blocks: " << reg_blocks << std::endl;


    /*************************************************************************/
    for (int i = 0; i < 3; i++) {
      HIPCHECK(hipStreamCreate(&streams[i]));
    }

    /*************************************************************************/
    /* Set up data to pass into the kernel ***********************************/

    for (int i = 0; i < 3; i++) {
      HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dev_array[i]),
                         warp_size * sizeof(long long)));
      HIPCHECK(hipMemsetAsync(dev_array[i], 0, warp_size * sizeof(long long),
                              streams[i]));
    }

    HIPCHECK(hipDeviceSynchronize());

    /*************************************************************************/
    /* Launch the kernels ****************************************************/
    void *coop_params[3][3];
    for (int i = 0; i < 3; i++) {
      coop_params[i][0] = reinterpret_cast<void*>(&loops);
      coop_params[i][1] = reinterpret_cast<void*>(&dev_array[i]);
      coop_params[i][2] = reinterpret_cast<void*>(&totalTicks);
    }

    // Verify over capacity
    HIPCHECK_API(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                     max_active_blocks + 1, warp_size,
                                     coop_params[0], 0, streams[0]), hipErrorCooperativeLaunchTooLarge);

    std::cout << "Launching an initial single cooperative kernel..." << std::endl;
    // We need exclude the the initial launching as it will need time to load code obj.
    auto single_start0 = std::chrono::system_clock::now();
    HIPCHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                        max_active_blocks, warp_size,
                                        coop_params[0], 0, streams[0]));
    HIPCHECK(hipDeviceSynchronize());
    auto single_end0 = std::chrono::system_clock::now();

    std::cout << "Launching a single cooperative kernel..." << std::endl;
    auto single_start = std::chrono::system_clock::now();
    HIPCHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                        coop_blocks, warp_size,
                                        coop_params[0], 0, streams[0]));

    HIPCHECK(hipDeviceSynchronize());
    auto single_end = std::chrono::system_clock::now();
    std::cout << "Launching 2 cooperative kernels to different streams...";
    std::cout << std::endl;

    auto double_start = std::chrono::system_clock::now();
    HIPCHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                        coop_blocks, warp_size,
                                        coop_params[0], 0, streams[0]));
    HIPCHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                        coop_blocks, warp_size,
                                        coop_params[1], 0, streams[1]));

    HIPCHECK(hipDeviceSynchronize());
    auto double_end = std::chrono::system_clock::now();
    std::cout << "Launching 2 cooperative kernels and 1 normal kernel...";
    std::cout << std::endl;

    auto triple_start = std::chrono::system_clock::now();
    HIPCHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                        coop_blocks, warp_size,
                                        coop_params[0], 0, streams[0]));
    HIPCHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                        coop_blocks, warp_size,
                                        coop_params[1], 0, streams[1]));
    hipLaunchKernelGGL(test_kernel, dim3(reg_blocks), dim3(warp_size),
                       0, streams[2], loops, dev_array[2], totalTicks);

    HIPCHECK(hipDeviceSynchronize());
    auto triple_end = std::chrono::system_clock::now();
    std::chrono::duration<double> single_kernel_time0 =
                                  (single_end0 - single_start0);
    std::chrono::duration<double> single_kernel_time =
                                  (single_end - single_start);
    std::chrono::duration<double> double_kernel_time =
                                  (double_end - double_start);
    std::chrono::duration<double> triple_kernel_time =
                                  (triple_end - triple_start);

    std::cout << "Initial single kernel took:" << std::endl;
    std::cout << "    " << single_kernel_time0.count();
    std::cout << " seconds" << std::endl;
    std::cout << std::endl;
    std::cout << "A single kernel took:" << std::endl;
    std::cout << "    " << single_kernel_time.count();
    std::cout << " seconds" << std::endl;
    std::cout << std::endl;
    std::cout << "Two cooperative kernels that could run together took:";
    std::cout << std::endl;
    std::cout << "    " << double_kernel_time.count();
    std::cout << " seconds" << std::endl;
    std::cout << std::endl;
    std::cout << "Two coop kernels and a third regular kernel took:";
    std::cout << std::endl << "    ";
    std::cout << triple_kernel_time.count();
    std::cout << " seconds" << std::endl;

    std::cout << "Testing whether these times make sense.." << std::endl;

    FailFlag = !verify(p_tests, single_kernel_time,
                       double_kernel_time, triple_kernel_time);

    for (int k = 0; k < 3; ++k) {
      HIPCHECK(hipFree(dev_array[k]));
      HIPCHECK(hipStreamDestroy(streams[k]));
    }
    if (FailFlag) {
      failed("Failed to exit ");
    }
  }
  passed();
}
