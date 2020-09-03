/*
Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
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
/*The general idea of the application is to test how multi-GPU Cooperative
Groups kernel launches to a stream interact with other things that may be
simultaneously running in the same streams.

The HIP specification says that a multi-GPU cooperative launch will wait
until all of the streams it's using finish their work. Only then will the
cooperative kernel be launched to all of the devices. Then no other work
can take part in the any of the streams until all of the multi-GPU
cooperative work is done.

However, there are flags that allow you to disable each of these
serialization points: hipCooperativeLaunchMultiDeviceNoPreSync and
hipCooperativeLaunchMultiDeviceNoPostSync.

As such, this benchmark tests the following five situations launching
to two GPUs (and thus two streams):

    1. Normal multi-GPU cooperative kernel:
        This should result in the following pattern:
        Stream 0: Cooperative
        Stream 1: Cooperative
    2. Regular kernel launches and multi-GPU cooperative kernel launches
       with the default flags, resulting in the following pattern:
        Stream 0: Regular --> Cooperative
        Stream 1:         --> Cooperative --> Regular

    3. Regular kernel launches and multi-GPU cooperative kernel launches
       that turn off "pre-sync". This should allow a cooperative kernel
       to launch even if work is already in a stream pointing to
       another GPU.
        This should result in the following pattern:
        Stream 0: Regular --> Cooperative
        Stream 1: Cooperative            --> Regular

    4. Regular kernel launches and multi-GPU cooperative kernel launches
       that turn off "post-sync". This should allow a new kernel to enter
       a GPU even if another GPU still has a cooperative kernel on it.
        This should result in the following pattern:
        Stream 0: Regular --> Cooperative
        Stream 1:         --> Cooperative--> Regular

    5. Regular kernel launches and multi-GPU cooperative kernel launches
       that turn off both pre- and post-sync. This should allow any of
       the kernels to launch to their GPU regardless of the status of
       other kernels in other multi-GPU stream groups.
        This should result in the following pattern:
        Stream 0: Regular --> Cooperative
        Stream 1: Cooperative --> Regular

We time how long it takes to run each of these benchmarks and print it as
the output of the benchmark. The kernels themselves are just useless time-
wasting code so that the kernel takes a meaningful amount of time on the
GPU before it exits. We only launch a single wavefront for each kernel, so
any serialization should not be because of GPU occupancy concerns.

If tests 2, 3, and 4 take roughly 3x as long as #1, that implies that
cooperative kernels are serialized as expected.

If test #5 takes roughly twice as long as #1, that implies that the
overlap-allowing flags work as expected.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
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
    std::cerr << "    Location: " << file << ":" << line << std::endl;
    failed("");
  }
  if (last_err != errval) {
    std::cerr << "Error: the return value of a function was not the same ";
    std::cerr << "as the value returned by hipGetLastError()" << std::endl;
    std::cerr << "    Location: " << file << ":" << line << std::endl;
    std::cerr << "    Function returned: " << hipGetErrorString(errval);
    std::cerr << " (" << errval << ")" << std::endl;
    std::cerr << "hipGetLastError() returned: " << hipGetErrorString(last_err);
    std::cerr << " (" << last_err << ")" << std::endl;
    failed("");
  }
}
#define hipCheckErr(errval) \
  do { hipCheckAndFail((errval), __FILE__, __LINE__); } while (0)

static int cooperative_groups_support(int device_id) {
  hipError_t err;
  int cooperative_attribute;
  HIPCHECK(hipDeviceGetAttribute(&cooperative_attribute,
          hipDeviceAttributeCooperativeLaunch, device_id));
  if (!cooperative_attribute) {
    std::cerr << "Cooperative launch support not available in ";
    std::cerr << "the device attribute for device " << device_id;
    std::cerr << std::endl;
    return 0;
  }

  int multi_gpu_cooperative_attribute;
  HIPCHECK(hipDeviceGetAttribute(&multi_gpu_cooperative_attribute,
           hipDeviceAttributeCooperativeMultiDeviceLaunch, device_id));
  if (!multi_gpu_cooperative_attribute) {
    std::cerr << "Multi-GPU cooperative launch support not available in ";
    std::cerr << "the device attribute for device " << device_id;
    std::cerr << std::endl;
    return 0;
  }

  hipDeviceProp_t device_properties;
  HIPCHECK(hipGetDeviceProperties(&device_properties, device_id));
  if (device_properties.cooperativeLaunch == 0) {
    std::cerr << "Cooperative group support not available in ";
    std::cerr << "device properties." << std::endl;
    return 0;
  }
  if (device_properties.cooperativeMultiDeviceLaunch == 0) {
    std::cerr << "Multi-GPU cooperative group support not available in ";
    std::cerr << "device properties." << std::endl;
    return 0;
  }
  return 1;
}

__global__ void test_coop_kernel(unsigned int loops, long long *array,
                                 int fast_gpu) {
  cooperative_groups::multi_grid_group mgrid =
  cooperative_groups::this_multi_grid();
  unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;

  if (mgrid.grid_rank() == fast_gpu) {
    return;
  }

  for (int i = 0; i < loops; i++) {
    long long start_clock = clock64();
    while (clock64() < (start_clock+1000000)) {}
    array[rank] += clock64();
  }
}

__global__ void test_kernel(uint32_t loops, unsigned long long *array) {
  unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < loops; i++) {
    long long start_clock = clock64();
    while (clock64() < (start_clock+1000000)) {}
    array[rank] += clock64();
  }
}

int main(int argc, char** argv) {
  hipError_t err;
  int device_num, FailFlag = 0;
  uint32_t loops = 2000;
  uint32_t fast_loops = 1;
  int32_t fast_gpu = -1;
  HIPCHECK(hipGetDeviceCount(&device_num));
  if (device_num < 2) {
    std::cout << "This test requires atleast two gpus but the system has ";
    std::cout << " only "<< device_num <<std::endl;
    std::cout << "The test is skipping with Pass result" << std::endl;
    passed();
  }
  for (int dev = 0; dev < (device_num-1); ++dev) {
    std::cout << "First device number: " << dev << std::endl;
    std::cout << "Second device number: " << (dev + 1) << std::endl;
    std::cout << "Loops: " << loops << std::endl;

    /*************************************************************************/
    /* Test whether target devices support cooperative groups ****************/
    for (int i = 0; i < 2; i++) {
      if (!cooperative_groups_support(dev + i)) {
        std::cout << "Skipping the test with Pass result.\n";
        passed();
      }
    }

    /*************************************************************************/
    /* We will launch enough waves to fill up all of the GPU *****************/
    int warp_sizes[2];
    int num_sms[2];
    hipDeviceProp_t device_properties[2];
    int warp_size = INT_MAX;
    int num_sm = INT_MAX;
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipGetDeviceProperties(&device_properties[i], (dev + i)));
      warp_sizes[i] = device_properties[i].warpSize;
      if (warp_sizes[i] < warp_size) {
        warp_size = warp_sizes[i];
      }
      num_sms[i] = device_properties[i].multiProcessorCount;
      if (num_sms[i] < num_sm) {
        num_sm = num_sms[i];
      }
      std::cout << "Device " << (i + 1);
      std::cout << " name: " << device_properties[i].name << std::endl;
    }
    std::cout << std::endl;

    // Calculate the device occupancy to know how many blocks can be run.
    int max_blocks_per_sm_arr[2];
    int max_blocks_per_sm = INT_MAX;
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipSetDevice(dev + i));
      HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
               &max_blocks_per_sm_arr[i], test_kernel, warp_size, 0));
      if (max_blocks_per_sm_arr[i] < max_blocks_per_sm) {
        max_blocks_per_sm = max_blocks_per_sm_arr[i];
      }
    }
    int desired_blocks = 1;

    if (desired_blocks > max_blocks_per_sm * num_sm) {
      std::cerr << "The requested number of blocks will not fit on the GPU";
      std::cerr << std::endl;
      std::cerr << "You requested " << desired_blocks << " but we can only ";
      std::cerr << "fit " << (max_blocks_per_sm * num_sm) << std::endl;
      failed("");
    }

    /*************************************************************************/
    /* Create the streams we will use in this test. **************************/
    hipStream_t streams[2];
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipSetDevice(dev + i));
      HIPCHECK(hipStreamCreate(&streams[i]));
    }

    /*************************************************************************/
    /* Set up data to pass into the kernelx **********************************/

    // Alocate the host input buffer, and two device-focused buffers that we
    // will use for our test.
    unsigned long long *dev_array[2];
    for (int i = 0; i < 2; i++) {
      int good_size = desired_blocks * warp_size * sizeof(long long);
      HIPCHECK(hipSetDevice(dev + i));
      HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dev_array[i]), good_size));
      HIPCHECK(hipMemsetAsync(dev_array[i], 0, good_size, streams[i]));
    }
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipSetDevice(dev + i));
      HIPCHECK(hipDeviceSynchronize());
    }

    /*************************************************************************/
    /* Launch the kernels ****************************************************/
    void *dev_params[2][3];
    hipLaunchParams md_params[2];
    std::chrono::time_point<std::chrono::system_clock> start_time[6];
    std::chrono::time_point<std::chrono::system_clock> end_time[6];

    std::cout << "Test 0: Launching a multi-GPU cooperative kernel...\n";
    std::cout << "This should result in the following pattern:" << std::endl;
    std::cout << "GPU " << dev << ": Long Coop Kernel" << std::endl;
    std::cout << "GPU " << (dev + 1) << ": Long Coop Kernel" << std::endl;

    for (int i = 0; i < 2; i++) {
      dev_params[i][0] = reinterpret_cast<void*>(&loops);
      dev_params[i][1] = reinterpret_cast<void*>(&dev_array[i]);
      dev_params[i][2] = reinterpret_cast<void*>(&fast_gpu);
      md_params[i].func = reinterpret_cast<void*>(test_coop_kernel);
      md_params[i].gridDim = desired_blocks;
      md_params[i].blockDim = warp_size;
      md_params[i].sharedMem = 0;
      md_params[i].stream = streams[i];
      md_params[i].args = dev_params[i];
    }

    start_time[0] = std::chrono::system_clock::now();
    HIPCHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0));
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipSetDevice(dev + i));
      HIPCHECK(hipDeviceSynchronize());
    }
    end_time[0] = std::chrono::system_clock::now();

    std::cout << std::endl;
    std::cout << "Test 1: Launching a multi-GPU cooperative kernel with the ";
    std::cout << "following pattern:" << std::endl;
    std::cout << "GPU " << dev << ": Standard  Kernel --> Long Coop Kernel\n";
    std::cout << "GPU " << (dev + 1) << ":                  --> Coop        ";
    std::cout << "--> Standard  Kernel\n";
    fast_gpu = 1;
    start_time[1] = std::chrono::system_clock::now();
    HIPCHECK(hipSetDevice(dev));
    hipLaunchKernelGGL(test_kernel, dim3(desired_blocks), dim3(warp_size), 0,
                       streams[0], loops, dev_array[0]);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0));
    HIPCHECK(hipSetDevice(dev + 1));
    hipLaunchKernelGGL(test_kernel, dim3(desired_blocks), dim3(warp_size), 0,
                       streams[1], loops, dev_array[1]);
    HIPCHECK(hipGetLastError());
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipSetDevice(dev + i));
      HIPCHECK(hipDeviceSynchronize());
    }
    end_time[1] = std::chrono::system_clock::now();
    fast_gpu = -1;

    std::cout << std::endl;
    std::cout << "Test 2: Launching a multi-GPU cooperative kernel with the ";
    std::cout << "following pattern:" << std::endl;
    std::cout << "GPU " << dev << ": Standard  Kernel --> Coop" << std::endl;
    std::cout << "GPU " << (dev + 1) << ":                  --> Long Coop";
    std::cout << " Kernel --> ";
    std::cout << "Standard  Kernel\n";
    fast_gpu = 0;
    start_time[2] = std::chrono::system_clock::now();
    HIPCHECK(hipSetDevice(dev));
    hipLaunchKernelGGL(test_kernel, dim3(desired_blocks), dim3(warp_size), 0,
                       streams[0], loops, dev_array[0]);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0));
    HIPCHECK(hipSetDevice(dev + 1));
    hipLaunchKernelGGL(test_kernel, dim3(desired_blocks), dim3(warp_size), 0,
                       streams[1], loops, dev_array[1]);
    HIPCHECK(hipGetLastError());
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipSetDevice(dev + i));
      HIPCHECK(hipDeviceSynchronize());
    }
    end_time[2] = std::chrono::system_clock::now();
    fast_gpu = -1;

    std::cout << std::endl;
    std::cout << "Test 3: Launching a multi-GPU cooperative kernel with the ";
    std::cout << "ability to overlap regular and cooperative kernels ";
    std::cout << "only at the beginning." << std::endl;
    std::cout << "This should result in the following pattern:" << std::endl;
    std::cout << "GPU " << dev << ": Standard  Kernel --> Coop" << std::endl;
    std::cout << "GPU " << (dev + 1) << ": Long Coop Kernel -->      Standard";
    std::cout<< "  Kernel\n";
    fast_gpu = 0;
    start_time[3] = std::chrono::system_clock::now();
    HIPCHECK(hipSetDevice(dev));
    hipLaunchKernelGGL(test_kernel, dim3(desired_blocks), dim3(warp_size), 0,
                       streams[0], loops, dev_array[0]);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2,
             hipCooperativeLaunchMultiDeviceNoPreSync));
    HIPCHECK(hipSetDevice(dev + 1));
    hipLaunchKernelGGL(test_kernel, dim3(desired_blocks), dim3(warp_size), 0,
                       streams[1], loops, dev_array[1]);
    HIPCHECK(hipGetLastError());
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipSetDevice(dev + i));
      HIPCHECK(hipDeviceSynchronize());
    }
    end_time[3] = std::chrono::system_clock::now();
    fast_gpu = -1;

    std::cout << std::endl;
    std::cout << "Test 4: Launching a multi-GPU cooperative kernel with the ";
    std::cout << "ability to overlap regular and cooperative kernels ";
    std::cout << "only at the end." << std::endl;
    std::cout << "This should result in the following pattern:" << std::endl;
    std::cout << "GPU " << dev << ": Standard  Kernel --> Long Coop Kernel\n";
    std::cout << "GPU " << (dev + 1) << ":                  --> Coop --> ";
    std::cout << "Standard  Kernel\n";
    fast_gpu = 1;
    start_time[4] = std::chrono::system_clock::now();
    HIPCHECK(hipSetDevice(dev));
    hipLaunchKernelGGL(test_kernel, dim3(desired_blocks), dim3(warp_size), 0,
                       streams[0], loops, dev_array[0]);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2,
             hipCooperativeLaunchMultiDeviceNoPostSync));
    HIPCHECK(hipSetDevice(dev + 1));
    hipLaunchKernelGGL(test_kernel, dim3(desired_blocks), dim3(warp_size), 0,
                       streams[1], loops, dev_array[1]);
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipSetDevice(dev + i));
      HIPCHECK(hipDeviceSynchronize());
    }
    end_time[4] = std::chrono::system_clock::now();
    fast_gpu = -1;

    std::cout << std::endl;
    std::cout << "Test 5: Launching a multi-GPU cooperative kernel with the ";
    std::cout << "ability to overlap regular and cooperative kernels";
    std::cout << std::endl;
    std::cout << "This should result in the following pattern:" << std::endl;
    std::cout << "GPU " << dev << ": Standard  Kernel --> Long Coop Kernel\n";
    std::cout << "GPU " << (dev + 1) << ": Long Coop Kernel --> Standard";
    std::cout << "  Kernel\n";
    start_time[5] = std::chrono::system_clock::now();
    HIPCHECK(hipSetDevice(dev));
    hipLaunchKernelGGL(test_kernel, dim3(desired_blocks), dim3(warp_size), 0,
                       streams[0], loops, dev_array[0]);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2,
             hipCooperativeLaunchMultiDeviceNoPreSync |
             hipCooperativeLaunchMultiDeviceNoPostSync));
    HIPCHECK(hipSetDevice(dev + 1));
    hipLaunchKernelGGL(test_kernel, dim3(desired_blocks), dim3(warp_size), 0,
                       streams[1], loops, dev_array[1]);
    HIPCHECK(hipGetLastError());
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipSetDevice(dev + i));
      HIPCHECK(hipDeviceSynchronize());
    }
    end_time[5] = std::chrono::system_clock::now();

    std::chrono::duration<double> single_kernel_time =
                                  (end_time[0] - start_time[0]);
    std::chrono::duration<double> serialized_gpu0_time =
                                  (end_time[1] - start_time[1]);
    std::chrono::duration<double> serialized_gpu1_time =
                                  (end_time[2] - start_time[2]);
    std::chrono::duration<double> pre_overlapped_time =
                                  (end_time[3] - start_time[3]);
    std::chrono::duration<double> post_overlapped_time =
                                  (end_time[4] - start_time[4]);
    std::chrono::duration<double> overlapped_time =
                                  (end_time[5] - start_time[5]);

    std::cout << "Test 0: A single kernel on both GPUs took:" << std::endl;
    std::cout << "    " << single_kernel_time.count();
    std::cout << " seconds" << std::endl;
    std::cout << std::endl;
    std::cout << "Test 1: Serialized set of three kernels with GPU0";
    std::cout << " being long took:";
    std::cout << "    " << serialized_gpu0_time.count();
    std::cout << " seconds" << std::endl;
    std::cerr << "Expect between " << (2.7 * single_kernel_time.count());
    std::cerr << " and ";
    std::cerr << (3.3 * single_kernel_time.count()) << " seconds.\n";
    std::cout << std::endl;
    std::cout << "Test 2: Serialized set of three kernels with GPU1";
    std::cout << " being long took:" << std::endl;
    std::cout << "    " << serialized_gpu1_time.count();
    std::cout << " seconds" << std::endl;
    std::cerr << "Expect between " << (2.7 * single_kernel_time.count());
    std::cerr << " and ";
    std::cerr << (3.3 * single_kernel_time.count()) << " seconds.\n";
    std::cout << std::endl;
    std::cout << "Test 3: Multiple kernels with pre-overlap allowed took:\n";
    std::cout << "    " << pre_overlapped_time.count();
    std::cout << " seconds" << std::endl;
    std::cerr << "Expect between " << (1.7 * single_kernel_time.count());
    std::cerr << " and ";
    std::cerr << (2.3 * single_kernel_time.count()) << " seconds.\n";
    std::cout << std::endl;
    std::cout << "Test 4: Multiple kernels with post-overlap allowed took:\n";
    std::cout << "    " << post_overlapped_time.count();
    std::cout << " seconds" << std::endl;
    std::cerr << "Expect between " << (1.7 * single_kernel_time.count());
    std::cerr << " and ";
    std::cerr << (2.3 * single_kernel_time.count()) << " seconds.";
    std::cout << std::endl;
    std::cout << "Test 5: Multiple kernels with overlap allowed took:\n";
    std::cout << "    " << overlapped_time.count();
    std::cout << " seconds" << std::endl;
    std::cerr << "Expect between " << (1.8 * single_kernel_time.count());
    std::cerr << " and ";
    std::cerr << (2.2 * single_kernel_time.count()) << " seconds.\n";

    // Test that fully not-overlapped kernels take roughly 3x as long as one
    // cooperative kernel.
    if (serialized_gpu0_time > 3.3 * single_kernel_time ||
        serialized_gpu0_time < 2.7 * single_kernel_time) {
      std::cerr << "ERROR!" << std::endl;
      std::cerr << "Test 1, the first case where all kernels should be ";
      std::cerr << "serialized, had a runtime that was very different ";
      std::cerr << "than what was expected." << std::endl;
      std::cerr << "Was " << serialized_gpu0_time.count() << " seconds.\n";
      std::cerr << "Expected between ";
      std::cerr << (2.7 * single_kernel_time.count()) << " and ";
      std::cerr << (3.3 * single_kernel_time.count()) << " seconds.\n";
      std::cerr << "Were they truly serialized?" << std::endl;
      FailFlag = 1;
    }

    // Test that fully not-overlapped kernels take roughly 3x as long as one
    // cooperative kernel.
    if (serialized_gpu1_time > 3.3 * single_kernel_time ||
        serialized_gpu1_time < 2.7 * single_kernel_time) {
      std::cerr << "ERROR!" << std::endl;
      std::cerr << "Test 2, the second case where all kernels should be ";
      std::cerr << "serialized, had a runtime that was very different ";
      std::cerr << "than what was expected." << std::endl;
      std::cerr << "Was " << serialized_gpu1_time.count();
      std::cerr << " seconds." << std::endl;
      std::cerr << "Expected between ";
      std::cerr << (2.7 * single_kernel_time.count()) << " and ";
      std::cerr << (3.3 * single_kernel_time.count()) << " seconds.\n";
      std::cerr << "Were they truly serialized?" << std::endl;
      FailFlag = 1;
    }

    // Test that kernels that can overlap only before the cooperative kernel
    // launches kernels take roughly the same time (in this case)
    if (pre_overlapped_time > 2.3 * single_kernel_time ||
        pre_overlapped_time < 1.7 * single_kernel_time) {
      std::cerr << "ERROR!" << std::endl;
      std::cerr << "Test 3, the case where the last kernel is serialized, had ";
      std::cerr << "a runtime that was very different than what was ";
      std::cerr << "expected." << std::endl;
      std::cerr << "Was " << pre_overlapped_time.count() << " seconds.\n";
      std::cerr << "Expected between ";
      std::cerr << (1.7 * single_kernel_time.count()) << " and ";
      std::cerr << (2.3 * single_kernel_time.count()) << " seconds.\n";
      FailFlag = 1;
    }

    // Test that kernels that can overlap only after the cooperative kernel
    // launches kernels take roughly the same time (in this case)
    if (post_overlapped_time > 2.3 * single_kernel_time ||
        post_overlapped_time < 1.7 * single_kernel_time) {
      std::cerr << "ERROR!" << std::endl;
      std::cerr << "Teste 4, the case where the first kernel is ";
      std::cerr << "serialized, had a runtime that was very different ";
      std::cerr << "than what was expected." << std::endl;
      std::cerr << "Was " << post_overlapped_time.count() << " seconds.\n";
      std::cerr << "Expected between ";
      std::cerr << (1.7 * single_kernel_time.count()) << " and ";
      std::cerr << (2.3 * single_kernel_time.count()) << " seconds.\n";
      FailFlag = 1;
    }

    // Test that, with the right flags on the kernel launch, that we prevent
    // incomplete launches from serializing the cooperative launch streams.
    if (overlapped_time > 2.2 * single_kernel_time ||
        overlapped_time < 1.8 * single_kernel_time) {
      std::cerr << "ERROR!" << std::endl;
      std::cerr << "Test 5, the case where normal and cooperative kernel ";
      std::cerr << "launches should overlap, does not appear to have done so.";
      std::cerr << std::endl;
      std::cerr << "Was " << overlapped_time.count() << " seconds.\n";
      std::cerr << "Expected between ";
      std::cerr << (1.8 * single_kernel_time.count()) << " and ";
      std::cerr << (2.2 * single_kernel_time.count()) << " seconds.\n";
      std::cerr << "Is the normal kernel being serialized with the ";
      std::cerr << "cooperative kernels on different streams?" << std::endl;
      FailFlag = 1;
    }
    for (int k = 0; k < 2; ++k) {
      HIPCHECK(hipFree(dev_array[k]));
      HIPCHECK(hipStreamDestroy(streams[k]));
    }
    if (FailFlag == 1) {
      break;
    }
  }
  if (FailFlag == 1) {
    failed("");
  } else {
    passed();
  }
}
