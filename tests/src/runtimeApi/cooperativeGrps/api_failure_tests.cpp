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
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/
// Test Description:
/*The general idea of the application is to test how Cooperative Groups kernel
launches work when launching too many warps to the target device. This test
first queries the nominal warp size of the target device. It then walks through
block sizes from 1 thread, 1 warp, 2 warps, ... `maximum_warps_in_a_block`. For
each of these, it queries the maximum number of blocks that can fit in each SM.
It then queries the number of SMs on the target device. This will yield a
calculation for the maximum number of blocks that can be co-scheduled on this
device.

The Cooperative Groups API says that users should not launch more than this
many warps (or blocks, etc.) to the target device. This test first tires to
launch 2x as many blcoks, to confirm that the runtime prevents such a launch
by returning a proper error value (`hipErrorCooperativeLaunchTooLarge`).

It then ensures that trying to launch too large of a kernel invocation does
not break the GPU by launching a kernel with exactly the maximum number of
blocks.

Finally, we run the same test for a block size that is larger than the maximum
allowed by the device, to ensure that this case is properly detected by the
runtime and that nothing breaks.*/



/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */


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

static inline bool hipCheckExpected(hipError_t errval,
        hipError_t expected_err, const char *file, int line) {
  hipError_t last_err = hipGetLastError();
  if (errval != expected_err) {
    std::cerr << "hip error: " << hipGetErrorString(errval);
    std::cerr << std::endl;
    std::cerr << "    Location: " << file << ":" << line << std::endl;
    return false;
  }
  if (last_err != errval) {
    std::cerr << "Error: the return value of a function was not the same ";
    std::cerr << "as the value returned by hipGetLastError()" << std::endl;
    std::cerr << "    Location: " << file << ":" << line << std::endl;
    std::cerr << "    Function returned: " << hipGetErrorString(errval);
    std::cerr << " (" << errval << ")" << std::endl;
    std::cerr << "hipGetLastError() returned: " << hipGetErrorString(last_err);
    std::cerr << " (" << last_err << ")" << std::endl;
    return false;
  }
  return true;
}

static bool cooperative_groups_support(int device_id) {
  hipError_t err;
  int cooperative_attribute;
  HIPCHECK(hipDeviceGetAttribute(&cooperative_attribute,
           hipDeviceAttributeCooperativeLaunch, device_id));
  if (!cooperative_attribute) {
    std::cerr << "Cooperative launch support not available in ";
    std::cerr << "the device attribute for device " << device_id;
    std::cerr << std::endl;
    return false;
  }
  hipDeviceProp_t device_properties;
  HIPCHECK(hipGetDeviceProperties(&device_properties, device_id));
  if (device_properties.cooperativeLaunch == 0) {
    std::cerr << "Cooperative group support not available in ";
    std::cerr << "device properties." << std::endl;
    return false;
  }
  return true;
}

__global__ void test_kernel(long long *array) {
  unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;
  array[rank] += clock64();
}

int main(int argc, char** argv) {
  hipError_t err;
  int device_num, FailFlag = 0;
  // Alocate the host input buffer, and two device-focused buffers that we
  // will use for our test.
  unsigned int *dev_array[2];
  HIPCHECK(hipGetDeviceCount(&device_num));
  for (int dev = 0; dev < device_num; ++dev) {
    /*************************************************************************/
    /* Test whether target device supports cooperative groups ****************/
    HIPCHECK(hipSetDevice(dev));
    if (!cooperative_groups_support(dev)) {
      std::cout << "Skipping the test with Pass result.\n";
      passed();
    }

    /*************************************************************************/
    /* Create the streams we will use in this test. **************************/
    hipStream_t streams[2];
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipStreamCreate(&streams[i]));
    }

    /*************************************************************************/
    /* We will try to launch more waves than the GPU can fit. ***************/
    hipDeviceProp_t device_properties;
    HIPCHECK(hipGetDeviceProperties(&device_properties, dev));
    int warp_size = device_properties.warpSize;
    int num_sms = device_properties.multiProcessorCount;
    int max_num_threads = device_properties.maxThreadsPerBlock;

    // Check single-thread block, all numbers of warps, then too-large block
    for (int block_size = 0; block_size <= (max_num_threads + warp_size);
         block_size += warp_size) {
      if (block_size == 0) {
        block_size = 1;
      }
      int max_blocks_per_sm;
      // Calculate the device occupancy to know how many blocks can be run.
      HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
              &max_blocks_per_sm, test_kernel, block_size, 0,
              hipOccupancyDefault));

      if ((block_size > max_num_threads) && (max_blocks_per_sm != 0)) {
        std::cerr << "ERROR! Occupancy API indicated that we can have >0 ";
        std::cerr << "blocks in a kernel when the block size is too large ";
        std::cerr << "to work on the device." << std::endl;
        std::cerr << "This is incorrect, and could possibly lead users ";
        std::cerr << "to try to launch kernels that will fail." << std::endl;
        //failed("");
        FailFlag = 1;
        break;
      }

      int desired_blocks = max_blocks_per_sm * num_sms;
      bool expect_fail = false;
      if (desired_blocks == 0) {
        desired_blocks = 1;
        expect_fail = true;
      }

      /**********************************************************************/
      /* Set up data to pass into the kernel ********************************/

      for (int i = 0; i < 2; i++) {
        int test_size;
        // Case where we expect to fail at launch.
        if (i == 0) {
          test_size = 2 * desired_blocks;
        } else {
          test_size = desired_blocks;
        }
        HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dev_array[i]),
                           test_size * block_size * sizeof(long long)));
        HIPCHECK(hipMemsetAsync(dev_array[i], 0,
                                test_size * block_size * sizeof(long long),
                                streams[i]));
      }

      HIPCHECK(hipDeviceSynchronize());

      /***********************************************************************/
      /* Launch the kernels **************************************************/
      void *coop_params[2][1];
      for (int i = 0; i < 2; i++) {
        coop_params[i][0] = reinterpret_cast<void*>(&dev_array[i]);
      }

      err = hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                       2 * desired_blocks, block_size,
                                       coop_params[0], 0, streams[0]);

      hipError_t expect_to_see;
      if (expect_fail) {
        expect_to_see = hipErrorInvalidConfiguration;
      } else {
        expect_to_see = hipErrorCooperativeLaunchTooLarge;
      }
      if (!hipCheckExpected(err, expect_to_see, __FILE__, __LINE__)) {
        std::cerr << "ERROR! Tried to launch a cooperative kernel with ";
        std::cerr << "too many warps." << std::endl;
        std::cerr << "This SHOULD have failed with the error ";
        std::cerr << hipGetErrorString(expect_to_see);
        std::cerr << " (" << expect_to_see << ")." << std::endl;
        std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
        std::cerr << " (" << err << ")" << std::endl;
        FailFlag = 1;
        break;
      }

      HIPCHECK(hipDeviceSynchronize());
      err = hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                       desired_blocks, block_size,
                                       coop_params[1], 0, streams[1]);

      if (expect_fail) {
        expect_to_see = hipErrorInvalidConfiguration;
      } else {
        expect_to_see = hipSuccess;
      }
      if (!hipCheckExpected(err, expect_to_see, __FILE__, __LINE__)) {
        std::cerr << "ERROR! Tried to launch a cooperative kernel ";
        std::cerr << "with a normal size, but a block size of ";
        std::cerr << desired_blocks << std::endl;
        std::cerr << "This SHOULD have returned ";
        std::cerr << hipGetErrorString(expect_to_see);
        std::cerr << " (" << expect_to_see << ")." << std::endl;
        std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
        std::cerr << " (" << err << ")" << std::endl;
        FailFlag = 1;
        break;
      }

      HIPCHECK(hipDeviceSynchronize());

      if (block_size == 1) {
        block_size = 0;
      }
      for (int m = 0; m < 2; ++m) {
        HIPCHECK(hipFree(dev_array[m]));
      }
    }
    for (int m = 0; m < 2; ++m) {
      HIPCHECK(hipStreamDestroy(streams[m]));
    }
    if (FailFlag == 1) {
      for (int m = 0; m < 2; ++m) {
        HIPCHECK(hipFree(dev_array[m]));
      }
      failed("");
    }
  }
  passed();
}
