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
/*The general idea of the application is to test how Cooperative Groups kernel
launches work when launching too many warps to multiple target devices. This
tests the following failure modes for hipLaunchCooperativeKernelMultiDevice:
  1) Do not launch more warps to any device than can fit on that device
  2) All device targets for the multi-device launch function must be different
  3) All streams must be explicit (non-NULL)
  4) The kernels sent in must be identical between devices
  5) The grid and block sizes must be identical between devices
  6) The block dimensions must be non-zero
  7) The dynamic shared memory size must be identical between devices.

This test ensures that the proper error conditions are returned, even if the
target kernel does not actually use any fo the cooperative groups features.

Note that tests 4, 5, and 7 only hold on Nvidia GPUs. AMD GPUs running ROCm
do not have these constraints. As such, the test checks to see whether they
should fail or succeed and compares this to what actually happens.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
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

static int support_for_separate_kernels(int device_id) {
  hipError_t err;

  int separate_kernel_supported;
  HIPCHECK(hipDeviceGetAttribute(&separate_kernel_supported,
           hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
           device_id));
  if (!separate_kernel_supported) {
    return 0;
  }

  hipDeviceProp_t device_properties;
  HIPCHECK(hipGetDeviceProperties(&device_properties, device_id));
  if (device_properties.cooperativeMultiDeviceUnmatchedFunc == 0) {
    return 0;
  }
  return 1;
}

static int support_for_separate_grid_sizes(int device_id) {
  hipError_t err;
  int separate_sizes_supported;
  HIPCHECK(hipDeviceGetAttribute(&separate_sizes_supported,
           hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
           device_id));
  if (!separate_sizes_supported) {
    return 0;
  }

  hipDeviceProp_t device_properties;
  HIPCHECK(hipGetDeviceProperties(&device_properties, device_id));
  if (device_properties.cooperativeMultiDeviceUnmatchedGridDim == 0) {
    return 0;
  }
  return 1;
}

static int support_for_separate_block_dims(int device_id) {
  hipError_t err;
  int separate_sizes_supported;
  HIPCHECK(hipDeviceGetAttribute(&separate_sizes_supported,
           hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
           device_id));
  if (!separate_sizes_supported) {
    return 0;
  }

  hipDeviceProp_t device_properties;
  HIPCHECK(hipGetDeviceProperties(&device_properties, device_id));
  if (device_properties.cooperativeMultiDeviceUnmatchedBlockDim == 0) {
    return 0;
  }
  return 1;
}

static int support_for_separate_shared_sizes(int device_id) {
  hipError_t err;
  int separate_sizes_supported;
  HIPCHECK(hipDeviceGetAttribute(&separate_sizes_supported,
           hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
           device_id));
  if (!separate_sizes_supported) {
    return 0;
  }

  hipDeviceProp_t device_properties;
  HIPCHECK(hipGetDeviceProperties(&device_properties, device_id));
  if (device_properties.cooperativeMultiDeviceUnmatchedSharedMem == 0) {
    return 0;
  }
  return 1;
}

__global__ void test_kernel(long long *array) {
    unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;
    array[rank] += clock64();
}

__global__ void second_test_kernel(long long *array) {
    unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;
    array[rank] += clock64();
}

int main(int argc, char** argv) {
  hipError_t err;
  /*************************************************************************/
  /* Parse the command line parameters *************************************/
  // Arguments to pull out of the command line.
  int device_num, FailFlag = 0;
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

    /*************************************************************************/
    /* Test whether target devices support cooperative groups ****************/
    for (int i = 0; i < 2; i++) {
       if (!cooperative_groups_support((dev + i))) {
         std::cout << "Skipping the test with Pass result.\n";
         passed();
        }
    }

    /*************************************************************************/
    /* We will try to launch more waves than the GPUs can fit. ***************/
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
      std::cout << "Device " << (dev + i);
      std::cout << " name: " << device_properties[i].name << std::endl;
    }
    std::cout << std::endl;

    // Calculate the device occupancy to know how many blocks can be run.
    int max_blocks_per_sm_arr[2];
    int max_blocks_per_sm = INT_MAX;
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipSetDevice((dev + i)));
      HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
               &max_blocks_per_sm_arr[i], test_kernel, warp_size, 0));
      if (max_blocks_per_sm_arr[i] < max_blocks_per_sm) {
          max_blocks_per_sm = max_blocks_per_sm_arr[i];
      }
    }

    int desired_blocks = max_blocks_per_sm * num_sm;

    /*************************************************************************/
    /* Create the streams we will use in this test. **************************/
    hipStream_t streams[2];
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipSetDevice((dev + i)));
      HIPCHECK(hipStreamCreate(&streams[i]));
    }

    /*************************************************************************/
    /* Set up data to pass into the kernel ***********************************/

    // Alocate the host input buffer, and two device-focused buffers per GPU
    // that we will use for our test.
    unsigned int *good_dev_array[2];
    unsigned int *bad_dev_array[2];
    for (int i = 0; i < 2; i++) {
      int good_size = desired_blocks * warp_size * sizeof(long long);
      int bad_size = 2 * desired_blocks * warp_size * sizeof(long long);

      HIPCHECK(hipSetDevice((dev + i)));
      HIPCHECK(hipMalloc(reinterpret_cast<void**>(&good_dev_array[i]),
                         good_size));
      HIPCHECK(hipMemsetAsync(good_dev_array[i], 0, good_size, streams[i]));
      HIPCHECK(hipMalloc(reinterpret_cast<void**>(&bad_dev_array[i]),
                         bad_size));
      HIPCHECK(hipMemsetAsync(bad_dev_array[i], 0, bad_size, streams[i]));
    }
    HIPCHECK(hipDeviceSynchronize());

    /*************************************************************************/
    /* Launch the kernels ****************************************************/
    std::cout << "Launching a multi-GPU cooperative kernel with too many ";
    std::cout << "warps..." << std::endl;

    void *dev_params[2][1];
    hipLaunchParams md_params[2];
    for (int i = 0; i < 2; i++) {
      dev_params[i][0] = reinterpret_cast<void*>(&bad_dev_array[i]);

      md_params[i].func = reinterpret_cast<void*>(test_kernel);
      md_params[i].gridDim = 2 * desired_blocks;
      md_params[i].blockDim = warp_size;
      md_params[i].sharedMem = 0;
      md_params[i].stream = streams[i];
      md_params[i].args = dev_params[i];
    }

    err = hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0);
    if (err != hipErrorCooperativeLaunchTooLarge) {
      std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
      std::cerr << "with too many warps." << std::endl;
      std::cerr << "This SHOULD have failed with the error ";
      std::cerr << "hipErrorCooperativeLaunchTooLarge (";
      std::cerr << hipErrorCooperativeLaunchTooLarge << ")." << std::endl;
      std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
      std::cerr << " (" << err << ")" << std::endl;
      FailFlag = 1;
    } else {
      std::cout << "\tProperly saw this return ";
      std::cout << "hipErrorCooperativeLaunchTooLarge" << std::endl;
    }
    HIPCHECK(hipDeviceSynchronize());

    std::cout << "Launching a multi-GPU cooperative kernel to the same ";
    std::cout << "device twice..." << std::endl;
    for (int i = 0; i < 2; i++) {
      dev_params[i][0] = reinterpret_cast<void*>(&good_dev_array[i]);
      md_params[i].gridDim = desired_blocks;
      md_params[i].stream = streams[0];
    }
    err = hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0);
    if (err != hipErrorInvalidDevice) {
      std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
      std::cerr << "to the same device twice." << std::endl;
      std::cerr << "This SHOULD have failed with the error ";
      std::cerr << "hipErrorInvalidDevice (";
      std::cerr << hipErrorInvalidDevice << ")." << std::endl;
      std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
      std::cerr << " (" << err << ")" << std::endl;
      FailFlag = 1;
    } else {
      std::cout << "\tProperly saw this return ";
      std::cout << "hipErrorInvalidDevice" << std::endl;
    }
    HIPCHECK(hipDeviceSynchronize());

    std::cout << "Launching a multi-GPU cooperative kernel to the NULL ";
    std::cout << "stream" << std::endl;
    for (int i = 0; i < 2; i++) {
      md_params[i].stream = NULL;
    }
    err = hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0);
    if (err != hipErrorInvalidResourceHandle) {
      std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
      std::cerr << "to the NULL stream." << std::endl;
      std::cerr << "This SHOULD have failed with the error ";
      std::cerr << "hipErrorInvalidResourceHandle (";
      std::cerr << hipErrorInvalidResourceHandle << ")." << std::endl;
      std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
      std::cerr << " (" << err << ")" << std::endl;
      FailFlag = 1;
    } else {
      std::cout << "\tProperly saw this return ";
      std::cout << "hipErrorInvalidResourceHandle" << std::endl;
    }
    HIPCHECK(hipDeviceSynchronize());
#ifndef __HIP_PLATFORM_NVIDIA__
    std::cout << "Launching a multi-GPU cooperative kernel with two ";
    std::cout << "different kernels." << std::endl;
    bool supports_sep_kernels = true;
    for (int i = 0; i < 2; i++) {
      md_params[i].stream = streams[i];
      if (!support_for_separate_kernels((dev + i))) {
        supports_sep_kernels = false;
      }
    }
    md_params[1].func = reinterpret_cast<void*>(second_test_kernel);
    err = hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0);
    if ((supports_sep_kernels && err != hipSuccess) ||
        (!supports_sep_kernels && err != hipErrorInvalidValue)) {
      if (supports_sep_kernels) {
        std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
        std::cerr << "with two different kernels." << std::endl;
        std::cerr << "This SHOULD have succeeded with hipSuccess (";
        std::cerr << hipSuccess << ")." << std::endl;
        std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
        std::cerr << " (" << err << ")" << std::endl;
      } else {
        std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
        std::cerr << "with two different kernels." << std::endl;
        std::cerr << "This SHOULD have failed with the error ";
        std::cerr << "hipErrorInvalidValue (";
        std::cerr << hipErrorInvalidValue << ")." << std::endl;
        std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
        std::cerr << " (" << err << ")" << std::endl;
      }
      FailFlag = 1;
    } else {
      std::cout << "\tProperly saw this return ";
      if (supports_sep_kernels) {
        std::cout << "hipSuccess" << std::endl;
      } else {
        std::cout << "hipErrorInvalidValue" << std::endl;
      }
    }
    HIPCHECK(hipDeviceSynchronize());

    std::cout << "Launching a multi-GPU cooperative kernel with two ";
    std::cout << "different grid sizes." << std::endl;
    bool supports_sep_sizes = true;
    for (int i = 0; i < 2; i++) {
      md_params[i].func = reinterpret_cast<void*>(test_kernel);
      md_params[i].gridDim = i+1;
      if (!support_for_separate_grid_sizes((dev + i))) {
        supports_sep_sizes = false;
      }
    }
    err = hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0);
    if ((supports_sep_sizes && err != hipSuccess) ||
      (!supports_sep_sizes && err == hipErrorInvalidValue)) {
      if (supports_sep_sizes) {
        std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
        std::cerr << "with two different grid sizes." << std::endl;
        std::cerr << "This SHOULD have succeeded with hipSuccess (";
        std::cerr << hipSuccess << ")." << std::endl;
        std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
        std::cerr << " (" << err << ")" << std::endl;
      } else {
        std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
        std::cerr << "with two different grid sizes." << std::endl;
        std::cerr << "This SHOULD have failed with the error ";
        std::cerr << "hipErrorInvalidValue (";
        std::cerr << hipErrorInvalidValue << ")." << std::endl;
        std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
        std::cerr << " (" << err << ")" << std::endl;
        FailFlag = 1;
      }
    } else {
      std::cout << "\tProperly saw this return ";
      if (supports_sep_kernels) {
        std::cout << "hipSuccess" << std::endl;
      } else {
        std::cout << "hipErrorInvalidValue" << std::endl;
      }
    }
    HIPCHECK(hipDeviceSynchronize());

    std::cout << "Launching a multi-GPU cooperative kernel with two ";
    std::cout << "different block dimensions." << std::endl;
    supports_sep_sizes = true;
    for (int i = 0; i < 2; i++) {
      md_params[i].gridDim = desired_blocks;
      md_params[i].blockDim = i+1;
      if (!support_for_separate_block_dims((dev + i))) {
        supports_sep_sizes = false;
      }
    }
    err = hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0);
    if ((supports_sep_sizes && err != hipSuccess) ||
          (!supports_sep_sizes && err == hipErrorInvalidValue)) {
      if (supports_sep_sizes) {
        std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
        std::cerr << "with two different block dimensions." << std::endl;
        std::cerr << "This SHOULD have succeeded with hipSuccess (";
        std::cerr << hipSuccess << ")." << std::endl;
        std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
        std::cerr << " (" << err << ")" << std::endl;
      } else {
        std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
        std::cerr << "with two different block dimensions." << std::endl;
        std::cerr << "This SHOULD have failed with the error ";
        std::cerr << "hipErrorInvalidValue (";
        std::cerr << hipErrorInvalidValue << ")." << std::endl;
        std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
        std::cerr << " (" << err << ")" << std::endl;
        FailFlag = 1;
      }
    } else {
      std::cout << "\tProperly saw this return ";
      if (supports_sep_kernels) {
        std::cout << "hipSuccess" << std::endl;
      } else {
        std::cout << "hipErrorInvalidValue" << std::endl;
      }
    }
    HIPCHECK(hipDeviceSynchronize());

    std::cout << "Launching a multi-GPU cooperative kernel with block ";
    std::cout << "dimensions of zero." << std::endl;
    for (int i = 0; i < 2; i++) {
      md_params[i].blockDim = 0;
    }
    err = hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0);
    if (err != hipErrorInvalidConfiguration) {
      std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
      std::cerr << "with block dimensions of zero." << std::endl;
      std::cerr << "This SHOULD have failed with the error ";
      std::cerr << "hipErrorInvalidConfiguration (";
      std::cerr << hipErrorInvalidConfiguration << ")." << std::endl;
      std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
      std::cerr << " (" << err << ")" << std::endl;
      FailFlag = 1;
    } else {
      std::cout << "\tProperly saw this return ";
      std::cout << "hipErrorInvalidConfiguration" << std::endl;
    }
    HIPCHECK(hipDeviceSynchronize());

    std::cout << "Launching a multi-GPU cooperative kernel with two ";
    std::cout << "different shared memory sizes." << std::endl;
    supports_sep_sizes = true;
    for (int i = 0; i < 2; i++) {
      md_params[i].blockDim = warp_size;
      md_params[i].sharedMem = i;
      if (!support_for_separate_shared_sizes((dev + i))) {
        supports_sep_sizes = false;
      }
    }
    err = hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0);
    if ((supports_sep_sizes && err != hipSuccess) ||
          (!supports_sep_sizes && err == hipErrorInvalidValue)) {
      if (supports_sep_sizes) {
        std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
        std::cerr << "with two different shared memory sizes." << std::endl;
        std::cerr << "This SHOULD have succeeded with hipSuccess (";
        std::cerr << hipSuccess << ")." << std::endl;
        std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
        std::cerr << " (" << err << ")" << std::endl;
      } else {
        std::cerr << "ERROR! Tried to launch a multi-GPU cooperative kernel ";
        std::cerr << "with two different shared memory sizes." << std::endl;
        std::cerr << "This SHOULD have failed with the error ";
        std::cerr << "hipErrorInvalidValue (";
        std::cerr << hipErrorInvalidValue << ")." << std::endl;
        std::cerr << "Instead, the launch returned " << hipGetErrorName(err);
        std::cerr << " (" << err << ")" << std::endl;
        FailFlag = 1;
      }
    } else {
      std::cout << "\tProperly saw this return ";
      if (supports_sep_kernels) {
        std::cout << "hipSuccess" << std::endl;
      } else {
        std::cout << "hipErrorInvalidValue" << std::endl;
      }
    }
    HIPCHECK(hipDeviceSynchronize());

    std::cout << "Launching a multi-GPU cooperative kernel with maximum ";
    std::cout << "number of warps..." << std::endl;
    for (int i = 0; i < 2; i++) {
      md_params[i].sharedMem = 0;
    }
    HIPCHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0));
    std::cout << "\tProperly launched." << std::endl;

    HIPCHECK(hipDeviceSynchronize());
#endif
    for (int m = 0; m < 2; ++m) {
      HIPCHECK(hipFree(good_dev_array[m]));
      HIPCHECK(hipFree(bad_dev_array[m]));
      HIPCHECK(hipStreamDestroy(streams[m]));
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
