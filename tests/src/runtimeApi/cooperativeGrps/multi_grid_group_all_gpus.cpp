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
/*The general idea of the application is to launch N warps to all GPUs detected
in the HIP system. N is a command-line parameter, but the user should set N
small enough that all warps can be on each of the GPUs at the same time.

All of the warps do a "work loop". Within the work loop, every warp
atomically increments a global variable that is shared between both fo the
target GPUs. The value returned from this atomic increment entriely depends
on the order the warps from the GPUs arrive at the atomic instruction. Each
warp then stores the result into a global array based on its warp ID.

We also add a sleep/wait loop into the code so that the last warp runs much
slower than everyone else. As such, it should store much larger values than
all the other warps.

If there are no barrier within the loop, then warp 0 will likely ge to the
global variable the first time while all the other warps have each
incremented it many times. If the barrier properly works, then each warp
will increment the variable once per time through the loop, and all threads
will sleep on the barrier waiting for the last warp to finally catch up.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -rdc=true -gencode arch=compute_70,code=sm_70
 * TEST: %t
 * HIT_END
 */

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "test_common.h"

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

static int verify_barrier_buffer(unsigned int loops, unsigned int warps,
                                 unsigned int *host_buffer,
                                 unsigned int num_devs) {
  unsigned int max_in_this_loop = 0;
  for (unsigned int i = 0; i < loops; i++) {
    max_in_this_loop += (warps * num_devs);
    for (unsigned int j = 0; j < warps; j++) {
      if (host_buffer[i*warps+j] > max_in_this_loop) {
        std::cerr << "Barrier failure!" << std::endl;
        std::cerr << "    Buffer entry " << i*warps+j;
        std::cerr << " contains the value " << host_buffer[i*warps+j];
        std::cerr << " but it should not be more than ";
        std::cerr << max_in_this_loop << std::endl;
        return -1;
      }
    }
  }
  std::cout << "\tBarriers work properly!" << std::endl;
  return 0;
}

static int verify_multi_gpu_buffer(unsigned int loops, unsigned int array_val) {
  unsigned int desired_val = 0;
  for (int i = 0; i < loops; i++) {
    if (i % 2 == 0) {
      desired_val += 2;
    } else {
      desired_val *= 2;
    }
  }
  std::cout << "Desired value is " << desired_val << std::endl;
  if (array_val != desired_val) {
    std::cerr << "ERROR! Multi-grid barrier does not appear to work.";
    std::cerr << std::endl;
    std::cerr << "Expected the multi-GPUs to work together to produce ";
    std::cerr << "the value " << desired_val << std::endl;
    std::cerr << "However, the entry returned from the multi-GPU ";
    std::cerr << "kernel was " << array_val << std::endl;
    return -1;
  }
    std::cout << "\tMulti-GPU barriers appear to work here." << std::endl;
    return 0;
}

__global__ void
test_kernel(unsigned int *atomic_val, unsigned int *global_array,
            unsigned int *array, uint32_t loops) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  cooperative_groups::multi_grid_group mgrid =
                      cooperative_groups::this_multi_grid();
  unsigned rank = grid.thread_rank();
  unsigned global_rank = mgrid.thread_rank();

  int offset = blockIdx.x;
  for (int i = 0; i < loops; i++) {
    // Make the last thread run way behind everyone else.
    // If the grid barrier below fails, then the other threads may hit the
    // atomicInc instruction many times before the last thread ever gets
    // to it.
    // As such, without the barrier, the last array entry will eventually
    // contain a very large value, defined by however many times the other
    // wavefronts make it through this loop.
    // If the barrier works, then it will likely contain some number
    // near "total number of blocks". It will be the last wavefront to
    // reach the atomicInc, but everyone will have only hit the atomic once.
    if (rank == (grid.size() - 1)) {
      long long start_clock = clock64();
      while (clock64() < (start_clock+1000000)) {}
    }
    if (threadIdx.x == 0) {
      array[offset] = atomicInc(atomic_val, UINT_MAX);
    }
    grid.sync();

    // Make the last thread in the entire multi-grid run way behind
    // everyone else.
    // If the mgrid barrier below fails, then the two global_array entries
    // will end up being out of sync, because the intermingling of adds
    // and multiplies will not be aligned between to the two GPUs.
    if (global_rank == (mgrid.size() - 1)) {
      long long start_clock = clock64();
      while (clock64() < (start_clock+100000000)) {}
    }
    // During even iterations, add into your own array entry
    // During odd iterations, add into your partner's array entry
    unsigned grid_rank = mgrid.grid_rank();
    unsigned inter_gpu_offset = (grid_rank + i) % mgrid.num_grids();
    if (rank == (grid.size() - 1)) {
      if (i % mgrid.num_grids() == 0) {
        global_array[grid_rank] += 2;
      } else {
        global_array[inter_gpu_offset] *= 2;
      }
    }
    mgrid.sync();
    offset += gridDim.x;
  }
}

int main(int argc, char** argv) {
  hipError_t err;
  int num_devices = 0;
  uint32_t loops = 2;
  uint32_t warps = 10;
  uint32_t block_size = 1;

  std::cout << "Loops: " << loops << std::endl;
  std::cout << "Warps: " << warps << std::endl;
  std::cout << "Block size: " << block_size << std::endl;

  HIPCHECK(hipGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    std::cout << "Not enough GPUs to run test." << std::endl;
    std::cout << "We require at least 2 GPUs, but only found ";
    std::cout << num_devices << std::endl;
    std::cout << "Skipping the test with PASSED result\n";
    passed();
  }

  uint32_t device_num[num_devices];

  /*************************************************************************/
  /* Test whether target device supports cooperative groups ****************/
  for (int i = 0; i < num_devices; i++) {
    device_num[i] = i;
    if (!cooperative_groups_support(device_num[i])) {
      std::cout << "Skipping the test with Pass result.\n";
      passed();
    }
  }

  /*************************************************************************/
  /* Test whether the requested size will fit on the GPU *******************/
  int warp_sizes[num_devices];
  int num_sms[num_devices];
  hipDeviceProp_t device_properties[num_devices];
  int warp_size = INT_MAX;
  int num_sm = INT_MAX;
  for (int i = 0; i < num_devices; i++) {
    HIPCHECK(hipGetDeviceProperties(&device_properties[i], device_num[i]));
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

  int num_threads_in_block = block_size * warp_size;

  // Calculate the device occupancy to know how many blocks can be run.
  int max_blocks_per_sm_arr[num_devices];
  int max_blocks_per_sm = INT_MAX;
  for (int i = 0; i < num_devices; i++) {
    HIPCHECK(hipSetDevice(device_num[i]));
    HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm_arr[i], test_kernel, num_threads_in_block, 0));
    if (max_blocks_per_sm_arr[i] < max_blocks_per_sm) {
      max_blocks_per_sm = max_blocks_per_sm_arr[i];
    }
  }

  int requested_blocks = warps / block_size;
  if (requested_blocks > max_blocks_per_sm * num_sm) {
    std::cerr << "Requesting to run " << requested_blocks << " blocks, ";
    std::cerr << "but we can only guarantee to simultaneously run ";
    std::cerr << (max_blocks_per_sm * num_sm) << std::endl;
    failed("");
  }

  /*************************************************************************/
  /* Set up data to pass into the kernel ***********************************/
  // Each block will output a single value per loop.
  uint32_t total_buffer_len = requested_blocks*loops;

  // Alocate the buffer that will hold the kernel's output, and which will
  // also be used to globally synchronize during GWS initialization
  unsigned int *host_buffer[num_devices];
  unsigned int *kernel_buffer[num_devices];
  unsigned int *kernel_atomic[num_devices];
  hipStream_t streams[num_devices];
  for (int i = 0; i < num_devices; i++) {
    host_buffer[i] = (unsigned int*)calloc(total_buffer_len,
                                           sizeof(unsigned int));
    HIPCHECK(hipSetDevice(device_num[i]));
    HIPCHECK(hipMalloc(reinterpret_cast<void**>(&kernel_buffer[i]),
                       total_buffer_len * sizeof(unsigned int)));
    HIPCHECK(hipMemcpy(kernel_buffer[i], host_buffer[i],
                       total_buffer_len * sizeof(unsigned int),
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMalloc(reinterpret_cast<void**>(&kernel_atomic[i]),
                       sizeof(unsigned int)));
    HIPCHECK(hipMemset(kernel_atomic[i], 0, sizeof(unsigned int)));
    HIPCHECK(hipStreamCreate(&streams[i]));
  }

  // Single kernel atomic shared between both devices; put it on the host
  unsigned int* global_array;
  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&global_array),
                         num_devices * sizeof(unsigned int), 0));
  HIPCHECK(hipMemset(global_array, 0, num_devices * sizeof(unsigned int)));

  /*************************************************************************/
  /* Launch the kernels ****************************************************/
  std::cout << "Launching a kernel with " << warps << " warps ";
  std::cout << "in " << requested_blocks << " thread blocks.";
  std::cout << std::endl;

  void *dev_params[num_devices][4];
  hipLaunchParams md_params[num_devices];
  for (int i = 0; i < num_devices; i++) {
    dev_params[i][0] = reinterpret_cast<void*>(&kernel_atomic[i]);
    dev_params[i][1] = reinterpret_cast<void*>(&global_array);
    dev_params[i][2] = reinterpret_cast<void*>(&kernel_buffer[i]);
    dev_params[i][3] = reinterpret_cast<void*>(&loops);
    md_params[i].func = reinterpret_cast<void*>(test_kernel);
    md_params[i].gridDim = requested_blocks;
    md_params[i].blockDim = num_threads_in_block;
    md_params[i].sharedMem = 0;
    md_params[i].stream = streams[i];
    md_params[i].args = dev_params[i];
  }

  HIPCHECK(hipLaunchCooperativeKernelMultiDevice(md_params, num_devices, 0));
  HIPCHECK(hipDeviceSynchronize());

  /*************************************************************************/
  /* Read back the buffers and print out its data **************************/
  for (int dev = 0; dev < num_devices; dev++) {
    HIPCHECK(hipMemcpy(host_buffer[dev], kernel_buffer[dev],
                       total_buffer_len * sizeof(unsigned int),
                       hipMemcpyDeviceToHost));
  }

  for (unsigned int i = 0; i < loops; i++) {
    for (int dev = 0; dev < num_devices; dev++) {
      std::cout << "+++++++++++++++++ Device " << dev;
      std::cout << "+++++++++++++++++" << std::endl;
      for (unsigned int j = 0; j < requested_blocks; j++) {
        std::cout << "Buffer entry " << (i*warps+j);
        std::cout << " (written by warp " << j << ")";
        std::cout << " is " << host_buffer[dev][i*requested_blocks+j];
        std::cout << std::endl;
      }
    }
    std::cout << "==========================\n";
  }
  for (unsigned int dev = 0; dev < num_devices; dev++) {
    std::cout << "Testing output from device " << dev << std::endl;
    int local_ret_val = verify_barrier_buffer(loops, requested_blocks,
                                              host_buffer[dev], num_devices);
    if (local_ret_val) {
      failed("");
    }
  }

  std::cout << std::endl << "The multi-GPU shared updates contain:\n";
  for (int i = 0; i < num_devices; i++) {
    std::cout << "Entry " << i << ": ";
    std::cout << global_array[i] << std::endl;
  }
  int flag = 0;
  for (int dev = 0; dev < num_devices; dev++) {
    std::cout << "Testing multi-GPU output for entry " << dev << std::endl;
    int local_ret_val = verify_multi_gpu_buffer(loops, global_array[dev]);
    if (local_ret_val) {
      flag = 1;
    }
  }
  for (int k = 0; k < num_devices; ++k) {
    HIPCHECK(hipFree(kernel_buffer[k]));
    HIPCHECK(hipFree(kernel_atomic[k]));
    HIPCHECK(hipStreamDestroy(streams[k]));
    free(host_buffer[k]);
  }
  if (flag == 1) {
    failed("");
  } else {
    passed();
  }
}
