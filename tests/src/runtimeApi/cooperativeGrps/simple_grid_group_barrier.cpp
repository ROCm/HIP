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
/*The general idea of the application is to launch N warps. N is a command-line
parameter, but the user should set N small enough that all warps can be on
the GPU at the same time.

All of the warps do a "work loop". Within the work loop, every warp
atomically increments a global variable. The value returned from this atomic
increment entriely depends on the order the threads arrive at the atomic
instruction. Each warp then stores the result into a global array based on its
warp ID.

We also add a sleep/wait loop into the code so that the last warp runs much
slower than everyone else. As such, it should store much larger values than
all the other warps.

If there are no barrier within the loop, then the last warp will likely get to
the global variable the first time after all the other warps have each
incremented it many times. If the barrier properly works, then each warp
will increment the variable once per time through the loop, and all threads
will sleep on the barrier waiting for the last warp to finally catch up.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
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

  hipDeviceProp_t device_properties;
  HIPCHECK(hipGetDeviceProperties(&device_properties, device_id));
  if (device_properties.cooperativeLaunch == 0) {
    std::cerr << "Cooperative group support not available in ";
    std::cerr << "device properties." << std::endl;
    return 0;
  }
  return 1;
}

static int verify_barrier_buffer(unsigned int loops, unsigned int warps,
                                 unsigned int *host_buffer) {
  unsigned int max_in_this_loop = 0;
  for (unsigned int i = 0; i < loops; i++) {
    max_in_this_loop += warps;
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
  std::cout << "Barriers work properly!" << std::endl;
  return 0;
}

__global__ void
test_kernel(unsigned int *atomic_val, unsigned int *array,
            unsigned int loops) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  unsigned rank = grid.thread_rank();

  int offset = blockIdx.x;
  for (int i = 0; i < loops; i++) {
    // Make the last thread run way behind everyone else.
    // If the barrier below fails, then the other threads may hit the
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
      array[offset] = atomicInc(&atomic_val[0], UINT_MAX);
    }
    grid.sync();
    offset += gridDim.x;
  }
}

int main(int argc, char** argv) {
  hipError_t err;
  int device_num;
  uint32_t loops = 2;
  uint32_t warps = 10;
  uint32_t block_size = 1;
  HIPCHECK(hipGetDeviceCount(&device_num));
  for (int dev = 0; dev < device_num; ++dev) {
    std::cout << "Device number: " << dev << std::endl;
    std::cout << "Loops: " << loops << std::endl;
    std::cout << "Warps: " << warps << std::endl;
    std::cout << "Block size: " << block_size << std::endl;

    /*************************************************************************/
    /* Test whether target device supports cooperative groups ****************/
    HIPCHECK(hipSetDevice(dev));
    if (!cooperative_groups_support(dev)) {
      std::cout << "Skipping the test with Pass result.\n";
      passed();
    }

    /*************************************************************************/
    /* Test whether the requested size will fit on the GPU *******************/
    int warp_size;
    int num_sms;
    int max_blocks_per_sm;
    hipDeviceProp_t device_properties;
    HIPCHECK(hipGetDeviceProperties(&device_properties, dev));
    warp_size = device_properties.warpSize;
    num_sms = device_properties.multiProcessorCount;

    std::cout << "Device name: " << device_properties.name << std::endl;
    std::cout << std::endl;

    int num_threads_in_block = block_size * warp_size;

    // Calculate the device occupancy to know how many blocks can be run.
    HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
             test_kernel, num_threads_in_block, 0));

    int requested_blocks = warps / block_size;
    if (requested_blocks > max_blocks_per_sm * num_sms) {
      std::cerr << "Requesting to run " << requested_blocks << " blocks, ";
      std::cerr << "but we can only guarantee to simultaneously run ";
      std::cerr << (max_blocks_per_sm * num_sms) << std::endl;
      failed("");
    }

    /*************************************************************************/
    /* Set up data to pass into the kernel ***********************************/
    // Each block will output a single value per loop.
    uint32_t total_buffer_len = requested_blocks*loops;

    // Alocate the buffer that will hold the kernel's output, and which will
    // also be used to globally synchronize during GWS initialization
    unsigned int *host_buffer = (unsigned int*)calloc(total_buffer_len,
            sizeof(unsigned int));

    unsigned int *kernel_buffer;
    HIPCHECK(hipMalloc(reinterpret_cast<void**>(&kernel_buffer),
                       total_buffer_len * sizeof(unsigned int)));
    HIPCHECK(hipMemcpy(kernel_buffer, host_buffer,
                       total_buffer_len * sizeof(unsigned int),
                       hipMemcpyHostToDevice));

    unsigned int *kernel_atomic;
    HIPCHECK(hipMalloc(reinterpret_cast<void**>(&kernel_atomic),
                       sizeof(unsigned int)));
    HIPCHECK(hipMemset(kernel_atomic, 0, sizeof(unsigned int)));

    /*************************************************************************/
    /* Launch the kernel *****************************************************/
    std::cout << "Launching a kernel with " << warps << " warps ";
    std::cout << "in " << requested_blocks << " thread blocks.";
    std::cout << std::endl;

    void *params[3];
    params[0] = reinterpret_cast<void*>(&kernel_atomic);
    params[1] = reinterpret_cast<void*>(&kernel_buffer);
    params[2] = reinterpret_cast<void*>(&loops);
    HIPCHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                        requested_blocks,
                                        num_threads_in_block, params, 0, NULL));

    /*************************************************************************/
    /* Read back the buffer and print out its data****************************/
    HIPCHECK(hipMemcpy(host_buffer, kernel_buffer,
                       total_buffer_len * sizeof(unsigned int),
                       hipMemcpyDeviceToHost));

    for (unsigned int i = 0; i < loops; i++) {
      for (unsigned int j = 0; j < requested_blocks; j++) {
        std::cout << "Buffer entry " << (i*warps+j);
        std::cout << " (written by warp " << j << ")";
        std::cout << " is " << host_buffer[i * requested_blocks + j];
        std::cout << std::endl;
      }
      std::cout << "==========================\n";
    }
    int ret_val = verify_barrier_buffer(loops, requested_blocks, host_buffer);
    HIPCHECK(hipFree(kernel_buffer));
    HIPCHECK(hipFree(kernel_atomic));
    if (ret_val == -1) {
      failed("");
    } else {
      passed();
    }
  }
}
