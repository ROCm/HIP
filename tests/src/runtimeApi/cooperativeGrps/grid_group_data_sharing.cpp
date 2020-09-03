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
/*The general idea of the application is to create a buffer of width N. N is a
command line parameter, and the user will need to make sure that we can fit
two buffers of N unsigned integers onto the target GPU at the same time.

We then launch a fixed number of warps to the GPU. This number is calculated
to fill the GPU with as many warps as can simultaneously run on the GPU.
The threads in these warps then walk over two arrays. First, values from
A[offset] are added into B[offset]. After all of A is added into all of B
in this element-wise manner, all of the waves barrier with one another.

After the barrier, the waves start adding values from B[mirror_offset] into
A[offset]. Mirror offset means that the wave that is writing into A[7] is
reading from B[7 before the last value]. This was probably written by a
different thread before the barrier.

After going through this loop a certain number of times, the kernel ends and
we read the arrays back out and recalculate this algorithm serially on the
CPU. We compare the serial version to the version that has inter-thread data
sharing and barriers and ensure they result in the same answer.

If they do have the same answer, then we can pretty confidently say that
writing from thread X and then hitting a barrier allows thread Y to see the
values.*/

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
    exit(errval);
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
#define hipCheckErr(errval)\
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

  hipDeviceProp_t device_properties;
  HIPCHECK(hipGetDeviceProperties(&device_properties, device_id));
  if (device_properties.cooperativeLaunch == 0) {
    std::cerr << "Cooperative group support not available in ";
    std::cerr << "device properties." << std::endl;
    return 0;
  }
  return 1;
}

static int verify_coop_arrays(unsigned int loops, unsigned int *host_input,
                              unsigned int *first_array,
                              unsigned int *second_array,
                              unsigned int array_len) {
  unsigned int *host_first_array = host_input;
  unsigned int *host_second_array = (unsigned int*)calloc(array_len,
                                                          sizeof(int));

  for (int i = 0; i < loops; i++) {
    for (int offset = 0; offset < array_len; offset++) {
      host_second_array[offset] += host_first_array[offset];
    }

    for (int offset = 0; offset < array_len; offset++) {
      unsigned int swizzle_offset = array_len - offset - 1;
      host_first_array[offset] += host_second_array[swizzle_offset];
    }
  }

  for (int i = 0; i < array_len; i++) {
    if (host_first_array[i] != first_array[i]) {
      std::cerr << "Test failure!" << std::endl;
      std::cerr << "    host_first_array[" << i << "] contains the ";
      std::cerr << "value " << host_first_array[i] << std::endl;
      std::cerr << "    GPU first_array[" << i << "] contains the ";
      std::cerr << "value " << first_array[i] << std::endl;
      return -1;
    }
    if (host_second_array[i] != second_array[i]) {
      std::cerr << "Test failure!" << std::endl;
      std::cerr << "    host_second_array[" << i << "] contains the ";
      std::cerr << "value " << host_second_array[i] << std::endl;
      std::cerr << "    GPU second_array[" << i << "] contains the ";
      std::cerr << "value " << second_array[i] << std::endl;
      return -1;
    }
  }

  std::cout << "Coop test appears to work properly!" << std::endl;
  free(host_second_array);
  return 0;
}

__global__ void
coop_kernel(unsigned int *first_array, unsigned int *second_array,
            unsigned int loops, unsigned int array_len) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  unsigned int rank = grid.thread_rank();
  unsigned int grid_size = grid.size();

  for (int i = 0; i < loops; i++) {
    // The goal of this loop is to directly add in values from
    // array one into array two, on a per-wave basis.
    for (int offset = rank; offset < array_len; offset += grid_size) {
      second_array[offset] += first_array[offset];
    }

    grid.sync();

    // The goal of this loop is to pull data the "mirror" lane in
    // array two and add it back into array one. This causes inter-
    // thread swizzling.
    for (int offset = rank; offset < array_len; offset += grid_size) {
      unsigned int swizzle_offset = array_len - offset - 1;
      first_array[offset] += second_array[swizzle_offset];
    }

    grid.sync();
  }
}

int main(int argc, char** argv) {
  hipError_t err;
  /*************************************************************************/
  /* Parse the command line parameters *************************************/
  // Arguments to pull out of the command line.
  int device_num = 0, loops = 2, width = 4096, flag = 0;
  HIPCHECK(hipGetDeviceCount(&device_num));
  for (int dev = 0; dev < device_num; ++dev) {
    std::cout << "Device number: " << dev << std::endl;
    std::cout << "Loops: " << loops << std::endl;
    std::cout << "Width: " << width << std::endl;

    /*************************************************************************/
    /* Test whether target device supports cooperative groups ****************/
    HIPCHECK(hipSetDevice(dev));

    if (!cooperative_groups_support(dev)) {
      std::cout << "Skipping the test with Pass result.\n";
      passed();
    }

    /*************************************************************************/
    /* We will launch enough waves to fill up all of the GPU *****************/
    hipDeviceProp_t device_properties;
    HIPCHECK(hipGetDeviceProperties(&device_properties, dev));

    int warp_size = device_properties.warpSize;
    int num_sms = device_properties.multiProcessorCount;

    std::cout << "Device name: " << device_properties.name << std::endl;
    std::cout << std::endl;

    // Calculate the device occupancy to know how many blocks can be run.
    int max_blocks_per_sm;
    HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                          coop_kernel,
                                                          warp_size, 0));

    int total_blocks = max_blocks_per_sm * num_sms;

    /*************************************************************************/
    /* Create the streams we will use in this test. **************************/
    hipStream_t streams[2];
    for (int i = 0; i < 2; i++) {
      HIPCHECK(hipStreamCreate(&streams[i]));
    }

    /*************************************************************************/
    /* Set up data to pass into the kernel ***********************************/

    // Alocate the host input buffer, and two device-focused buffers that we
    // will use for our test.
    unsigned int *input_buffer = (unsigned int*)calloc(width,
                                                       sizeof(unsigned int));
    for (int i = 0; i < width; i++) {
      input_buffer[i] = i;
    }

    unsigned int *first_dev_array;
    HIPCHECK(hipMalloc(reinterpret_cast<void**>(&first_dev_array),
                       width * sizeof(unsigned int)));

    HIPCHECK(hipMemcpyAsync(first_dev_array, input_buffer,
                            width * sizeof(unsigned int),
                            hipMemcpyHostToDevice, streams[0]));

    unsigned int *second_dev_array;
    HIPCHECK(hipMalloc(reinterpret_cast<void**>(&second_dev_array),
                       width * sizeof(unsigned int)));
    HIPCHECK(hipMemsetAsync(second_dev_array, 0, width * sizeof(unsigned int),
                            streams[0]));

    /*************************************************************************/
    /* Launch the kernels ****************************************************/
    std::cout << "Launching a cooperative kernel with " << total_blocks;
    std::cout << " thread blocks, each with " << warp_size << " threads";
    std::cout << std::endl;

    void *coop_params[4];
    coop_params[0] = reinterpret_cast<void*>(&first_dev_array);
    coop_params[1] = reinterpret_cast<void*>(&second_dev_array);
    coop_params[2] = reinterpret_cast<void*>(&loops);
    coop_params[3] = reinterpret_cast<void*>(&width);
    HIPCHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(coop_kernel),
                                        total_blocks, warp_size, coop_params,
                                        0, streams[0]));

    /*************************************************************************/
    /* Read back the buffers and print out their data ************************/
    unsigned int *first_array = (unsigned int*)calloc(width,
                                                      sizeof(unsigned int));
    unsigned int *second_array = (unsigned int*)calloc(width,
                                                       sizeof(unsigned int));
    HIPCHECK(hipMemcpyAsync(first_array, first_dev_array,
                            width * sizeof(unsigned int),
                            hipMemcpyDeviceToHost, streams[0]));

    HIPCHECK(hipMemcpyAsync(second_array, second_dev_array,
                            width * sizeof(unsigned int),
                            hipMemcpyDeviceToHost, streams[0]));

    std::cout << "Waiting for cooperative work to finish..." << std::endl;
    std::cout << std::flush;

    HIPCHECK(hipStreamSynchronize(streams[0]));


    int ret_val = 0;

    std::cout << "Attemping to verify buffers." << std::endl;
    std::cout << std::flush;
    ret_val = verify_coop_arrays(loops, input_buffer, first_array,
                                 second_array, width);
    if (!ret_val) {
      std::cout << "It appears that inter-thread data sharing at ";
      std::cout << "grid_group sync points works properly!" << std::endl;
    } else {
      flag = 1;
    }
    for (int k = 0; k < 2; ++k) {
      HIPCHECK(hipStreamDestroy(streams[k]));
    }
    HIPCHECK(hipFree(first_dev_array));
    HIPCHECK(hipFree(second_dev_array));
    free(input_buffer);
    free(first_array);
    free(second_array);
  }
  if (!flag) {
    passed();
  } else {
    failed("");
  }
}
