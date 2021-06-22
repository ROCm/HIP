/*
Copyright (c) 2015-2021 Advanced Micro Devices, Inc. All rights reserved.
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

/* HIT_START
 * BUILD: %t %s ../test_common.cpp HIPCC_OPTIONS -std=c++11 EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include "AtomicsTest.h"

// Configurable variables
#define NUM_THREADS (16 * 16)
#define NUM_X (4)
#define NUM_Y (4)
#define NUM_BLOCKS (NUM_X * NUM_Y)

// Device Kernel - Initally all bits are set to 0. num_iter is sizeof(T), we set all bits in 
// this kernel and host function. Expected result should be max_value(all bits set to 1).
template <typename T>
__global__ void SystemAtomicsDeviceKernel_OR(T* t_device_ptr, size_t num_iter, size_t grid_size,
                                             int* sync_device_ptr) {
  // Sync variable is used to sync the operation timing between both device and host.
  atomicAdd_system(sync_device_ptr, 1);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t iter_idx = 0; iter_idx < num_iter; ++iter_idx) {
    size_t move_bit = (tid + (iter_idx * grid_size));
    atomicOr_system(t_device_ptr, 1 << move_bit);
  }
}

// Host Function - Initally all bits are set to 0. num_iter is sizeof(T), we set all bits in
// this function and device kernel. Expected result should be max_value(all bits set to 1).
template <typename T>
void SystemAtomicsHostFunction_OR(T* t_host_ptr, size_t num_iter, int* sync_host_ptr) {
  // Sync variable is used to sync the operation timing between both device and host.
  __atomic_fetch_add(sync_host_ptr, 1, __ATOMIC_RELAXED);

  for (size_t iter_idx = 0; iter_idx < num_iter; ++iter_idx) {
    __atomic_or_fetch(t_host_ptr, 1 << iter_idx, __ATOMIC_RELAXED);
  }
}

template <typename T>
bool runTest() {
  bool test_result = true;
  AtomicsTest atmcs_test;
  T* t_host_ptr = nullptr;
  T* t_device_ptr = nullptr;
  int* sync_host_ptr = nullptr;
  int* sync_device_ptr = nullptr;

  // Iteration is the number of bits
  size_t num_iter = sizeof(T) * 8;
  assert((num_iter % (NUM_X * NUM_Y)) == 0);

  // Get a pointer that is accesible both by host and device.
  atmcs_test.get_system_atomics_ptr<int>(&sync_host_ptr, &sync_device_ptr, 1);
  assert(sync_host_ptr != nullptr);
  assert(sync_device_ptr != nullptr);

  atmcs_test.get_system_atomics_ptr<T>(&t_host_ptr, &t_device_ptr, 1);
  assert(t_host_ptr != nullptr);
  assert(t_device_ptr != nullptr);

  // 1.Launch Device Kernel
  SystemAtomicsDeviceKernel_OR<T><<<NUM_X, NUM_Y>>>(t_device_ptr, num_iter/(NUM_X * NUM_Y),
                                                    (NUM_X * NUM_Y), sync_device_ptr);

  // 2.Launch Host Function
  SystemAtomicsHostFunction_OR<T>(t_host_ptr, num_iter, sync_device_ptr);

  // 3.Validate Results
  HIPCHECK(hipDeviceSynchronize());
  // same pointer address was incremented by 1 by both host and device ptr for num_iterations.
  if(*t_host_ptr != std::numeric_limits<T>::max()) {
    test_result = false;
  }

  atmcs_test.free_system_atomics_ptr<int>(sync_host_ptr);
  atmcs_test.free_system_atomics_ptr<T>(t_host_ptr);

  return test_result;
}

int main() {
  bool result = true;
  result &= runTest<unsigned int>();
  result &= runTest<unsigned long>();
  result &= runTest<unsigned long long>();

  if (result) {
    passed();
  }

  return 0;
}
