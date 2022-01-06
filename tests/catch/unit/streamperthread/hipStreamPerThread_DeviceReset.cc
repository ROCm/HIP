/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <vector>
#include <thread>

/*
 hipDeviceReset deletes all active streams including hipStreamPerThread.
 Scenario: App calls hipDeviceReset while in other thread some Async operation is in
          progress on hipStreamPerThread.
 Watch out: hipDeviceRest should be successfull without any crash
 */
static void Copy_to_device() {
  unsigned int ele_size = (32 * 1024);  // 32KB
  int* A_h = nullptr;
  int* A_d = nullptr;

  hipError_t status =  hipHostMalloc(&A_h, ele_size*sizeof(int));
  if (status != hipSuccess) return;

  status = hipMalloc(&A_d, ele_size * sizeof(int));
  if (status != hipSuccess) return;

  for(unsigned int i = 0; i < ele_size; ++i) {
    A_h[i] = 123;
  }
  hipMemcpyAsync(A_d, A_h, ele_size * sizeof(int), hipMemcpyHostToDevice,
                 hipStreamPerThread);
}

TEST_CASE("Unit_hipStreamPerThread_DeviceReset_1") {
  constexpr unsigned int MAX_THREAD_CNT = 10;
  std::vector<std::thread> threads(MAX_THREAD_CNT);

  for (auto &th : threads) {
    th = std::thread(Copy_to_device);
    th.detach();
  }
  HIP_CHECK(hipDeviceReset());
}

/*
 hipDeviceReset deletes all active streams including hipStreamPerThread.
 Scenario: i) Launch Async task on hipStreamPerThread and waits for it to complete.
          ii) Call hipDeviceReset to delete all active stream
         iii) Again try to launch Async task on hipStreamPerThread
 Watch out: Since hipStreamPerThread is an implicit stream hence even after device reset
            it should available to use.
 */
TEST_CASE("Unit_hipStreamPerThread_DeviceReset_2") {
  unsigned int ele_size = (32 * 1024);  // 32KB
  int* A_h = nullptr;
  int* A_d = nullptr;

  hipError_t status =  hipHostMalloc(&A_h, ele_size*sizeof(int));
  if (status != hipSuccess) return;
  status = hipMalloc(&A_d, ele_size * sizeof(int));
  if (status != hipSuccess) return;

  for (unsigned int i = 0; i < ele_size; ++i) {
    A_h[i] = 123;
  }
  hipMemcpyAsync(A_d, A_h, ele_size * sizeof(int), hipMemcpyHostToDevice,
                 hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);

  hipDeviceReset();

  hipMemcpyAsync(A_d, A_h, ele_size * sizeof(int), hipMemcpyHostToDevice,
                 hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);
}