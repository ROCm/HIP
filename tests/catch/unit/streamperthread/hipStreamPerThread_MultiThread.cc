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

static void Copy_to_device() {
  unsigned int ele_size = (32 * 1024);  // 32KB
  int* A_h = nullptr;
  int* A_d = nullptr;

  hipHostMalloc(&A_h, ele_size*sizeof(int));
  hipMalloc(&A_d, ele_size * sizeof(int));

  for (unsigned int i = 0; i < ele_size; ++i) {
    A_h[i] = 123;
  }
  hipMemcpyAsync(A_d, A_h, ele_size * sizeof(int), hipMemcpyHostToDevice,
                 hipStreamPerThread);
}

/*
hipStreamPerThread is an implicit stream which gets destroyed once thread is completed.
Scenario : App pushes Async task(s) into hipStreamPerThread and did not wait for it to complete.
Watch out : Incomplete task in hipStreamPerThread should not cause any crash due to thread exit.
 */
TEST_CASE("Unit_hipStreamPerThread_MultiThread") {
  constexpr unsigned int MAX_THREAD_CNT = 10;
  std::vector<std::thread> threads(MAX_THREAD_CNT);

  for (auto &th : threads) {
    th = std::thread(Copy_to_device);
  }

  for (auto& th : threads) {
    th.detach();
  }
}
