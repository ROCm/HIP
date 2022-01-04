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

#define MEM_SIZE (1024*1024*32)
#define SEED 5

constexpr unsigned int MAX_THREAD_CNT = 10;

__global__ void copy_kernl(int* devPtr) {
  for (int i = 0; i < MEM_SIZE; ++i) {
    devPtr[i] = (i+1) + SEED;
  }
}

TEST_CASE("Unit_hipStreamPerThread_Basic") {
  constexpr int size = sizeof(int) * MEM_SIZE;
  int* hostMem = nullptr;
  int* devMem = nullptr;

  HIP_CHECK(hipHostMalloc(&hostMem, size));
  HIP_CHECK(hipMalloc(&devMem, size));

  // Init host mem with different values
  for (int i = 0; i < MEM_SIZE; ++i) {
    hostMem[i] = i;
  }

  /*
   hipStreamPerThread is an implicit stream which works independent of null stream.
   Null stream synchronize will account hipStreamPerThread into account.
   test scenario: Launch kernel + Async mem copy on hipStreamPerThread and call synchronize on null stream.
   Result : Null stream synchronize should sync hipStreamPerThread as well
   */
  copy_kernl<<<1, 1, 0, hipStreamPerThread>>>(devMem);

  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, size, hipMemcpyDeviceToHost, hipStreamPerThread));

  HIP_CHECK(hipStreamSynchronize(0));

  // validate result
  for (int i = MEM_SIZE-1; i >= 0; --i) {
    CHECK(hostMem[i] == (i+1+SEED));
  }
}

TEST_CASE("Unit_hipStreamPerThread_StreamQuery") {
  std::vector<std::thread> threads(MAX_THREAD_CNT);

  for (auto &th : threads) {
    th = std::thread([](){HIP_CHECK(hipStreamQuery(hipStreamPerThread));});
  }

  for (auto& th : threads) {
    th.join();
  }
  REQUIRE(true);
}

TEST_CASE("Unit_hipStreamPerThread_StreamSynchronize") {
  constexpr unsigned int MAX_THREAD_CNT = 10;
  std::vector<std::thread> threads(MAX_THREAD_CNT);

  for (auto &th : threads) {
    th = std::thread([](){HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));});
  }

  for (auto& th : threads) {
    th.join();
  }
  REQUIRE(true);
}

TEST_CASE("Unit_hipStreamPerThread_StreamGetPriority") {
  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(hipStreamPerThread, &priority));
}

TEST_CASE("Unit_hipStreamPerThread_StreamGetFlags") {
  unsigned int flags = 0;
  HIP_CHECK(hipStreamGetFlags(hipStreamPerThread, &flags));
}

TEST_CASE("Unit_hipStreamPerThread_StreamDestroy") {
  hipError_t status = hipStreamDestroy(hipStreamPerThread);
  REQUIRE(status != hipSuccess);
}

TEST_CASE("Unit_hipStreamPerThread_MemcpyAsync") {
  unsigned int ele_size = (16 * 1024);  // 16KB
  int* A_h = nullptr;
  int* A_d = nullptr;

  HIP_CHECK(hipHostMalloc(&A_h, ele_size*sizeof(int)));
  HIP_CHECK(hipMalloc(&A_d, ele_size * sizeof(int)));

  for (unsigned int i = 0; i < ele_size; ++i) {
    A_h[i] = 123;
  }

  HIP_CHECK(hipMemcpy(A_d, A_h, ele_size * sizeof(int), hipMemcpyHostToDevice));

  // Rest host memory
  for (unsigned int i = 0; i < ele_size; ++i) {
    A_h[i] = 0;
  }

  HIP_CHECK(hipMemcpyAsync(A_h, A_d, ele_size * sizeof(int), hipMemcpyDeviceToHost,
                           hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  // Verify result
  for (unsigned int i = 0; i < ele_size; ++i) {
    REQUIRE(A_h[i] == 123);
  }
}