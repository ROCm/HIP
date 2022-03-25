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

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

/*
This testfile verifies the following scenarios of hipHostMalloc API
1. Basic scenario of hipHostMalloc API
2. Negative Scenarios of hipHostMalloc API
3. Allocating memory using hipHostMalloc with Coherent flag
4. Allocating memory using hipHostMalloc with NonCoherent flag
5. Allocating memory using hipHostMalloc with default flag
*/

#include<hip_test_checkers.hh>
#include<kernels.hh>
#include<hip_test_common.hh>
// #include<Set.hpp>

#define SYNC_EVENT 0
#define SYNC_STREAM 1
#define SYNC_DEVICE 2

std::vector<std::string> syncMsg = {"event", "stream", "device"};
static constexpr int numElements{1024 * 16};
static constexpr size_t sizeBytes{numElements * sizeof(int)};

// __global__ void Set(int* Ad, int val) {
//     int tx = threadIdx.x + blockIdx.x * blockDim.x;
//     Ad[tx] = val;
// }

void CheckHostPointer(int numElements, int* ptr, unsigned eventFlags,
                      int syncMethod, std::string msg) {
    std::cerr << "test: CheckHostPointer "
              << msg
              << " eventFlags = " << std::hex << eventFlags
              << ((eventFlags & hipEventReleaseToDevice) ?
                 " hipEventReleaseToDevice" : "")
              << ((eventFlags & hipEventReleaseToSystem) ?
                 " hipEventReleaseToSystem" : "")
              << " ptr=" << ptr << " syncMethod="
              << syncMsg[syncMethod] << "\n";

    hipStream_t s;
    hipEvent_t e;

    // Init:
    HIP_CHECK(hipStreamCreate(&s));
    HIP_CHECK(hipEventCreateWithFlags(&e, eventFlags))
    dim3 dimBlock(64, 1, 1);
    dim3 dimGrid(numElements / dimBlock.x, 1, 1);

    const int expected = 13;

    // Init array to know state:
    // hipLaunchKernelGGL(Set, dimGrid, dimBlock, 0, 0x0, ptr, -42);
    HipTest::launchKernel(Set, dimGrid, dimBlock, 0, 0x0, ptr, -42);
    HIP_CHECK(hipDeviceSynchronize());

    // hipLaunchKernelGGL(Set, dimGrid, dimBlock, 0, s, ptr, expected);
    HipTest::launchKernel(Set, dimGrid, dimBlock, 0, s, ptr, expected);
    HIP_CHECK(hipEventRecord(e, s));

    // Host waits for event :
    switch (syncMethod) {
        case SYNC_EVENT:
            HIP_CHECK(hipEventSynchronize(e));
            break;
        case SYNC_STREAM:
            HIP_CHECK(hipStreamSynchronize(s));
            break;
        case SYNC_DEVICE:
            HIP_CHECK(hipDeviceSynchronize());
            break;
        default:
            assert(0);
    }

    for (int i = 0; i < numElements; i++) {
        if (ptr[i] != expected) {
            printf("mismatch at %d: %d != %d\n", i, ptr[i], expected);
            REQUIRE(ptr[i] == expected);
        }
    }

    HIP_CHECK(hipStreamDestroy(s));
    HIP_CHECK(hipEventDestroy(e));
}
/*
This testcase performs the basic scenario of hipHostMalloc API
Allocates the memory using hipHostMalloc API
Launches the kernel and performs vector addition.
validates thes result.
*/
TEST_CASE("Unit_hipHostMalloc_Basic") {
  static constexpr auto LEN{1024 * 1024};
  static constexpr auto SIZE{LEN * sizeof(float)};

  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  if (prop.canMapHostMemory != 1) {
    SUCCEED("Does support HostPinned Memory");
  } else {
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), SIZE,
                           hipHostMallocWriteCombined | hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&B_h), SIZE,
                           hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&C_h), SIZE,
                           hipHostMallocMapped));

    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&C_d), C_h, 0));

    HipTest::setDefaultData<float>(LEN, A_h, B_h, C_h);

    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&B_d), SIZE));
    HIP_CHECK(hipMemcpy(B_d, B_h, SIZE, hipMemcpyHostToDevice));

    dim3 dimGrid(LEN / 512, 1, 1);
    dim3 dimBlock(512, 1, 1);
    HipTest::launchKernel<float>(HipTest::vectorADD<float>, dimGrid, dimBlock,
                       0, 0, static_cast<const float*>(A_d),
                       static_cast<const float*>(B_d), C_d, LEN);
    HIP_CHECK(hipMemcpy(C_h, C_d, LEN*sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HipTest::checkVectorADD<float>(A_h, B_h, C_h, numElements);

    HIP_CHECK(hipHostFree(A_h));
    HIP_CHECK(hipHostFree(B_h));
    HIP_CHECK(hipHostFree(C_h));
  }
}
/*
This testcase verifies the hipHostMalloc API by passing nullptr
to the pointer variable
*/
TEST_CASE("Unit_hipHostMalloc_Negative") {
#if HT_AMD
  {
    // Stimulate error condition:
    int* A = nullptr;
    REQUIRE(hipHostMalloc(reinterpret_cast<void**>(&A), sizeBytes,
            hipHostMallocCoherent | hipHostMallocNonCoherent)
            != hipSuccess);
    REQUIRE(A == nullptr);
  }
#endif
}
/*
This testcase verifies the hipHostMalloc API by
1.Allocating memory using noncoherent flag
2. Launches the kernel and modifies the variable
   using different synchronization
   techniquies
3. validates the result.
*/
TEST_CASE("Unit_hipHostMalloc_NonCoherent") {
  int* A = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A),
                          sizeBytes, hipHostMallocNonCoherent));
  const char* ptrType = "non-coherent";
  CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                   SYNC_DEVICE, ptrType);
  CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                   SYNC_STREAM, ptrType);
  CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                   SYNC_EVENT, ptrType);
}

/*
This testcase verifies the hipHostMalloc API by
1.Allocating memory using coherent flag
2. Launches the kernel and modifies the variable
   using different synchronization
   techniquies
3. validates the result.
*/
TEST_CASE("Unit_hipHostMalloc_Coherent") {
  int* A = nullptr;
  if (hipHostMalloc(reinterpret_cast<void**>(&A), sizeBytes,
                    hipHostMallocCoherent) == hipSuccess) {
    const char* ptrType = "coherent";
    CheckHostPointer(numElements, A, hipEventReleaseToDevice,
                     SYNC_DEVICE, ptrType);
    CheckHostPointer(numElements, A, hipEventReleaseToDevice,
                     SYNC_STREAM, ptrType);
    CheckHostPointer(numElements, A, hipEventReleaseToDevice,
                     SYNC_EVENT, ptrType);

    CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                     SYNC_DEVICE, ptrType);
    CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                     SYNC_STREAM, ptrType);
    CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                     SYNC_EVENT, ptrType);
  } else {
    SUCCEED("Coherence memory allocation failed. Is SVM atomic supported?");
  }
}

/*
This testcase verifies the hipHostMalloc API by
1.Allocating memory using default flag
2. Launches the kernel and modifies the variable
   using different synchronization
   techniquies
3. validates the result.
*/
TEST_CASE("Unit_hipHostMalloc_Default") {
  int* A = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A), sizeBytes));
  const char* ptrType = "default";
  CheckHostPointer(numElements, A, 0, SYNC_DEVICE, ptrType);
  CheckHostPointer(numElements, A, 0, SYNC_STREAM, ptrType);
  CheckHostPointer(numElements, A, 0, SYNC_EVENT, ptrType);
}
