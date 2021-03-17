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
/*
    This unit test is written to test Stream Write and Stream Wait API.
    Stream Write:
      Both 32 and 64 bit version of this APIs are tested by writing a specific value and checking
   the correctness. Various mememory objects (host registered, device and Signal Memory) are tested.
    Stream Wait:
      Wait API is tested using two memory locations (DataPr and SignalPtr). Following
      commands are executed for each type of wait operaitons (GEQ, EQ, AND and NOR) in the order
   specified.
        1. CPU : An intial values are written to DataPtr and SignalPtr
        2. GPU : Wait operation (with false condition that blocks the stream) is enqued.
        3. GPU : Write operation on DataPtr to update its value is enqued.
        4. CPU : A query or CPU wait to make sure all commands are processed by GPU.
        5. CPU : streamQuery is performed to make sure it is not finshed executing the commands,
   since step-2 is blocking.
        6. CPU : A new value is written to SignalPtr memory that make wait condition defined in
   step-2 to be true. This causes step-3 to be executed.
        7. CPU : Synchronize the stream and read value at DataPtr, it should be equal to updated
   value (step-3).
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11 EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include <unistd.h>
#include <hip/hip_runtime.h>
#include "test_common.h"

// Random predefiend 32 and 64 bit values
constexpr int32_t value32 =         0x70F0F0FF;
constexpr int64_t value64 =         0x7FFF0000FFFF0000;
constexpr unsigned int writeFlag = 0;

constexpr float SLEEP_MS = 100;
void testWrite() {

  int64_t* signalPtr;

  hipStream_t stream;
  hipStreamCreate(&stream);

  int64_t* host_ptr64 = (int64_t *) malloc(sizeof(int64_t));
  int32_t* host_ptr32 = (int32_t *) malloc(sizeof(int32_t));
  std::cout << " hipStreamWriteValue: testing ... \n";

  HIPCHECK(hipExtMallocWithFlags((void **)&signalPtr, 8, hipMallocSignalMemory));

  void* device_ptr64;
  void* device_ptr32;

  *host_ptr64 = 0x0;
  *host_ptr32 = 0x0;
  *signalPtr = 0x0;

  hipHostRegister(host_ptr64, sizeof(int64_t), 0);
  hipHostRegister(host_ptr32, sizeof(int32_t), 0);

  // Test writting registered host pointer
  HIPCHECK(hipStreamWriteValue64(stream, host_ptr64, value64, writeFlag));
  HIPCHECK(hipStreamWriteValue32(stream, host_ptr32, value32, writeFlag));
  hipStreamSynchronize(stream);

  HIPASSERT(*host_ptr64 == value64);
  HIPASSERT(*host_ptr32 == value32);

  // Test writting device pointer
  hipHostGetDevicePointer((void**)&device_ptr64, host_ptr64, 0);
  hipHostGetDevicePointer((void**)&device_ptr32, host_ptr32, 0);

  // Reset values
  *host_ptr64 = 0x0;
  *host_ptr32 = 0x0;

  HIPCHECK(hipStreamWriteValue64(stream, device_ptr64, value64, writeFlag));
  HIPCHECK(hipStreamWriteValue32(stream, device_ptr32, value32, writeFlag));
  hipStreamSynchronize(stream);

  HIPASSERT(*host_ptr64 == value64);
  HIPASSERT(*host_ptr32 == value32);

  // Test Writing to Signal Memory
  HIPCHECK(hipStreamWriteValue64(stream, signalPtr, value64, writeFlag));
  hipStreamSynchronize(stream);

  HIPASSERT(*signalPtr == value64);

  // Cleanup
  hipStreamDestroy(stream);
  hipHostUnregister(host_ptr64);
  hipHostUnregister(host_ptr32);
  HIPCHECK(hipFree(signalPtr));
  free(host_ptr32);
  free(host_ptr64);
}

bool streamWaitValueSupported() {
  int device_num = 0;
  HIPCHECK(hipGetDeviceCount(&device_num));
  int waitValueSupport;
  for (int device_id = 0; device_id < device_num; ++device_id) {
    HIPCHECK(hipSetDevice(device_id));
    waitValueSupport = 0;
    HIPCHECK(hipDeviceGetAttribute(&waitValueSupport, hipDeviceAttributeCanUseStreamWaitValue,
                                   device_id));
    if (waitValueSupport == 1) return true;
  }
  return false;
}

void testWait() {
  int64_t* signalPtr;
  // random data values
  int32_t DATA_INIT = 0x1234;
  int32_t DATA_UPDATE = 0X4321;

  struct TEST_WAIT {
    int compareOp;
    uint64_t mask;
    int64_t waitValue;
    int64_t signalValueFail;
    int64_t signalValuePass;
  };

  TEST_WAIT testCases[] = {
    {
      // mask will ignore few MSB bits
      hipStreamWaitValueGte,
      0x0000FFFFFFFFFFFF,
      0x000000007FFF0001,
      0x7FFF00007FFF0000,
      0x000000007FFF0001
    },
    {
      hipStreamWaitValueGte,
      0xF,
      0x4,
      0x3,
      0x6
    },
    {
      // mask will ignore few MSB bits
      hipStreamWaitValueEq,
      0x0000FFFFFFFFFFFF,
      0x000000000FFF0001,
      0x7FFF00000FFF0000,
      0x7F0000000FFF0001
    },
    {
      hipStreamWaitValueEq,
      0xFF,
      0x11,
      0x25,
      0x11
    },
    {
      // mask will discard bits 8 to 11
      hipStreamWaitValueAnd,
      0xFF,
      0xF4A,
      0xF35,
      0X02
    },
    {
      // mask is set to ignore the sign bit.
      hipStreamWaitValueNor,
      0x7FFFFFFFFFFFFFFF,
      0x7FFFFFFFFFFFF247,
      0x7FFFFFFFFFFFFdbd,
      0x7FFFFFFFFFFFFdb5
    },
    {
      // mask is set to apply NOR for bits 0 to 3.
      hipStreamWaitValueNor,
      0xF,
      0x7E,
      0x7D,
      0x76
    }
  };

  struct TEST_WAIT32_NO_MASK {
    int compareOp;
    int32_t waitValue;
    int32_t signalValueFail;
    int32_t signalValuePass;
  };

  // default mask 0xFFFFFFFF will be used.
  TEST_WAIT32_NO_MASK testCasesNoMask32[] = {
    {
      hipStreamWaitValueGte,
      0x7FFF0001,
      0x7FFF0000,
      0x7FFF0010
    },
    {
      hipStreamWaitValueEq,
      0x7FFFFFFF,
      0x7FFF0000,
      0x7FFFFFFF
    },
    {
      hipStreamWaitValueAnd,
      0x70F0F0F0,
      0x0F0F0F0F,
      0X1F0F0F0F
    },
    {
      hipStreamWaitValueNor,
      0x7AAAAAAA,
      static_cast<int32_t>(0x85555555),
      static_cast<int32_t>(0x9AAAAAAA)
    }
  };

  struct TEST_WAIT64_NO_MASK {
    int compareOp;
    int64_t waitValue;
    int64_t signalValueFail;
    int64_t signalValuePass;
  };

  // default mask 0xFFFFFFFFFFFFFFFF will be used.
  TEST_WAIT64_NO_MASK testCasesNoMask64[] = {
    {
      hipStreamWaitValueGte,
      0x7FFFFFFFFFFF0001,
      0x7FFFFFFFFFFF0000,
      0x7FFFFFFFFFFF0001
    },
    {
      hipStreamWaitValueEq,
      0x7FFFFFFFFFFFFFFF,
      0x7FFFFFFF0FFF0000,
      0x7FFFFFFFFFFFFFFF
    },
    {
      hipStreamWaitValueAnd,
      0x70F0F0F0F0F0F0F0,
      0x0F0F0F0F0F0F0F0F,
      0X1F0F0F0F0F0F0F0F
    },
    {
      hipStreamWaitValueNor,
      0x4724724747247247,
      static_cast<int64_t>(0xbddbddbdbddbddbd),
      static_cast<int64_t>(0xbddbddbdbddbddb3)
    }
  };

  if (!streamWaitValueSupported()) {
    std::cout << " hipStreamWaitValue: not supported on this device , skipping ... \n";
    return;
  }
  std::cout << " hipStreamWaitValue32: testing ... \n";
  std::cout << " hipStreamWaitValue64: testing ... \n";
  hipStream_t stream;
  hipStreamCreate(&stream);

  HIPCHECK(hipExtMallocWithFlags((void **)&signalPtr, 8, hipMallocSignalMemory));
  int64_t* dataPtr64 = (int64_t *) malloc(sizeof(int64_t));
  int32_t* dataPtr32 = (int32_t *) malloc(sizeof(int32_t));
  hipHostRegister(dataPtr64, sizeof(int64_t), 0);
  hipHostRegister(dataPtr32, sizeof(int32_t), 0);

  // We run all test cases twice

  // Run-1: streamWait is blocking (wait conditions is false)
  // Run-2: streamWait is non-blocking (wait condition is true)
  for (int run = 0; run < 2; run++) {
    bool isBlocking = run == 0;

    for (const auto& tc : testCases) {
      *signalPtr = isBlocking ? tc.signalValueFail : tc.signalValuePass;
      *dataPtr64 = DATA_INIT;

      HIPCHECK(hipStreamWaitValue64(stream, signalPtr, tc.waitValue, tc.compareOp, tc.mask));
      HIPCHECK(hipStreamWriteValue64(stream, dataPtr64, DATA_UPDATE, writeFlag));

      if (isBlocking) {
        // Trigger an implict flush and verify stream has pending work.
        HIPASSERT(hipStreamQuery(stream) == hipErrorNotReady);

        // update signal to unblock the wait.
        *signalPtr = tc.signalValuePass;
      }
      // HIPASSERT(hipStreamQuery(stream) == hipSuccess);
      hipStreamSynchronize(stream);
      HIPASSERT(*dataPtr64 == DATA_UPDATE);

      // 32-bit API
      *signalPtr = isBlocking ? tc.signalValueFail : tc.signalValuePass;
      *dataPtr32 = DATA_INIT;

      HIPCHECK(hipStreamWaitValue32(stream, signalPtr, static_cast<int32_t>(tc.waitValue),
                                    tc.compareOp, static_cast<uint32_t>(tc.mask)));
      HIPCHECK(hipStreamWriteValue32(stream, dataPtr32, DATA_UPDATE, writeFlag));

      if (isBlocking) {
        // For DEBUG only
        // usleep(500);
        // HIPASSERT(*dataPtr32 == DATA_INIT);

        // Trigger an implict flush and verify stream has pending work.
        HIPASSERT(hipStreamQuery(stream) == hipErrorNotReady);

        // update signal to unblock the wait.
        *signalPtr = static_cast<int32_t>(tc.signalValuePass);
      }
      hipStreamSynchronize(stream);
      HIPASSERT(*dataPtr32 == DATA_UPDATE);
    }
  }

  std::cout << " hipStreamWaitValue32 with default mask: testing ... \n";
  // Run-1: streamWait is blocking (wait conditions is false)
  // Run-2: streamWait is non-blocking (wait condition is true)
  for (int run = 0; run < 2; run++) {
    bool isBlocking = run == 0;

    for (const auto& tc : testCasesNoMask32) {
      *signalPtr = isBlocking ? tc.signalValueFail : tc.signalValuePass;
      *dataPtr32 = DATA_INIT;

      HIPCHECK(hipStreamWaitValue32(stream, signalPtr, tc.waitValue, tc.compareOp));
      HIPCHECK(hipStreamWriteValue32(stream, dataPtr32, DATA_UPDATE, writeFlag));

      if (isBlocking) {
        // For DEBUG only
        // usleep(500);
        // HIPASSERT(*dataPtr32 == DATA_INIT);

        // Trigger an implict flush and verify stream has pending work.
        HIPASSERT(hipStreamQuery(stream) == hipErrorNotReady);

        // update signal to unblock the wait.
        *signalPtr = tc.signalValuePass;
      }
      hipStreamSynchronize(stream);
      HIPASSERT(*dataPtr32 == DATA_UPDATE);
    }
  }

  std::cout << " hipStreamWaitValue64 with default mask: testing ... \n";
  // Run-1: streamWait is blocking (wait conditions is false)
  // Run-2: streamWait is non-blocking (wait condition is true)
  for (int run = 0; run < 2; run++) {
    bool isBlocking = run == 0;

    for (const auto& tc : testCasesNoMask64) {
      *signalPtr = isBlocking ? tc.signalValueFail : tc.signalValuePass;
      *dataPtr64 = DATA_INIT;

      HIPCHECK(hipStreamWaitValue64(stream, signalPtr, tc.waitValue, tc.compareOp));
      HIPCHECK(hipStreamWriteValue64(stream, dataPtr64, DATA_UPDATE, writeFlag));

      if (isBlocking) {
        // For DEBUG only
        // usleep(500);
        // HIPASSERT(*dataPtr64 == DATA_INIT);

        // Trigger an implict flush and verify stream has pending work.
        HIPASSERT(hipStreamQuery(stream) == hipErrorNotReady);

        // update signal to unblock the wait.
        *signalPtr = tc.signalValuePass;
      }
      hipStreamSynchronize(stream);
      HIPASSERT(*dataPtr64 == DATA_UPDATE);
    }
  }

  // Cleanup
  HIPCHECK(hipFree(signalPtr));
  hipHostUnregister(dataPtr64);
  hipHostUnregister(dataPtr32);
  free(dataPtr64);
  free(dataPtr32);
  hipStreamDestroy(stream);
}


int main() {
  testWrite();
  testWait();
  passed();
}
