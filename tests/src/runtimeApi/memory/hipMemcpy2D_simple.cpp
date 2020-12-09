/*
Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.
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
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */
/* Intension : Multidevice test to verify hipMemcpy2D(Async) behavior. Test verifies hipMemcpy2D behavior
               when one of the device memory is not owned by current device.
               i)  H2D & D2H -> Device memory is not owned by current device
               ii) D2D -> source memory is not owned by current device
   Note : To make it simple at present checking API functionality not validating values.
 */
#include "test_common.h"
using namespace std;
//#defines
#define Nrows 8
#define Ncols 8
// Globals
int Nbytes = Nrows * Ncols * sizeof(char);
bool Copy2D(bool syncCopy, hipMemcpyKind kind) {
  char* hPtr = nullptr;
  char* devPtr = nullptr;
  size_t pitch;
  int canAccess = 0;
  HIPCHECK(hipDeviceCanAccessPeer(&canAccess, 1, 0));
  if (!canAccess) {
    cout << "Exit early as Non-Peer config\n";
    // Returning true as test should not be executed on non-peer configs
    return true;
  }
  hPtr = (char*)malloc(Nrows * Ncols);
  HIPCHECK(hipSetDevice(0));
  HIPCHECK(hipMallocPitch((void**)&devPtr, (size_t*)&pitch, Ncols * sizeof(char), Nrows));
  HIPCHECK(hipSetDevice(1));
  if (syncCopy) {
    // API under test : Copy triggered from dev1 but device memory allocated on dev0
    if (kind == hipMemcpyHostToDevice) {
      HIPCHECK(hipMemcpy2D(devPtr, pitch, hPtr, Ncols * sizeof(char), Ncols * sizeof(char), Nrows,
                           hipMemcpyHostToDevice));
    } else if (kind == hipMemcpyDeviceToHost) {
      HIPCHECK(hipMemcpy2D(hPtr, Ncols * sizeof(char), devPtr, pitch, Ncols * sizeof(char), Nrows,
                           hipMemcpyDeviceToHost));
    } else if (kind == hipMemcpyDeviceToDevice) {
      char* devPtr1;
      size_t pitch1;
      HIPCHECK(hipMallocPitch((void**)&devPtr1, (size_t*)&pitch1, Ncols * sizeof(char), Nrows));
      // API under test : Copy triggered from dev1 but device memory allocated on dev0
      HIPCHECK(hipMemcpy2D(devPtr1, pitch1, devPtr, pitch, Ncols * sizeof(char), Nrows,
                           hipMemcpyDeviceToDevice));
      HIPCHECK(hipFree(devPtr1));
    }
  } else {
    hipStream_t pStream;
    HIPCHECK(hipStreamCreate(&pStream));
    if (kind == hipMemcpyHostToDevice) {
      HIPCHECK(hipMemcpy2DAsync(devPtr, pitch, hPtr, Ncols * sizeof(char), Ncols * sizeof(char),
                                Nrows, hipMemcpyHostToDevice, pStream));
    } else if (kind == hipMemcpyDeviceToHost) {
      HIPCHECK(hipMemcpy2DAsync(hPtr, Ncols * sizeof(char), devPtr, pitch, Ncols * sizeof(char),
                                Nrows, hipMemcpyDeviceToHost, pStream));
    } else if (kind == hipMemcpyDeviceToDevice) {
      char* devPtr1;
      size_t pitch1;
      HIPCHECK(hipMallocPitch((void**)&devPtr1, (size_t*)&pitch1, Ncols * sizeof(char), Nrows));
      HIPCHECK(hipMemcpy2DAsync(devPtr1, pitch1, devPtr, pitch, Ncols * sizeof(char), Nrows,
                                hipMemcpyDeviceToDevice, pStream));
      HIPCHECK(hipFree(devPtr1));
    }
    HIPCHECK(hipStreamSynchronize(pStream));
    HIPCHECK(hipStreamDestroy(pStream));
  }
  // Free allocations
  HIPCHECK(hipFree(devPtr));
  free(hPtr);
  return true;
}
int main() {
  int numDev = 0;
  HIPCHECK(hipGetDeviceCount(&numDev));
  if (numDev == 0) {
    failed("No device found");
  } else if (numDev == 1) {
    passed();
  }
  bool status = true;
  status &= Copy2D(true, hipMemcpyHostToDevice);     // Sync copy, H2D
  status &= Copy2D(false, hipMemcpyHostToDevice);    // Async copy, H2D
  status &= Copy2D(true, hipMemcpyDeviceToHost);     // Sync copy, D2H
  status &= Copy2D(false, hipMemcpyDeviceToHost);    // Async copy, D2H
  status &= Copy2D(true, hipMemcpyDeviceToDevice);   // Sync copy, D2D
  status &= Copy2D(false, hipMemcpyDeviceToDevice);  // Async copy, D2D
  // Validate final result
  if (!status) {
    failed("Failed");
  }
  passed();
}