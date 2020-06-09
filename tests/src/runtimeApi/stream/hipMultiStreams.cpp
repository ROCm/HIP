/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include <iostream>
#include <vector>
#include "test_common.h"

using namespace std;

__global__ void kernel_do_nothing() {
  // empty kernel
}

int main(int argc, char* argv[]) {

  constexpr int nLoops = 100000;
  constexpr int nStreams = 2;
  vector<hipStream_t> streams(nStreams);

  int nGpu = 0;
  HIPCHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
      cout << "info: didn't find any GPU! skipping the test!\n";
      passed();
      return 0;
  }

  static int device = 0;
  HIPCHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, device));
  cout << "info: running on bus " << "0x" << props.pciBusID << " " << props.name << endl;

 for (int i = 0; i < nStreams; i++) {
   HIPCHECK(hipStreamCreate(&streams[i]));
 }

  for (int k = 0; k <= nLoops; ++k) {
    HIPCHECK(hipDeviceSynchronize());

    // Launch kernel with default stream
    hipLaunchKernelGGL((kernel_do_nothing), dim3(1), dim3(1), 0, 0);

    // Launch kernel on all streams
    for (int i = 0; i < nStreams; i++) {
      hipLaunchKernelGGL((kernel_do_nothing), dim3(1), dim3(1), 0, streams[i]);
    }

    // Sync stream 1
    HIPCHECK(hipStreamSynchronize(streams[0]));

    if (k % 10000 == 0 || k == nLoops) {
      cout << "Info: Iteration = " << k << endl;
    }
  }

  HIPCHECK(hipDeviceSynchronize());

  // Clean up
  for (int i = 0; i < nStreams; i++) {
    HIPCHECK(hipStreamDestroy(streams[i]));
  }

  HIPCHECK(hipDeviceReset());

  passed();
}
