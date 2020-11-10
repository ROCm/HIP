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
#include <chrono>
#include "test_common.h"

using namespace std;

int main(int argc, char* argv[]) {
  hipStream_t stream;
  vector<uint32_t> cuMask(6);
  stringstream ss;

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
  cout << "info: running on bus " << "0x" << props.pciBusID << " " << props.name <<
      " with " << props.multiProcessorCount << " CUs" << endl;

  std::string str_out, str_err = "hipErrorInvalidValue";

  char* gCUMask = NULL;
  string globalCUMask("");

  gCUMask = getenv("ROC_GLOBAL_CU_MASK");
  if (gCUMask != NULL && gCUMask[0] != '\0') {
    globalCUMask.assign(gCUMask);
    for_each(globalCUMask.begin(), globalCUMask.end(), [](char & c) {
      c = ::tolower(c);
    });
  }

  // make a default CU mask bit-array where all CUs are active
  // this default mask is expected to be returned when there is no
  // custom or global CU mask defined
  std::vector<uint32_t> defaultCUMask;
  uint32_t temp = 0;
  uint32_t bit_index = 0;
  for (uint32_t i = 0; i < props.multiProcessorCount; i++) {
    temp |= 1UL << bit_index;
    if (bit_index >= 32) {
      defaultCUMask.push_back(temp);
      temp = 0;
      bit_index = 0;
      temp |= 1UL << bit_index;
    }
    bit_index += 1;
  }
  if (bit_index != 0) {
    defaultCUMask.push_back(temp);
  }

  str_out = hipGetErrorString(hipExtStreamGetCUMask(0, cuMask.size(), 0));
  if ((str_err.compare(str_out)) != 0) {
    failed("hipExtStreamGetCUMask returned wrong error code!");
  }

  str_out = hipGetErrorString(hipExtStreamGetCUMask(0, 0, &cuMask[0]));
  if ((str_err.compare(str_out)) != 0) {
    failed("hipExtStreamGetCUMask returned wrong error code!");
  }

  // read the CU mask for the null stream, when this call returns
  // the content of cuMask should be either the global CU mask (if it is defined) or
  // the defautl CU mask where all CUs are active
  HIPCHECK(hipExtStreamGetCUMask(0, cuMask.size(), &cuMask[0]));

  ss << std::hex;
  for (int i = cuMask.size() - 1; i >= 0; i--) {
    ss << cuMask[i];
  }

  // remove extra 0 from ss if any present
  size_t found = ss.str().find_first_not_of("0");
  if (found != string::npos) {
    ss.str(ss.str().substr(found, ss.str().length()));
  }

  if (globalCUMask.size() > 0) {
    if (ss.str().compare(globalCUMask) != 0) {
      failed("Error! expected the CU mask: %s but got %s", globalCUMask.c_str(), ss.str().c_str());
    }
  } else {
    for (auto i = 0 ; i < min(cuMask.size(), defaultCUMask.size()); i++) {
      if (cuMask[i] != defaultCUMask[i]) {
        failed("Error! expected the CU mask: %u but got %u", defaultCUMask[i], cuMask[i]);
      }
    }
  }

  cout << "info: CU mask for the default stream is: 0x" << ss.str().c_str() << endl;

  vector<uint32_t> cuMask1(defaultCUMask);
  cuMask1[0] = 0xe;

  HIPCHECK(hipExtStreamCreateWithCUMask(&stream, cuMask1.size(), cuMask1.data()));
  ss.str("");
  for (int i = cuMask1.size() - 1; i >= 0; i--) {
    ss << cuMask1[i];
  }
  cout << "info: setting a custom CU mask 0x" << ss.str() << " for stream " << stream << endl;

  HIPCHECK(hipExtStreamGetCUMask(stream, cuMask.size(), &cuMask[0]));


  if (!gCUMask) {
    for (int i = 0; i < cuMask1.size(); i++) {
      if (cuMask1[i] != cuMask[i]) {
        HIPCHECK(hipStreamDestroy(stream));
        failed("Error! expected the CU mask: %u but got %u", cuMask1[i], cuMask[i]);
      }
    }
  }

  ss.str("");
  for (int i = cuMask.size() - 1; i >= 0; i--) {
    ss << cuMask[i];
  }
  found = ss.str().find_first_not_of("0");
  if (found != string::npos) {
    ss.str(ss.str().substr(found, ss.str().length()));
  }

  cout << "info: reading back CU mask 0x" << ss.str() << " for stream " << stream << endl;

  HIPCHECK(hipStreamDestroy(stream));

  passed();
}
