/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

void runTest(int srcDevice, int dstDevice) {
  std::cout<<"Between Device "<<srcDevice<<" and Device "<<dstDevice<<std::endl;
  for (int p2p_attr_idx = hipDevP2PAttrPerformanceRank;
        p2p_attr_idx <= hipDevP2PAttrHipArrayAccessSupported; ++p2p_attr_idx) {
    int value = -1;
    HIPCHECK(hipDeviceGetP2PAttribute(&value, static_cast<hipDeviceP2PAttr>(p2p_attr_idx),
                                      srcDevice, dstDevice));
    std::cout<<"Attr["<<p2p_attr_idx<<"] is "<<value<<std::endl;
  }
}

int main() {

  int count = -1;
  HIPCHECK(hipGetDeviceCount(&count))

  if (count >= 2){
    for (int dev_idx = 0; dev_idx < (count-1); ++dev_idx) {
      runTest(dev_idx, 1 + dev_idx);
    }
  } else {
    std::cout<<"Not enough GPUs to run the single GPU tests"<<std::endl;
  }

  passed();
}
