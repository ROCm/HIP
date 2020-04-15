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
 * BUILD: %t %s ../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

bool NegativeTests(){
  
  hipEvent_t e;
  // 1. NULL check
  HIPASSERT(hipEventElapsedTime(nullptr, e, e) == hipErrorInvalidValue);
  
  float f;
  // CUDA crashes hence skipping them
#ifndef __HIP_PLATFORM_NVCC__
  // 2. Null events
  HIPASSERT(hipEventElapsedTime(&f, nullptr, e) == hipErrorInvalidHandle);
  HIPASSERT(hipEventElapsedTime(&f, e, nullptr) == hipErrorInvalidHandle);
#endif
  
  // 3. Same event
  HIPCHECK(hipEventCreate(&e));
  HIPASSERT(hipEventElapsedTime(&f, e, e) == hipErrorInvalidHandle);
  
  // 4. Event created using disabled timing
  float timeElapsed = 1.0f;
  hipEvent_t start, stop;
  HIPCHECK(hipEventCreateWithFlags(&start,hipEventDisableTiming));
  HIPCHECK(hipEventCreateWithFlags(&stop,hipEventDisableTiming));
  HIPASSERT(hipEventElapsedTime(&timeElapsed, start, stop) == hipErrorInvalidHandle); 

  // 5. Events created on different device  
  int devCount = 0;
  HIPCHECK(hipGetDeviceCount(&devCount));
  if (devCount > 1){
      // create event on dev=0
      HIPCHECK(hipSetDevice(0));
      hipEvent_t start;
      HIPCHECK(hipEventCreate(&start));

      // create event on dev=1
      HIPCHECK(hipSetDevice(1));
      hipEvent_t stop;
      HIPCHECK(hipEventCreate(&stop));

      HIPCHECK(hipEventRecord(start, nullptr));
      HIPCHECK(hipEventSynchronize(start));

      HIPCHECK(hipEventRecord(stop, nullptr));
      HIPCHECK(hipEventSynchronize(stop));

      float tElapsed = 1.0f;
      HIPASSERT(hipEventElapsedTime(&tElapsed, start, stop) == hipErrorInvalidHandle);
  }
      
  return true;
}

bool PositiveTests(){
  
  // 1. Events on default stream
  {
    hipEvent_t start;
    HIPCHECK(hipEventCreate(&start));

    hipEvent_t stop;
    HIPCHECK(hipEventCreate(&stop));

    HIPCHECK(hipEventRecord(start, nullptr));
    HIPCHECK(hipEventSynchronize(start));

    HIPCHECK(hipEventRecord(stop, nullptr));
    HIPCHECK(hipEventSynchronize(stop));

    float tElapsed = 1.0f;
    HIPCHECK(hipEventElapsedTime(&tElapsed,start,stop));
  }
  
  // 2. Elapsed time works if events are swapped 
  {
    hipEvent_t start;
    HIPCHECK(hipEventCreate(&start));

    hipEvent_t stop;
    HIPCHECK(hipEventCreate(&stop));

    HIPCHECK(hipEventRecord(start, nullptr));
    HIPCHECK(hipEventSynchronize(start));

    HIPCHECK(hipEventRecord(stop, nullptr));
    HIPCHECK(hipEventSynchronize(stop));

    float tElapsed = 1.0f;
    HIPCHECK(hipEventElapsedTime(&tElapsed, stop, start));
  }
  
  // 3. Events recorded on different streams
  {
    hipEvent_t start, stop;
    hipStream_t S1,S2;
    
    HIPCHECK(hipStreamCreate(&S1));
    HIPCHECK(hipStreamCreate(&S2));
    HIPCHECK(hipEventCreate(&start));
    HIPCHECK(hipEventCreate(&stop));
    
    HIPCHECK(hipEventRecord(start,S1));
    HIPCHECK(hipEventRecord(stop,S2));
    
    HIPCHECK(hipEventSynchronize(start));
    HIPCHECK(hipEventSynchronize(stop));
    
    float tElapsed = 1.0f;
    HIPCHECK(hipEventElapsedTime(&tElapsed, start, stop));
  }
  
  
  return true;
}

int main(){
  
  bool status = true;
  status &= NegativeTests();
  status &= PositiveTests();
  
  if (status){
    passed();
  }
  return 0;
}