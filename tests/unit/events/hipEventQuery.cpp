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

#include <thread>
#include <chrono>
#include <atomic>


bool NegativeTests(){
  hipError_t status ;
  status = hipEventQuery(nullptr);

  hipEvent_t et;
  HIPCHECK(hipEventCreate(&et));
  status = hipEventQuery(et);
  
  return true;
}

__global__ void kernel(int *input){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // dummy operation
  if( tid == 0){
    *input = 0;
    for(int i = 1; i<= 100000000; i++){
      *input += 1;
    }
  }
}

enum class ExecState
{
   EXEC_NOT_STARTED,
   EXEC_STARTED,
   EXEC_CB_STARTED,
   EXEC_STATE_VERIFIED,
   EXEC_CB_FINISHED,
   EXEC_FINISHED
};

std::atomic<ExecState> gData(ExecState::EXEC_NOT_STARTED);

void cbFunction(hipStream_t stream, hipError_t status, void* userData)
{
  if(gData.load() != ExecState::EXEC_STARTED)
    return; // Error hence return early
    
    gData.store(ExecState::EXEC_CB_STARTED);
    int count = 10000;
    while(count--){
      if(gData.load() ==  ExecState::EXEC_STATE_VERIFIED)
      {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  gData.store(ExecState::EXEC_CB_FINISHED);
}

bool PositiveTests(){
    size_t *C_h, *C_d;
    size_t sizeBytes = sizeof(size_t);
    
    HIPCHECK(hipMalloc(&C_d, sizeBytes));
    HIPCHECK(hipHostMalloc(&C_h, sizeBytes));

    hipStream_t stream;
    HIPCHECK(hipStreamCreateWithFlags(&stream, 0x0));
    
    hipEvent_t end;
    HIPCHECK(hipEventCreate(&end));
    
    gData.store(ExecState::EXEC_STARTED);
       
    HIPCHECK(hipStreamAddCallback(stream, cbFunction, nullptr, 0));
    
    HIPCHECK(hipEventRecord(end, stream));
    
    // Event should not be ready as callback is still running
    while(1){
      HIPASSERT(hipEventQuery(end) == hipErrorNotReady);
      if(gData.load() == ExecState::EXEC_CB_STARTED){
        gData.store(ExecState::EXEC_STATE_VERIFIED);
        break;
      }
    }
    
    HIPCHECK(hipEventSynchronize(end));
    HIPASSERT(hipEventQuery(end) == hipSuccess);
    
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