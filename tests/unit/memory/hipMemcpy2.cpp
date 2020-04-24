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
 * TEST_NAMED: %t hipMemcpyDtoDAsync-simple --async
 * HIT_END
 */
 
/*
 * Unit test for hipMemcpyHtoD(), hipMemcpyDtoH(), hipMemcpyDtoD(), hipMemcpyHtoDAsync(), hipMemcpyDtoHAsync(), hipMemcpyDtoDAsync()
 * 
 * -------------------------------------------------------------------------------------------------------------------------------------------------------|
 * Negative test : Null check , Invalid size, Memory kind does not match passed dst & src ptr                                                             |
 * -------------------------------------------------------------------------------------------------------------------------------------------------------|
 * Positive test : simpleMemcpyTest() : Single GPU | H2D->D2H, H2D->Kernel->D2H | Pinned/UnPinned host Memory, different size/data types                  |
 * -------------------------------------------------------------------------------------------------------------------------------------------------------|
 *               : simpleMemcpyD2DTest() : Multi GPU | H2D -> D -> D -> D2H | Copy GPU-0<-->GPU-1, Pinned/UnPinned host Memory, different size/data types |
 * -------------------------------------------------------------------------------------------------------------------------------------------------------|
 */
 
#include "test_common.h"
#include <vector>

// Globals

#define SYNC  0
#define ASYNC 1

int gMemcpyType = -1;
 
 // ------------------------------ Utility Functions --------------------------
template<typename T>
void updateHostArray(T* arr, size_t arraySize, T value){
  for(int i = 0; i< arraySize; ++i)
    arr[i] = value;
}

hipError_t memcopyDtoH(void* dst, void* src, size_t sizeBytes, hipStream_t stream = NULL) {
    hipError_t status= hipSuccess;
    
    if(gMemcpyType == SYNC)
    {
      status = hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), sizeBytes);
    }
    else if (gMemcpyType == ASYNC) {
        status = hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), sizeBytes, stream);
        if(status == hipSuccess)
          status = hipStreamSynchronize(stream);
    }
    else{
      status = hipErrorInvalidValue;
    } 
    return status;
}

hipError_t memcopyHtoD(void* dst, void* src, size_t sizeBytes, hipStream_t stream = NULL) {
    hipError_t status= hipSuccess;
    
    if(gMemcpyType == SYNC)
    {
      status = hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, sizeBytes);
    }
    else if (gMemcpyType == ASYNC) {
        status = hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), src, sizeBytes, stream);
        if(status == hipSuccess)
          status = hipStreamSynchronize(stream);
    }
    else{
      status = hipErrorInvalidValue;
    } 
    return status;
}

 hipError_t memcopyDtoD(void* dst, void* src, size_t sizeBytes, hipStream_t stream=NULL) {
    hipError_t status= hipSuccess;
    
    switch(gMemcpyType){
    case SYNC: 
          status = hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst), reinterpret_cast<hipDeviceptr_t>(src), sizeBytes);
          break;
    case ASYNC:
          status = hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), reinterpret_cast<hipDeviceptr_t>(src), sizeBytes, stream);
          if(status == hipSuccess)
            status = hipStreamSynchronize(stream); 
          break;
    default : status  = hipErrorInvalidValue;
    }
    return status;
}
 
 // ------------------------------- End of Utility ----------------------------
 template<typename T>
 bool simpleMemcpyD2DTest(size_t arrSize, bool pinnedHostMemeory, int gpu0, int gpu1, T initData){
  T *A1_h0, *A2_h0, *A1_d0, *A2_d0;
  T *A1_h1, *A2_h1, *A1_d1, *A2_d1;

  size_t numBytes = sizeof(T) * arrSize;
  
  // set device 0 first
  HIPCHECK(hipSetDevice(gpu0));

  // Allocate host memory (pinned/unpinned)
  HipTest::initArraysForHost(&A1_h0, &A2_h0, numBytes, pinnedHostMemeory);
  // Allocate device memory
  HipTest::initArrays(&A1_d0, &A2_d0, numBytes);
  
  hipStream_t stream0 = NULL;
  if(gMemcpyType != SYNC){
    HIPCHECK(hipStreamCreate(&stream0));
  }
  
   // set device 1 first
  HIPCHECK(hipSetDevice(gpu1));

  // Allocate host memory (pinned/unpinned)
  HipTest::initArraysForHost(&A1_h1, &A2_h1, numBytes, pinnedHostMemeory);
  // Allocate device memory
  HipTest::initArrays(&A1_d1, &A2_d1, numBytes);
  
  hipStream_t stream1 = NULL;
  if(gMemcpyType != SYNC){
    HIPCHECK(hipStreamCreate(&stream1));
  } 
  
  /*
   * GPU0 -> GPU1
   * Memcpy between H-> D(gpu0) -> D(gpu1) ->H(pinned & unpinned)
   */
  updateHostArray(A1_h0, arrSize, initData++);
  HIPCHECK(memcopyHtoD(A1_d0, A1_h0, numBytes, stream0));
  HIPCHECK(memcopyDtoD(A1_d1, A1_d0, numBytes,stream0));
  HIPCHECK(memcopyDtoH(A2_h0, A1_d1, numBytes, stream0)); 
  HIPASSERT(A1_h0[arrSize-1] == A2_h0[arrSize-1]); 
  
  /*
   * GPU1 -> GPU0
   * Memcpy between H-> D(gpu1) -> D(gpu0) ->H(pinned & unpinned)
   */
  updateHostArray(A2_h1, arrSize, initData++);
  HIPCHECK(memcopyHtoD(A2_d1, A2_h1, numBytes, stream1));
  HIPCHECK(memcopyDtoD(A2_d0, A2_d1, numBytes, stream1));
  HIPCHECK(memcopyDtoH(A2_h0, A2_d0, numBytes, stream1)); 
  HIPASSERT(A2_h1[arrSize-1] == A2_h0[arrSize-1]);   
 
  // free memory
  HipTest::freeArrays(A1_d0, A2_d0);
  HipTest::freeArrays(A1_d1, A2_d1);
  HipTest::freeArraysForHost(A1_h0, A2_h0, pinnedHostMemeory);
  HipTest::freeArraysForHost(A1_h1, A2_h1, pinnedHostMemeory);  
  return true;
} 
 
bool runSimpleD2DTests(){
  bool status = true;
  
  int devCount = 0;
  HIPCHECK(hipGetDeviceCount(&devCount));
  if(devCount == 0)
    return false;
  
  std::vector<int> Bytes = {4, 255, 20, 20, 4, 4, 512*512, 255, 20, 1024, 4, 4, 1024*1024 };
  
  // Will go over all device and perform D2D memcpy with different size and different data types
  for(int gpu0 = 0; gpu0 < devCount; ++gpu0){
    for(int gpu1 = 0; gpu1 < devCount; ++gpu1){
      // pinned and unpinned D2D memcopy using different data types
      for(auto it : Bytes){
        status  &= simpleMemcpyD2DTest<float>(it, true /*pinned*/, gpu0, gpu1, 3.0f);
        status  &= simpleMemcpyD2DTest<int>(it, true /*pinned*/, gpu0, gpu1, 5);
        
        status  &= simpleMemcpyD2DTest<float>(it, false /*un-pinned*/, gpu0, gpu1, 4.0f);
        status  &= simpleMemcpyD2DTest<int>(it, false /*un-pinned*/, gpu0, gpu1, 4);
        
        status  &= simpleMemcpyD2DTest<double>(it, true /*pinned*/, gpu0, gpu1, 5.01f);
        status  &= simpleMemcpyD2DTest<double>(it, false /*un-pinned*/, gpu0, gpu1, 4.02f);
      }
    }
  }
  return status;
}

template<typename T>
bool simpleMemcpyTest(size_t arrSize, bool pinnedHostMemeory, int gpu, T initData){
  T *A1_h, *A2_h;
  T *A1_d, *A2_d;
   
  size_t numBytes = sizeof(T) * arrSize;
  
  // set device first
  HIPCHECK(hipSetDevice(gpu));
  
  // Allocate host memory (pinned/unpinned)
  HipTest::initArraysForHost(&A1_h, &A2_h, numBytes, pinnedHostMemeory);
  // Allocate device memory
  HipTest::initArrays(&A1_d, &A2_d, numBytes);
  
  hipStream_t stream = NULL;
  if(gMemcpyType != SYNC){
    HIPCHECK(hipStreamCreate(&stream));
  }
     
  /*
  * Memcpy between H->D->H(pinned & unpinned)
  */
  updateHostArray(A1_h, arrSize, initData++);
  HIPCHECK(memcopyHtoD(A1_d, A1_h, numBytes, stream));
  HIPCHECK(memcopyDtoH(A2_h, A1_d, numBytes,stream));
  HIPASSERT(A1_h[arrSize-1] == A2_h[arrSize-1]); 
  
  // use kernel 
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, arrSize);
  updateHostArray(A1_h, arrSize, initData++);
  HIPCHECK(memcopyHtoD(A1_d, A1_h, numBytes, stream));
  hipLaunchKernelGGL(HipTest::addCount, dim3(blocks), dim3(threadsPerBlock), 0, stream,
                           static_cast<const T*>(A1_d), A2_d, arrSize, 1);
                           
  HIPCHECK(memcopyDtoH(A2_h, A2_d, numBytes,stream));
  HIPASSERT(++A1_h[arrSize-1] == A2_h[arrSize-1]); 
  
  // free memory
  HipTest::freeArrays(A1_d, A2_d);
  HipTest::freeArraysForHost(A1_h, A2_h, pinnedHostMemeory);  
  return true;
} 
 
bool runSimpleTests(){
  bool status = true;
  
  int devCount = 0;
  HIPCHECK(hipGetDeviceCount(&devCount));
  if(devCount == 0)
    return false;
  
  std::vector<int> Bytes = {4, 255, 20, 20, 4, 4, 512*512, 255, 20, 1024, 4, 4, 1024*1024 };
  
  int gpu = 0;
  // Will go over all device and perform D2H memcpy with different size and different data types
  do{
    // pinned and unpinned D2H memcopy using different data types
    for(auto it : Bytes){
      status  &= simpleMemcpyTest<float>(it, true /*pinned*/, gpu, 3.0f);
      status  &= simpleMemcpyTest<int>(it, true /*pinned*/, gpu, 5);
      status  &= simpleMemcpyTest<float>(it, false /*un-pinned*/, gpu, 4.0f);
      status  &= simpleMemcpyTest<int>(it, false /*un-pinned*/, gpu, 4);
      status  &= simpleMemcpyTest<double>(it, true /*pinned*/, gpu, 5.01f);
      status  &= simpleMemcpyTest<double>(it, false /*un-pinned*/, gpu, 4.02f);
    }
    gpu++;
  }while(gpu < devCount);
  return status;
}
 
bool PositiveTests(){
  bool status = true;
  
  status &= runSimpleTests();
  status &= runSimpleD2DTests();
  
  return status;
}
 
 // Parse arguments specific to this test.
bool parseMyArguments(int argc, char* argv[]) {
    int more_argc = HipTest::parseStandardArguments(argc, argv, false);

    gMemcpyType = SYNC;
    
    // parse args for this test:
    for (int i = 1; i < more_argc; i++) {
        const char* arg = argv[i];

        if (!strcmp(arg, "--async")) {
            gMemcpyType = ASYNC;

        }
        else {
            failed("BAD Argument(s)");
        }
    }
    return true;
};

bool NegativeTests(){

  float A1 = 100.3f;
  float *A1d, *A2d;
  HIPCHECK(hipMalloc((void**)&A1d, sizeof(float)));
  HIPCHECK(hipMalloc((void**)&A2d, sizeof(float)));
  
  // NULL check 
  HIPASSERT(memcopyHtoD(nullptr, &A1, sizeof(float)) == hipErrorInvalidValue);
  HIPASSERT(memcopyHtoD(&A1d, nullptr, sizeof(float)) == hipErrorInvalidValue);
  
  HIPASSERT(memcopyDtoH(nullptr, &A1d, sizeof(float)) == hipErrorInvalidValue);
  HIPASSERT(memcopyDtoH(&A1, nullptr, sizeof(float)) == hipErrorInvalidValue);
  
  HIPASSERT(memcopyDtoD(nullptr, &A1d, sizeof(float)) == hipErrorInvalidValue);
  HIPASSERT(memcopyDtoD(&A1d, nullptr, sizeof(float)) == hipErrorInvalidValue);
  
  // Invalid size
  // TODO : HIP yet to support
#ifdef __HIP_PLATFORM_NVCC__  
  HIPASSERT(memcopyHtoD(A1d, &A1, -1) == hipErrorInvalidValue);
  HIPASSERT(memcopyDtoH(&A1, A1d, -1) == hipErrorInvalidValue);
  HIPASSERT(memcopyDtoD(A1d, A1d, -1) == hipErrorInvalidValue);
      
  HIPASSERT(memcopyHtoD(A1d, &A1, sizeof(float)*1024) == hipErrorInvalidValue);   
  HIPASSERT(memcopyDtoH(&A1, A1d, sizeof(float)*1024) == hipErrorInvalidValue); 
  HIPASSERT(memcopyDtoD(A1d, A1d, sizeof(float)*1024) == hipErrorInvalidValue);   
#endif 
  // clear allocations  
  HipTest::freeArrays(A1d, A2d);

  return true;
}

int main(int argc, char* argv[]){
 bool status = true;
 if(!parseMyArguments(argc, argv))
     return 0; // Parsing error hence return early

#ifdef __HIP_PLATFORM_NVCC__
   // TODO : HIP Yet to support negative scenarios
 status &= NegativeTests();
#endif
 status &= PositiveTests();
 
 if(status)
   passed();
 
 return 0;
}
 
 