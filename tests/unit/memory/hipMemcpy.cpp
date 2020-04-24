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
 * TEST_NAMED: %t hipMemcpyWithStream-simple --syncWithStream
 * TEST_NAMED: %t hipMemcpyAsync-simple --async
 * HIT_END
 */

/*
 * Unit test for hipMemcpy(), hipMemcpyWithStream(), hipMemcpyAsync()
 * 
 * ------------------------------------------------------------------------------------------------------------------------------------------
 * Negative test : Null check , Invalid size, Memory kind does not match passed dst & src ptr
 * ------------------------------------------------------------------------------------------------------------------------------------------
 * Positive test : SimpleMemcopy() : Single GPU | H2H, H2D, D2H, D2D | Pinned/UnPinned host Memory 
 * ------------------------------------------------------------------------------------------------------------------------------------------
 *               : SimpleMemcpy2() : Single GPU | H2D -> Kernel -> D2H | Kernel operation included
 * ------------------------------------------------------------------------------------------------------------------------------------------
 *               : SimpleMemcopyWithDefaultKind() : Single GPU | H2H, H2D, D2H, D2D | Pinned/UnPinned host Memory with hipMemcpyDefault kind 
 * ------------------------------------------------------------------------------------------------------------------------------------------
 *               : MultiDeviceInterLeavedCopy(): Multi GPU | H2H, H2D, D2H, D2D | lnterleaved copies between GPUs
 * ------------------------------------------------------------------------------------------------------------------------------------------
 *               : MultiDeviceCopy(): Multi GPU | H2H, H2D, D2H, D2D | switch device -> memcopy
 * ------------------------------------------------------------------------------------------------------------------------------------------
 *               : stressTest() : Test Memcopy on single & milti device with different sizes, different data types.
 * ------------------------------------------------------------------------------------------------------------------------------------------ 
 */

#include "test_common.h"
#include <vector>

bool isMultiDevice(){
  int devCount =0;
  HIPCHECK(hipGetDeviceCount(&devCount));
  return (devCount > 1) ? true : false;
}

template<typename T>

void updateHostArray(T* arr, size_t arraySize, T value){
  for(int i = 0; i< arraySize; ++i)
    arr[i] = value;
}

bool getGpuIdRand(int& gpu0, int& gpu1){
  int devCount;
  HIPCHECK(hipGetDeviceCount(&devCount));
  
  gpu0 = 0;
  gpu1 = 0;
  
  if(devCount <= 1)
    return true;
  
  std::vector<bool> gpuTbl(devCount,false);
  while(1){
    int gpuId = rand()%devCount;
    if(gpuTbl[gpuId] == false){
      if(gpu0 == INT_MAX)
        gpu0 = gpuId;
      else if(gpu1 == INT_MAX)
        gpu1 = gpuId;
      else
        break;
    }
  }
  return true;
}


#define SYNC  0
#define ASYNC 1
#define SYNC_WITH_STREAM 2

int gMemcpyType = -1;

 hipError_t memcopy(void* dst, const void* src, size_t sizeBytes, enum hipMemcpyKind kind, hipStream_t stream = NULL) {
    hipError_t status= hipSuccess;
    
    if(gMemcpyType == SYNC)
    {
      status = hipMemcpy(dst, src, sizeBytes, kind);
    }
    else if(gMemcpyType == SYNC_WITH_STREAM){
        status = hipMemcpyWithStream(dst, src, sizeBytes, kind, stream);
    }
    else if (gMemcpyType == ASYNC) {
        status = hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
        if(status == hipSuccess)
          status = hipStreamSynchronize(stream);
    }
    else{
      status = hipErrorInvalidValue;
    } 
    return status;
}

 // Parse arguments specific to this test.
bool parseMyArguments(int argc, char* argv[]) {
    int more_argc = HipTest::parseStandardArguments(argc, argv, false);
    
    gMemcpyType = SYNC;//default
    
    // parse args for this test:
    for (int i = 1; i < more_argc; i++) {
        const char* arg = argv[i];

        if (!strcmp(arg, "--async")) {
            gMemcpyType = ASYNC;

        }
        else if (!strcmp(arg, "--syncWithStream")){
          gMemcpyType = SYNC_WITH_STREAM;
        }
         else {
            failed("Bad Input");
        }
    }
    return true;
}

/*
 * NegativeTests() : 
 *      Test includes: 
 *          1) NULL checks
 *          2) Invalid memcopy size 
 *          3) Memory kind and dst/src pointer mismatch
 *             Note: Behavior is undefined hence not verifying results.  
 */
bool NegativeTests(){

  float A1 = 100.3f;
  float A2 = 20.4f;
  float *A1d, *A2d;
  HIPCHECK(hipMalloc((void**)&A1d, sizeof(float)));
  HIPCHECK(hipMalloc((void**)&A2d, sizeof(float)));
  
  // NULL check  
  HIPASSERT(memcopy(nullptr, &A1, sizeof(float), hipMemcpyHostToHost, NULL) == hipErrorInvalidValue);
  HIPASSERT(memcopy(&A1, nullptr, sizeof(float), hipMemcpyHostToHost, NULL) == hipErrorInvalidValue);

#ifdef __HIP_PLATFORM_NVCC__ 
  // TODO : HIP yet to support 
  // Invalid size
  HIPASSERT(memcopy(A1d, &A1, -1, hipMemcpyHostToDevice, NULL) == hipErrorInvalidValue);  
  //status = hipMemcpy(A1d, &A1, 0, hipMemcpyHostToDevice); // Not invalid value on nvcc 
  HIPASSERT(memcopy(A1d, &A1, sizeof(float)*1024, hipMemcpyHostToDevice, NULL) == hipErrorInvalidValue); 
#endif
  
  /*
   * Test : Invalid memcpy kind
   * Expected behavior : Calling hipMemcpy API with destination and source pointers which 
   *                     do not match passed memcpy kind results in error.
   */
   // Note: Since behavior is undefined hence return is not validated. 
   //       Intension of the test is to make sure API call does not crash.
   
  memcopy(A1d, &A1, sizeof(float), hipMemcpyHostToHost, NULL);
  memcopy(A1d, A2d, sizeof(float), hipMemcpyHostToHost, NULL);
  memcopy(&A1, A1d, sizeof(float), hipMemcpyHostToHost, NULL);

  memcopy(&A2, &A1, sizeof(float), hipMemcpyHostToDevice, NULL);
  memcopy(A1d, A2d, sizeof(float), hipMemcpyHostToDevice, NULL);
  memcopy(&A1, A1d, sizeof(float), hipMemcpyHostToDevice, NULL);
  
  memcopy(&A2, &A1, sizeof(float), hipMemcpyDeviceToHost, NULL);
  memcopy(A2d, A1d, sizeof(float), hipMemcpyDeviceToHost, NULL);
  memcopy(A1d, &A1, sizeof(float), hipMemcpyDeviceToHost, NULL);
  
  memcopy(&A2, &A1, sizeof(float), hipMemcpyDeviceToDevice, NULL);
  memcopy(&A1, A1d, sizeof(float), hipMemcpyDeviceToDevice, NULL);
  memcopy(A1d, &A1, sizeof(float), hipMemcpyDeviceToDevice, NULL);
 
  // clear allocations  
  HipTest::freeArrays(A1d, A2d);

  return true;
}

/*
 * SimpleMemcopy() : 
 *      Test includes: 
 *          It covers all possible valid functionalities e.g. H2H, H2D, D2H & D2D, uses pinned/unpinned host memory   
 */
template<typename T>
bool SimpleMemcopy(int arrSize, T initData){
  size_t numBytes = sizeof(T) * arrSize;
  
  // Variables
  T *A1_d, *A2_d;
  T *A1_h, *A2_h;
  T *A1_pin_h,*A2_pin_h;
  
  // Allocate host memory 
  HipTest::initArraysForHost(&A1_h, &A2_h, numBytes, false);
  HipTest::initArraysForHost(&A1_pin_h, &A2_pin_h, numBytes, true);
  // Allocate device memory
  HipTest::initArrays(&A1_d, &A2_d, numBytes);
  
  hipStream_t stream;
  if(gMemcpyType == SYNC){
    stream = NULL;
  }
  else{
    HIPCHECK(hipStreamCreate(&stream));
  }
  
  /*
   * Memcpy between host memory (pinned & unpinned)
   */
  updateHostArray(A1_h, arrSize, initData); 
  HIPCHECK(memcopy(A2_h,     A1_h,     numBytes, hipMemcpyHostToHost, stream));
  HIPCHECK(memcopy(A2_pin_h, A2_h,     numBytes, hipMemcpyHostToHost, stream));
  HIPCHECK(memcopy(A1_pin_h, A2_pin_h, numBytes, hipMemcpyHostToHost, stream));
  HIPASSERT(A1_h[arrSize-1] == A1_pin_h[arrSize-1]);
  
  /*
   * Memcpy between host -> device -> host pinned memory
   */  
  updateHostArray(A1_h, arrSize, initData++);
  HIPCHECK(memcopy(A1_d,     A1_h,     numBytes, hipMemcpyHostToDevice, stream));
  HIPCHECK(memcopy(A1_pin_h, A1_d,     numBytes, hipMemcpyDeviceToHost, stream));
  HIPASSERT(A1_h[arrSize-1] == A1_pin_h[arrSize-1]);  
  
  /*
   * Memcpy between host -> device -> device -> host pinned memory
   */
  updateHostArray(A1_h, arrSize, initData++);
  HIPCHECK(memcopy(A1_d,     A1_h,     numBytes, hipMemcpyHostToDevice, stream));
  HIPCHECK(memcopy(A2_d,     A1_d,     numBytes, hipMemcpyDeviceToDevice, stream));  
  HIPCHECK(memcopy(A1_pin_h, A2_d,     numBytes, hipMemcpyDeviceToHost, stream));
  HIPASSERT(A1_h[arrSize-1] == A1_pin_h[arrSize-1]); 

  HipTest::freeArrays(A1_d, A2_d);
  HipTest::freeArraysForHost(A1_h, A2_h, false);
  HipTest::freeArraysForHost(A1_pin_h, A2_pin_h, true);
  return true;
}

bool SimpleMemcopy2(size_t arrSize) {
  size_t Nbytes = arrSize * sizeof(int);
  
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, arrSize, false);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, arrSize);
  
  hipStream_t stream = NULL;
  if(gMemcpyType != SYNC){
    HIPCHECK(hipStreamCreate(&stream));
  }
  if(gMemcpyType == ASYNC){
    HIPCHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
    HIPCHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream));
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, stream, 
                        static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d, arrSize);
    HIPCHECK(hipStreamSynchronize(stream));
  }
  else{
    HIPCHECK(memcopy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIPCHECK(memcopy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));
    
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, stream,
                  static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d, arrSize);
  }
  HIPCHECK(memcopy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  HipTest::checkVectorADD(A_h, B_h, C_h, arrSize);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  return true;
}

/*
 * SimpleMemcopyWithDefaultKind() : 
 *      Test includes: 
 *          It covers all possible valid memcopy operations using hipMemcpuyDefault as kind.
 *      Note: In case of hipMemcpyDefault dst/src pointers should be used to find correct direction
 */
template<typename T>
bool SimpleMemcopyWithDefaultKind(int arrSize, T initData){
  size_t numBytes = sizeof(T) * arrSize;
  
  // Variables
  T *A1_d, *A2_d;
  T *A1_h, *A2_h;
  T *A1_pin_h,*A2_pin_h;
  
  // Allocate host memory 
  HipTest::initArraysForHost(&A1_h, &A2_h, numBytes, false);
  HipTest::initArraysForHost(&A1_pin_h, &A2_pin_h, numBytes, true);
  // Allocate device memory
  HipTest::initArrays(&A1_d, &A2_d, numBytes);
  
  hipStream_t stream;
  if(gMemcpyType == SYNC){
    stream = NULL;
  }
  else{
    HIPCHECK(hipStreamCreate(&stream));
  }
  
  /*
   * Memcpy between host memory (pinned & unpinned)
   */
  updateHostArray(A1_h, arrSize, initData); 
  HIPCHECK(memcopy(A2_h,     A1_h,     numBytes, hipMemcpyDefault, stream));
  HIPCHECK(memcopy(A2_pin_h, A2_h,     numBytes, hipMemcpyDefault, stream ));
  HIPCHECK(memcopy(A1_pin_h, A2_pin_h, numBytes, hipMemcpyDefault, stream));
  HIPASSERT(A1_h[arrSize-1] == A1_pin_h[arrSize-1]);
  
  /*
   * Memcpy between host -> device -> host pinned memory
   */  
  updateHostArray(A1_h, arrSize, initData++);
  HIPCHECK(memcopy(A1_d,     A1_h,     numBytes, hipMemcpyDefault, stream));
  HIPCHECK(memcopy(A1_pin_h, A1_d,     numBytes, hipMemcpyDefault, stream));
  HIPASSERT(A1_h[arrSize-1] == A1_pin_h[arrSize-1]);  
  
  /*
   * Memcpy between host -> device -> device -> host pinned memory
   */
  updateHostArray(A1_h, arrSize, initData++);
  HIPCHECK(memcopy(A1_d,     A1_h,     numBytes, hipMemcpyDefault, stream));
  HIPCHECK(memcopy(A2_d,     A1_d,     numBytes, hipMemcpyDefault, stream));  
  HIPCHECK(memcopy(A1_pin_h, A2_d,     numBytes, hipMemcpyDefault, stream));
  HIPASSERT(A1_h[arrSize-1] == A1_pin_h[arrSize-1]); 
  
  HipTest::freeArrays(A1_d, A2_d);
  HipTest::freeArraysForHost(A1_h, A2_h, false);
  HipTest::freeArraysForHost(A1_pin_h, A2_pin_h, true);
  return true;
}

/*
 * runSimpleCopy() : 
 *    Test includes: 
 *        Test runs SimpleMemcopy & SimpleMemcopyWithDefaultKind on every device
 */
bool runSimpleCopy(){
  bool status = true;
  int numDev;
  HIPCHECK(hipGetDevice(&numDev));
  int gpu = 0;
  do{
      HIPCHECK(hipSetDevice(gpu));
      status &= SimpleMemcopy<float>(1024*1024, 1.0f);
      status &= SimpleMemcopy2(1024*1024);
      status &= SimpleMemcopyWithDefaultKind<int>(1024*1024, 1);
      gpu++;
    }while(numDev > gpu);
  return status;
}

/*
 * MultiDeviceInterLeavedCopy() : 
 *    Test includes: 
 *      1.) Copy uses destination and source pointers from different devices e.g. D2D (src=gpu0 & dst=gpu1)  
 */
 
template<typename T>
bool MultiDeviceInterLeavedCopy(size_t arrSize, T initData, int gpu0, int gpu1){

  if(!isMultiDevice()){
    skipped();
    return true;
  }

  size_t numBytes = sizeof(T) * arrSize;
  
  // Allocations
  T *A1_d0, *A2_d0, *A1_d1, *A2_d1;
  T *A1_pin_h0,*A2_pin_h0,*A1_pin_h1,*A2_pin_h1;
                   
  HIPCHECK(hipSetDevice(gpu0)); 
  // Allocate host memory 
  HipTest::initArraysForHost(&A1_pin_h0, &A2_pin_h0, numBytes, true);
  // Allocate device memory
  HipTest::initArrays(&A1_d0, &A2_d0, numBytes);
  
  HIPCHECK(hipSetDevice(gpu1)); 
  // Allocate host memory 
  HipTest::initArraysForHost(&A1_pin_h1, &A2_pin_h1, numBytes, true);
  // Allocate device memory
  HipTest::initArrays(&A1_d1, &A2_d1, numBytes); 
  
  hipStream_t stream = NULL;
  if(gMemcpyType != SYNC){
    HIPCHECK(hipStreamCreate(&stream));
  }
  
  /*
   * Memcpy between host memory pinned allocated from defferent device
   */
  updateHostArray(A1_pin_h0, arrSize, initData); 
  HIPCHECK(memcopy(A1_pin_h1, A1_pin_h0,     numBytes, hipMemcpyHostToHost, stream));
  HIPCHECK(memcopy(A2_pin_h0, A1_pin_h1,     numBytes, hipMemcpyHostToHost, stream));
  HIPASSERT(A1_pin_h0[arrSize-1] == A2_pin_h0[arrSize-1]);
  
  /*
   * Memcpy between host(GPU0) -> device(GPU1) -> host(GPU0)
   */  
  updateHostArray(A1_pin_h0, arrSize, initData++);
  HIPCHECK(memcopy(A1_d1,     A1_pin_h0, numBytes, hipMemcpyHostToDevice, stream));
  HIPCHECK(memcopy(A2_pin_h0, A1_d1,     numBytes, hipMemcpyDeviceToHost, stream));
  HIPASSERT(A2_pin_h0[arrSize-1] == A1_pin_h0[arrSize-1]);  
  
  /*
   * Memcpy between host(GPU0) -> device(GPU1) -> device(GPU0) -> host(GPU1)
   */
  updateHostArray(A1_pin_h0, arrSize, initData++);
  HIPCHECK(memcopy(A1_d1,     A1_pin_h0, numBytes, hipMemcpyHostToDevice, stream));
  HIPCHECK(memcopy(A2_d0,     A1_d1,     numBytes, hipMemcpyDeviceToDevice, stream));  
  HIPCHECK(memcopy(A1_pin_h1, A2_d0,     numBytes, hipMemcpyDeviceToHost, stream));
  HIPASSERT(A1_pin_h1[arrSize-1] == A1_pin_h0[arrSize-1]); 
  
  HipTest::freeArrays(A1_d0, A2_d0);
  HipTest::freeArrays(A1_d1, A2_d1);
  HipTest::freeArraysForHost(A1_pin_h1, A2_pin_h1, true);
  HipTest::freeArraysForHost(A1_pin_h0, A2_pin_h0, true);  
  
  return true;
}

/*
 * MultiDeviceCopy() : 
 *    Test includes: 
 *      1.) Copy happens on same device but device switch is keep on happening
 */
template<typename T>
bool MultiDeviceCopy(size_t arrSize, T initData, int gpu0, int gpu1){

  if(!isMultiDevice()){
    skipped();
    return true;
  }

  size_t numBytes = sizeof(T) * arrSize;
  
  // Allocations
  T *A1_d0, *A2_d0, *A1_d1, *A2_d1;
  T *A1_pin_h0,*A2_pin_h0,*A1_pin_h1,*A2_pin_h1;
                   
  HIPCHECK(hipSetDevice(gpu0)); 
  // Allocate host memory 
  HipTest::initArraysForHost(&A1_pin_h0, &A2_pin_h0, numBytes, true);
  // Allocate device memory
  HipTest::initArrays(&A1_d0, &A2_d0, numBytes);
  
  hipStream_t stream0 = NULL;
  if(gMemcpyType != SYNC){
    HIPCHECK(hipStreamCreate(&stream0));
  }
  
  HIPCHECK(hipSetDevice(gpu1)); 
  // Allocate host memory 
  HipTest::initArraysForHost(&A1_pin_h1, &A2_pin_h1, numBytes, true);
  // Allocate device memory
  HipTest::initArrays(&A1_d1, &A2_d1, numBytes); 

  hipStream_t stream1 = NULL;
  if(gMemcpyType != SYNC){
    HIPCHECK(hipStreamCreate(&stream1));
  }
  /*
   * Memcpy H2H on GPU0
   */
  HIPCHECK(hipSetDevice(gpu0));
  updateHostArray(A1_pin_h0, arrSize, initData); 
  HIPCHECK(memcopy(A2_pin_h0, A1_pin_h0, numBytes, hipMemcpyHostToHost,stream0));
  HIPASSERT(A1_pin_h0[arrSize-1] == A2_pin_h0[arrSize-1]);
  
  /*
   * Memcpy H2D -> D2H on GPU1
   */  
  HIPCHECK(hipSetDevice(gpu1));
  updateHostArray(A1_pin_h1, arrSize, initData++);
  HIPCHECK(memcopy(A1_d1,     A1_pin_h1, numBytes, hipMemcpyHostToDevice, stream1));
  HIPCHECK(memcopy(A2_pin_h1, A1_d1,     numBytes, hipMemcpyDeviceToHost, stream1));
  HIPASSERT(A2_pin_h1[arrSize-1] == A1_pin_h1[arrSize-1]);  
  
  /*
   * Memcpy H2D -> D2D(GPU0->GPU0) -> D2H on GPU0
   */
  HIPCHECK(hipSetDevice(gpu0));
  updateHostArray(A1_pin_h0, arrSize, initData++);
  HIPCHECK(memcopy(A1_d0,     A1_pin_h0, numBytes, hipMemcpyHostToDevice, stream0));
  HIPCHECK(memcopy(A2_d0,     A1_d0,     numBytes, hipMemcpyDeviceToDevice, stream0));  
  HIPCHECK(memcopy(A2_pin_h0, A2_d0,     numBytes, hipMemcpyDeviceToHost, stream0));
  HIPASSERT(A2_pin_h0[arrSize-1] == A1_pin_h0[arrSize-1]); 

  /*
   * Memcpy H2H on GPU1
   */
  HIPCHECK(hipSetDevice(gpu1));
  updateHostArray(A1_pin_h1, arrSize, initData); 
  HIPCHECK(memcopy(A2_pin_h1, A1_pin_h1, numBytes, hipMemcpyHostToHost, stream1));
  HIPASSERT(A1_pin_h1[arrSize-1] == A2_pin_h1[arrSize-1]);
    
  /*
   * Memcpy H2D -> D2H on GPU0
   */  
  HIPCHECK(hipSetDevice(gpu0));
  updateHostArray(A1_pin_h0, arrSize, initData++);
  HIPCHECK(memcopy(A1_d0,     A1_pin_h0, numBytes, hipMemcpyHostToDevice, stream0));
  HIPCHECK(memcopy(A2_pin_h0, A1_d0,     numBytes, hipMemcpyDeviceToHost, stream0));
  HIPASSERT(A2_pin_h0[arrSize-1] == A1_pin_h0[arrSize-1]);  
  
  /*
   * Memcpy H2D -> D2D(GPU1->GPU1) -> D2H on GPU1
   */
  HIPCHECK(hipSetDevice(gpu1));
  updateHostArray(A1_pin_h1, arrSize, initData++);
  HIPCHECK(memcopy(A1_d1,     A1_pin_h1, numBytes, hipMemcpyHostToDevice, stream1));
  HIPCHECK(memcopy(A2_d1,     A1_d1,     numBytes, hipMemcpyDeviceToDevice, stream1));  
  HIPCHECK(memcopy(A2_pin_h1, A2_d1,     numBytes, hipMemcpyDeviceToHost, stream1));
  HIPASSERT(A2_pin_h1[arrSize-1] == A1_pin_h1[arrSize-1]); 
    
  HipTest::freeArrays(A1_d0, A2_d0);
  HipTest::freeArrays(A1_d1, A2_d1);
  HipTest::freeArraysForHost(A1_pin_h1, A2_pin_h1, true);
  HipTest::freeArraysForHost(A1_pin_h0, A2_pin_h0, true);  
  return true;
}

// ----------------------------- Stress Testing ------------------------------- //
/*
 * stressSimpleCopyTest() : 
 *    Test includes: 
 *      1.) Test simple memcopy with differnet data types float, int, chat, double
 */
bool stressSimpleCopyTest(size_t N, bool isMemcpyAsync =false){
  bool status = true;
  
  status &= SimpleMemcopy<float>(N, 1.0f);
  status &= SimpleMemcopy<int>(N, 3);
  status &= SimpleMemcopy<double>(N, 1.01f);
  status &= SimpleMemcopy<char>(N, 'a');
  
  return status;
}

/*
 * stressMultiDeviceTest() : 
 *    Test includes: 
 *      1.) It selects two GPUs randomly and test multi device interleaved and non-interveaved 
 *          memcopy with differnet data types float, int, chat, double
 */
bool stressMultiDeviceTest(size_t N, bool isMemcpyAsync = false)
{
  bool status = true;
  
  if(!isMultiDevice()){
    skipped();
    return true;
  }
  
  int gpu0 = INT_MAX;;
  int gpu1 = INT_MAX;
  getGpuIdRand(gpu0,gpu1);
    
  status &= MultiDeviceCopy<char>(N, 'a', gpu0, gpu1);
  status &= MultiDeviceCopy<int>(N, 1, gpu0, gpu1);
  status &= MultiDeviceCopy<float>(N, 1.0f, gpu0, gpu1);
  status &= MultiDeviceCopy<double>(N, 1.01f, gpu0, gpu1);
  
  status &= MultiDeviceInterLeavedCopy<char>(N, 'a', gpu0, gpu1);
  status &= MultiDeviceInterLeavedCopy<int>(N, 1, gpu0, gpu1);
  status &= MultiDeviceInterLeavedCopy<float>(N, 1.0f, gpu0, gpu1);
  status &= MultiDeviceInterLeavedCopy<double>(N, 1.01f, gpu0, gpu1);
  return status;
}

/*
 * StressTests() : 
 *      Test includes: 
 *          1) Memcpy of different size [4Bytes to 1MB] + random offset
 *          2) Different data types char, int, double, float
 *          3) mGPU test randomly choose two GPUs and execute interleaved copy and D2D copy  
 */
bool StressTests(){
        
  // loop over different memory size up to 1MB
  bool status =  true;
  int offset;
  for(size_t n = 2; n < 20; ++n)
  {
    offset  = std::rand() % 4;
    size_t N = pow(2,n) + offset;
    status &= stressSimpleCopyTest(N);
    status &= stressMultiDeviceTest(N);
  }
    
  return status;
}

bool PositiveTests(){
  bool status = true;
  
  status &= runSimpleCopy();
  status &= MultiDeviceInterLeavedCopy<float>(1024*1024, 1.0f, 0, 1);
  status &= MultiDeviceCopy<int>(1024*1024, 1, 0, 1);
  
  return status;
}

int main(int argc, char* argv[]){

  std::srand(std::time(nullptr));
  if(!parseMyArguments(argc, argv))
  {
    return 0; // return early
  }
  
  bool status = true;

#ifdef __HIP_PLATFORM_NVCC__  
  status &= NegativeTests();
#endif
  status &= PositiveTests();
  status &= StressTests();
  
  if (status){
    passed();
  }
  return 0;
}