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
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM all
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

void printSep()
{
    printf ("======================================================================================\n");
}


//---
// Test copies of a matrix numW by numH
// The subroutine allocates memory , copies to device, runs a vector add kernel, copies back, and checks the result.
//
// IN: numW: number of elements in the 1st dimension used for allocation
// IN: numH: number of elements in the 2nd dimension used for allocation
// IN: usePinnedHost : If true, allocate host with hipHostMalloc and is pinned ; else allocate host memory with malloc.
//
template <typename T>
void memcpy2Dtest(size_t numW, size_t numH, bool usePinnedHost)
{

  size_t width = numW * sizeof(T);
  size_t sizeElements = width * numH;

  printf("memcpy2Dtest: %s<%s> size=%lu (%6.2fMB) W: %d, H:%d, usePinnedHost: %d\n",
         __func__,
         TYPENAME(T),
         sizeElements, sizeElements/1024.0/1024.0,
         (int)numW, (int)numH, usePinnedHost);

  T *A_d, *B_d, *C_d;
  T *A_h, *B_h, *C_h;

  size_t pitch_A, pitch_B, pitch_C;

  hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
  HipTest::initArrays2DPitch(&A_d, &B_d, &C_d, &pitch_A, &pitch_B, &pitch_C, numW, numH);
  HipTest::initArraysForHost(&A_h, &B_h, &C_h, numW*numH, usePinnedHost);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numW*numH);

  HIPCHECK (hipMemcpy2D (A_d, pitch_A, A_h, width, width, numH, hipMemcpyHostToDevice) );
  HIPCHECK (hipMemcpy2D (B_d, pitch_B, B_h, width, width, numH, hipMemcpyHostToDevice) );

  hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, B_d, C_d, (pitch_C/sizeof(T))*numH);

  HIPCHECK (hipMemcpy2D (C_h, width, C_d, pitch_C, width, numH, hipMemcpyDeviceToHost) );

  HIPCHECK ( hipDeviceSynchronize() );
  HipTest::checkVectorADD(A_h, B_h, C_h, numW*numH);

  HipTest::freeArrays (A_d, B_d, C_d, A_h, B_h, C_h, usePinnedHost);

  printf ("  %s success\n", __func__);
}

//---
// Test copies of a matrix numW by numH into a hipArray data structure
// The subroutine allocates memory , copies to device, runs a vector add kernel, copies back, and checks the result.
//
// IN: numW: number of elements in the 1st dimension used for allocation
// IN: numH: number of elements in the 2nd dimension used for allocation. If this is 1, then the 1-dimensional copy API
//           would be used
// IN: usePinnedHost : If true, allocate host with hipHostMalloc and is pinned ; else allocate host memory with malloc.
// IN: usePitch: If true, pads additional memory. This is only valid in the 2-dimensional case
//
template <typename T>
void memcpyArraytest(size_t numW, size_t numH, bool usePinnedHost, bool usePitch=false)
{

  size_t width = numW * sizeof(T);
  size_t sizeElements = width * numH;

  printf("memcpyArraytest: %s<%s> size=%lu (%6.2fMB) W: %d, H: %d, usePinnedHost: %d, usePitch: %d\n",
         __func__,
         TYPENAME(T),
         sizeElements, sizeElements/1024.0/1024.0,
         (int)numW, (int)numH, usePinnedHost, usePitch);

  hipArray *A_d, *B_d, *C_d;
  T *A_h, *B_h, *C_h;

  // 1D
  if ((numW >= 1) && (numH == 1)) {
    hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
    HipTest::initHIPArrays(&A_d, &B_d, &C_d, &desc, numW, 1, 0);
    HipTest::initArraysForHost(&A_h, &B_h, &C_h, numW*numH, usePinnedHost);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numW*numH);

    HIPCHECK (hipMemcpyToArray (A_d, 0, 0, (void *)A_h, width, hipMemcpyHostToDevice) );
    HIPCHECK (hipMemcpyToArray (B_d, 0, 0, (void *)B_h, width, hipMemcpyHostToDevice) );

    hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, (T*)A_d->data, (T*)B_d->data, (T*)C_d->data, numW);

    HIPCHECK (hipMemcpy (C_h, C_d->data, width, hipMemcpyDeviceToHost) );

    HIPCHECK ( hipDeviceSynchronize() );
    HipTest::checkVectorADD(A_h, B_h, C_h, numW);

  }
  // 2D
  else if ((numW >= 1) && (numH >= 1)) {


    hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
    HipTest::initHIPArrays(&A_d, &B_d, &C_d, &desc, numW, numH, 0);
    HipTest::initArraysForHost(&A_h, &B_h, &C_h, numW*numH, usePinnedHost);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numW*numH);

    if (usePitch) {
      T *A_p, *B_p, *C_p;
      size_t pitch_A, pitch_B, pitch_C;

      HipTest::initArrays2DPitch(&A_p, &B_p, &C_p, &pitch_A, &pitch_B, &pitch_C, numW, numH);
      HIPCHECK (hipMemcpy2D (A_p, pitch_A, A_h, width, width, numH, hipMemcpyHostToDevice) );
      HIPCHECK (hipMemcpy2D (B_p, pitch_B, B_h, width, width, numH, hipMemcpyHostToDevice) );

      HIPCHECK (hipMemcpy2DToArray (A_d, 0, 0, (void *)A_p, pitch_A, width, numH, hipMemcpyDeviceToDevice) );
      HIPCHECK (hipMemcpy2DToArray (B_d, 0, 0, (void *)B_p, pitch_B, width, numH, hipMemcpyDeviceToDevice) );

      hipFree(A_p);
      hipFree(B_p);
      hipFree(C_p);
    }
    else {
      HIPCHECK (hipMemcpy2DToArray (A_d, 0, 0, (void *)A_h, width, width, numH, hipMemcpyHostToDevice) );
      HIPCHECK (hipMemcpy2DToArray (B_d, 0, 0, (void *)B_h, width, width, numH, hipMemcpyHostToDevice) );
    }

    hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, (T*)A_d->data, (T*)B_d->data, (T*)C_d->data, numW*numH);

    HIPCHECK (hipMemcpy2D ((void*)C_h, width, (void*)C_d->data, width, width, numH, hipMemcpyDeviceToHost) );

    HIPCHECK ( hipDeviceSynchronize() );
    HipTest::checkVectorADD(A_h, B_h, C_h, numW*numH);
  }
  // Unknown
  else {
    HIPASSERT("Incompatible dimensions" && 0);
  }

  hipFreeArray(A_d);
  hipFreeArray(B_d);
  hipFreeArray(C_d);
  HipTest::freeArraysForHost(A_h, B_h, C_h, usePinnedHost);

  printf ("  %s success\n", __func__);

}

//---
//Try many different sizes to memory copy.
template <typename T>
void memcpyArraytest_size(size_t maxElem=0, size_t offset=0)
{
  printf ("test: %s<%s>\n", __func__,  TYPENAME(T));

  int deviceId;
  HIPCHECK(hipGetDevice(&deviceId));

  size_t free, total;
  HIPCHECK(hipMemGetInfo(&free, &total));

  if (maxElem == 0) {
      maxElem = free/sizeof(T)/5;
  }

  printf ("  device#%d: hipMemGetInfo: free=%zu (%4.2fMB) total=%zu (%4.2fMB)    maxSize=%6.1fMB offset=%lu\n",
          deviceId, free, (float)(free/1024.0/1024.0), total, (float)(total/1024.0/1024.0), maxElem*sizeof(T)/1024.0/1024.0, offset);

  // Test 1D
  for (size_t elem=64; elem+offset<=maxElem; elem*=2) {
      HIPCHECK ( hipDeviceReset() );
      memcpyArraytest<T>(elem+offset, 1, 0);  // unpinned host
      HIPCHECK ( hipDeviceReset() );
      memcpyArraytest<T>(elem+offset, 1, 1);  // pinned host
  }

  // Test 2D
  size_t maxElem2D = sqrt(maxElem);

  for (size_t elem=64; elem+offset<=maxElem2D; elem*=2) {
      HIPCHECK ( hipDeviceReset() );
      memcpyArraytest<T>(elem+offset, elem+offset, 0, 1);  // use pitch
  }
}

int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true);

    printf ("info: set device to %d\n", p_gpuDevice);
    HIPCHECK(hipSetDevice(p_gpuDevice));

    if (p_tests & 0x1) {
        printf ("\n\n=== tests&1 (types)\n");
        printSep();
        HIPCHECK ( hipDeviceReset() );
        size_t width = N/6;
        size_t height = N/6;
        memcpy2Dtest<float>(321, 211, 0);
        memcpy2Dtest<double>(322, 211, 0);
        memcpy2Dtest<char>(320, 211, 0);
        memcpy2Dtest<int>(323, 211, 0);
        printf ("===\n\n\n");

        printf ("\n\n=== tests&1 (types)\n");
        printSep();
        // 2D
        memcpyArraytest<float>(320, 211, 0, 0);
        memcpyArraytest<unsigned int>(322, 211, 0, 0);
        memcpyArraytest<int>(320, 211, 0, 0);
        memcpyArraytest<float>(320, 211, 0, 1);
        memcpyArraytest<float>(322, 211, 0, 1);
        memcpyArraytest<int>(320, 211, 0, 1);
        printSep();
        // 1D
        memcpyArraytest<float>(320, 1, 0);
        memcpyArraytest<unsigned int>(322, 1, 0);
        memcpyArraytest<int>(320, 1, 0);
        printf ("===\n\n\n");
    }

    if (p_tests & 0x4) {
        printf ("\n\n=== tests&4 (test sizes and offsets)\n");
        printSep();
        HIPCHECK ( hipDeviceReset() );
        printSep();
        memcpyArraytest_size<float>(0,0);
        printSep();
        memcpyArraytest_size<float>(0,64);
        printSep();
        memcpyArraytest_size<float>(1024*1024,13);
        printSep();
        memcpyArraytest_size<float>(1024*1024,50);
    }

    passed();

}
