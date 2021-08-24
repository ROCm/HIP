/*
 Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../../src/test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include <printf/printf_common.h>
#include <iostream>
#include <chrono>
#include <sys/time.h>

#define SIMPLY_ASSIGN 0
#define USE_HIPTEST_SETNUMBLOCKS 0

using namespace std;

template<class T>
__global__ void vec_fill(T *x, T coef, int N) {
  const int istart = threadIdx.x + blockIdx.x * blockDim.x;
  const int ishift = blockDim.x * gridDim.x;
  for (int i = istart; i < N; i += ishift) {
#if SIMPLY_ASSIGN
    x[i] = coef;
#else
    x[i] = coef * i;
#endif
  }
}

__device__ void print_log(int i, double value, double expected) {
  printf("failed at %d: val=%g, expected=%g\n", i, value, expected);
}

__device__ void print_log(int i, int value, int expected) {
  printf("failed at %d: val=%d, expected=%d\n", i, value, expected);
}

template<class T>
__global__ void vec_verify(T *x, T coef, int N) {
  const int istart = threadIdx.x + blockIdx.x * blockDim.x;
  const int ishift = blockDim.x * gridDim.x;
  for (int i = istart; i < N; i += ishift) {
#if SIMPLY_ASSIGN
    if(x[i] != coef) {
      print_log(i, x[i], coef);
    }
#else
    if(x[i] != coef * i) {
      print_log(i, x[i], coef * i);
    }
#endif
  }
}

template<class T>
__global__ void daxpy(T *__restrict__ x, T *__restrict__ y,
    const T coef, int Niter, int N) {
  const int istart = threadIdx.x + blockIdx.x * blockDim.x;
  const int ishift = blockDim.x * gridDim.x;
  for (int iter = 0; iter < Niter; ++iter) {
    T iv = coef * iter;
    for (int i = istart; i < N; i += ishift)
    y[i] = iv * x[i] + y[i];
  }
}

template<class T>
class hipPerfMemFill {
 private:
  static constexpr int NUM_START = 27;
  static constexpr int NUM_SIZE = 5;
  static constexpr int NUM_ITER = 10;
  size_t totalSizes_[NUM_SIZE];
  hipDeviceProp_t props_;
  const T coef_ = getCoefficient(3.14159);
  const unsigned int blocksPerCU_;
  const unsigned int threadsPerBlock_;

 public:
  hipPerfMemFill(unsigned int blocksPerCU, unsigned int threadsPerBlock) :
    blocksPerCU_(blocksPerCU), threadsPerBlock_(threadsPerBlock) {
    for (int i = 0; i < NUM_SIZE; i++) {
      totalSizes_[i] = 1ull << (i + NUM_START); // 128M, 256M, 512M, 1024M, 2048M
    }
  }

  ~hipPerfMemFill() {
  }

  bool supportLargeBar() {
    return props_.isLargeBar != 0;
  }

  bool supportManagedMemory() {
    return props_.managedMemory != 0;
  }

  const T getCoefficient(double val) {
    return static_cast<T>(val);
  }

  void setHostBuffer(T *A, T val, size_t size) {
    size_t len = size / sizeof(T);
    for (int i = 0; i < len; i++) {
      A[i] = val;
    }
  }

  void open(int deviceId) {
    int nGpu = 0;
    HIPCHECK(hipGetDeviceCount(&nGpu));
    if (nGpu < 1) {
      failed("No GPU!");
    } else if (deviceId >= nGpu) {
      failed("Info: wrong GPU Id %d\n", deviceId);
    }

    HIPCHECK(hipSetDevice(deviceId));
    HIPCHECK(hipGetDeviceProperties(&props_, deviceId));
    std::cout << "Info: running on device: id: " << deviceId << ", bus: 0x"
        << props_.pciBusID << " " << props_.name << " with "
        << props_.multiProcessorCount << " CUs, large bar: "
        << supportLargeBar() << ", managed memory: " << supportManagedMemory()
        << ", DeviceMallocFinegrained: " << supportDeviceMallocFinegrained()
        << std::endl;
  }

  void log_host(const char* title, double GBytes, double sec) {
    cout << title << " [" << setw(7) << GBytes << " GB]: cost " << setw(10) << sec
        << " s in bandwidth " << setw(10) << GBytes / sec << " [GB/s]" << endl;
  }

  void log_kernel(const char* title, double GBytes, double sec, double sec_hv, double sec_kv) {
    cout << title << " [" << setw(7) << GBytes << " GB]: cost " << setw(10) << sec
        << " s in bandwidth " << setw(10) << GBytes / sec << " [GB/s]" << ", hostVerify cost "
        << setw(10) << sec_hv << " s in bandwidth " << setw(10) << GBytes / sec_hv << " [GB/s]"
        << ", kernelVerify cost "<< setw(10) << sec_kv << " s in bandwidth " << setw(10)
        << GBytes / sec_kv << " [GB/s]" << endl;
  }

  void hostFill(size_t size, T *data, T coef, double &sec) {
    size_t num = size / sizeof(T);  // Size of elements
    auto start = chrono::steady_clock::now();
    for (int i = 0; i < num; ++i) {
#if SIMPLY_ASSIGN
      data[i] = coef;
#else
      data[i] = coef * i;
#endif
    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double> diff = end - start;  // in second
    sec = diff.count();
  }

  void kernelFill(size_t size, T *data, T coef, double &sec) {
    size_t num = size / sizeof(T);  // Size of elements
    unsigned blocks = setNumBlocks(num);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(vec_fill<T>), dim3(blocks),
                           dim3(threadsPerBlock), 0, 0, data, 0, num);  // kernel will be loaded first time
    HIPCHECK(hipDeviceSynchronize());

    auto start = chrono::steady_clock::now();

    for (int iter = 0; iter < NUM_ITER; ++iter) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(vec_fill<T>), dim3(blocks),
                             dim3(threadsPerBlock), 0, 0, data, coef, num);
    }
    HIPCHECK(hipDeviceSynchronize());

    auto end = chrono::steady_clock::now();
    chrono::duration<double> diff = end - start;  // in second
    sec = diff.count() / NUM_ITER;  // in second
  }

  void hostVerify(size_t size, T *data, T coef, double &sec) {
    size_t num = size / sizeof(T);  // Size of elements
    auto start = chrono::steady_clock::now();
    for (int i = 0; i < num; ++i) {
#if SIMPLY_ASSIGN
      if(data[i] != coef) {
        cout << "hostVerify failed: i=" << i << ", data[i]=" << data[i] << ", expected=" << coef << endl;
        failed("failed\n");
      }
#else
      if(data[i] != coef * i) {
        cout << "hostVerify failed: i=" << i << ", data[i]=" << data[i] << ", expected=" << coef * i << endl;
        failed("failed\n");
      }
#endif
    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double> diff = end - start;  // in second
    sec = diff.count();
  }

  void kernelVerify(size_t size, T *data, T coef, double &sec) {
    size_t num = size / sizeof(T);  // Size of elements
    unsigned blocks = setNumBlocks(num);

    CaptureStream *capture = new CaptureStream(stdout);
    capture->Begin();

    hipLaunchKernelGGL(HIP_KERNEL_NAME(vec_verify<T>), dim3(blocks),
                       dim3(threadsPerBlock), 0, 0, data, coef, num);  // kernel will be loaded first time
    HIPCHECK(hipDeviceSynchronize());

    capture->End();
    capture->Truncate(1000); // Don't want too long log if existing
    std::string device_output = capture->getData();
    delete capture;
    if (device_output.length() > 0) {
      failed("kernelVerify failed:\n%s\n", device_output.c_str());
    }

    // Now all data verified. The following is to test bandwidth.
    auto start = chrono::steady_clock::now();

    for (int iter = 0; iter < NUM_ITER; ++iter) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(vec_verify<T>), dim3(blocks),
                             dim3(threadsPerBlock), 0, 0, data, coef, num);
    }
    HIPCHECK(hipDeviceSynchronize());

    auto end = chrono::steady_clock::now();
    chrono::duration<double> diff = end - start;  // in second
    sec = diff.count() / NUM_ITER;  // in second
  }

  bool testLargeBarDeviceMemoryHostFill(size_t size) {
    if (!supportLargeBar()) {
      return false;
    }

    double GBytes = (double) size / (1024.0 * 1024.0 * 1024.0);

    T *A;
    HIPCHECK(hipMalloc(&A, size));
    double sec = 0;
    hostFill(size, A, coef_, sec);  // Cpu can access device mem in LB
    HIPCHECK(hipFree(A));

    log_host("Largebar: host   fill", GBytes, sec);
    return true;
  }

  bool testLargeBar() {
    if (!supportLargeBar()) {
      return false;
    }

    cout << "Test large bar device memory host filling" << endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testLargeBarDeviceMemoryHostFill(totalSizes_[i])) {
        return false;
      }
    }

    return true;
  }

  bool testManagedMemoryHostFill(size_t size) {
    if (!supportManagedMemory()) {
      return false;
    }
    double GBytes = (double) size / (1024.0 * 1024.0 * 1024.0);

    T *A;
    HIPCHECK(hipMallocManaged(&A, size));
    double sec = 0;
    hostFill(size, A, coef_, sec);  // Cpu can access HMM mem
    HIPCHECK(hipFree(A));

    log_host("Managed: host   fill", GBytes, sec);
    return true;
  }

  bool testManagedMemoryKernelFill(size_t size) {
    if (!supportManagedMemory()) {
      return false;
    }
    double GBytes = (double) size / (1024.0 * 1024.0 * 1024.0);

    T *A;
    HIPCHECK(hipMallocManaged(&A, size));

    double sec = 0, sec_hv = 0, sec_kv = 0;
    kernelFill(size, A, coef_, sec);
    hostVerify(size, A, coef_, sec_hv);  // Managed memory can be verified by host
    kernelVerify(size, A, coef_, sec_kv);
    HIPCHECK(hipFree(A));

    log_kernel("Managed: kernel fill", GBytes, sec, sec_hv, sec_kv);

    return true;
  }

  bool testManagedMemory() {
    if (!supportManagedMemory()) {
      return false;
    }

    cout << "Test managed memory host filling" << endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testManagedMemoryHostFill(totalSizes_[i])) {
        return false;
      }
    }

    cout << "Test managed memory kernel filling" << endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testManagedMemoryKernelFill(totalSizes_[i])) {
        return false;
      }
    }

    return true;
  }

  bool testHostMemoryHostFill(size_t size, unsigned int flags) {
    double GBytes = (double) size / (1024.0 * 1024.0 * 1024.0);
    T *A;
    HIPCHECK(hipHostMalloc(&A, size, flags));
    double sec = 0;
    hostFill(size, A, coef_, sec);
    HIPCHECK(hipHostFree(A));

    log_host("Host: host   fill", GBytes, sec);
    return true;
  }

  bool testHostMemoryKernelFill(size_t size, unsigned int flags) {
    double GBytes = (double) size / (1024.0 * 1024.0 * 1024.0);

    T *A;
    HIPCHECK(hipHostMalloc((void** ) &A, size, flags));
    double sec = 0, sec_hv = 0, sec_kv = 0;
    kernelFill(size, A, coef_, sec);
    hostVerify(size, A, coef_, sec_hv);
    kernelVerify(size, A, coef_, sec_kv);
    HIPCHECK(hipHostFree(A));

    log_kernel("Host: kernel fill", GBytes, sec, sec_hv, sec_kv);
    return true;
  }

  bool testHostMemory() {
    cout << "Test coherent host memory host filling" << endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testHostMemoryHostFill(totalSizes_[i], hipHostMallocCoherent)) {
        return false;
      }
    }

    cout << "Test non-coherent host memory host filling" << endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testHostMemoryHostFill(totalSizes_[i], hipHostMallocNonCoherent)) {
        return false;
      }
    }

    cout << "Test coherent host memory kernel filling" << endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testHostMemoryKernelFill(totalSizes_[i], hipHostMallocCoherent)) {
        return false;
      }
    }

    cout << "Test non-coherent host memory kernel filling" << endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testHostMemoryKernelFill(totalSizes_[i], hipHostMallocNonCoherent)) {
        return false;
      }
    }

    return true;
  }

  /* This fuction should be via device attribute query*/
  bool supportDeviceMallocFinegrained() {
    T *A = nullptr;
    hipExtMallocWithFlags((void **)&A, sizeof(T), hipDeviceMallocFinegrained);
    if (!A) {
      return false;
    }
    HIPCHECK(hipFree(A));
    return true;
  }

  unsigned int setNumBlocks(size_t size) {
    size_t num = size/sizeof(T);

#if USE_HIPTEST_SETNUMBLOCKS
    return HipTest::setNumBlocks(blocksPerCU_, threadsPerBlock_,
                                 num);
#else
    return (num + threadsPerBlock_ - 1) / threadsPerBlock_;
#endif
  }

  bool testExtDeviceMemoryHostFill(size_t size, unsigned int flags) {
    double GBytes = (double) size / (1024.0 * 1024.0 * 1024.0);

    T *A = nullptr;
    HIPCHECK(hipExtMallocWithFlags((void **)&A, size, flags));
    if (!A) {
      cout << "failed hipExtMallocWithFlags() with size =" << size << " flags="
           << std::hex << flags << endl;
      return false;
    }

    double sec = 0;
    hostFill(size, A, coef_, sec);  // Cpu can access this mem
    HIPCHECK(hipFree(A));

    log_host("ExtDevice: host   fill", GBytes, sec);
    return true;
  }

  bool testExtDeviceMemoryKernelFill(size_t size, unsigned int flags) {
    double GBytes = (double) size / (1024.0 * 1024.0 * 1024.0);

    T *A = nullptr;
    HIPCHECK(hipExtMallocWithFlags((void **)&A, size, flags));
    if (!A) {
      cout << "failed hipExtMallocWithFlags() with size =" << size << " flags="
           << std::hex << flags << endl;
      return false;
    }

    double sec = 0, sec_hv = 0, sec_kv = 0;
    kernelFill(size, A, coef_, sec);
    hostVerify(size, A, coef_, sec_hv);  // Fine grained device memory can be verified by host
    kernelVerify(size, A, coef_, sec_kv);
    HIPCHECK(hipFree(A));

    log_kernel("ExtDevice: kernel fill", GBytes, sec, sec_hv, sec_kv);

    return true;
  }

  bool testExtDeviceMemory() {
    cout << "Test fine grained device memory host filling"
        << endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testExtDeviceMemoryHostFill(totalSizes_[i],
                                       hipDeviceMallocFinegrained)) {
        return false;
      }
    }

    cout << "Test fine grained device memory kernel filling"
        << endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testExtDeviceMemoryKernelFill(totalSizes_[i],
                                         hipDeviceMallocFinegrained)) {
        return false;
      }
    }

    return true;
  }

  bool run() {
    if (supportLargeBar()) {
      if (!testLargeBar()) {
        return false;
      }
    }

    if (supportManagedMemory()) {
      if (!testManagedMemory()) {
        return false;
      }
    }

    if (!testHostMemory()) {
      return false;
    }

    if (supportDeviceMallocFinegrained()) {
      if (!testExtDeviceMemory()) {
        return false;
      }
    }
    return true;
  }

};

int main(int argc, char *argv[]) {
  HipTest::parseStandardArguments(argc, argv, true); // For ::p_gpuDevice, ::blocksPerCU, ::threadsPerBlock
  cout << "Test int" << endl;
  hipPerfMemFill<int> hipPerfMemFillInt(::blocksPerCU, ::threadsPerBlock);
  hipPerfMemFillInt.open(::p_gpuDevice);
  HIPASSERT(hipPerfMemFillInt.run());

  cout << "Test double" << endl;
  hipPerfMemFill<double> hipPerfMemFillDouble(::blocksPerCU, ::threadsPerBlock);
  hipPerfMemFillDouble.open(::p_gpuDevice);
  HIPASSERT(hipPerfMemFillDouble.run());

  passed();
}
