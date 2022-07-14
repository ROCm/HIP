/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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


/* Test Description:
   Scenario-1: Launch a kernel in hipStreamPerThread, while it is in flight
   check for hipStreamQuery(hipStreamPerThread) it should return
   hipErrorNotReady.
   Scenario-2: Testing hipStreamPerThread stream object with hipMallocManaged()
   memory
   Scenario-3: To check the working of hipStreamPerThread in forked process
   Scenario-4: The following test case tests the working of hipEventSynchronize
   in multiple threads which are launched in quick succession
   Scenario-5: The following test case checks the working of
   hipStreamWaitEvent() with hipStreamWaitEvent()
   Scenario-6: Testing hipLaunchCooperativeKernel() api with hipStreamPerThread
   Scenario-7: Testing hipLaunchCooperativeKernelMultiDevice() with
   hipStreamPerThread
*/
#include <vector>
#include <thread>
#include <chrono>
#ifdef _WIN32
  #include <Windows.h>
  #define sleep(x) _sleep(x)
#endif
#ifdef __linux__
  #include <unistd.h>
  #include <sys/mman.h>
  #include <sys/wait.h>
#endif

#include <hip_test_common.hh>
#ifdef HT_AMD
  #include "hip/hip_cooperative_groups.h"
#endif
using namespace std::chrono;
using namespace cooperative_groups;
#if HT_AMD
#define HIPRT_CB
#endif


static bool IfTestPassed = false;
// kernel
__global__ void StreamPerThrd(int *Ad, int *Ad1, size_t n, int Pk_Clk,
                              int Wait, int WaitEvnt  = 0) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    Ad[index] = Ad[index] + 10;
  }
  if (Wait) {
    int64_t GpuFrq = (Pk_Clk * 1000);
    int64_t StrtTck = clock64();
    if (index == 0) {
      // The following while loop checks the value in ptr for around 4 seconds
      while ((clock64() - StrtTck) <= (6 * GpuFrq)) {
      }
      if (WaitEvnt == 1) {
        *Ad1 = 1;
      }
    }
  }
}


__global__ void StreamPerThrd1(int *A, int Pk_Clk) {
  int64_t GpuFrq = (Pk_Clk * 1000);
  int64_t StrtTck = clock64();
  // The following while loop checks the value in ptr for around 3-4 seconds
  while ((clock64() - StrtTck) <= (3 * GpuFrq)) {
  }
  *A = 1;
}

__global__ void MiniKernel(int *A) {
  if (*A == 0) {
    *A = 2;  //  Fail condition
  } else if (*A == 1) {
    *A = 3;  //  Pass condition
  } else {
     *A = 4;  //  Garbage value found in A
  }
}

__global__ void StreamPerThrdCoopKrnl(int *Ad, int *n) {
  int NumElms = (*n);
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < NumElms) {
    Ad[index] = Ad[index] + 10;
  }
}

#if HT_AMD
__global__ void test_gwsPerThrd(uint* buf, uint bufSize, int64_t* tmpBuf,
                                int64_t* result) {
    extern __shared__ int64_t tmp[];
    uint groups = gridDim.x;
    uint group_id = blockIdx.x;
    uint local_id = threadIdx.x;
    uint chunk = gridDim.x * blockDim.x;

    uint i = group_id * blockDim.x + local_id;
    int64_t sum = 0;
    while (i < bufSize) {
      sum += buf[i];
      i += chunk;
    }
    tmp[local_id] = sum;
    __syncthreads();
    i = 0;
    if (local_id == 0) {
        sum = 0;
        while (i < blockDim.x) {
          sum += tmp[i];
          i++;
        }
        tmpBuf[group_id] = sum;
    }

    // wait
    cooperative_groups::this_grid().sync();

    if (((blockIdx.x * blockDim.x) + threadIdx.x) == 0) {
        for (uint i = 1; i < groups; ++i) {
          sum += tmpBuf[i];
       }
       // *result = sum;
       result[1 + cooperative_groups::this_multi_grid().grid_rank()] = sum;
    }
    cooperative_groups::this_multi_grid().sync();
    if (cooperative_groups::this_multi_grid().grid_rank() == 0) {
      sum = 0;
      for (uint i = 1; i <= cooperative_groups::this_multi_grid().num_grids();
           ++i) {
        sum += result[i];
      }
      *result = sum;
    }
}
#endif
static const uint BufferSizeInDwords = 256 * 1024 * 1024;
static constexpr uint NumKernelArgs = 4;
static constexpr uint MaxGPUs = 8;
// callback function
static void HIPRT_CB CallBackFunctn(hipStream_t strm, hipError_t err,
                                    void *ChkVal) {
  // The following HIPASSERT() is just to satisfy catch2 framework.
  // As it ensures the use of all the variables.
  HIPASSERT(strm);
  HIPCHECK(err);
  if (*(reinterpret_cast<int*>(ChkVal)) == 1) {
    IfTestPassed = true;
  } else {
    IfTestPassed = false;
  }
}

static void EventSync() {
  int *Ad = nullptr, *Ah = nullptr, NumElms = 4096, CONST_NUM = 123;
  int blockSize = 32, peak_clk;
  HIP_CHECK(hipMalloc(&Ad, NumElms * sizeof(int)));
  Ah = new int[NumElms];
  for (int i = 0; i < NumElms; ++i) {
    Ah[i] = CONST_NUM;
  }
  // creating event objects
  hipEvent_t start, end;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&end));
  HIP_CHECK(hipMemcpy(Ad, Ah, NumElms * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceGetAttribute(&peak_clk, hipDeviceAttributeClockRate, 0));
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
  HIP_CHECK(hipEventRecord(start, hipStreamPerThread));
  StreamPerThrd<<<dimGrid, dimBlock, 0, hipStreamPerThread>>>(Ad, NULL, NumElms,
                                                              peak_clk, 0);
  HIP_CHECK(hipEventRecord(end, hipStreamPerThread));
  HIP_CHECK(hipEventSynchronize(end));
  HIP_CHECK(hipMemcpy(Ah, Ad, NumElms * sizeof(int), hipMemcpyDeviceToHost));
  int MisMatch = 0;
  for (int i = 0; i < NumElms; ++i) {
    if (Ah[i] != (CONST_NUM + 10)) {
      MisMatch++;
    }
  }
  delete[] Ah;
  HIP_CHECK(hipFree(Ad));
  if (MisMatch) {
    WARN("Data Mismatch observed!!\n");
    IfTestPassed = false;
  } else {
    IfTestPassed = true;
  }
}

/* Launch a kernel in hipStreamPerThread, while it is in flight check for
   hipStreamQuery(hipStreamPerThread) it should return hipErrorNotReady.*/
TEST_CASE("Unit_hipStreamPerThreadTst_StrmQuery") {
  int *Ad = nullptr, *Ah = nullptr, NumElms = 4096, CONST_NUM = 123;
  int blockSize = 32, peak_clk;
  hipError_t err;
  HIP_CHECK(hipMalloc(&Ad, NumElms * sizeof(int)));
  Ah = new int[NumElms];
  for (int i = 0; i < NumElms; ++i) {
    Ah[i] = CONST_NUM;
  }
  HIP_CHECK(hipMemcpy(Ad, Ah, NumElms * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceGetAttribute(&peak_clk, hipDeviceAttributeClockRate, 0));
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
  SECTION("Test working of hipStreamQuery") {
    StreamPerThrd<<<dimGrid, dimBlock, 0, hipStreamPerThread>>>(Ad, NULL,
                    NumElms, peak_clk, 1);
    err = hipStreamQuery(hipStreamPerThread);
    if (err != hipErrorNotReady) {
      WARN("hipStreamQuery on hipStreamPerThread didnt return expected error!");
      IfTestPassed = false;
    } else {
      IfTestPassed = true;
    }
  }
  SECTION("check working of hipStreamAddCallback() with hipStreamPerThread") {
    int *Hptr = nullptr, *A_d = nullptr;
    HIP_CHECK(hipHostMalloc(&Hptr, sizeof(int)));
    *Hptr = 0;
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), Hptr, 0));
    StreamPerThrd1<<<1, 1, 0, hipStreamPerThread>>>(A_d, peak_clk);
    HIP_CHECK(hipStreamAddCallback(hipStreamPerThread, CallBackFunctn, A_d, 0));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    HIP_CHECK(hipHostFree(Hptr));
  }
  HIP_CHECK(hipFree(Ad));
  delete[] Ah;
  REQUIRE(IfTestPassed);
}

/* Testing hipStreamPerThread stream object with hipMallocManaged() memory*/
TEST_CASE("Unit_hipStreamPerThread_MangdMem") {
  int managed = 0;
  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                  0));
  if (managed == 1) {
    int *Hmm = nullptr, NumElms = 4096, CONST_NUM = 123, blockSize = 32;
    SECTION("Using Managed memory") {
      HIP_CHECK(hipMallocManaged(&Hmm, NumElms * sizeof(int)));
      for (int i = 0; i < NumElms; ++i) {
        Hmm[i] = CONST_NUM;
      }
    }
    SECTION("Prefetching Managed memory to device") {
      HIP_CHECK(hipMallocManaged(&Hmm, NumElms * sizeof(int)));
      for (int i = 0; i < NumElms; ++i) {
        Hmm[i] = CONST_NUM;
      }
      HIP_CHECK(hipMemPrefetchAsync(Hmm, NumElms * sizeof(int), 0,
                hipStreamPerThread));
    }
    int peak_clk;
    HIP_CHECK(hipDeviceGetAttribute(&peak_clk, hipDeviceAttributeClockRate, 0));
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
    StreamPerThrd<<<dimGrid, dimBlock, 0, hipStreamPerThread>>>(Hmm, NULL,
                    NumElms, peak_clk, 0);
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    // Validating the result
    int MisMatch = 0;
    for (int i = 0; i < NumElms; ++i) {
      if (Hmm[i] != (CONST_NUM + 10)) {
        MisMatch++;
      }
    }
    HIP_CHECK(hipFree(Hmm));
    if (MisMatch) {
      WARN("Data mismatch observed!!\n");
      REQUIRE(false);
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
            "attribute. Hence skipping the testing with Pass result.\n");
  }
}

/*  To check the working of hipStreamPerThread in forked process*/
#ifdef __linux__
TEST_CASE("Unit_hipStreamPerThread_ChildProc") {
  if (fork() == 0) {  //  child process
    int *Ad = nullptr, *Ah = nullptr, NumElms = 4096, CONST_NUM = 123;
    int blockSize = 32, peak_clk;
    HIP_CHECK(hipMalloc(&Ad, NumElms * sizeof(int)));
    Ah = new int[NumElms];
    for (int i = 0; i < NumElms; ++i) {
      Ah[i] = CONST_NUM;
    }
    HIP_CHECK(hipMemcpy(Ad, Ah, NumElms * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceGetAttribute(&peak_clk, hipDeviceAttributeClockRate, 0));
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
    StreamPerThrd<<<dimGrid, dimBlock, 0, hipStreamPerThread>>>(Ad, NULL,
                    NumElms, peak_clk, 0);
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    HIP_CHECK(hipMemcpy(Ah, Ad, NumElms * sizeof(int), hipMemcpyDeviceToHost));
    int MisMatch = 0;
    for (int i = 0; i < NumElms; ++i) {
      if (Ah[i] != (CONST_NUM + 10)) {
        MisMatch++;
      }
    }
    delete[] Ah;
    HIP_CHECK(hipFree(Ad));
    if (MisMatch) {
      WARN("Data Mismatch observed!!\n");
      exit(9);
    } else {
      exit(10);
    }
  } else {  //  Parent process
    int stat;
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif

/* The following test case tests the working of hipEventSynchronize in
   multiple threads which are launched in quick succession*/
TEST_CASE("Unit_hipStreamPerThread_EvtRcrdMThrd") {
  IfTestPassed = true;
  int MAX_THREAD_CNT = 20;
  std::vector<std::thread> threads(MAX_THREAD_CNT);
  for (auto &th : threads) {
    th = std::thread(EventSync);
  }
  for (auto& th : threads) {
    th.join();
  }
  REQUIRE(IfTestPassed);
}

/* The following test case checks the working of hipStreamWaitEvent() with
   hipStreamWaitEvent()*/
TEST_CASE("Unit_hipStreamPerThread_StrmWaitEvt") {
  IfTestPassed = true;
  int *Ad = nullptr, NumElms = 4096, CONST_NUM = 123, blockSize = 32, *Ah = nullptr;
  int *Ad1 = nullptr, *Ah1 = nullptr;
  Ah = new int[NumElms];
  Ah1 = new int;
  hipStream_t Strm;
  HIP_CHECK(hipStreamCreate(&Strm));
  for (int i = 0; i < NumElms; ++i) {
    Ah[i] = CONST_NUM;
  }
  Ah1[0] = 0;
  HIP_CHECK(hipMalloc(&Ad, NumElms * sizeof(int)));
  HIP_CHECK(hipMemcpy(Ad, Ah, NumElms * sizeof(int), hipMemcpyHostToDevice));
  memset(Ah, 0, NumElms * sizeof(int));
  HIP_CHECK(hipMalloc(&Ad1, sizeof(int)));
  HIP_CHECK(hipMemset(Ad1, 0, sizeof(int)));
  int peak_clk;
  HIP_CHECK(hipDeviceGetAttribute(&peak_clk, hipDeviceAttributeClockRate, 0));
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
  hipEvent_t e1;
  HIPCHECK(hipEventCreate(&e1));
  StreamPerThrd<<<dimGrid, dimBlock, 0, Strm>>>(Ad, Ad1, NumElms,
  peak_clk, 1, 1);
  HIP_CHECK(hipEventRecord(e1, Strm));
  HIP_CHECK(hipStreamWaitEvent(hipStreamPerThread, e1, 0 /*flags*/));
  MiniKernel<<<1, 1, 0, hipStreamPerThread>>>(Ad1);
  sleep(1);
  HIP_CHECK(hipMemcpy(Ah1, Ad1, sizeof(int), hipMemcpyDeviceToHost));
  if (*Ah1  != 3) {
    IfTestPassed = false;
    if (*Ah1 == 2) {
      WARN("hipStreamPerThread didn't honour hipStreamWaitEvent()");
    } else if (*Ah1 == 4) {
      WARN("Unexpected behavior observed with hipStreamPerThread");
    }
  }
  // Validating the result
  HIP_CHECK(hipMemcpy(Ah, Ad, NumElms * sizeof(int), hipMemcpyDeviceToHost));
  int MisMatch = 0;
  for (int i = 0; i < NumElms; ++i) {
    if (Ah[i] != (CONST_NUM + 10)) {
      MisMatch++;
    }
  }
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Ad1));
  HIP_CHECK(hipStreamDestroy(Strm));
  delete[] Ah;
  delete Ah1;
  if (MisMatch) {
    WARN("Data mismatch observed!!\n");
    IfTestPassed = false;
  }
  REQUIRE(IfTestPassed);
}


/* Testing hipLaunchCooperativeKernel() api with hipStreamPerThread*/
TEST_CASE("Unit_hipStreamPerThread_CoopLaunch") {
  hipDeviceProp_t device_properties;
  HIPCHECK(hipGetDeviceProperties(&device_properties, 0));
  /* Test whether target device supports cooperative groups ****************/
  if (device_properties.cooperativeLaunch == 0) {
    SUCCEED("Cooperative group support not available...");
  } else {
    /* We will launch enough waves to fill up all of the GPU *****************/
    int warp_size = device_properties.warpSize;
    int num_sms = device_properties.multiProcessorCount;
    // long long totalTicks = device_properties.clockRate ;
    int max_blocks_per_sm = 0;
    // Calculate the device occupancy to know how many blocks can be run.
    HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                          StreamPerThrdCoopKrnl,
                                                          warp_size, 0));
    int max_active_blocks = max_blocks_per_sm * num_sms;
    int *Ad = nullptr, *Ah = nullptr, *DNumElms = nullptr, NumElms = 4096;
    int Const = 123;
    Ah = new int[NumElms];
    for (int i = 0; i < NumElms; ++i) {
      Ah[i] = Const;
    }
    HIP_CHECK(hipMalloc(&Ad, sizeof(int) * NumElms));
    HIP_CHECK(hipMalloc(&DNumElms, sizeof(int)));
    HIP_CHECK(hipMemcpyAsync(Ad, Ah, sizeof(int) * NumElms,
                             hipMemcpyHostToDevice, hipStreamPerThread));
    HIP_CHECK(hipMemcpyAsync(DNumElms, &NumElms, sizeof(int),
                             hipMemcpyHostToDevice, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    void *coop_params[2];
    coop_params[0] = reinterpret_cast<void*>(&Ad);
    coop_params[1] = reinterpret_cast<void*>(&DNumElms);
    HIP_CHECK(hipLaunchCooperativeKernel(
              reinterpret_cast<void*>(StreamPerThrdCoopKrnl),
              max_active_blocks, warp_size,
              coop_params, 0, hipStreamPerThread));
    HIP_CHECK(hipMemcpy(Ah, Ad, sizeof(int) * NumElms, hipMemcpyDeviceToHost));
    // Verifying the result
    int DataMismatch = 0;
    for (int i = 0; i < NumElms; ++i) {
      if (Ah[i] != (Const + 10)) {
        DataMismatch++;
      }
    }
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(DNumElms));
    delete[] Ah;
    if (DataMismatch > 0) {
      REQUIRE(false);
    }
  }
}

/* Testing hipLaunchCooperativeKernelMultiDevice() with hipStreamPerThread*/
#if HT_AMD
TEST_CASE("Unit_hipStreamPerThread_CoopLaunchMDev") {
  uint* dA[MaxGPUs];
  int64_t* dB[MaxGPUs];
  int64_t* dC;

  uint32_t* init = new uint32_t[BufferSizeInDwords];
  for (uint32_t i = 0; i < BufferSizeInDwords; ++i) {
    init[i] = i;
  }

  int nGpu = 0;
  HIPCHECK(hipGetDeviceCount(&nGpu));
  size_t copySizeInDwords = BufferSizeInDwords / nGpu;
  hipDeviceProp_t deviceProp[MaxGPUs];

  for (int i = 0; i < nGpu; i++) {
    HIPCHECK(hipSetDevice(i));

    // Calculate the device occupancy to know how many blocks can be
    // run concurrently
    hipGetDeviceProperties(&deviceProp[i], 0);
    if (!deviceProp[i].cooperativeMultiDeviceLaunch) {
      WARN("Device doesn't support cooperative launch!");
      SUCCEED("");
    }
    size_t SIZE = copySizeInDwords * sizeof(uint);

    HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dA[i]), SIZE));
    HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dB[i]),
             64 * deviceProp[i].multiProcessorCount * sizeof(int64_t)));
    if (i == 0) {
      HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&dC),
      (nGpu + 1) * sizeof(int64_t)));
    }
    HIPCHECK(hipMemcpy(dA[i], &init[i * copySizeInDwords] , SIZE,
             hipMemcpyHostToDevice));
    hipDeviceSynchronize();
  }

  dim3 dimBlock;
  dim3 dimGrid;
  dimGrid.x = 1;
  dimGrid.y = 1;
  dimGrid.z = 1;
  dimBlock.x = 64;
  dimBlock.y = 1;
  dimBlock.z = 1;

  int numBlocks = 0;
  uint workgroups[3] = {64, 128, 256};

  hipLaunchParams* launchParamsList = new hipLaunchParams[nGpu];
  std::time_t end_time;
  double time = 0;
  for (uint set = 0; set < 3; ++set) {
    void* args[MaxGPUs * NumKernelArgs];
    WARN("---------- Test#" << set << ", size: "<< BufferSizeInDwords <<
      " dwords ---------------\n");
    for (int i = 0; i < nGpu; i++) {
      HIPCHECK(hipSetDevice(i));
      dimBlock.x = workgroups[set];
      HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
      test_gwsPerThrd, dimBlock.x * dimBlock.y * dimBlock.z,
      dimBlock.x * sizeof(int64_t)));

      WARN("GPU(" << i << ") Block size: " << dimBlock.x <<
        " Num blocks per CU: " << numBlocks << "\n");

      dimGrid.x = deviceProp[i].multiProcessorCount * (std::min)(numBlocks, 32);

      args[i * NumKernelArgs]     = reinterpret_cast<void*>(&dA[i]);
      args[i * NumKernelArgs + 1] = reinterpret_cast<void*>(&copySizeInDwords);
      args[i * NumKernelArgs + 2] = reinterpret_cast<void*>(&dB[i]);
      args[i * NumKernelArgs + 3] = reinterpret_cast<void*>(&dC);

      launchParamsList[i].func = reinterpret_cast<void*>(test_gwsPerThrd);
      launchParamsList[i].gridDim = dimGrid;
      launchParamsList[i].blockDim = dimBlock;
      launchParamsList[i].sharedMem = dimBlock.x * sizeof(int64_t);

      launchParamsList[i].stream = hipStreamPerThread;
      launchParamsList[i].args = &args[i * NumKernelArgs];
    }

    system_clock::time_point start = system_clock::now();
    hipLaunchCooperativeKernelMultiDevice(launchParamsList, nGpu, 0);
    for (int i = 0; i < nGpu; i++) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipDeviceSynchronize());
    }
    system_clock::time_point end = system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    end_time = std::chrono::system_clock::to_time_t(end);

    time += elapsed_seconds.count();

    size_t processedDwords = copySizeInDwords * nGpu;
    if (*dC != (((int64_t)(processedDwords) * (processedDwords - 1)) / 2)) {
      WARN("Data validation failed ("<< *dC << " != " <<
        (((int64_t)(BufferSizeInDwords) * (BufferSizeInDwords - 1)) / 2) <<
        ") for grid size = " << dimGrid.x << " and block size = " <<
        dimBlock.x << "\n");
      WARN("Test failed!");
    }
  }

  delete [] launchParamsList;

  WARN("finished computation at " << std::ctime(&end_time));
  WARN("elapsed time: " << time << "s\n");

  hipSetDevice(0);
  hipFree(dC);
  for (int i = 0; i < nGpu; i++) {
    hipFree(dA[i]);
    hipFree(dB[i]);
  }
  delete [] init;
}
#endif
