/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

/**

Testcase Scenarios
------------------
Functional:
1) Validate usecase of Event handle along with memory handle across multiple
processes with complex scenario.

Negative/Argument Validation:
1) Get event handle with eventHandle(nullptr).
2) Get event handle with event(nullptr).
3) Get event handle with invalid event object.
4) Get event handle for event allocated without Interprocess flag.
5) Open event handle with event(nullptr).
6) Open event handle with eventHandle as invalid.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>


#define BUF_SIZE        4096
#define MAX_DEVICES     8


typedef struct ipcEventInfo {
  int device;
  pid_t pid;
  hipIpcEventHandle_t eventHandle;
  hipIpcMemHandle_t memHandle;
} ipcEventInfo_t;

typedef struct ipcDevices {
  int count;
  int ordinals[MAX_DEVICES];
} ipcDevices_t;

typedef struct ipcBarrier {
  int count;
  bool sense;
  bool allExit;
} ipcBarrier_t;

/**
  Get device count and list down devices with
  P2P access with Device 0.
*/
void getDevices(ipcDevices_t *devices) {
  pid_t pid = fork();

  if (!pid) {
    // HIP APIs are called in child process,
    // to avoid HIP Initialization in main process.
    int i, devCnt{};
    HIP_CHECK(hipGetDeviceCount(&devCnt));

    if (devCnt < 2) {
        devices->count = 0;
        WARN("Count less than expected number of devices");
        exit(EXIT_SUCCESS);
    }

    // Device 0
    devices->ordinals[0] = 0;
    devices->count = 1;

    // Check possibility for peer accesses, relevant to our tests
    INFO("Checking GPU(s) for support of p2p memory access ");
    INFO("Between GPU0 and other GPU(s)");

    int canPeerAccess_0i, canPeerAccess_i0;
    for (i = 1; i < devCnt; i++) {
        HIP_CHECK(hipDeviceCanAccessPeer(&canPeerAccess_0i, 0, i));
        HIP_CHECK(hipDeviceCanAccessPeer(&canPeerAccess_i0, i, 0));

        if (canPeerAccess_0i * canPeerAccess_i0) {
            devices->ordinals[i] = i;
            INFO("Two-way peer access is available between GPU"
            << devices->ordinals[0] <<" and GPU"
            << devices->ordinals[devices->count]);
            devices->count += 1;
        }
    }

    exit(EXIT_SUCCESS);
  } else {
      int status;
      waitpid(pid, &status, 0);
      HIP_ASSERT(!status);
  }
}

static ipcBarrier_t *g_Barrier{};
static bool g_procSense;
static int g_processCnt;

/**
 Calling process waits for other processes to signal/complete.
*/
void processBarrier() {
  int newCount = __sync_add_and_fetch(&g_Barrier->count, 1);

  if (newCount == g_processCnt) {
    g_Barrier->count = 0;
    g_Barrier->sense = !g_procSense;

  } else {
    while (g_Barrier->sense == g_procSense) {
        if (!g_Barrier->allExit) {
          sched_yield();
        } else {
          exit(EXIT_FAILURE);
        }
    }
  }

  g_procSense = !g_procSense;
}


__global__ void computeKernel(int *dst, int *src, int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] / num;
}

/**
 * 1) Process 0 allocates buffer in GPU0 memory and exports the memory handle.
 * 2) Other processes opens memory handle of GPU0 memory, performs computation
 * and records event.
 * 3) Process 0 synchronizes event and validates the resulting buffer.
 */
void runMultiProcKernel(ipcEventInfo_t *shmEventInfo, int index) {
  int *d_ptr;
  int hData[BUF_SIZE]{};
  unsigned int seed = time(nullptr);

  // Randomize data before computation
  for (int i = 0; i < BUF_SIZE; i++) {
      hData[i] = rand_r(&seed);
  }

  HIP_CHECK(hipSetDevice(shmEventInfo[index].device));

  if (index == 0) {
    int h_results[BUF_SIZE * MAX_DEVICES];
    hipEvent_t event[MAX_DEVICES];

    HIP_CHECK(hipMalloc(&d_ptr, BUF_SIZE * g_processCnt * sizeof(int)));
    HIP_CHECK(hipIpcGetMemHandle(&shmEventInfo[0].memHandle, d_ptr));
    HIP_CHECK(hipMemcpy(d_ptr, hData,
                          BUF_SIZE * sizeof(int), hipMemcpyHostToDevice));

    // Barrier 1: Process0 will wait for all processes to create event handles,
    // signals device memory creation.
    processBarrier();

    for (int i = 1; i < g_processCnt; i++) {
      HIP_CHECK(hipIpcOpenEventHandle(&event[i], shmEventInfo[i].eventHandle));
    }

    // Barrier 2: Process0 waits for kernels to be launched
    // and the events to be recorded.
    processBarrier();

    for (int i = 1; i < g_processCnt; i++) {
      HIP_CHECK(hipEventSynchronize(event[i]));
    }

    HIP_CHECK(hipMemcpy(h_results, d_ptr + BUF_SIZE,
        BUF_SIZE * (g_processCnt - 1) * sizeof(int), hipMemcpyDeviceToHost));

    // Barrier 3: Process0 signals event usage is done.
    processBarrier();
    HIP_CHECK(hipFree(d_ptr));
    for (int n = 1; n < g_processCnt; n++) {
        for (int i = 0; i < BUF_SIZE; i++) {
            if (hData[i]/(n + 1) != h_results[(n-1) * BUF_SIZE + i]) {
                WARN("Data validation error at index " << i << " n" << n);
                g_Barrier->allExit = true;
                exit(EXIT_FAILURE);
            }
        }
    }
  } else {
    hipEvent_t event;
    HIP_CHECK(hipEventCreateWithFlags(&event,
                               hipEventDisableTiming | hipEventInterprocess));
    HIP_CHECK(hipIpcGetEventHandle(&shmEventInfo[index].eventHandle, event));

    // Barrier 1 : wait until proc 0 initializes device memory,
    // signals event creation.
    processBarrier();
    HIP_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void **>(&d_ptr),
                                               shmEventInfo[0].memHandle,
                                   hipIpcMemLazyEnablePeerAccess));
    const dim3 threads(512, 1);
    const dim3 blocks(BUF_SIZE / threads.x, 1);
    hipLaunchKernelGGL(computeKernel, dim3(blocks), dim3(threads), 0, 0,
                                    d_ptr + index *BUF_SIZE, d_ptr, index + 1);
    HIP_CHECK(hipEventRecord(event));

    // Barrier 2 : Signals that event is recorded
    processBarrier();
    HIP_CHECK(hipIpcCloseMemHandle(d_ptr));

    // Barrier 3 : wait for all the events to be used up by processes
    processBarrier();
    HIP_CHECK(hipEventDestroy(event));
  }
}

/**
 Functional test demonstrating IPC event usage along with IPC memory handle
*/
TEST_CASE("Unit_hipIpcEventHandle_Functional") {
  ipcDevices_t *shmDevices;
  ipcEventInfo_t *shmEventInfo;
  shmDevices = reinterpret_cast<ipcDevices_t *> (mmap(NULL, sizeof(*shmDevices),
                    PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0));
  REQUIRE(MAP_FAILED != shmDevices);

  getDevices(shmDevices);

  if (shmDevices->count < 2) {
    WARN("Test requires atleast two GPUs with P2P access. Skipping test.");
    return;
  }

  g_processCnt = shmDevices->count;

  // Barrier is used to synchronize processes created.
  g_Barrier = reinterpret_cast<ipcBarrier_t *> (mmap(NULL, sizeof(*g_Barrier),
                   PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0));
  REQUIRE(MAP_FAILED != g_Barrier);
  memset(g_Barrier, 0, sizeof(*g_Barrier));

  // set local barrier sense flag
  g_procSense = 0;

  // shared memory for Event and memHandle Info
  shmEventInfo = reinterpret_cast<ipcEventInfo_t *>(mmap(NULL,
                                          g_processCnt * sizeof(*shmEventInfo),
                    PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0));
  REQUIRE(MAP_FAILED != shmEventInfo);

  // initialize shared memory
  memset(shmEventInfo, 0, g_processCnt * sizeof(*shmEventInfo));

  int index = 0;

  for (int i = 1; i < g_processCnt; i++) {
      int pid = fork();

      if (!pid) {
          index = i;
          break;
      } else {
          shmEventInfo[i].pid = pid;
      }
  }

  shmEventInfo[index].device = shmDevices->ordinals[index];

  // Run the test
  runMultiProcKernel(shmEventInfo, index);

  // Cleanup
  if (index == 0) {
    for (int i = 1; i < g_processCnt; i++) {
        int status;
        waitpid(shmEventInfo[i].pid, &status, 0);
        HIP_ASSERT(WIFEXITED(status));
    }
  }
}

/**
 Performs API Parameter validation.
*/
TEST_CASE("Unit_hipIpcEventHandle_ParameterValidation") {
  hipEvent_t event;
  hipIpcEventHandle_t eventHandle;
  hipError_t ret;
  HIP_CHECK(hipEventCreateWithFlags(&event,
                             hipEventDisableTiming | hipEventInterprocess));
#if HT_AMD
  // Test disabled for nvidia due to segfault with cuda api
  SECTION("Get event handle with eventHandle(nullptr)") {
    ret = hipIpcGetEventHandle(nullptr, event);
    REQUIRE(ret == hipErrorInvalidValue);
  }
#endif

  SECTION("Get event handle with event(nullptr)") {
    ret = hipIpcGetEventHandle(&eventHandle, nullptr);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("Get event handle with invalid event object") {
    hipEvent_t eventUninit{};
    ret = hipIpcGetEventHandle(&eventHandle, eventUninit);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("Get event handle for event allocated without Interprocess flag") {
    hipEvent_t eventNoIpc;
    HIP_CHECK(hipEventCreateWithFlags(&eventNoIpc, hipEventDisableTiming));

    ret = hipIpcGetEventHandle(&eventHandle, eventNoIpc);
    if ((ret != hipErrorInvalidResourceHandle) &&
       (ret != hipErrorInvalidConfiguration)) {
      INFO("Error returned : " << ret);
      REQUIRE(false);
    }
  }

  SECTION("Open event handle with event(nullptr)") {
    hipIpcEventHandle_t ipc_handle{};
    ret = hipIpcOpenEventHandle(nullptr, ipc_handle);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("Open event handle with eventHandle as invalid") {
    hipIpcEventHandle_t ipc_handle{};
    hipEvent_t eventOut;
    ret = hipIpcOpenEventHandle(&eventOut, ipc_handle);
    if ((ret != hipErrorInvalidValue) && (ret != hipErrorMapFailed)) {
      INFO("Error returned : " << ret);
      REQUIRE(false);
    }
  }
}

#endif
