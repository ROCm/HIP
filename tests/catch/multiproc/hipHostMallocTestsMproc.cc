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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :
 1) Validate page table allocation of hipHostMalloc() api when
 HIP_VISIBLE_DEVICES set to single device.
 2) Validate page table allocation of hipHostMalloc() api when
 HIP_VISIBLE_DEVICES set to list of multiple devices.
*/

#include <hip_test_common.hh>
#include <hip_test_smi.hh>

#if defined(__linux__)
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <dlfcn.h>

#if HT_AMD
/* AMD specific as below tests use rocm-smi */

/**
 * Defines
 */
#define LIB_ROCMSMI "librocm_smi64.so"
#define ALLOC_SIZE (1*1024*1024)

/**
 * Global variables
 */
static rsmi_status_t (*rsmi_dev_memory_usage_get_fp)(uint32_t,
                                                     rsmi_memory_type_t,
                                                     uint64_t *);
static rsmi_status_t (*rsmi_init_fp)(uint64_t);
static rsmi_status_t (*rsmi_shut_down_fp)();
static void *rocm_smi_h;


/**
 * Fetches Gpu device count
 */
static void getDeviceCount(int *pdevCnt) {
  int fd[2], val = 0;
  pid_t childpid;

  // create pipe descriptors
  pipe(fd);

  // disable visible_devices env from shell
  unsetenv("ROCR_VISIBLE_DEVICES");
  unsetenv("HIP_VISIBLE_DEVICES");

  childpid = fork();

  if (childpid > 0) {  // Parent
    close(fd[1]);
    // parent will wait to read the device cnt
    read(fd[0], &val, sizeof(val));

    // close the read-descriptor
    close(fd[0]);

    // wait for child exit
    wait(nullptr);

    *pdevCnt = val;
  } else if (!childpid) {  // Child
    int devCnt = 1;
    // writing only, no need for read-descriptor
    close(fd[0]);

    HIP_CHECK(hipGetDeviceCount(&devCnt));
    // send the value on the write-descriptor:
    write(fd[1], &devCnt, sizeof(devCnt));

    // close the write descriptor:
    close(fd[1]);
    exit(0);
  } else {  // failure
    *pdevCnt = 0;
    return;
  }
}

/**
 * Initializes rocm smi library handles
 */
static bool rocm_smi_init() {
  // Open ROCm SMI Library
  if (!(rocm_smi_h = dlopen(LIB_ROCMSMI, RTLD_LAZY))) {
    printf("Error opening rocm smi library!\n");
    return false;
  }

  void* fnsym = dlsym(rocm_smi_h, "rsmi_dev_memory_usage_get");
  if (!fnsym) {
    printf("Error getting rsmi_dev_memory_usage_get() function\n");
    dlclose(rocm_smi_h);
    return false;
  }
  rsmi_dev_memory_usage_get_fp = reinterpret_cast<rsmi_status_t (*)(uint32_t,
                                       rsmi_memory_type_t, uint64_t *)>(fnsym);

  fnsym = dlsym(rocm_smi_h, "rsmi_init");
  if (!fnsym) {
    printf("Error getting rsmi_init() function\n");
    dlclose(rocm_smi_h);
    return false;
  }
  rsmi_init_fp = reinterpret_cast<rsmi_status_t (*)(uint64_t)>(fnsym);

  fnsym = dlsym(rocm_smi_h, "rsmi_shut_down");
  if (!fnsym) {
    printf("Error getting rsmi_shut_down() function\n");
    dlclose(rocm_smi_h);
    return false;
  }
  rsmi_shut_down_fp = reinterpret_cast<rsmi_status_t (*)()>(fnsym);

  uint64_t init_flags = 0;
  rsmi_status_t retsmi_init;
  retsmi_init = rsmi_init_fp(init_flags);
  if (RSMI_STATUS_SUCCESS != retsmi_init) {
    printf("Error when initializing rocm_smi\n");
    dlclose(rocm_smi_h);
    return false;
  }

  return true;
}

/**
 * Exits rocm smi library
 */
static void rocm_smi_exit() {
  rsmi_shut_down_fp();
  dlclose(rocm_smi_h);
}

/**
 * Validates page table memory allocations
 * by setting visible devices selected.
 */
static bool validatePageTableAllocations(const char *devList, int visDevCnt) {
  int fd[2];
  bool testResult = false;
  pid_t pid;
  int numdev = 0;

  getDeviceCount(&numdev);
  REQUIRE(numdev > 0);

  if (pipe(fd) < 0) {
    printf("Pipe system call failed\n");
    return false;
  }

  pid = fork();

  if (!pid) {  // Child process
    rsmi_status_t ret;
    std::vector<int> prev, current;
    uint64_t used = 0;
    int tmpdev = 0, changeCnt = 0, indx = 0;
    char *ptr = nullptr;
    bool testPassed = true;

    // Disable visible_devices env from shell
    unsetenv("ROCR_VISIBLE_DEVICES");
    unsetenv("HIP_VISIBLE_DEVICES");

    setenv("HIP_VISIBLE_DEVICES", devList, 1);

    // First Call to initialize hip api
    hipGetDeviceCount(&tmpdev);


    // Get memory snapshot before hostmalloc
    for (indx = 0; indx < numdev; indx++) {
      ret = rsmi_dev_memory_usage_get_fp(indx, RSMI_MEM_TYPE_VRAM, &used);
      if (RSMI_STATUS_SUCCESS != ret) {
        printf("Error while running rsmi_dev_memory_usage_get func\n");
        dlclose(rocm_smi_h);
        rsmi_shut_down_fp();
        return false;
      }
      prev.push_back(used);
    }

    HIP_CHECK(hipHostMalloc(&ptr, ALLOC_SIZE));

    // Get memory snapshot after hostmalloc
    for (indx = 0; indx < numdev; indx++) {
      ret = rsmi_dev_memory_usage_get_fp(indx, RSMI_MEM_TYPE_VRAM, &used);
      if (RSMI_STATUS_SUCCESS != ret) {
        printf("Error while running rsmi_dev_memory_usage_get func\n");
        dlclose(rocm_smi_h);
        rsmi_shut_down_fp();
        hipHostFree(ptr);
        return false;
      }
      current.push_back(used);
    }

    for (indx = 0; indx < numdev; indx++) {
      if (current[indx] - prev[indx])
        changeCnt++;
    }

    // Check if memory allocation happened only for visible devices
    if (changeCnt == visDevCnt) {
      testPassed = true;
    } else {
      testPassed = false;
    }

    hipHostFree(ptr);

    // writing only, no need for read-descriptor
    close(fd[0]);

    // send the value on the write-descriptor:
    write(fd[1], &testPassed, sizeof(testPassed));

    // close the write descriptor:
    close(fd[1]);
    exit(0);
  } else if (pid > 0) {  // parent
    close(fd[1]);
    read(fd[0], &testResult, sizeof(testResult));
    close(fd[0]);
    wait(nullptr);
  } else {
    printf("fork() failed\n");
    HIP_ASSERT(false);
  }

  return testResult;
}

/**
 * Test page table allocation when HIP_VISIBLE_DEVICES set to
 * single device
 */
TEST_CASE("Unit_hipHostMalloc_SingleVisibleDevicePageAlloc") {
  int devCnt;
  std::string str;

  if (!rocm_smi_init()) {
    WARN("Testcase skipped as rocm smi not initialized/present");
    return;
  }

  getDeviceCount(&devCnt);
  REQUIRE(devCnt > 0);

  // Select single visible device and validate memory usage
  for (int i = 0; i < devCnt; i++) {
    str = std::to_string(i);
    REQUIRE(validatePageTableAllocations(str.c_str(), 1) == true);
  }

  rocm_smi_exit();
}

/**
 * Test page table allocation when HIP_VISIBLE_DEVICES set to
 * multiple devices
 */
TEST_CASE("Unit_hipHostMalloc_MultipleVisibleDevicesPageAlloc") {
  int devCnt = 0, vdCnt = 0;
  std::string str;

  if (!rocm_smi_init()) {
    WARN("Testcase skipped as rocm smi not initialized/present");
    return;
  }

  getDeviceCount(&devCnt);
  REQUIRE(devCnt > 0);

  // Select multiple visible devices and validate memory usage
  for (int i = 0; i < devCnt; i++) {
    if (i == 0)
      str += std::to_string(i);
    else
      str += "," + std::to_string(i);

    vdCnt++;
    REQUIRE(validatePageTableAllocations(str.c_str(), vdCnt) == true);
  }

  rocm_smi_exit();
}
#endif  // HT_AMD
#endif  // __linux__
