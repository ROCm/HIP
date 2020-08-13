/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_common.h"
#include <iostream>
#include <time.h>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <numaif.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include "hip/hip_runtime.h"
/* HIT_START
 * BUILD_CMD: hipPerfHostNumaAlloc %hc -I%S/../../src %S/%s %S/../../src/test_common.cpp -lnuma -o %T/%t EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

// To run it correctly, we must not export HIP_VISIBLE_DEVICES.
// And we must explicitly link libnuma because of numa api move_pages().
#define NUM_PAGES 4
char *h = nullptr;
char *d_h = nullptr;
char *m = nullptr;
char *d_m = nullptr;
int page_size = 0;
const int mode[] = { MPOL_DEFAULT, MPOL_BIND, MPOL_PREFERRED, MPOL_INTERLEAVE };
const char* modeStr[] = { "MPOL_DEFAULT", "MPOL_BIND", "MPOL_PREFERRED", "MPOL_INTERLEAVE" };

std::string exeCommand(const char* cmd) {
  std::array<char, 128> buff;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    return result;
  }
  while (fgets(buff.data(), buff.size(), pipe.get()) != nullptr) {
    result += buff.data();
  }
  return result;
}

int getCpuAgentCount() {
  const char* cmd = "cat /proc/cpuinfo | grep \"physical id\" | sort | uniq | wc -l";
  int cpuAgentCount = std::atoi(exeCommand(cmd).c_str());
  return cpuAgentCount;
}

bool test(int cpuId, int gpuId, int numaMode, unsigned int hostMallocflags) {
  void *pages[NUM_PAGES];
  int status[NUM_PAGES];
  int nodes[NUM_PAGES];
  int ret_code;

  printf("set cpu %d, gpu %d, numaMode %d, hostMallocflags 0x%x\n", cpuId,
         gpuId, numaMode, hostMallocflags);

  if (cpuId >= 0) {
    unsigned long nodeMask = 1 << cpuId;
    unsigned long maxNode = sizeof(nodeMask) * 8;
    if (set_mempolicy(numaMode, numaMode == MPOL_DEFAULT ? NULL : &nodeMask,
                      numaMode == MPOL_DEFAULT ? 0 : maxNode) == -1) {
      printf("set_mempolicy() failed with err %d\n", errno);
      return false;
    }
  }

  if (gpuId >= 0) {
    HIPCHECK(hipSetDevice(gpuId));
  }

  posix_memalign((void**) &m, page_size, page_size * NUM_PAGES);
  hipHostRegister(m, page_size * NUM_PAGES, hipHostRegisterMapped);
  hipHostGetDevicePointer((void**) &d_m, m, 0);

  status[0] = -1;
  pages[0] = m;
  for (int i = 1; i < NUM_PAGES; i++) {
    pages[i] = (char*) pages[0] + page_size;
  }
  ret_code = move_pages(0, NUM_PAGES, pages, NULL, status, 0);
  printf("Memory (malloc) ret %d at %p (dev %p) is at node: ", ret_code, m, d_m);
  for (int i = 0; i < NUM_PAGES; i++) {
    printf("%d ", status[i]); // Don't verify as it's out of our control
  }
  printf("\n");

  HIPCHECK(hipHostMalloc((void**) &h, page_size*NUM_PAGES, hostMallocflags));
  pages[0] = h;
  for (int i = 1; i < NUM_PAGES; i++) {
    pages[i] = (char*) pages[0] + page_size;
  }
  ret_code = move_pages(0, NUM_PAGES, pages, NULL, status, 0);
  d_h = nullptr;
  if (hostMallocflags & hipHostMallocMapped) {
    hipHostGetDevicePointer((void**) &d_h, h, 0);
    printf("Memory (hipHostMalloc) ret %d at %p (dev %p) is at node: ",
           ret_code, h, d_h);
  } else {
    printf("Memory (hipHostMalloc) ret %d at %p is at node: ", ret_code, h);
  }
  for (int i = 0; i < NUM_PAGES; i++) {
    printf("%d ", status[i]);  // Always print it even if it's wrong. Verify later
  }
  printf("\n");

  HIPCHECK(hipHostFree((void* )h));
  hipHostUnregister(m);
  free(m);

  if (cpuId >= 0 && (numaMode == MPOL_BIND || numaMode == MPOL_PREFERRED)) {
    for (int i = 0; i < NUM_PAGES; i++) {
      if (status[i] != cpuId) {  // Now verify
        printf("Failed at %d", i);
        return false;
      }
    }
  }
  return true;
}

bool runTest(const int &cpuCount, const int &gpuCount,
             const unsigned int &hostMallocflags, const std::string &str) {
  printf("%s\n", str.c_str());

  for (int m = 0; m < sizeof(mode) / sizeof(mode[0]); m++) {
    printf("Testing %s\n", modeStr[m]);

    for (int i = 0; i < cpuCount; i++) {
      for (int j = 0; j < gpuCount; j++) {
        if (!test(i, j, mode[m], hostMallocflags)) {
          return false;
        }
      }
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  int gpuCount = 0;
  HIPCHECK(hipGetDeviceCount(&gpuCount));
  int cpuCount = getCpuAgentCount();
  page_size = getpagesize();
  printf("Cpu count %d, Gpu count %d, Page size %d\n", cpuCount, gpuCount,
         page_size);

  if (cpuCount < 0 || gpuCount < 0) {
    failed("Bad device count\n");
    return -1;
  }

  if (!runTest(cpuCount, gpuCount, hipHostMallocDefault | hipHostMallocNumaUser,
               "Testing hipHostMallocDefault | hipHostMallocNumaUser........................")) {
    failed("Failed testing hipHostMallocDefault | hipHostMallocNumaUser\n");
    return -1;
  }

  if (!runTest(cpuCount, gpuCount, hipHostMallocMapped | hipHostMallocNumaUser,
               "Testing hipHostMallocMapped | hipHostMallocNumaUser.........................")) {
    failed("Failed testing hipHostMallocMapped | hipHostMallocNumaUser\n");
    return -1;
  }

  passed();
}
