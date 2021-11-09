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
#include "test_common.h"

#include <thread>
#ifdef __linux__
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

// standard global variables that can be set on command line
size_t N = 4 * 1024 * 1024;
char memsetval = 0x42;
int memsetD32val = 0xDEADBEEF;
short memsetD16val = 0xDEAD;
char memsetD8val = 0xDE;
int iterations = 1;
unsigned blocksPerCU = 6;  // to hide latency
unsigned threadsPerBlock = 256;
int textureFilterMode = 0; // 0: hipFilterModePoint; 1: hipFilterModeLinear
int p_gpuDevice = 0;
unsigned p_verbose = 0;
int p_tests = -1; /*which tests to run. Interpretation is left to each test.  default:all*/
int debug_test = 0;
#ifdef _WIN64
const char* HIP_VISIBLE_DEVICES_STR = "HIP_VISIBLE_DEVICES=";
const char* CUDA_VISIBLE_DEVICES_STR = "CUDA_VISIBLE_DEVICES=";
const char* PATH_SEPERATOR_STR = "\\";
const char* NULL_DEVICE = "NUL:";
#else
const char* HIP_VISIBLE_DEVICES_STR = "HIP_VISIBLE_DEVICES";
const char* CUDA_VISIBLE_DEVICES_STR = "CUDA_VISIBLE_DEVICES";
const char* PATH_SEPERATOR_STR = "/";
const char* NULL_DEVICE = "/dev/null";
#endif

#ifdef _WIN64
// Windows does not have rand_r, use srand and rand instead.
int rand_r(unsigned int* s) {
  srand(*s);
  return rand();
}
#endif

// Get Free Memory from the system
static size_t getMemoryAmount() {
#if __linux__
  struct sysinfo info;
  int _ = sysinfo(&info);
  return info.freeram / (1024 * 1024);  // MB
#elif defined(_WIN32)
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  GlobalMemoryStatusEx(&statex);
  return (statex.ullAvailPhys / (1024 * 1024));  // MB
#endif
}

size_t getHostThreadCount(const size_t memPerThread, const size_t maxThreads) {
  if (memPerThread == 0) return 0;
  auto memAmount = getMemoryAmount();
  const auto processor_count = std::thread::hardware_concurrency();
  if (processor_count == 0 || memAmount == 0) return 0;
  size_t thread_count = 0;
  if ((processor_count * memPerThread) < memAmount)
    thread_count = processor_count;
  else
    thread_count = reinterpret_cast<size_t>(memAmount / memPerThread);
  if (maxThreads > 0) {
    return (thread_count > maxThreads) ? maxThreads : thread_count;
  }
  return thread_count;
}

namespace HipTest {


double elapsed_time(long long startTimeUs, long long stopTimeUs) {
    return ((double)(stopTimeUs - startTimeUs)) / ((double)(1000));
}


int parseSize(const char* str, size_t* output) {
    char* next;
    *output = strtoull(str, &next, 0);
    int l = strlen(str);
    if (l) {
        char c = str[l - 1];  // last char.
        if ((c == 'k') || (c == 'K')) {
            *output *= 1024;
        }
        if ((c == 'm') || (c == 'M')) {
            *output *= (1024 * 1024);
        }
        if ((c == 'g') || (c == 'G')) {
            *output *= (1024 * 1024 * 1024);
        }
    }
    return 1;
}


int parseUInt(const char* str, unsigned int* output) {
    char* next;
    *output = strtoul(str, &next, 0);
    return !strlen(next);
}


int parseInt(const char* str, int* output) {
    char* next;
    *output = strtol(str, &next, 0);
    return !strlen(next);
}


int parseStandardArguments(int argc, char* argv[], bool failOnUndefinedArg) {
    int extraArgs = 1;
    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];

        if (!strcmp(arg, " ")) {
            // skip NULL args.
        } else if (!strcmp(arg, "--N") || (!strcmp(arg, "-N"))) {
            if (++i >= argc || !HipTest::parseSize(argv[i], &N)) {
                failed("Bad N size argument");
            }
        } else if (!strcmp(arg, "--threadsPerBlock")) {
            if (++i >= argc || !HipTest::parseUInt(argv[i], &threadsPerBlock)) {
                failed("Bad threadsPerBlock argument");
            }
        } else if (!strcmp(arg, "--blocksPerCU")) {
            if (++i >= argc || !HipTest::parseUInt(argv[i], &blocksPerCU)) {
                failed("Bad blocksPerCU argument");
            }
        } else if (!strcmp(arg, "--memsetval")) {
            int ex;
            if (++i >= argc || !HipTest::parseInt(argv[i], &ex)) {
                failed("Bad memsetval argument");
            }
            memsetval = ex;
        } else if (!strcmp(arg, "--memsetD32val")) {
            int ex;
            if (++i >= argc || !HipTest::parseInt(argv[i], &ex)) {
                failed("Bad memsetD32val argument");
            }
            memsetD32val = ex;
        } else if (!strcmp(arg, "--memsetD16val")) {
            int ex;
            if (++i >= argc || !HipTest::parseInt(argv[i], &ex)) {
                failed("Bad memsetD16val argument");
            }
            memsetD16val = ex;
        } else if (!strcmp(arg, "--memsetD8val")) {
            int ex;
            if (++i >= argc || !HipTest::parseInt(argv[i], &ex)) {
                failed("Bad memsetD8val argument");
            }
            memsetD8val = ex;
        } else if (!strcmp(arg, "--textureFilterMode")) {
          int mode;
          if (++i >= argc || !HipTest::parseInt(argv[i], &mode)) {
              failed("Bad textureFilterMode argument");
          }
          textureFilterMode = mode;
        } else if (!strcmp(arg, "--iterations") || (!strcmp(arg, "-i"))) {
            if (++i >= argc || !HipTest::parseInt(argv[i], &iterations)) {
                failed("Bad iterations argument");
            }
        } else if (!strcmp(arg, "--gpu") || (!strcmp(arg, "-gpuDevice")) || (!strcmp(arg, "-g"))) {
            if (++i >= argc || !HipTest::parseInt(argv[i], &p_gpuDevice)) {
                failed("Bad gpuDevice argument");
            }

        } else if (!strcmp(arg, "--verbose") || (!strcmp(arg, "-v"))) {
            if (++i >= argc || !HipTest::parseUInt(argv[i], &p_verbose)) {
                failed("Bad verbose argument");
            }
        } else if (!strcmp(arg, "--tests") || (!strcmp(arg, "-t"))) {
            if (++i >= argc || !HipTest::parseInt(argv[i], &p_tests)) {
                failed("Bad tests argument");
            }

        } else if (!strcmp(arg, "--debug") || (!strcmp(arg, "-d"))) {
            if (++i >= argc || !HipTest::parseInt(argv[i], &debug_test)) {
                failed("Bad tests argument");
            }
        } else {
            if (failOnUndefinedArg) {
                failed("Bad argument '%s'", arg);
            } else {
                argv[extraArgs++] = argv[i];
            }
        }
    };

    return extraArgs;
}


unsigned setNumBlocks(unsigned blocksPerCU, unsigned threadsPerBlock, size_t N) {
    int device;
    HIPCHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIPCHECK(hipGetDeviceProperties(&props, device));

    unsigned blocks = props.multiProcessorCount * blocksPerCU;
    if (blocks * threadsPerBlock > N) {
        blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    }

    return blocks;
}


}  // namespace HipTest
