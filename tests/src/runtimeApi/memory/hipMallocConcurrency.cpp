/*
Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
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

 (TestCase 1)::
 1) Test hipMalloc() api passing zero size and confirming *ptr returning
 nullptr. Also pass nullptr to hipFree() api.
 2) Pass maximum value of size_t for hipMalloc() api and make sure appropriate
 error is returned.
 3) Check for hipMalloc() error code, passing invalid/null pointer.

 (TestCase 2)::
 4) Regress hipMalloc()/hipFree() in loop for bigger chunk of allocation
 with adequate number of iterations and later test for kernel execution on
 default gpu.
 5) Regress hipMalloc()/hipFree() in loop while allocating smaller chunks
 keeping maximum number of iterations and then run kernel code on default
 gpu, perfom data validation.

 (TestCase 3)::
 6) Check hipMalloc() api adaptability when app creates small chunks of memory
 continuously, stores it for later use and then frees it at later point
 of time.

 (TestCase 4)::
 7) Run hipMalloc() api/kernel code on same gpu parallely from parent and child
 processes, validate the results.

 (TestCase 5)::
 8) Execute hipMalloc() api simultaneously on all the gpus by spawning multiple
 child processes. Validate buffers allocated after running kernel code.

 (TestCase 6)::
 9) Multithread Scenario : Exercise hipMalloc() api parellely on all gpus from
 multiple threads and regress the api.

 (TestCases 2, 3, 4, 5, 6)::
 10) Validate memory usage with hipMemGetInfo() while regressing hipMalloc()
 api. Check for any possible memory leaks.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * TEST_NAMED: %t hipMalloc_ArgValidation  --tests 1
 * TEST_NAMED: %t hipMalloc_LoopRegression_AllocFreeCycle --tests 2
 * TEST_NAMED: %t hipMalloc_LoopRegression_AllocPool --tests 3
 * TEST_NAMED: %t hipMallocChild_Concurrency_DefaultGpu --tests 4
 * TEST_NAMED: %t hipMallocChild_Concurrency_MultiGpu --tests 5
 * TEST_NAMED: %t hipMalloc_MultiThreaded_MultiGpu --tests 6
 * HIT_END
 */

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <iostream>
#include <vector>
#include <limits>
#include <atomic>

#include "test_common.h"

/* Max alloc/free iterations for bigger chunks */
#define MAX_ALLOCFREE_BC (10000)

/* Buffer size for alloc/free cycles */
#define BUFF_SIZE_AF (5*1024*1024)

/* Max alloc/free iterations for smaller chunks */
#define MAX_ALLOCFREE_SC (5000000)

/* Max alloc and pool iterations (TBD) */
#define MAX_ALLOCPOOL_ITER (2000000)

/**
 * Validates data consitency on supplied gpu
 */
bool validateMemoryOnGPU(int gpu) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t prevAvl, prevTot, curAvl, curTot;
  bool TestPassed = true;

  HIPCHECK(hipSetDevice(gpu));
  HIPCHECK(hipMemGetInfo(&prevAvl, &prevTot));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                  0, 0, static_cast<const int*>(A_d),
                  static_cast<const int*>(B_d), C_d, N);

  HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  if (!HipTest::checkVectorADD(A_h, B_h, C_h, N)) {
    printf("Validation PASSED for gpu %d from pid %d\n", gpu, getpid());
  } else {
    printf("%s : Validation FAILED for gpu %d from pid %d\n",
        __func__, gpu, getpid());
    TestPassed &= false;
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIPCHECK(hipMemGetInfo(&curAvl, &curTot));

  if ((prevAvl != curAvl) || (prevTot != curTot)) {
    printf("%s : Memory allocation mismatch observed."
        "Possible memory leak.", __func__);
    TestPassed &= false;
  }

  return TestPassed;
}

/**
 * Fetches Gpu device count
 */
void getDeviceCount(int *pdevCnt) {
#ifdef __linux__
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
    wait(NULL);

    *pdevCnt = val;
  } else if (!childpid) {  // Child
    int devCnt = 1;
    // writing only, no need for read-descriptor
    close(fd[0]);

    HIPCHECK(hipGetDeviceCount(&devCnt));
    // send the value on the write-descriptor:
    write(fd[1], &devCnt, sizeof(devCnt));

    // close the write descriptor:
    close(fd[1]);
    exit(0);
  } else {  // failure
    *pdevCnt = 1;
    return;
  }

#else
  HIPCHECK(hipGetDeviceCount(pdevCnt));
#endif
}

/**
 * Regress memory allocation and free in loop
 */
bool regressAllocInLoop(int gpu) {
  bool TestPassed = true;
  size_t tot, avail, ptot, pavail;
  int i = 0;
  int *ptr;

  HIPCHECK(hipSetDevice(gpu));

  // Exercise allocation in loop with bigger chunks
  for (i = 0; i < MAX_ALLOCFREE_BC; i++) {
    size_t numBytes = BUFF_SIZE_AF;

    HIPCHECK(hipMemGetInfo(&pavail, &ptot));
    HIPCHECK(hipMalloc(&ptr, numBytes));
    HIPCHECK(hipMemGetInfo(&avail, &tot));

    if (pavail-avail != numBytes) {
      printf("LoopAllocation : Memory allocation of %6.2fMB"
             "not matching with hipMemGetInfo - FAIL\n",
              numBytes/(1024.0*1024.0));
      TestPassed &= false;
      HIPCHECK(hipFree(ptr));
      break;
    }

    HIPCHECK(hipFree(ptr));
  }

  // Exercise allocation in loop with smaller chunks and max iters
  HIPCHECK(hipMemGetInfo(&pavail, &ptot));

  for (i = 0; i < MAX_ALLOCFREE_SC; i++) {
    size_t numBytes = 16;

    HIPCHECK(hipMalloc(&ptr, numBytes));

    HIPCHECK(hipFree(ptr));
  }

  HIPCHECK(hipMemGetInfo(&avail, &tot));

  if ((pavail != avail) || (ptot != tot)) {
    printf("LoopAllocation : Memory allocation mismatch observed."
        "Possible memory leak.");
    TestPassed &= false;
  }

  return TestPassed;
}

/*
 * Thread func to regress alloc and check data consistency
 */

std::atomic<bool> g_thTestPassed(true);

void threadFunc(int gpu) {
  g_thTestPassed = g_thTestPassed & regressAllocInLoop(gpu);
  g_thTestPassed = g_thTestPassed & validateMemoryOnGPU(gpu);

  printf("thread execution status on gpu(%d) : %d\n", gpu, g_thTestPassed.load());
}

int main(int argc, char* argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);

  if (p_tests == 1) {  // Arg validation
    // Test hipMalloc for zero size
    bool TestPassed = true;
    int *ptr;

    HIPCHECK(hipMalloc(&ptr, 0));

    // ptr expected to be reset to null ptr
    if (ptr) {
      printf("ArgValidation : Failed in zero size test\n");
      TestPassed &= false;
    }

    // Free null ptr
    HIPCHECK(hipFree(ptr));

    // Test hipMalloc for invalid arguments
    hipError_t ret;

    if ((ret = hipMalloc(NULL, 100)) != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropritate error value returned"
          " for invalid argument. Error: '%s'(%d)\n",
          hipGetErrorString(ret), ret);
      TestPassed &= false;
    }

    // Test hipMalloc for Maximum value of size_t
    if ((ret = hipMalloc(&ptr, std::numeric_limits<std::size_t>::max()))
        != hipErrorMemoryAllocation) {
      printf("ArgValidation : Invalid error returned for max size_t."
          " Error: '%s'(%d)\n", hipGetErrorString(ret), ret);
      TestPassed &= false;
    }

    if (TestPassed) {
      passed();
    } else {
      failed("hipMalloc ArgumentValidation Failure!");
    }

  } else if (p_tests == 2) {  // Loop Regression Alloc/Free Cycle
    bool TestPassed = true;

    TestPassed &= regressAllocInLoop(0);
    TestPassed &= validateMemoryOnGPU(0);

    if (TestPassed) {
      passed();
    } else {
      failed("hipMalloc_LoopRegression_AllocFreeCycle Failure!");
    }

  } else if (p_tests == 3) {  // Loop Regression Alloc and Pool
    size_t avail, tot, pavail, ptot;
    bool TestPassed = true;
    hipError_t err;
    int *ptr;

    std::vector<int *> ptrlist;

    HIPCHECK(hipMemGetInfo(&pavail, &ptot));

    // Allocate small chunks of memory million times
    for (int i = 0; i < MAX_ALLOCPOOL_ITER; i++) {  // Iterations TBD
      if ((err = hipMalloc(&ptr, 10)) != hipSuccess) {
        HIPCHECK(hipMemGetInfo(&avail, &tot));

        printf("Loop regression pool allocation failure. "
        "Total gpu memory : %6.2fMB, Free memory %6.2fMB iter %d error '%s'\n",
        tot/(1024.0*1024.0), avail/(1024.0*1024.0), i, hipGetErrorString(err));

        TestPassed &= false;
        break;
      }

      // Store pointers allocated to emulate memory pool of app
      ptrlist.push_back(ptr);
    }

    // Free ptrs at later point of time
    for ( auto &t : ptrlist ) {
      HIPCHECK(hipFree(t));
    }

    HIPCHECK(hipMemGetInfo(&avail, &tot));

    TestPassed &= validateMemoryOnGPU(0);

    if ((pavail != avail) || (ptot != tot)) {
      printf("%s : Memory allocation mismatch observed. Possible memory leak.",
          __func__);
      TestPassed &= false;
    }

    if (TestPassed) {
      passed();
    } else {
      failed("hipMalloc_LoopRegression_AllocPool failure!");
    }

  } else if (p_tests == 4) {
    bool TestPassed = true;

#ifdef __linux__
    // Parallel execution of parent and child on gpu0
    int pid;

    if ((pid = fork()) < 0) {
      printf("Child_Concurrency_Gpu0 : fork() returned error %d.", pid);
      TestPassed &= false;

    } else if (!pid) {   // Child process
      bool TestPassedChild = true;

      TestPassedChild = validateMemoryOnGPU(0);

      if (TestPassedChild) {
        exit(0);  // child exit with success status
      } else {
        printf("Child_Concurrency_Gpu0 : childpid %d failed\n", getpid());
        exit(1);  // child exit with failure status
      }

    } else {  // Parent process
      int exitStatus;
      TestPassed = validateMemoryOnGPU(0);

      pid = wait(&exitStatus);
      if ( WEXITSTATUS(exitStatus) || ( pid < 0 ) )
        TestPassed &= false;
    }
#else
    printf("Test hipMallocChild_Concurrency_DefaultGpu skipped on non-linux\n");
#endif

    // TC scenarios specific to linux
    // are treated as pass in windows.
    if (TestPassed) {
      passed();
    } else {
      failed("hipMallocChild_Concurrency_DefaultGpu Failed!");
    }

  } else if (p_tests == 5) {
    bool TestPassed = true;
#ifdef __linux__
    // Parallel execution on multiple gpus from different child processes
    int devCnt = 1, pid = 0, cumStatus = 0;

    // Get GPU count
    getDeviceCount(&devCnt);

    // Spawn child for each GPU
    for (int gpu = 0; gpu < devCnt; gpu++) {
      if ((pid = fork()) < 0) {
         printf("Child_Concurrency_MultiGpu : fork() returned error %d\n", pid);
         failed("Test Failed!");

      } else if (!pid) {  // Child process
         bool TestPassedChild = true;
         TestPassedChild = validateMemoryOnGPU(gpu);

         if (TestPassedChild) {
            exit(0);  // child exit with success status
         } else {
            printf("Child_Concurrency_MultiGpu : childpid %d failed\n",
                getpid());
            exit(1);  // child exit with failure status
         }
      }
    }

    // Parent shall wait for child to complete
    for (int i = 0; i < devCnt; i++) {
      int pidwait = 0, exitStatus;
      pidwait = wait(&exitStatus);

      if (pidwait < 0) {
        TestPassed &= false;
        break;
      }

      cumStatus |= WEXITSTATUS(exitStatus);
    }

    // Cummulative status of all child
    if (cumStatus) {
       TestPassed &= false;
    }

#else
    printf("Test hipMallocChild_Concurrency_MultiGpu skipped on non-linux\n");
#endif


    // TC scenarios specific to linux
    // are treated as pass in windows.
    if (TestPassed) {
      passed();
    } else {
      failed("hipMallocChild_Concurrency_MultiGpu Failed!");
    }

  } else if (p_tests == 6) {  // Multithreaded multiple gpu execution
    std::vector<std::thread> threadlist;
    int devCnt = 1;

    // Get GPU count
    getDeviceCount(&devCnt);


    for (int i = 0; i < devCnt; i++) {
      threadlist.push_back(std::thread(threadFunc, i));
    }

    for (auto &t : threadlist) {
      t.join();
    }

    if (g_thTestPassed) {
      passed();
    } else {
      failed("hipMalloc_MultiThreaded_MultiGpu Failed!");
    }
  } else {
    failed("Didnt receive any valid option. Try options 1 to 6\n");
  }
}

