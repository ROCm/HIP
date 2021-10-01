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

 1) Run hipMalloc() api/kernel code on same gpu parallely from parent and child
 processes, validate the results.

 2) Execute hipMalloc() api simultaneously on all the gpus by spawning multiple
 child processes. Validate buffers allocated after running kernel code.

*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#ifdef __linux__
#include <sys/wait.h>
#include <sys/types.h>
#include <unistd.h>

/**
 * Fetches Gpu device count
 */
static void getDeviceCount(int* pdevCnt) {
  int fd[2], val = 0;
  pid_t childpid;

  // create pipe descriptors
  pipe(fd);

  // disable visible_devices env from shell
#ifdef HT_NVIDIA
  unsetenv("CUDA_VISIBLE_DEVICES");
#else
  unsetenv("ROCR_VISIBLE_DEVICES");
  unsetenv("HIP_VISIBLE_DEVICES");
#endif

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
 * Validates data consistency on supplied gpu
 */
static bool validateMemoryOnGPU(int gpu, bool concurOnOneGPU = false) {
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t prevAvl, prevTot, curAvl, curTot;
  bool TestPassed = true;
  constexpr auto N = 4 * 1024 * 1024;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  size_t Nbytes = N * sizeof(int);

  HIP_CHECK(hipSetDevice(gpu));
  HIP_CHECK(hipMemGetInfo(&prevAvl, &prevTot));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                     0, 0, static_cast<const int*>(A_d),
                     static_cast<const int*>(B_d), C_d, N);

  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  if (!HipTest::checkVectorADD(A_h, B_h, C_h, N)) {
    printf("Validation PASSED for gpu %d from pid %d\n", gpu, getpid());
  } else {
    printf("Validation FAILED for gpu %d from pid %d\n", gpu, getpid());
    TestPassed = false;
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipMemGetInfo(&curAvl, &curTot));

  if (!concurOnOneGPU && (prevAvl != curAvl || prevTot != curTot)) {
    // In concurrent calls on one GPU, we cannot verify leaking in this way
    printf(
        "%s : Memory allocation mismatch observed."
        "Possible memory leak.\n",
        __func__);
    TestPassed = false;
  }

  return TestPassed;
}

/**
 * Parallel execution of parent and child on gpu0
 */
TEST_CASE("Unit_hipMalloc_ChildConcurrencyDefaultGpu") {
  int devCnt = 0, pid = 0;
  constexpr auto resSuccess = 1, resFailure = 2;
  bool TestPassed = true;

  // Get GPU count
  getDeviceCount(&devCnt);
  REQUIRE(devCnt > 0);

  if ((pid = fork()) < 0) {
    INFO("Child_Concurrency_DefaultGpu : fork() returned error : " << pid);
    HIP_ASSERT(false);

  } else if (!pid) {  // Child process
    bool TestPassedChild = false;

    // Allocates and validates memory on Gpu0 simultaneously with parent
    TestPassedChild = validateMemoryOnGPU(0, true);

    if (TestPassedChild) {
      exit(resSuccess);  // child exit with success status
    } else {
      exit(resFailure);  // child exit with failure status
    }

  } else {  // Parent process
    int exitStatus;

    // Allocates and validates memory on Gpu0 simultaneously with child
    TestPassed = validateMemoryOnGPU(0, true);

    // Wait and get result from child
    pid = wait(&exitStatus);
    if ((WEXITSTATUS(exitStatus) ==  resFailure) || (pid < 0))
      TestPassed = false;
  }

  REQUIRE(TestPassed == true);
}

/**
 * Parallel execution of api on multiple gpus from
 * different child processes.
 */
TEST_CASE("Unit_hipMalloc_ChildConcurrencyMultiGpu") {
  int devCnt = 0, pid = 0;
  constexpr auto resSuccess = 1, resFailure = 2;

  // Get GPU count
  getDeviceCount(&devCnt);
  REQUIRE(devCnt > 0);

  // Spawn child for each GPU
  for (int gpu = 0; gpu < devCnt; gpu++) {
    if ((pid = fork()) < 0) {
      INFO("Child_Concurrency_MultiGpu : fork() returned error : " << pid);
      REQUIRE(false);

    } else if (!pid) {  // Child process
      bool TestPassedChild = false;
      TestPassedChild = validateMemoryOnGPU(gpu);

      if (TestPassedChild) {
        exit(resSuccess);  // child exit with success status
      } else {
        exit(resFailure);  // child exit with failure status
      }
    }
  }

  // Parent shall wait for child to complete
  int passCnt = 0;
  for (int i = 0; i < devCnt; i++) {
    int pidwait = 0, exitStatus;
    pidwait = wait(&exitStatus);

    printf("exitStatus for dev:%d is %d\n", i, WEXITSTATUS(exitStatus));
    if (pidwait < 0) {
      break;
    }

    if (WEXITSTATUS(exitStatus) == resSuccess) passCnt++;
  }
  REQUIRE(passCnt == devCnt);
}
#endif  // __linux__
