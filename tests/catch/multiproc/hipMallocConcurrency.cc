#include <sys/types.h>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#ifdef __linux__
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <iostream>
#include <vector>
#include <limits>
#include <atomic>

#include <hip_test_common.hh>


static constexpr size_t N = 4 * 1024 * 1024;
static constexpr unsigned blocksPerCU = 6;  // to hide latency
static constexpr unsigned threadsPerBlock = 256;
/**
 * Validates data consitency on supplied gpu
 */
bool validateMemoryOnGPU(int gpu, bool concurOnOneGPU = false) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t prevAvl, prevTot, curAvl, curTot;
  bool TestPassed = true;

  HIP_CHECK(hipSetDevice(gpu));
  HIP_CHECK(hipMemGetInfo(&prevAvl, &prevTot));
  printf("tgs allocating..\n");
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                     static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d, N);

  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  if (!HipTest::checkVectorADD(A_h, B_h, C_h, N)) {
    printf("Validation PASSED for gpu %d from pid %d\n", gpu, getpid());
  } else {
    printf("%s : Validation FAILED for gpu %d from pid %d\n", __func__, gpu, getpid());
    TestPassed &= false;
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipMemGetInfo(&curAvl, &curTot));

  if (!concurOnOneGPU && (prevAvl != curAvl || prevTot != curTot)) {
    // In concurrent calls on one GPU, we cannot verify leaking in this way
    printf(
        "%s : Memory allocation mismatch observed."
        "Possible memory leak.\n",
        __func__);
    TestPassed &= false;
  }

  return TestPassed;
}


#if 1
/**
 * Fetches Gpu device count
 */
void getDeviceCount1(int* pdevCnt) {
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

    HIP_CHECK(hipGetDeviceCount(&devCnt));
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
  HIP_CHECK(hipGetDeviceCount(pdevCnt));
#endif
}
#endif


TEST_CASE("hipMallocChild_Concurrency_MultiGpu") {
  bool TestPassed = false;
#ifdef __linux__
  // Parallel execution on multiple gpus from different child processes
  int devCnt = 1, pid = 0;

  // Get GPU count
  getDeviceCount1(&devCnt);

  // Spawn child for each GPU
  for (int gpu = 0; gpu < devCnt; gpu++) {
    if ((pid = fork()) < 0) {
      INFO("Child_Concurrency_MultiGpu : fork() returned error" << pid);
      REQUIRE(false);

    } else if (!pid) {  // Child process
      bool TestPassedChild = false;
      TestPassedChild = validateMemoryOnGPU(gpu);

      if (TestPassedChild) {
        printf("returning exit(1) for success\n");
        exit(1);  // child exit with success status
      } else {
        printf("Child_Concurrency_MultiGpu : childpid %d failed\n", getpid());
        exit(2);  // child exit with failure status
      }
    }
  }

  // Parent shall wait for child to complete
  int cnt = 0;

  for (int i = 0; i < devCnt; i++) {
    int pidwait = 0, exitStatus;
    pidwait = wait(&exitStatus);

    printf("exitStatus for iter:%d is %d\n", i, exitStatus);
    if (pidwait < 0) {
      break;
    }

    if (WEXITSTATUS(exitStatus) == 1) cnt++;
  }

  if (cnt && (cnt == devCnt)) TestPassed = true;

#else
  INFO("Test hipMallocChild_Concurrency_MultiGpu skipped on non-linux");
#endif
  REQUIRE(TestPassed == true);
}
