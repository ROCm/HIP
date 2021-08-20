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
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
This testcase verifies the hipIpcMemAccess APIs by creating memory handle
in parent process and access it in child process.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#ifdef __linux__
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>

typedef struct mem_handle {
  int device;
  hipIpcMemHandle_t memHandle;
  bool IfTestPassed;
} hip_ipc_t;


// This testcase verifies the hipIpcMemAccess APIs as follows
// The following program spawns a child process and does the following
// Parent iterate through each device, create memory -- create hipIpcMemhandle
// stores the mem handle in mmaped memory, release the child using sem_post()
// and wait for child to release itself(parent process)
// child process:
// Child process get the ipc mem handle using hipIpcOpenMemHandle
// Iterate through all the available gpus and do Device to Device copies
// and check for data consistencies and close the hipIpcCloseMemHandle
// release the parent and wait for parent to release itself(child)

TEST_CASE("Unit_hipIpcMemAccess_Semaphores") {
  hip_ipc_t *shrd_mem = NULL;
  pid_t pid;
  size_t N = 1024;
  size_t Nbytes = N * sizeof(int);
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  sem_t *sem_ob1{nullptr}, *sem_ob2{nullptr};
  int Num_devices = 0, CanAccessPeer = 0;

  std::string cmd_line = "rm -rf /dev/shm/sem.my-sem-object*";
  int res = system(cmd_line.c_str());
  REQUIRE(res != -1);
  sem_ob1 = sem_open("/my-sem-object1", O_CREAT|O_EXCL, 0660, 0);
  sem_ob2 = sem_open("/my-sem-object2", O_CREAT|O_EXCL, 0660, 0);
  REQUIRE(sem_ob1 != SEM_FAILED);
  REQUIRE(sem_ob2 != SEM_FAILED);

  shrd_mem = reinterpret_cast<hip_ipc_t *>(mmap(NULL, sizeof(hip_ipc_t),
                                           PROT_READ | PROT_WRITE,
                                           MAP_SHARED | MAP_ANONYMOUS,
                                           0, 0));
  REQUIRE(shrd_mem != NULL);
  shrd_mem->IfTestPassed = true;
  HipTest::initArrays<int>(nullptr, nullptr, nullptr,
                           &A_h, nullptr, &C_h, N, false);
  pid = fork();
  if (pid != 0) {
    // Parent process
    HIP_CHECK(hipGetDeviceCount(&Num_devices));
    for (int i = 0; i < Num_devices; ++i) {
      if (shrd_mem->IfTestPassed == true) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(&A_d, Nbytes));
        HIP_CHECK(hipIpcGetMemHandle(reinterpret_cast<hipIpcMemHandle_t *>
                                     (&shrd_mem->memHandle),
                                     A_d));
        HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
        shrd_mem->device = i;
        if ((sem_post(sem_ob1)) == -1) {
          // Need to use inline function to release resources.
          shrd_mem->IfTestPassed = false;
          WARN("sem_post() call failed in parent process.");
        }
        if ((sem_wait(sem_ob2)) == -1) {
          shrd_mem->IfTestPassed = false;
          WARN("sem_wait() call failed in parent process.");
        }
        HIP_CHECK(hipFree(A_d));
      }
    }
  } else {
    // Child process
    HIP_CHECK(hipGetDeviceCount(&Num_devices));
    for (int j = 0; j < Num_devices; ++j) {
        HIP_CHECK(hipSetDevice(j));
      if ((sem_wait(sem_ob1)) == -1) {
        shrd_mem->IfTestPassed = false;
        WARN("sem_wait() call failed in child process.");
        if ((sem_post(sem_ob2)) == -1) {
          shrd_mem->IfTestPassed = false;
          WARN("sem_post() call on sem_ob2 failed");
          exit(1);
        }
      }
      for (int i = 0; i < Num_devices; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(&C_d, Nbytes));
        HIP_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void **>(&B_d),
                                      shrd_mem->memHandle,
                                      hipIpcMemLazyEnablePeerAccess));
        HIP_CHECK(hipDeviceCanAccessPeer(&CanAccessPeer, i, shrd_mem->device));
        if (CanAccessPeer == 1) {
          HIP_CHECK(hipMemcpy(C_d, B_d, Nbytes, hipMemcpyDeviceToDevice));
          HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
          HipTest::checkTest<int>(A_h, C_h, N);
          memset(reinterpret_cast<void*>(C_h), 0, Nbytes);
          // Checking if the data obtained from Ipc shared memory is consistent
          HIP_CHECK(hipMemcpy(C_h, B_d, Nbytes, hipMemcpyDeviceToHost));
          HipTest::checkTest<int>(A_h, C_h, N);
        }
        HIP_CHECK(hipIpcCloseMemHandle(reinterpret_cast<void*>(B_d)));
        HIP_CHECK(hipFree(C_d));
      }
      if ((sem_post(sem_ob2)) == -1) {
        shrd_mem->IfTestPassed = false;
        WARN("sem_post() call on sem_ob2 failed");
        exit(1);
      }
    }
    exit(0);
  }
  if ((sem_unlink("/my-sem-object1")) == -1) {
    WARN("sem_unlink() call on /my-sem-object1 failed");
  }
  if ((sem_unlink("/my-sem-object2")) == -1) {
    WARN("sem_unlink() call on /my-sem-object2 failed");
  }
  int rFlag = 0;
  waitpid(pid, &rFlag, 0);
  REQUIRE(shrd_mem->IfTestPassed == true);
}
#endif
