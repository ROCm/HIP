/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include "test_common.h"

#define HIPCHECK_NO_RETURN(lastError, error)                                                      \
    {                                                                                             \
      if (lastError == hipSuccess) {                                                              \
        hipError_t localError = error;                                                            \
        if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {     \
          printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, hipGetErrorString(localError),   \
              localError, #error, __FILE__, __LINE__, KNRM);                                      \
          lastError = localError;                                                                 \
          if (shrd_mem)                                                                           \
            shrd_mem->IfTestPassed = false;                                                       \
        }                                                                                         \
      }                                                                                           \
    }

#ifdef __linux__
sem_t *sem_ob1 = NULL, *sem_ob2 = NULL;
typedef struct mem_handle {
  int device;
  hipIpcMemHandle_t memHandle;
  bool IfTestPassed;
} hip_ipc_t;

class IpcMemHandleTest {
 public:
  bool InitFlag = true;
  hip_ipc_t *shrd_mem = NULL;
  pid_t pid;
  size_t N = 1024;
  size_t Nbytes = N * sizeof(int);
  int *A_d = NULL, out = 0;
  int *A_h, *C_h;
  int Num_devices = 0, Data_mismatch, CanAccessPeer = 0;
  int *Ad1 = NULL, *Ad2 = NULL;
  IpcMemHandleTest();
  bool Test();
  ~IpcMemHandleTest();
};


bool IpcMemHandleTest::Test() {
  if (InitFlag == false) {
    // Abort the test if the initialization fails
    printf("Resource initialization failed. Hence test skipped!");
    return false;
  }
  hipError_t status = hipSuccess;

  pid = fork();
  if (pid != 0) {
    // Parent process
    HIPCHECK_NO_RETURN(status, hipGetDeviceCount(&Num_devices));
    for (int i = 0; i < Num_devices; ++i) {
      if (shrd_mem->IfTestPassed == true) {
        HIPCHECK_NO_RETURN(status, hipSetDevice(i));
        HIPCHECK_NO_RETURN(status, hipMalloc(&A_d, Nbytes));
        HIPCHECK_NO_RETURN(status, hipIpcGetMemHandle((hipIpcMemHandle_t *) &shrd_mem->memHandle,
                                    A_d));
        HIPCHECK_NO_RETURN(status, hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
        shrd_mem->device = i;
        if ((out=sem_post(sem_ob1)) == -1) {
          // Need to use inline function to release resources.
          shrd_mem->IfTestPassed = false;
          failed("sem_post() call failed in parent process.");
        }
        if ((out=sem_wait(sem_ob2)) == -1) {
          shrd_mem->IfTestPassed = false;
          failed("sem_wait() call failed in parent process.");
        }
        HIPCHECK_NO_RETURN(status, hipFree(A_d));
      }
    }
  } else {
    // Child process
    HIPCHECK_NO_RETURN(status, hipGetDeviceCount(&Num_devices));
    for (int j = 0; j < Num_devices; ++j) {
      if ((out=sem_wait(sem_ob1)) == -1) {
        shrd_mem->IfTestPassed = false;
        printf("sem_wait() call failed in child process.");
        if ((out=sem_post(sem_ob2)) == -1) {
          printf("sem_post() call on sem_ob2 failed");
          exit(1);
        }
      }
      for (int i = 0; i < Num_devices; ++i) {
        Data_mismatch = 0;
        HIPCHECK_NO_RETURN(status, hipSetDevice(i));
        HIPCHECK_NO_RETURN(status, hipMalloc(&Ad2, Nbytes));
        HIPCHECK_NO_RETURN(status, hipIpcOpenMemHandle((void **) &Ad1, shrd_mem->memHandle,
                                     hipIpcMemLazyEnablePeerAccess));
        HIPCHECK_NO_RETURN(status, hipDeviceCanAccessPeer(&CanAccessPeer, i, shrd_mem->device));
        if (CanAccessPeer == 1) {
          HIPCHECK_NO_RETURN(status, hipMemcpy(Ad2, Ad1, Nbytes, hipMemcpyDeviceToDevice));
          HIPCHECK_NO_RETURN(status, hipMemcpy(C_h, Ad2, Nbytes, hipMemcpyDeviceToHost));
          for (int i = 0; i < N; ++i) {
            if (C_h[i] != 123)
              Data_mismatch++;
          }
          if (Data_mismatch != 0) {
            printf("Data mismatch found when data copied from Ipc memhandle");
            printf(" to Device: %d\n", i);
            shrd_mem->IfTestPassed = false;
          }
          memset(reinterpret_cast<void*>(C_h), 0, Nbytes);
          // Checking if the data obtained from Ipc shared memory is consistent
          HIPCHECK_NO_RETURN(status, hipMemcpy(C_h, Ad1, Nbytes, hipMemcpyDeviceToHost));
          for (int i = 0; i < N; ++i) {
            if (C_h[i] != 123)
              Data_mismatch++;
          }
          if (Data_mismatch != 0) {
            printf("Data mismatch found when data copied from Ipc memhandle");
            printf(" Host.\n");
            shrd_mem->IfTestPassed = false;
          }
        }
        HIPCHECK_NO_RETURN(status, hipIpcCloseMemHandle(reinterpret_cast<void*>(Ad1)));
      }
    HIPCHECK_NO_RETURN(status, hipFree(Ad2));
    if ((out=sem_post(sem_ob2)) == -1) {
      shrd_mem->IfTestPassed = false;
      printf("sem_post() call on sem_ob2 failed");
      exit(1);
    }
  }
  exit(0);
  }

  if ((out = sem_unlink("/my-sem-object1")) == -1) {
    printf("sem_unlink() call on /my-sem-object1 failed");
  }
  if ((out = sem_unlink("/my-sem-object2")) == -1) {
    printf("sem_unlink() call on /my-sem-object2 failed");
  }
  int rFlag = 0; // return flag
  waitpid(pid, &rFlag, 0);
  if (shrd_mem->IfTestPassed == false) {
    return false;
  } else {
    return true;
  }
}

IpcMemHandleTest::IpcMemHandleTest() {
  std::string cmd_line = "rm -rf /dev/shm/sem.my-sem-object*";
  int res = system(cmd_line.c_str());
  if (res == -1) {
    InitFlag = false;
    printf("System call to remove existing shared objects failed!");
  }
  int out;
  if ((sem_ob1 = sem_open ("/my-sem-object1", O_CREAT|O_EXCL, 0660, 0)) ==
      SEM_FAILED) {
    InitFlag = false;
    printf("Initialization of 1st semaphore object failed");
  }
  if ((sem_ob2 = sem_open ("/my-sem-object2", O_CREAT|O_EXCL, 0660, 0)) ==
      SEM_FAILED) {
    InitFlag = false;
    printf("Initialization of 2nd semaphore object failed");
  }

  shrd_mem = reinterpret_cast<hip_ipc_t *>(mmap(NULL, sizeof(hip_ipc_t),
                                                PROT_READ | PROT_WRITE,
                                                MAP_SHARED | MAP_ANONYMOUS,
                                                0, 0));
  if (shrd_mem == NULL) {
    InitFlag = false;
    printf("mmap() call failed!");
  }
  shrd_mem->IfTestPassed = true;
  A_h = reinterpret_cast<int*>(malloc(Nbytes));
  C_h = reinterpret_cast<int*>(malloc(Nbytes));
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 123;
  }
}

IpcMemHandleTest::~IpcMemHandleTest() {
  munmap(shrd_mem, sizeof(hip_ipc_t));
  HIPCHECK(hipFree((A_d)));
  free(A_h);
  free(C_h);
  HIPCHECK(hipFree((Ad1)));
  HIPCHECK(hipFree((Ad2)));
}
#endif

int main() {
  bool IfTestPassed = true;
  // The following program spawns a child process and does the following
  // Parent iterate through each device, create memory -- create hipIpcMemhandle
  // stores the mem handle in mmaped memory, release the child using sem_post()
  // and wait for child to release itself(parent process)
  // child process:
  // Child process get the ipc mem handle using hipIpcOpenMemHandle
  // Iterate through all the available gpus and do Device to Device copies
  // and check for data consistencies and close the hipIpcCloseMemHandle
  // release the parent and wait for parent to release itself(child)
#ifdef __linux__
  IpcMemHandleTest obj;
  IfTestPassed = obj.Test();
#else
  printf("This is not a Linux platform. Hence Skipping the test!\n");
  IfTestPassed = true;
#endif
  if (IfTestPassed == false) {
    failed("");
  }
  passed();
}
