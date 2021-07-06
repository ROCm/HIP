#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#ifdef __linux__
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <dlfcn.h>
#endif


bool testMallocFromChild() {
  int fd[2];
  pid_t childpid;
  bool testResult = false;

  // create pipe descriptors
  pipe(fd);

  childpid = fork();
  if (childpid > 0) {  // Parent
    close(fd[1]);
    // parent will wait to read the device cnt
    read(fd[0], &testResult, sizeof(testResult));

    // close the read-descriptor
    close(fd[0]);

    // wait for child exit
    wait(NULL);

    return testResult;

  } else if (!childpid) {  // Child
    // writing only, no need for read-descriptor
    close(fd[0]);

    char* A_d = nullptr;
    hipError_t ret = hipMalloc(&A_d, 1024);

    printf("hipMalloc returned : %s\n", hipGetErrorString(ret));
    if (ret == hipSuccess)
      testResult = true;
    else
      testResult = false;

    // send the value on the write-descriptor:
    write(fd[1], &testResult, sizeof(testResult));

    // close the write descriptor:
    close(fd[1]);
    exit(0);
  }
  return false;
}


TEST_CASE("ChildMalloc") {
  auto res = testMallocFromChild();
  REQUIRE(res == true);
}
