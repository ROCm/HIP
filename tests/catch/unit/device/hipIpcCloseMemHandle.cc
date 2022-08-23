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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

TEST_CASE("Unit_hipIpcCloseMemHandle_Positive_Reference_Counting") {
  int fd[2];
  REQUIRE(pipe(fd) == 0);

  // The fork must be performed before the runtime is initialized(so before any API that implicitly
  // initializes it). The pipe in conjunction with wait is then used to impose total ordering
  // between parent and child process. Because total ordering is imposed regular CATCH assertions
  // should be safe to use
  auto pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);

    hipIpcMemHandle_t handle;
    REQUIRE(read(fd[0], &handle, sizeof(handle)) >= 0);
    REQUIRE(close(fd[0]) == 0);

    void *child_ptr1, *child_ptr2;
    HIP_CHECK(hipIpcOpenMemHandle(&child_ptr1, handle, hipIpcMemLazyEnablePeerAccess));
    HIP_CHECK(hipIpcOpenMemHandle(&child_ptr2, handle, hipIpcMemLazyEnablePeerAccess));

    HIP_CHECK(hipIpcCloseMemHandle(child_ptr1));
    hipPointerAttribute_t attributes;
    HIP_CHECK(hipPointerGetAttributes(&attributes, child_ptr1));
    HIP_CHECK(hipPointerGetAttributes(&attributes, child_ptr2));

    HIP_CHECK(hipIpcCloseMemHandle(child_ptr2));
    HIP_CHECK_ERROR(hipPointerGetAttributes(&attributes, child_ptr1), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipPointerGetAttributes(&attributes, child_ptr2), hipErrorInvalidValue);

    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);

    void* ptr;
    hipIpcMemHandle_t handle;
    HIP_CHECK(hipMalloc(&ptr, 1024));
    HIP_CHECK(hipIpcGetMemHandle(&handle, ptr));

    REQUIRE(write(fd[1], &handle, sizeof(handle)) >= 0);
    REQUIRE(close(fd[1]) == 0);

    REQUIRE(wait(NULL) >= 0);

    hipPointerAttribute_t attributes;
    HIP_CHECK(hipPointerGetAttributes(&attributes, ptr));

    HIP_CHECK(hipFree(ptr));
  }
}

TEST_CASE("Unit_hipIpcCloseMemHandle_Negative_Close_In_Originating_Process") {
  void* ptr;
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipMalloc(&ptr, 1024));
  HIP_CHECK(hipIpcGetMemHandle(&handle, ptr));

  HIP_CHECK_ERROR(hipIpcCloseMemHandle(ptr), hipErrorInvalidValue);
  HIP_CHECK(hipFree(ptr));
}