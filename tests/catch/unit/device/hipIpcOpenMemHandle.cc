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

#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

TEST_CASE("Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process") {
  hipDeviceptr_t ptr1, ptr2;
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr1), 1024));
  HIP_CHECK(hipIpcGetMemHandle(&handle, reinterpret_cast<void*>(ptr1)));
  HIP_CHECK_ERROR(
      hipIpcOpenMemHandle(reinterpret_cast<void**>(&ptr2), handle, hipIpcMemLazyEnablePeerAccess),
      hipErrorInvalidContext);
  HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr1)));
}

TEST_CASE("Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device") {
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

    hipDeviceptr_t ptr_child;
    HIP_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void**>(&ptr_child), handle,
                                  hipIpcMemLazyEnablePeerAccess));

    HIP_CHECK(hipInit(0));
    hipCtx_t ctx;
    HIP_CHECK(hipCtxCreate(&ctx, 0, 0));

    hipDeviceptr_t ptr_child_ctx;
    HIP_CHECK_ERROR(hipIpcOpenMemHandle(reinterpret_cast<void**>(&ptr_child_ctx), handle,
                                        hipIpcMemLazyEnablePeerAccess),
                    hipErrorInvalidResourceHandle);

    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);

    hipDeviceptr_t ptr;
    hipIpcMemHandle_t handle;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), 1024));
    HIP_CHECK(hipIpcGetMemHandle(&handle, reinterpret_cast<void*>(ptr)));

    REQUIRE(write(fd[1], &handle, sizeof(handle)) >= 0);
    REQUIRE(close(fd[1]) == 0);

    REQUIRE(wait(NULL) >= 0);

    HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr)));
  }
}