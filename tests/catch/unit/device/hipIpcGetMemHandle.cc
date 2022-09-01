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

#ifdef __linux__

#include <cstring>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

TEST_CASE("Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations") {
  void *ptr1, *ptr2;
  hipIpcMemHandle_t handle1, handle2;
  HIP_CHECK(hipMalloc(&ptr1, 1024));
  HIP_CHECK(hipMalloc(&ptr2, 1024));
  HIP_CHECK(hipIpcGetMemHandle(&handle1, ptr1));
  HIP_CHECK(hipIpcGetMemHandle(&handle2, ptr2));

  CHECK(memcmp(&handle1, &handle2, sizeof(handle1)) != 0);

  HIP_CHECK(hipFree(ptr1));
  HIP_CHECK(hipFree(ptr2));
}

TEST_CASE("Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Reused_Memory") {
  void *ptr1 = nullptr, *ptr2 = nullptr;
  hipIpcMemHandle_t handle1, handle2;
  HIP_CHECK(hipMalloc(&ptr1, 1024));
  HIP_CHECK(hipIpcGetMemHandle(&handle1, ptr1));
  HIP_CHECK(hipFree(ptr1));

  HIP_CHECK(hipMalloc(&ptr2, 1024));
  HIP_CHECK(hipIpcGetMemHandle(&handle2, ptr2));

  if (ptr1 == ptr2) CHECK(memcmp(&handle1, &handle2, sizeof(handle1)) != 0);

  HIP_CHECK(hipFree(ptr2));
}

TEST_CASE("Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory") {
  void* ptr;
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipMalloc(&ptr, 1024));
  HIP_CHECK(hipFree(ptr));
  HIP_CHECK_ERROR(hipIpcGetMemHandle(&handle, ptr), hipErrorInvalidDevicePointer);
}

TEST_CASE("Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer") {
  int* ptr;
  constexpr size_t n = 1024;
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(*ptr)));
  HIP_CHECK_ERROR(hipIpcGetMemHandle(&handle, reinterpret_cast<void*>(ptr + n)),
                  hipErrorInvalidDevicePointer);
  HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr)));
}

#endif