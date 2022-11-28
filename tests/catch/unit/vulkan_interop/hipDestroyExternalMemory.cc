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

#include "vulkan_test.hh"

constexpr bool enable_validation = false;

TEST_CASE("Unit_hipDestroyExternalMemory_Vulkan_Negative_Parameters") {
  SECTION("extMem == nullptr") {
    HIP_CHECK_ERROR(hipDestroyExternalMemory(nullptr), hipErrorInvalidValue);
  }

// Segfaults in CUDA
// Disabled on AMD due to defect - EXSWHTEC-187
#if HT_AMD && 0
  SECTION("Double free") {
    VulkanTest vkt(enable_validation);
    const auto storage = vkt.CreateMappedStorage<int>(1, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    auto desc = vkt.BuildMemoryDescriptor(storage.memory, sizeof(*storage.host_ptr));
    hipExternalMemory_t ext_memory;
    HIP_CHECK(hipImportExternalMemory(&ext_memory, &desc));

    HIP_CHECK(hipDestroyExternalMemory(ext_memory));
    HIP_CHECK_ERROR(hipDestroyExternalMemory(ext_memory), hipErrorInvalidValue);
  }
#endif
}