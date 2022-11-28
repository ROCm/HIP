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

template <typename T> __global__ void Set(T* ptr, const T val) { ptr[threadIdx.x] = val; }

TEST_CASE("Unit_hipExternalMemoryGetMappedBuffer_Vulkan_Positive_Read_Write") {
  VulkanTest vkt(enable_validation);
  using type = uint8_t;
  constexpr uint32_t count = 3;

  const auto vk_storage =
      vkt.CreateMappedStorage<type>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);

  const auto hip_ext_mem_desc = vkt.BuildMemoryDescriptor(vk_storage.memory, vk_storage.size);
  hipExternalMemory_t hip_ext_memory;
  HIP_CHECK(hipImportExternalMemory(&hip_ext_memory, &hip_ext_mem_desc));

  hipExternalMemoryBufferDesc external_mem_buffer_desc = {};
  external_mem_buffer_desc.size = vk_storage.size;

  type* hip_dev_ptr = nullptr;
  HIP_CHECK(hipExternalMemoryGetMappedBuffer(reinterpret_cast<void**>(&hip_dev_ptr), hip_ext_memory,
                                             &external_mem_buffer_desc));
  REQUIRE(nullptr != hip_dev_ptr);

  vk_storage.host_ptr[0] = 41;
  vk_storage.host_ptr[1] = 40;
  vk_storage.host_ptr[2] = 43;

  std::vector<type> read_buffer(count, 0);
  HIP_CHECK(
      hipMemcpy(read_buffer.data(), hip_dev_ptr, count * sizeof(type), hipMemcpyDeviceToHost));
  REQUIRE(41 == read_buffer[0]);
  REQUIRE(40 == read_buffer[1]);
  REQUIRE(43 == read_buffer[2]);

  Set<<<1, 1>>>(hip_dev_ptr + 1, static_cast<type>(42));
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(41 == vk_storage.host_ptr[0]);
  REQUIRE(42 == vk_storage.host_ptr[1]);
  REQUIRE(43 == vk_storage.host_ptr[2]);

  // Defect - EXSWHTEC-181
  // HIP_CHECK(hipFree(hip_dev_ptr));
  HIP_CHECK(hipDestroyExternalMemory(hip_ext_memory));
}

// Disabled on AMD due to defect - EXSWHTEC-175
#if HT_NVIDIA
TEST_CASE("Unit_hipExternalMemoryGetMappedBuffer_Vulkan_Positive_Read_Write_With_Offset") {
  VulkanTest vkt(enable_validation);
  using type = uint8_t;
  constexpr uint32_t count = 2;

  const auto vk_storage =
      vkt.CreateMappedStorage<type>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);

  const auto hip_ext_mem_desc = vkt.BuildMemoryDescriptor(vk_storage.memory, vk_storage.size);
  hipExternalMemory_t hip_ext_memory;
  HIP_CHECK(hipImportExternalMemory(&hip_ext_memory, &hip_ext_mem_desc));

  hipExternalMemoryBufferDesc external_mem_buffer_desc = {};
  constexpr auto offset = (count - 1) * sizeof(type);
  external_mem_buffer_desc.size = vk_storage.size - offset;
  external_mem_buffer_desc.offset = offset;

  type* hip_dev_ptr = nullptr;
  HIP_CHECK(hipExternalMemoryGetMappedBuffer(reinterpret_cast<void**>(&hip_dev_ptr), hip_ext_memory,
                                             &external_mem_buffer_desc));

  vk_storage.host_ptr[0] = 41;
  vk_storage.host_ptr[1] = 42;
  type read_val = 0;
  HIP_CHECK(hipMemcpy(&read_val, hip_dev_ptr, 1, hipMemcpyDeviceToHost));
  REQUIRE(42 == read_val);

  // Defect - EXSWHTEC-181
  // HIP_CHECK(hipFree(hip_dev_ptr));
  HIP_CHECK(hipDestroyExternalMemory(hip_ext_memory));
}
#endif

TEST_CASE("Unit_hipExternalMemoryGetMappedBuffer_Vulkan_Negative_Parameters") {
  VulkanTest vkt(enable_validation);
  const auto vk_storage = vkt.CreateMappedStorage<int>(1, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);

  const auto hip_ext_mem_desc = vkt.BuildMemoryDescriptor(vk_storage.memory, vk_storage.size);
  hipExternalMemory_t hip_ext_memory;
  HIP_CHECK(hipImportExternalMemory(&hip_ext_memory, &hip_ext_mem_desc));

  hipExternalMemoryBufferDesc external_mem_buffer_desc = {};
  external_mem_buffer_desc.size = vk_storage.size;
  void* hip_dev_ptr = nullptr;

// Disabled on AMD due to defect - EXSWHTEC-176
#if HT_NVIDIA
  SECTION("devPtr == nullptr") {
    HIP_CHECK_ERROR(
        hipExternalMemoryGetMappedBuffer(nullptr, hip_ext_memory, &external_mem_buffer_desc),
        hipErrorInvalidValue);
  }
#endif

// Disabled on AMD due to defect - EXSWHTEC-177
#if HT_NVIDIA
  SECTION("bufferDesc == nullptr") {
    HIP_CHECK_ERROR(hipExternalMemoryGetMappedBuffer(&hip_dev_ptr, hip_ext_memory, nullptr),
                    hipErrorInvalidValue);
  }
#endif

// Disabled on AMD due to defect - EXSWHTEC-179
#if HT_NVIDIA
  SECTION("bufferDesc.flags != 0") {
    external_mem_buffer_desc.flags = 1;
    HIP_CHECK_ERROR(
        hipExternalMemoryGetMappedBuffer(&hip_dev_ptr, hip_ext_memory, &external_mem_buffer_desc),
        hipErrorInvalidValue);
  }
#endif

// Disabled on AMD due to defect - EXSWHTEC-180
#if HT_NVIDIA
  SECTION("bufferDesc.offset + bufferDesc.size > hipExternalMemHandleDesc.size") {
    external_mem_buffer_desc.offset = 1;
    HIP_CHECK_ERROR(
        hipExternalMemoryGetMappedBuffer(&hip_dev_ptr, hip_ext_memory, &external_mem_buffer_desc),
        hipErrorInvalidValue);
  }
#endif
}