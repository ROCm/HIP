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

TEST_CASE("Unit_hipSignalExternalSemaphoresAsync_Vulkan_Positive_Binary_Semaphore") {
  VulkanTest vkt(enable_validation);

  constexpr uint32_t count = 1;
  const auto src_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  const auto dst_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  const auto command_buffer = vkt.GetCommandBuffer();
  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK_RESULT(vkBeginCommandBuffer(command_buffer, &begin_info));
  VkBufferCopy buffer_copy = {};
  buffer_copy.size = count * sizeof(*src_storage.host_ptr);
  vkCmdCopyBuffer(command_buffer, src_storage.buffer, dst_storage.buffer, 1, &buffer_copy);
  VK_CHECK_RESULT(vkEndCommandBuffer(command_buffer));

  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  const auto hip_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_BINARY);
  hipExternalSemaphore_t hip_ext_semaphore;
  HIP_CHECK(hipImportExternalSemaphore(&hip_ext_semaphore, &hip_sem_handle_desc));

  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = &semaphore;
  const auto fence = vkt.CreateFence();
  VK_CHECK_RESULT(vkQueueSubmit(vkt.GetQueue(), 1, &submit_info, fence));

  REQUIRE(vkGetFenceStatus(vkt.GetDevice(), fence) == VK_NOT_READY);

  hipExternalSemaphoreSignalParams signal_params = {};
  signal_params.params.fence.value = 0;
  HIP_CHECK(hipSignalExternalSemaphoresAsync(&hip_ext_semaphore, &signal_params, 1, nullptr));
  PollStream(nullptr, hipSuccess);

  VK_CHECK_RESULT(
      vkWaitForFences(vkt.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000 /*5 seconds*/));

  HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
}

// Timeline semaphores unsupported on AMD
#if HT_NVIDIA
TEST_CASE("Unit_hipSignalExternalSemaphoresAsync_Vulkan_Positive_Timeline_Semaphore") {
  VulkanTest vkt(enable_validation);
  constexpr uint64_t signal_value = 2;

  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_TIMELINE);
  const auto hip_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_TIMELINE);
  hipExternalSemaphore_t hip_ext_semaphore;
  HIP_CHECK(hipImportExternalSemaphore(&hip_ext_semaphore, &hip_sem_handle_desc));

  hipExternalSemaphoreSignalParams signal_params = {};
  signal_params.params.fence.value = signal_value;

  HIP_CHECK(hipSignalExternalSemaphoresAsync(&hip_ext_semaphore, &signal_params, 1, nullptr));
  PollStream(nullptr, hipSuccess);

  uint64_t sem_value = 0u;
  VK_CHECK_RESULT(vkGetSemaphoreCounterValue(vkt.GetDevice(), semaphore, &sem_value));

  REQUIRE(2 == sem_value);

  HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
}
#endif

TEST_CASE("Unit_hipSignalExternalSemaphoresAsync_Vulkan_Positive_Multiple_Semaphores") {
  VulkanTest vkt(enable_validation);

  constexpr uint32_t count = 1;
  const auto src_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  const auto dst_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

#if HT_AMD
  constexpr auto second_semaphore_type = VK_SEMAPHORE_TYPE_BINARY;
#else
  constexpr auto second_semaphore_type = VK_SEMAPHORE_TYPE_TIMELINE;
#endif

  const auto command_buffer = vkt.GetCommandBuffer();
  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK_RESULT(vkBeginCommandBuffer(command_buffer, &begin_info));
  VkBufferCopy buffer_copy = {};
  buffer_copy.size = count * sizeof(*src_storage.host_ptr);
  vkCmdCopyBuffer(command_buffer, src_storage.buffer, dst_storage.buffer, 1, &buffer_copy);
  VK_CHECK_RESULT(vkEndCommandBuffer(command_buffer));

  const auto binary_semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  const auto hip_binary_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(binary_semaphore, VK_SEMAPHORE_TYPE_BINARY);
  hipExternalSemaphore_t hip_binary_ext_semaphore;
  HIP_CHECK(hipImportExternalSemaphore(&hip_binary_ext_semaphore, &hip_binary_sem_handle_desc));

  const auto timeline_semaphore = vkt.CreateExternalSemaphore(second_semaphore_type);
  const auto hip_timeline_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(timeline_semaphore, second_semaphore_type);
  hipExternalSemaphore_t hip_timeline_ext_semaphore;
  HIP_CHECK(hipImportExternalSemaphore(&hip_timeline_ext_semaphore, &hip_timeline_sem_handle_desc));

  uint64_t wait_values[] = {1, 0};
  VkTimelineSemaphoreSubmitInfo timeline_submit_info = {};
  timeline_submit_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timeline_submit_info.waitSemaphoreValueCount = 2;
  timeline_submit_info.pWaitSemaphoreValues = wait_values;

  VkSemaphore wait_semaphores[] = {timeline_semaphore, binary_semaphore};
  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  submit_info.waitSemaphoreCount = 2;
  submit_info.pWaitSemaphores = wait_semaphores;
  submit_info.pNext =
      second_semaphore_type == VK_SEMAPHORE_TYPE_TIMELINE ? &timeline_submit_info : nullptr;
  const auto fence = vkt.CreateFence();
  VK_CHECK_RESULT(vkQueueSubmit(vkt.GetQueue(), 1, &submit_info, fence));

  REQUIRE(vkGetFenceStatus(vkt.GetDevice(), fence) == VK_NOT_READY);

  hipExternalSemaphoreSignalParams binary_signal_params = {};
  binary_signal_params.params.fence.value = 0;
  hipExternalSemaphoreSignalParams timeline_signal_params = {};
  timeline_signal_params.params.fence.value =
      second_semaphore_type == VK_SEMAPHORE_TYPE_TIMELINE ? 2 : 0;
  hipExternalSemaphore_t ext_semaphores[] = {hip_binary_ext_semaphore, hip_timeline_ext_semaphore};
  hipExternalSemaphoreSignalParams signal_params[] = {binary_signal_params, timeline_signal_params};
  HIP_CHECK(hipSignalExternalSemaphoresAsync(ext_semaphores, signal_params, 2, nullptr));

  VK_CHECK_RESULT(
      vkWaitForFences(vkt.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000 /*5 seconds*/));

  HIP_CHECK(hipDestroyExternalSemaphore(hip_binary_ext_semaphore));
  HIP_CHECK(hipDestroyExternalSemaphore(hip_timeline_ext_semaphore));
}

TEST_CASE("Unit_hipSignalExternalSemaphoresAsync_Vulkan_Negative_Parameters") {
  VulkanTest vkt(enable_validation);
  hipExternalSemaphoreSignalParams signal_params = {};
  signal_params.params.fence.value = 1;

  SECTION("extSemArray == nullptr") {
    HIP_CHECK_ERROR(hipSignalExternalSemaphoresAsync(nullptr, &signal_params, 1, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("paramsArray == nullptr") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    HIP_CHECK_ERROR(hipSignalExternalSemaphoresAsync(&hip_ext_semaphore, nullptr, 1, nullptr),
                    hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }

  SECTION("Wait params flags  != 0") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    signal_params.flags = 1;
    HIP_CHECK_ERROR(
        hipSignalExternalSemaphoresAsync(&hip_ext_semaphore, &signal_params, 1, nullptr),
        hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }

  SECTION("Invalid stream") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipSignalExternalSemaphoresAsync(&hip_ext_semaphore, &signal_params, 1, stream),
                    hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }
}