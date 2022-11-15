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

#pragma once

#include <vulkan/vulkan.h>
#include <vector>

#ifdef _WIN64
#include <VersionHelpers.h>
#endif

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

#define VK_CHECK_RESULT(code)                                                                      \
  {                                                                                                \
    VkResult res = (code);                                                                         \
    if (res != VK_SUCCESS) {                                                                       \
      INFO("Vulkan error: " << std::to_string(res) << "\n In File: " << __FILE__                   \
                            << "\n At line: " << __LINE__);                                        \
      REQUIRE(false);                                                                              \
    }                                                                                              \
  }

class VulkanTest {
 public:
  VulkanTest(bool enable_validation)
      : _enable_validation{enable_validation},
        _sem_handle_type{GetVkSemHandlePlatformType()},
        _mem_handle_type{GetVkMemHandlePlatformType()} {
    CreateInstance();
    CreateDevice();
    CreateCommandBuffer();
  }

  ~VulkanTest() {
    for (const auto s : _semaphores) {
      vkDestroySemaphore(_device, s, nullptr);
    }

    for (const auto f : _fences) {
      vkDestroyFence(_device, f, nullptr);
    }

    for (const auto& s : _stores) {
      vkUnmapMemory(_device, s.memory);
      vkDestroyBuffer(_device, s.buffer, nullptr);
      vkFreeMemory(_device, s.memory, nullptr);
    }

    if (_command_buffer != VK_NULL_HANDLE)
      vkFreeCommandBuffers(_device, _command_pool, 1, &_command_buffer);

    if (_command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(_device, _command_pool, nullptr);

    if (_device != VK_NULL_HANDLE) vkDestroyDevice(_device, nullptr);

    if (_instance != VK_NULL_HANDLE) vkDestroyInstance(_instance, nullptr);
  }

  VulkanTest(const VulkanTest&) = delete;

  VulkanTest(VulkanTest&&) = delete;

  template <typename T> struct MappedBuffer {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    uint32_t size = 0;
    T* host_ptr = nullptr;
  };

  template <typename T>
  MappedBuffer<T> CreateMappedStorage(uint32_t count, VkBufferUsageFlagBits transfer_flags,
                                      bool external = false);

  VkFence CreateFence();

  VkSemaphore CreateExternalSemaphore(VkSemaphoreType sem_type, uint64_t initial_value = 0);

  hipExternalSemaphoreHandleDesc BuildSemaphoreDescriptor(VkSemaphore vk_sem,
                                                          VkSemaphoreType sem_type);

  hipExternalMemoryHandleDesc BuildMemoryDescriptor(VkDeviceMemory vk_mem, uint32_t size);


  VkDevice GetDevice() const { return _device; }

  VkCommandBuffer GetCommandBuffer() const { return _command_buffer; }

  VkQueue GetQueue() const { return _queue; }

 private:
  void CreateInstance();

  void CreateDevice();

  void CreateCommandBuffer();

  bool CheckExtensionSupport(std::vector<const char*> expected_extensions);

  void EnableValidationLayer();

  uint32_t GetComputeQueueFamilyIndex();

  void FindPhysicalDevice();

  uint32_t FindMemoryType(uint32_t memory_type_bits, VkMemoryPropertyFlags properties);

  hipExternalSemaphoreHandleType VulkanSemHandleTypeToHIPHandleType(VkSemaphoreType sem_type);

  hipExternalMemoryHandleType VulkanMemHandleTypeToHIPHandleType();

#ifdef _WIN64
  HANDLE
  GetSemaphoreHandle(VkSemaphore semaphore);
#else
  int GetSemaphoreHandle(VkSemaphore semaphore);
#endif

#ifdef _WIN64
  HANDLE
  GetMemoryHandle(VkDeviceMemory memory);
#else
  int GetMemoryHandle(VkDeviceMemory memory);
#endif

  VkExternalSemaphoreHandleTypeFlagBits GetVkSemHandlePlatformType() const;

  VkExternalMemoryHandleTypeFlagBits GetVkMemHandlePlatformType() const;

  struct Storage {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t size = 0u;
  };

 private:
  const bool _enable_validation = false;
  const VkExternalSemaphoreHandleTypeFlagBits _sem_handle_type;
  const VkExternalMemoryHandleTypeFlagBits _mem_handle_type;
  VkInstance _instance = VK_NULL_HANDLE;
  VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
  VkDevice _device = VK_NULL_HANDLE;
  VkQueue _queue = VK_NULL_HANDLE;
  VkCommandPool _command_pool = VK_NULL_HANDLE;
  VkCommandBuffer _command_buffer = VK_NULL_HANDLE;
  uint32_t _compute_family_queue_idx = 0u;
  std::vector<const char*> _enabled_layers;

  std::vector<VkSemaphore> _semaphores;
  std::vector<VkFence> _fences;
  std::vector<Storage> _stores;

  std::vector<const char*> _required_instance_extensions{
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME};
#ifdef _WIN64
  std::vector<const char*> _required_device_extensions{
      VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME};
#else
  std::vector<const char*> _required_device_extensions{
      VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
#endif
};


template <typename T>
VulkanTest::MappedBuffer<T> VulkanTest::CreateMappedStorage(uint32_t count,
                                                            VkBufferUsageFlagBits transfer_flags,
                                                            bool external) {
  Storage storage;
  const auto size = count * sizeof(T);

  VkBufferCreateInfo buffer_create_info = {};
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.size = size;
  buffer_create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | transfer_flags;
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkExternalMemoryBufferCreateInfo external_memory_buffer_info = {};
  if (external) {
    external_memory_buffer_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_memory_buffer_info.handleTypes = _mem_handle_type;
    buffer_create_info.pNext = &external_memory_buffer_info;
  }
  VK_CHECK_RESULT(vkCreateBuffer(_device, &buffer_create_info, nullptr, &storage.buffer));

  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(_device, storage.buffer, &memory_requirements);
  storage.size = memory_requirements.size;

  VkMemoryAllocateInfo allocate_info = {};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = memory_requirements.size;
  allocate_info.memoryTypeIndex =
      FindMemoryType(memory_requirements.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  REQUIRE(allocate_info.memoryTypeIndex != VK_MAX_MEMORY_TYPES);

  VkExportMemoryAllocateInfoKHR vulkan_export_memory_allocate_info = {};
  if (external) {
    vulkan_export_memory_allocate_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    vulkan_export_memory_allocate_info.handleTypes = _mem_handle_type;

#ifdef _WIN64
    WindowsSecurityAttributes winSecurityAttributes;

    VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
    vulkanExportMemoryWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    vulkanExportMemoryWin32HandleInfoKHR.pNext = NULL;
    vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
    vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)NULL;

    vulkan_export_memory_allocate_info.pNext =
        _mem_handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
        ? &vulkanExportMemoryWin32HandleInfoKHR
        : NULL;
#endif
    allocate_info.pNext = &vulkan_export_memory_allocate_info;
  }

  VK_CHECK_RESULT(vkAllocateMemory(_device, &allocate_info, nullptr, &storage.memory));
  VK_CHECK_RESULT(vkBindBufferMemory(_device, storage.buffer, storage.memory, 0));

  T* host_ptr = nullptr;
  VK_CHECK_RESULT(vkMapMemory(_device, storage.memory, 0, storage.size, 0,
                              reinterpret_cast<void**>(&host_ptr)));

  _stores.push_back(storage);
  return MappedBuffer<T>{storage.memory, storage.buffer, storage.size, host_ptr};
}

// Sometimes in CUDA the stream is not immediately ready after a semaphore has been signaled
void PollStream(hipStream_t stream, hipError_t expected, uint32_t num_iterations = 5);

hipExternalSemaphore_t ImportBinarySemaphore(VulkanTest& vkt);