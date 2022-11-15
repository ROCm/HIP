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

#include <iostream>
#include <algorithm>

VkFence VulkanTest::CreateFence() {
  VkFence fence;
  VkFenceCreateInfo fence_create_info = {};
  fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.flags = 0;
  VK_CHECK_RESULT(vkCreateFence(_device, &fence_create_info, nullptr, &fence));

  _fences.push_back(fence);
  return fence;
}

VkSemaphore VulkanTest::CreateExternalSemaphore(VkSemaphoreType sem_type, uint64_t initial_value) {
  VkExportSemaphoreCreateInfoKHR export_sem_create_info = {};
  export_sem_create_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
  export_sem_create_info.handleTypes = _sem_handle_type;

  if (sem_type == VK_SEMAPHORE_TYPE_TIMELINE) {
    VkSemaphoreTypeCreateInfo timeline_create_info = {};
    timeline_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_create_info.initialValue = initial_value;
    export_sem_create_info.pNext = &timeline_create_info;
  } else {
    export_sem_create_info.pNext = nullptr;
  }

  VkSemaphoreCreateInfo semaphore_create_info = {};
  semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_create_info.pNext = &export_sem_create_info;

  VkSemaphore semaphore;
  VK_CHECK_RESULT(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &semaphore));

  _semaphores.push_back(semaphore);
  return semaphore;
}

hipExternalSemaphoreHandleDesc VulkanTest::BuildSemaphoreDescriptor(VkSemaphore vk_sem,
                                                                    VkSemaphoreType sem_type) {
  hipExternalSemaphoreHandleDesc sem_handle_desc = {};
  sem_handle_desc.type = VulkanSemHandleTypeToHIPHandleType(sem_type);
#ifdef _WIN64
  sem_handle_desc.handle.win32.handle = GetSemaphoreHandle(vk_sem);
#else
  sem_handle_desc.handle.fd = GetSemaphoreHandle(vk_sem);
#endif
  sem_handle_desc.flags = 0;

  return sem_handle_desc;
}

hipExternalMemoryHandleDesc VulkanTest::BuildMemoryDescriptor(VkDeviceMemory vk_mem,
                                                              uint32_t size) {
  hipExternalMemoryHandleDesc mem_handle_desc = {};
  mem_handle_desc.type = VulkanMemHandleTypeToHIPHandleType();
#ifdef _WIN64
  mem_handle_desc.handle.win32.handle = GetMemoryHandle(ck_mem);
#else
  mem_handle_desc.handle.fd = GetMemoryHandle(vk_mem);
#endif
  mem_handle_desc.size = size;

  return mem_handle_desc;
}

void VulkanTest::CreateInstance() {
  UNSCOPED_INFO("Not all of the required instance extensions are supported");
  REQUIRE(CheckExtensionSupport(_required_instance_extensions));
  if (_enable_validation) {
    EnableValidationLayer();
  }

  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = static_cast<uint32_t>(_required_instance_extensions.size());
  create_info.ppEnabledExtensionNames = _required_instance_extensions.data();
  create_info.enabledLayerCount = static_cast<uint32_t>(_enabled_layers.size());
  create_info.ppEnabledLayerNames = _enabled_layers.data();

  VK_CHECK_RESULT(vkCreateInstance(&create_info, nullptr, &_instance));
}

void VulkanTest::CreateDevice() {
  UNSCOPED_INFO("Not all of the required device extensions are supported");
  REQUIRE(CheckExtensionSupport(_required_device_extensions));

  FindPhysicalDevice();

  VkDeviceQueueCreateInfo queue_create_info = {};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = _compute_family_queue_idx = GetComputeQueueFamilyIndex();
  queue_create_info.queueCount = 1;
  float queue_priorities = 1.0;
  queue_create_info.pQueuePriorities = &queue_priorities;

  VkPhysicalDeviceVulkan12Features features = {};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features.timelineSemaphore = true;

  VkDeviceCreateInfo device_create_info = {};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.enabledLayerCount = _enabled_layers.size();
  device_create_info.ppEnabledLayerNames = _enabled_layers.data();
  device_create_info.enabledExtensionCount = _required_device_extensions.size();
  device_create_info.ppEnabledExtensionNames = _required_device_extensions.data();
  device_create_info.pQueueCreateInfos = &queue_create_info;
  device_create_info.queueCreateInfoCount = 1;
  device_create_info.pNext = &features;

  VK_CHECK_RESULT(vkCreateDevice(_physical_device, &device_create_info, nullptr, &_device));
  vkGetDeviceQueue(_device, _compute_family_queue_idx, 0, &_queue);
}

void VulkanTest::CreateCommandBuffer() {
  VkCommandPoolCreateInfo command_pool_create_info = {};
  command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_create_info.flags = 0;
  command_pool_create_info.queueFamilyIndex = _compute_family_queue_idx;
  VK_CHECK_RESULT(vkCreateCommandPool(_device, &command_pool_create_info, nullptr, &_command_pool));

  VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
  command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_buffer_allocate_info.commandPool = _command_pool;
  command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_allocate_info.commandBufferCount = 1;
  VK_CHECK_RESULT(
      vkAllocateCommandBuffers(_device, &command_buffer_allocate_info, &_command_buffer));
}

bool VulkanTest::CheckExtensionSupport(std::vector<const char*> expected_extensions) {
  uint32_t extension_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> extension_properties(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extension_properties.data());

  std::vector<const char*> supported_extensions;
  supported_extensions.reserve(extension_count);
  std::transform(extension_properties.begin(), extension_properties.end(),
                 std::back_inserter(supported_extensions),
                 [](const auto& p) { return p.extensionName; });

  constexpr auto p = [](const char* l, const char* r) { return strcmp(l, r) < 0; };
  std::sort(expected_extensions.begin(), expected_extensions.end(), p);
  std::sort(supported_extensions.begin(), supported_extensions.end(), p);

  return std::includes(supported_extensions.begin(), supported_extensions.end(),
                       expected_extensions.begin(), expected_extensions.end(),
                       [](const char* l, const char* r) { return strcmp(l, r) == 0; });
}

void VulkanTest::EnableValidationLayer() {
  uint32_t layer_count = 0;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  std::vector<VkLayerProperties> layer_properties(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, layer_properties.data());
  const bool found_val_layer =
      std::any_of(layer_properties.cbegin(), layer_properties.cend(), [](const auto& props) {
        return strcmp(props.layerName, "VK_LAYER_KHRONOS_validation") == 0;
      });


  if (found_val_layer) {
    _enabled_layers.push_back("VK_LAYER_KHRONOS_validation");
  } else {
    UNSCOPED_INFO("Validation was requested, but the validation layer could not be located");
    REQUIRE(found_val_layer);
  }
}

uint32_t VulkanTest::GetComputeQueueFamilyIndex() {
  uint32_t queue_family_count = 0u;

  vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &queue_family_count,
                                           queue_families.data());

  const auto it =
      std::find_if(queue_families.cbegin(), queue_families.cend(), [](const auto& props) {
        return props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT);
      });
  REQUIRE(it != queue_families.cend());

  return std::distance(queue_families.cbegin(), it);
}

void VulkanTest::FindPhysicalDevice() {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(_instance, &device_count, nullptr);
  REQUIRE(device_count != 0u);

  std::vector<VkPhysicalDevice> physical_devices(device_count);
  vkEnumeratePhysicalDevices(_instance, &device_count, physical_devices.data());

  _physical_device = physical_devices[0];
}

uint32_t VulkanTest::FindMemoryType(uint32_t memory_type_bits, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memory_properties;
  vkGetPhysicalDeviceMemoryProperties(_physical_device, &memory_properties);
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if ((memory_type_bits & (1 << i)) &&
        ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties)) {
      return i;
    }
  }
  return VK_MAX_MEMORY_TYPES;
}

hipExternalSemaphoreHandleType VulkanTest::VulkanSemHandleTypeToHIPHandleType(
    VkSemaphoreType sem_type) {
  if (sem_type == VK_SEMAPHORE_TYPE_BINARY) {
    if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
      return hipExternalSemaphoreHandleTypeOpaqueWin32;
    } else if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
      return hipExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    } else if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
      return hipExternalSemaphoreHandleTypeOpaqueFd;
    }
  } else if (sem_type == VK_SEMAPHORE_TYPE_TIMELINE) {
#if HT_AMD
    throw std::invalid_argument("Timeline semaphore unsupported on AMD");
#else
    if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
      return hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    } else if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
      return hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    } else if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
      return hipExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    }
#endif
  }

  throw std::invalid_argument("Invalid vulkan semaphore handle type");
}

hipExternalMemoryHandleType VulkanTest::VulkanMemHandleTypeToHIPHandleType() {
  if (_mem_handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
    return hipExternalMemoryHandleTypeOpaqueWin32;
  } else if (_mem_handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
    return hipExternalMemoryHandleTypeOpaqueWin32Kmt;
  } else if (_mem_handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
    return hipExternalMemoryHandleTypeOpaqueFd;
  }

  throw std::invalid_argument("Invalid vulkan memory handle type");
}

#ifdef _WIN64
HANDLE
VulkanTest::GetSemaphoreHandle(VkSemaphore semaphore) {
  HANDLE handle = 0;

  VkSemaphoreGetWin32HandleInfoKHR semaphoreGetWin32HandleInfoKHR = {};
  semaphoreGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  semaphoreGetWin32HandleInfoKHR.pNext = NULL;
  semaphoreGetWin32HandleInfoKHR.semaphore = semaphore;
  semaphoreGetWin32HandleInfoKHR.handleType = _sem_handle_type;

  PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
  fpGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
      _device, "vkGetSemaphoreWin32HandleKHR");
  if (!fpGetSemaphoreWin32HandleKHR) {
    throw std::runtime_error("Failed to retrieve vkGetSemaphoreWin32HandleKHR");
  }
  if (fpGetSemaphoreWin32HandleKHR(_device, &semaphoreGetWin32HandleInfoKHR, &handle) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }

  return handle;
}
#else
int VulkanTest::GetSemaphoreHandle(VkSemaphore semaphore) {
  int fd;

  VkSemaphoreGetFdInfoKHR semaphoreGetFdInfoKHR = {};
  semaphoreGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  semaphoreGetFdInfoKHR.pNext = NULL;
  semaphoreGetFdInfoKHR.semaphore = semaphore;
  semaphoreGetFdInfoKHR.handleType = _sem_handle_type;

  PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
  fpGetSemaphoreFdKHR =
      (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(_device, "vkGetSemaphoreFdKHR");
  if (!fpGetSemaphoreFdKHR) {
    throw std::runtime_error("Failed to retrieve vkGetSemaphoreFdKHR");
  }
  if (fpGetSemaphoreFdKHR(_device, &semaphoreGetFdInfoKHR, &fd) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve semaphore handle");
  }

  return fd;
}
#endif

#ifdef _WIN64
HANDLE
VulkanTest::GetMemoryHandle(VkDeviceMemory memory) {
  Handle handle = 0;

  VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
  vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  vkMemoryGetWin32HandleInfoKHR.memory = memory;
  vkMemoryGetWin32HandleInfoKHR.handleType = _mem_handle_type;

  PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR =
      (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(m_device, "vkGetMemoryWin32HandleKHR");

  if (!fpGetMemoryWin32HandleKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR");
  }
  if (fpGetMemoryWin32HandleKHR(_device, &vkMemoryGetWin32HandleInfoKHR, &handle) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve memory handle");
  }

  return handle;
}
#else
int VulkanTest::GetMemoryHandle(VkDeviceMemory memory) {
  int fd;

  VkMemoryGetFdInfoKHR memoryGetFdInfoKHR = {};
  memoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  memoryGetFdInfoKHR.memory = memory;
  memoryGetFdInfoKHR.handleType = _mem_handle_type;

  PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(_device, "vkGetMemoryFdKHR");
  if (!fpGetMemoryFdKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryFdKHR");
  }
  if (fpGetMemoryFdKHR(_device, &memoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve memory handle");
  }

  return fd;
}
#endif

VkExternalSemaphoreHandleTypeFlagBits VulkanTest::GetVkSemHandlePlatformType() const {
#ifdef _WIN64
  return IsWindows8OrGreater() ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
                               : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

VkExternalMemoryHandleTypeFlagBits VulkanTest::GetVkMemHandlePlatformType() const {
#ifdef _WIN64
  return IsWindows8OrGreater() ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                               : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

// Sometimes in CUDA the stream is not immediately ready after a semaphore has been signaled
void PollStream(hipStream_t stream, hipError_t expected, uint32_t num_iterations) {
  hipError_t query_result;
  for (uint32_t _ = 0; _ < num_iterations; ++_) {
    if ((query_result = hipStreamQuery(stream)) != expected) {
      std::this_thread::sleep_for(std::chrono::milliseconds{5});
    } else {
      break;
    }
  }
  REQUIRE(expected == query_result);
}

hipExternalSemaphore_t ImportBinarySemaphore(VulkanTest& vkt) {
  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  const auto sem_handle_desc = vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_BINARY);
  hipExternalSemaphore_t hip_ext_semaphore;
  HIP_CHECK(hipImportExternalSemaphore(&hip_ext_semaphore, &sem_handle_desc));

  return hip_ext_semaphore;
}