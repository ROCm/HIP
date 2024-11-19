// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "example_utils.hpp"
#include "vulkan_utils.hpp"

#include "nvidia_hip_fix.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "sinewave.frag.spv.h"
#include "sinewave.vert.spv.h"

// Currently hip external semaphores are not working under Linux when
// compiling for the AMD platform.
// TODO: Remove once this is implemented in hipamd.
// See https://github.com/ROCm-Developer-Tools/hipamd/issues/48.
#ifndef USE_EXTERNAL_SEMAPHORES
    #if defined(__HIP_PLATFORM_AMD__) && !defined(_WIN64)
        #define USE_EXTERNAL_SEMAPHORES 0
    #else
        #define USE_EXTERNAL_SEMAPHORES 1
    #endif
#endif

// Currently it seems like waiting on an external semaphore that is signaled
// from hip is not working under windows
#ifndef USE_SIGNAL_SEMAPHORE
    #if defined(__HIP_PLATFORM_AMD__) && defined(_WIN64)
        #define USE_SIGNAL_SEMAPHORE 0

    #else
        #define USE_SIGNAL_SEMAPHORE 1
    #endif
#endif

/// \brief The maximum number of frames that can be rendered at the same time. By
/// setting this value to more than one, we can allow the presentation engine to
/// draw the rendered frame to the monitor while we already render the next frame
/// in the background.
constexpr size_t max_frames_in_flight = 2;

/// \brief Time maximum time (in nanoseconds) that we are willing to wait on the next
/// image from the swapchain.
constexpr uint64_t frame_timeout = std::numeric_limits<uint64_t>::max();

/// \brief The number of triangles that the example's grid is in width.
constexpr uint32_t grid_width = 256;
/// \brief The number of triangles that the example's grid is in height.
constexpr uint32_t grid_height = 256;

/// \brief The Vulkan instance extensions required for sharing HIP- and Vulkan
/// types. \p VK_KHR_external_memory_capabilities is required to share buffers, and
/// \p VK_KHR_external_semaphore_capabilities is required to share semaphores.
/// \p VK_KHR_get_physical_device_properties2 is required for the other two, as well
/// as for querying the device's UUID.
constexpr const char* required_instance_extensions[] = {
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
};

/// \brief The general Vulkan extensions that a particular device needs to support in order
/// for it to be able to run this example.
/// \p VK_KHR_swapchain is required in order to draw to the example's window, and \p VK_KHR_external_memory
/// and \p VK_KHR_external_semaphore are required to share memory and semaphores respectively with HIP.
constexpr const char* required_device_extensions[]
    = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
       VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
       VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
       VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
       VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
#else
       VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
       VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME};
#endif

/// \brief This structure represents a device UUID, obtained either from Vulkan or
/// from HIP.
struct uuid
{
    uint8_t bytes[VK_UUID_SIZE];

    /// \brief This function fetches a Vulkan-compatible device UUID from a HIP device.
    ///
    /// The use of this function should actually be replaced by \p hipDeviceGetUuid. However,
    /// on AMD it returns a device UUID that is not compatible with that returned by Vulkan, and
    /// when compiling for NVIDIA it yields a linker error. For this reason we provide our own
    /// implementation that is compatible with both the Mesa (RADV) and AMD (AMDVLK) implementations
    /// of Vulkan on AMD, and call into the CUDA API directly when compiling for NVIDIA.
    static uuid get_hip_device_uuid(hipDevice_t device)
    {
#if defined(__HIP_PLATFORM_AMD__)
        // The value that hipDeviceGetUuid returns does not correspond with those returned
        // by mesa (see https://gitlab.freedesktop.org/mesa/mesa/-/blob/5cd3e395037250946ba2519600836341df02c8ca/src/amd/common/ac_gpu_info.c#L1366-1382)
        // and by xgl (see https://github.com/GPUOpen-Drivers/xgl/blob/4118707939c2f4783d28ce2a383184a3794ca477/icd/api/vk_physical_device.cpp#L4363-L4421)
        // Those drivers _do_ align with each other, so we can create our own UUID here.
        // \see https://github.com/ROCm-Developer-Tools/hipamd/issues/50.
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, device));

        struct uuid result    = {};
        uint32_t*   uuid_ints = reinterpret_cast<uint32_t*>(result.bytes);
        uuid_ints[0]          = props.pciDomainID;
        uuid_ints[1]          = props.pciBusID;
        uuid_ints[2]          = props.pciDeviceID;
        // Note: function is 0 anyway.

        return result;
#elif defined(__HIP_PLATFORM_NVIDIA__)
        // Work around a compile error related to hipDeviceGetUuid when compiling for NVIDIA:
        // "undefined reference to `cuDeviceGetUuid'"
        // \see https://github.com/ROCm-Developer-Tools/hipamd/issues/51.
        cudaDeviceProp props;
        HIP_CHECK(hipCUDAErrorTohipError(cudaGetDeviceProperties(&props, device)));

        struct uuid result = {};
        std::memcpy(result.bytes, props.uuid.bytes, VK_UUID_SIZE);

        return result;
#else
    #error unsupported platform
#endif
    }
};

/// \brief \p std::ostream print operator overload for \p uuid.
/// \see uuid.
std::ostream& operator<<(std::ostream& os, const uuid uuid)
{
    for(size_t i = 0; i < VK_UUID_SIZE * 2; ++i)
    {
        // Extract the current nibble.
        const uint8_t c = (uuid.bytes[i / 2] >> (4 - (i % 2) * 4)) & 0xF;
        os << static_cast<char>(c < 10 ? c + '0' : c + 'a' - 10);
        if(i == 8 || i == 12 || i == 16 || i == 20)
        {
            os << '-';
        }
    }
    return os;
}

/// \brief This structure represents a candidate HIP-device that we can use
/// for this example.
struct hip_device_candidate
{
    /// The HIP device index representing this device.
    hipDevice_t device;
    /// The Vulkan-compatible device UUID.
    uuid device_uuid;
};

/// \brief This structure represents a candidate device that we can use for this
/// example.
struct physical_device_candidate
{
    /// The Vulkan physical device handle of the device to be used.
    VkPhysicalDevice pdev;

    /// The candidate device's Vulkan device properties.
    VkPhysicalDeviceProperties props;

    /// The HIP device candidate that this Vulkan device corresponds to.
    hip_device_candidate hip_candidate;

    /// The queue allocation that contains details about which queues will be
    /// used throughout this example.
    queue_allocation queues;
};

/// \brief Checks if a particular Vulkan physical device is qualified to run this example:
/// - It needs to support the Vulkan surface which we want to render to.
/// - It needs to support the required generic and platform-specific Vulkan device extensions.
/// - It needs to be a HIP-supported device. This is checked by fetching the device
///   UUID from Vulkan, and checking if it appears in the device UUIDs fetched from HIP
///   (passed through \p hip_uuids).
/// - It needs to support graphics- and present queues that can render to the surface.
/// If all of these are satisfied, the \p candidate structure is filled with information
/// about the physical device that is required later, and the function returns \p true.
/// Otherwise, \p false is returned.
///
/// \param hip_devices - A vector of \p hipDevice_t and their corresponding Vulkan-compatible
///   device UUID.
/// \param pdev - The Vulkan physical device to check suitability off.
/// \p surface - The Vulkan surface that the physical device needs to support.
bool is_physical_device_suitable(const instance_dispatch&                dispatch,
                                 const std::vector<hip_device_candidate> hip_devices,
                                 VkPhysicalDevice                        pdev,
                                 VkSurfaceKHR                            surface,
                                 physical_device_candidate&              candidate)
{
    // Check if HIP supports this device by checking if there is any device with the same UUID.
    {
        // Query the Vulkan device UUID using vkGetPhysicalDeviceProperties2.
        VkPhysicalDeviceIDPropertiesKHR id_props = {};
        id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR;

        VkPhysicalDeviceProperties2KHR props2 = {};
        props2.sType                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
        props2.pNext                          = &id_props;

        dispatch.get_physical_device_properties2(pdev, &props2);

        const auto cmp_device_uuid = [&](const hip_device_candidate& hip_candidate)
        {
            return std::equal(std::begin(hip_candidate.device_uuid.bytes),
                              std::end(hip_candidate.device_uuid.bytes),
                              std::begin(id_props.deviceUUID),
                              std::end(id_props.deviceUUID));
        };

        // Try to find a HIP device UUID that matches the UUID reported by Vulkan - if any such exists,
        // we know that the device supports both Vulkan and HIP, and we can use it to run this example.
        const auto it = std::find_if(hip_devices.begin(), hip_devices.end(), cmp_device_uuid);
        if(it == hip_devices.end())
        {
            // This device does not support HIP.
            return false;
        }

        candidate.props         = props2.properties;
        candidate.hip_candidate = *it;
    }

    // Check if the device supports our surface at all.
    if(!check_surface_support(dispatch, pdev, surface))
    {
        return false;
    }

    // Check if the device supports the required extensions.
    if(!check_device_extensions(dispatch,
                                pdev,
                                required_device_extensions,
                                std::size(required_device_extensions)))
    {
        return false;
    }

    // Try to allocate device queues for the candidate device.
    if(!allocate_device_queues(dispatch, pdev, surface, candidate.queues))
    {
        return false;
    }

    candidate.pdev = pdev;
    return true;
}

/// \brief Try to find a physical device that can run this example. This is done by fetching
/// all supported devices from HIP and from Vulkan, and checking each of these to see if the required
/// features are supported.
///
/// To check whether a Vulkan and HIP device are the same, their UUIDs are compared.
/// \see \p uuid::get_hip_device_uuid.
/// \see \p is_physical_device_suitable.
void find_physical_device(const instance_dispatch&   dispatch,
                          VkInstance                 instance,
                          VkSurfaceKHR               surface,
                          physical_device_candidate& candidate)
{
    uint32_t physical_device_count;
    VK_CHECK(dispatch.enumerate_physical_devices(instance, &physical_device_count, nullptr));
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    VK_CHECK(dispatch.enumerate_physical_devices(instance,
                                                 &physical_device_count,
                                                 physical_devices.data()));

    if(physical_device_count == 0)
    {
        std::cerr << "System has no physical devices\n";
        std::exit(error_exit_code);
    }

    // Fetch the number of HIP devices that are currently present on the system.
    // Note: This depends on the current HIP platform, and may report different
    // devices depending on that.
    int hip_device_count;
    HIP_CHECK(hipGetDeviceCount(&hip_device_count));

    // For each HIP device, check to see if we can use it all, and then query
    // its Vulkan-compatible device UUID.
    std::vector<hip_device_candidate> hip_devices;
    for(hipDevice_t hip_device = 0; hip_device < hip_device_count; ++hip_device)
    {
        hipDeviceProp_t hip_properties;
        HIP_CHECK(hipGetDeviceProperties(&hip_properties, hip_device));
        if(hip_properties.computeMode == hipComputeModeProhibited)
            continue;

        const uuid device_uuid = uuid::get_hip_device_uuid(hip_device);
        hip_devices.push_back({hip_device, device_uuid});
    }

    for(VkPhysicalDevice pdev : physical_devices)
    {
        if(is_physical_device_suitable(dispatch, hip_devices, pdev, surface, candidate))
        {
            return;
        }
    }

    std::cerr << "No suitable device\n";
    std::exit(error_exit_code);
}

/// \brief Allocate and bind memory for a Vulkan buffer
/// \param buffer - The buffer to allocate create memory for.
/// \param properties - The memory properties for the allocated memory.
/// \param external - Whether to allocate this memory such that it can be exported.
VkDeviceMemory allocate_buffer_memory(const graphics_context&     ctx,
                                      const VkBuffer              buffer,
                                      const VkMemoryPropertyFlags properties,
                                      const bool                  external = false)
{
    VkMemoryRequirements mem_reqs;
    ctx.vkd->get_buffer_memory_requirements(ctx.dev, buffer, &mem_reqs);

    const uint32_t memory_type = ctx.find_memory_type_index(mem_reqs.memoryTypeBits, properties);

    VkMemoryAllocateInfo allocate_info = {};
    allocate_info.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize       = mem_reqs.size;
    allocate_info.memoryTypeIndex      = memory_type;

    VkExportMemoryAllocateInfoKHR export_info = {};
    export_info.sType                         = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;
#else
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

    if(external)
    {
        allocate_info.pNext = &export_info;
    }

    VkDeviceMemory memory;
    VK_CHECK(ctx.vkd->allocate_memory(ctx.dev, &allocate_info, nullptr, &memory));
    VK_CHECK(ctx.vkd->bind_buffer_memory(ctx.dev, buffer, memory, 0));
    return memory;
}

/// \brief Create and allocate a Vulkan buffer.
/// \param size - The size (in bytes) that this buffer should be allocated for.
/// \param usage - The Vulkan usage that this buffer will be used for.
/// \param external - If true, this buffer will be created so that it can later be exported to a
///   platform-native handle, that may be imported to HIP.
VkBuffer create_buffer(const graphics_context&  ctx,
                       const VkDeviceSize       size,
                       const VkBufferUsageFlags usage,
                       const bool               external = false)
{
    VkBufferCreateInfo create_info = {};
    create_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_info.size               = size;
    create_info.usage              = usage;
    create_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

    // In order to be able to export the buffer handle, we need to supply Vulkan with this
    // VkExternalMemoryBufferCreateInfoKHR, and set the handleTypes to the native handle type
    // that we want to export. Which handle type to export depends on the platform we are
    // currently compiling for.
    VkExternalMemoryBufferCreateInfoKHR external_create_info = {};
    external_create_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR;
#ifdef _WIN64
    external_create_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;
#else
    external_create_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
    // If exporting, add the external buffer create information to the buffer's create info
    // so that it gets passed to Vulkan.
    if(external)
    {
        create_info.pNext = &external_create_info;
    }

    VkBuffer buffer;
    VK_CHECK(ctx.vkd->create_buffer(ctx.dev, &create_info, nullptr, &buffer));
    return buffer;
}

/// \brief This function converts a Vulkan memory handle to its equivalent HIP handle. The
/// VkDeviceMemory passed to this function and the returned HIP memory represents the same
/// physical area of GPU memory, through the handles of each respective API. Writing to the
/// buffer in one API will allow us to read the results through the other. Note that access
/// to the buffer should be synchronized between the APIs, for example using queue syncs or
/// semaphores.
/// \param memory - The Vulkan memory handle to convert. This memory needs to be created with
///   the appropriate fields set in VkExportMemoryAllocateInfoKHR.
///   \see allocate_buffer_memory for allocating such a memory handle, and
///   \see create_buffer for creating a Vulkan buffer that is compatible with that memory.
hipExternalMemory_t
    memory_to_hip(const graphics_context& ctx, const VkDeviceMemory memory, const VkDeviceSize size)
{
    // [Sphinx vulkan memory to hip start]
    // Prepare the HIP external semaphore descriptor with the platform-specific
    // handle type that we wish to import. This value should correspond to the
    // handleTypes field set in VkExportMemoryAllocateInfoKHR while creating the
    // Vulkan buffer.
    hipExternalMemoryHandleDesc desc = {};
    desc.size                        = size;

    // Export the Vulkan buffer handle to a platform-specific native handle, depending
    // on the current platform: On Windows the buffer is converted to a HANDLE, and on Linux
    // to a file descriptor representing the driver's GPU handle to the memory.
    // This native handle is then passed to the HIP external memory descriptor so that it
    // may be imported.
#ifdef _WIN64
    desc.type = hipExternalMemoryHandleTypeOpaqueWin32Kmt;

    VkMemoryGetWin32HandleInfoKHR get_handle_info = {};
    get_handle_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    get_handle_info.memory     = memory;
    get_handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;

    VK_CHECK(
        ctx.vkd->get_memory_win32_handle(ctx.dev, &get_handle_info, &desc.handle.win32.handle));
#else
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;

    VkMemoryGetFdInfoKHR get_fd_info = {};
    get_fd_info.sType                = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    get_fd_info.memory               = memory;
    get_fd_info.handleType           = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    VK_CHECK(ctx.vkd->get_memory_fd(ctx.dev, &get_fd_info, &desc.handle.fd));
#endif

    // Import the native memory handle to HIP to create an external memory.
    hipExternalMemory_t hip_memory;
    HIP_CHECK(hipImportExternalMemory(&hip_memory, &desc));
    return hip_memory;
    // [Sphinx vulkan memory to hip end]
}

/// \brief Utility function to create a Vulkan semaphore.
/// \param external - If true, this semaphore is created so that it can later be exported
///   to a platform-native handle, which may be imported to HIP later.
VkSemaphore create_semaphore(const graphics_context& ctx, const bool external = false)
{
    VkSemaphoreCreateInfo create_info = {};
    create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    // Similar to buffers, in order to be able to export the semaphore handle we need to supply
    // Vulkan with this VkExportSemaphoreCreateInfoKHR structure, and set the handleTypes to the
    // value appropriate for the platform that we are currently compiling for.
    VkExportSemaphoreCreateInfoKHR export_create_info = {};
    export_create_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
#ifdef _WIN64
    export_create_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#else
    export_create_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

    // If exporting, add the export structure to the create info chain.
    if(external)
    {
        create_info.pNext = &export_create_info;
    }

    VkSemaphore sema;
    VK_CHECK(ctx.vkd->create_semaphore(ctx.dev, &create_info, nullptr, &sema));
    return sema;
}

/// \brief This function converts a Vulkan semaphore to its equivalent HIP handle. The passed
/// semaphore and the returned HIP semaphore represent the same backing semaphore, though the
/// handles of the respective API. Signaling on the semaphore in one API will allow the other
/// API to wait on it, which is how we can guarantee synchronized access to resources in a
/// cross-API manner.
/// \param sema - The Vulkan semaphore to convert. This semaphore needs to be created with
///   \p the appropriate fields set in VkExportSemaphoreCreateInfoKHR.
///   \see create_semaphore for creating such a semaphore.
hipExternalSemaphore_t semaphore_to_hip(const graphics_context& ctx, const VkSemaphore sema)
{
    // [Sphinx semaphore import start]
    // Prepare the HIP external semaphore descriptor with the platform-specific handle type
    // that we wish to import. This value should correspond to the handleTypes field set in
    // the VkExportSemaphoreCreateInfoKHR structure that was passed to Vulkan when creating
    // the semaphore.
    hipExternalSemaphoreHandleDesc desc = {};

    // Export the Vulkan semaphore to a platform-specific handle depending on the current
    // platform: On Windows, we convert the semaphore into a HANDLE, and on Linux it is
    // converted to a file descriptor.
    // This native handle is then passed to the HIP external semaphore descriptor.
#ifdef _WIN64
    desc.type = hipExternalSemaphoreHandleTypeOpaqueWin32;

    VkSemaphoreGetWin32HandleInfoKHR get_handle_info = {};
    get_handle_info.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    get_handle_info.semaphore  = sema;
    get_handle_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;

    VK_CHECK(
        ctx.vkd->get_semaphore_win32_handle(ctx.dev, &get_handle_info, &desc.handle.win32.handle));

#else
    desc.type = hipExternalSemaphoreHandleTypeOpaqueFd;

    VkSemaphoreGetFdInfoKHR get_fd_info = {};
    get_fd_info.sType                   = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    get_fd_info.semaphore               = sema;
    get_fd_info.handleType              = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    VK_CHECK(ctx.vkd->get_semaphore_fd(ctx.dev, &get_fd_info, &desc.handle.fd));
#endif

    // Import the native semaphore to HIP to create a HIP external semaphore.
    hipExternalSemaphore_t hip_sema;
    HIP_CHECK(hipImportExternalSemaphore(&hip_sema, &desc));
    // [Sphinx semaphore import end]
    return hip_sema;
}

/// \brief When the HIP external memory is exported from Vulkan and imported to HIP, it
/// is not yet ready for use. To actually use the memory, we need to map it to a pointer
/// so that we may pass it to the kernel so that it can be read from and written to.
void* map_hip_external_memory(const hipExternalMemory_t mem, const VkDeviceSize size)
{
    // [Sphinx map external memory start]
    hipExternalMemoryBufferDesc desc = {};
    desc.offset                      = 0;
    desc.size                        = size;
    desc.flags                       = 0;

    void* ptr;
    HIP_CHECK(hipExternalMemoryGetMappedBuffer(&ptr, mem, &desc));
    // [Sphinx map external memory end]
    return ptr;
}

// [Sphinx sinewave kernel start]
/// \brief The main HIP kernel for this example - computes a simple sine wave over a
/// 2-dimensional grid of points.
/// \param height_map - the grid of points to compute a sine wave for. It is expected to be
///   a \p grid_width by \p grid_height array packed into memory.(y on the inner axis).
/// \param time - The current time relative to the start of the program.
__global__ void sinewave_kernel(float* height_map, const float time)
{
    const float        freq = 10.f;
    const unsigned int x    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y    = blockIdx.y * blockDim.y + threadIdx.y;
    const float        u    = (2.f * x) / grid_width - 1.f;
    const float        v    = (2.f * y) / grid_height - 1.f;

    if(x < grid_width && y < grid_height)
    {
        height_map[x * grid_width + y] = sinf(u * freq + time) * cosf(v * freq + time);
    }
}
// [Sphinx sinewave kernel end]

/// \brief In order to increase efficiency, we pipeline the rendering process. This allows us to render
/// the next frame already while another frame is being presented by Vulkan. The \p frame structure
/// contains the relevant Vulkan handles that are duplicated for each phase of the pipeline.
struct frame
{
    const graphics_context& ctx;

    /// The semaphore that guards the use of the swapchain image before it is ready.
    VkSemaphore image_acquired;
    /// The semaphore that guards the present before the image is rendered.
    VkSemaphore render_finished;
    /// A fence that allows us to synchronize on CPU until this frame is ready
    /// to be re-rendered again after it has been submitted to the GPU.
    VkFence frame_fence;
    /// The command pool that the command buffer for this frame will is allocated from.
    /// By having a separate pool for each frame we can reset the command for the frame simply
    /// by resetting the pool.
    VkCommandPool cmd_pool;
    /// The main command buffer for this frame.
    VkCommandBuffer cmd_buf;

    /// \brief Create a new <tt>frame</tt>.
    explicit frame(const graphics_context& ctx) : ctx(ctx)
    {
        this->image_acquired  = create_semaphore(ctx);
        this->render_finished = create_semaphore(ctx);

        VkFenceCreateInfo fence_create_info = {};
        fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_create_info.flags             = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(ctx.vkd->create_fence(ctx.dev, &fence_create_info, nullptr, &this->frame_fence));

        VkCommandPoolCreateInfo cmd_pool_create_info = {};
        cmd_pool_create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmd_pool_create_info.queueFamilyIndex        = ctx.graphics_queue.family;
        VK_CHECK(
            ctx.vkd->create_command_pool(ctx.dev, &cmd_pool_create_info, nullptr, &this->cmd_pool));

        VkCommandBufferAllocateInfo cmd_buf_allocate_info = {};
        cmd_buf_allocate_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_buf_allocate_info.commandPool        = this->cmd_pool;
        cmd_buf_allocate_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_buf_allocate_info.commandBufferCount = 1;
        VK_CHECK(
            ctx.vkd->allocate_command_buffers(ctx.dev, &cmd_buf_allocate_info, &this->cmd_buf));
    }

    ~frame()
    {
        this->ctx.vkd->destroy_command_pool(this->ctx.dev, this->cmd_pool, nullptr);
        this->ctx.vkd->destroy_fence(this->ctx.dev, this->frame_fence, nullptr);
        this->ctx.vkd->destroy_semaphore(this->ctx.dev, this->image_acquired, nullptr);
        this->ctx.vkd->destroy_semaphore(this->ctx.dev, this->render_finished, nullptr);
    }

    /// \brief Wait until the GPU-work for this frame has been completed, so that we
    /// can render to it again.
    void wait() const
    {
        VK_CHECK(this->ctx.vkd->wait_for_fences(this->ctx.dev,
                                                1,
                                                &this->frame_fence,
                                                VK_TRUE,
                                                frame_timeout));
    }

    /// \brief Reset the fence that backs this frame.
    void reset() const
    {
        VK_CHECK(this->ctx.vkd->reset_fences(this->ctx.dev, 1, &this->frame_fence));
    }
};

/// \brief This structure contains all the rendering related information for this example.
/// Its contents differ itself from the \p graphics_context in that in a typical Vulkan programs
/// there is usually only one <tt>graphics_context<tt>-like structure, but there may be multiple
/// <tt>renderer</tt>-like structures. In this example though, there is only one.
///
/// This renderer renders a grid of triangles to the window, the color of which is determined by
/// a HIP computation. Rendering is done using 3 buffers:
/// - One buffer contains the height of each triangle (rendered as color).
/// - One buffer holds the x- and y-coordinates for each of the corners of the triangle. Note: these
///   coordinates are unique, as the triangles that are made up from these points are defined by the
/// - Index buffer, that holds indices into the former two buffers to make up a list of triangles.
struct renderer
{
    /// The total number of vertices for the triangles.
    constexpr static size_t num_verts = grid_width * grid_height;
    /// The number of bytes in the x- and y-coordinates buffer. Each x/y coordinate is encoded as
    /// a pair of floats, which are stored in a packed  array-of-structures format: | x | y | x | y | ... |.
    constexpr static size_t grid_buffer_size = num_verts * sizeof(float) * 2;
    /// The number of bytes in the height buffer. Each height is encoded as a floating point value.
    /// This buffer will be shared with HIP, which is why these coordinates are
    /// stored in a separate buffer.
    constexpr static size_t height_buffer_size = num_verts * sizeof(float);

    /// The number of indices in the index buffer. Each triangle has 3 points, each square in the grid
    /// is made up of 2 triangles. There are (width - 1) by (height - 1) squares in the grid.
    constexpr static size_t num_indices = (grid_width - 1) * (grid_height - 1) * 3 * 2;
    /// The number of bytes in the index buffer. Each index is encoded as a 32-bit int.
    constexpr static size_t index_buffer_size = num_indices * sizeof(uint32_t);

    const graphics_context& ctx;
    swapchain&              sc;

    hipDevice_t hip_device;
    hipStream_t hip_stream;

    VkRenderPass render_pass;

    /// The frames in the rendering pipeline.
    std::vector<frame> frames;
    /// The index of the frame we are currently rendering to.
    uint32_t frame_index = 0;

    /// The Vulkan frame buffers to render to - each corresponds to a swapchain
    /// image with the same index in <tt>sc</tt>
    std::vector<VkFramebuffer> framebuffers;

    /// The pipeline layout and pipeline of the rendering pipeline for the Vulkan part
    /// of this example.
    VkPipelineLayout pipeline_layout;
    VkPipeline       pipeline;

    /// Whether the swapchain is out-of-date and needs to be recreated.
    bool swapchain_out_of_date = false;

    /// The buffer and memory holding the grid coordinates.
    VkBuffer       grid_buffer;
    VkDeviceMemory grid_memory;
    /// The buffer and memory holding the grid heights.
    /// This buffer will be exported to HIP.
    /// \see hip_height_memory.
    /// \see hip_height_buffer.
    VkBuffer       height_buffer;
    VkDeviceMemory height_memory;
    /// The buffer and memory holding the indices for the triangles to render.
    VkBuffer       index_buffer;
    VkDeviceMemory index_memory;

    /// The HIP-imported version of \p height_buffer.
    hipExternalMemory_t hip_height_memory;
    /// The HIP-imported version of \p height_buffer mapped into the program's memory.
    float* hip_height_buffer;

    /// The semaphore that guards between when the buffer has been rendered from the
    /// Vulkan side and when we can simulate it again from the HIP side, and
    /// its hip-imported version.
    VkSemaphore            buffer_ready;
    hipExternalSemaphore_t hip_buffer_ready;

    /// The semaphore that guards between when the simulation has finished from the HIP
    /// side and when we can render it to the swapchain in the Vulkan side, and its HIP-
    /// imported version.
    VkSemaphore            simulation_finished;
    hipExternalSemaphore_t hip_simulation_finished;

    /// The time at which this example started.
    std::chrono::high_resolution_clock::time_point start_time;

    /// Counters used to keep track of the current performance.
    uint32_t                                       fps_start_frame = 0;
    std::chrono::high_resolution_clock::time_point fps_start_time;

    /// \brief Initialize a new renderer.
    renderer(const graphics_context& ctx, swapchain& sc, const hipDevice_t hip_device)
        : ctx(ctx), sc(sc), hip_device(hip_device)
    {
        // Create a HIP stream for the (hip) device that was selected, which compute commands will be scheduled to later.
        HIP_CHECK(hipSetDevice(this->hip_device));
        HIP_CHECK(hipStreamCreate(&this->hip_stream));

        // Initialize the Vulkan resources related to this renderer.
        this->render_pass     = sc.create_render_pass();
        this->pipeline_layout = this->ctx.create_pipeline_layout();
        this->create_pipeline();

        this->frames.reserve(max_frames_in_flight);
        for(size_t i = 0; i < max_frames_in_flight; ++i)
        {
            this->frames.emplace_back(ctx);
        }

        this->sc.recreate_framebuffers(this->render_pass, this->framebuffers);

        // Create each of the buffers, and allocate memory for them.

        this->grid_buffer
            = create_buffer(ctx,
                            grid_buffer_size,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        // This buffer is going to be exported to HIP, so we should create it as
        // an external buffer.
        this->height_buffer
            = create_buffer(ctx,
                            height_buffer_size,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            true);

        this->index_buffer
            = create_buffer(ctx,
                            index_buffer_size,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        // Allocate the memory for each buffer.

        this->grid_memory
            = allocate_buffer_memory(ctx, this->grid_buffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        // Allocate this memory in a way that supports exporting.
        this->height_memory = allocate_buffer_memory(ctx,
                                                     this->height_buffer,
                                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                     true);
        this->index_memory
            = allocate_buffer_memory(ctx, this->index_buffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Upload the initial data to the buffers.
        this->initialize_buffer_data();

        // Export the height buffer and import it in HIP.
        this->hip_height_memory = memory_to_hip(this->ctx, this->height_memory, height_buffer_size);
        // Map it into memory.
        this->hip_height_buffer = reinterpret_cast<float*>(
            map_hip_external_memory(this->hip_height_memory, height_buffer_size));

        // Create the Vulkan-HIP synchronization resources from Vulkan and import them in HIP.
#if USE_EXTERNAL_SEMAPHORES == 1
        this->buffer_ready     = create_semaphore(this->ctx, true);
        this->hip_buffer_ready = semaphore_to_hip(this->ctx, this->buffer_ready);

        this->simulation_finished     = create_semaphore(this->ctx, true);
        this->hip_simulation_finished = semaphore_to_hip(this->ctx, this->simulation_finished);
#endif

        // Initialize performance counters.
        this->start_time     = std::chrono::high_resolution_clock::now();
        this->fps_start_time = this->start_time;
    }

    ~renderer()
    {
        // Be sure that rendering is finished
        this->wait_all_frames();

        // Make sure that all work has been finished before destroying the stream.
        HIP_CHECK(hipStreamSynchronize(this->hip_stream));
        HIP_CHECK(hipStreamDestroy(this->hip_stream));

        // Destroy Vulkan-HIP synchronization resources.
#if USE_EXTERNAL_SEMAPHORES == 1
        HIP_CHECK(hipDestroyExternalSemaphore(this->hip_buffer_ready));
        HIP_CHECK(hipDestroyExternalSemaphore(this->hip_simulation_finished));

        this->ctx.vkd->destroy_semaphore(this->ctx.dev, this->buffer_ready, nullptr);
        this->ctx.vkd->destroy_semaphore(this->ctx.dev, this->simulation_finished, nullptr);
#endif

        // Destroy the HIP external memory handle. We don't need to unmap it.
        HIP_CHECK(hipDestroyExternalMemory(this->hip_height_memory));

        // Destroy Vulkan device memory & buffer handles.
        this->ctx.vkd->free_memory(this->ctx.dev, this->index_memory, nullptr);
        this->ctx.vkd->free_memory(this->ctx.dev, this->height_memory, nullptr);
        this->ctx.vkd->free_memory(this->ctx.dev, this->grid_memory, nullptr);
        this->ctx.vkd->destroy_buffer(this->ctx.dev, this->index_buffer, nullptr);
        this->ctx.vkd->destroy_buffer(this->ctx.dev, this->height_buffer, nullptr);
        this->ctx.vkd->destroy_buffer(this->ctx.dev, this->grid_buffer, nullptr);

        this->ctx.vkd->destroy_pipeline_layout(this->ctx.dev, this->pipeline_layout, nullptr);
        this->ctx.vkd->destroy_pipeline(this->ctx.dev, this->pipeline, nullptr);

        for(const VkFramebuffer fb : this->framebuffers)
        {
            this->ctx.vkd->destroy_framebuffer(this->ctx.dev, fb, nullptr);
        }

        this->ctx.vkd->destroy_render_pass(this->ctx.dev, this->render_pass, nullptr);
    }

    renderer(const renderer&)            = delete;
    renderer& operator=(const renderer&) = delete;

    renderer(renderer&&)            = delete;
    renderer& operator=(renderer&&) = delete;

    /// \brief Block until all current frames have finished rendering.
    void wait_all_frames()
    {
        for(const frame& frame : this->frames)
        {
            frame.wait();
        }
    }

    /// \brief Upload the initial values for each buffer to Vulkan.
    void initialize_buffer_data()
    {
        // Create a "staging" buffer that is accessible from the CPU, that we will be using to
        // upload data to. We can re-use the same staging buffer for all three buffers, so create it
        // so that it is able to hold the maximum size of all three buffers.
        constexpr size_t staging_buffer_size = std::max(grid_buffer_size, index_buffer_size);
        VkBuffer         staging_buffer
            = create_buffer(ctx, staging_buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        VkDeviceMemory staging_memory = allocate_buffer_memory(
            ctx,
            staging_buffer,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // Map the staging buffer into host memory.
        void* staging;
        VK_CHECK(
            this->ctx.vkd
                ->map_memory(this->ctx.dev, staging_memory, 0, staging_buffer_size, 0, &staging));

        // Initialize the height buffer
        {
            std::memset(staging, 0, height_buffer_size);
            this->ctx.copy_buffer(this->height_buffer, staging_buffer, height_buffer_size);
        }

        // Initialize the grid buffer
        {
            float* grid = reinterpret_cast<float*>(staging);
            for(uint32_t y = 0; y < grid_height; ++y)
            {
                for(uint32_t x = 0; x < grid_width; ++x)
                {
                    *grid++ = (2.0f * x) / (grid_width - 1) - 1;
                    *grid++ = (2.0f * y) / (grid_height - 1) - 1;
                }
            }

            this->ctx.copy_buffer(this->grid_buffer, staging_buffer, grid_buffer_size);
        }

        // Initialize the index buffer
        {
            uint32_t* indices = reinterpret_cast<uint32_t*>(staging);
            for(uint32_t y = 0; y < grid_height - 1; ++y)
            {
                for(uint32_t x = 0; x < grid_width - 1; ++x)
                {
                    *indices++ = (y + 0) * grid_width + (x + 0);
                    *indices++ = (y + 1) * grid_width + (x + 0);
                    *indices++ = (y + 0) * grid_width + (x + 1);
                    *indices++ = (y + 1) * grid_width + (x + 0);
                    *indices++ = (y + 1) * grid_width + (x + 1);
                    *indices++ = (y + 0) * grid_width + (x + 1);
                }
            }

            this->ctx.copy_buffer(this->index_buffer, staging_buffer, index_buffer_size);
        }

        // We are done with the staging buffer so clean it up.
        this->ctx.vkd->unmap_memory(this->ctx.dev, staging_memory);
        this->ctx.vkd->free_memory(this->ctx.dev, staging_memory, nullptr);
        this->ctx.vkd->destroy_buffer(this->ctx.dev, staging_buffer, nullptr);
    }

    /// \brief Initialize the Vulkan pipeline for the renderer.
    void create_pipeline()
    {
        VkShaderModule vert
            = create_shader_module(this->ctx, std::size(sinewave_vert), sinewave_vert);
        VkShaderModule frag
            = create_shader_module(this->ctx, std::size(sinewave_frag), sinewave_frag);

        // Keep in sync with shaders!
        VkPipelineShaderStageCreateInfo pssci[2] = {};
        pssci[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pssci[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        pssci[0].module = vert;
        pssci[0].pName  = "main";
        pssci[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pssci[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        pssci[1].module = frag;
        pssci[1].pName  = "main";

        // Keep in sync with shaders!
        VkVertexInputBindingDescription bindings[2] = {};
        bindings[0].binding                         = 0;
        bindings[0].stride                          = sizeof(float);
        bindings[0].inputRate                       = VK_VERTEX_INPUT_RATE_VERTEX;
        bindings[1].binding                         = 1;
        bindings[1].stride                          = sizeof(float) * 2;
        bindings[1].inputRate                       = VK_VERTEX_INPUT_RATE_VERTEX;

        // Keep in sync with shaders!
        VkVertexInputAttributeDescription attribs[2] = {};
        attribs[0].binding                           = 0;
        attribs[0].location                          = 0;
        attribs[0].format                            = VK_FORMAT_R32_SFLOAT;
        attribs[1].binding                           = 1;
        attribs[1].location                          = 1;
        attribs[1].format                            = VK_FORMAT_R32G32_SFLOAT;

        this->pipeline = this->ctx.create_simple_pipeline(this->pipeline_layout,
                                                          this->render_pass,
                                                          pssci,
                                                          std::size(pssci),
                                                          bindings,
                                                          std::size(bindings),
                                                          attribs,
                                                          std::size(attribs));

        // Shader modules do not need to be kept around in memory.
        this->ctx.vkd->destroy_shader_module(this->ctx.dev, vert, nullptr);
        this->ctx.vkd->destroy_shader_module(this->ctx.dev, frag, nullptr);
    }

    /// \brief Re-create the backing swapchain and re-initialize frame buffers if the swapchain
    /// has become outdated.
    bool recreate_swapchain(GLFWwindow* const window)
    {
        VK_CHECK(this->ctx.vkd->queue_wait_idle(this->ctx.present_queue.queue));
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if(width == 0 || height == 0)
        {
            return false;
        }

        this->sc.recreate({static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
        this->sc.recreate_framebuffers(this->render_pass, this->framebuffers);

        return true;
    }

    /// \brief Start rendering the next frame
    /// \returns if the frame can be rendered at all. This may not be the case on
    /// some operating systems for example if the window is minimized and has a
    /// surface extent of 0 by 0 pixels.
    bool begin_frame(GLFWwindow* const window)
    {
        const frame& frame = frames[this->frame_index % this->frames.size()];
        // Wait until the previous instance of this frame is done rendering.
        frame.wait();

        // Acquire the next image index from the swapchain.
        // Re-create the swapchain if it has become outdated in the meantime.
        if(this->swapchain_out_of_date)
        {
            if(!this->recreate_swapchain(window))
                return false;
            this->swapchain_out_of_date = false;
        }

        const swapchain::present_state present_state
            = this->sc.acquire_next_image(frame.image_acquired, frame_timeout);
        switch(present_state)
        {
            case swapchain::present_state::optimal: break;
            case swapchain::present_state::suboptimal:
                // Sub-optimal, but semaphore is already signaled.
                // Continue rendering this frame and re-create on the next.
                this->swapchain_out_of_date = true;
                break;
            case swapchain::present_state::out_of_date:
                // Need to re-create immediately.
                this->swapchain_out_of_date = true;
                return false;
        }

        // Reset the fence backing the frame now that we are creating work.
        frame.reset();

        // Reset the command pool and initialize the command buffer so that we can start submitting
        // draw commands to it.
        VK_CHECK(this->ctx.vkd->reset_command_pool(this->ctx.dev, frame.cmd_pool, 0));
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(this->ctx.vkd->begin_command_buffer(frame.cmd_buf, &begin_info));

        return true;
    }

    /// \brief End the current frame and submit it to the graphics queue for rendering and the
    /// present queue for presenting.
    void end_frame()
    {
        const frame& frame = frames[this->frame_index % this->frames.size()];

        VK_CHECK(this->ctx.vkd->end_command_buffer(frame.cmd_buf));

        // The semaphores that we need to wait on before this frame can be rendered completely:
        // - The frame needs to wait before the image is completely acquired from Vulkan. In
        //   vkAcquireNextImageKHR the implementation may already know _which_ image is going to
        //   be rendered to next, but it may not be quite ready for it yet. This is why we need
        //   to wait on it here.
        // - HIP needs to be finished with the height buffer, and so it also need to wait on the
        //   semaphore that signals that its ready.
#if USE_EXTERNAL_SEMAPHORES == 1 && USE_SIGNAL_SEMAPHORE == 1
        VkSemaphore wait_semaphores[] = {frame.image_acquired, this->simulation_finished};
#else
        VkSemaphore wait_semaphores[] = {frame.image_acquired};
#endif

        // The pipeline stage at which each of the corresponding \p wait_semaphores need to be
        // waited upon. This allows Vulkan to start with some rendering processes even though
        // the semaphores are not yet signaled:
        // - We only need the swapchain image when we are actually going to draw to it, we can
        //   already perform the vertex shader for example and the fragment shader to some extent
        //   before the output is actually drawn to the swap image.
        // - The buffer passed to HIP is used for vertex coordinates during when drawing in Vulkan,
        //   so that buffer needs to be finished (and its associated \p simulation_finished semaphore
        //   needs to be signaled) when we vertex inputs are bound.
        const VkPipelineStageFlags wait_dst_stage_masks[]
            = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT};

        // The semaphores that need to be signaled after this step is finished:
        // - The \p render_finished semaphore allows us to guard the time between when the rendering
        //   commands are finished (and so when the result is on the swapchain image) and when it can
        //   be copied to the GLFW window.
        // - The \p buffer_ready semaphore signals that the rendering process is finished, and that we
        //   can perform the next step of the simulation. This prevents that HIP is already modifying the
        //   buffer while Vulkan has not completely rendered it to the swapchain image.
#if USE_EXTERNAL_SEMAPHORES == 1
        VkSemaphore signal_semaphores[] = {frame.render_finished, this->buffer_ready};
#else
        VkSemaphore signal_semaphores[] = {frame.render_finished};
#endif

        // Submit the current frame's command buffer to the GPU.
        VkSubmitInfo submit_info         = {};
        submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount   = std::size(wait_semaphores);
        submit_info.pWaitSemaphores      = wait_semaphores;
        submit_info.pWaitDstStageMask    = wait_dst_stage_masks;
        submit_info.signalSemaphoreCount = std::size(signal_semaphores);
        submit_info.pSignalSemaphores    = signal_semaphores;
        submit_info.commandBufferCount   = 1;
        submit_info.pCommandBuffers      = &frame.cmd_buf;
        VK_CHECK(this->ctx.vkd->queue_submit(this->ctx.graphics_queue.queue,
                                             1,
                                             &submit_info,
                                             frame.frame_fence));

        // Then finally ask the swapchain to draw the current image to the GLFW window, when rendering
        // is finished.
        const swapchain::present_state present_state = this->sc.present(frame.render_finished);
        if(present_state != swapchain::present_state::optimal)
            this->swapchain_out_of_date = true;

        ++this->frame_index;
    }

    /// \brief This function updates the height buffer with new coordinates.
    void step_simulation()
    {
        // Take care that we are not going to modify the buffer before it is ready.
#if USE_EXTERNAL_SEMAPHORES == 1
        // If semaphores are supported and used, we need to wait on it so that it is
        // certain that Vulkan is no longer using the buffer.
        // Note: This semaphore is not signaled in the first frame, so we don't need to wait
        // on it then.
        if(this->frame_index != 0)
        {
            // [Sphinx wait semaphore start]
            hipExternalSemaphoreWaitParams wait_params = {};
            HIP_CHECK(hipWaitExternalSemaphoresAsync(&this->hip_buffer_ready,
                                                     &wait_params,
                                                     1,
                                                     this->hip_stream));
            // [Sphinx wait semaphore end]
        }
#else
        // If semaphores are not supported or not used, then we need to perform a full queue
        // sync to be sure that Vulkan is not using the buffer anymore.
        VK_CHECK(this->ctx.vkd->queue_wait_idle(this->ctx.graphics_queue.queue));
#endif

        const auto  now = std::chrono::high_resolution_clock::now();
        const float time
            = std::chrono::duration<float, std::chrono::seconds::period>(now - this->start_time)
                  .count();

        // The tile size to be used for each block of the computation. A tile is
        // tile_size by tile_size threads in this case, since we are invoking the
        // computation over a 2D-grid.
        constexpr size_t tile_size = 8;

        // [Sphinx kernel call start]
        // Launch the HIP kernel to advance the simulation.
        sinewave_kernel<<<dim3(ceiling_div(grid_width, tile_size),
                               ceiling_div(grid_height, tile_size)),
                          dim3(tile_size, tile_size),
                          0,
                          this->hip_stream>>>(this->hip_height_buffer, time);
        HIP_CHECK(hipGetLastError());
        // [Sphinx kernel call end]

        // Signal to Vulkan that we are done with the buffer and that it can proceed
        // with rendering.
#if USE_EXTERNAL_SEMAPHORES == 1 && USE_SIGNAL_SEMAPHORE == 1
        // If semaphores are supported and used, signal the semaphore that indicates
        // that the simulation has finished.
        // [Sphinx signal semaphore start]
        hipExternalSemaphoreSignalParams signal_params = {};
        HIP_CHECK(hipSignalExternalSemaphoresAsync(&this->hip_simulation_finished,
                                                   &signal_params,
                                                   1,
                                                   this->hip_stream));
        // [Sphinx signal semaphore end]
#else
        // If semaphores are not used or not supported, we need to again perform a full
        // queue sync from the HIP side this time.
        HIP_CHECK(hipStreamSynchronize(this->hip_stream));
#endif
    }

    /// \brief Draw the next frame to the window.
    void draw(GLFWwindow* const window)
    {
        if(!this->begin_frame(window))
            return;

        // Advance the simulation on the HIP side.
        this->step_simulation();

        // Render the grid to the screen from the Vulkan side.
        const frame&          frame   = frames[this->frame_index % this->frames.size()];
        const VkCommandBuffer cmd_buf = frame.cmd_buf;

        // Initialize the rendering pass
        VkClearValue clear_color = {};

        VkViewport viewport = {};
        viewport.width      = this->sc.extent.width;
        viewport.height     = this->sc.extent.height;
        viewport.minDepth   = 0;
        viewport.maxDepth   = 1;

        VkRect2D scissor = {};
        scissor.extent   = this->sc.extent;

        const device_dispatch& vkd = *this->ctx.vkd;

        vkd.cmd_set_viewport(cmd_buf, 0, 1, &viewport);
        vkd.cmd_set_scissor(cmd_buf, 0, 1, &scissor);

        VkRenderPassBeginInfo rp_begin_info = {};
        rp_begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_begin_info.renderPass            = this->render_pass;
        rp_begin_info.framebuffer           = this->framebuffers[this->sc.image_index];
        rp_begin_info.renderArea            = scissor;
        rp_begin_info.clearValueCount       = 1;
        rp_begin_info.pClearValues          = &clear_color;
        vkd.cmd_begin_render_pass(cmd_buf, &rp_begin_info, VK_SUBPASS_CONTENTS_INLINE);

        // Bind the pipeline that we are using to render with.
        vkd.cmd_bind_pipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, this->pipeline);

        VkBuffer     vertex_buffers[] = {this->height_buffer, this->grid_buffer};
        VkDeviceSize offsets[]        = {0, 0};
        vkd.cmd_bind_vertex_buffers(cmd_buf, 0, std::size(vertex_buffers), vertex_buffers, offsets);
        vkd.cmd_bind_index_buffer(cmd_buf, this->index_buffer, 0, VK_INDEX_TYPE_UINT32);

        // Draw the triangles.
        vkd.cmd_draw_indexed(cmd_buf, num_indices, 1, 0, 0, 0);

        vkd.cmd_end_render_pass(cmd_buf);

        this->end_frame();

        // Output a native performance measurement.
        const auto frame_time = std::chrono::high_resolution_clock::now();
        const auto time_diff  = frame_time - this->fps_start_time;
        if(time_diff > std::chrono::seconds{5})
        {
            const auto time_diff_sec
                = std::chrono::duration_cast<std::chrono::duration<float>>(time_diff).count();
            const uint32_t frames = this->frame_index - this->fps_start_frame;
            std::cout << "Average FPS (over " << double_precision(time_diff_sec, 2, true)
                      << " seconds): " << double_precision(frames / time_diff_sec, 2, true) << " ("
                      << double_precision((time_diff_sec * 1000) / frames, 2, true)
                      << " ms per frame)" << std::endl;
            this->fps_start_frame = this->frame_index;
            this->fps_start_time  = frame_time;
        }
    }
};

/// \brief GLFW window resize callback: If the window is resized then we need to re-create the
/// swapchain on the next frame.
void resize_callback(GLFWwindow* const window, const int, const int)
{
    renderer* r              = reinterpret_cast<renderer*>(glfwGetWindowUserPointer(window));
    r->swapchain_out_of_date = true;
}

/// \brief Program entry point.
int main()
{
    // The initial size of the GLFW window when the example is first started.
    constexpr VkExtent2D initial_window_extent = {1280, 800};

    // Initialize GLFW.
    glfwSetErrorCallback(
        [](int code, const char* const message)
        { std::cerr << "A glfw error encountered: " << message << "(" << code << ")\n"; });

    if(glfwInit() != GLFW_TRUE)
    {
        std::cerr << "failed to initialize GLFW\n";
        return error_exit_code;
    }

    // Initialize the window.
    VkApplicationInfo app_info  = {};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = "HIP-Vulkan interop example";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName        = "rocm-examples";
    app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion         = VK_MAKE_VERSION(1, 0, 0);

    GLFWwindow* window = create_window(app_info, initial_window_extent);

    // Create the base Vulkan types: Load base function pointers, create instance, load
    // instance function pointers, create the surface.
    const auto         vkb      = std::make_unique<base_dispatch>(glfwGetInstanceProcAddress);
    const VkInstance   instance = create_instance(*vkb,
                                                app_info,
                                                required_instance_extensions,
                                                std::size(required_instance_extensions));
    const auto         vki      = std::make_unique<instance_dispatch>(*vkb, instance);
    const VkSurfaceKHR surface  = create_surface(instance, window);

    // Try to find a physical device that we can use for this example.
    physical_device_candidate candidate;
    find_physical_device(*vki, instance, surface, candidate);

    const hipDevice_t hip_device = candidate.hip_candidate.device;

    // Let the user know which device we are using, on both the Vulkan and HIP sides.
    hipDeviceProp_t hip_props;
    HIP_CHECK(hipGetDeviceProperties(&hip_props, hip_device));

    std::cout << "Using device " << candidate.props.deviceName << " (hip device " << hip_device
              << ", UUID " << candidate.hip_candidate.device_uuid << ", compute capability "
              << hip_props.major << "." << hip_props.minor << ")\n";

    {
        // Initialize the rendering resources, both the Vulkan and HIP ones.
        // These are defined in a sub-scope so that the destructors are
        // invoked before we call `glfwDestroyWindow` and `glfwTerminate`.
        graphics_context ctx(vki.get(),
                             instance,
                             surface,
                             candidate.pdev,
                             candidate.queues,
                             required_device_extensions,
                             std::size(required_device_extensions));

        swapchain swapchain(ctx, initial_window_extent);
        renderer  renderer(ctx, swapchain, hip_device);

        glfwSetWindowUserPointer(window, reinterpret_cast<void*>(&renderer));
        glfwSetFramebufferSizeCallback(window, resize_callback);

        // The main rendering loop.
        // Repeat for as long as the window is not closed.
        while(glfwWindowShouldClose(window) == GLFW_FALSE)
        {
            renderer.draw(window);
            glfwPollEvents();
        }

        glfwSetFramebufferSizeCallback(window, nullptr);
        glfwSetWindowUserPointer(window, nullptr);
    }

    // Destroy the surface and instance now that we are done with them.
    vki->destroy_surface(instance, surface, nullptr);
    vki->destroy_instance(instance, nullptr);

    // Clean up GLFW.
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
