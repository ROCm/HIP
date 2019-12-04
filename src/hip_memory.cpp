/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include <hc_am.hpp>
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"

#include <functional>
#include <fstream>

__device__ char __hip_device_heap[__HIP_SIZE_OF_HEAP];
__device__ uint32_t __hip_device_page_flag[__HIP_NUM_PAGES];

// Internal HIP APIS:
namespace hip_internal {

namespace {
    inline
    const char* hsa_to_string(hsa_status_t err) noexcept
    {
        const char* r{};

        if (hsa_status_string(err, &r) != HSA_STATUS_SUCCESS) return r;

        return "Unknown.";
    }

    template<std::size_t m, std::size_t n>
    inline
    void throwing_result_check(hsa_status_t res, const char (&file)[m],
                               const char (&function)[n], int line) {
        if (res == HSA_STATUS_SUCCESS) return;
        if (res == HSA_STATUS_INFO_BREAK) return;

        throw std::runtime_error{"Failed in file " + (file +
                                 (", in function \"" + (function +
                                 ("\", on line " + std::to_string(line))))) +
                                 ", with error: " + hsa_to_string(res)};
    }

    inline
    hsa_agent_t cpu_agent() noexcept {
        static hsa_agent_t cpu{[]() {
            hsa_agent_t r{};
            throwing_result_check(
                hsa_iterate_agents([](hsa_agent_t x, void* pr) {
                    hsa_device_type_t t{};
                    hsa_agent_get_info(x, HSA_AGENT_INFO_DEVICE, &t);

                    if (t != HSA_DEVICE_TYPE_CPU) return HSA_STATUS_SUCCESS;

                    *static_cast<hsa_agent_t *>(pr) = x;

                    return HSA_STATUS_INFO_BREAK;
                }, &r), __FILE__, __func__, __LINE__);

            return r;
        }()};

        return cpu;
    }

    inline
    hsa_device_type_t type(hsa_agent_t x) noexcept
    {
        hsa_device_type_t r{};
        throwing_result_check(hsa_agent_get_info(x, HSA_AGENT_INFO_DEVICE, &r),
                              __FILE__, __func__, __LINE__);

        return r;
    }

    const auto is_large_BAR{[](){
        std::unique_ptr<void, void (*)(void*)> hsa{
            (throwing_result_check(hsa_init(), __FILE__, __func__, __LINE__),
             nullptr),
            [](void*) { hsa_shut_down(); }};
        bool r{true};

        throwing_result_check(hsa_iterate_agents([](hsa_agent_t x, void* pr) {
            if (x.handle == cpu_agent().handle) return HSA_STATUS_SUCCESS;

            throwing_result_check(
                hsa_agent_iterate_regions(x, [](hsa_region_t y, void* p) {
                    hsa_region_segment_t seg{};
                    throwing_result_check(
                        hsa_region_get_info(y, HSA_REGION_INFO_SEGMENT, &seg),
                        __FILE__, __func__, __LINE__);

                    if (seg != HSA_REGION_SEGMENT_GLOBAL) {
                        return HSA_STATUS_SUCCESS;
                    }

                    uint32_t flags{};
                    throwing_result_check(hsa_region_get_info(
                        y, HSA_REGION_INFO_GLOBAL_FLAGS, &flags),
                        __FILE__, __func__, __LINE__);

                    if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
                        hsa_amd_memory_pool_access_t tmp{};
                        throwing_result_check(
                            hsa_amd_agent_memory_pool_get_info(
                                cpu_agent(),
                                hsa_amd_memory_pool_t{y.handle},
                                HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                &tmp),
                            __FILE__, __func__, __LINE__);

                        *static_cast<bool*>(p) &=
                            tmp != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
                    }

                    return HSA_STATUS_SUCCESS;
                }, pr), __FILE__, __func__, __LINE__);

            return HSA_STATUS_SUCCESS;
        }, &r), __FILE__, __func__, __LINE__);

        return r;
    }()};

    inline
    hsa_amd_pointer_info_t info(const void* p) noexcept
    {
        hsa_amd_pointer_info_t r{sizeof(hsa_amd_pointer_info_t)};
        throwing_result_check(
            hsa_amd_pointer_info(
                const_cast<void*>(p), &r, nullptr, nullptr, nullptr),
            __FILE__, __func__, __LINE__);

        r.size = is_large_BAR || (type(r.agentOwner) == HSA_DEVICE_TYPE_CPU) ?
            UINT32_MAX : sizeof(hsa_amd_pointer_info_t);

        return r;
    }

    constexpr size_t staging_sz{4 * 1024 * 1024}; // 2 Pages.

    thread_local const std::unique_ptr<void, void (*)(void *)> staging_buffer{
        []() {
            hsa_region_t r{};
            throwing_result_check(hsa_agent_iterate_regions(
                cpu_agent(), [](hsa_region_t x, void *p) {
                hsa_region_segment_t seg{};
                throwing_result_check(
                    hsa_region_get_info(x, HSA_REGION_INFO_SEGMENT, &seg),
                    __FILE__, __func__, __LINE__);

                if (seg != HSA_REGION_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

                uint32_t flags{};
                throwing_result_check(hsa_region_get_info(
                    x, HSA_REGION_INFO_GLOBAL_FLAGS, &flags),
                    __FILE__, __func__, __LINE__);

                if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
                    *static_cast<hsa_region_t *>(p) = x;

                    return HSA_STATUS_INFO_BREAK;
                }

                return HSA_STATUS_SUCCESS;
            }, &r), __FILE__, __func__, __LINE__);

            void *tp{};
            throwing_result_check(hsa_memory_allocate(r, staging_sz, &tp),
                                  __FILE__, __func__, __LINE__);

            return tp;
        }(),
        [](void *ptr) { hsa_memory_free(ptr); }};

    thread_local hsa_signal_t copy_signal{[]() {
        hsa_agent_t cpu{cpu_agent()};
        hsa_signal_t sgn{};
        throwing_result_check(hsa_signal_create(1, 1, &cpu, &sgn),
                              __FILE__, __func__, __LINE__);

        return sgn;
    }()};
} // Unnamed namespace.

inline
void do_copy(void* __restrict dst, const void* __restrict src, std::size_t n,
             hsa_agent_t da, hsa_agent_t sa) {
    hsa_signal_silent_store_relaxed(copy_signal, 1);
    throwing_result_check(
        hsa_amd_memory_async_copy(dst, da, src, sa, n, 0, nullptr, copy_signal),
        __FILE__, __func__, __LINE__);

    while (hsa_signal_wait_relaxed(copy_signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                   UINT64_MAX, HSA_WAIT_STATE_ACTIVE));
}

inline
void do_std_memcpy(
    void* __restrict dst, const void* __restrict src, std::size_t n) {
    std::memcpy(dst, src, n);

    return std::atomic_thread_fence(std::memory_order_seq_cst);
}

inline
void d2h_copy(void* __restrict dst, const void* __restrict src, size_t n,
              hsa_amd_pointer_info_t si) {
    if (si.size == UINT32_MAX) return do_std_memcpy(dst, src, n);

    const auto di{info(dst)};

    if (di.type == HSA_EXT_POINTER_TYPE_LOCKED) {
        dst = static_cast<char*>(di.agentBaseAddress) +
              (static_cast<char*>(dst) -
               static_cast<char*>(di.hostBaseAddress));
        do_copy(dst, src, n, si.agentOwner, si.agentOwner);
    }
    else if (n <= staging_sz) {
        do_copy(staging_buffer.get(), src, n, si.agentOwner, si.agentOwner);
        std::memcpy(dst, staging_buffer.get(), n);
    }
    else {
        std::unique_ptr<void, void (*)(void*)> lck{
            dst, [](void* p) { hsa_amd_memory_unlock(p); }};

        throwing_result_check(hsa_amd_memory_lock(dst, n, &si.agentOwner, 1,
                                                  const_cast<void**>(&dst)),
            __FILE__, __func__, __LINE__);

        do_copy(dst, src, n, si.agentOwner, si.agentOwner);
    }
}

inline
void h2d_copy(void* __restrict dst, const void* __restrict src, size_t n,
              hsa_amd_pointer_info_t di) {
    if (di.size == UINT32_MAX) return do_std_memcpy(dst, src, n);

    const auto si{info(const_cast<void*>(src))};

    if (si.type == HSA_EXT_POINTER_TYPE_LOCKED) {
        src = static_cast<char*>(si.agentBaseAddress) +
             (static_cast<const char*>(src) -
              static_cast<char*>(si.hostBaseAddress));
        do_copy(dst, src, n, di.agentOwner, di.agentOwner);
    }
    else if (n <= staging_sz) {
        std::memcpy(staging_buffer.get(), src, n);
        do_copy(dst, staging_buffer.get(), n, di.agentOwner, di.agentOwner);
    }
    else {
        std::unique_ptr<void, void (*)(void*)> lck{
            const_cast<void*>(src), [](void* p) { hsa_amd_memory_unlock(p); }};

        throwing_result_check(hsa_amd_memory_lock(const_cast<void*>(src), n,
                                                  &di.agentOwner, 1,
                                                  const_cast<void**>(&src)),
            __FILE__, __func__, __LINE__);

        do_copy(dst, src, n, di.agentOwner, di.agentOwner);
    }
}

inline
void generic_copy(void* __restrict dst, const void* __restrict src, size_t n,
                  hsa_amd_pointer_info_t di, hsa_amd_pointer_info_t si) {
    if (di.size == UINT32_MAX && si.size == UINT32_MAX) {
        return do_std_memcpy(dst, src, n);
    }

    std::unique_ptr<void, void (*)(void*)> lck0{
        nullptr, [](void* p) { hsa_amd_memory_unlock(p); }};
    std::unique_ptr<void, void (*)(void*)> lck1{nullptr, lck0.get_deleter()};

    switch (si.type) {
    case HSA_EXT_POINTER_TYPE_HSA:
        if (di.type == HSA_EXT_POINTER_TYPE_HSA) {
            hsa_memory_copy(dst, src, n);
            return; // TODO: do_copy(dst, src, n, di.agentOwner, si.agentOwner);
        }

        if (di.type == HSA_EXT_POINTER_TYPE_UNKNOWN ||
            di.type == HSA_EXT_POINTER_TYPE_LOCKED) {
            return d2h_copy(dst, src, n, si);
        }
        break;
    case HSA_EXT_POINTER_TYPE_LOCKED:
        if (di.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
            std::memcpy(dst, si.hostBaseAddress, n);

            return;
        }
        if (di.type == HSA_EXT_POINTER_TYPE_LOCKED) {
            std::memcpy(di.hostBaseAddress, si.hostBaseAddress, n);

            return;
        }
        src = si.agentBaseAddress;
        si.agentOwner = di.agentOwner;
        break;
    case HSA_EXT_POINTER_TYPE_UNKNOWN:
        if (di.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
            std::memcpy(dst, src, n);

            return;
        }
        if (di.type == HSA_EXT_POINTER_TYPE_LOCKED) {
            std::memcpy(di.hostBaseAddress, src, n);

            return;
        }
        return h2d_copy(dst, src, n, di);
    default: do_copy(dst, src, n, di.agentOwner, si.agentOwner); break;
    }
}

inline
void memcpy_impl(void* __restrict dst, const void* __restrict src, size_t n,
                 hipMemcpyKind k) noexcept {
    switch (k) {
    case hipMemcpyHostToHost: std::memcpy(dst, src, n); break;
    case hipMemcpyHostToDevice:
        return is_large_BAR ? do_std_memcpy(dst, src, n)
                            : h2d_copy(dst, src, n, info(dst));
    case hipMemcpyDeviceToHost:
        return is_large_BAR ? do_std_memcpy(dst, src, n)
                            : d2h_copy(dst, src, n, info(src));
    case hipMemcpyDeviceToDevice:
        return do_copy(dst, src, n, info(dst).agentOwner,
                       info(const_cast<void*>(src)).agentOwner);
    default: return generic_copy(dst, src, n, info(dst), info(src));
    }
}

hipError_t memcpyAsync(void* dst, const void* src, size_t sizeBytes,
                       hipMemcpyKind kind, hipStream_t stream) {
    if (sizeBytes == 0) return hipSuccess;
    if (!dst || !src) return hipErrorInvalidValue;

    try {
        stream = ihipSyncAndResolveStream(stream);

        if (!stream) return hipErrorInvalidValue;

        stream->locked_copyAsync(dst, src, sizeBytes, kind);
    }
    catch (const ihipException& ex) {
        return ex._code;
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        throw;
    }
    catch (...) {
        return hipErrorUnknown;
    }

    return hipSuccess;
}

hipError_t memcpySync(void* dst, const void* src, size_t sizeBytes,
                      hipMemcpyKind kind, hipStream_t stream) {
    if (sizeBytes == 0) return hipSuccess;
    if (!dst || !src) return hipErrorInvalidValue;

    try {
        stream = ihipSyncAndResolveStream(stream);

        if (!stream) return hipErrorInvalidValue;

        LockedAccessor_StreamCrit_t cs{stream->criticalData()};
        cs->_av.wait();

        memcpy_impl(dst, src, sizeBytes, kind);
    }
    catch (const ihipException& ex) {
        return ex._code;
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        throw;
    }
    catch (...) {
        return hipErrorUnknown;
    }

    return hipSuccess;
}

// return 0 on success or -1 on error:
int sharePtr(void* ptr, ihipCtx_t* ctx, bool shareWithAll, unsigned hipFlags) {
    int ret = 0;

    auto device = ctx->getWriteableDevice();

    if (shareWithAll) {
        // shareWithAll memory is not mapped to any device
        hc::am_memtracker_update(ptr, -1, hipFlags);
        hsa_status_t s = hsa_amd_agents_allow_access(g_deviceCnt + 1, g_allAgents, NULL, ptr);
        tprintf(DB_MEM, "    allow access to CPU + all %d GPUs (shareWithAll)\n", g_deviceCnt);
        if (s != HSA_STATUS_SUCCESS) {
            ret = -1;
        }
    } else {
#if USE_APP_PTR_FOR_CTX
        hc::am_memtracker_update(ptr, device->_deviceId, hipFlags, ctx);
#else
        hc::am_memtracker_update(ptr, device->_deviceId, hipFlags);
#endif
        int peerCnt = 0;
        {
            LockedAccessor_CtxCrit_t crit(ctx->criticalData());
            // the peerCnt always stores self so make sure the trace actually
            peerCnt = crit->peerCnt();
            tprintf(DB_MEM, "  allow access to %d other peer(s)\n", peerCnt - 1);
            if (peerCnt > 1) {
                // printf ("peer self access\n");

                // TODOD - remove me:
                for (auto iter = crit->_peers.begin(); iter != crit->_peers.end(); iter++) {
                    tprintf(DB_MEM, "    allow access to peer: %s%s\n", (*iter)->toString().c_str(),
                            (iter == crit->_peers.begin()) ? " (self)" : "");
                };

                hsa_status_t s =
                    hsa_amd_agents_allow_access(crit->peerCnt(), crit->peerAgents(), NULL, ptr);
                if (s != HSA_STATUS_SUCCESS) {
                    ret = -1;
                }
            }
        }
    }

    return ret;
}


// Allocate a new pointer with am_alloc and share with all valid peers.
// Returns null-ptr if a memory error occurs (either allocation or sharing)
void* allocAndSharePtr(const char* msg, size_t sizeBytes, ihipCtx_t* ctx, bool shareWithAll,
                       unsigned amFlags, unsigned hipFlags, size_t alignment) {
    void* ptr = nullptr;

    auto device = ctx->getWriteableDevice();

#if (__hcc_workweek__ >= 17332)
    if (alignment != 0) {
        ptr = hc::am_aligned_alloc(sizeBytes, device->_acc, amFlags, alignment);
    } else
#endif
    {
        ptr = hc::am_alloc(sizeBytes, device->_acc, amFlags);
    }
    tprintf(DB_MEM, " alloc %s ptr:%p-%p size:%zu on dev:%d\n", msg, ptr,
            static_cast<char*>(ptr) + sizeBytes, sizeBytes, device->_deviceId);

    if (HIP_INIT_ALLOC != -1) {
        // TODO , dont' call HIP API directly here:
        hipMemset(ptr, HIP_INIT_ALLOC, sizeBytes);
    }

    if (ptr != nullptr) {
        int r = sharePtr(ptr, ctx, shareWithAll, hipFlags);
        if (r != 0) {
            ptr = nullptr;
        }
    }

    return ptr;
}

hipError_t ihipHostMalloc(TlsData *tls, void** ptr, size_t sizeBytes, unsigned int flags) {
    hipError_t hip_status = hipSuccess;

    if (HIP_SYNC_HOST_ALLOC) {
        hipDeviceSynchronize();
    }

    auto ctx = ihipGetTlsDefaultCtx();
    if ((ctx == nullptr) || (ptr == nullptr)) {
        hip_status = hipErrorInvalidValue;
    }
    else if (sizeBytes == 0) {
        hip_status = hipSuccess;
        // TODO - should size of 0 return err or be siliently ignored?
    } else {
        unsigned trueFlags = flags;
        if (flags == hipHostMallocDefault) {
            // HCC/ROCM provide a modern system with unified memory and should set both of these
            // flags by default:
            trueFlags = hipHostMallocMapped | hipHostMallocPortable;
        }


        const unsigned supportedFlags = hipHostMallocPortable | hipHostMallocMapped |
                                        hipHostMallocWriteCombined | hipHostMallocCoherent |
                                        hipHostMallocNonCoherent;


        const unsigned coherencyFlags = hipHostMallocCoherent | hipHostMallocNonCoherent;

        if ((flags & ~supportedFlags) || ((flags & coherencyFlags) == coherencyFlags)) {
            *ptr = nullptr;
            // can't specify unsupported flags, can't specify both Coherent + NonCoherent
            hip_status = hipErrorInvalidValue;
        } else {
            auto device = ctx->getWriteableDevice();
#if (__hcc_workweek__ >= 19115)
            //Avoid mapping host pinned memory to all devices by HCC
            unsigned amFlags = amHostUnmapped;
#else
            unsigned amFlags = 0;
#endif
            if (flags & hipHostMallocCoherent) {
                amFlags |= amHostCoherent;
            } else if (flags & hipHostMallocNonCoherent) {
                amFlags |= amHostNonCoherent;
            } else {
                // depends on env variables:
                amFlags |= HIP_HOST_COHERENT ? amHostCoherent : amHostNonCoherent;
            }


            *ptr = hip_internal::allocAndSharePtr(
                (amFlags & amHostCoherent) ? "finegrained_host" : "pinned_host", sizeBytes, ctx,
                true  /*shareWithAll*/, amFlags, flags, 0);

            if (sizeBytes && (*ptr == NULL)) {
                hip_status = hipErrorMemoryAllocation;
            }
        }
    }

    if (HIP_SYNC_HOST_ALLOC) {
        hipDeviceSynchronize();
    }
    return hip_status;
}

hipError_t ihipHostFree(TlsData *tls, void* ptr) {

    // Synchronize to ensure all work has finished.
    ihipGetTlsDefaultCtx()->locked_waitAllStreams();  // ignores non-blocking streams, this waits
                                                      // for all activity to finish.

    hipError_t hipStatus = hipErrorInvalidValue;
    if (ptr) {
        hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if (status == AM_SUCCESS) {
            if (amPointerInfo._hostPointer == ptr) {
                hc::am_free(ptr);
                hipStatus = hipSuccess;
            }
        }
    } else {
        // free NULL pointer succeeds and is common technique to initialize runtime
        hipStatus = hipSuccess;
    }

    return hipStatus;
}


}  // end namespace hip_internal

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Memory
//
//
//
// HIP uses several "app*" fields HC memory tracker to track state necessary for the HIP API.
//_appId : DeviceID.  For device mem, this is device where the memory is physically allocated.
//         For host or registered mem, this is the current device when the memory is allocated or
//         registered.  This device will have a GPUVM mapping for the host mem.
//
//_appAllocationFlags : These are flags provided by the user when allocation is performed. They are
//returned to user in hipHostGetFlags and other APIs.
// TODO - add more info here when available.
//
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
    HIP_INIT_API(hipPointerGetAttributes, attributes, ptr);

    hipError_t e = hipSuccess;
    if ((attributes == nullptr) || (ptr == nullptr)) {
        e = hipErrorInvalidValue;
    } else {
        hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if (status == AM_SUCCESS) {
            attributes->memoryType =
                amPointerInfo._isInDeviceMem ? hipMemoryTypeDevice : hipMemoryTypeHost;
            attributes->hostPointer = amPointerInfo._hostPointer;
            attributes->devicePointer = amPointerInfo._devicePointer;
            attributes->isManaged = 0;
            if (attributes->memoryType == hipMemoryTypeHost) {
                attributes->hostPointer = (void*)ptr;
            }
            if (attributes->memoryType == hipMemoryTypeDevice) {
                attributes->devicePointer = (void*)ptr;
            }
            attributes->allocationFlags = amPointerInfo._appAllocationFlags;
            attributes->device = amPointerInfo._appId;

            if (attributes->device < -1) {
                e = hipErrorInvalidDevice;
            }
        } else {
            attributes->memoryType = hipMemoryTypeDevice;
            attributes->hostPointer = 0;
            attributes->devicePointer = 0;
            attributes->device = -2;
            attributes->isManaged = 0;
            attributes->allocationFlags = 0;

            e = hipErrorInvalidValue;
        }
    }
    return ihipLogStatus(e);
}


hipError_t hipHostGetDevicePointer(void** devicePointer, void* hostPointer, unsigned flags) {
    HIP_INIT_API(hipHostGetDevicePointer, devicePointer, hostPointer, flags);

    hipError_t e = hipSuccess;

    // Flags must be 0:
    if ((flags != 0) || (devicePointer == nullptr) || (hostPointer == nullptr)) {
        e = hipErrorInvalidValue;
    } else {
        hc::accelerator acc;
        *devicePointer = NULL;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, hostPointer);
        if (status == AM_SUCCESS) {
            *devicePointer =
                static_cast<char*>(amPointerInfo._devicePointer) +
                (static_cast<char*>(hostPointer) - static_cast<char*>(amPointerInfo._hostPointer));
            tprintf(DB_MEM, " host_ptr=%p returned device_pointer=%p\n", hostPointer,
                    *devicePointer);
        } else {
            e = hipErrorMemoryAllocation;
        }
    }
    return ihipLogStatus(e);
}


hipError_t hipMalloc(void** ptr, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMalloc, (TRACE_MEM), ptr, sizeBytes);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    // return NULL pointer when malloc size is 0
    if ( nullptr == ctx || nullptr == ptr)  {
        hip_status = hipErrorInvalidValue;
    }
    else if (sizeBytes == 0) {
        *ptr = NULL;
        hip_status = hipSuccess;
    } else {
        auto device = ctx->getWriteableDevice();
        *ptr = hip_internal::allocAndSharePtr("device_mem", sizeBytes, ctx, false /*shareWithAll*/,
                                              0 /*amFlags*/, 0 /*hipFlags*/, 0);

        if (sizeBytes && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        }
    }


    return ihipLogStatus(hip_status);
}

hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags) {
    HIP_INIT_SPECIAL_API(hipExtMallocWithFlags, (TRACE_MEM), ptr, sizeBytes, flags);
    HIP_SET_DEVICE();

#if (__hcc_workweek__ >= 19115)
    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    // return NULL pointer when malloc size is 0
    if (sizeBytes == 0) {
        *ptr = NULL;
        hip_status = hipSuccess;
    } else if ((ctx == nullptr) || (ptr == nullptr)) {
        hip_status = hipErrorInvalidValue;
    } else {
        unsigned amFlags = 0;
        if (flags & hipDeviceMallocFinegrained) {
            amFlags = amDeviceFinegrained;
        } else if (flags != hipDeviceMallocDefault) {
            hip_status = hipErrorInvalidValue;
            return ihipLogStatus(hip_status);
        }
        auto device = ctx->getWriteableDevice();
        *ptr = hip_internal::allocAndSharePtr("device_mem", sizeBytes, ctx, false /*shareWithAll*/,
                                              amFlags /*amFlags*/, 0 /*hipFlags*/, 0);

        if (sizeBytes && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        }
    }
#else
    hipError_t hip_status = hipErrorMemoryAllocation;
#endif

    return ihipLogStatus(hip_status);
}


hipError_t hipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags) {
    HIP_INIT_SPECIAL_API(hipHostMalloc, (TRACE_MEM), ptr, sizeBytes, flags);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;
    hip_status = hip_internal::ihipHostMalloc(tls, ptr, sizeBytes, flags);
    return ihipLogStatus(hip_status);
}

hipError_t hipMallocManaged(void** devPtr, size_t size, unsigned int flags) {
    HIP_INIT_SPECIAL_API(hipMallocManaged, (TRACE_MEM), devPtr, size, flags);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;
    if(flags != hipMemAttachGlobal)
        hip_status = hipErrorInvalidValue;
    else
        hip_status = hip_internal::ihipHostMalloc(tls, devPtr, size, hipHostMallocDefault);
    return ihipLogStatus(hip_status);
}

// Deprecated function:
hipError_t hipMallocHost(void** ptr, size_t sizeBytes) { return hipHostMalloc(ptr, sizeBytes, 0); }

// Deprecated function:
hipError_t hipMemAllocHost(void** ptr, size_t sizeBytes) { return hipHostMalloc(ptr, sizeBytes, 0); }

// Deprecated function:
hipError_t hipHostAlloc(void** ptr, size_t sizeBytes, unsigned int flags) {
    return hipHostMalloc(ptr, sizeBytes, flags);
};

hipError_t allocImage(TlsData* tls,hsa_ext_image_geometry_t geometry, int width, int height, int depth, hsa_ext_image_channel_order_t channelOrder, hsa_ext_image_channel_type_t channelType,void ** ptr, hsa_ext_image_data_info_t &imageInfo, int array_size __dparm(0)) {
   auto ctx = ihipGetTlsDefaultCtx();
   if (ctx) {
      hc::accelerator acc = ctx->getDevice()->_acc;
      hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());
      if (!agent)
         return hipErrorInvalidResourceHandle;
      size_t allocGranularity = 0;
      hsa_amd_memory_pool_t* allocRegion = static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_region());
      hsa_amd_memory_pool_get_info(*allocRegion, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &allocGranularity);

      hsa_ext_image_descriptor_t imageDescriptor;
      imageDescriptor.geometry = geometry;
      imageDescriptor.width = width;
      imageDescriptor.height = height;
      imageDescriptor.depth = depth;
      imageDescriptor.array_size = array_size;
      imageDescriptor.format.channel_order = channelOrder;
      imageDescriptor.format.channel_type = channelType;

      hsa_access_permission_t permission = HSA_ACCESS_PERMISSION_RW;
      hsa_status_t status =
         hsa_ext_image_data_get_info_with_layout(*agent, &imageDescriptor, permission, HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &imageInfo);
      if(imageInfo.size == 0 || HSA_STATUS_SUCCESS != status){
         return hipErrorRuntimeOther;
      }
      size_t alignment = imageInfo.alignment <= allocGranularity ? 0 : imageInfo.alignment;
      const unsigned am_flags = 0;
      *ptr = hip_internal::allocAndSharePtr("device_array", imageInfo.size, ctx,
                                                                          false /*shareWithAll*/, am_flags, 0, alignment);
      if (*ptr == NULL) {
         return hipErrorMemoryAllocation;
      }
      return hipSuccess;
   }
   else {
      return hipErrorMemoryAllocation;
   }
}

// width in bytes
hipError_t ihipMallocPitch(TlsData* tls, void** ptr, size_t* pitch, size_t width, size_t height, size_t depth) {
    hipError_t hip_status = hipSuccess;
    if(ptr==NULL || pitch == NULL){
        return hipErrorInvalidValue;
    }
    hsa_ext_image_data_info_t imageInfo;
    if (depth == 0)
        hip_status = allocImage(tls,HSA_EXT_IMAGE_GEOMETRY_2D,width,height,0,HSA_EXT_IMAGE_CHANNEL_ORDER_R,
                                HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32,ptr,imageInfo);
    else
        hip_status = allocImage(tls,HSA_EXT_IMAGE_GEOMETRY_3D,width,height,depth,HSA_EXT_IMAGE_CHANNEL_ORDER_R,
                               HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32,ptr,imageInfo);

    if(hip_status == hipSuccess)
        *pitch = imageInfo.size/(height == 0 ? 1:height)/(depth == 0 ? 1:depth);

    return hip_status;
}

// width in bytes
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
    HIP_INIT_SPECIAL_API(hipMallocPitch, (TRACE_MEM), ptr, pitch, width, height);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;

    if (width == 0 || height == 0) return ihipLogStatus(hipErrorUnknown);

    hip_status = ihipMallocPitch(tls, ptr, pitch, width, height, 0);
    return ihipLogStatus(hip_status);
}

hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes){
    HIP_INIT_SPECIAL_API(hipMemAllocPitch, (TRACE_MEM), dptr, pitch, widthInBytes, height,elementSizeBytes);
    HIP_SET_DEVICE();

    if (widthInBytes == 0 || height == 0) return ihipLogStatus(hipErrorInvalidValue);

    return ihipLogStatus(ihipMallocPitch(tls, dptr, pitch, widthInBytes, height, 0));
}

hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
    HIP_INIT_API(hipMalloc3D, pitchedDevPtr, &extent);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;

    if (extent.width == 0 || extent.height == 0) return ihipLogStatus(hipErrorUnknown);
    if (!pitchedDevPtr) return ihipLogStatus(hipErrorInvalidValue);
    void* ptr;
    size_t pitch;

    hip_status =
        ihipMallocPitch(tls, &pitchedDevPtr->ptr, &pitch, extent.width, extent.height, extent.depth);
    if (hip_status == hipSuccess) {
        pitchedDevPtr->pitch = pitch;
        pitchedDevPtr->xsize = extent.width;
        pitchedDevPtr->ysize = extent.height;
    }
    return ihipLogStatus(hip_status);
}

hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f) {
    hipChannelFormatDesc cd;
    cd.x = x;
    cd.y = y;
    cd.z = z;
    cd.w = w;
    cd.f = f;
    return cd;
}

extern void getChannelOrderAndType(const hipChannelFormatDesc& desc,
                                   enum hipTextureReadMode readMode,
                                   hsa_ext_image_channel_order_t* channelOrder,
                                   hsa_ext_image_channel_type_t* channelType);

hipError_t GetImageInfo(hsa_ext_image_geometry_t geometry,int width, int height, int depth, hipChannelFormatDesc desc, hsa_ext_image_data_info_t &imageInfo,int array_size __dparm(0))
{
    hsa_ext_image_descriptor_t imageDescriptor;
    imageDescriptor.geometry = geometry;
    imageDescriptor.width = width;
    imageDescriptor.height = height;
    imageDescriptor.depth = depth;
    imageDescriptor.array_size = array_size;
    hsa_ext_image_channel_order_t channelOrder;
    hsa_ext_image_channel_type_t channelType;
    getChannelOrderAndType(desc, hipReadModeElementType, &channelOrder, &channelType);
    imageDescriptor.format.channel_order = channelOrder;
    imageDescriptor.format.channel_type = channelType;

    hsa_access_permission_t permission = HSA_ACCESS_PERMISSION_RW;
    // Get the current device agent.
    hc::accelerator acc;
    hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());
    if (!agent)
        return hipErrorInvalidResourceHandle;
    hsa_status_t status =
        hsa_ext_image_data_get_info_with_layout(*agent, &imageDescriptor, permission, HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &imageInfo);
    if(HSA_STATUS_SUCCESS != status){
        return hipErrorRuntimeOther;
    }

    return hipSuccess;
}

hipError_t ihipArrayToImageFormat(hipArray_Format format,hsa_ext_image_channel_type_t &channelType) {
   switch (format) {
       case HIP_AD_FORMAT_UNSIGNED_INT8:
          channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
          break;
       case HIP_AD_FORMAT_UNSIGNED_INT16:
          channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
          break;
       case HIP_AD_FORMAT_UNSIGNED_INT32:
          channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
          break;
       case HIP_AD_FORMAT_SIGNED_INT8:
          channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
          break;
       case HIP_AD_FORMAT_SIGNED_INT16:
          channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
          break;
       case HIP_AD_FORMAT_SIGNED_INT32:
          channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
          break;
       case HIP_AD_FORMAT_HALF:
          channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
          break;
       case HIP_AD_FORMAT_FLOAT:
          channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT;
          break;
       default:
          return hipErrorUnknown;
          break;
   }
   return hipSuccess;
}

hipError_t hipArrayCreate(hipArray** array, const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
    HIP_INIT_SPECIAL_API(hipArrayCreate, (TRACE_MEM), array, pAllocateArray);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;
    if (pAllocateArray->Width > 0) {
        *array = (hipArray*)malloc(sizeof(hipArray));
        array[0]->width = pAllocateArray->Width;
        array[0]->height = pAllocateArray->Height;
        array[0]->Format = pAllocateArray->Format;
        array[0]->NumChannels = pAllocateArray->NumChannels;
        array[0]->isDrv = true;
        array[0]->textureType = hipTextureType2D;
        void** ptr = &array[0]->data;
        hsa_ext_image_channel_type_t channelType;
        hsa_ext_image_channel_order_t channelOrder;

        hip_status = ihipArrayToImageFormat(pAllocateArray->Format,channelType);
        if(hipSuccess != hip_status)
           return ihipLogStatus(hip_status);

        if (pAllocateArray->NumChannels == 4) {
           channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
        } else if (pAllocateArray->NumChannels == 2) {
           channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RG;
        } else if (pAllocateArray->NumChannels == 1) {
           channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_R;
        }
        hsa_ext_image_data_info_t imageInfo;
        return ihipLogStatus(allocImage(tls,HSA_EXT_IMAGE_GEOMETRY_2D,pAllocateArray->Width,
                                pAllocateArray->Height,0,channelOrder,channelType,ptr,imageInfo));
    } else {
        return ihipLogStatus(hipErrorInvalidValue);
    }
}

hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc, size_t width,
                          size_t height, unsigned int flags) {
    HIP_INIT_SPECIAL_API(hipMallocArray, (TRACE_MEM), array, desc, width, height, flags);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;
    if (width > 0) {
        *array = (hipArray*)malloc(sizeof(hipArray));
        array[0]->type = flags;
        array[0]->width = width;
        array[0]->height = height;
        array[0]->depth = 1;
        array[0]->desc = *desc;
        array[0]->isDrv = false;
        array[0]->textureType = hipTextureType2D;
        void** ptr = &array[0]->data;

        hsa_ext_image_channel_order_t channelOrder;
        hsa_ext_image_channel_type_t channelType;
        getChannelOrderAndType(*desc, hipReadModeElementType, &channelOrder, &channelType);
        hsa_ext_image_data_info_t imageInfo;
        switch (flags) {
            case hipArrayLayered:
            case hipArrayCubemap:
            case hipArraySurfaceLoadStore:
            case hipArrayTextureGather:
               assert(0);
               break;
            case hipArrayDefault:
            default:
               hip_status = allocImage(tls,HSA_EXT_IMAGE_GEOMETRY_2D,width,height,0,channelOrder,channelType,ptr,imageInfo);
               break;
        }
    } else {
        hip_status = hipErrorInvalidValue;
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipArray3DCreate(hipArray** array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray) {
    HIP_INIT_SPECIAL_API(hipArray3DCreate, (TRACE_MEM), array, pAllocateArray);
    hipError_t hip_status = hipSuccess;

    *array = (hipArray*)malloc(sizeof(hipArray));
    array[0]->type = pAllocateArray->Flags;
    array[0]->width = pAllocateArray->Width;
    array[0]->height = pAllocateArray->Height;
    array[0]->depth = pAllocateArray->Depth;
    array[0]->Format = pAllocateArray->Format;
    array[0]->NumChannels = pAllocateArray->NumChannels;
    array[0]->isDrv = true;
    void** ptr = &array[0]->data;

    hsa_ext_image_channel_order_t channelOrder;
    if (pAllocateArray->NumChannels == 4) {
       channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
    } else if (pAllocateArray->NumChannels == 2) {
       channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RG;
    } else if (pAllocateArray->NumChannels == 1) {
       channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_R;
    }

    hsa_ext_image_channel_type_t channelType;
    hip_status = ihipArrayToImageFormat(pAllocateArray->Format,channelType);
    hsa_ext_image_data_info_t imageInfo;
    switch (pAllocateArray->Flags) {
       case hipArrayLayered:
          hip_status = allocImage(tls,HSA_EXT_IMAGE_GEOMETRY_2DA,pAllocateArray->Width,pAllocateArray->Height,0,
                                  channelOrder,channelType,ptr,imageInfo,pAllocateArray->Depth);
          array[0]->textureType = hipTextureType2DLayered;
          break;
       case hipArraySurfaceLoadStore:
       case hipArrayTextureGather:
          assert(0);
          break;
       case hipArrayDefault:
       case hipArrayCubemap:
       default:
          hip_status = allocImage(tls,HSA_EXT_IMAGE_GEOMETRY_3D,pAllocateArray->Width,pAllocateArray->Height,
                                  pAllocateArray->Depth,channelOrder,channelType,ptr,imageInfo);
          array[0]->textureType = hipTextureType3D;
          break;
    }

    return ihipLogStatus(hip_status);
}

hipError_t hipMalloc3DArray(hipArray** array, const struct hipChannelFormatDesc* desc,
                            struct hipExtent extent, unsigned int flags) {

    HIP_INIT_API(hipMalloc3DArray, array, desc, &extent, flags);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;

    if(array==NULL ){
        return ihipLogStatus(hipErrorInvalidValue);
    }
    *array = (hipArray*)malloc(sizeof(hipArray));
    array[0]->type = flags;
    array[0]->width = extent.width;
    array[0]->height = extent.height;
    array[0]->depth = extent.depth;
    array[0]->desc = *desc;
    array[0]->isDrv = false;
    void** ptr = &array[0]->data;
    hsa_ext_image_channel_order_t channelOrder;
    hsa_ext_image_channel_type_t channelType;
    getChannelOrderAndType(*desc, hipReadModeElementType, &channelOrder, &channelType);
    hsa_ext_image_data_info_t imageInfo;
    switch (flags) {
       case hipArrayLayered:
          hip_status = allocImage(tls,HSA_EXT_IMAGE_GEOMETRY_2DA,extent.width,extent.height,0,channelOrder,channelType,ptr,imageInfo,extent.depth);
          array[0]->textureType = hipTextureType2DLayered;
          break;
       case hipArraySurfaceLoadStore:
       case hipArrayTextureGather:
          assert(0);
          break;
       case hipArrayDefault:
       case hipArrayCubemap:
       default:
          hip_status = allocImage(tls,HSA_EXT_IMAGE_GEOMETRY_3D,extent.width,extent.height,extent.depth,channelOrder,channelType,ptr,imageInfo);
          array[0]->textureType = hipTextureType3D;
          break;
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
    HIP_INIT_API(hipHostGetFlags, flagsPtr, hostPtr);

    hipError_t hip_status = hipSuccess;

    hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
    hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
    am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, hostPtr);
    if (status == AM_SUCCESS) {
        *flagsPtr = amPointerInfo._appAllocationFlags;
        //0 is valid flag hipHostMallocDefault, and during hipHostMalloc if unsupported flags are passed as parameter it throws error
        hip_status = hipSuccess;
        tprintf(DB_MEM, " %s: host ptr=%p\n", __func__, hostPtr);
    } else {
        hip_status = hipErrorInvalidValue;
    }
    return ihipLogStatus(hip_status);
}


// TODO - need to fix several issues here related to P2P access, host memory fallback.
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
    HIP_INIT_API(hipHostRegister, hostPtr, sizeBytes, flags);

    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (hostPtr == NULL) {
        return ihipLogStatus(hipErrorInvalidValue);
    }

    hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
    hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
    am_status_t am_status = hc::am_memtracker_getinfo(&amPointerInfo, hostPtr);

    if (am_status == AM_SUCCESS) {
        hip_status = hipErrorHostMemoryAlreadyRegistered;
    } else {
        auto ctx = ihipGetTlsDefaultCtx();
        if (hostPtr == NULL) {
            return ihipLogStatus(hipErrorInvalidValue);
        }
        // TODO-test : multi-gpu access to registered host memory.
        if (ctx) {
            if ((flags == hipHostRegisterDefault) || (flags & hipHostRegisterPortable) ||
                (flags & hipHostRegisterMapped) || (flags == hipExtHostRegisterCoarseGrained)) {
                auto device = ctx->getWriteableDevice();
                std::vector<hc::accelerator> vecAcc;
                for (int i = 0; i < g_deviceCnt; i++) {
                    vecAcc.push_back(ihipGetDevice(i)->_acc);
                }
#if (__hcc_workweek__ >= 19183)
                if(flags & hipExtHostRegisterCoarseGrained) {
                    am_status = hc::am_memory_host_lock(device->_acc, hostPtr, sizeBytes, &vecAcc[0],
                                                    vecAcc.size());
                } else {
                    am_status = hc::am_memory_host_lock_with_flag(device->_acc, hostPtr, sizeBytes, &vecAcc[0],
                                                    vecAcc.size());
                }
#else
                am_status = hc::am_memory_host_lock(device->_acc, hostPtr, sizeBytes, &vecAcc[0],
                                                    vecAcc.size());
#endif
                if ( am_status == AM_SUCCESS ) {
                     am_status = hc::am_memtracker_getinfo(&amPointerInfo, hostPtr);

                if ( am_status == AM_SUCCESS ) {
                    void *devPtr = amPointerInfo._devicePointer;
 #if USE_APP_PTR_FOR_CTX
                    hc::am_memtracker_update(hostPtr, device->_deviceId, flags, ctx);
                    hc::am_memtracker_update(devPtr, device->_deviceId, flags, ctx);
 #else
                    hc::am_memtracker_update(hostPtr, device->_deviceId, flags);
                    hc::am_memtracker_update(devPtr, device->_deviceId, flags);
 #endif
                    tprintf(DB_MEM, " %s registered ptr=%p and allowed access to %zu peers\n", __func__,
                                 hostPtr, vecAcc.size());
                };
                };
                if (am_status == AM_SUCCESS) {
                    hip_status = hipSuccess;
                } else {
                    hip_status = hipErrorMemoryAllocation;
                }
            } else {
                hip_status = hipErrorInvalidValue;
            }
        }
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipHostUnregister(void* hostPtr) {
    HIP_INIT_API(hipHostUnregister, hostPtr);
    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t hip_status = hipSuccess;
    if (hostPtr == NULL) {
        hip_status = hipErrorInvalidValue;
    } else {
        auto device = ctx->getWriteableDevice();
        am_status_t am_status = hc::am_memory_host_unlock(device->_acc, hostPtr);
        tprintf(DB_MEM, " %s unregistered ptr=%p\n", __func__, hostPtr);
        if (am_status != AM_SUCCESS) {
            hip_status = hipErrorHostMemoryNotRegistered;
        }
    }
    return ihipLogStatus(hip_status);
}

namespace hip_impl {
hipError_t hipMemcpyToSymbol(void* dst, const void* src, size_t count,
                             size_t offset, hipMemcpyKind kind,
                             const char* symbol_name) {
    HIP_INIT_SPECIAL_API(hipMemcpyToSymbol, (TRACE_MCMD), symbol_name, src,
                         count, offset, kind);

    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbol_name, dst);

    return ihipLogStatus(
        hipMemcpy(static_cast<char*>(dst) + offset, src, count, kind));
}

hipError_t hipMemcpyFromSymbol(void* dst, const void* src, size_t count,
                               size_t offset, hipMemcpyKind kind,
                               const char* symbol_name) {
    HIP_INIT_SPECIAL_API(hipMemcpyFromSymbol, (TRACE_MCMD), symbol_name, dst,
                         count, offset, kind);

    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbol_name, dst);

    return ihipLogStatus(
        hipMemcpy(dst, static_cast<const char*>(src) + offset, count, kind));
}


hipError_t hipMemcpyToSymbolAsync(void* dst, const void* src, size_t count,
                                  size_t offset, hipMemcpyKind kind,
                                  hipStream_t stream, const char* symbol_name) {
    HIP_INIT_SPECIAL_API(hipMemcpyToSymbolAsync, (TRACE_MCMD), symbol_name, src,
                         count, offset, kind, stream);

    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbol_name, dst);

    if (dst == nullptr) {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipError_t e = hipSuccess;
    if (stream) {
        try {
            hip_internal::memcpyAsync((char*)dst+offset, src, count, kind, stream);
        } catch (ihipException& ex) {
            e = ex._code;
        }
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* src, size_t count,
                                    size_t offset, hipMemcpyKind kind,
                                    hipStream_t stream, const char* symbol_name) {
    HIP_INIT_SPECIAL_API(hipMemcpyFromSymbolAsync, (TRACE_MCMD), symbol_name,
                         dst, count, offset, kind, stream);

    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbol_name, src);

    if (src == nullptr || dst == nullptr) {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipError_t e = hipSuccess;
    stream = ihipSyncAndResolveStream(stream);
    if (stream) {
        try {
            hip_internal::memcpyAsync(dst, (char*)src+offset, count, kind, stream);
        } catch (ihipException& ex) {
            e = ex._code;
        }
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}
} // Namespace hip_impl.

//---
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpy, (TRACE_MCMD), dst, src, sizeBytes, kind);

    return ihipLogStatus(hip_internal::memcpySync(dst, src, sizeBytes, kind,
                                                  hipStreamNull));
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMemcpyHtoD, (TRACE_MCMD), dst, src, sizeBytes);

    return ihipLogStatus(hip_internal::memcpySync(dst, src, sizeBytes,
                                                  hipMemcpyHostToDevice,
                                                  hipStreamNull));
}

hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMemcpyDtoH, (TRACE_MCMD), dst, src, sizeBytes);

    return ihipLogStatus(hip_internal::memcpySync(dst, src, sizeBytes,
                                                  hipMemcpyDeviceToHost,
                                                  hipStreamNull));
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMemcpyDtoD, (TRACE_MCMD), dst, src, sizeBytes);

    return ihipLogStatus(hip_internal::memcpySync(dst, src, sizeBytes,
                                                  hipMemcpyDeviceToDevice,
                                                  hipStreamNull));
}

hipError_t hipMemcpyHtoH(void* dst, void* src, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMemcpyHtoH, (TRACE_MCMD), dst, src, sizeBytes);

    return ihipLogStatus(hip_internal::memcpySync(dst, src, sizeBytes,
                                                  hipMemcpyHostToHost,
                                                  hipStreamNull));
}

hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,
                               hipMemcpyKind kind, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyWithStream, (TRACE_MCMD), dst, src, sizeBytes,
                         kind, stream);

    return ihipLogStatus(hip_internal::memcpySync(dst, src, sizeBytes, kind,
                                                  stream));
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyAsync, (TRACE_MCMD), dst, src, sizeBytes, kind, stream);

    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, kind, stream));
}

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyHtoDAsync, (TRACE_MCMD), dst, src, sizeBytes, stream);

    return ihipLogStatus(
        hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyHostToDevice, stream));
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyDtoDAsync, (TRACE_MCMD), dst, src, sizeBytes, stream);

    return ihipLogStatus(
        hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream));
}

hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyDtoHAsync, (TRACE_MCMD), dst, src, sizeBytes, stream);

    return ihipLogStatus(
        hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToHost, stream));
}

hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpy2DToArray, (TRACE_MCMD), dst, wOffset, hOffset, src, spitch, width, height, kind);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hipError_t e = hipSuccess;

    size_t byteSize;
    if (dst) {
        switch (dst[0].desc.f) {
            case hipChannelFormatKindSigned:
                byteSize = sizeof(int);
                break;
            case hipChannelFormatKindUnsigned:
                byteSize = sizeof(unsigned int);
                break;
            case hipChannelFormatKindFloat:
                byteSize = sizeof(float);
                break;
            case hipChannelFormatKindNone:
                byteSize = sizeof(size_t);
                break;
            default:
                byteSize = 0;
                break;
        }
    } else {
        return ihipLogStatus(hipErrorUnknown);
    }

    if ((wOffset + width > (dst->width * byteSize)) || width > spitch) {
        return ihipLogStatus(hipErrorUnknown);
    }

    size_t src_w = spitch;
    size_t dst_w = (dst->width) * byteSize;

    try {
        for (int i = 0; i < height; ++i) {
            stream->locked_copySync((unsigned char*)dst->data + i * dst_w,
                                    (unsigned char*)src + i * src_w, width, kind);
        }
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpyToArray, (TRACE_MCMD), dst, wOffset, hOffset, src, count, kind);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hipError_t e = hipSuccess;

    try {
        stream->locked_copySync((char*)dst->data + wOffset, src, count, kind);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpyFromArray, (TRACE_MCMD), dst, srcArray, wOffset, hOffset, count, kind);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hipError_t e = hipSuccess;

    try {
        stream->locked_copySync((char*)dst, (char*)srcArray->data + wOffset, count, kind);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost, size_t count) {
    HIP_INIT_SPECIAL_API(hipMemcpyHtoA, (TRACE_MCMD), dstArray, dstOffset, srcHost, count);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hipError_t e = hipSuccess;
    try {
        stream->locked_copySync((char*)dstArray->data + dstOffset, srcHost, count,
                                hipMemcpyHostToDevice);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset, size_t count) {
    HIP_INIT_SPECIAL_API(hipMemcpyAtoH, (TRACE_MCMD), dst, srcArray, srcOffset, count);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hipError_t e = hipSuccess;

    try {
        stream->locked_copySync((char*)dst, (char*)srcArray->data + srcOffset, count,
                                hipMemcpyDeviceToHost);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t ihipMemcpy3D(const struct hipMemcpy3DParms* p, hipStream_t stream, bool isAsync) {
    hipError_t e = hipSuccess;
    if(p) {
        size_t byteSize, width, height, depth, widthInBytes, srcPitch, dstPitch, ySize;
        hipChannelFormatDesc desc;
        void* srcPtr;void* dstPtr;
        if (p->dstArray != nullptr) {
            if (p->dstArray->isDrv == false) {
                switch (p->dstArray->desc.f) {
                    case hipChannelFormatKindSigned:
                        byteSize = sizeof(int);
                        break;
                    case hipChannelFormatKindUnsigned:
                        byteSize = sizeof(unsigned int);
                        break;
                    case hipChannelFormatKindFloat:
                        byteSize = sizeof(float);
                        break;
                    case hipChannelFormatKindNone:
                        byteSize = sizeof(size_t);
                        break;
                    default:
                        byteSize = 0;
                        break;
                }
                depth = p->extent.depth;
                height = p->extent.height;
                width =  p->extent.width;
                widthInBytes = p->extent.width * byteSize;
                srcPitch = p->srcPtr.pitch;
                srcPtr = p->srcPtr.ptr;
                ySize = p->srcPtr.ysize;
                desc = p->dstArray->desc;
                dstPtr = p->dstArray->data;
            } else {
                depth = p->Depth;
                height = p->Height;
                widthInBytes = p->WidthInBytes;
                width =  p->dstArray->width;
                desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindSigned);
                srcPitch = p->srcPitch;
                srcPtr = (void*)p->srcHost;
                ySize = p->srcHeight;
                dstPtr = p->dstArray->data;
            }
            hsa_ext_image_data_info_t imageInfo;
            if(hipTextureType2DLayered == p->dstArray->textureType)
                GetImageInfo(HSA_EXT_IMAGE_GEOMETRY_2DA, width, height, 0, desc, imageInfo, depth);
            else
                GetImageInfo(HSA_EXT_IMAGE_GEOMETRY_3D, width, height, depth, desc, imageInfo);

            dstPitch = imageInfo.size/(height == 0 ? 1:height)/(depth == 0 ? 1:depth);

        } else {
            // Non array destination
            depth = p->extent.depth;
            height = p->extent.height;
            widthInBytes = p->extent.width;
            srcPitch = p->srcPtr.pitch;
            srcPtr = p->srcPtr.ptr;
            dstPtr = p->dstPtr.ptr;
            ySize = p->srcPtr.ysize;
            dstPitch = p->dstPtr.pitch;
        }

        stream = ihipSyncAndResolveStream(stream);
        try {
            if((widthInBytes == dstPitch) && (widthInBytes == srcPitch)) {
                if(isAsync)
                    stream->locked_copyAsync((void*)dstPtr, (void*)srcPtr, widthInBytes*height*depth, p->kind);
                else
                    stream->locked_copySync((void*)dstPtr, (void*)srcPtr, widthInBytes*height*depth, p->kind, false);
            } else {
                for (int i = 0; i < depth; i++) {
                    for (int j = 0; j < height; j++) {
                        // TODO: p->srcPos or p->dstPos are not 0.
                        unsigned char* src =
                             (unsigned char*)srcPtr + i * ySize * srcPitch + j * srcPitch;
                        unsigned char* dst =
                             (unsigned char*)dstPtr + i * height * dstPitch + j * dstPitch;
                        if(isAsync)
                            stream->locked_copyAsync(dst, src, widthInBytes, p->kind);
                        else
                            stream->locked_copySync(dst, src, widthInBytes, p->kind);
                     }
                }
           }
        } catch (ihipException ex) {
            e = ex._code;
        }
    } else {
        e = hipErrorInvalidValue;
    }
    return e;
}

hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p) {
    HIP_INIT_SPECIAL_API(hipMemcpy3D, (TRACE_MCMD), p);
    hipError_t e = hipSuccess;
    e = ihipMemcpy3D(p, hipStreamNull, false);
    return ihipLogStatus(e);
}

hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpy3DAsync, (TRACE_MCMD), p, stream);
    hipError_t e = hipSuccess;
    e = ihipMemcpy3D(p, stream, true);
    return ihipLogStatus(e);
}

namespace {
template <uint32_t block_dim, uint32_t items_per_lane,
          typename RandomAccessIterator, typename N, typename T>
__global__ void hip_fill_n(RandomAccessIterator f, N n, T value) {
    const auto grid_dim = gridDim.x * blockDim.x * items_per_lane;
    const auto gidx = blockIdx.x * block_dim + threadIdx.x;

    size_t idx = gidx * items_per_lane;
    while (idx + items_per_lane <= n) {
        for (auto i = 0u; i != items_per_lane; ++i) {
            __builtin_nontemporal_store(value, &f[idx + i]);
        }
        idx += grid_dim;
    }

    if (gidx < n % grid_dim) {
        __builtin_nontemporal_store(value, &f[n - gidx - 1]);
    }
}

template <typename T, typename std::enable_if<std::is_integral<T>{}>::type* = nullptr>
inline const T& clamp_integer(const T& x, const T& lower, const T& upper) {
    assert(!(upper < lower));

    return std::min(upper, std::max(x, lower));
}

template <typename T>
__global__ void hip_copy2d_n(T* dst, const T* src, size_t width, size_t height, size_t destPitch, size_t srcPitch) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    size_t floorWidth = (width/sizeof(T));
    T *dstPtr = (T *)((uint8_t*) dst + idy * destPitch);
    T *srcPtr = (T *)((uint8_t*) src + idy * srcPitch);
    if((idx < floorWidth) && (idy < height)){
        dstPtr[idx] = srcPtr[idx];
    } else if((idx < width) && (idy < height)){
        size_t bytesToCopy = width - (floorWidth * sizeof(T));
        dstPtr += floorWidth;
        srcPtr += floorWidth;
        __builtin_memcpy(reinterpret_cast<uint8_t*>(dstPtr), reinterpret_cast<const uint8_t*>(srcPtr),bytesToCopy);
    }
}
}  // namespace

//Get the allocated size
hipError_t ihipMemPtrGetInfo(void* ptr, size_t* size) {
    hipError_t e = hipSuccess;
    if (ptr != nullptr && size != nullptr) {
        *size = 0;
        hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if (status == AM_SUCCESS) {
            *size = amPointerInfo._sizeBytes;
        } else {
            e = hipErrorInvalidValue;
        }
    } else {
        e = hipErrorInvalidValue;
    }
    return e;
}

template <typename T>
void ihipMemsetKernel(hipStream_t stream, T* ptr, T val, size_t count) {
    static constexpr uint32_t block_dim = 256;
    static constexpr uint32_t max_write_width = 4 * sizeof(std::uint32_t); // 4 DWORDs
    static constexpr uint32_t items_per_lane = max_write_width / sizeof(T);

    const uint32_t grid_dim = clamp_integer<size_t>(
        count / (block_dim * items_per_lane), 1, UINT32_MAX);

    hipLaunchKernelGGL(hip_fill_n<block_dim, items_per_lane>, dim3(grid_dim),
                       dim3{block_dim}, 0u, stream, ptr, count, std::move(val));
}

template <typename T>
void ihipMemcpy2dKernel(hipStream_t stream, T* dst, const T* src, size_t width, size_t height, size_t destPitch, size_t srcPitch) {
    size_t threadsPerBlock_x = 64;
    size_t threadsPerBlock_y = 4;
    uint32_t grid_dim_x = clamp_integer<size_t>( (width+(threadsPerBlock_x*sizeof(T)-1)) / (threadsPerBlock_x*sizeof(T)), 1, UINT32_MAX);
    uint32_t grid_dim_y = clamp_integer<size_t>( (height+(threadsPerBlock_y-1)) / threadsPerBlock_y, 1, UINT32_MAX);
    hipLaunchKernelGGL(hip_copy2d_n, dim3(grid_dim_x,grid_dim_y), dim3(threadsPerBlock_x,threadsPerBlock_y), 0u, stream, dst, src,
                       width, height, destPitch, srcPitch);
}

typedef enum ihipMemsetDataType {
    ihipMemsetDataTypeChar   = 0,
    ihipMemsetDataTypeShort  = 1,
    ihipMemsetDataTypeInt    = 2
}ihipMemsetDataType;

hipError_t ihipMemsetAsync(void* dst, int  value, size_t count, hipStream_t stream, enum ihipMemsetDataType copyDataType) {
    if (count == 0) return hipSuccess;
    if (!dst) return hipErrorInvalidValue;

    try {
        if (copyDataType == ihipMemsetDataTypeChar) {
            if ((count & 0x3) == 0) {
                // use a faster dword-per-workitem copy:
                value = value & 0xff;
                uint32_t value32 = (value << 24) | (value << 16) | (value << 8) | (value) ;
                ihipMemsetKernel<uint32_t> (stream, static_cast<uint32_t*> (dst), value32, count/sizeof(uint32_t));
            } else {
                // use a slow byte-per-workitem copy:
                ihipMemsetKernel<char> (stream, static_cast<char*> (dst), value, count);
            }
        } else if (copyDataType == ihipMemsetDataTypeInt) { // 4 Bytes value
            ihipMemsetKernel<uint32_t> (stream, static_cast<uint32_t*> (dst), value, count);
        } else if (copyDataType == ihipMemsetDataTypeShort) {
            value = value & 0xffff;
            ihipMemsetKernel<uint16_t> (stream, static_cast<uint16_t*> (dst), value, count);
        }
    } catch (...) {
        return hipErrorInvalidValue;
    }

    if (HIP_API_BLOCKING) {
        tprintf (DB_SYNC, "%s LAUNCH_BLOCKING wait for hipMemsetAsync.\n", ToString(stream).c_str());
        stream->locked_wait();
    }

    return hipSuccess;
}

namespace {
    template<typename T>
    void handleHeadTail(T* dst, std::size_t n_head, std::size_t n_body,
                        std::size_t n_tail, hipStream_t stream, int value) {
        struct Cleaner {
            static
            __global__
            void clean(T* p, std::size_t nh, std::size_t nb, int x) noexcept {
                p[(threadIdx.x < nh) ? threadIdx.x : (threadIdx.x - nh + nb)] = x;
            }
        };

        hipLaunchKernelGGL(Cleaner::clean, 1, n_head + n_tail, 0, stream,
                           dst, n_head,
                           n_body * sizeof(std::uint32_t) / sizeof(T), value);

    }
} // Anonymous namespace.

hipError_t ihipMemsetSync(void* dst, int  value, size_t count, hipStream_t stream, ihipMemsetDataType copyDataType) {
    if (count == 0) return hipSuccess;
    if (!dst) return hipErrorInvalidValue;

    try {
        size_t n = count;
        auto aligned_dst{(copyDataType == ihipMemsetDataTypeInt) ? dst :
            reinterpret_cast<void*>(
                hip_impl::round_up_to_next_multiple_nonnegative(
                    reinterpret_cast<std::uintptr_t>(dst), 4ul))};
        size_t n_head{};
        size_t n_tail{};
        int original_value = value;

        switch (copyDataType) {
            case ihipMemsetDataTypeChar:
                value &= 0xff;
                value = (value << 24) | (value << 16) | (value << 8) | value;
                n_head = static_cast<std::uint8_t*>(aligned_dst) -
                    static_cast<std::uint8_t*>(dst);
                n -= n_head;
                n /= sizeof(std::uint32_t);
                n_tail = count % sizeof(std::uint32_t);
                break;
            case ihipMemsetDataTypeShort:
                value &= 0xffff;
                value = (value << 16) | value;
                n_head = static_cast<std::uint16_t*>(aligned_dst) -
                    static_cast<std::uint16_t*>(dst);
                n = (count - n_head) *
                    sizeof(std::uint16_t) / sizeof(std::uint32_t);
                n_tail = ((count - n_head) *
                    sizeof(std::uint16_t)) % sizeof(std::uint32_t);
                break;
            default: break;
        }

        // queue the memset kernel for the remainder of the buffer before the HSA call below
        if (aligned_dst != dst || n_tail != 0) {
            switch (copyDataType) {
            case ihipMemsetDataTypeChar:
                handleHeadTail(static_cast<std::uint8_t*>(dst), n_head, n,
                               n_tail, stream, value & 0xff);
                break;
            case ihipMemsetDataTypeShort:
                handleHeadTail(static_cast<std::uint16_t*>(dst), n_head, n,
                               n_tail, stream, value & 0xffff);
                break;
            default: break;
            }
        }

        // The stream must be locked from all other op insertions to guarantee
        // that the following HSA call can complete before any other ops.
        // Flush the stream while locked. Once the stream is empty, we can safely perform
        // the out-of-band HSA call. Lastly, the stream will unlock via RAII.
	    if (!stream) stream = ihipSyncAndResolveStream(stream);
	    if (!stream) return hipErrorInvalidValue;

        LockedAccessor_StreamCrit_t crit(stream->criticalData());
        crit->_av.wait(stream->waitMode());
        const auto s = hsa_amd_memory_fill(aligned_dst, value, n);
        if (s != HSA_STATUS_SUCCESS) return hipErrorInvalidValue;
    }
    catch (...) {
        return hipErrorInvalidValue;
    }

    if (HIP_API_BLOCKING) {
        tprintf (DB_SYNC, "%s LAUNCH_BLOCKING wait for hipMemsetSync.\n", ToString(stream).c_str());
        stream->locked_wait();
    }

    return hipSuccess;
}

hipError_t getLockedPointer(void *hostPtr, size_t dataLen, void **devicePtrPtr)
{
        hc::accelerator acc;

#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, hostPtr);
        if (status == AM_SUCCESS) {
            *devicePtrPtr = static_cast<char*>(amPointerInfo._devicePointer) +
                (static_cast<char*>(hostPtr) - static_cast<char*>(amPointerInfo._hostPointer));
            return(hipSuccess);
        };
        return(hipErrorHostMemoryNotRegistered);
}

// TODO - review and optimize
hipError_t ihipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                        size_t height, hipMemcpyKind kind) {
    if (dst == nullptr || src == nullptr || width > dpitch || width > spitch) return hipErrorInvalidValue;

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);
    int isLockedOrD2D = 0;
    void *pinnedPtr=NULL;
    void *actualSrc = (void*)src;
    void *actualDest = dst;
    if(kind == hipMemcpyHostToDevice ) {
        if(getLockedPointer((void*)src, spitch, &pinnedPtr) == hipSuccess ){
            isLockedOrD2D = 1;
            actualSrc = pinnedPtr;
        }
    } else if(kind == hipMemcpyDeviceToHost) {
          if(getLockedPointer((void*)dst, dpitch, &pinnedPtr) == hipSuccess ){
            isLockedOrD2D = 1;
            actualDest = pinnedPtr;
          }
    } else if(kind == hipMemcpyDeviceToDevice) {
        isLockedOrD2D = 1;
    }

    hc::completion_future marker;

    hipError_t e = hipSuccess;
    if((width == dpitch) && (width == spitch)) {
        stream->locked_copySync((void*)dst, (void*)src, width*height, kind, false);
    } else {
        try {
            if(!isLockedOrD2D) {
                for (int i = 0; i < height; ++i)
                    stream->locked_copySync((unsigned char*)dst + i * dpitch,
                                    (unsigned char*)src + i * spitch, width, kind);
            } else {
                if(!stream->locked_copy2DSync(dst, src, width, height, spitch, dpitch, kind)){
                    ihipMemcpy2dKernel<uint8_t> (stream, static_cast<uint8_t*> (dst), static_cast<const uint8_t*> (src), width, height, dpitch, spitch);
                    stream->locked_wait();
                }
            }
        } catch (ihipException& ex) {
            e = ex._code;
        }
    }

    return e;
}

hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpy2D, (TRACE_MCMD), dst, dpitch, src, spitch, width, height, kind);
    hipError_t e = hipSuccess;
    e = ihipMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
    return ihipLogStatus(e);
}

hipError_t ihipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream) {
    if (dst == nullptr || src == nullptr || width > dpitch || width > spitch) return hipErrorInvalidValue;
    hipError_t e = hipSuccess;
    int isLockedOrD2D = 0;
    void *pinnedPtr=NULL;
    void *actualSrc = (void*)src;
    void *actualDest = dst;
    stream = ihipSyncAndResolveStream(stream);
    if(kind == hipMemcpyHostToDevice ) {
        if(getLockedPointer((void*)src, spitch, &pinnedPtr) == hipSuccess ){
            isLockedOrD2D = 1;
            actualSrc = pinnedPtr;
        }
    } else if(kind == hipMemcpyDeviceToHost) {
          if(getLockedPointer((void*)dst, dpitch, &pinnedPtr) == hipSuccess ){
            isLockedOrD2D = 1;
            actualDest = pinnedPtr;
          }
    } else if(kind == hipMemcpyDeviceToDevice) {
        isLockedOrD2D = 1;
    }

    if((width == dpitch) && (width == spitch)) {
            hip_internal::memcpyAsync(dst, src, width*height, kind, stream);
    } else {
        try {
            if(!isLockedOrD2D){
                for (int i = 0; i < height; ++i)
                    e = hip_internal::memcpyAsync((unsigned char*)dst + i * dpitch,
                                          (unsigned char*)src + i * spitch, width, kind, stream);
            } else{
                if(!stream->locked_copy2DAsync(dst, src, width, height, spitch, dpitch, kind)){
                    ihipMemcpy2dKernel<uint8_t> (stream, static_cast<uint8_t*> (dst), static_cast<const uint8_t*> (src), width, height, dpitch, spitch);
                }
            }
        } catch (ihipException& ex) {
            e = ex._code;
        }
    }

    return e;
}

hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpy2DAsync, (TRACE_MCMD), dst, dpitch, src, spitch, width, height, kind, stream);
    hipError_t e = hipSuccess;
    e = ihipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
    return ihipLogStatus(e);
}

hipError_t ihip2dOffsetMemcpy(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, size_t srcXOffsetInBytes, size_t srcYOffset,
                            size_t dstXOffsetInBytes, size_t dstYOffset,hipMemcpyKind kind,
                            hipStream_t stream, bool isAsync) {
    if((spitch < width + srcXOffsetInBytes) || (srcYOffset >= height)){
        return hipErrorInvalidValue;
    } else if((dpitch < width + dstXOffsetInBytes) || (dstYOffset >= height)){
        return hipErrorInvalidValue;
    }
    src = (void*)((char*)src+ srcYOffset*spitch + srcXOffsetInBytes);
    dst = (void*)((char*)dst+ dstYOffset*dpitch + dstXOffsetInBytes);
    if(isAsync){
        return ihipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, hipMemcpyDefault, stream);
    } else{
        return ihipMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyDefault);
    }
}

hipError_t ihipMemcpyParam2D(const hip_Memcpy2D* pCopy, hipStream_t stream, bool isAsync) {
    if (pCopy == nullptr) {
        return hipErrorInvalidValue;
    }
    void* dst; const void* src;
    size_t spitch = pCopy->srcPitch;
    size_t dpitch = pCopy->dstPitch;
    switch(pCopy->srcMemoryType){
        case hipMemoryTypeHost:
            src = pCopy->srcHost;
            break;
        case hipMemoryTypeArray:
            src = pCopy->srcArray->data;
            spitch = pCopy->WidthInBytes;
            break;
        case hipMemoryTypeUnified:
        case hipMemoryTypeDevice:
            src = pCopy->srcDevice;
            break;
        default:
            return hipErrorInvalidValue;
    }
    switch(pCopy->dstMemoryType){
        case hipMemoryTypeHost:
            dst = pCopy->dstHost;
            break;
        case hipMemoryTypeArray:
            dst = pCopy->dstArray->data;
            dpitch = pCopy->WidthInBytes;
            break;
        case hipMemoryTypeUnified:
        case hipMemoryTypeDevice:
            dst = pCopy->dstDevice;
            break;
        default:
            return hipErrorInvalidValue;
    }
    return ihip2dOffsetMemcpy(dst, dpitch, src, spitch, pCopy->WidthInBytes,
                            pCopy->Height, pCopy->srcXInBytes, pCopy->srcY,
                            pCopy->dstXInBytes, pCopy->dstY, hipMemcpyDefault,
                            stream, isAsync);
}

hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
    HIP_INIT_SPECIAL_API(hipMemcpyParam2D, (TRACE_MCMD), pCopy);
    return ihipLogStatus(ihipMemcpyParam2D(pCopy, hipStreamNull, false));
}

hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyParam2DAsync, (TRACE_MCMD), pCopy, stream);
    return ihipLogStatus(ihipMemcpyParam2D(pCopy, stream, true));
}

hipError_t hipMemcpy2DFromArray( void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind ){
    HIP_INIT_SPECIAL_API(hipMemcpy2DFromArray, (TRACE_MCMD), dst, dpitch, src, wOffset, hOffset, width, height, kind);
    size_t byteSize;
    if(src) {
        switch (src->desc.f) {
            case hipChannelFormatKindSigned:
                byteSize = sizeof(int);
                break;
            case hipChannelFormatKindUnsigned:
                byteSize = sizeof(unsigned int);
                break;
            case hipChannelFormatKindFloat:
                byteSize = sizeof(float);
                break;
            case hipChannelFormatKindNone:
                byteSize = sizeof(size_t);
                break;
            default:
                byteSize = 0;
                break;
        }
    } else {
        return ihipLogStatus(hipErrorInvalidValue);
    }
    return ihipLogStatus(ihip2dOffsetMemcpy(dst, dpitch, src->data, src->width*byteSize, width, height, wOffset, hOffset, 0, 0, kind, hipStreamNull, false));
}

hipError_t hipMemcpy2DFromArrayAsync( void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream ){
    HIP_INIT_SPECIAL_API(hipMemcpy2DFromArrayAsync, (TRACE_MCMD), dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
    size_t byteSize;
    if(src) {
        switch (src->desc.f) {
            case hipChannelFormatKindSigned:
                byteSize = sizeof(int);
                break;
            case hipChannelFormatKindUnsigned:
                byteSize = sizeof(unsigned int);
                break;
            case hipChannelFormatKindFloat:
                byteSize = sizeof(float);
                break;
            case hipChannelFormatKindNone:
                byteSize = sizeof(size_t);
                break;
            default:
                byteSize = 0;
                break;
        }
    } else {
        return ihipLogStatus(hipErrorInvalidValue);
    }
    return ihipLogStatus(ihip2dOffsetMemcpy(dst, dpitch, src->data, src->width*byteSize, width, height, wOffset, hOffset, 0, 0, kind, stream, true));
}

// TODO-sync: function is async unless target is pinned host memory - then these are fully sync.
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemsetAsync, (TRACE_MCMD), dst, value, sizeBytes, stream);
    return ihipLogStatus(ihipMemsetAsync(dst, value, sizeBytes, stream, ihipMemsetDataTypeChar));
}

hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemsetD32Async, (TRACE_MCMD), dst, value, count, stream);
    return ihipLogStatus(ihipMemsetAsync(dst, value, count, stream, ihipMemsetDataTypeInt));
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMemset, (TRACE_MCMD), dst, value, sizeBytes);
    return ihipLogStatus(ihipMemsetSync(dst, value, sizeBytes, nullptr, ihipMemsetDataTypeChar));
}

hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
    HIP_INIT_SPECIAL_API(hipMemset2D, (TRACE_MCMD), dst, pitch, value, width, height);
    size_t sizeBytes = pitch * height;
    return ihipLogStatus(ihipMemsetSync(dst, value, sizeBytes, nullptr, ihipMemsetDataTypeChar));
}

hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream ) {
    HIP_INIT_SPECIAL_API(hipMemset2DAsync, (TRACE_MCMD), dst, pitch, value, width, height, stream);
    size_t sizeBytes = pitch * height;
    return ihipLogStatus(ihipMemsetAsync(dst, value, sizeBytes, stream, ihipMemsetDataTypeChar));
}

hipError_t hipMemsetD8(hipDeviceptr_t dst, unsigned char value, size_t count) {
    HIP_INIT_SPECIAL_API(hipMemsetD8, (TRACE_MCMD), dst, value, count);
    return ihipLogStatus(ihipMemsetSync(dst, value, count, nullptr, ihipMemsetDataTypeChar));
}

hipError_t hipMemsetD8Async(hipDeviceptr_t dst, unsigned char value, size_t count , hipStream_t stream ) {
    HIP_INIT_SPECIAL_API(hipMemsetD8Async, (TRACE_MCMD), dst, value, count, stream);
    return ihipLogStatus(ihipMemsetAsync(dst, value, count, stream, ihipMemsetDataTypeChar));
}

hipError_t hipMemsetD16(hipDeviceptr_t dst, unsigned short value, size_t count){
    HIP_INIT_SPECIAL_API(hipMemsetD16, (TRACE_MCMD), dst, value, count);
    return ihipLogStatus(ihipMemsetSync(dst, value, count, nullptr, ihipMemsetDataTypeShort));
}

hipError_t hipMemsetD16Async(hipDeviceptr_t dst, unsigned short value, size_t count, hipStream_t stream ){
    HIP_INIT_SPECIAL_API(hipMemsetD16Async, (TRACE_MCMD), dst, value, count, stream);
    return ihipLogStatus(ihipMemsetAsync(dst, value, count, stream, ihipMemsetDataTypeShort));
}

hipError_t hipMemsetD32(hipDeviceptr_t dst, int value, size_t count) {
    HIP_INIT_SPECIAL_API(hipMemsetD32, (TRACE_MCMD), dst, value, count);
    return ihipLogStatus(ihipMemsetSync(dst, value, count, nullptr, ihipMemsetDataTypeInt));
}

hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent) {
    HIP_INIT_SPECIAL_API(hipMemset3D, (TRACE_MCMD), &pitchedDevPtr, value, &extent);
    size_t sizeBytes = pitchedDevPtr.pitch * extent.height * extent.depth;
    return ihipLogStatus(ihipMemsetSync(pitchedDevPtr.ptr, value, sizeBytes, nullptr, ihipMemsetDataTypeChar));
}

hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent ,hipStream_t stream ) {
    HIP_INIT_SPECIAL_API(hipMemset3DAsync, (TRACE_MCMD), &pitchedDevPtr, value, &extent);
    size_t sizeBytes = pitchedDevPtr.pitch * extent.height * extent.depth;
    return ihipLogStatus(ihipMemsetAsync(pitchedDevPtr.ptr, value, sizeBytes, stream, ihipMemsetDataTypeChar));
}

hipError_t hipMemGetInfo(size_t* free, size_t* total) {
    HIP_INIT_API(hipMemGetInfo, free, total);

    hipError_t e = hipSuccess;

    ihipCtx_t* ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        auto device = ctx->getWriteableDevice();
        if (total) {
            *total = device->_props.totalGlobalMem;
        } else {
            e = hipErrorInvalidValue;
        }

        if (free) {
		if (!device->_driver_node_id) return ihipLogStatus(hipErrorInvalidDevice);

		std::string fileName = std::string("/sys/class/kfd/kfd/topology/nodes/") + std::to_string(device->_driver_node_id) + std::string("/mem_banks/0/used_memory");
		std::ifstream file;
		file.open(fileName);
		if (!file) return ihipLogStatus(hipErrorFileNotFound);

                std::string deviceSize;
		size_t deviceMemSize;

		file >> deviceSize;
		file.close();
                if ((deviceMemSize=strtol(deviceSize.c_str(),NULL,10))){
		    *free = device->_props.totalGlobalMem - deviceMemSize;
		    // Deduct the amount of memory from the free memory reported from the system
		    if (HIP_HIDDEN_FREE_MEM) *free -= (size_t)HIP_HIDDEN_FREE_MEM * 1024 * 1024;
		} else {
 		    return ihipLogStatus(hipErrorInvalidValue);
		}
        } else {
            e = hipErrorInvalidValue;
        }

    } else {
        e = hipErrorInvalidDevice;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
    HIP_INIT_API(hipMemPtrGetInfo, ptr, size);

    return ihipLogStatus(ihipMemPtrGetInfo(ptr, size));
}


hipError_t hipFree(void* ptr) {
    HIP_INIT_SPECIAL_API(hipFree, (TRACE_MEM), ptr);

    hipError_t hipStatus = hipErrorInvalidDevicePointer;

    if (ptr) {
        hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if (status == AM_SUCCESS) {
            /*if (amPointerInfo._hostPointer == NULL) */ //TODO: Fix it when there is proper managed memory support
            {
                if (HIP_SYNC_FREE) {
                    // Synchronize all devices, all streams
                    // to ensure all work has finished on all devices.
                    // This is disabled by default.
                    for (unsigned i = 0; i < g_deviceCnt; i++) {
                        ihipGetPrimaryCtx(i)->locked_waitAllStreams();
                    }
                }
                else {
                    ihipCtx_t* ctx;
                    if (amPointerInfo._appId != -1) {
#if USE_APP_PTR_FOR_CTX
                        ctx = static_cast<ihipCtx_t*>(amPointerInfo._appPtr);
#else
                        ctx = ihipGetPrimaryCtx(amPointerInfo._appId);
#endif
                    } else {
                        ctx = ihipGetTlsDefaultCtx();
                    }
                    // Synchronize to ensure all work has finished on device owning the memory.
                    ctx->locked_waitAllStreams();  // ignores non-blocking streams, this waits
                                                   // for all activity to finish.
                }
                hc::am_free(ptr);
                hipStatus = hipSuccess;
            }
        }
    } else {
        // free NULL pointer succeeds and is common technique to initialize runtime
        hipStatus = hipSuccess;
    }

    return ihipLogStatus(hipStatus);
}


hipError_t hipHostFree(void* ptr) {
    HIP_INIT_SPECIAL_API(hipHostFree, (TRACE_MEM), ptr);

    hipError_t hipStatus = hipSuccess;
    hipStatus = hip_internal::ihipHostFree(tls, ptr);

    return ihipLogStatus(hipStatus);
};


// Deprecated:
hipError_t hipFreeHost(void* ptr) { return hipHostFree(ptr); }

hipError_t hipFreeArray(hipArray* array) {
    HIP_INIT_SPECIAL_API(hipFreeArray, (TRACE_MEM), array);

    hipError_t hipStatus = hipErrorInvalidDevicePointer;

    // Synchronize to ensure all work has finished.
    ihipGetTlsDefaultCtx()->locked_waitAllStreams();  // ignores non-blocking streams, this waits
                                                      // for all activity to finish.

    if (array->data) {
        hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, array->data);
        if (status == AM_SUCCESS) {
            if (amPointerInfo._hostPointer == NULL) {
                hc::am_free(array->data);
                hipStatus = hipSuccess;
            }
        }
    }

    return ihipLogStatus(hipStatus);
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
    HIP_INIT_API(hipMemGetAddressRange, pbase, psize, dptr);
    hipError_t hipStatus = hipSuccess;
    hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
    hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
    am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, dptr);
    if (status == AM_SUCCESS) {
        *pbase = amPointerInfo._devicePointer;
        *psize = amPointerInfo._sizeBytes;
    } else
        hipStatus = hipErrorInvalidDevicePointer;
    return ihipLogStatus(hipStatus);
}


// TODO: IPC implementaiton:

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
    HIP_INIT_API(hipIpcGetMemHandle, handle, devPtr);
    hipError_t hipStatus = hipSuccess;
    // Get the size of allocated pointer
    size_t psize = 0u;
    hc::accelerator acc;
    if ((handle == NULL) || (devPtr == NULL)) {
        hipStatus = hipErrorInvalidResourceHandle;
    } else {
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, devPtr);
        if (status == AM_SUCCESS) {
            psize = (size_t)amPointerInfo._sizeBytes;
        } else {
            hipStatus = hipErrorInvalidResourceHandle;
        }
        ihipIpcMemHandle_t* iHandle = (ihipIpcMemHandle_t*)handle;
        // Save the size of the pointer to hipIpcMemHandle
        iHandle->psize = psize;

#if USE_IPC
        // Create HSA ipc memory
        hsa_status_t hsa_status =
            hsa_amd_ipc_memory_create(devPtr, psize, (hsa_amd_ipc_memory_t*)&(iHandle->ipc_handle));
        if (hsa_status != HSA_STATUS_SUCCESS) hipStatus = hipErrorMemoryAllocation;
#else
        hipStatus = hipErrorRuntimeOther;
#endif
    }
    return ihipLogStatus(hipStatus);
}

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
    HIP_INIT_API(hipIpcOpenMemHandle, devPtr, &handle, flags);
    hipError_t hipStatus = hipSuccess;
    if (devPtr == NULL)
        return ihipLogStatus(hipErrorInvalidValue);

#if USE_IPC
    // Get the current device agent.
    hc::accelerator acc;
    hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());
    if (!agent)
        return ihipLogStatus(hipErrorInvalidResourceHandle);

    ihipIpcMemHandle_t* iHandle = (ihipIpcMemHandle_t*)&handle;
    // Attach ipc memory
    auto ctx = ihipGetTlsDefaultCtx();
    {
        LockedAccessor_CtxCrit_t crit(ctx->criticalData());
        auto device = ctx->getWriteableDevice();
        // the peerCnt always stores self so make sure the trace actually
        if(hsa_amd_ipc_memory_attach(
            (hsa_amd_ipc_memory_t*)&(iHandle->ipc_handle), iHandle->psize, crit->peerCnt(),
             crit->peerAgents(), devPtr) != HSA_STATUS_SUCCESS)
            return ihipLogStatus(hipErrorRuntimeOther);

        hc::AmPointerInfo ampi(NULL, *devPtr, *devPtr, sizeof(*devPtr), acc, true, true);
        am_status_t am_status = hc::am_memtracker_add(*devPtr,ampi);
        if (am_status != AM_SUCCESS)
            return ihipLogStatus(hipErrorMapBufferObjectFailed);

#if USE_APP_PTR_FOR_CTX
        am_status = hc::am_memtracker_update(*devPtr, device->_deviceId, 0, ctx);
#else
        am_status = hc::am_memtracker_update(*devPtr, device->_deviceId, 0);
#endif
        if(am_status != AM_SUCCESS)
            return ihipLogStatus(hipErrorMapBufferObjectFailed);
    }
#else
    hipStatus = hipErrorRuntimeOther;
#endif

    return ihipLogStatus(hipStatus);
}

hipError_t hipIpcCloseMemHandle(void* devPtr) {
    HIP_INIT_API(hipIpcCloseMemHandle, devPtr);
    hipError_t hipStatus = hipSuccess;
    if (devPtr == NULL)
        return ihipLogStatus(hipErrorInvalidValue);

#if USE_IPC
    if(hc::am_memtracker_remove(devPtr) != AM_SUCCESS)
        return ihipLogStatus(hipErrorInvalidValue);

    if (hsa_amd_ipc_memory_detach(devPtr) != HSA_STATUS_SUCCESS)
        return ihipLogStatus(hipErrorInvalidResourceHandle);
#else
    hipStatus = hipErrorRuntimeOther;
#endif

    return ihipLogStatus(hipStatus);
}

// hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle){
//     return hipSuccess;
// }
