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

#include "hip/hip_runtime.h"
#include "hip/hcc_detail/elfio/elfio.hpp"
#include "hip/hcc_detail/hsa_helpers.hpp"
#include "hip/hcc_detail/program_state.hpp"
#include "hip_hcc_internal.h"
#include "hip/hip_ext.h"
#include "program_state.inl"
#include "trace_helper.h"
#include "hc_am.hpp"

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "../include/hip/hcc_detail/code_object_bundle.hpp"
#include "hip_fatbin.h"
// TODO Use Pool APIs from HCC to get memory regions.

using namespace ELFIO;
using namespace std;

// For HIP implicit kernargs.
static const size_t HIP_IMPLICIT_KERNARG_SIZE = 56;
static const size_t HIP_IMPLICIT_KERNARG_ALIGNMENT = 8;

struct amd_kernel_code_v3_t {
  uint32_t group_segment_fixed_size;
  uint32_t private_segment_fixed_size;
  uint8_t reserved0[8];
  int64_t kernel_code_entry_byte_offset;
  uint8_t reserved1[24];
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint16_t kernel_code_properties;
  uint8_t reserved2[6];
};

// calculate MD5 checksum
inline std::string checksum(size_t size, const char *source) {
    // FNV-1a hashing, 64-bit version
    const uint64_t FNV_prime = 0x100000001b3;
    const uint64_t FNV_basis = 0xcbf29ce484222325;
    uint64_t hash = FNV_basis;

    const char *str = static_cast<const char *>(source);
    for (auto i = 0; i < size; ++i) {
        hash ^= *str++;
        hash *= FNV_prime;
    }
    return std::to_string(hash);
}

inline uint64_t alignTo(uint64_t Value, uint64_t Align, uint64_t Skew = 0) {
    assert(Align != 0u && "Align can't be 0.");
    Skew %= Align;
    return (Value + Align - 1 - Skew) / Align * Align + Skew;
}


struct ihipKernArgInfo {
    vector<uint32_t> Size;
    vector<uint32_t> Align;
    vector<string> ArgType;
    vector<string> ArgName;
    uint32_t totalSize;
};

map<string, ihipKernArgInfo> kernelArguments;

struct ihipModuleSymbol_t {
    uint64_t _object{};  // The kernel object.
    amd_kernel_code_t const* _header{};
    string _name;  // TODO - review for performance cost.  Name is just used for debug.
    vector<pair<size_t, size_t>> _kernarg_layout{};
    bool _is_code_object_v3{};
};

template <>
string ToString(hipFunction_t v) {
    std::ostringstream ss;
    ss << "0x" << std::hex << v->_object;
    return ss.str();
};

const std::string& FunctionSymbol(const hipFunction_t f) { return f->_name; };

extern hipError_t ihipGetDeviceProperties(hipDeviceProp_t* props, int device);

#define CHECK_HSA(hsaStatus, hipStatus)                                                            \
    if (hsaStatus != HSA_STATUS_SUCCESS) {                                                         \
        return hipStatus;                                                                          \
    }

#define CHECKLOG_HSA(hsaStatus, hipStatus)                                                         \
    if (hsaStatus != HSA_STATUS_SUCCESS) {                                                         \
        return ihipLogStatus(hipStatus);                                                           \
    }

hipError_t ihipModuleLaunchKernel(TlsData *tls, hipFunction_t f, uint32_t globalWorkSizeX,
                                  uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                  uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                  uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                  hipStream_t hStream, void** kernelParams, void** extra,
                                  hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags, bool isStreamLocked = 0,
                                  void** impCoopParams = 0) {
    using namespace hip_impl;

    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t ret = hipSuccess;

    if (ctx == nullptr) {
        ret = hipErrorInvalidDevice;

    } else {
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t* currentDevice = ihipGetDevice(deviceId);
        hsa_agent_t gpuAgent = (hsa_agent_t)currentDevice->_hsaAgent;

        std::vector<char> kernargs{};
        if (kernelParams) {
            if (extra) return hipErrorInvalidValue;

            for (auto&& x : f->_kernarg_layout) {
                const auto p{static_cast<const char*>(*kernelParams)};

                kernargs.insert(
                    kernargs.cend(),
                    round_up_to_next_multiple_nonnegative(
                        kernargs.size(), x.second) - kernargs.size(),
                    '\0');
                kernargs.insert(kernargs.cend(), p, p + x.first);

                ++kernelParams;
            }
        } else if (extra) {
            if (extra[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER &&
                extra[2] == HIP_LAUNCH_PARAM_BUFFER_SIZE && extra[4] == HIP_LAUNCH_PARAM_END) {
                auto args = (char*)extra[1];
                size_t argSize = *(size_t*)(extra[3]);
                kernargs.insert(kernargs.end(), args, args+argSize);
            } else {
                return hipErrorNotInitialized;
            }

        } else {
            return hipErrorInvalidValue;
        }

        // Insert 56-bytes at the end for implicit kernel arguments and fill with value zero.
        size_t padSize = (~kernargs.size() + 1) & (HIP_IMPLICIT_KERNARG_ALIGNMENT - 1);
        kernargs.insert(kernargs.end(), padSize + HIP_IMPLICIT_KERNARG_SIZE, 0);

        if (impCoopParams) {
            const auto p{static_cast<const char*>(*impCoopParams)};
            // The sixth index is for multi-grid synchronization
            kernargs.insert((kernargs.cend() - padSize - HIP_IMPLICIT_KERNARG_SIZE) + 6 * HIP_IMPLICIT_KERNARG_ALIGNMENT,
                            p, p + HIP_IMPLICIT_KERNARG_ALIGNMENT);
        }

        /*
          Kernel argument preparation.
        */
        grid_launch_parm lp;
        lp.dynamic_group_mem_bytes =
            sharedMemBytes;  // TODO - this should be part of preLaunchKernel.
        hStream = ihipPreLaunchKernel(
            hStream, dim3(globalWorkSizeX/localWorkSizeX, globalWorkSizeY/localWorkSizeY, globalWorkSizeZ/localWorkSizeZ),
            dim3(localWorkSizeX, localWorkSizeY, localWorkSizeZ), &lp, f->_name.c_str(), isStreamLocked);

        hsa_kernel_dispatch_packet_t aql;

        memset(&aql, 0, sizeof(aql));

        // aql.completion_signal._handle = 0;
        // aql.kernarg_address = 0;

        aql.workgroup_size_x = localWorkSizeX;
        aql.workgroup_size_y = localWorkSizeY;
        aql.workgroup_size_z = localWorkSizeZ;
        aql.grid_size_x = globalWorkSizeX;
        aql.grid_size_y = globalWorkSizeY;
        aql.grid_size_z = globalWorkSizeZ;
        if (f->_is_code_object_v3) {
            const auto* header =
                reinterpret_cast<const amd_kernel_code_v3_t*>(f->_header);
            aql.group_segment_size =
                header->group_segment_fixed_size + sharedMemBytes;
            aql.private_segment_size =
                header->private_segment_fixed_size;
        } else {
            aql.group_segment_size =
                f->_header->workgroup_group_segment_byte_size + sharedMemBytes;
            aql.private_segment_size =
                f->_header->workitem_private_segment_byte_size;
        }
        aql.kernel_object = f->_object;
        aql.setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
        aql.header =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
        if((flags & 0x1)== 0 ) {
            //in_order
            aql.header |= (1 << HSA_PACKET_HEADER_BARRIER);
        }

        aql.header |= lp.launch_fence;

        hc::completion_future cf;

        lp.av->dispatch_hsa_kernel(&aql, kernargs.data(), kernargs.size(),
                                   (startEvent || stopEvent) ? &cf : nullptr
#if (__hcc_workweek__ > 17312)
                                   ,
                                   f->_name.c_str()
#endif
        );


        if (startEvent) {
            startEvent->attachToCompletionFuture(&cf, hStream, hipEventTypeStartCommand);
        }
        if (stopEvent) {
            stopEvent->attachToCompletionFuture(&cf, hStream, hipEventTypeStopCommand);
        }

        ihipPostLaunchKernel(f->_name.c_str(), hStream, lp, isStreamLocked);


    }

    return ret;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, uint32_t gridDimX, uint32_t gridDimY,
                                 uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
                                 uint32_t blockDimZ, uint32_t sharedMemBytes, hipStream_t hStream,
                                 void** kernelParams, void** extra) {
    HIP_INIT_API(hipModuleLaunchKernel, f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
                 hStream, kernelParams, extra);
    return ihipLogStatus(ihipModuleLaunchKernel(tls,
        f, blockDimX * gridDimX, blockDimY * gridDimY, gridDimZ * blockDimZ, blockDimX, blockDimY,
        blockDimZ, sharedMemBytes, hStream, kernelParams, extra, nullptr, nullptr, 0));
}

hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags) {
    HIP_INIT_API(hipExtModuleLaunchKernel, f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX,
                 localWorkSizeY, localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra);
    return ihipLogStatus(ihipModuleLaunchKernel(tls,
        f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
        localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags));
}

hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent, hipEvent_t stopEvent) {
    HIP_INIT_API(hipHccModuleLaunchKernel, f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX,
                 localWorkSizeY, localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra);
    return ihipLogStatus(ihipModuleLaunchKernel(tls,
        f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
        localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, 0));
}

__attribute__((visibility("default")))
hipError_t ihipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
                                              int  numDevices, unsigned int  flags, hip_impl::program_state& ps) {
    hipError_t result;

    if ((numDevices > g_deviceCnt) || (launchParamsList == nullptr)) {
        return hipErrorInvalidValue;
    }

    hipFunction_t* kds = reinterpret_cast<hipFunction_t*>(malloc(sizeof(hipFunction_t) * numDevices));
    if (kds == nullptr) {
        return hipErrorNotInitialized;
    }

    // prepare all kernel descriptors for each device as all streams will be locked in the next loop
    for (int i = 0; i < numDevices; ++i) {
        const hipLaunchParams& lp = launchParamsList[i];
        if (lp.stream == nullptr) {
            free(kds);
            return hipErrorNotInitialized;
        }
        kds[i] = ps.kernel_descriptor(reinterpret_cast<std::uintptr_t>(lp.func),
                hip_impl::target_agent(lp.stream));
        if (kds[i] == nullptr) {
            free(kds);
            return hipErrorInvalidValue;
        }
        hip_impl::kernargs_size_align kargs = ps.get_kernargs_size_align(
                reinterpret_cast<std::uintptr_t>(lp.func));
        kds[i]->_kernarg_layout = *reinterpret_cast<const std::vector<std::pair<std::size_t, std::size_t>>*>(
                kargs.getHandle());
    }

    // lock all streams before launching kernels to each device
    for (int i = 0; i < numDevices; ++i) {
        LockedAccessor_StreamCrit_t streamCrit(launchParamsList[i].stream->criticalData(), false);
 #if (__hcc_workweek__ >= 19213)
        streamCrit->_av.acquire_locked_hsa_queue();
 #endif
     }

    GET_TLS();
    // launch kernels for each  device
    for (int i = 0; i < numDevices; ++i) {
        const hipLaunchParams& lp = launchParamsList[i];

        result = ihipModuleLaunchKernel(tls, kds[i],
                lp.gridDim.x * lp.blockDim.x,
                lp.gridDim.y * lp.blockDim.y,
                lp.gridDim.z * lp.blockDim.z,
                lp.blockDim.x, lp.blockDim.y,
                lp.blockDim.z, lp.sharedMem,
                lp.stream, lp.args, nullptr, nullptr, nullptr, 0,
                true /* stream is already locked above and will be unlocked
                        in the below code after launching kernels on all devices*/);
    }

    // unlock all streams
    for (int i = 0; i < numDevices; ++i) {
        launchParamsList[i].stream->criticalData().unlock();
 #if (__hcc_workweek__ >= 19213)
        launchParamsList[i].stream->criticalData()._av.release_locked_hsa_queue();
 #endif
     }

    free(kds);

    return result;
}

namespace {
// kernel for initializing GWS
// nwm1 is the total number of work groups minus 1
__global__ void init_gws(uint nwm1) {
    __ockl_gws_init(nwm1, 0);
}
}

__attribute__((visibility("default")))
hipError_t ihipLaunchCooperativeKernel(const void* f, dim3 gridDim,
        dim3 blockDimX, void** kernelParams, unsigned int sharedMemBytes,
        hipStream_t stream, hip_impl::program_state& ps) {

    hipError_t result;


    if ((f == nullptr) || (stream == nullptr) || (kernelParams == nullptr)) {
        return hipErrorNotInitialized;
    }

    if (!stream->getDevice()->_props.cooperativeLaunch) {
        return hipErrorInvalidConfiguration;
    }

    // Prepare the kernel descriptor for initializing the GWS
    hipFunction_t gwsKD = ps.kernel_descriptor(
            reinterpret_cast<std::uintptr_t>(&init_gws),
            hip_impl::target_agent(stream));

    if (gwsKD == nullptr) {
        return hipErrorInvalidValue;
    }
    hip_impl::kernargs_size_align gwsKargs = ps.get_kernargs_size_align(
                    reinterpret_cast<std::uintptr_t>(&init_gws));

    gwsKD->_kernarg_layout = *reinterpret_cast<const std::vector<
            std::pair<std::size_t, std::size_t>>*>(gwsKargs.getHandle());

    // Prepare the kernel descriptor for the main kernel
    hipFunction_t kd = ps.kernel_descriptor(
            reinterpret_cast<std::uintptr_t>(f),
            hip_impl::target_agent(stream));
    if (kd == nullptr) {
        return hipErrorInvalidValue;
    }
    hip_impl::kernargs_size_align kargs =
            ps.get_kernargs_size_align(
                    reinterpret_cast<std::uintptr_t>(f));

    kd->_kernarg_layout = *reinterpret_cast<const std::vector<
            std::pair<std::size_t, std::size_t>>*>(kargs.getHandle());


    void *gwsKernelParam[1];
    // calculate total number of work groups minus 1 for the main kernel
    uint nwm1 = (gridDim.x * gridDim.y * gridDim.z) - 1;
    gwsKernelParam[0] = &nwm1;

    LockedAccessor_StreamCrit_t streamCrit(stream->criticalData(), false);
#if (__hcc_workweek__ >= 19213)
    streamCrit->_av.acquire_locked_hsa_queue();
#endif

    GET_TLS();
    // launch the init_gws kernel to initialize the GWS
    result = ihipModuleLaunchKernel(tls, gwsKD, 1, 1, 1, 1, 1, 1,
             0, stream, gwsKernelParam, nullptr, nullptr, nullptr, 0, true);

    if (result != hipSuccess) {
        stream->criticalData().unlock();
#if (__hcc_workweek__ >= 19213)
        stream->criticalData()._av.release_locked_hsa_queue();
#endif

        return hipErrorLaunchFailure;
    }

    size_t impCoopArg = 1;
    void* impCoopParams[1];
    impCoopParams[0] = &impCoopArg;

    // launch the main kernel
    result = ihipModuleLaunchKernel(tls, kd,
            gridDim.x * blockDimX.x,
            gridDim.y * blockDimX.y,
            gridDim.z * blockDimX.z,
            blockDimX.x, blockDimX.y, blockDimX.z,
            sharedMemBytes, stream, kernelParams, nullptr, nullptr,
            nullptr, 0, true, impCoopParams);

    stream->criticalData().unlock();
#if (__hcc_workweek__ >= 19213)
    stream->criticalData()._av.release_locked_hsa_queue();
#endif

    return result;
}

__attribute__((visibility("default")))
hipError_t ihipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList,
        int  numDevices, unsigned int  flags, hip_impl::program_state& ps) {

    hipError_t result;

    if (numDevices > g_deviceCnt || launchParamsList == nullptr || numDevices > MAX_COOPERATIVE_GPUs) {
        return hipErrorInvalidValue;
    }

    for (int i = 0; i < numDevices; ++i) {
        if (!launchParamsList[i].stream->getDevice()->_props.cooperativeMultiDeviceLaunch) {
            return hipErrorInvalidConfiguration;
        }
    }

    hipFunction_t* gwsKds = reinterpret_cast<hipFunction_t*>(malloc(sizeof(hipFunction_t) * numDevices));
    hipFunction_t* kds    = reinterpret_cast<hipFunction_t*>(malloc(sizeof(hipFunction_t) * numDevices));

    if (kds == nullptr || gwsKds == nullptr) {
        return hipErrorNotInitialized;
    }

    // prepare all kernel descriptors for initializing the GWS and the main kernels per device
    for (int i = 0; i < numDevices; ++i) {
        const hipLaunchParams& lp = launchParamsList[i];
        if (lp.stream == nullptr) {
            free(gwsKds);
            free(kds);
            return hipErrorNotInitialized;
        }

        gwsKds[i] = ps.kernel_descriptor(reinterpret_cast<std::uintptr_t>(&init_gws),
                hip_impl::target_agent(lp.stream));
        if (gwsKds[i] == nullptr) {
            free(gwsKds);
            free(kds);
            return hipErrorInvalidValue;
        }
        hip_impl::kernargs_size_align gwsKargs = ps.get_kernargs_size_align(
                reinterpret_cast<std::uintptr_t>(&init_gws));
        gwsKds[i]->_kernarg_layout = *reinterpret_cast<const std::vector<std::pair<std::size_t, std::size_t>>*>(
                gwsKargs.getHandle());


        kds[i] = ps.kernel_descriptor(reinterpret_cast<std::uintptr_t>(lp.func),
                hip_impl::target_agent(lp.stream));
        if (kds[i] == nullptr) {
            free(gwsKds);
            free(kds);
            return hipErrorInvalidValue;
        }
        hip_impl::kernargs_size_align kargs = ps.get_kernargs_size_align(
                reinterpret_cast<std::uintptr_t>(lp.func));
        kds[i]->_kernarg_layout = *reinterpret_cast<const std::vector<std::pair<std::size_t, std::size_t>>*>(
                kargs.getHandle());
    }

    mg_sync *mg_sync_ptr = 0;
    mg_info *mg_info_ptr[MAX_COOPERATIVE_GPUs] = {0};

    GET_TLS();
    result = hip_internal::ihipHostMalloc(tls, (void **)&mg_sync_ptr, sizeof(mg_sync), hipHostMallocDefault);
    if (result != hipSuccess) {
        return hipErrorInvalidValue;
    }
    mg_sync_ptr->w0 = 0;
    mg_sync_ptr->w1 = 0;

    uint all_sum = 0;
    for (int i = 0; i < numDevices; ++i) {
        result = hip_internal::ihipHostMalloc(tls, (void **)&mg_info_ptr[i], sizeof(mg_info), hipHostMallocDefault);
        if (result != hipSuccess) {
            hip_internal::ihipHostFree(tls, mg_sync_ptr);
            for (int j = 0; j < i; ++j) {
                hip_internal::ihipHostFree(tls, mg_info_ptr[j]);
            }
            return hipErrorInvalidValue;
        }
        // calculate the sum of sizes of all grids
        const hipLaunchParams& lp = launchParamsList[i];
        all_sum += lp.blockDim.x * lp.blockDim.y * lp.blockDim.z *
                   lp.gridDim.x  * lp.gridDim.y  * lp.gridDim.z;
    }

    // lock all streams before launching the blit kernels for initializing the GWS and main kernels to each device
    for (int i = 0; i < numDevices; ++i) {
        LockedAccessor_StreamCrit_t streamCrit(launchParamsList[i].stream->criticalData(), false);
#if (__hcc_workweek__ >= 19213)
        streamCrit->_av.acquire_locked_hsa_queue();
#endif
    }

    // launch the init_gws kernel to initialize the GWS for each device
    for (int i = 0; i < numDevices; ++i) {
        const hipLaunchParams& lp = launchParamsList[i];

        void *gwsKernelParam[1];
        uint nwm1 = (lp.gridDim.x * lp.gridDim.y * lp.gridDim.z) - 1;
        gwsKernelParam[0] = &nwm1;

        result = ihipModuleLaunchKernel(tls, gwsKds[i], 1, 1, 1, 1, 1, 1,
                0, lp.stream, gwsKernelParam, nullptr, nullptr, nullptr, 0, true);

        if (result != hipSuccess) {
            for (int j = 0; j < numDevices; ++j) {
                launchParamsList[j].stream->criticalData().unlock();
#if (__hcc_workweek__ >= 19213)
                launchParamsList[j].stream->criticalData()._av.release_locked_hsa_queue();
#endif
            }
            hip_internal::ihipHostFree(tls, mg_sync_ptr);
            for (int j = 0; j < numDevices; ++j) {
                hip_internal::ihipHostFree(tls, mg_info_ptr[j]);
            }

            return hipErrorLaunchFailure;
        }
    }

    void* impCoopParams[1];
    ulong prev_sum = 0;
    // launch the main kernels for each device
    for (int i = 0; i < numDevices; ++i) {
        const hipLaunchParams& lp = launchParamsList[i];

        //initialize and setup the implicit kernel argument for multi-grid sync
        mg_info_ptr[i]->mgs       = mg_sync_ptr;
        mg_info_ptr[i]->grid_id   = i;
        mg_info_ptr[i]->num_grids = numDevices;
        mg_info_ptr[i]->all_sum   = all_sum;
        mg_info_ptr[i]->prev_sum  = prev_sum;
        prev_sum += lp.blockDim.x * lp.blockDim.y * lp.blockDim.z *
                    lp.gridDim.x  * lp.gridDim.y  * lp.gridDim.z;


        impCoopParams[0] = &mg_info_ptr[i];

        result = ihipModuleLaunchKernel(tls, kds[i],
                lp.gridDim.x * lp.blockDim.x,
                lp.gridDim.y * lp.blockDim.y,
                lp.gridDim.z * lp.blockDim.z,
                lp.blockDim.x, lp.blockDim.y,
                lp.blockDim.z, lp.sharedMem,
                lp.stream, lp.args, nullptr, nullptr, nullptr, 0,
                true, impCoopParams);

        if (result != hipSuccess) {
            for (int j = 0; j < numDevices; ++j) {
                launchParamsList[j].stream->criticalData().unlock();
#if (__hcc_workweek__ >= 19213)
                launchParamsList[j].stream->criticalData()._av.release_locked_hsa_queue();
#endif
            }
            hip_internal::ihipHostFree(tls, mg_sync_ptr);
            for (int j = 0; j < numDevices; ++j) {
                hip_internal::ihipHostFree(tls, mg_info_ptr[j]);
            }

            return hipErrorLaunchFailure;
        }

    }

    // unlock all streams
    for (int i = 0; i < numDevices; ++i) {
        launchParamsList[i].stream->criticalData().unlock();
#if (__hcc_workweek__ >= 19213)
        launchParamsList[i].stream->criticalData()._av.release_locked_hsa_queue();
#endif
    }

    free(gwsKds);
    free(kds);

    hip_internal::ihipHostFree(tls, mg_sync_ptr);
    for (int j = 0; j < numDevices; ++j) {
        hip_internal::ihipHostFree(tls, mg_info_ptr[j]);
    }

    return result;
}

namespace hip_impl {
    hsa_executable_t executable_for(hipModule_t hmod) {
        return hmod->executable;
    }

    const char* hash_for(hipModule_t hmod) {
        return hmod->hash.c_str();
    }

    hsa_agent_t this_agent() {
        GET_TLS();
        auto ctx = ihipGetTlsDefaultCtx();

        if (!ctx) throw runtime_error{"No active HIP context."};

        auto device = ctx->getDevice();

        if (!device) throw runtime_error{"No device available for HIP."};

        ihipDevice_t* currentDevice = ihipGetDevice(device->_deviceId);

        if (!currentDevice) throw runtime_error{"No active device for HIP."};

        return currentDevice->_hsaAgent;
    }

    struct Agent_global {
        Agent_global() : name(nullptr), address(nullptr), byte_cnt(0) {}
        Agent_global(const char* name, hipDeviceptr_t address, uint32_t byte_cnt) 
            : name(nullptr), address(address), byte_cnt(byte_cnt) {
                if (name)
                    this->name = strdup(name);
        }

        Agent_global& operator=(Agent_global&& t) {
            if (this == &t) return *this;

            if (name) free(name);
            name = t.name;
            address = t.address;
            byte_cnt = t.byte_cnt;

            t.name = nullptr;
            t.address = nullptr;
            t.byte_cnt = 0;

            return *this;
        }

        Agent_global(Agent_global&& t) 
            : name(nullptr), address(nullptr), byte_cnt(0) {
                *this = std::move(t);
        }

        // not needed, delete them to prevent bugs
        Agent_global(const Agent_global&) = delete;
        Agent_global& operator=(Agent_global& t) = delete;

        ~Agent_global() { if (name) free(name); }

        char* name;
        hipDeviceptr_t address;
        uint32_t byte_cnt;
    };

    template<typename ForwardIterator>
    std::pair<hipDeviceptr_t, std::size_t> read_global_description(
        ForwardIterator f, ForwardIterator l, const char* name) {
        const auto it = std::find_if(f, l, [=](const Agent_global& x) {
            return strcmp(x.name, name) == 0;
        });

        return it == l ?
            std::make_pair(nullptr, 0u) : std::make_pair(it->address, it->byte_cnt);
    }

    std::vector<Agent_global> read_agent_globals(hsa_agent_t agent,
                                            hsa_executable_t executable);
    class agent_globals_impl {
        private:
            std::pair<
                std::mutex,
                std::unordered_map<
                    std::string, std::vector<Agent_global>>> globals_from_module;

            std::unordered_map<
                hsa_agent_t,
                std::pair<
                    std::once_flag,
                    std::vector<Agent_global>>> globals_from_process;

        public:

            hipError_t read_agent_global_from_module(hipDeviceptr_t* dptr, size_t* bytes,
                    hipModule_t hmod, const char* name) {
                // the key of the map would the hash of code object associated with the
                // hipModule_t instance
                std::string key(hash_for(hmod));

                if (globals_from_module.second.count(key) == 0) {
                    std::lock_guard<std::mutex> lck{globals_from_module.first};

                    if (globals_from_module.second.count(key) == 0) {
                        globals_from_module.second.emplace(
                                key, read_agent_globals(this_agent(), executable_for(hmod)));
                    }
                }

                const auto it0 = globals_from_module.second.find(key);
                if (it0 == globals_from_module.second.cend()) {
                    hip_throw(
                            std::runtime_error{"agent_globals data structure corrupted."});
                }

                std::tie(*dptr, *bytes) = read_global_description(it0->second.cbegin(),
                        it0->second.cend(), name);
                // HACK for SWDEV-173477
                //
                // For code objects with global symbols of length 0, ROCR runtime's fix
                // may not be working correctly. Therefore the
                // result from read_agent_globals() can't be trusted entirely.
                //
                // As a workaround to tame applications which depend on the existence of
                // global symbols with length 0, always return hipSuccess here.
                //
                // This behavior shall be reverted once ROCR runtime has been fixed to
                // address SWDEV-173477 and SWDEV-190701

                //return *dptr ? hipSuccess : hipErrorNotFound;
                return hipSuccess;                
            }

            hipError_t read_agent_global_from_process(hipDeviceptr_t* dptr, size_t* bytes,
                    const char* name) {

                auto agent = this_agent();

                std::call_once(globals_from_process[agent].first, [this](hsa_agent_t aa) {
                    std::vector<Agent_global> tmp0;
                    for (auto&& executable : hip_impl::get_program_state().impl->get_executables(aa)) {
                        auto tmp1 = read_agent_globals(aa, executable);
                        tmp0.insert(tmp0.end(), make_move_iterator(tmp1.begin()),
                            make_move_iterator(tmp1.end()));
                    }
                    globals_from_process[aa].second = move(move(tmp0));
                }, agent);

                const auto it = globals_from_process.find(agent);

                if (it == globals_from_process.cend()) return hipErrorNotInitialized;

                std::tie(*dptr, *bytes) = read_global_description(it->second.second.cbegin(),
                        it->second.second.cend(), name);

                return *dptr ? hipSuccess : hipErrorNotFound;
            }

    };

    agent_globals::agent_globals() : impl(new agent_globals_impl()) { 
        if (!impl) 
            hip_throw(
                std::runtime_error{"Error when constructing agent global data structures."});
    }
    agent_globals::~agent_globals() { delete impl; }

    hipError_t agent_globals::read_agent_global_from_module(hipDeviceptr_t* dptr, size_t* bytes,
                                             hipModule_t hmod, const char* name) {
        return impl->read_agent_global_from_module(dptr, bytes, hmod, name);
    }

    hipError_t agent_globals::read_agent_global_from_process(hipDeviceptr_t* dptr, size_t* bytes,
                                              const char* name) {
        return impl->read_agent_global_from_process(dptr, bytes, name);
    }

} // Namespace hip_impl.

hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
                              hipModule_t hmod, const char* name) {
    HIP_INIT_API(hipModuleGetGlobal, dptr, bytes, hmod, name);
    if (!dptr || !bytes || !hmod) return hipErrorInvalidValue;

    if (!name) return hipErrorNotInitialized;

    return hip_impl::get_agent_globals().read_agent_global_from_module(dptr, bytes, hmod, name);
}

namespace {
inline void track(const hip_impl::Agent_global& x, hsa_agent_t agent) {
    GET_TLS();
    tprintf(DB_MEM, "  add variable '%s' with ptr=%p size=%u to tracker\n", x.name,
            x.address, x.byte_cnt);

    int deviceIndex =0;
    for ( deviceIndex = 0; deviceIndex < g_deviceCnt; deviceIndex++) {
        if(g_allAgents[deviceIndex] == agent)
           break;
    }
    auto device = ihipGetDevice(deviceIndex - 1);
    hc::AmPointerInfo ptr_info(nullptr, x.address, x.address, x.byte_cnt, device->_acc, true,
                               false);
    hc::am_memtracker_add(x.address, ptr_info);
#if USE_APP_PTR_FOR_CTX
    hc::am_memtracker_update(x.address, device->_deviceId, 0u, ihipGetTlsDefaultCtx());
#else
    hc::am_memtracker_update(x.address, device->_deviceId, 0u);
#endif

}

template <typename Container = vector<hip_impl::Agent_global>>
inline hsa_status_t copy_agent_global_variables(hsa_executable_t, hsa_agent_t agent,
                                                hsa_executable_symbol_t x, void* out) {
    using namespace hip_impl;

    assert(out);

    hsa_symbol_kind_t t = {};
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &t);

    if (t == HSA_SYMBOL_KIND_VARIABLE) {
        hip_impl::Agent_global tmp(name(x).c_str(), address(x), size(x));
        static_cast<Container*>(out)->push_back(std::move(tmp));

        track(static_cast<Container*>(out)->back(),agent);
    }

    return HSA_STATUS_SUCCESS;
}

inline hsa_status_t remove_agent_global_variables(hsa_executable_t, hsa_agent_t agent,
                                                  hsa_executable_symbol_t x, void* unused) {
    hsa_symbol_kind_t t = {};
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &t);

    if (t == HSA_SYMBOL_KIND_VARIABLE) {
        hc::am_memtracker_remove(hip_impl::address(x));
    }

    return HSA_STATUS_SUCCESS;
}

hsa_executable_symbol_t find_kernel_by_name(hsa_executable_t executable, const char* kname,
                                            hsa_agent_t* agent = nullptr) {
    using namespace hip_impl;

    pair<const char*, hsa_executable_symbol_t> r{kname, {}};

    hsa_executable_iterate_agent_symbols(
        executable, agent ? *agent : this_agent(),
        [](hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t x, void* s) {
            auto p = static_cast<pair<const char*, hsa_executable_symbol_t>*>(s);

            if (type(x) != HSA_SYMBOL_KIND_KERNEL) {
                return HSA_STATUS_SUCCESS;
            }
            if (name(x) != p->first) return HSA_STATUS_SUCCESS;

            p->second = x;

            return HSA_STATUS_INFO_BREAK;
        },
        &r);

    return r.second;
}


string read_elf_file_as_string(const void* file) {
    // Precondition: file points to an ELF image that was BITWISE loaded
    //               into process accessible memory, and not one loaded by
    //               the loader. This is because in the latter case
    //               alignment may differ, which will break the size
    //               computation.
    //               the image is Elf64, and matches endianness i.e. it is
    //               Little Endian.
    if (!file) return {};

    auto h = static_cast<const ELFIO::Elf64_Ehdr*>(file);
    auto s = static_cast<const char*>(file);
    // This assumes the common case of SHT being the last part of the ELF.
    auto sz =
        sizeof(ELFIO::Elf64_Ehdr) + h->e_shoff + h->e_shentsize * h->e_shnum;

    return string{s, s + sz};
}

string code_object_blob_for_agent(const void* maybe_bundled_code, hsa_agent_t agent) {
    using namespace hip_impl;

    if (!maybe_bundled_code) return {};

    Bundled_code_header tmp{maybe_bundled_code};

    if (!valid(tmp)) return {};

    const auto agent_isa = isa(agent);

    const auto it = find_if(bundles(tmp).cbegin(), bundles(tmp).cend(), [=](const Bundled_code& x) {
        return agent_isa == triple_to_hsa_isa(x.triple);
        ;
    });

    if (it == bundles(tmp).cend()) return {};

    return string{it->blob.cbegin(), it->blob.cend()};
}
} // Unnamed namespace.

namespace hip_impl {
    vector<Agent_global> read_agent_globals(hsa_agent_t agent,
                                            hsa_executable_t executable) {
        vector<Agent_global> r;

        hsa_executable_iterate_agent_symbols(
            executable, agent, copy_agent_global_variables, &r);

        return r;
    }
    void remove_agent_globals_from_tracker(hsa_agent_t agent, hsa_executable_t executable) {
        hsa_executable_iterate_agent_symbols(executable, agent, remove_agent_global_variables, NULL);
    }
} // Namespace hip_impl.

hipError_t hipModuleUnload(hipModule_t hmod) {
    HIP_INIT_API(hipModuleUnload, hmod);

    // TODO - improve this synchronization so it is thread-safe.
    // Currently we want for all inflight activity to complete, but don't prevent another
    // thread from launching new kernels before we finish this operation.
    ihipSynchronize(tls);

    // deleting ihipModule_t does not remove agent globals from hc_am memtracker
    hip_impl::remove_agent_globals_from_tracker(hip_impl::this_agent(), hip_impl::executable_for(hmod));

    delete hmod;  // The ihipModule_t dtor will clean everything up.
    hmod = nullptr;

    return ihipLogStatus(hipSuccess);
}

hipError_t ihipModuleGetFunction(TlsData *tls, hipFunction_t* func, hipModule_t hmod, const char* name,
                                 hsa_agent_t *agent = nullptr) {
    using namespace hip_impl;

    if (!func || !name) return hipErrorInvalidValue;

    auto ctx = ihipGetTlsDefaultCtx();

    if (!ctx) return hipErrorInvalidContext;

    *func = new ihipModuleSymbol_t;

    if (!*func) return hipErrorInvalidValue;

    std::string name_str(name);
    auto kernel = find_kernel_by_name(hmod->executable, name_str.c_str(), agent);

    if (kernel.handle == 0u) {
        name_str.append(".kd");
        kernel = find_kernel_by_name(hmod->executable, name_str.c_str(), agent);
    }

    if (kernel.handle == 0u) return hipErrorNotFound;

    // TODO: refactor the whole ihipThisThat, which is a mess and yields the
    //       below, due to hipFunction_t being a pointer to ihipModuleSymbol_t.
    func[0][0] = *static_cast<hipFunction_t>(
        Kernel_descriptor{kernel_object(kernel), name_str, hmod->kernargs[name_str]});

    return hipSuccess;
}

// Get kernel for the current hsa agent.
hipError_t hipModuleGetFunction(hipFunction_t* hfunc, hipModule_t hmod, const char* name) {
    HIP_INIT_API(hipModuleGetFunction, hfunc, hmod, name);
    return ihipLogStatus(ihipModuleGetFunction(tls, hfunc, hmod, name));
}

// Get kernel for the given hsa agent. Internal use only.
hipError_t hipModuleGetFunctionEx(hipFunction_t* hfunc, hipModule_t hmod,
                                  const char* name, hsa_agent_t *agent) {
    HIP_INIT_API(hipModuleGetFunctionEx, hfunc, hmod, name, agent);
    return ihipLogStatus(ihipModuleGetFunction(tls, hfunc, hmod, name, agent));
}

namespace {
const amd_kernel_code_v3_t *header_v3(const ihipModuleSymbol_t& kd) {
  return reinterpret_cast<const amd_kernel_code_v3_t*>(kd._header);
}

hipFuncAttributes make_function_attributes(TlsData *tls, const ihipModuleSymbol_t& kd) {
    hipFuncAttributes r{};

    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, ihipGetTlsDefaultCtx()->getDevice()->_deviceId);
    // TODO: at the moment there is no way to query the count of registers
    //       available per CU, therefore we hardcode it to 64 KiRegisters.
    prop.regsPerBlock = prop.regsPerBlock ? prop.regsPerBlock : 64 * 1024;

    if (kd._is_code_object_v3) {
        r.localSizeBytes = header_v3(kd)->private_segment_fixed_size;
        r.sharedSizeBytes = header_v3(kd)->group_segment_fixed_size;
        r.numRegs = ((header_v3(kd)->compute_pgm_rsrc1 & 0x3F) + 1) << 2;
        r.binaryVersion = 0; // FIXME: should it be the ISA version or code
                             //        object format version?
    } else {
        r.localSizeBytes = kd._header->workitem_private_segment_byte_size;
        r.sharedSizeBytes = kd._header->workgroup_group_segment_byte_size;
        r.numRegs = kd._header->workitem_vgpr_count;
        r.binaryVersion =
            kd._header->amd_machine_version_major * 10 +
            kd._header->amd_machine_version_minor;
    }
    r.maxDynamicSharedSizeBytes = prop.sharedMemPerBlock - r.sharedSizeBytes;
    r.maxThreadsPerBlock = r.numRegs ?
        std::min(prop.maxThreadsPerBlock, prop.regsPerBlock / r.numRegs) :
        prop.maxThreadsPerBlock;
    r.ptxVersion = prop.major * 10 + prop.minor; // HIP currently presents itself as PTX 3.0.

    return r;
}
} // Unnamed namespace.

hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func)
{
    HIP_INIT_API(hipFuncGetAttributes, attr, func);
    using namespace hip_impl;

    if (!attr) return ihipLogStatus(hipErrorInvalidValue);
    if (!func) return ihipLogStatus(hipErrorInvalidDeviceFunction);

    auto agent = this_agent();
    auto kd = get_program_state().kernel_descriptor(reinterpret_cast<uintptr_t>(func), agent);

    if (!kd->_header) throw runtime_error{"Ill-formed Kernel_descriptor."};

    *attr = make_function_attributes(tls, *kd);

    return ihipLogStatus(hipSuccess);
}

hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc)
{
    HIP_INIT_API(hipFuncGetAttribute, value, attrib, hfunc);
    using namespace hip_impl;
    
    hipError_t retVal = hipSuccess;
    if (!value) return ihipLogStatus(hipErrorInvalidValue);
    hipFuncAttributes attr{};
    attr = make_function_attributes(tls, *hfunc);
    switch(attrib) {
        case HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
            *value = (int) attr.sharedSizeBytes;
            break;
        case HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            *value = attr.maxThreadsPerBlock;
            break;
        case HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
            *value = (int) attr.constSizeBytes;
            break;
        case HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
            *value = (int) attr.localSizeBytes;
            break;
        case HIP_FUNC_ATTRIBUTE_NUM_REGS:
            *value = attr.numRegs;
            break;
        case HIP_FUNC_ATTRIBUTE_PTX_VERSION:
            *value = attr.ptxVersion;
            break;
        case HIP_FUNC_ATTRIBUTE_BINARY_VERSION:
            *value = attr.binaryVersion;
            break;
        case HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA:
            *value = attr.cacheModeCA;
            break;
        case HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
            *value = attr.maxDynamicSharedSizeBytes;
            break;
        case HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
            *value = attr.preferredShmemCarveout;
            break;
        default:
             retVal = hipErrorInvalidValue;
    }
    return ihipLogStatus(retVal);
}

hipError_t ihipModuleLoadData(TlsData *tls, hipModule_t* module, const void* image) {
    using namespace hip_impl;

    if (!module) return hipErrorInvalidValue;

    *module = new ihipModule_t;

    auto ctx = ihipGetTlsDefaultCtx();
    if (!ctx) return hipErrorInvalidContext;

    // try extracting code object from image as fatbin.
    char name[64] = {};
    hsa_agent_get_info(this_agent(), HSA_AGENT_INFO_NAME, name);
    if (auto *code_obj = __hipExtractCodeObjectFromFatBinary(image, name))
      image = code_obj;

    hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                              &(*module)->executable);

    auto tmp = code_object_blob_for_agent(image, this_agent());

    auto content = tmp.empty() ? read_elf_file_as_string(image) : tmp;

    (*module)->executable = get_program_state().load_executable(
                                            content.data(), content.size(), (*module)->executable,
                                            this_agent());

    program_state_impl::read_kernarg_metadata(content, (*module)->kernargs);

    // compute the hash of the code object
    (*module)->hash = checksum(content.length(), content.data());

    return (*module)->executable.handle ? hipSuccess : hipErrorUnknown;
}

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
    HIP_INIT_API(hipModuleLoadData, module, image);
    return ihipLogStatus(ihipModuleLoadData(tls,module,image));
}

hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
    HIP_INIT_API(hipModuleLoad, module, fname);

    if (!fname) return ihipLogStatus(hipErrorInvalidValue);

    ifstream file{fname};

    if (!file.is_open()) return ihipLogStatus(hipErrorFileNotFound);

    vector<char> tmp{istreambuf_iterator<char>{file}, istreambuf_iterator<char>{}};

    return ihipLogStatus(ihipModuleLoadData(tls, module, tmp.data()));
}

hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions,
                               hipJitOption* options, void** optionValues) {
    HIP_INIT_API(hipModuleLoadDataEx, module, image, numOptions, options, optionValues);
    return ihipLogStatus(ihipModuleLoadData(tls, module, image));
}

hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name) {
    using namespace hip_impl;

    HIP_INIT_API(hipModuleGetTexRef, texRef, hmod, name);

    hipError_t ret = hipErrorNotFound;
    if (!texRef) return ihipLogStatus(hipErrorInvalidValue);

    if (!hmod || !name) return ihipLogStatus(hipErrorNotInitialized);
    
    auto addr = get_program_state().global_addr_by_name(name);
    if (addr == nullptr) return ihipLogStatus(hipErrorInvalidValue);

    *texRef = reinterpret_cast<textureReference*>(addr);
    return ihipLogStatus(hipSuccess);
}

void getGprsLdsUsage(hipFunction_t f, size_t* usedVGPRS, size_t* usedSGPRS, size_t* usedLDS)
{
    if (f->_is_code_object_v3) {
        const auto header = reinterpret_cast<const amd_kernel_code_v3_t*>(f->_header);
        // GRANULATED_WAVEFRONT_VGPR_COUNT is specified in 0:5 bits of COMPUTE_PGM_RSRC1
        // the granularity for gfx6-gfx9 is max(0, ceil(vgprs_used / 4) - 1)
        *usedVGPRS = ((header->compute_pgm_rsrc1 & 0x3F) + 1) << 2;
        // GRANULATED_WAVEFRONT_SGPR_COUNT is specified in 6:9 bits of COMPUTE_PGM_RSRC1
        // the granularity for gfx9+ is 2 * max(0, ceil(sgprs_used / 16) - 1)
        *usedSGPRS = ((((header->compute_pgm_rsrc1 & 0x3C0) >> 6) >> 1) + 1) << 4;
        *usedLDS = header->group_segment_fixed_size;
    }
    else {
        const auto header = f->_header;
        // VGPRs granularity is 4
        *usedVGPRS = ((header->workitem_vgpr_count + 3) >> 2) << 2;
        // adding 2 to take into account the 2 VCC registers & handle the granularity of 16
        *usedSGPRS = header->wavefront_sgpr_count + 2;
        *usedSGPRS = ((*usedSGPRS + 15) >> 4) << 4;
        *usedLDS = header->workgroup_group_segment_byte_size;
    }
}

hipError_t ihipOccupancyMaxPotentialBlockSize(TlsData *tls, uint32_t* gridSize, uint32_t* blockSize,
                                              hipFunction_t f, size_t dynSharedMemPerBlk,
                                              uint32_t blockSizeLimit)
{
    using namespace hip_impl;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx == nullptr) {
        return hipErrorInvalidDevice;
    }

    hipDeviceProp_t prop{};
    ihipGetDeviceProperties(&prop, ihipGetTlsDefaultCtx()->getDevice()->_deviceId);

    prop.regsPerBlock = prop.regsPerBlock ? prop.regsPerBlock : 64 * 1024;

    size_t usedVGPRS = 0;
    size_t usedSGPRS = 0;
    size_t usedLDS = 0;
    getGprsLdsUsage(f, &usedVGPRS, &usedSGPRS, &usedLDS);

    // try different workgroup sizes to find the maximum potential occupancy
    // based on the usage of VGPRs and LDS
    size_t wavefrontSize = prop.warpSize;
    size_t maxWavefrontsPerBlock = prop.maxThreadsPerBlock / wavefrontSize;

    // Due to SPI and private memory limitations, the max of wavefronts per CU in 32
    size_t maxWavefrontsPerCU = min(prop.maxThreadsPerMultiProcessor / wavefrontSize, 32);

    const size_t numSIMD = 4;
    size_t maxActivWaves = 0;
    size_t maxWavefronts = 0;
    for (int i = 0; i < maxWavefrontsPerBlock; i++) {
        size_t wavefrontsPerWG = i + 1;

        // workgroup per CU is 40 for WG size of 1 wavefront; otherwise it is 16
        size_t maxWorkgroupPerCU = (wavefrontsPerWG == 1) ? 40 : 16;
        size_t maxWavesWGLimited = min(wavefrontsPerWG * maxWorkgroupPerCU, maxWavefrontsPerCU);

        // Compute VGPR limited wavefronts per block
        size_t wavefrontsVGPRS;
        if (usedVGPRS == 0) {
            wavefrontsVGPRS = maxWavesWGLimited;
        }
        else {
            // find how many VGPRs are available for each SIMD
            size_t numVGPRsPerSIMD = (prop.regsPerBlock / wavefrontSize / numSIMD);
            wavefrontsVGPRS = (numVGPRsPerSIMD / usedVGPRS) * numSIMD;
        }

        size_t maxWavesVGPRSLimited = 0;
        if (wavefrontsVGPRS > maxWavesWGLimited) {
            maxWavesVGPRSLimited = maxWavesWGLimited;
        }
        else {
            maxWavesVGPRSLimited = (wavefrontsVGPRS / wavefrontsPerWG) * wavefrontsPerWG;
        }

        // Compute SGPR limited wavefronts per block
        size_t wavefrontsSGPRS;
        if (usedSGPRS == 0) {
            wavefrontsSGPRS = maxWavesWGLimited;
        }
        else {
            const size_t numSGPRsPerSIMD = (prop.gcnArch < 800) ? 512 : 800;
            wavefrontsSGPRS = (numSGPRsPerSIMD / usedSGPRS) * numSIMD;
        }

        size_t maxWavesSGPRSLimited = 0;
        if (wavefrontsSGPRS > maxWavesWGLimited) {
            maxWavesSGPRSLimited = maxWavesWGLimited;
        }
        else {
            maxWavesSGPRSLimited = (wavefrontsSGPRS / wavefrontsPerWG) * wavefrontsPerWG;
        }

        // Compute LDS limited wavefronts per block
        size_t wavefrontsLDS;
        if (usedLDS == 0) {
            wavefrontsLDS = maxWorkgroupPerCU * wavefrontsPerWG;
        }
        else {
            size_t availableSharedMemPerCU = prop.maxSharedMemoryPerMultiProcessor;
            size_t workgroupPerCU = availableSharedMemPerCU / (usedLDS + dynSharedMemPerBlk);
            wavefrontsLDS = min(workgroupPerCU, maxWorkgroupPerCU) * wavefrontsPerWG;
        }

        size_t maxWavesLDSLimited = min(wavefrontsLDS, maxWavefrontsPerCU);

        size_t activeWavefronts = 0;
        size_t tmp_min = (size_t)min(maxWavesLDSLimited, maxWavesWGLimited);
        tmp_min = min(maxWavesSGPRSLimited, tmp_min);
        activeWavefronts = min(maxWavesVGPRSLimited, tmp_min);

        if (maxActivWaves <= activeWavefronts) {
            maxActivWaves = activeWavefronts;
            maxWavefronts = wavefrontsPerWG;
        }
    }

    // determine the grid and block sizes for maximum potential occupancy
    size_t maxThreadsCnt = prop.maxThreadsPerMultiProcessor*prop.multiProcessorCount;
    if (blockSizeLimit > 0) {
       maxThreadsCnt = min(maxThreadsCnt, blockSizeLimit);
    }

    *blockSize = maxWavefronts * wavefrontSize;
    *gridSize = min((maxThreadsCnt + *blockSize - 1) / *blockSize, prop.multiProcessorCount);

    return hipSuccess;
}

hipError_t hipOccupancyMaxPotentialBlockSize(uint32_t* gridSize, uint32_t* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             uint32_t blockSizeLimit)
{
    HIP_INIT_API(hipOccupancyMaxPotentialBlockSize, gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);

    return ihipLogStatus(ihipOccupancyMaxPotentialBlockSize(tls,
        gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit));
}

hipError_t ihipOccupancyMaxActiveBlocksPerMultiprocessor(
   TlsData *tls, uint32_t* numBlocks, hipFunction_t f, uint32_t blockSize, size_t dynSharedMemPerBlk)
{
    using namespace hip_impl;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx == nullptr) {
        return hipErrorInvalidDevice;
    }

    hipDeviceProp_t prop{};
    ihipGetDeviceProperties(&prop, ihipGetTlsDefaultCtx()->getDevice()->_deviceId);

    prop.regsPerBlock = prop.regsPerBlock ? prop.regsPerBlock : 64 * 1024;

    size_t usedVGPRS = 0;
    size_t usedSGPRS = 0;
    size_t usedLDS = 0;
    getGprsLdsUsage(f, &usedVGPRS, &usedSGPRS, &usedLDS);

    // Due to SPI and private memory limitations, the max of wavefronts per CU in 32
    size_t wavefrontSize = prop.warpSize;
    size_t maxWavefrontsPerCU = min(prop.maxThreadsPerMultiProcessor / wavefrontSize, 32);

    const size_t simdPerCU = 4;
    const size_t maxWavesPerSimd = maxWavefrontsPerCU / simdPerCU;

    size_t numWavefronts = (blockSize + wavefrontSize - 1) / wavefrontSize;

    size_t availableVGPRs = (prop.regsPerBlock / wavefrontSize / simdPerCU);
    size_t vgprs_alu_occupancy = simdPerCU * (usedVGPRS == 0 ? maxWavesPerSimd
        : std::min(maxWavesPerSimd, availableVGPRs / usedVGPRS));

    // Calculate blocks occupancy per CU based on VGPR usage
    *numBlocks = vgprs_alu_occupancy / numWavefronts;

    const size_t availableSGPRs = (prop.gcnArch < 800) ? 512 : 800;
    size_t sgprs_alu_occupancy = simdPerCU * (usedSGPRS == 0 ? maxWavesPerSimd
        : std::min(maxWavesPerSimd, availableSGPRs / usedSGPRS));

    // Calculate blocks occupancy per CU based on SGPR usage
    *numBlocks = std::min(*numBlocks, (uint32_t) (sgprs_alu_occupancy / numWavefronts));

    size_t total_used_lds = usedLDS + dynSharedMemPerBlk;
    if (total_used_lds != 0) {
      // Calculate LDS occupacy per CU. lds_per_cu / (static_lsd + dynamic_lds)
      size_t lds_occupancy = prop.maxSharedMemoryPerMultiProcessor / total_used_lds;
      *numBlocks = std::min(*numBlocks, (uint32_t) lds_occupancy);
    }

    return hipSuccess;
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
   uint32_t* numBlocks, hipFunction_t f, uint32_t blockSize, size_t dynSharedMemPerBlk)
{
    HIP_INIT_API(hipOccupancyMaxActiveBlocksPerMultiprocessor, numBlocks, f, blockSize, dynSharedMemPerBlk);

    return ihipLogStatus(ihipOccupancyMaxActiveBlocksPerMultiprocessor(
        tls, numBlocks, f, blockSize, dynSharedMemPerBlk));
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
   uint32_t* numBlocks, hipFunction_t f, uint32_t  blockSize, size_t dynSharedMemPerBlk,
   unsigned int flags)
{
    HIP_INIT_API(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, numBlocks, f, blockSize, dynSharedMemPerBlk, flags);

    return ihipLogStatus(ihipOccupancyMaxActiveBlocksPerMultiprocessor(
        tls, numBlocks, f, blockSize, dynSharedMemPerBlk));
}

hipError_t hipLaunchKernel(
    const void* func_addr, dim3 numBlocks, dim3 dimBlocks, void** args,
    size_t sharedMemBytes, hipStream_t stream)
{
   HIP_INIT_API(hipLaunchKernel,func_addr,numBlocks,dimBlocks,args,sharedMemBytes,stream);

   hipFunction_t kd = hip_impl::get_program_state().kernel_descriptor((std::uintptr_t)func_addr,
                                                           hip_impl::target_agent(stream));

   if(kd == nullptr || kd->_header == nullptr)
       return ihipLogStatus(hipErrorInvalidValue);

   size_t szKernArg = kd->_header->kernarg_segment_byte_size;

   if(args == NULL && szKernArg != 0)
      return ihipLogStatus(hipErrorInvalidValue);

   void* config[]{
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
	    &szKernArg,
        HIP_LAUNCH_PARAM_END};

   return ihipLogStatus(ihipModuleLaunchKernel(tls, kd, numBlocks.x * dimBlocks.x, numBlocks.y * dimBlocks.y, numBlocks.z * dimBlocks.z,
                          dimBlocks.x, dimBlocks.y, dimBlocks.z, sharedMemBytes, stream, nullptr, (void**)&config, nullptr, nullptr, 0));
}
