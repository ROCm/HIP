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
#include "program_state.inl"
#include "trace_helper.h"

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
static const size_t HIP_IMPLICIT_KERNARG_SIZE = 48;
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

hipError_t hipModuleUnload(hipModule_t hmod) {
    HIP_INIT_API(hipModuleUnload, hmod);

    // TODO - improve this synchronization so it is thread-safe.
    // Currently we want for all inflight activity to complete, but don't prevent another
    // thread from launching new kernels before we finish this operation.
    ihipSynchronize();

    delete hmod;  // The ihipModule_t dtor will clean everything up.
    hmod = nullptr;

    return ihipLogStatus(hipSuccess);
}

hipError_t ihipModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                  uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                  uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                  uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                  hipStream_t hStream, void** kernelParams, void** extra,
                                  hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags, bool isStreamLocked = 0) {
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

        // Insert 48-bytes at the end for implicit kernel arguments and fill with value zero.
        size_t padSize = (~kernargs.size() + 1) & (HIP_IMPLICIT_KERNARG_ALIGNMENT - 1);
        kernargs.insert(kernargs.end(), padSize + HIP_IMPLICIT_KERNARG_SIZE, 0);

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
        bool is_code_object_v3 = f->_name.find(".kd") != std::string::npos;
        if (is_code_object_v3) {
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

        if (HCC_OPT_FLUSH) {
            aql.header |= (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                          (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
        } else {
            aql.header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
        };


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
    return ihipLogStatus(ihipModuleLaunchKernel(
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
    return ihipLogStatus(ihipModuleLaunchKernel(
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
    return ihipLogStatus(ihipModuleLaunchKernel(
        f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
        localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, 0));
}

hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
                                              int  numDevices, unsigned int  flags) {

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
        kds[i] = hip_impl::get_program_state().kernel_descriptor(reinterpret_cast<std::uintptr_t>(lp.func),
                hip_impl::target_agent(lp.stream));
        if (kds[i] == nullptr) {
            free(kds);
            return hipErrorInvalidValue;
        }
        hip_impl::kernargs_size_align kargs = hip_impl::get_program_state().get_kernargs_size_align(
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

    // launch kernels for each  device
    for (int i = 0; i < numDevices; ++i) {
        const hipLaunchParams& lp = launchParamsList[i];

        result = ihipModuleLaunchKernel(kds[i],
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

namespace hip_impl {
    hsa_executable_t executable_for(hipModule_t hmod) {
        return hmod->executable;
    }

    const char* hash_for(hipModule_t hmod) {
        return hmod->hash.c_str();
    }

    hsa_agent_t this_agent() {
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
} // Namespace hip_impl.

hipError_t ihipModuleGetFunction(hipFunction_t* func, hipModule_t hmod, const char* name,
                                 hsa_agent_t *agent = nullptr) {
    using namespace hip_impl;

    if (!func || !name) return hipErrorInvalidValue;

    auto ctx = ihipGetTlsDefaultCtx();

    if (!ctx) return hipErrorInvalidContext;

    *func = new ihipModuleSymbol_t;

    if (!*func) return hipErrorInvalidValue;

    auto kernel = find_kernel_by_name(hmod->executable, name, agent);

    if (kernel.handle == 0u) {
        std::string name_str(name);
        name_str.append(".kd");
        kernel = find_kernel_by_name(hmod->executable, name_str.c_str(), agent);
    }

    if (kernel.handle == 0u) return hipErrorNotFound;

    // TODO: refactor the whole ihipThisThat, which is a mess and yields the
    //       below, due to hipFunction_t being a pointer to ihipModuleSymbol_t.
    func[0][0] = *static_cast<hipFunction_t>(
        Kernel_descriptor{kernel_object(kernel), name, hmod->kernargs[name]});

    return hipSuccess;
}

// Get kernel for the current hsa agent.
hipError_t hipModuleGetFunction(hipFunction_t* hfunc, hipModule_t hmod, const char* name) {
    HIP_INIT_API(hipModuleGetFunction, hfunc, hmod, name);
    return ihipLogStatus(ihipModuleGetFunction(hfunc, hmod, name));
}

// Get kernel for the given hsa agent. Internal use only.
hipError_t hipModuleGetFunctionEx(hipFunction_t* hfunc, hipModule_t hmod,
                                  const char* name, hsa_agent_t *agent) {
    HIP_INIT_API(hipModuleGetFunctionEx, hfunc, hmod, name, agent);
    return ihipLogStatus(ihipModuleGetFunction(hfunc, hmod, name, agent));
}

namespace {
const amd_kernel_code_v3_t *header_v3(const ihipModuleSymbol_t& kd) {
  return reinterpret_cast<const amd_kernel_code_v3_t*>(kd._header);
}

hipFuncAttributes make_function_attributes(const ihipModuleSymbol_t& kd) {
    hipFuncAttributes r{};

    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, ihipGetTlsDefaultCtx()->getDevice()->_deviceId);
    // TODO: at the moment there is no way to query the count of registers
    //       available per CU, therefore we hardcode it to 64 KiRegisters.
    prop.regsPerBlock = prop.regsPerBlock ? prop.regsPerBlock : 64 * 1024;

    bool is_code_object_v3 = kd._name.find(".kd") != std::string::npos;
    if (is_code_object_v3) {
        r.localSizeBytes = header_v3(kd)->private_segment_fixed_size;
        r.sharedSizeBytes = header_v3(kd)->group_segment_fixed_size;
    } else {
        r.localSizeBytes = kd._header->workitem_private_segment_byte_size;
        r.sharedSizeBytes = kd._header->workgroup_group_segment_byte_size;
    }
    r.maxDynamicSharedSizeBytes = prop.sharedMemPerBlock - r.sharedSizeBytes;
    if (is_code_object_v3) {
        r.numRegs = ((header_v3(kd)->compute_pgm_rsrc1 & 0x3F) + 1) << 2;
    } else {
        r.numRegs = kd._header->workitem_vgpr_count;
    }
    r.maxThreadsPerBlock = r.numRegs ?
        std::min(prop.maxThreadsPerBlock, prop.regsPerBlock / r.numRegs) :
        prop.maxThreadsPerBlock;
    if (is_code_object_v3) {
        r.binaryVersion = 0; // FIXME: should it be the ISA version or code
                             //        object format version?
    } else {
        r.binaryVersion =
            kd._header->amd_machine_version_major * 10 +
            kd._header->amd_machine_version_minor;
    }
    r.ptxVersion = prop.major * 10 + prop.minor; // HIP currently presents itself as PTX 3.0.

    return r;
}
} // Unnamed namespace.

hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func)
{
    using namespace hip_impl;

    if (!attr) return hipErrorInvalidValue;
    if (!func) return hipErrorInvalidDeviceFunction;

    auto agent = this_agent();
    auto kd = get_program_state().kernel_descriptor(reinterpret_cast<uintptr_t>(func), agent);

    if (!kd->_header) throw runtime_error{"Ill-formed Kernel_descriptor."};

    *attr = make_function_attributes(*kd);

    return hipSuccess;
}

hipError_t ihipModuleLoadData(hipModule_t* module, const void* image) {
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

    std::vector<char> blob(content.cbegin(), content.cend());
    program_state_impl::read_kernarg_metadata(blob, (*module)->kernargs);

    // compute the hash of the code object
    (*module)->hash = checksum(content.length(), content.data());

    return (*module)->executable.handle ? hipSuccess : hipErrorUnknown;
}

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
    HIP_INIT_API(hipModuleLoadData, module, image);
    return ihipLogStatus(ihipModuleLoadData(module,image));
}

hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
    HIP_INIT_API(hipModuleLoad, module, fname);

    if (!fname) return ihipLogStatus(hipErrorInvalidValue);

    ifstream file{fname};

    if (!file.is_open()) return ihipLogStatus(hipErrorFileNotFound);

    vector<char> tmp{istreambuf_iterator<char>{file}, istreambuf_iterator<char>{}};

    return ihipLogStatus(ihipModuleLoadData(module, tmp.data()));
}

hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions,
                               hipJitOption* options, void** optionValues) {
    HIP_INIT_API(hipModuleLoadDataEx, module, image, numOptions, options, optionValues);
    return ihipLogStatus(ihipModuleLoadData(module, image));
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

hipError_t ihipOccupancyMaxPotentialBlockSize(uint32_t* gridSize, uint32_t* blockSize,
                                              hipFunction_t f, size_t dynSharedMemPerBlk,
                                              uint32_t blockSizeLimit)
{
    using namespace hip_impl;

    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t ret = hipSuccess;

    if (ctx == nullptr) {
        ret = hipErrorInvalidDevice;
    }

    hipDeviceProp_t prop{};
    ihipGetDeviceProperties(&prop, ihipGetTlsDefaultCtx()->getDevice()->_deviceId);

    prop.regsPerBlock = prop.regsPerBlock ? prop.regsPerBlock : 64 * 1024;

    size_t usedVGPRS = 0;
    size_t usedSGPRS = 0;
    size_t usedLDS = 0;
    bool is_code_object_v3 = f->_name.find(".kd") != std::string::npos;
    if (is_code_object_v3) {
        const auto header = reinterpret_cast<const amd_kernel_code_v3_t*>(f->_header);
        // GRANULATED_WAVEFRONT_VGPR_COUNT is specified in 0:5 bits of COMPUTE_PGM_RSRC1
        // the granularity for gfx6-gfx9 is max(0, ceil(vgprs_used / 4) - 1)
        usedVGPRS = ((header->compute_pgm_rsrc1 & 0x3F) + 1) << 2;
        // GRANULATED_WAVEFRONT_SGPR_COUNT is specified in 6:9 bits of COMPUTE_PGM_RSRC1
        // the granularity for gfx9+ is 2 * max(0, ceil(sgprs_used / 16) - 1)
        usedSGPRS = ((((header->compute_pgm_rsrc1 & 0x3C0) >> 6) >> 1) + 1) << 4;
        usedLDS = header->group_segment_fixed_size;
    }
    else {
        const auto header = f->_header;
        // VGPRs granularity is 4
        usedVGPRS = ((header->workitem_vgpr_count + 3) >> 2) << 2;
        // adding 2 to take into account the 2 VCC registers & handle the granularity of 16
        usedSGPRS = header->wavefront_sgpr_count + 2;
        usedSGPRS = ((usedSGPRS + 15) >> 4) << 4;
        usedLDS = header->workgroup_group_segment_byte_size;
    }

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

        if (maxActivWaves < activeWavefronts) {
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

    return ret;
}


hipError_t hipOccupancyMaxPotentialBlockSize(uint32_t* gridSize, uint32_t* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             uint32_t blockSizeLimit)
{
    HIP_INIT_API(hipOccupancyMaxPotentialBlockSize, gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);

    return ihipLogStatus(ihipOccupancyMaxPotentialBlockSize(
        gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit));
}
