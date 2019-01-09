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

#include "elfio/elfio.hpp"
#include "hip/hip_runtime.h"
#include "hip/hcc_detail/program_state.hpp"
#include "hip_hcc_internal.h"
#include "hsa_helpers.hpp"
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
using namespace hip_impl;
using namespace std;

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
};

template <>
string ToString(hipFunction_t v) {
    std::ostringstream ss;
    ss << "0x" << std::hex << v->_object;
    return ss.str();
};

std::string& FunctionSymbol(hipFunction_t f) { return f->_name; };

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
                                  hipEvent_t startEvent, hipEvent_t stopEvent) {
    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t ret = hipSuccess;

    if (ctx == nullptr) {
        ret = hipErrorInvalidDevice;

    } else {
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t* currentDevice = ihipGetDevice(deviceId);
        hsa_agent_t gpuAgent = (hsa_agent_t)currentDevice->_hsaAgent;

        void* config[5] = {0};
        size_t kernArgSize;

        if (kernelParams != NULL) {
            std::string name = f->_name;
            struct ihipKernArgInfo pl = kernelArguments[name];
            char* argBuf = (char*)malloc(pl.totalSize);
            memset(argBuf, 0, pl.totalSize);
            int index = 0;
            for (int i = 0; i < pl.Size.size(); i++) {
                memcpy(argBuf + index, kernelParams[i], pl.Size[i]);
                index += pl.Align[i];
            }
            config[1] = (void*)argBuf;
            kernArgSize = pl.totalSize;
        } else if (extra != NULL) {
            memcpy(config, extra, sizeof(size_t) * 5);
            if (config[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER &&
                config[2] == HIP_LAUNCH_PARAM_BUFFER_SIZE && config[4] == HIP_LAUNCH_PARAM_END) {
                kernArgSize = *(size_t*)(config[3]);
            } else {
                return hipErrorNotInitialized;
            }

        } else {
            return hipErrorInvalidValue;
        }


        /*
          Kernel argument preparation.
        */
        grid_launch_parm lp;
        lp.dynamic_group_mem_bytes =
            sharedMemBytes;  // TODO - this should be part of preLaunchKernel.
        hStream = ihipPreLaunchKernel(
            hStream, dim3(globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ),
            dim3(localWorkSizeX, localWorkSizeY, localWorkSizeZ), &lp, f->_name.c_str());


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
        aql.group_segment_size =
            f->_header->workgroup_group_segment_byte_size + sharedMemBytes;
        aql.private_segment_size =
            f->_header->workitem_private_segment_byte_size;
        aql.kernel_object = f->_object;
        aql.setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
        aql.header =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
            (1 << HSA_PACKET_HEADER_BARRIER);  // TODO - honor queue setting for execute_in_order

        if (HCC_OPT_FLUSH) {
            aql.header |= (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                          (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
        } else {
            aql.header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
        };


        hc::completion_future cf;

        lp.av->dispatch_hsa_kernel(&aql, config[1] /* kernarg*/, kernArgSize,
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


        if (kernelParams != NULL) {
            free(config[1]);
        }
        ihipPostLaunchKernel(f->_name.c_str(), hStream, lp);
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
        blockDimZ, sharedMemBytes, hStream, kernelParams, extra, nullptr, nullptr));
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
        localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent));
}

namespace {
struct Agent_global {
    string name;
    hipDeviceptr_t address;
    uint32_t byte_cnt;
};

inline void track(const Agent_global& x, hsa_agent_t agent) {
    tprintf(DB_MEM, "  add variable '%s' with ptr=%p size=%u to tracker\n", x.name.c_str(),
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

template <typename Container = vector<Agent_global>>
inline hsa_status_t copy_agent_global_variables(hsa_executable_t, hsa_agent_t agent,
                                                hsa_executable_symbol_t x, void* out) {
    assert(out);

    hsa_symbol_kind_t t = {};
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &t);

    if (t == HSA_SYMBOL_KIND_VARIABLE) {
        static_cast<Container*>(out)->push_back(Agent_global{name(x), address(x), size(x)});

        track(static_cast<Container*>(out)->back(),agent);
    }

    return HSA_STATUS_SUCCESS;
}

inline hsa_agent_t this_agent() {
    auto ctx = ihipGetTlsDefaultCtx();

    if (!ctx) throw runtime_error{"No active HIP context."};

    auto device = ctx->getDevice();

    if (!device) throw runtime_error{"No device available for HIP."};

    ihipDevice_t* currentDevice = ihipGetDevice(device->_deviceId);

    if (!currentDevice) throw runtime_error{"No active device for HIP."};

    return currentDevice->_hsaAgent;
}

inline vector<Agent_global> read_agent_globals(hsa_agent_t agent, hsa_executable_t executable) {
    vector<Agent_global> r;

    hsa_executable_iterate_agent_symbols(executable, agent, copy_agent_global_variables, &r);

    return r;
}

template <typename ForwardIterator>
pair<hipDeviceptr_t, size_t> read_global_description(ForwardIterator f, ForwardIterator l,
                                                     const char* name) {
    const auto it = std::find_if(f, l, [=](const Agent_global& x) { return x.name == name; });

    return it == l ? make_pair(nullptr, 0u) : make_pair(it->address, it->byte_cnt);
}

hipError_t read_agent_global_from_module(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                                         const char* name) {
    // the key of the map would the hash of code object associated with the
    // hipModule_t instance
    static unordered_map<std::string, vector<Agent_global>> agent_globals;
    auto key = hmod->hash;

    if (agent_globals.count(key) == 0) {
        static mutex mtx;
        lock_guard<mutex> lck{mtx};

        if (agent_globals.count(key) == 0) {
            agent_globals.emplace(key, read_agent_globals(this_agent(), hmod->executable));
        }
    }

    const auto it0 = agent_globals.find(key);
    if (it0 == agent_globals.cend()) {
        throw runtime_error{"agent_globals data structure corrupted."};
    }

    tie(*dptr, *bytes) = read_global_description(it0->second.cbegin(), it0->second.cend(), name);

    return *dptr ? hipSuccess : hipErrorNotFound;
}

hipError_t read_agent_global_from_process(hipDeviceptr_t* dptr, size_t* bytes, const char* name) {
    static unordered_map<hsa_agent_t, vector<Agent_global>> agent_globals;
    static std::once_flag f;

    call_once(f, []() {
        for (auto&& agent_executables : hip_impl::executables()) {
            vector<Agent_global> tmp0;
            for (auto&& executable : agent_executables.second) {
                auto tmp1 = read_agent_globals(agent_executables.first, executable);
                tmp0.insert(tmp0.end(), make_move_iterator(tmp1.begin()),
                            make_move_iterator(tmp1.end()));
            }
            agent_globals.emplace(agent_executables.first, move(tmp0));
        }
    });

    const auto it = agent_globals.find(this_agent());

    if (it == agent_globals.cend()) return hipErrorNotInitialized;

    tie(*dptr, *bytes) = read_global_description(it->second.cbegin(), it->second.cend(), name);

    return *dptr ? hipSuccess : hipErrorNotFound;
}

hsa_executable_symbol_t find_kernel_by_name(hsa_executable_t executable, const char* kname) {
    pair<const char*, hsa_executable_symbol_t> r{kname, {}};

    hsa_executable_iterate_agent_symbols(
        executable, this_agent(),
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

string read_elf_file_as_string(
    const void* file) {  // Precondition: file points to an ELF image that was BITWISE loaded
    //               into process accessible memory, and not one loaded by
    //               the loader. This is because in the latter case
    //               alignment may differ, which will break the size
    //               computation.
    //               the image is Elf64, and matches endianness i.e. it is
    //               Little Endian.
    if (!file) return {};

    auto h = static_cast<const Elf64_Ehdr*>(file);
    auto s = static_cast<const char*>(file);
    // This assumes the common case of SHT being the last part of the ELF.
    auto sz = sizeof(Elf64_Ehdr) + h->e_shoff + h->e_shentsize * h->e_shnum;

    return string{s, s + sz};
}

string code_object_blob_for_agent(const void* maybe_bundled_code, hsa_agent_t agent) {
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
}  // namespace

hipError_t ihipModuleGetFunction(hipFunction_t* func, hipModule_t hmod, const char* name) {

    if (!func || !name) return hipErrorInvalidValue;

    auto ctx = ihipGetTlsDefaultCtx();

    if (!ctx) return hipErrorInvalidContext;

    *func = new ihipModuleSymbol_t;

    if (!*func) return hipErrorInvalidValue;

    auto kernel = find_kernel_by_name(hmod->executable, name);

    if (kernel.handle == 0u) return hipErrorNotFound;

    // TODO: refactor the whole ihipThisThat, which is a mess and yields the
    //       below, due to hipFunction_t being a pointer to ihipModuleSymbol_t.
    func[0][0] = *static_cast<hipFunction_t>(
        Kernel_descriptor{kernel_object(kernel), name});

    return hipSuccess;
}

hipError_t hipModuleGetFunction(hipFunction_t* hfunc, hipModule_t hmod, const char* name) {
    HIP_INIT_API(hipModuleGetFunction, hfunc, hmod, name);
    return ihipLogStatus(ihipModuleGetFunction(hfunc, hmod, name));
}

hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                              const char* name) {
    HIP_INIT_API(hipModuleGetGlobal, dptr, bytes, hmod, name);

    return ihipLogStatus(ihipModuleGetGlobal(dptr, bytes, hmod, name));
}

hipError_t ihipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                               const char* name) {
    if (!dptr || !bytes) return hipErrorInvalidValue;

    if (!name) return hipErrorNotInitialized;

    const auto r = hmod ? read_agent_global_from_module(dptr, bytes, hmod, name)
                        : read_agent_global_from_process(dptr, bytes, name);

    return r;
}

namespace
{
    inline
    hipFuncAttributes make_function_attributes(const amd_kernel_code_t& header)
    {
        hipFuncAttributes r{};

        hipDeviceProp_t prop{};
        hipGetDeviceProperties(
            &prop, ihipGetTlsDefaultCtx()->getDevice()->_deviceId);
        // TODO: at the moment there is no way to query the count of registers
        //       available per CU, therefore we hardcode it to 64 KiRegisters.
        prop.regsPerBlock = prop.regsPerBlock ? prop.regsPerBlock : 64 * 1024;
        
        r.localSizeBytes = header.workitem_private_segment_byte_size;
        r.sharedSizeBytes = header.workgroup_group_segment_byte_size;
        r.maxDynamicSharedSizeBytes =
            prop.sharedMemPerBlock - r.sharedSizeBytes;
        r.numRegs = header.workitem_vgpr_count;
        r.maxThreadsPerBlock = r.numRegs ?
            std::min(prop.maxThreadsPerBlock, prop.regsPerBlock / r.numRegs) :
            prop.maxThreadsPerBlock;
        r.binaryVersion =
            header.amd_machine_version_major * 10 +
            header.amd_machine_version_minor;
        r.ptxVersion = prop.major * 10 + prop.minor; // HIP currently presents itself as PTX 3.0.

        return r;
    }
}

hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func)
{
    if (!attr) return hipErrorInvalidValue;
    if (!func) return hipErrorInvalidDeviceFunction;

    const auto it0 = functions().find(reinterpret_cast<uintptr_t>(func));

    if (it0 == functions().cend()) return hipErrorInvalidDeviceFunction;

    auto agent = this_agent();
    const auto it1 = find_if(
        it0->second.cbegin(),
        it0->second.cend(),
        [=](const pair<hsa_agent_t, Kernel_descriptor>& x) {
        return x.first == agent;
    });

    if (it1 == it0->second.cend()) return hipErrorInvalidDeviceFunction;

    const auto header = static_cast<hipFunction_t>(it1->second)->_header;

    if (!header) throw runtime_error{"Ill-formed Kernel_descriptor."};

    *attr = make_function_attributes(*header);

    return hipSuccess;
}

hipError_t ihipModuleLoadData(hipModule_t* module, const void* image) {

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

    (*module)->executable = hip_impl::load_executable(content,
                                                      (*module)->executable,
                                                      this_agent());

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
    HIP_INIT_API(hipModuleGetTexRef, texRef, hmod, name);

    hipError_t ret = hipErrorNotFound;
    if (!texRef) return ihipLogStatus(hipErrorInvalidValue);

    if (!hmod || !name) return ihipLogStatus(hipErrorNotInitialized);

    const auto it = globals().find(name);
    if (it == globals().end()) return ihipLogStatus(hipErrorInvalidValue);

    *texRef = reinterpret_cast<textureReference*>(it->second);
    return ihipLogStatus(hipSuccess);
}
