/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <elf.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/amd_hsa_kernel_code.h"

#include "hip/hip_runtime.h"
#include "hip_hcc.h"
#include "trace_helper.h"

//TODO Use Pool APIs from HCC to get memory regions.

#define CHECK_HSA(hsaStatus, hipStatus) \
if (hsaStatus != HSA_STATUS_SUCCESS) {\
    return hipStatus;\
}

#define CHECKLOG_HSA(hsaStatus, hipStatus) \
if (hsaStatus != HSA_STATUS_SUCCESS) {\
    return ihipLogStatus(hipStatus);\
}

namespace hipdrv {

    hsa_status_t findSystemRegions(hsa_region_t region, void *data){
        hsa_region_segment_t segment_id;
        hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment_id);

        if(segment_id != HSA_REGION_SEGMENT_GLOBAL){
            return HSA_STATUS_SUCCESS;
        }

        hsa_region_global_flag_t flags;
        hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);

        hsa_region_t *reg = (hsa_region_t*)data;

        if(flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED){
            *reg = region;
        }

        return HSA_STATUS_SUCCESS;
    }

}   // End namespace hipdrv

uint64_t PrintSymbolSizes(const void *emi, const char *name){
    const Elf64_Ehdr *ehdr = (const Elf64_Ehdr*)emi;
    if(NULL == ehdr || EV_CURRENT != ehdr->e_version){}
    const Elf64_Shdr * shdr = (const Elf64_Shdr*)((char*)emi + ehdr->e_shoff);
    for(uint16_t i=0;i<ehdr->e_shnum;++i){
        if(shdr[i].sh_type == SHT_SYMTAB){
            const Elf64_Sym *syms = (const Elf64_Sym*)((char*)emi + shdr[i].sh_offset);
            assert(syms);
            uint64_t numSyms = shdr[i].sh_size/shdr[i].sh_entsize;
            const char* strtab = (const char*)((char*)emi + shdr[shdr[i].sh_link].sh_offset);
            assert(strtab);
            for(uint64_t i=0;i<numSyms;++i){
                const char *symname = strtab + syms[i].st_name;
                assert(symname);
                uint64_t size = syms[i].st_size;
                if(strcmp(name, symname) == 0){
                    return size;
                }
            }
        }
    }
    return 0;
}

uint64_t ElfSize(const void *emi){
    const Elf64_Ehdr *ehdr = (const Elf64_Ehdr*)emi;
    const Elf64_Shdr *shdr = (const Elf64_Shdr*)((char*)emi + ehdr->e_shoff);

    uint64_t max_offset = ehdr->e_shoff;
    uint64_t total_size = max_offset + ehdr->e_shentsize * ehdr->e_shnum;

    for(uint16_t i=0;i < ehdr->e_shnum;++i){
        uint64_t cur_offset = static_cast<uint64_t>(shdr[i].sh_offset);
        if(max_offset < cur_offset){
            max_offset = cur_offset;
            total_size = max_offset;
            if(SHT_NOBITS != shdr[i].sh_type){
                total_size += static_cast<uint64_t>(shdr[i].sh_size);
            }
        }
    }
    return total_size;
}


hipError_t hipModuleLoad(hipModule_t *module, const char *fname){
    HIP_INIT_API(module, fname);
    hipError_t ret = hipSuccess;
    *module = new ihipModule_t;

    if(module == NULL){
        return ihipLogStatus(hipErrorInvalidValue);
    }

    auto ctx = ihipGetTlsDefaultCtx();
    if(ctx == nullptr){
        ret = hipErrorInvalidContext;

    }else{
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t *currentDevice = ihipGetDevice(deviceId);
        std::ifstream in(fname, std::ios::binary | std::ios::ate);

        if(!in.is_open() ){
            return ihipLogStatus(hipErrorFileNotFound);

        } else {

            size_t size = std::string::size_type(in.tellg());
            void *p = NULL;
            hsa_agent_t agent = currentDevice->_hsaAgent;
            hsa_region_t sysRegion;
            hsa_status_t status = hsa_agent_iterate_regions(agent, hipdrv::findSystemRegions, &sysRegion);
            status = hsa_memory_allocate(sysRegion, size, (void**)&p);

            if(status != HSA_STATUS_SUCCESS){
                return ihipLogStatus(hipErrorOutOfMemory);
            }

            char *ptr = (char*)p;
            if(!ptr){
                return ihipLogStatus(hipErrorOutOfMemory);
            }
            (*module)->ptr = p;
            (*module)->size = size;
            in.seekg(0, std::ios::beg);
            std::copy(std::istreambuf_iterator<char>(in),
                      std::istreambuf_iterator<char>(), ptr);

            status = hsa_code_object_deserialize(ptr, size, NULL, &(*module)->object);

            if(status != HSA_STATUS_SUCCESS){
                return ihipLogStatus(hipErrorSharedObjectInitFailed);
            }

            status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, NULL, &(*module)->executable);
            CHECKLOG_HSA(status, hipErrorNotInitialized);

            status = hsa_executable_load_code_object((*module)->executable, agent, (*module)->object, NULL);
            CHECKLOG_HSA(status, hipErrorNotInitialized);

            status = hsa_executable_freeze((*module)->executable, NULL);
            CHECKLOG_HSA(status, hipErrorNotInitialized);
        }
    }

    return ihipLogStatus(ret);
}


hipError_t hipModuleUnload(hipModule_t hmod)
{
    // TODO - improve this synchronization so it is thread-safe.
    // Currently we want for all inflight activity to complete, but don't prevent another
    // thread from launching new kernels before we finish this operation.
    ihipSynchronize();
    hipError_t ret = hipSuccess;
    hsa_status_t status = hsa_executable_destroy(hmod->executable);
    if(status != HSA_STATUS_SUCCESS)
		{
				ret = hipErrorInvalidValue;
		}
    status = hsa_code_object_destroy(hmod->object);
    if(status != HSA_STATUS_SUCCESS)
		{
				ret = hipErrorInvalidValue;
		}
    status = hsa_memory_free(hmod->ptr);
    if(status != HSA_STATUS_SUCCESS)
		{
				ret = hipErrorInvalidValue;
		}
    delete hmod;
    return ihipLogStatus(ret);
}

hipError_t ihipModuleGetSymbol(hipFunction_t *func, hipModule_t hmod, const char *name){
    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t ret = hipSuccess;

    if(name == nullptr){
        return ihipLogStatus(hipErrorInvalidValue);
    }

    if(ctx == nullptr){
        ret = hipErrorInvalidContext;

    }else{
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t *currentDevice = ihipGetDevice(deviceId);
        hsa_agent_t gpuAgent = (hsa_agent_t)currentDevice->_hsaAgent;

        hsa_status_t status;
        hsa_executable_symbol_t 	symbol;
        status = hsa_executable_get_symbol(hmod->executable, NULL, name, gpuAgent, 0, &symbol);
        if(status != HSA_STATUS_SUCCESS){
            return ihipLogStatus(hipErrorNotFound);
        }

        status = hsa_executable_symbol_get_info(symbol,
                                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                   &func->_object);
        CHECK_HSA(status, hipErrorNotFound);

        status = hsa_executable_symbol_get_info(symbol,
                                    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                    &func->_groupSegmentSize);
        CHECK_HSA(status, hipErrorNotFound);

        status = hsa_executable_symbol_get_info(symbol,
                                    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                    &func->_privateSegmentSize);
        CHECK_HSA(status, hipErrorNotFound);

        strncpy(func->_name, name, sizeof(func->_name));
    }
    return ihipLogStatus(ret);
}


hipError_t hipModuleGetFunction(hipFunction_t *hfunc, hipModule_t hmod,
                                const char *name){
    HIP_INIT_API(hfunc, hmod, name);
    return ihipModuleGetSymbol(hfunc, hmod, name);
}


hipError_t hipModuleLaunchKernel(hipFunction_t f,
            uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
            uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
            uint32_t sharedMemBytes, hipStream_t hStream,
            void **kernelParams, void **extra)
{
    HIP_INIT_API(f, gridDimX, gridDimY, gridDimZ,
                 blockDimX, blockDimY, blockDimZ,
                 sharedMemBytes, hStream,
                 kernelParams, extra);

    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t ret = hipSuccess;

    if(ctx == nullptr){
        ret = hipErrorInvalidDevice;

    }else{
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t *currentDevice = ihipGetDevice(deviceId);
        hsa_agent_t gpuAgent = (hsa_agent_t)currentDevice->_hsaAgent;

        void *config[5] = {0};
        size_t kernArgSize;

        if(extra != NULL){
            memcpy(config, extra, sizeof(size_t)*5);
            if(config[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER && config[2] == HIP_LAUNCH_PARAM_BUFFER_SIZE && config[4] == HIP_LAUNCH_PARAM_END){
                kernArgSize = *(size_t*)(config[3]);
            } else {
                return ihipLogStatus(hipErrorNotInitialized);
            }
        }else{
            return ihipLogStatus(hipErrorInvalidValue);
        }



        /*
          Kernel argument preparation.
        */
        grid_launch_parm lp; 
        lp.dynamic_group_mem_bytes = f._groupSegmentSize;  // TODO - this should be part of preLaunchKernel.
        hStream = ihipPreLaunchKernel(hStream, dim3(gridDimX, gridDimY, gridDimZ), dim3(blockDimX, blockDimY, blockDimZ), &lp, f._name);


        hsa_kernel_dispatch_packet_t aql;

        memset(&aql, 0, sizeof(aql));

        //aql.completion_signal._handle = 0;
        //aql.kernarg_address = 0;

        aql.workgroup_size_x = blockDimX;
        aql.workgroup_size_y = blockDimY;
        aql.workgroup_size_z = blockDimZ;
        aql.grid_size_x = blockDimX * gridDimX;
        aql.grid_size_y = blockDimY * gridDimY;
        aql.grid_size_z = blockDimZ * gridDimZ;
        aql.group_segment_size = f._groupSegmentSize + sharedMemBytes;
        aql.private_segment_size = f._privateSegmentSize;
        aql.kernel_object = f._object;
        aql.setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
        aql.header =   (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                                    (1 << HSA_PACKET_HEADER_BARRIER) |
                                    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                                    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

        lp.av->dispatch_hsa_kernel(&aql, config[1] /* kernarg*/, kernArgSize);


        ihipPostLaunchKernel(f._name, hStream, lp);
    }

    return ihipLogStatus(ret);
}


hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes,
                              hipModule_t hmod, const char* name)
{
    HIP_INIT_API(dptr, bytes, hmod, name);
    hipError_t ret = hipSuccess;
    if(dptr == NULL || bytes == NULL){
        return ihipLogStatus(hipErrorInvalidValue);
    }
    if(name == NULL || hmod == NULL){
        return ihipLogStatus(hipErrorNotInitialized);
    }
    else{
        hipFunction_t func;
        ihipModuleGetSymbol(&func, hmod, name);
        *bytes = PrintSymbolSizes(hmod->ptr, name) + sizeof(amd_kernel_code_t);
        *dptr = reinterpret_cast<void*>(func._object);
        return ihipLogStatus(ret);
    }
}


hipError_t hipModuleLoadData(hipModule_t *module, const void *image)
{
    HIP_INIT_API(module, image);
    hipError_t ret = hipSuccess;
    if(image == NULL || module == NULL){
        return ihipLogStatus(hipErrorNotInitialized);
    } else {
        auto ctx = ihipGetTlsDefaultCtx();
        *module = new ihipModule_t;
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t *currentDevice = ihipGetDevice(deviceId);

        void *p;
        uint64_t size = ElfSize(image);
        hsa_agent_t agent = currentDevice->_hsaAgent;
        hsa_region_t sysRegion;
        hsa_status_t status = hsa_agent_iterate_regions(agent, hipdrv::findSystemRegions, &sysRegion);
        status = hsa_memory_allocate(sysRegion, size, (void**)&p);

        if(status != HSA_STATUS_SUCCESS){
            return ihipLogStatus(hipErrorOutOfMemory);
        }

        char *ptr = (char*)p;
        if(!ptr){
           return ihipLogStatus(hipErrorOutOfMemory);
        }
        (*module)->ptr = p;
        (*module)->size = size;

        memcpy(ptr, image, size);

        status = hsa_code_object_deserialize(ptr, size, NULL, &(*module)->object);

        if(status != HSA_STATUS_SUCCESS){
            return ihipLogStatus(hipErrorSharedObjectInitFailed);
        }

        status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, NULL, &(*module)->executable);
        CHECKLOG_HSA(status, hipErrorNotInitialized);

        status = hsa_executable_load_code_object((*module)->executable, agent, (*module)->object, NULL);
        CHECKLOG_HSA(status, hipErrorNotInitialized);

        status = hsa_executable_freeze((*module)->executable, NULL);
        CHECKLOG_HSA(status, hipErrorNotInitialized);
    }
    return ihipLogStatus(ret);
}
