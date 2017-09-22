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

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <libelf.h>
#include <elf.h>
#include <gelf.h>
#include <map>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/amd_hsa_kernel_code.h"

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"

//TODO Use Pool APIs from HCC to get memory regions.

#include <cassert>
inline uint64_t alignTo(uint64_t Value, uint64_t Align, uint64_t Skew = 0) {
  assert(Align != 0u && "Align can't be 0.");
  Skew %= Align;
  return (Value + Align - 1 - Skew) / Align * Align + Skew;
}

struct ihipKernArgInfo{
  std::vector<uint32_t> Size;
  std::vector<uint32_t> Align;
  std::vector<std::string> ArgType;
  std::vector<std::string> ArgName;
  uint32_t totalSize;
};

std::map<std::string,struct ihipKernArgInfo> kernelArguments;

struct MyElfNote {
  uint32_t n_namesz = 0;
  uint32_t n_descsz = 0;
  uint32_t n_type = 0;

  MyElfNote() = default;
};

struct ihipModuleSymbol_t{
    uint64_t    _object;             // The kernel object.
    uint32_t    _groupSegmentSize;
    uint32_t    _privateSegmentSize;
    std::string        _name;       // TODO - review for performance cost.  Name is just used for debug.
};

template <>
std::string ToString(hipFunction_t v)
{
    std::ostringstream ss;
    ss << "0x" << std::hex << v->_object;
    return ss.str();
};


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
            std::copy(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), ptr);
/*
            Elf *e = elf_memory((char*)p, size);
            if(elf_kind(e) != ELF_K_ELF){
              return ihipLogStatus(hipErrorInvalidValue);
            }
            size_t numpHdrs;
            if(elf_getphdrnum(e, &numpHdrs) != 0){
              return ihipLogStatus(hipErrorInvalidValue);
            }
            for(size_t i=0;i<numpHdrs;i++){
              GElf_Phdr pHdr;
              if(gelf_getphdr(e, i, &pHdr) != &pHdr){
                continue;
              }
              if(pHdr.p_type == PT_NOTE && pHdr.p_align >= sizeof(int)){
                char* ptr = (char*) p + pHdr.p_offset;
                char* segmentEnd = ptr + pHdr.p_filesz;
                while(ptr < segmentEnd){
                  MyElfNote *note = (MyElfNote*) ptr;
                  char *name = (char*) &note[1];
                  char *desc = name + alignTo(note->n_namesz, sizeof(int));
                  if (note->n_type == 8) {
                    std::string metadatastr((const char*)desc, (size_t)note->n_descsz);
                    AMDGPU::RuntimeMD::Program::Metadata meta;
                    meta = meta.fromYAML(metadatastr);
                    for(int i=0;i<meta.Kernels.size();i++){
                      struct ihipKernArgInfo pl;
                      size_t totalSizeOfArgs = 0;
                      for(int j=0;j<meta.Kernels[i].Args.size();j++){
                        if(meta.Kernels[i].Args[j].Kind < 7) {
                          pl.Size.push_back(meta.Kernels[i].Args[j].Size);
                          pl.Align.push_back(meta.Kernels[i].Args[j].Align);
                          pl.ArgType.push_back(meta.Kernels[i].Args[j].TypeName);
                          pl.ArgName.push_back(meta.Kernels[i].Args[j].Name);
                          totalSizeOfArgs += meta.Kernels[i].Args[j].Align;
                        }
                      }
                      pl.totalSize = totalSizeOfArgs;
                      kernelArguments[meta.Kernels[i].Name] = pl;
                    }
                    break;
                  }

                  ptr += sizeof(*note)
                      + alignTo(note->n_namesz, sizeof(int))
                      + alignTo(note->n_descsz, sizeof(int));
                }
              }
            }
*/
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
    for(auto f = hmod->funcTrack.begin(); f != hmod->funcTrack.end(); ++f) {
      delete *f;
    }
    delete hmod;
    return ihipLogStatus(ret);
}


hipError_t ihipModuleGetSymbol(hipFunction_t *func, hipModule_t hmod, const char *name)
{
    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t ret = hipSuccess;

    if (name == nullptr){
        return ihipLogStatus(hipErrorInvalidValue);
    }

    if (ctx == nullptr){
        ret = hipErrorInvalidContext;

    } else {
        std::string str(name);
        for(auto f = hmod->funcTrack.begin(); f != hmod->funcTrack.end(); ++f) {
            if((*f)->_name == str) {
                *func = *f;
                return ret;
            }
        }
        ihipModuleSymbol_t *sym = new ihipModuleSymbol_t;
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
                                   &sym->_object);
        CHECK_HSA(status, hipErrorNotFound);

        status = hsa_executable_symbol_get_info(symbol,
                                    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                    &sym->_groupSegmentSize);
        CHECK_HSA(status, hipErrorNotFound);

        status = hsa_executable_symbol_get_info(symbol,
                                    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                    &sym->_privateSegmentSize);
        CHECK_HSA(status, hipErrorNotFound);

        sym->_name = name;
        *func = sym;
        hmod->funcTrack.push_back(*func);
    }
    return ret;
}


hipError_t hipModuleGetFunction(hipFunction_t *hfunc, hipModule_t hmod,
                                const char *name){
    HIP_INIT_API(hfunc, hmod, name);
    return ihipLogStatus(ihipModuleGetSymbol(hfunc, hmod, name));
}


hipError_t ihipModuleLaunchKernel(hipFunction_t f,
                                  uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                  uint32_t localWorkSizeX, uint32_t localWorkSizeY, uint32_t localWorkSizeZ,
                                  size_t sharedMemBytes, hipStream_t hStream,
                                  void **kernelParams, void **extra,
                                  hipEvent_t startEvent, hipEvent_t stopEvent)
{

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

        if(kernelParams != NULL){
          std::string name = f->_name;
          struct ihipKernArgInfo pl = kernelArguments[name];
          char* argBuf = (char*)malloc(pl.totalSize);
          memset(argBuf, 0, pl.totalSize);
          int index = 0;
          for(int i=0;i<pl.Size.size();i++){
            memcpy(argBuf + index, kernelParams[i], pl.Size[i]);
            index += pl.Align[i];
          }
          config[1] = (void*)argBuf;
          kernArgSize = pl.totalSize;
        } else if(extra != NULL){
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
        lp.dynamic_group_mem_bytes = sharedMemBytes;  // TODO - this should be part of preLaunchKernel.
        hStream = ihipPreLaunchKernel(hStream, dim3(globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ), dim3(localWorkSizeX, localWorkSizeY, localWorkSizeZ), &lp, f->_name.c_str());


        hsa_kernel_dispatch_packet_t aql;

        memset(&aql, 0, sizeof(aql));

        //aql.completion_signal._handle = 0;
        //aql.kernarg_address = 0;

        aql.workgroup_size_x = localWorkSizeX;
        aql.workgroup_size_y = localWorkSizeY;
        aql.workgroup_size_z = localWorkSizeZ;
        aql.grid_size_x = globalWorkSizeX;
        aql.grid_size_y = globalWorkSizeY;
        aql.grid_size_z = globalWorkSizeZ;
        aql.group_segment_size = f->_groupSegmentSize + sharedMemBytes;
        aql.private_segment_size = f->_privateSegmentSize;
        aql.kernel_object = f->_object;
        aql.setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
        aql.header =   (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
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
                                  , f->_name.c_str()
#endif
                                  );



        if (startEvent) {
            startEvent->attachToCompletionFuture(&cf, hStream, hipEventTypeStartCommand);
        }
        if (stopEvent) {
            stopEvent->attachToCompletionFuture (&cf, hStream, hipEventTypeStopCommand);
        }


        if(kernelParams != NULL){
          free(config[1]);
        }
        ihipPostLaunchKernel(f->_name.c_str(), hStream, lp);
    }

    return ret;
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
    return ihipLogStatus(ihipModuleLaunchKernel(f,
                blockDimX * gridDimX, blockDimY * gridDimY, gridDimZ * blockDimZ,
                blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream, kernelParams, extra,
                nullptr, nullptr));
}


hipError_t hipHccModuleLaunchKernel(hipFunction_t f,
            uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
            uint32_t localWorkSizeX, uint32_t localWorkSizeY, uint32_t localWorkSizeZ,
            size_t sharedMemBytes, hipStream_t hStream,
            void **kernelParams, void **extra,
            hipEvent_t startEvent, hipEvent_t stopEvent)
{
    HIP_INIT_API(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
                 localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                 sharedMemBytes, hStream,
                 kernelParams, extra);
    return ihipLogStatus(ihipModuleLaunchKernel(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
                localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent));
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
        *dptr = reinterpret_cast<void*>(func->_object);
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

hipError_t hipModuleLoadDataEx(hipModule_t *module, const void *image, unsigned int numOptions, hipJitOption *options, void **optionValues)
{
    return hipModuleLoadData(module, image);
}
