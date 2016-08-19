/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip_runtime.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/amd_hsa_kernel_code.h"
#include "hcc_detail/hip_hcc.h"
#include "hcc_detail/trace_helper.h"
#include <fstream>

//TODO Use Pool APIs from HCC to get memory regions.

namespace hipdrv{

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

    hsa_status_t findKernArgRegions(hsa_region_t region, void *data){
        hsa_region_segment_t segment_id;
        hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment_id);

        if(segment_id != HSA_REGION_SEGMENT_GLOBAL){
            return HSA_STATUS_SUCCESS;
        }

        hsa_region_global_flag_t flags;
        hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);

        hsa_region_t *reg = (hsa_region_t*)data;

        if(flags & HSA_REGION_GLOBAL_FLAG_KERNARG){
            *reg = region;
        }

        return HSA_STATUS_SUCCESS;
    }

}   // End namespace hipdrv



hipError_t hipModuleLoad(hipModule *module, const char *fname){
    HIP_INIT_API(fname);
    hipError_t ret = hipSuccess;

    if(module == NULL){
        return hipErrorInvalidValue;
    }

    auto ctx = ihipGetTlsDefaultCtx();
    if(ctx == nullptr){
        ret = hipErrorInvalidContext;

    }else{
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t *currentDevice = ihipGetDevice(deviceId);
        std::ifstream in(fname, std::ios::binary | std::ios::ate);

        if(!in){
            return hipErrorFileNotFound;

        }else{
            size_t size = std::string::size_type(in.tellg());
            void *p = NULL;
            hsa_agent_t agent = currentDevice->_hsaAgent;
            hsa_region_t sysRegion;
            hsa_status_t status = hsa_agent_iterate_regions(agent, hipdrv::findSystemRegions, &sysRegion);
            status = hsa_memory_allocate(sysRegion, size, (void**)&p);
            if(status != HSA_STATUS_SUCCESS){
                return hipErrorOutOfMemory;
            }

            char *ptr = (char*)p;
            if(!ptr){
                return hipErrorOutOfMemory;
            }

            hsa_code_object_t obj;
            in.seekg(0, std::ios::beg);
            std::copy(std::istreambuf_iterator<char>(in),
                      std::istreambuf_iterator<char>(), ptr);
            status = hsa_code_object_deserialize(ptr, size, NULL, &obj);

            if(status != HSA_STATUS_SUCCESS){
                return hipErrorSharedObjectInitFailed;
            }

            module->object = obj.handle;

            hsa_executable_t executable;
            status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, NULL, &executable);
            if(status != HSA_STATUS_SUCCESS){
                return hipErrorNotInitialized;
            }

            module->executable = executable.handle;

        }
    }

    return ret;
}

hipError_t hipModuleGetFunction(hipFunction *func, hipModule hmod, const char *name){
    HIP_INIT_API(name);
    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t ret = hipSuccess;

    if(name == nullptr){
        return hipErrorInvalidValue;
    }

    if(ctx == nullptr){
        ret = hipErrorInvalidContext;

    }else{
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t *currentDevice = ihipGetDevice(deviceId);
        hsa_agent_t gpuAgent = (hsa_agent_t)currentDevice->_hsaAgent;

        hsa_status_t status;
        hsa_executable_t executable;
        hsa_executable_symbol_t kernel_symbol;
        executable.handle = hmod.executable;
        if(status != HSA_STATUS_SUCCESS){
            return hipErrorNotInitialized;
        }

        hsa_code_object_t obj;
        obj.handle = hmod.object;
        status = hsa_executable_load_code_object(executable, gpuAgent, obj, NULL);
        if(status != HSA_STATUS_SUCCESS){
            return hipErrorNotInitialized;
        }

        status = hsa_executable_freeze(executable, NULL);
        status = hsa_executable_get_symbol(executable, NULL, name, gpuAgent, 0, &kernel_symbol);
        if(status != HSA_STATUS_SUCCESS){
            return hipErrorNotFound;
        }

        status = hsa_executable_symbol_get_info(kernel_symbol,
                                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                   &func->kernel);

        if(status != HSA_STATUS_SUCCESS){
            return hipErrorNotFound;
        }

        func->symbol = kernel_symbol.handle;

    }

    return ret;
}

hipError_t hipLaunchModuleKernel(hipFunction f,
            uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
            uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
            uint32_t sharedMemBytes, hipStream_t hStream,
            void **kernelParams, void **extra){
    HIP_INIT_API(f.kernel);
    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t ret = hipSuccess;

    if(ctx == nullptr){
        ret = hipErrorInvalidDevice;

    }else{
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t *currentDevice = ihipGetDevice(deviceId);
        hsa_agent_t gpuAgent = (hsa_agent_t)currentDevice->_hsaAgent;

        void *config[5] = {0};
        size_t kernSize;

        if(extra != NULL){
            memcpy(config, extra, sizeof(size_t)*5);
            if(config[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER && config[2] == HIP_LAUNCH_PARAM_BUFFER_SIZE && config[4] == HIP_LAUNCH_PARAM_END){
                kernSize = *(size_t*)(config[3]);
            }else{
                return hipErrorNotInitialized;
            }
        }else{
            return hipErrorInvalidValue;
        }
/*
Kernel argument preparation.
*/

        hsa_region_t kernArg;
        hsa_status_t status = hsa_agent_iterate_regions(gpuAgent, hipdrv::findKernArgRegions, &kernArg);
        void *kern;
        status = hsa_memory_allocate(kernArg, kernSize, &kern);

        if(status != HSA_STATUS_SUCCESS){
            return hipErrorLaunchOutOfResources;
        }

        memcpy(kern, config[1], kernSize);



/*
Pre kernel launch 

    stream = ihipSyncAndResolveStream(stream);
    stream->lockopen_preKernelCommand();
    hc::accelerator_view av = &stream->_av;
    hc::completion_future cf = new hc::completion_future;
*/

        hStream = ihipSyncAndResolveStream(hStream);
        hc::accelerator_view *av = &hStream->_av;
        hsa_queue_t *Queue = (hsa_queue_t*)av->get_hsa_queue();
        hsa_signal_t signal;
        status = hsa_signal_create(1, 0, NULL, &signal);

//        hsa_memory_pool_t pool = av->get_hsa_kernarg_region();
//        status = hsa_amd_memory_pool_allocate(pool, kernSize, 0, &kernArg);
//        assert(status == HSA_STATUS_SUCCESS);

/*
Creating the packets
*/

        const uint32_t queue_mask = Queue->size-1;
        uint32_t packet_index = hsa_queue_load_write_index_relaxed(Queue);
        hsa_kernel_dispatch_packet_t *dispatch_packet = &(((hsa_kernel_dispatch_packet_t*)(Queue->base_address))[packet_index & queue_mask]);

        dispatch_packet->completion_signal = signal;
        dispatch_packet->workgroup_size_x = blockDimX;
        dispatch_packet->workgroup_size_y = blockDimY;
        dispatch_packet->workgroup_size_z = blockDimZ;
        dispatch_packet->grid_size_x = blockDimX * gridDimX;
        dispatch_packet->grid_size_y = blockDimY * gridDimY;
        dispatch_packet->grid_size_z = blockDimZ * gridDimZ;
        dispatch_packet->group_segment_size = 0;
        dispatch_packet->private_segment_size = sharedMemBytes;
        dispatch_packet->kernarg_address = kern;
        dispatch_packet->kernel_object = f.kernel;
        uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

        uint16_t setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
        uint32_t header32 = header | (setup << 16);

        __atomic_store_n((uint32_t*)(dispatch_packet), header32, __ATOMIC_RELEASE);

        hsa_queue_store_write_index_relaxed(Queue, packet_index+1);
        hsa_signal_store_relaxed(Queue->doorbell_signal, packet_index);
        hsa_signal_value_t value = hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    }

    return ret;
}
