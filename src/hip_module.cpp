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

}   // End namespace hipdrv



hipError_t hipModuleLoad(hipModule *module, const char *fname){
    HIP_INIT_API(fname);
    hipError_t ret = hipSuccess;
    *module = new ihipModule_t;

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

            *module = new ihipModule_t;
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

            in.seekg(0, std::ios::beg);
            std::copy(std::istreambuf_iterator<char>(in),
                      std::istreambuf_iterator<char>(), ptr);
            status = hsa_code_object_deserialize(ptr, size, NULL, &(*module)->object);

            if(status != HSA_STATUS_SUCCESS){
                return hipErrorSharedObjectInitFailed;
            }

            status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, NULL, &(*module)->executable);
            if(status != HSA_STATUS_SUCCESS){
                return hipErrorNotInitialized;
            }
        }
    }

    return ret;
}

hipError_t hipModuleUnload(hipModule hmod){
    hipError_t ret = hipSuccess;
    hsa_status_t status = hsa_executable_destroy(hmod->executable);
    if(status != HSA_STATUS_SUCCESS){ret = hipErrorInvalidValue; }
    status = hsa_code_object_destroy(hmod->object);
    if(status != HSA_STATUS_SUCCESS){ret = hipErrorInvalidValue; }
    delete hmod;
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
        *func = new ihipFunction_t;
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t *currentDevice = ihipGetDevice(deviceId);
        hsa_agent_t gpuAgent = (hsa_agent_t)currentDevice->_hsaAgent;

        hsa_status_t status;
        if(status != HSA_STATUS_SUCCESS){
            return hipErrorNotInitialized;
        }

        status = hsa_executable_load_code_object(hmod->executable, gpuAgent, hmod->object, NULL);
        if(status != HSA_STATUS_SUCCESS){
            return hipErrorNotInitialized;
        }

        status = hsa_executable_freeze(hmod->executable, NULL);
        status = hsa_executable_get_symbol(hmod->executable, NULL, name, gpuAgent, 0, &(*func)->kernel_symbol);
        if(status != HSA_STATUS_SUCCESS){
            return hipErrorNotFound;
        }

        status = hsa_executable_symbol_get_info((*func)->kernel_symbol,
                                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                   &(*func)->kernel);

        if(status != HSA_STATUS_SUCCESS){
            return hipErrorNotFound;
        }


    }

    return ret;
}

hipError_t hipLaunchModuleKernel(hipFunction f,
            uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
            uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
            uint32_t sharedMemBytes, hipStream_t hStream,
            void **kernelParams, void **extra){
    HIP_INIT_API(f->kernel);
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
        hsa_status_t status;
        grid_launch_parm lp;
        hStream = ihipPreLaunchKernel(hStream, 0, 0, &lp);

/*
  Create signal
*/

        hsa_signal_t signal;
        status = hsa_signal_create(1, 0, NULL, &signal);

/*
  Launch AQL packet
*/
        hStream->launchModuleKernel(signal, blockDimX, blockDimY, blockDimZ,
                  gridDimX, gridDimY, gridDimZ, sharedMemBytes, config[1], kernSize, f->kernel);

/*
  Wait for signal
*/

        hsa_signal_value_t value = hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);


        ihipPostLaunchKernel(hStream, lp);
                
    }

    return ret;
}
