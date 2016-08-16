#include "hip_runtime.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hcc_detail/hip_hcc.h"
#include "hcc_detail/trace_helper.h"
#include <fstream>

hipError_t hipModuleLoad(hipModule *module, const char *fname){
    HIP_INIT_API(fname);
    hipError_t ret = hipSuccess;
    auto ctx = ihipGetTlsDefaultCtx();
    if(ctx == nullptr){
        ret = hipErrorInvalidDevice;
    }else{
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t *currentDevice = ihipGetDevice(deviceId);
        hc::accelerator acc = currentDevice->_acc;
        std::ifstream in(fname, std::ios::binary | std::ios::ate);
        if(!in){
            std::cout<<"Couldn't read file "<<fname<<std::endl;
        }else{
            size_t size = std::string::size_type(in.tellg());
            void *p = NULL;
            hsa_amd_memory_pool_t *pool = (hsa_amd_memory_pool_t*)acc.get_hsa_am_system_region();
            hsa_status_t status = hsa_amd_memory_pool_allocate(*pool, size, 0, (void**)&p);
            assert(status = HSA_STATUS_SUCCESS);
            char *ptr = (char*)p;
            if(!ptr){
                std::cout<<"Error: failed to allocate memory for code object"<<std::endl;
            }
            in.seekg(0, std::ios::beg);
            std::copy(std::istreambuf_iterator<char>(in),
                      std::istreambuf_iterator<char>(), ptr);
            hsa_code_object_t obj;
            status = hsa_code_object_deserialize(ptr, size, NULL, &obj);
            *module = obj.handle;
            assert(status == HSA_STATUS_SUCCESS);
        }
    }
    return ret;
}

hipError_t hipModuleGetFunction(hipFunction *func, hipModule hmod, const char *name){
    HIP_INIT_API(name);
    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t ret = hipSuccess;
    if(ctx == nullptr){
      ret = hipErrorInvalidDevice;
    }else{
      int deviceId = ctx->getDevice()->_deviceId;
      ihipDevice_t *currentDevice = ihipGetDevice(deviceId);
      hc::accelerator acc = currentDevice->_acc;
      hsa_agent_t *gpuAgent = (hsa_agent_t*)acc.get_hsa_agent();

      assert(gpuAgent != NULL);
      hsa_status_t status;
      hsa_executable_symbol_t kernel_symbol;
      hsa_executable_t executable;
      status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, NULL, &executable);
      assert(status == HSA_STATUS_SUCCESS);
      hsa_code_object_t obj;
      obj.handle = hmod;
      status = hsa_executable_load_code_object(executable, *gpuAgent, obj, NULL);
      assert(status == HSA_STATUS_SUCCESS);
      status = hsa_executable_freeze(executable, NULL);
      assert(status == HSA_STATUS_SUCCESS);
      status = hsa_executable_get_symbol(executable, NULL, name, *gpuAgent, 0, &kernel_symbol);
      assert(status == HSA_STATUS_SUCCESS);
      status = hsa_executable_symbol_get_info(kernel_symbol,
                                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                   func);
      assert(status == HSA_STATUS_SUCCESS);
    }
    return ret;
}
