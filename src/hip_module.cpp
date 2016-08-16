#include "hip_runtime.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hcc_detail/hip_hcc.h"
#include "hcc_detail/trace_helper.h"
#include <fstream>

hsa_status_t FindRegions(hsa_region_t region, void *data){
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

hipError_t hipModuleLoad(hipModule *module, const char *fname){
    HIP_INIT_API(fname);
    hipError_t ret = hipSuccess;
    auto ctx = ihipGetTlsDefaultCtx();
    if(ctx == nullptr){
        ret = hipErrorInvalidDevice;
    }else{
        int deviceId = ctx->getDevice()->_deviceId;
        ihipDevice_t *currentDevice = ihipGetDevice(deviceId);
        std::ifstream in(fname, std::ios::binary | std::ios::ate);
        if(!in){
            std::cout<<"Couldn't read file "<<fname<<std::endl;
        }else{
            size_t size = std::string::size_type(in.tellg());
            void *p = NULL;
            hsa_agent_t agent = currentDevice->_hsaAgent;
            hsa_region_t sysRegion;
            hsa_status_t status = hsa_agent_iterate_regions(agent, FindRegions, &sysRegion);
            assert(status == HSA_STATUS_SUCCESS);
            status = hsa_memory_allocate(sysRegion, size, (void**)&p);
            char *ptr = (char*)p;
            if(!ptr){
                std::cout<<"Error: failed to allocate memory for code object"<<std::endl;
            }
            hsa_code_object_t obj;
            in.seekg(0, std::ios::beg);
            std::copy(std::istreambuf_iterator<char>(in),
                      std::istreambuf_iterator<char>(), ptr);
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
      hsa_agent_t gpuAgent = (hsa_agent_t)currentDevice->_hsaAgent;

      hsa_status_t status;
      hsa_executable_symbol_t kernel_symbol;
      hsa_executable_t executable;
      status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, NULL, &executable);
      
      assert(status == HSA_STATUS_SUCCESS);
      hsa_code_object_t obj;
      obj.handle = hmod;
      status = hsa_executable_load_code_object(executable, gpuAgent, obj, NULL);
      assert(status == HSA_STATUS_SUCCESS);
      status = hsa_executable_freeze(executable, NULL);
      assert(status == HSA_STATUS_SUCCESS);
      status = hsa_executable_get_symbol(executable, NULL, name, gpuAgent, 0, &kernel_symbol);
      assert(status == HSA_STATUS_SUCCESS);
      status = hsa_executable_symbol_get_info(kernel_symbol,
                                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                   func);
      assert(status == HSA_STATUS_SUCCESS);
    }
    return ret;
}
