#include <hc_am.hpp>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"
#include "hip/hip_cbapi.h"

///////////////////////////////////////////////////////////////////////////////////
// HIP API callbacks/activity

extern "C" size_t HSAOp_get_record_size();
extern "C" void HSAOp_set_activity_callback(hip_cb_async_t callback, void* arg);

HIP_CALLBACKS_INSTANCE;

extern std::string& FunctionSymbol(hipFunction_t f);
const char* hipKernelNameRef(const hipFunction_t f) { return FunctionSymbol(f).c_str(); }

hipError_t hipValidateActivityRecord() {
  return (sizeof(hip_ops_record_t) == HSAOp_get_record_size()) ? hipSuccess : hipErrorInvalidValue;
}

hipError_t hipRegisterApiCallback(uint32_t id, hip_cb_fun_t fun, void* arg) {
  return HIP_SET_CALLBACK(id, fun, arg) ? hipSuccess : hipErrorInvalidValue;
}

hipError_t hipRemoveApiCallback(uint32_t id) {
  return HIP_SET_CALLBACK(id, NULL, NULL) ? hipSuccess : hipErrorInvalidValue;
}

hipError_t hipRegisterActivityCallback(uint32_t id, hip_cb_act_t cb_act, hip_cb_async_t cb_async, void* arg) {
  HSAOp_set_activity_callback(cb_async, arg);
  return HIP_SET_ACTIVITY(id, cb_act, arg) ? hipSuccess : hipErrorInvalidValue;
}

hipError_t hipRemoveActivityCallback(uint32_t id) {
  HSAOp_set_activity_callback(NULL, NULL);
  return HIP_SET_ACTIVITY(id, NULL, NULL) ? hipSuccess : hipErrorInvalidValue;
}

///////////////////////////////////////////////////////////////////////////////////
