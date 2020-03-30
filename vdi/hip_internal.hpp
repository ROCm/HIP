/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef HIP_SRC_HIP_INTERNAL_H
#define HIP_SRC_HIP_INTERNAL_H

#include "vdi_common.hpp"
#include "hip_prof_api.h"
#include "trace_helper.h"
#include "utils/debug.hpp"
#include "hip_formatting.hpp"
#include <unordered_set>
#include <thread>
#include <stack>
#include <mutex>
#include <iterator>


/*! IHIP IPC MEMORY Structure */
#define IHIP_IPC_MEM_HANDLE_SIZE   32
#define IHIP_IPC_MEM_RESERVED_SIZE LP64_SWITCH(28,24)

typedef struct ihipIpcMemHandle_st {
  char ipc_handle[IHIP_IPC_MEM_HANDLE_SIZE];  ///< ipc memory handle on ROCr
  size_t psize;
  char reserved[IHIP_IPC_MEM_RESERVED_SIZE];
} ihipIpcMemHandle_t;

#define HIP_INIT() \
  std::call_once(hip::g_ihipInitialized, hip::init);       \
  if (hip::g_device == nullptr && g_devices.size() > 0) {  \
    hip::g_device = g_devices[0];                          \
  }

// This macro should be called at the beginning of every HIP API.
#define HIP_INIT_API(cid, ...)                               \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[%zx] %s ( %s )", std::this_thread::get_id(), __func__, ToString( __VA_ARGS__ ).c_str()); \
  amd::Thread* thread = amd::Thread::current();              \
  if (!VDI_CHECK_THREAD(thread)) {                           \
    HIP_RETURN(hipErrorOutOfMemory);                         \
  }                                                          \
  HIP_INIT()                                                 \
  HIP_CB_SPAWNER_OBJECT(cid);

#define HIP_RETURN(ret)          \
  hip::g_lastError = ret;  \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[%zx] %s: Returned %s", std::this_thread::get_id(), __func__, hipGetErrorName(hip::g_lastError)); \
  return hip::g_lastError;

namespace hc {
class accelerator;
class accelerator_view;
};

namespace hip {

  /// HIP Device class
  class Device {
    amd::Monitor lock_{"Device lock"};
    /// VDI context
    amd::Context* context_;
    /// Device's ID
    /// Store it here so we don't have to loop through the device list every time
    int deviceId_;
    //Maintain list of user enabled peers
    std::list<int> userEnabledPeers;
  public:
    Device(amd::Context* ctx, int devId): context_(ctx), deviceId_(devId) { assert(ctx != nullptr); }
    ~Device() {}

    amd::Context* asContext() const { return context_; }
    int deviceId() const { return deviceId_; }
    void retain() const { context_->retain(); }
    void release() const { context_->release(); }
    const std::vector<amd::Device*>& devices() const { return context_->devices(); }
    hipError_t EnablePeerAccess(int peerDeviceId){
      amd::ScopedLock lock(lock_);
      bool found = (std::find(userEnabledPeers.begin(), userEnabledPeers.end(), peerDeviceId) != userEnabledPeers.end());
      if (found) {
        return hipErrorPeerAccessAlreadyEnabled;
      }
      userEnabledPeers.push_back(peerDeviceId);
      return hipSuccess;
    }
    hipError_t DisablePeerAccess(int peerDeviceId) {
      amd::ScopedLock lock(lock_);
      bool found = (std::find(userEnabledPeers.begin(), userEnabledPeers.end(), peerDeviceId) != userEnabledPeers.end());
      if (found) {
        userEnabledPeers.remove(peerDeviceId);
        return hipSuccess;
      } else {
        return hipErrorPeerAccessNotEnabled;
      }
    }
  };

  extern std::once_flag g_ihipInitialized;
  /// Current thread's device
  extern thread_local Device* g_device;
  extern thread_local hipError_t g_lastError;
  /// Device representing the host - for pinned memory
  extern Device* host_device;

  extern void init();

  extern Device* getCurrentDevice();
  extern void setCurrentDevice(unsigned int index);

  /// Get VDI queue associated with hipStream
  /// Note: This follows the CUDA spec to sync with default streams
  ///       and Blocking streams
  extern amd::HostQueue* getQueue(hipStream_t s);
  /// Get default stream of the device
  extern amd::HostQueue* getNullStream(Device&);
  /// Get default stream associated with the VDI context
  extern amd::HostQueue* getNullStream(amd::Context&);
  /// Get default stream of the thread
  extern amd::HostQueue* getNullStream();
  /// Sync Blocking streams on the current device
  extern void syncStreams();
  /// Sync blocking streams on the given device
  extern void syncStreams(int devId);


  struct Function {
    amd::Kernel* function_;
    amd::Monitor lock_;

    Function(amd::Kernel* f) : function_(f), lock_("function lock") {}
    hipFunction_t asHipFunction() { return reinterpret_cast<hipFunction_t>(this); }

    static Function* asFunction(hipFunction_t f) { return reinterpret_cast<Function*>(f); }
  };

  struct Stream {
    amd::HostQueue* queue;
    amd::Monitor lock;
    Device* device;
    amd::CommandQueue::Priority priority;
    unsigned int flags;

    Stream(Device* dev, amd::CommandQueue::Priority p, unsigned int f);
    void create();
    amd::HostQueue* asHostQueue();
    void destroy();
    void finish();
  };

};

struct ihipExec_t {
  dim3 gridDim_;
  dim3 blockDim_;
  size_t sharedMem_;
  hipStream_t hStream_;
  std::vector<char> arguments_;
};

class PlatformState {
  amd::Monitor lock_{"Guards global function map"};

  std::unordered_map<const void*, std::vector<std::pair<hipModule_t, bool>>> modules_;
  bool initialized_{false};

  void digestFatBinary(const void* data, std::vector<std::pair<hipModule_t, bool>>& programs);
public:
  void init();
  std::vector<std::pair<hipModule_t, bool>>* addFatBinary(const void*data)
  {
    if (initialized_) {
      digestFatBinary(data, modules_[data]);
    }
    return &modules_[data];
  }
  void removeFatBinary(std::vector<std::pair<hipModule_t, bool>>* module)
  {
    for (auto& mod : modules_) {
      if (&mod.second == module) {
        modules_.erase(&mod);
        return;
      }
    }
  }

  struct RegisteredVar {
  public:
    RegisteredVar(): size_(0), devicePtr_(nullptr), amd_mem_obj_(nullptr) {}
    ~RegisteredVar() {}

    hipDeviceptr_t getdeviceptr() const { return devicePtr_; };
    size_t getvarsize() const { return size_; };

    size_t size_;               // Size of the variable
    hipDeviceptr_t devicePtr_;  //Device Memory Address of the variable.
    amd::Memory* amd_mem_obj_;
  };

  struct DeviceFunction {
    std::string deviceName;
    std::vector< std::pair< hipModule_t, bool > >* modules;
    std::vector<hipFunction_t> functions;
  };
  struct DeviceVar {
    void* shadowVptr;
    std::string hostVar;
    size_t size;
    std::vector< std::pair< hipModule_t, bool > >* modules;
    std::vector<RegisteredVar> rvars;
    bool dyn_undef;
  };
private:
  class Module {
  public:
    Module(hipModule_t hip_module_) : hip_module(hip_module_) {}
    std::unordered_map<std::string, DeviceFunction > functions_;
  private:
    hipModule_t hip_module;
  };
  std::unordered_map<hipModule_t, Module*> module_map_;

  std::unordered_map<const void*, DeviceFunction > functions_;
  std::unordered_multimap<std::string, DeviceVar > vars_;

  static PlatformState* platform_;

  PlatformState() {}
  ~PlatformState() {}
public:
  static PlatformState& instance() {
    return *platform_;
  }

  bool unregisterFunc(hipModule_t hmod);
  std::vector< std::pair<hipModule_t, bool> >* unregisterVar(hipModule_t hmod);


  PlatformState::DeviceVar* findVar(std::string hostVar, int deviceId, hipModule_t hmod);
  void registerVar(const void* hostvar, const DeviceVar& var);
  void registerFunction(const void* hostFunction, const DeviceFunction& func);

  bool registerModFuncs(std::vector<std::string>& func_names, hipModule_t* module);
  bool findModFunc(hipFunction_t* hfunc, hipModule_t hmod, const char* name);
  bool createFunc(hipFunction_t* hfunc, hipModule_t hmod, const char* name);
  hipFunction_t getFunc(const void* hostFunction, int deviceId);
  bool getFuncAttr(const void* hostFunction, hipFuncAttributes* func_attr);
  bool getGlobalVar(const void* hostVar, int deviceId, hipModule_t hmod,
                    hipDeviceptr_t* dev_ptr, size_t* size_ptr);
  bool getTexRef(const char* hostVar, hipModule_t hmod, textureReference** texRef);

  bool getShadowVarInfo(std::string var_name, hipModule_t hmod,
                        void** var_addr, size_t* var_size);
  void setupArgument(const void *arg, size_t size, size_t offset);
  void configureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream);

  void popExec(ihipExec_t& exec);
};

extern std::vector<hip::Device*> g_devices;
extern hipError_t ihipDeviceGetCount(int* count);
extern int ihipGetDevice();
extern hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags);
extern amd::Memory* getMemoryObject(const void* ptr, size_t& offset);
extern bool CL_CALLBACK getSvarInfo(cl_program program, std::string var_name, void** var_addr,
                                    size_t* var_size);

#endif // HIP_SRC_HIP_INTERNAL_H
