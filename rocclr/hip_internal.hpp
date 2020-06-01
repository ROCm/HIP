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
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

/*! IHIP IPC MEMORY Structure */
#define IHIP_IPC_MEM_HANDLE_SIZE   32
#define IHIP_IPC_MEM_RESERVED_SIZE LP64_SWITCH(28,24)

typedef struct ihipIpcMemHandle_st {
  char ipc_handle[IHIP_IPC_MEM_HANDLE_SIZE];  ///< ipc memory handle on ROCr
  size_t psize;
  char reserved[IHIP_IPC_MEM_RESERVED_SIZE];
} ihipIpcMemHandle_t;

#ifdef _WIN32
  inline int getpid() { return _getpid(); }
#endif

#define HIP_INIT() \
  std::call_once(hip::g_ihipInitialized, hip::init);       \
  if (hip::g_device == nullptr && g_devices.size() > 0) {  \
    hip::g_device = g_devices[0];                          \
  }

// This macro should be called at the beginning of every HIP API.
#define HIP_INIT_API(cid, ...)                               \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%-5d: [%zx] %s ( %s )", getpid(), std::this_thread::get_id(), __func__, ToString( __VA_ARGS__ ).c_str()); \
  amd::Thread* thread = amd::Thread::current();              \
  if (!VDI_CHECK_THREAD(thread)) {                           \
    HIP_RETURN(hipErrorOutOfMemory);                         \
  }                                                          \
  HIP_INIT()                                                 \
  HIP_CB_SPAWNER_OBJECT(cid);

#define HIP_RETURN(ret)          \
  hip::g_lastError = ret;  \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%-5d: [%zx] %s: Returned %s", getpid(), std::this_thread::get_id(), __func__, hipGetErrorName(hip::g_lastError)); \
  return hip::g_lastError;

namespace hc {
class accelerator;
class accelerator_view;
};

namespace hip {
  class Device;

  class Stream {
    amd::HostQueue* queue_;
    mutable amd::Monitor lock_;
    Device* device_;
    amd::CommandQueue::Priority priority_;
    unsigned int flags_;
    bool null_;

  public:
    Stream(Device* dev, amd::CommandQueue::Priority p, unsigned int f = 0, bool null_stream = false);

    /// Creates the hip stream object, including AMD host queue
    bool Create();

    /// Get device AMD host queue object. The method can allocate the queue
    amd::HostQueue* asHostQueue(bool skip_alloc = false);

    void Destroy();
    void Finish() const;
    /// Get device ID associated with the current stream;
    int DeviceId() const;
    /// Returns if stream is null stream
    bool Null() const { return null_; }
    /// Returns the lock object for the current stream
    amd::Monitor& Lock() const { return lock_; }
    /// Returns the creation flags for the current stream
    unsigned int Flags() const { return flags_; }
    /// Returns the priority for the current stream
    amd::CommandQueue::Priority Priority() const { return priority_; }

    /// Sync all non-blocking streams
    static void syncNonBlockingStreams();
  };

  /// HIP Device class
  class Device {
    amd::Monitor lock_{"Device lock"};
    /// ROCclr context
    amd::Context* context_;
    /// Device's ID
    /// Store it here so we don't have to loop through the device list every time
    int deviceId_;
    /// ROCclr host queue for default streams
    Stream null_stream_;
    //Maintain list of user enabled peers
    std::list<int> userEnabledPeers;

  public:
    Device(amd::Context* ctx, int devId):
      context_(ctx), deviceId_(devId), null_stream_(this, amd::CommandQueue::Priority::Normal, 0, true)
        { assert(ctx != nullptr); }
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
    amd::HostQueue* NullStream(bool skip_alloc = false);
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

  /// Get ROCclr queue associated with hipStream
  /// Note: This follows the CUDA spec to sync with default streams
  ///       and Blocking streams
  extern amd::HostQueue* getQueue(hipStream_t s);
  /// Get default stream associated with the ROCclr context
  extern amd::HostQueue* getNullStream(amd::Context&);
  /// Get default stream of the thread
  extern amd::HostQueue* getNullStream();

  struct Function {
    amd::Kernel* function_;
    amd::Monitor lock_;

    Function(amd::Kernel* f) : function_(f), lock_("function lock") {}
    ~Function() { function_->release(); }
    hipFunction_t asHipFunction() { return reinterpret_cast<hipFunction_t>(this); }

    static Function* asFunction(hipFunction_t f) { return reinterpret_cast<Function*>(f); }
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
  amd::Monitor lock_{"Guards global function map", true};

  std::unordered_map<const void*, std::vector<std::pair<hipModule_t, bool>>> modules_;
  bool initialized_{false};

  void digestFatBinary(const void* data, std::vector<std::pair<hipModule_t, bool>>& programs);
public:
  void init();
  std::vector<std::pair<hipModule_t, bool>>* addFatBinary(const void*data)
  {
    amd::ScopedLock lock(lock_);
    if (initialized_) {
      digestFatBinary(data, modules_[data]);
    }
    return &modules_[data];
  }
  void removeFatBinary(std::vector<std::pair<hipModule_t, bool>>* module)
  {
    amd::ScopedLock lock(lock_);
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
  enum DeviceVarKind {
    DVK_Variable,
    DVK_Surface,
    DVK_Texture
  };
  struct DeviceVar {
    DeviceVarKind kind;
    void* shadowVptr;
    std::string hostVar;
    size_t size;
    std::vector< std::pair< hipModule_t, bool > >* modules;
    std::vector<RegisteredVar> rvars;
    bool dyn_undef;
    int type; // surface/texture type
    int norm; // texture has normalized output
    bool shadowAllocated = false; // shadow ptr is allocated on-demand and needs freeing.
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
  // Map from the host shadow symbol to its device name. As different modules
  // may have the same name, each symbol is uniquely identified by a pair of
  // module handle and its name.
  std::unordered_map<const void*,
                     std::pair<hipModule_t, std::string>> symbols_;

  static PlatformState* platform_;

  PlatformState() {}
  ~PlatformState() {}
public:
  static PlatformState& instance() {
    if (platform_ == nullptr) {
       // __hipRegisterFatBinary() will call this when app starts, thus
       // there is no multiple entry issue here.
       platform_ =  new PlatformState();
    }
    return *platform_;
  }

  bool unregisterFunc(hipModule_t hmod);
  std::vector< std::pair<hipModule_t, bool> >* unregisterVar(hipModule_t hmod);


  bool findSymbol(const void *hostVar, hipModule_t &hmod, std::string &devName);
  PlatformState::DeviceVar* findVar(std::string hostVar, int deviceId, hipModule_t hmod);
  void registerVarSym(const void *hostVar, hipModule_t hmod, const char *symbolName);
  void registerVar(const char* symbolName, const DeviceVar& var);
  void registerFunction(const void* hostFunction, const DeviceFunction& func);

  bool registerModFuncs(std::vector<std::string>& func_names, hipModule_t* module);
  bool findModFunc(hipFunction_t* hfunc, hipModule_t hmod, const char* name);
  bool createFunc(hipFunction_t* hfunc, hipModule_t hmod, const char* name);
  hipFunction_t getFunc(const void* hostFunction, int deviceId);
  bool getFuncAttr(const void* hostFunction, hipFuncAttributes* func_attr);
  bool getGlobalVar(const char* hostVar, int deviceId, hipModule_t hmod,
                    hipDeviceptr_t* dev_ptr, size_t* size_ptr);
  bool getTexRef(const char* hostVar, hipModule_t hmod, textureReference** texRef);

  bool getGlobalVarFromSymbol(const void* hostVar, int deviceId,
                              hipDeviceptr_t* dev_ptr, size_t* size_ptr);

  bool getShadowVarInfo(std::string var_name, hipModule_t hmod,
                        void** var_addr, size_t* var_size);
  void setupArgument(const void *arg, size_t size, size_t offset);
  void configureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream);

  void popExec(ihipExec_t& exec);
};

/// Wait all active streams on the blocking queue. The method enqueues a wait command and
/// doesn't stall the current thread
extern void iHipWaitActiveStreams(amd::HostQueue* blocking_queue, bool wait_null_stream = false);

extern std::vector<hip::Device*> g_devices;
extern hipError_t ihipDeviceGetCount(int* count);
extern int ihipGetDevice();
extern hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags);
extern amd::Memory* getMemoryObject(const void* ptr, size_t& offset);
extern bool CL_CALLBACK getSvarInfo(cl_program program, std::string var_name, void** var_addr,
                                    size_t* var_size);

#endif // HIP_SRC_HIP_INTERNAL_H
