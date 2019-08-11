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

#ifndef HIP_SRC_HIP_INTERNAL_H
#define HIP_SRC_HIP_INTERNAL_H

#include "cl_common.hpp"
#include "trace_helper.h"
#include "utils/debug.hpp"
#include <unordered_set>
#include <thread>
#include <stack>
#include <mutex>

/*! IHIP IPC MEMORY Structure */
#define IHIP_IPC_MEM_HANDLE_SIZE   32
#define IHIP_IPC_MEM_RESERVED_SIZE LP64_SWITCH(28,24)

typedef struct ihipIpcMemHandle_st {
  char ipc_handle[IHIP_IPC_MEM_HANDLE_SIZE];  ///< ipc memory handle on ROCr
  size_t psize;
  char reserved[IHIP_IPC_MEM_RESERVED_SIZE];
} ihipIpcMemHandle_t;

#define HIP_INIT() \
  std::call_once(hip::g_ihipInitialized, hip::init);        \
  if (hip::g_context == nullptr && g_devices.size() > 0) {  \
    hip::g_context = g_devices[0];                          \
  }

// This macro should be called at the beginning of every HIP API.
#define HIP_INIT_API(...)                                    \
  LogPrintfInfo("[%zx] %s ( %s )", std::this_thread::get_id(), __func__, ToString( __VA_ARGS__ ).c_str()); \
  amd::Thread* thread = amd::Thread::current();              \
  if (!CL_CHECK_THREAD(thread)) {                            \
    HIP_RETURN(hipErrorOutOfMemory);                         \
  }                                                          \
  HIP_INIT();

namespace hc {
class accelerator;
class accelerator_view;
};

namespace hip {
  extern std::once_flag g_ihipInitialized;
  extern thread_local amd::Context* g_context;
  extern thread_local hipError_t g_lastError;
  extern amd::Context* host_context;

  extern void init();

  extern amd::Context* getCurrentContext();
  extern void setCurrentContext(unsigned int index);

  extern amd::HostQueue* getNullStream(amd::Context&);
  extern amd::HostQueue* getNullStream();
  extern void syncStreams();


  struct Function {
    amd::Kernel* function_;
    amd::Monitor lock_;

    Function(amd::Kernel* f) : function_(f), lock_("function lock") {}
    hipFunction_t asHipFunction() { return reinterpret_cast<hipFunction_t>(this); }

    static Function* asFunction(hipFunction_t f) { return reinterpret_cast<Function*>(f); }
  };

  struct Stream {
    amd::HostQueue* queue;

    amd::Device* device;
    amd::Context* context;
    amd::CommandQueue::Priority priority;
    unsigned int flags;

    Stream(amd::Device* dev, amd::Context* ctx, amd::CommandQueue::Priority p, unsigned int f);
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
  amd::Monitor lock_;

public:
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
  };
private:
  std::unordered_map<const void*, DeviceFunction > functions_;
  std::unordered_map<std::string, DeviceVar > vars_;

  static PlatformState* platform_;

  PlatformState() : lock_("Guards global function map") {}
  ~PlatformState() {}
public:
  static PlatformState& instance() {
    return *platform_;
  }

  void unregisterVar(hipModule_t hmod);

  void registerVar(const void* hostvar, const DeviceVar& var);
  void registerFunction(const void* hostFunction, const DeviceFunction& func);

  hipFunction_t getFunc(const void* hostFunction, int deviceId);
  bool getFuncAttr(const void* hostFunction, hipFuncAttributes* func_attr);
  bool getGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                    size_t* size_ptr);
  bool getShadowVarInfo(std::string var_name, void** var_addr, size_t* var_size);
  void setupArgument(const void *arg, size_t size, size_t offset);
  void configureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream);

  void popExec(ihipExec_t& exec);
};

extern std::vector<amd::Context*> g_devices;
extern hipError_t ihipDeviceGetCount(int* count);
extern int ihipGetDevice();
extern hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags);
extern amd::Memory* getMemoryObject(const void* ptr, size_t& offset);

#define HIP_RETURN(ret)          \
        hip::g_lastError = ret;  \
        LogPrintfInfo("[%zx] %s: Returned %s", std::this_thread::get_id(), __func__, hipGetErrorName(hip::g_lastError)); \
        return hip::g_lastError; \

inline std::ostream& operator<<(std::ostream& os, const dim3& s) {
    os << '{';
    os << s.x;
    os << ',';
    os << s.y;
    os << ',';
    os << s.z;
    os << '}';
    return os;
}

#endif // HIP_SRC_HIP_INTERNAL_H
