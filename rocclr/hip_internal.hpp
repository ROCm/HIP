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

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

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

#define HIP_API_PRINT(...)                                 \
  uint64_t startTimeUs=0 ; HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs, "%-5d: [%zx] %s%s ( %s )%s",  getpid(), std::this_thread::get_id(), KGRN,    \
          __func__, ToString( __VA_ARGS__ ).c_str(),KNRM);

#define HIP_ERROR_PRINT(err, ...)                             \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%-5d: [%zx] %s: Returned %s : %s", getpid(), std::this_thread::get_id(),  \
          __func__, hipGetErrorName(err), ToString( __VA_ARGS__ ).c_str());

// This macro should be called at the beginning of every HIP API.
#define HIP_INIT_API(cid, ...)                               \
  HIP_API_PRINT(__VA_ARGS__)                                 \
  amd::Thread* thread = amd::Thread::current();              \
  if (!VDI_CHECK_THREAD(thread)) {                           \
    HIP_RETURN(hipErrorOutOfMemory);                         \
  }                                                          \
  HIP_INIT()                                                 \
  HIP_CB_SPAWNER_OBJECT(cid);

#define HIP_RETURN_DURATION(ret, ...)                      \
  hip::g_lastError = ret;                         \
  HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs, "%-5d: [%zx] %s: Returned %s : %s",  getpid(), std::this_thread::get_id(),  \
          __func__, hipGetErrorName(hip::g_lastError), ToString( __VA_ARGS__ ).c_str()); \
  return hip::g_lastError;

#define HIP_RETURN(ret, ...)                      \
  hip::g_lastError = ret;                         \
  HIP_ERROR_PRINT(hip::g_lastError, __VA_ARGS__)  \
  return hip::g_lastError;

#define HIP_RETURN_ONFAIL(func)          \
  do {                                   \
    hipError_t herror = (func);          \
    if (herror != hipSuccess) {          \
      HIP_RETURN(herror);                \
    }                                    \
  } while (0);

// Cannot be use in place of HIP_RETURN.
// Refrain from using for external HIP APIs
#define IHIP_RETURN_ONFAIL(func)         \
  do {                                   \
    hipError_t herror = (func);          \
    if (herror != hipSuccess) {          \
      return herror;                     \
    }                                    \
  } while (0);


namespace hc {
class accelerator;
class accelerator_view;
};

namespace hip {
  class Device;

  class Stream {
  public:
    enum Priority : int {High = -1, Normal = 0, Low = 1};
  private:
    amd::HostQueue* queue_;
    mutable amd::Monitor lock_;
    Device* device_;
    Priority priority_;
    unsigned int flags_;
    bool null_;
    const std::vector<uint32_t> cuMask_;

  public:
    Stream(Device* dev, Priority p = Priority::Normal, unsigned int f = 0, bool null_stream = false,
           const std::vector<uint32_t>& cuMask = {});

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
    Priority GetPriority() const { return priority_; }

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
    /// Store device flags
    unsigned int flags_;
    /// Maintain list of user enabled peers
    std::list<int> userEnabledPeers;

  public:
    Device(amd::Context* ctx, int devId):
      context_(ctx), deviceId_(devId), null_stream_(this, Stream::Priority::Normal, 0, true), flags_(hipDeviceScheduleSpin)
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
    unsigned int getFlags() const { return flags_; }
    void setFlags(unsigned int flags) { flags_ = flags; }
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
};

struct ihipExec_t {
  dim3 gridDim_;
  dim3 blockDim_;
  size_t sharedMem_;
  hipStream_t hStream_;
  std::vector<char> arguments_;
};

/// Wait all active streams on the blocking queue. The method enqueues a wait command and
/// doesn't stall the current thread
extern void iHipWaitActiveStreams(amd::HostQueue* blocking_queue, bool wait_null_stream = false);

extern std::vector<hip::Device*> g_devices;
extern hipError_t ihipDeviceGetCount(int* count);
extern int ihipGetDevice();
extern hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags);
extern amd::Memory* getMemoryObject(const void* ptr, size_t& offset);
extern amd::Memory* getMemoryObjectWithOffset(const void* ptr, const size_t size);
extern bool CL_CALLBACK getSvarInfo(cl_program program, std::string var_name, void** var_addr,
                                    size_t* var_size);

constexpr bool kOptionChangeable = true;
constexpr bool kNewDevProg = false;

constexpr bool kMarkerDisableFlush = true;   //!< Avoids command batch flush in ROCclr

#endif // HIP_SRC_HIP_INTERNAL_H
