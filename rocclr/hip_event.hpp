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

#ifndef HIP_EVENT_H
#define HIP_EVENT_H

#include "hip_internal.hpp"
#include "thread/monitor.hpp"

// Internal structure for stream callback handler
class StreamCallback {
   public:
    StreamCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                  amd::Command* command)
        : stream_(stream), callBack_(callback),
          userData_(userData), command_(command) {
        };
    hipStream_t stream_;
    hipStreamCallback_t callBack_;
    void* userData_;
    amd::Command* command_;
};

void CL_CALLBACK ihipStreamCallback(cl_event event, cl_int command_exec_status, void* user_data);


namespace hip {

class ProfileMarker: public amd::Marker {
public:
  ProfileMarker(amd::HostQueue& queue, bool disableFlush, bool markerTs = false)
  : amd::Marker(queue, disableFlush) {
    profilingInfo_.enabled_ = true;
    profilingInfo_.callback_ = nullptr;
    profilingInfo_.marker_ts_ = markerTs;
    profilingInfo_.clear();
  }
};

#define IPC_SIGNALS_PER_EVENT 32
typedef struct ihipIpcEventShmem_s {
  std::atomic<int> owners;
  std::atomic<int> read_index;
  std::atomic<int> write_index;
  std::atomic<int> signal[IPC_SIGNALS_PER_EVENT];
} ihipIpcEventShmem_t;

class Event {
public:
  Event(unsigned int flags) : flags(flags), lock_("hipEvent_t", true),
                              event_(nullptr), recorded_(false) {
    // No need to init event_ here as addMarker does that
  }

  ~Event() {
    if (event_ != nullptr) {
      event_->release();
    }
  }
  unsigned int flags;

  hipError_t query();
  hipError_t synchronize();
  hipError_t elapsedTime(Event& stop, float& ms);
  hipError_t streamWait(amd::HostQueue* queue, uint flags);

  void addMarker(amd::HostQueue* queue, amd::Command* command, bool record);
  bool isRecorded() { return recorded_; }
  amd::Monitor& lock() { return lock_; }

   //IPC Events
  struct ihipIpcEvent_t {
    std::string ipc_name_;
    int ipc_fd_;
    ihipIpcEventShmem_t* ipc_shmem_;
    ihipIpcEvent_t(): ipc_name_("dummy"), ipc_fd_(0), ipc_shmem_(nullptr) {
    }
    void setipcname(const char* name) {
      ipc_name_ = std::string(name);
    }
  };
  ihipIpcEvent_t ipc_evt_;
private:
  amd::Monitor lock_;
  amd::HostQueue* stream_;
  amd::Event* event_;

  //! Flag to indicate hipEventRecord has been called. This is needed except for
  //! hip*ModuleLaunchKernel API which takes start and stop events so no
  //! hipEventRecord is called. Cleanup needed once those APIs are deprecated.
  bool recorded_;

  bool ready();
};

};

#endif // HIP_EVEMT_H
