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

namespace hip {

class ProfileMarker: public amd::Marker {
public:
  ProfileMarker(amd::HostQueue& queue, bool disableFlush)
  : amd::Marker(queue, disableFlush) {
    profilingInfo_.enabled_ = true;
    profilingInfo_.callback_ = nullptr;
    profilingInfo_.clear();
  }
};

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

  amd::Monitor& lock() { return lock_; }

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
