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

#define HIP_INIT() \
  std::call_once(hip::g_ihipInitialized, hip::init);        \
  if (hip::g_context == nullptr && g_devices.size() > 0) {  \
    hip::g_context = g_devices[0];                          \
  }

// This macro should be called at the beginning of every HIP API.
#define HIP_INIT_API(...)                                    \
  LogPrintfInfo("%s ( %s )", __func__, ToString( __VA_ARGS__ ).c_str()); \
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

  extern void init();

  extern amd::Context* getCurrentContext();
  extern void setCurrentContext(unsigned int index);

  extern amd::HostQueue* getNullStream();
  extern void syncStreams();


  struct Function {
    amd::Kernel* function_;
    amd::Monitor lock_;

    Function(amd::Kernel* f) : function_(f), lock_("function lock") {}
    hipFunction_t asHipFunction() { return reinterpret_cast<hipFunction_t>(this); }

    static Function* asFunction(hipFunction_t f) { return reinterpret_cast<Function*>(f); }
  };
};
extern std::vector<amd::Context*> g_devices;
extern hipError_t ihipDeviceGetCount(int* count);
extern amd::Memory* getMemoryObject(const void* ptr, size_t& offset);

#define HIP_RETURN(ret)          \
        hip::g_lastError = ret;  \
        DebugInfoGuarantee(hip::g_lastError == hipSuccess); \
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
