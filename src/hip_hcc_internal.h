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

#ifndef HIP_SRC_HIP_HCC_INTERNAL_H
#define HIP_SRC_HIP_HCC_INTERNAL_H

#include <hc.hpp>
#include <hsa/hsa.h>
#include <unordered_map>
#include <stack>

#include "hsa/hsa_ext_amd.h"
#include "hip/hip_runtime.h"
#include "hip_util.h"
#include "env.h"


#if (__hcc_workweek__ < 16354)
#error("This version of HIP requires a newer version of HCC.");
#endif

// Use the __appPtr field in the am memtracker to store the context.
// Requires a bug fix in HCC
#if defined(__HCC_HAS_EXTENDED_AM_MEMTRACKER_UPDATE) and                                           \
    (__HCC_HAS_EXTENDED_AM_MEMTRACKER_UPDATE != 0)
#define USE_APP_PTR_FOR_CTX 1
#endif


#define USE_IPC 1

//---
// Environment variables:

// Intended to distinguish whether an environment variable should be visible only in debug mode, or
// in debug+release.
// static const int debug = 0;
extern const int release;

// TODO - this blocks both kernels and memory ops.  Perhaps should have separate env var for
// kernels?
extern int HIP_LAUNCH_BLOCKING;
extern int HIP_API_BLOCKING;

extern int HIP_PRINT_ENV;
extern int HIP_PROFILE_API;
// extern int HIP_TRACE_API;
extern int HIP_ATP;
extern int HIP_DB;
extern int HIP_STAGING_SIZE;    /* size of staging buffers, in KB */
extern int HIP_STREAM_SIGNALS;  /* number of signals to allocate at stream creation */
extern int HIP_VISIBLE_DEVICES; /* Contains a comma-separated sequence of GPU identifiers */
extern int HIP_FORCE_P2P_HOST;

extern int HIP_HOST_COHERENT;

extern int HIP_HIDDEN_FREE_MEM;
//---
// Chicken bits for disabling functionality to work around potential issues:
extern int HIP_SYNC_HOST_ALLOC;
extern int HIP_SYNC_STREAM_WAIT;

extern int HIP_SYNC_NULL_STREAM;
extern int HIP_INIT_ALLOC;
extern int HIP_FORCE_NULL_STREAM;

extern int HIP_DUMP_CODE_OBJECT;

// TODO - remove when this is standard behavior.
extern int HCC_OPT_FLUSH;

// Class to assign a short TID to each new thread, for HIP debugging purposes.
class TidInfo {
   public:
    TidInfo();

    int tid() const { return _shortTid; };
    pid_t pid() const { return _pid; }; 
    uint64_t incApiSeqNum() { return ++_apiSeqNum; };
    uint64_t apiSeqNum() const { return _apiSeqNum; };

   private:
    int _shortTid;
    pid_t _pid; 

    // monotonically increasing API sequence number for this threa.
    uint64_t _apiSeqNum;
};

struct ProfTrigger {
    static const uint64_t MAX_TRIGGER = std::numeric_limits<uint64_t>::max();

    void print(int tid) {
        std::cout << "Enabling tracing for ";
        for (auto iter = _profTrigger.begin(); iter != _profTrigger.end(); iter++) {
            std::cout << "tid:" << tid << "." << *iter << ",";
        }
        std::cout << "\n";
    };

    uint64_t nextTrigger() { return _profTrigger.empty() ? MAX_TRIGGER : _profTrigger.back(); };
    void add(uint64_t trigger) { _profTrigger.push_back(trigger); };
    void sort() { std::sort(_profTrigger.begin(), _profTrigger.end(), std::greater<int>()); };

   private:
    std::vector<uint64_t> _profTrigger;
};


//---
// Extern tls
extern thread_local hipError_t tls_lastHipError;
extern thread_local TidInfo tls_tidInfo;
extern thread_local bool tls_getPrimaryCtx;

extern std::vector<ProfTrigger> g_dbStartTriggers;
extern std::vector<ProfTrigger> g_dbStopTriggers;

//---
// Forward defs:
class ihipStream_t;
class ihipDevice_t;
class ihipCtx_t;
struct ihipEventData_t;

// Color defs for debug messages:
#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

extern const char* API_COLOR;
extern const char* API_COLOR_END;


// If set, thread-safety is enforced on all event/stream/ctx/device functions.
// Can disable for performance or functional experiments - in this case
// the code uses a dummy "no-op" mutex.
#define EVENT_THREAD_SAFE 1

#define STREAM_THREAD_SAFE 1

#define CTX_THREAD_SAFE 1

#define DEVICE_THREAD_SAFE 1


// Compile debug trace mode - this prints debug messages to stderr when env var HIP_DB is set.
// May be set to 0 to remove debug if checks - possible code size and performance difference?
#define COMPILE_HIP_DB 1


// Compile HIP tracing capability.
// 0x1 = print a string at function entry with arguments.
// 0x2 = prints a simple message with function name + return code when function exits.
// 0x3 = print both.
// Must be enabled at runtime with HIP_TRACE_API
#define COMPILE_HIP_TRACE_API 0x3


// Compile code that generates trace markers for CodeXL ATP at HIP function begin/end.
// ATP is standard CodeXL format that includes timestamps for kernels, HSA RT APIs, and HIP APIs.
#ifndef COMPILE_HIP_ATP_MARKER
#define COMPILE_HIP_ATP_MARKER 0
#endif


// Compile support for trace markers that are displayed on CodeXL GUI at start/stop of each function
// boundary.
// TODO - currently we print the trace message at the beginning. if we waited, we could also
// tls_tidInfo return codes, and any values returned through ptr-to-args (ie the pointers allocated
// by hipMalloc).
#if COMPILE_HIP_ATP_MARKER
#include "CXLActivityLogger.h"
#define MARKER_BEGIN(markerName, group) amdtBeginMarker(markerName, group, nullptr);
#define MARKER_END() amdtEndMarker();
#define RESUME_PROFILING amdtResumeProfiling(AMDT_ALL_PROFILING);
#define STOP_PROFILING amdtStopProfiling(AMDT_ALL_PROFILING);
#else
// Swallow scoped markers:
#define MARKER_BEGIN(markerName, group)
#define MARKER_END()
#define RESUME_PROFILING
#define STOP_PROFILING
#endif


//---
// HIP Trace modes - use with HIP_TRACE_API=...
#define TRACE_ALL 0    // 0x01
#define TRACE_KCMD 1   // 0x02, kernel command
#define TRACE_MCMD 2   // 0x04, memory command
#define TRACE_MEM 3    // 0x08, memory allocation or deallocation.
#define TRACE_SYNC 4   // 0x10, synchronization (host or hipStreamWaitEvent)
#define TRACE_QUERY 5  // 0x20, hipEventRecord, hipEventQuery, hipStreamQuery


//---
// HIP_DB Debug flags:
#define DB_API 0  /* 0x01 - shortcut to enable HIP_TRACE_API on single switch */
#define DB_SYNC 1 /* 0x02 - trace synchronization pieces */
#define DB_MEM 2  /* 0x04 - trace memory allocation / deallocation */
#define DB_COPY 3 /* 0x08 - trace memory copy and peer commands. . */
#define DB_WARN 4 /* 0x10 - warn about sub-optimal or shady behavior */
#define DB_FB 5   /* 0x20 - trace loading fat binary */
#define DB_MAX_FLAG 6
// When adding a new debug flag, also add to the char name table below.
//
//

struct DbName {
    const char* _color;
    const char* _shortName;
};

// This table must be kept in-sync with the defines above.
static const DbName dbName[] = {
    {KGRN, "api"},  // not used,
    {KYEL, "sync"}, {KCYN, "mem"}, {KMAG, "copy"}, {KRED, "warn"},
    {KBLU, "fatbin"},
};


#if COMPILE_HIP_DB
#define tprintf(trace_level, ...)                                                                                     \
    {                                                                                                                 \
        if (HIP_DB & (1 << (trace_level))) {                                                                          \
            char msgStr[1000];                                                                                        \
            snprintf(msgStr, sizeof(msgStr), __VA_ARGS__);                                                            \
            fprintf(stderr, "  %ship-%s pid:%d tid:%d:%s%s", dbName[trace_level]._color,                              \
                    dbName[trace_level]._shortName, tls_tidInfo.pid(), tls_tidInfo.tid(), msgStr, KNRM);              \
        }                                                                                                             \
    }
#else
/* Compile to empty code */
#define tprintf(trace_level, ...)
#endif


static inline uint64_t getTicks() { return hc::get_system_ticks(); }

//---
extern uint64_t recordApiTrace(std::string* fullStr, const std::string& apiStr);

#if COMPILE_HIP_ATP_MARKER || (COMPILE_HIP_TRACE_API & 0x1)
#define API_TRACE(forceTrace, ...)                                                                 \
    uint64_t hipApiStartTick = 0;                                                                  \
    {                                                                                              \
        tls_tidInfo.incApiSeqNum();                                                                \
        if (forceTrace ||                                                                          \
            (HIP_PROFILE_API || (COMPILE_HIP_DB && (HIP_TRACE_API & (1 << TRACE_ALL))))) {         \
            std::string apiStr = std::string(__func__) + " (" + ToString(__VA_ARGS__) + ')';       \
            std::string fullStr;                                                                   \
            hipApiStartTick = recordApiTrace(&fullStr, apiStr);                                    \
            if (HIP_PROFILE_API == 0x1) {                                                          \
                MARKER_BEGIN(__func__, "HIP")                                                      \
            } else if (HIP_PROFILE_API == 0x2) {                                                   \
                MARKER_BEGIN(fullStr.c_str(), "HIP");                                              \
            }                                                                                      \
        }                                                                                          \
    }

#else
// Swallow API_TRACE
#define API_TRACE(IS_CMD, ...) tls_tidInfo.incApiSeqNum();
#endif


// Just initialize the HIP runtime, but don't log any trace information.
#define HIP_INIT()                                                                                 \
    std::call_once(hip_initialized, ihipInit);                                                     \
    ihipCtxStackUpdate();
#define HIP_SET_DEVICE() ihipDeviceSetState();


// This macro should be called at the beginning of every HIP API.
// It initializes the hip runtime (exactly once), and
// generates a trace string that can be output to stderr or to ATP file.
#define HIP_INIT_API(cid, ...)                                                                     \
    HIP_INIT()                                                                                     \
    API_TRACE(0, __VA_ARGS__);                                                                     \
    HIP_CB_SPAWNER_OBJECT(cid);


// Like above, but will trace with a specified "special" bit.
// Replace HIP_INIT_API with this call inside HIP APIs that launch work on the GPU:
// kernel launches, copy commands, memory sets, etc.
#define HIP_INIT_SPECIAL_API(cid, tbit, ...)                                                       \
    HIP_INIT()                                                                                     \
    API_TRACE((HIP_TRACE_API & (1 << tbit)), __VA_ARGS__);                                         \
    HIP_CB_SPAWNER_OBJECT(cid);


// This macro should be called at the end of every HIP API, and only at the end of top-level hip
// APIS (not internal hip) It has dual function: logs the last error returned for use by
// hipGetLastError, and also prints the closing message when the debug trace is enabled.
#define ihipLogStatus(hipStatus)                                                                    \
    ({                                                                                              \
        hipError_t localHipStatus = hipStatus; /*local copy so hipStatus only evaluated once*/      \
        tls_lastHipError = localHipStatus;                                                          \
                                                                                                    \
        if ((COMPILE_HIP_TRACE_API & 0x2) && HIP_TRACE_API & (1 << TRACE_ALL)) {                    \
            auto ticks = getTicks() - hipApiStartTick;                                              \
            fprintf(stderr, "  %ship-api pid:%d tid:%d.%lu %-30s ret=%2d (%s)>> +%lu ns%s\n",       \
                    (localHipStatus == 0) ? API_COLOR : KRED, tls_tidInfo.pid(), tls_tidInfo.tid(), \
                    tls_tidInfo.apiSeqNum(), __func__, localHipStatus,                              \
                    ihipErrorString(localHipStatus), ticks, API_COLOR_END);                         \
        }                                                                                           \
        if (HIP_PROFILE_API) {                                                                      \
            MARKER_END();                                                                           \
        }                                                                                           \
        localHipStatus;                                                                             \
    })


class ihipException : public std::exception {
   public:
    explicit ihipException(hipError_t e) : _code(e){};

    hipError_t _code;
};


#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
}
#endif

const hipStream_t hipStreamNull = 0x0;


/**
 * HIP IPC Handle Size
 */
#define HIP_IPC_RESERVED_SIZE 24
class ihipIpcMemHandle_t {
   public:
#if USE_IPC
    hsa_amd_ipc_memory_t ipc_handle;  ///< ipc memory handle on ROCr
#endif
    size_t psize;
    char reserved[HIP_IPC_RESERVED_SIZE];
};


struct ihipModule_t {
    std::string fileName;
    hsa_executable_t executable = {};
    hsa_code_object_reader_t coReader = {};
    std::string hash;

    ~ihipModule_t() {
        if (executable.handle) hsa_executable_destroy(executable);
        if (coReader.handle) hsa_code_object_reader_destroy(coReader);
    }
};


//---
// Used to remove lock, for performance or stimulating bugs.
class FakeMutex {
   public:
    void lock() {}
    bool try_lock() { return true; }
    void unlock() {}
};

#if EVENT_THREAD_SAFE
typedef std::mutex EventMutex;
#else
#warning "Stream thread-safe disabled"
typedef FakeMutex EventMutex;
#endif

#if STREAM_THREAD_SAFE
typedef std::mutex StreamMutex;
#else
#warning "Stream thread-safe disabled"
typedef FakeMutex StreamMutex;
#endif

// Pair Device and Ctx together, these could also be toggled separately if desired.
#if CTX_THREAD_SAFE
typedef std::mutex CtxMutex;
#else
typedef FakeMutex CtxMutex;
#warning "Ctx thread-safe disabled"
#endif

#if DEVICE_THREAD_SAFE
typedef std::mutex DeviceMutex;
#else
typedef FakeMutex DeviceMutex;
#warning "Device thread-safe disabled"
#endif

//
//---
// Protects access to the member _data with a lock acquired on contruction/destruction.
// T must contain a _mutex field which meets the BasicLockable requirements (lock/unlock)
template <typename T>
class LockedAccessor {
   public:
    LockedAccessor(T& criticalData, bool autoUnlock = true)
        : _criticalData(&criticalData),
          _autoUnlock(autoUnlock)

    {
        tprintf(DB_SYNC, "locking criticalData=%p for %s..\n", _criticalData,
                ToString(_criticalData->_parent).c_str());
        _criticalData->_mutex.lock();
    };

    ~LockedAccessor() {
        if (_autoUnlock) {
            tprintf(DB_SYNC, "auto-unlocking criticalData=%p for %s...\n", _criticalData,
                    ToString(_criticalData->_parent).c_str());
            _criticalData->_mutex.unlock();
        }
    }

    void unlock() {
        tprintf(DB_SYNC, "unlocking criticalData=%p for %s...\n", _criticalData,
                ToString(_criticalData->_parent).c_str());
        _criticalData->_mutex.unlock();
    }

    // Syntactic sugar so -> can be used to get the underlying type.
    T* operator->() { return _criticalData; };

   private:
    T* _criticalData;
    bool _autoUnlock;
};


template <typename MUTEX_TYPE>
struct LockedBase {
    // Experts-only interface for explicit locking.
    // Most uses should use the lock-accessor.
    void lock() { _mutex.lock(); }
    void unlock() { _mutex.unlock(); }
    bool try_lock() { return _mutex.try_lock(); }

    MUTEX_TYPE _mutex;
};


template <typename MUTEX_TYPE>
class ihipStreamCriticalBase_t : public LockedBase<MUTEX_TYPE> {
   public:
    ihipStreamCriticalBase_t(ihipStream_t* parentStream, hc::accelerator_view av)
        : _kernelCnt(0), _av(av), _parent(parentStream){};

    ~ihipStreamCriticalBase_t() {}

    ihipStreamCriticalBase_t<StreamMutex>* mlock() {
        LockedBase<MUTEX_TYPE>::lock();
        return this;
    };

    void munlock() {
        tprintf(DB_SYNC, "munlocking criticalData=%p for %s...\n", this,
                ToString(this->_parent).c_str());
        LockedBase<MUTEX_TYPE>::unlock();
    };

    ihipStreamCriticalBase_t<StreamMutex>* mtry_lock() {
        bool gotLock = LockedBase<MUTEX_TYPE>::try_lock();
        tprintf(DB_SYNC, "mtry_locking=%d criticalData=%p for %s...\n", gotLock, this,
                ToString(this->_parent).c_str());
        return gotLock ? this : nullptr;
    };

   public:
    ihipStream_t* _parent;
    uint32_t _kernelCnt;  // Count of inflight kernels in this stream.  Reset at ::wait().

    hc::accelerator_view _av;

   private:
};


// if HIP code needs to acquire locks for both ihipCtx_t and ihipStream_t, it should first acquire
// the lock for the ihipCtx_t and then for the individual streams.  The locks should not be acquired
// in reverse order or deadlock may occur.  In some cases, it may be possible to reduce the range
// where the locks must be held. HIP routines should avoid acquiring and releasing the same lock
// during the execution of a single HIP API. Another option is to use try_lock in the innermost lock
// query.


typedef ihipStreamCriticalBase_t<StreamMutex> ihipStreamCritical_t;
typedef LockedAccessor<ihipStreamCritical_t> LockedAccessor_StreamCrit_t;

//---
// Internal stream structure.
class ihipStream_t {
   public:
    enum ScheduleMode { Auto, Spin, Yield };
    typedef uint64_t SeqNum_t;

    // TODOD -make av a reference to avoid shared_ptr overhead?
    ihipStream_t(ihipCtx_t* ctx, hc::accelerator_view av, unsigned int flags);
    ~ihipStream_t();

    // kind is hipMemcpyKind
    void locked_copySync(void* dst, const void* src, size_t sizeBytes, unsigned kind,
                         bool resolveOn = true);

    void locked_copy2DSync(void* dst, const void* src, size_t width, size_t height, size_t srcPitch, size_t dstPitch, unsigned kind,
                         bool resolveOn = true);

    void locked_copyAsync(void* dst, const void* src, size_t sizeBytes, unsigned kind);

    void locked_copy2DAsync(void* dst, const void* src, size_t width, size_t height, size_t srcPitch, size_t dstPitch, unsigned kind);

    void lockedSymbolCopySync(hc::accelerator& acc, void* dst, void* src, size_t sizeBytes,
                              size_t offset, unsigned kind);
    void lockedSymbolCopyAsync(hc::accelerator& acc, void* dst, void* src, size_t sizeBytes,
                               size_t offset, unsigned kind);

    //---
    // Member functions that begin with locked_ are thread-safe accessors - these acquire / release
    // the critical mutex.
    LockedAccessor_StreamCrit_t lockopen_preKernelCommand();
    void lockclose_postKernelCommand(const char* kernelName, hc::accelerator_view* av);


    void locked_wait();

    hc::accelerator_view* locked_getAv() {
        LockedAccessor_StreamCrit_t crit(_criticalData);
        return &(crit->_av);
    };

    void locked_streamWaitEvent(ihipEventData_t& event);
    hc::completion_future locked_recordEvent(hipEvent_t event);

    bool locked_eventIsReady(hipEvent_t event);
    void locked_eventWaitComplete(hc::completion_future& marker, hc::hcWaitMode waitMode);

    ihipStreamCritical_t& criticalData() { return _criticalData; };

    //---
    hc::hcWaitMode waitMode() const;

    // Use this if we already have the stream critical data mutex:
    void wait(LockedAccessor_StreamCrit_t& crit);

    void launchModuleKernel(hc::accelerator_view av, hsa_signal_t signal, uint32_t blockDimX,
                            uint32_t blockDimY, uint32_t blockDimZ, uint32_t gridDimX,
                            uint32_t gridDimY, uint32_t gridDimZ, uint32_t groupSegmentSize,
                            uint32_t sharedMemBytes, void* kernarg, size_t kernSize,
                            uint64_t kernel);


    //-- Non-racy accessors:
    // These functions access fields set at initialization time and are non-racy (so do not acquire
    // mutex)
    const ihipDevice_t* getDevice() const;
    ihipCtx_t* getCtx() const;

    // Before calling this function, stream must be resolved from "0" to the actual stream:
    bool isDefaultStream() const { return _id == 0; };

   public:
    //---
    // Public member vars - these are set at initialization and never change:
    SeqNum_t _id;  // monotonic sequence ID.  0 is the default stream.
    unsigned _flags;


   private:
    // The unsigned return is hipMemcpyKind
    unsigned resolveMemcpyDirection(bool srcInDeviceMem, bool dstInDeviceMem);
    void resolveHcMemcpyDirection(unsigned hipMemKind, const hc::AmPointerInfo* dstPtrInfo,
                                  const hc::AmPointerInfo* srcPtrInfo, hc::hcCommandKind* hcCopyDir,
                                  ihipCtx_t** copyDevice, bool* forceUnpinnedCopy);

    bool canSeeMemory(const ihipCtx_t* thisCtx, const hc::AmPointerInfo* dstInfo,
                      const hc::AmPointerInfo* srcInfo);

    void addSymbolPtrToTracker(hc::accelerator& acc, void* ptr, size_t sizeBytes);


   public:  // TODO - move private
    // Critical Data - MUST be accessed through LockedAccessor_StreamCrit_t
    ihipStreamCritical_t _criticalData;

   private:  // Data
    std::mutex _hasQueueLock;

    ihipCtx_t* _ctx;  // parent context that owns this stream.

    // Friends:
    friend std::ostream& operator<<(std::ostream& os, const ihipStream_t& s);
    friend hipError_t hipStreamQuery(hipStream_t);

    ScheduleMode _scheduleMode;
};


//----
// Internal structure for stream callback handler
class ihipStreamCallback_t {
   public:
    ihipStreamCallback_t(hipStream_t stream, hipStreamCallback_t callback, void* userData)
        : _stream(stream), _callback(callback), _userData(userData) {
    };
    hipStream_t _stream;
    hipStreamCallback_t _callback;
    void* _userData;
};


//----
// Internal event structure:
enum hipEventStatus_t {
    hipEventStatusUnitialized = 0,  // event is uninitialized, must be "Created" before use.
    hipEventStatusCreated = 1,      // event created, but not yet Recorded
    hipEventStatusRecording = 2,    // event has been recorded into a stream but not completed yet.
    hipEventStatusComplete = 3,     // event has been recorded - timestamps are valid.
};

// TODO - rename to ihip type of some kind
enum ihipEventType_t {
    hipEventTypeIndependent,
    hipEventTypeStartCommand,
    hipEventTypeStopCommand,
};


struct ihipEventData_t {
    ihipEventData_t() {
        _state = hipEventStatusCreated;
        _stream = NULL;
        _timestamp = 0;
        _type = hipEventTypeIndependent;
    };

    void marker(const hc::completion_future& marker) { _marker = marker; };
    hc::completion_future& marker() { return _marker; }
    uint64_t timestamp() const { return _timestamp; };
    ihipEventType_t type() const { return _type; };

    ihipEventType_t _type;
    hipEventStatus_t _state;
    hipStream_t _stream;  // Stream where the event is recorded.  Null stream is resolved to actual
                          // stream when recorded
    uint64_t _timestamp;  // store timestamp, may be set on host or by marker.
   private:
    hc::completion_future _marker;
};


//=============================================================================
// class ihipEventCriticalBase_t
template <typename MUTEX_TYPE>
class ihipEventCriticalBase_t : LockedBase<MUTEX_TYPE> {
   public:
    explicit ihipEventCriticalBase_t(const ihipEvent_t* parentEvent) : _parent(parentEvent) {}
    ~ihipEventCriticalBase_t(){};

    // Keep data in structure so it can be easily copied into snapshots
    // (used to reduce lock contention and preserve correct lock order)
    ihipEventData_t _eventData;

   private:
    const ihipEvent_t* _parent;
    friend class LockedAccessor<ihipEventCriticalBase_t>;
};

typedef ihipEventCriticalBase_t<EventMutex> ihipEventCritical_t;

typedef LockedAccessor<ihipEventCritical_t> LockedAccessor_EventCrit_t;

// internal hip event structure.
class ihipEvent_t {
   public:
    explicit ihipEvent_t(unsigned flags);
    void attachToCompletionFuture(const hc::completion_future* cf, hipStream_t stream,
                                  ihipEventType_t eventType);
    std::pair<hipEventStatus_t, uint64_t> refreshEventStatus();  // returns pair <state, timestamp>


    // Return a copy of the critical state. The critical data is locked during the copy.
    ihipEventData_t locked_copyCrit() {
        LockedAccessor_EventCrit_t crit(_criticalData);
        return _criticalData._eventData;
    };

    ihipEventCritical_t& criticalData() { return _criticalData; };

   public:
    unsigned _flags;

   private:
    ihipEventCritical_t _criticalData;

    friend hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
};


//=============================================================================
// class ihipDeviceCriticalBase_t
template <typename MUTEX_TYPE>
class ihipDeviceCriticalBase_t : LockedBase<MUTEX_TYPE> {
   public:
    explicit ihipDeviceCriticalBase_t(ihipDevice_t* parentDevice)
        : _parent(parentDevice), _ctxCount(0){};

    ~ihipDeviceCriticalBase_t() {}

    // Contexts:
    void addContext(ihipCtx_t* ctx);
    void removeContext(ihipCtx_t* ctx);
    std::list<ihipCtx_t*>& ctxs() { return _ctxs; };
    const std::list<ihipCtx_t*>& const_ctxs() const { return _ctxs; };
    int getcount() { return _ctxCount; };
    friend class LockedAccessor<ihipDeviceCriticalBase_t>;

   private:
    ihipDevice_t* _parent;

    //--- Context Tracker:
    std::list<ihipCtx_t*> _ctxs;  // contexts associated with this device across all threads.

    int _ctxCount;
};

typedef ihipDeviceCriticalBase_t<DeviceMutex> ihipDeviceCritical_t;

typedef LockedAccessor<ihipDeviceCritical_t> LockedAccessor_DeviceCrit_t;

//----
// Properties of the HIP device.
// Multiple contexts can point to same device.
class ihipDevice_t {
   public:
    ihipDevice_t(unsigned deviceId, unsigned deviceCnt, hc::accelerator& acc);
    ~ihipDevice_t();

    // Accessors:
    ihipCtx_t* getPrimaryCtx() const { return _primaryCtx; };
    void locked_removeContext(ihipCtx_t* c);
    void locked_reset();
    ihipDeviceCritical_t& criticalData() { return _criticalData; };

   public:
    unsigned _deviceId;  // device ID

    hc::accelerator _acc;
    hsa_agent_t _hsaAgent;  // hsa agent handle

    //! Number of compute units supported by the device:
    unsigned _computeUnits;
    hipDeviceProp_t _props;  // saved device properties.

    // TODO - report this through device properties, base on HCC API call.
    int _isLargeBar;

    ihipCtx_t* _primaryCtx;

    int _state;  // 1 if device is set otherwise 0

   private:
    hipError_t initProperties(hipDeviceProp_t* prop);

   private:
    ihipDeviceCritical_t _criticalData;
};
//=============================================================================


//---
//
struct ihipExec_t {
  dim3 _gridDim;
  dim3 _blockDim;
  size_t _sharedMem;
  hipStream_t _hStream;
  std::vector<char> _arguments;
};

//=============================================================================
// class ihipCtxCriticalBase_t
template <typename MUTEX_TYPE>
class ihipCtxCriticalBase_t : LockedBase<MUTEX_TYPE> {
   public:
    ihipCtxCriticalBase_t(ihipCtx_t* parentCtx, unsigned deviceCnt)
        : _parent(parentCtx), _peerCnt(0) {
        _peerAgents = new hsa_agent_t[deviceCnt];
    };

    ~ihipCtxCriticalBase_t() {
        if (_peerAgents != nullptr) {
            delete _peerAgents;
            _peerAgents = nullptr;
        }
        _peerCnt = 0;
    }

    // Streams:
    void addStream(ihipStream_t* stream);
    std::list<ihipStream_t*>& streams() { return _streams; };
    const std::list<ihipStream_t*>& const_streams() const { return _streams; };


    // Peer Accessor classes:
    bool isPeerWatcher(const ihipCtx_t* peer);  // returns True if peer has access to memory
                                                // physically located on this device.
    bool addPeerWatcher(const ihipCtx_t* thisCtx, ihipCtx_t* peer);
    bool removePeerWatcher(const ihipCtx_t* thisCtx, ihipCtx_t* peer);
    void resetPeerWatchers(ihipCtx_t* thisDevice);
    void printPeerWatchers(FILE* f) const;

    uint32_t peerCnt() const { return _peerCnt; };
    hsa_agent_t* peerAgents() const { return _peerAgents; };


    // TODO - move private
    std::list<ihipCtx_t*> _peers;  // list of enabled peer devices.
    //--- Execution stack:
    std::stack<ihipExec_t> _execStack; // Execution stack for this device.

    friend class LockedAccessor<ihipCtxCriticalBase_t>;

   private:
    ihipCtx_t* _parent;

    //--- Stream Tracker:
    std::list<ihipStream_t*> _streams;  // streams associated with this device.


    //--- Peer Tracker:
    // These reflect the currently Enabled set of peers for this GPU:
    // Enabled peers have permissions to access the memory physically allocated on this device.
    // Note the peers always contain the self agent for easy interfacing with HSA APIs.
    uint32_t _peerCnt;         // number of enabled peers
    hsa_agent_t* _peerAgents;  // efficient packed array of enabled agents (to use for allocations.)
   private:
    void recomputePeerAgents();
};
// Note Mutex type Real/Fake selected based on CtxMutex
typedef ihipCtxCriticalBase_t<CtxMutex> ihipCtxCritical_t;

// This type is used by functions that need access to the critical device structures.
typedef LockedAccessor<ihipCtxCritical_t> LockedAccessor_CtxCrit_t;
//=============================================================================


//=============================================================================
// class ihipCtx_t:
// A HIP CTX (context) points at one of the existing devices and contains the streams,
// peer-to-peer mappings, creation flags.  Multiple contexts can point to the same
// device.
//
class ihipCtx_t {
   public:  // Functions:
    ihipCtx_t(ihipDevice_t* device, unsigned deviceCnt,
              unsigned flags);  // note: calls constructor for _criticalData
    ~ihipCtx_t();

    // Functions which read or write the critical data are named locked_.
    // (might be better called "locking_"
    // ihipCtx_t does not use recursive locks so the ihip implementation must avoid calling a
    // locked_ function from within a locked_ function. External functions which call several
    // locked_ functions will acquire and release the lock for each function.  if this occurs in
    // performance-sensitive code we may want to refactor by adding non-locked functions and
    // creating a new locked_ member function to call them all.
    void locked_removeStream(ihipStream_t* s);
    void locked_reset();
    void locked_waitAllStreams();
    void locked_syncDefaultStream(bool waitOnSelf, bool syncHost);

    ihipCtxCritical_t& criticalData() { return _criticalData; };

    const ihipDevice_t* getDevice() const { return _device; };
    int getDeviceNum() const { return _device->_deviceId; };

    // TODO - review uses of getWriteableDevice(), can these be converted to getDevice()
    ihipDevice_t* getWriteableDevice() const { return _device; };

    std::string toString() const;

   public:  // Data
    // The NULL stream is used if no other stream is specified.
    // Default stream has special synchronization properties with other streams.
    ihipStream_t* _defaultStream;

    // Flags specified when the context is created:
    unsigned _ctxFlags;

   private:
    ihipDevice_t* _device;


   private:  // Critical data, protected with locked access:
    // Members of _protected data MUST be accessed through the LockedAccessor.
    // Search for LockedAccessor<ihipCtxCritical_t> for examples; do not access _criticalData
    // directly.
    ihipCtxCritical_t _criticalData;
};


//=================================================================================================
// Global variable definition:
extern std::once_flag hip_initialized;
extern unsigned g_deviceCnt;
extern hsa_agent_t g_cpu_agent;   // the CPU agent.
extern hsa_agent_t* g_allAgents;  // CPU agents + all the visible GPU agents.

//=================================================================================================
// Extern functions:
extern void ihipInit();
extern const char* ihipErrorString(hipError_t);
extern ihipCtx_t* ihipGetTlsDefaultCtx();
extern void ihipSetTlsDefaultCtx(ihipCtx_t* ctx);
extern hipError_t ihipSynchronize(void);
extern void ihipCtxStackUpdate();
extern hipError_t ihipDeviceSetState();

extern ihipDevice_t* ihipGetDevice(int);
ihipCtx_t* ihipGetPrimaryCtx(unsigned deviceIndex);


hipStream_t ihipSyncAndResolveStream(hipStream_t);
hipError_t ihipStreamSynchronize(hipStream_t stream);
void ihipStreamCallbackHandler(ihipStreamCallback_t* cb);

// Stream printf functions:
inline std::ostream& operator<<(std::ostream& os, const ihipStream_t& s) {
    os << "stream:";
    os << s.getDevice()->_deviceId;
    ;
    os << '.';
    os << s._id;
    return os;
}

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

inline std::ostream& operator<<(std::ostream& os, const gl_dim3& s) {
    os << '{';
    os << s.x;
    os << ',';
    os << s.y;
    os << ',';
    os << s.z;
    os << '}';
    return os;
}

// Stream printf functions:
inline std::ostream& operator<<(std::ostream& os, const hipEvent_t& e) {
    os << "event:" << std::hex << static_cast<void*>(e);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const ihipCtx_t* c) {
    os << "ctx:" << static_cast<const void*>(c) << ".dev:" << c->getDevice()->_deviceId;
    return os;
}


// Helper functions that are used across src files:
namespace hip_internal {
hipError_t memcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                       hipStream_t stream);
};


#endif
