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

/**
 * @file hip_hcc.cpp
 *
 * Contains definitions for functions that are large enough that we don't want to inline them everywhere.
 * This file is compiled and linked into apps running HIP / HCC path.
 */
#include <assert.h>
#include <stdint.h>
#include <iostream>
#include <sstream>
#include <list>
#include <sys/types.h>
#include <unistd.h>
#include <deque>
#include <vector>
#include <algorithm>
#include <atomic>

#include <hc.hpp>
#include <hc_am.hpp>
#include "hsa/hsa_ext_amd.h"

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"
#include "env.h"


//=================================================================================================
//Global variables:
//=================================================================================================
const int release = 1;

const char *API_COLOR = KGRN;
const char *API_COLOR_END = KNRM;

int HIP_LAUNCH_BLOCKING = 0;
std::string HIP_LAUNCH_BLOCKING_KERNELS;
std::vector<std::string> g_hipLaunchBlockingKernels;
int HIP_API_BLOCKING = 0;


int HIP_PRINT_ENV = 0;
int HIP_TRACE_API= 0;
std::string HIP_TRACE_API_COLOR("green");
int HIP_PROFILE_API= 0;

// TODO - DB_START/STOP need more testing.
std::string HIP_DB_START_API;
std::string HIP_DB_STOP_API;
int HIP_DB= 0;
int HIP_VISIBLE_DEVICES = 0; 
int HIP_WAIT_MODE = 0;

int HIP_FORCE_P2P_HOST = 0;
int HIP_FAIL_SOC = 0;
int HIP_DENY_PEER_ACCESS = 0;

int HIP_HIDDEN_FREE_MEM = 256;
// Force async copies to actually use the synchronous copy interface.
int HIP_FORCE_SYNC_COPY = 0;

// TODO - set these to 0 and 1
int HIP_EVENT_SYS_RELEASE=0;
int HIP_HOST_COHERENT = 1;

int HIP_SYNC_HOST_ALLOC = 1;


int HIP_INIT_ALLOC=-1;
int HIP_SYNC_STREAM_WAIT = 0;
int HIP_FORCE_NULL_STREAM=0;



#if (__hcc_workweek__ >= 17300)
// Make sure we have required bug fix in HCC
// Perform resolution on the GPU:
// Chicken bit to sync on host to implement null stream.
// If 0, null stream synchronization is performed on the GPU
int HIP_SYNC_NULL_STREAM = 0;
#else
int HIP_SYNC_NULL_STREAM = 1;
#endif

// HIP needs to change some behavior based on HCC_OPT_FLUSH :
#if (__hcc_workweek__ >= 17296)
int HCC_OPT_FLUSH = 1;
#else
#warning "HIP disabled HCC_OPT_FLUSH since HCC version does not yet support"
int HCC_OPT_FLUSH = 0;
#endif







std::once_flag hip_initialized;

// Array of pointers to devices.
ihipDevice_t **g_deviceArray;


bool g_visible_device = false;
unsigned g_deviceCnt;
std::vector<int> g_hip_visible_devices;
hsa_agent_t g_cpu_agent;
hsa_agent_t *g_allAgents; // CPU agents + all the visible GPU agents.
unsigned g_numLogicalThreads;

std::atomic<int> g_lastShortTid(1);

// Indexed by short-tid:
//
std::vector<ProfTrigger> g_dbStartTriggers;
std::vector<ProfTrigger> g_dbStopTriggers;

//=================================================================================================
// Thread-local storage:
//=================================================================================================

// This is the implicit context used by all HIP commands.
// It can be set by hipSetDevice or by the CTX manipulation commands:
thread_local hipError_t tls_lastHipError = hipSuccess;


thread_local TidInfo tls_tidInfo;




//=================================================================================================
// Top-level "free" functions:
//=================================================================================================
void recordApiTrace(std::string *fullStr, const std::string &apiStr)
{
    auto apiSeqNum = tls_tidInfo.apiSeqNum();
    auto tid = tls_tidInfo.tid();

    if ((tid < g_dbStartTriggers.size()) && (apiSeqNum >= g_dbStartTriggers[tid].nextTrigger())) {
        printf ("info: resume profiling at %lu\n", apiSeqNum);
        RESUME_PROFILING;
        g_dbStartTriggers.pop_back();
    };
    if ((tid < g_dbStopTriggers.size()) && (apiSeqNum >= g_dbStopTriggers[tid].nextTrigger())) {
        printf ("info: stop profiling at %lu\n", apiSeqNum);
        STOP_PROFILING;
        g_dbStopTriggers.pop_back();
    };

    fullStr->reserve(16 + apiStr.length());
    *fullStr = std::to_string(tid) + ".";
    *fullStr += std::to_string(apiSeqNum);
    *fullStr += " ";
    *fullStr += apiStr;


    if (COMPILE_HIP_DB && HIP_TRACE_API) {
        fprintf (stderr, "%s<<hip-api tid:%s%s\n" , API_COLOR, fullStr->c_str(), API_COLOR_END);
    }
}


static inline bool ihipIsValidDevice(unsigned deviceIndex)
{
    // deviceIndex is unsigned so always > 0
    return (deviceIndex < g_deviceCnt);
}


ihipDevice_t * ihipGetDevice(int deviceIndex)
{
    if (ihipIsValidDevice(deviceIndex)) {
        return g_deviceArray[deviceIndex];
    } else {
        return NULL;
    }
}

ihipCtx_t * ihipGetPrimaryCtx(unsigned deviceIndex)
{
    ihipDevice_t *device = ihipGetDevice(deviceIndex);
    return device ? device->getPrimaryCtx() : NULL;
};


static thread_local ihipCtx_t *tls_defaultCtx = nullptr;
void ihipSetTlsDefaultCtx(ihipCtx_t *ctx)
{
    tls_defaultCtx = ctx;
}


//---
//TODO - review the context creation strategy here.  Really should be:
//  - first "non-device" runtime call creates the context for this thread.  Allowed to call setDevice first.
//  - hipDeviceReset destroys the primary context for device?
//  - Then context is created again for next usage.
ihipCtx_t *ihipGetTlsDefaultCtx()
{
    // Per-thread initialization of the TLS:
    if ((tls_defaultCtx == nullptr) && (g_deviceCnt>0)) {
        ihipSetTlsDefaultCtx(ihipGetPrimaryCtx(0));
    }
    return tls_defaultCtx;
}

hipError_t ihipSynchronize(void)
{
    ihipGetTlsDefaultCtx()->locked_waitAllStreams(); // ignores non-blocking streams, this waits for all activity to finish.

    return (hipSuccess);
}

//=================================================================================================
// ihipStream_t:
//=================================================================================================
TidInfo::TidInfo()  :
    _apiSeqNum(0)
{
    _shortTid = g_lastShortTid.fetch_add(1);

    if (COMPILE_HIP_DB && HIP_TRACE_API) {
        std::stringstream tid_ss;
        std::stringstream tid_ss_num;
        tid_ss_num << std::this_thread::get_id();
        tid_ss << std::hex << std::stoull(tid_ss_num.str());

        tprintf(DB_API, "HIP initialized short_tid#%d (maps to full_tid: 0x%s)\n", _shortTid, tid_ss.str().c_str());
    };
}

//=================================================================================================
// ihipStream_t:
//=================================================================================================
//---
ihipStream_t::ihipStream_t(ihipCtx_t *ctx, hc::accelerator_view av, unsigned int flags) :
    _id(0), // will be set by add function.
    _flags(flags),
    _ctx(ctx),
    _criticalData(this, av)
{
    unsigned schedBits = ctx->_ctxFlags & hipDeviceScheduleMask;

    switch (schedBits) {
        case hipDeviceScheduleAuto          : _scheduleMode = Auto; break;
        case hipDeviceScheduleSpin          : _scheduleMode = Spin; break;
        case hipDeviceScheduleYield         : _scheduleMode = Yield; break;
        case hipDeviceScheduleBlockingSync  : _scheduleMode = Yield; break;
        default:_scheduleMode = Auto;
    };
};


//---
ihipStream_t::~ihipStream_t()
{
}


hc::hcWaitMode ihipStream_t::waitMode() const
{
    hc::hcWaitMode waitMode = hc::hcWaitModeActive;

    if (_scheduleMode == Auto) {
        if (g_deviceCnt > g_numLogicalThreads) {
            waitMode = hc::hcWaitModeActive;
        } else {
            waitMode = hc::hcWaitModeBlocked;
        }
    } else if (_scheduleMode == Spin) {
        waitMode = hc::hcWaitModeActive;
    } else if (_scheduleMode == Yield) {
        waitMode = hc::hcWaitModeBlocked;
    } else {
        assert(0); // bad wait mode.
    }

    if (HIP_WAIT_MODE == 1) {
        waitMode = hc::hcWaitModeBlocked;
    } else if (HIP_WAIT_MODE == 2) {
        waitMode = hc::hcWaitModeActive;
    }

    return waitMode;
}

//Wait for all kernel and data copy commands in this stream to complete.
//This signature should be used in routines that already have locked the stream mutex
void ihipStream_t::wait(LockedAccessor_StreamCrit_t &crit)
{
    tprintf (DB_SYNC, "%s wait for queue-empty..\n", ToString(this).c_str());

    crit->_av.wait(waitMode());

    crit->_kernelCnt = 0;
}

//---
//Wait for all kernel and data copy commands in this stream to complete.
void ihipStream_t::locked_wait()
{
    LockedAccessor_StreamCrit_t crit(_criticalData);

    wait(crit);

};

// Causes current stream to wait for specified event to complete:
// Note this does not provide any kind of host serialization.
void ihipStream_t::locked_streamWaitEvent(hipEvent_t event)
{
    LockedAccessor_StreamCrit_t crit(_criticalData);


    crit->_av.create_blocking_marker(event->marker(), hc::accelerator_scope);
}


// Causes current stream to wait for specified event to complete:
// Note this does not provide any kind of host serialization.
bool ihipStream_t::locked_eventIsReady(hipEvent_t event)
{
    // Event query that returns "Complete" may cause HCC to manipulate
    // internal queue state so lock the stream's queue here.
    LockedAccessor_StreamCrit_t crit(_criticalData);

    return (event->marker().is_ready());
}

void ihipStream_t::locked_eventWaitComplete(hipEvent_t event, hc::hcWaitMode waitMode)
{
    LockedAccessor_StreamCrit_t crit(_criticalData);

    event->marker().wait(waitMode);
}


// Create a marker in this stream.
// Save state in the event so it can track the status of the event.
void ihipStream_t::locked_recordEvent(hipEvent_t event)
{
    // Lock the stream to prevent simultaneous access
    LockedAccessor_StreamCrit_t crit(_criticalData);

    auto scopeFlag = hc::accelerator_scope;
    // The env var HIP_EVENT_SYS_RELEASE sets the default,
    // The explicit flags override the env var (if specified)
    if (event->_flags & hipEventReleaseToSystem) {
        scopeFlag = hc::system_scope;
    } else if (event->_flags & hipEventReleaseToDevice) {
        scopeFlag = hc::accelerator_scope;
    } else {
        scopeFlag = HIP_EVENT_SYS_RELEASE ? hc::system_scope : hc::accelerator_scope;
    }

    event->marker(crit->_av.create_marker(scopeFlag));
};

//=============================================================================



//-------------------------------------------------------------------------------------------------


//---
const ihipDevice_t * ihipStream_t::getDevice() const
{
    return _ctx->getDevice();
};



ihipCtx_t * ihipStream_t::getCtx() const
{
    return _ctx;
};




//--
// Lock the stream to prevent other threads from intervening.
LockedAccessor_StreamCrit_t ihipStream_t::lockopen_preKernelCommand()
{

    LockedAccessor_StreamCrit_t crit(_criticalData, false/*no unlock at destruction*/);


    return crit;
}


//---
// Must be called after kernel finishes, this releases the lock on the stream so other commands can submit.
void ihipStream_t::lockclose_postKernelCommand(const char * kernelName, hc::accelerator_view *av)
{
    bool blockThisKernel = false;

    if (!g_hipLaunchBlockingKernels.empty()) {
        std::string kernelNameString(kernelName);
        for (auto o=g_hipLaunchBlockingKernels.begin(); o!=g_hipLaunchBlockingKernels.end(); o++) {
            if ((*o == kernelNameString)) {
                //printf ("force blocking for kernel %s\n", o->c_str());
                blockThisKernel = true;
            }
        }
    }

    if (HIP_LAUNCH_BLOCKING || blockThisKernel) {
        // TODO - fix this so it goes through proper stream::wait() call.// direct wait OK since we know the stream is locked.
        av->wait(hc::hcWaitModeActive);
        tprintf(DB_SYNC, "%s LAUNCH_BLOCKING for kernel '%s' completion\n", ToString(this).c_str(), kernelName);
    }

    _criticalData.unlock(); // paired with lock from lockopen_preKernelCommand.
};



//=============================================================================
// Recompute the peercnt and the packed _peerAgents whenever a peer is added or deleted.
// The packed _peerAgents can efficiently be used on each memory allocation.
template<>
void ihipCtxCriticalBase_t<CtxMutex>::recomputePeerAgents()
{
    _peerCnt = 0;
    std::for_each (_peers.begin(), _peers.end(), [this](ihipCtx_t* ctx) {
        _peerAgents[_peerCnt++] = ctx->getDevice()->_hsaAgent;
    });
}


template<>
bool ihipCtxCriticalBase_t<CtxMutex>::isPeerWatcher(const ihipCtx_t *peer)
{
    auto match = std::find(_peers.begin(), _peers.end(), peer);
    return (match != std::end(_peers));
}


template<>
bool ihipCtxCriticalBase_t<CtxMutex>::addPeerWatcher(const ihipCtx_t *thisCtx, ihipCtx_t *peerWatcher)
{
    auto match = std::find(_peers.begin(), _peers.end(), peerWatcher);
    if (match == std::end(_peers)) {
        // Not already a peer, let's update the list:
        tprintf(DB_COPY, "addPeerWatcher.  Allocations on %s now visible to peerWatcher %s.\n",
                          thisCtx->toString().c_str(), peerWatcher->toString().c_str());
        _peers.push_back(peerWatcher);
        recomputePeerAgents();
        return true;
    }

    // If we get here - peer was already on list, silently ignore.
    return false;
}


template<>
bool ihipCtxCriticalBase_t<CtxMutex>::removePeerWatcher(const ihipCtx_t *thisCtx, ihipCtx_t *peerWatcher)
{
    auto match = std::find(_peers.begin(), _peers.end(), peerWatcher);
    if (match != std::end(_peers)) {
        // Found a valid peer, let's remove it.
        tprintf(DB_COPY, "removePeerWatcher.  Allocations on %s no longer visible to former peerWatcher %s.\n",
                          thisCtx->toString().c_str(), peerWatcher->toString().c_str());
        _peers.remove(peerWatcher);
        recomputePeerAgents();
        return true;
    } else {
        return false;
    }
}


template<>
void ihipCtxCriticalBase_t<CtxMutex>::resetPeerWatchers(ihipCtx_t *thisCtx)
{
    tprintf(DB_COPY, "resetPeerWatchers for context=%s\n", thisCtx->toString().c_str());
    _peers.clear();
    _peerCnt = 0;
    addPeerWatcher(thisCtx, thisCtx); // peer-list always contains self agent.
}


template<>
void ihipCtxCriticalBase_t<CtxMutex>::printPeerWatchers(FILE *f) const
{
    for (auto iter = _peers.begin(); iter!=_peers.end(); iter++) {
        fprintf (f, "%s ", (*iter)->toString().c_str());
    };
}





template<>
void ihipCtxCriticalBase_t<CtxMutex>::addStream(ihipStream_t *stream)
{
    stream->_id = _streams.size();
    _streams.push_back(stream);
    tprintf(DB_SYNC, " addStream: %s\n", ToString(stream).c_str());
}

template<>
void ihipDeviceCriticalBase_t<DeviceMutex>::addContext(ihipCtx_t *ctx)
{
    _ctxs.push_back(ctx);
    tprintf(DB_SYNC, " addContext: %s\n", ToString(ctx).c_str());
}

//=============================================================================

//=================================================================================================
// ihipDevice_t
//=================================================================================================
ihipDevice_t::ihipDevice_t(unsigned deviceId, unsigned deviceCnt, hc::accelerator &acc) :
    _deviceId(deviceId),
    _acc(acc),
    _state(0),
    _criticalData(this)
{
    hsa_agent_t *agent = static_cast<hsa_agent_t*> (acc.get_hsa_agent());
    if (agent) {
        int err = hsa_agent_get_info(*agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &_computeUnits);
        if (err != HSA_STATUS_SUCCESS) {
            _computeUnits = 1;
        }

        _hsaAgent = *agent;
    } else {
        _hsaAgent.handle = static_cast<uint64_t> (-1);
    }

    initProperties(&_props);


    _primaryCtx = new ihipCtx_t(this, deviceCnt, hipDeviceMapHost);
}


ihipDevice_t::~ihipDevice_t()
{
    delete _primaryCtx;
    _primaryCtx = NULL;
}

void ihipDevice_t::locked_removeContext(ihipCtx_t *c)
{
    LockedAccessor_DeviceCrit_t  crit(_criticalData);

    crit->ctxs().remove(c);
    tprintf(DB_SYNC, " locked_removeContext: %s\n", ToString(c).c_str());
}


void ihipDevice_t::locked_reset()
{
    // Obtain mutex access to the device critical data, release by destructor
    LockedAccessor_DeviceCrit_t  crit(_criticalData);


    //---
    //Wait for pending activity to complete?  TODO - check if this is required behavior:
    tprintf(DB_SYNC, "locked_reset waiting for activity to complete.\n");

    // Reset and remove streams:
    // Delete all created streams including the default one.
    for (auto ctxI=crit->const_ctxs().begin(); ctxI!=crit->const_ctxs().end(); ctxI++) {
        ihipCtx_t *ctx = *ctxI;
        (*ctxI)->locked_reset();
        tprintf(DB_SYNC, " ctx cleanup %s\n", ToString(ctx).c_str());

        delete ctx;
    }
    // Clear the list.
    crit->ctxs().clear();


    //reset _primaryCtx
    _primaryCtx->locked_reset();
    tprintf(DB_SYNC, " _primaryCtx cleanup %s\n", ToString(_primaryCtx).c_str());
    // Reset and release all memory stored in the tracker:
    // Reset will remove peer mapping so don't need to do this explicitly.
    // FIXME - This is clearly a non-const action!  Is this a context reset or a device reset - maybe should reference count?

    _state = 0;
    am_memtracker_reset(_acc);

};

#define ErrorCheck(x) error_check(x, __LINE__, __FILE__)

void error_check(hsa_status_t hsa_error_code, int line_num, std::string str) {
  if ((hsa_error_code != HSA_STATUS_SUCCESS)&& (hsa_error_code != HSA_STATUS_INFO_BREAK))  {
    printf("HSA reported error!\n In file: %s\nAt line: %d\n", str.c_str(),line_num);
  }
}



//---
// Helper for initProperties
// Determines if the given agent is of type HSA_DEVICE_TYPE_GPU and counts it.
static hsa_status_t countGpuAgents(hsa_agent_t agent, void *data) {
    if (data == NULL) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    hsa_device_type_t device_type;
    hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    if (status != HSA_STATUS_SUCCESS) {
        return status;
    }
    if (device_type == HSA_DEVICE_TYPE_GPU) {
        (*static_cast<int*>(data))++;
    }
    return HSA_STATUS_SUCCESS;
}




hsa_status_t FindGpuDevice(hsa_agent_t agent, void* data) {
    if (data == NULL) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    hsa_device_type_t hsa_device_type;
    hsa_status_t hsa_error_code =
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type);
    if (hsa_error_code != HSA_STATUS_SUCCESS) {
        return hsa_error_code;
    }

    if (hsa_device_type == HSA_DEVICE_TYPE_GPU) {
        *((hsa_agent_t*)data) = agent;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t GetDevicePool(hsa_amd_memory_pool_t pool, void* data) {
    if (NULL == data) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    hsa_status_t err;
    hsa_amd_segment_t segment;
    uint32_t flag;

    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    ErrorCheck(err);
    if (HSA_AMD_SEGMENT_GLOBAL != segment) return HSA_STATUS_SUCCESS;
    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
    ErrorCheck(err);
    *((hsa_amd_memory_pool_t*)data) = pool;
    return HSA_STATUS_SUCCESS;
}


int checkAccess(hsa_agent_t agent, hsa_amd_memory_pool_t pool)
{
    hsa_status_t err;
    hsa_amd_memory_pool_access_t access;
    err = hsa_amd_agent_memory_pool_get_info(agent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
    ErrorCheck(err);
    return access;
}

hsa_status_t get_region_info(hsa_region_t region, void* data)
{
    hsa_status_t err;
    hipDeviceProp_t* p_prop = reinterpret_cast<hipDeviceProp_t*>(data);
    uint32_t region_segment;

    // Get region segment
    err = hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &region_segment);
    ErrorCheck(err);

    switch(region_segment) {
    case HSA_REGION_SEGMENT_READONLY:
        err = hsa_region_get_info(region, HSA_REGION_INFO_SIZE, &(p_prop->totalConstMem)); break;
    /* case HSA_REGION_SEGMENT_PRIVATE:
        cout<<"PRIVATE"<<endl; private segment cannot be queried */
    case HSA_REGION_SEGMENT_GROUP:
        err = hsa_region_get_info(region, HSA_REGION_INFO_SIZE, &(p_prop->sharedMemPerBlock)); break;
    default: break;
    }
    return HSA_STATUS_SUCCESS;
}


// Determines if the given agent is of type HSA_DEVICE_TYPE_GPU and counts it.
static hsa_status_t findCpuAgent(hsa_agent_t agent, void *data)
{
    hsa_device_type_t device_type;
    hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    if (status != HSA_STATUS_SUCCESS) {
        return status;
    }
    if (device_type == HSA_DEVICE_TYPE_CPU) {
        (*static_cast<hsa_agent_t*>(data)) = agent;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}


#define DeviceErrorCheck(x) if (x != HSA_STATUS_SUCCESS) { return hipErrorInvalidDevice; }

//---
// Initialize properties for the device.
// Call this once when the ihipDevice_t is created:
hipError_t ihipDevice_t::initProperties(hipDeviceProp_t* prop)
{
    hipError_t e = hipSuccess;
    hsa_status_t err;

    // Set some defaults in case we don't find the appropriate regions:
    prop->totalGlobalMem = 0;
    prop->totalConstMem = 0;
    prop-> maxThreadsPerMultiProcessor = 0;
    prop->regsPerBlock = 0;

    if (_hsaAgent.handle == -1) {
        return hipErrorInvalidDevice;
    }

    // Iterates over the agents to determine Multiple GPU devices
    // using the countGpuAgents callback.
    //! @bug : on HCC, isMultiGpuBoard returns True if system contains multiple GPUS (rather than if GPU is on a multi-ASIC board)
    int gpuAgentsCount = 0;
    err = hsa_iterate_agents(countGpuAgents, &gpuAgentsCount);
    if (err == HSA_STATUS_INFO_BREAK) { err = HSA_STATUS_SUCCESS; }
    DeviceErrorCheck(err);
    prop->isMultiGpuBoard = 0 ? gpuAgentsCount < 2 : 1;

    // Get agent name

    err = hsa_agent_get_info(_hsaAgent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME, &(prop->name));
    char archName[256];
    err = hsa_agent_get_info(_hsaAgent, HSA_AGENT_INFO_NAME, &archName);

    prop->gcnArch = atoi(archName+3);

    DeviceErrorCheck(err);

    // Get agent node
    uint32_t node;
    err = hsa_agent_get_info(_hsaAgent, HSA_AGENT_INFO_NODE, &node);
    DeviceErrorCheck(err);

    // Get wavefront size
    err = hsa_agent_get_info(_hsaAgent, HSA_AGENT_INFO_WAVEFRONT_SIZE,&prop->warpSize);
    DeviceErrorCheck(err);

    // Get max total number of work-items in a workgroup
    err = hsa_agent_get_info(_hsaAgent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &prop->maxThreadsPerBlock );
    DeviceErrorCheck(err);

    // Get max number of work-items of each dimension of a work-group
    uint16_t work_group_max_dim[3];
    err = hsa_agent_get_info(_hsaAgent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, work_group_max_dim);
    DeviceErrorCheck(err);
    for( int i =0; i< 3 ; i++) {
        prop->maxThreadsDim[i]= work_group_max_dim[i];
    }

    hsa_dim3_t grid_max_dim;
    err = hsa_agent_get_info(_hsaAgent, HSA_AGENT_INFO_GRID_MAX_DIM, &grid_max_dim);
    DeviceErrorCheck(err);
    prop->maxGridSize[0]= (int) ((grid_max_dim.x == UINT32_MAX) ? (INT32_MAX) : grid_max_dim.x);
    prop->maxGridSize[1]= (int) ((grid_max_dim.y == UINT32_MAX) ? (INT32_MAX) : grid_max_dim.y);
    prop->maxGridSize[2]= (int) ((grid_max_dim.z == UINT32_MAX) ? (INT32_MAX) : grid_max_dim.z);

    // Get Max clock frequency
    err = hsa_agent_get_info(_hsaAgent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY, &prop->clockRate);
    prop->clockRate *= 1000.0;   // convert Mhz to Khz.
    DeviceErrorCheck(err);

    uint64_t counterHz;
    err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &counterHz);
    DeviceErrorCheck(err);
    prop->clockInstructionRate = counterHz / 1000;

    // Get Agent BDFID (bus/device/function ID)
    uint16_t bdf_id = 1;
    err = hsa_agent_get_info(_hsaAgent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf_id);
    DeviceErrorCheck(err);

    // BDFID is 16bit uint: [8bit - BusID | 5bit - Device ID | 3bit - Function/DomainID]
    prop->pciDomainID =  bdf_id & 0x7;
    prop->pciDeviceID =  (bdf_id>>3) & 0x1F;
    prop->pciBusID =  (bdf_id>>8) & 0xFF;

    // Masquerade as a 3.0-level device. This will change as more HW functions are properly supported.
    // Application code should use the arch.has* to do detailed feature detection.
    prop->major = 3;
    prop->minor = 0;

    // Get number of Compute Unit
    err = hsa_agent_get_info(_hsaAgent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &(prop->multiProcessorCount));
    DeviceErrorCheck(err);

    // TODO-hsart - this appears to return 0?
    uint32_t cache_size[4];
    err = hsa_agent_get_info(_hsaAgent, HSA_AGENT_INFO_CACHE_SIZE, cache_size);
    DeviceErrorCheck(err);
    prop->l2CacheSize = cache_size[1];

    /* Computemode for HSA Devices is always : cudaComputeModeDefault */
    prop->computeMode = 0;

    _isLargeBar = _acc.has_cpu_accessible_am();

    // Get Max Threads Per Multiprocessor
    uint32_t max_waves_per_cu;
    err = hsa_agent_get_info(_hsaAgent,(hsa_agent_info_t) HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU, &max_waves_per_cu);
    DeviceErrorCheck(err);
    prop-> maxThreadsPerMultiProcessor = prop->warpSize*max_waves_per_cu;

    // Get memory properties
    err = hsa_agent_iterate_regions(_hsaAgent, get_region_info, prop);
    DeviceErrorCheck(err);

    // Get the size of the region we are using for Accelerator Memory allocations:
    hsa_region_t *am_region = static_cast<hsa_region_t*>(_acc.get_hsa_am_region());
    err = hsa_region_get_info(*am_region, HSA_REGION_INFO_SIZE, &prop->totalGlobalMem);
    DeviceErrorCheck(err);
    // maxSharedMemoryPerMultiProcessor should be as the same as group memory size.
    // Group memory will not be paged out, so, the physical memory size is the total shared memory size, and also equal to the group region size.
    prop->maxSharedMemoryPerMultiProcessor = prop->totalGlobalMem;

    // Get Max memory clock frequency
    err = hsa_region_get_info(*am_region, (hsa_region_info_t)HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY, &prop->memoryClockRate);
    DeviceErrorCheck(err);
    prop->memoryClockRate *= 1000.0;   // convert Mhz to Khz.

    // Get global memory bus width in bits
    err = hsa_region_get_info(*am_region, (hsa_region_info_t)HSA_AMD_REGION_INFO_BUS_WIDTH, &prop->memoryBusWidth);
    DeviceErrorCheck(err);

    // Set feature flags - these are all mandatory for HIP on HCC path:
    // Some features are under-development and future revs may support flags that are currently 0.
    // Reporting of these flags should be synchronized with the HIP_ARCH* compile-time defines in hip_runtime.h

    prop->arch.hasGlobalInt32Atomics       = 1;
    prop->arch.hasGlobalFloatAtomicExch    = 1;
    prop->arch.hasSharedInt32Atomics       = 1;
    prop->arch.hasSharedFloatAtomicExch    = 1;
    prop->arch.hasFloatAtomicAdd           = 0;
    prop->arch.hasGlobalInt64Atomics       = 1;
    prop->arch.hasSharedInt64Atomics       = 1;
    prop->arch.hasDoubles                  = 1;
    prop->arch.hasWarpVote                 = 1;
    prop->arch.hasWarpBallot               = 1;
    prop->arch.hasWarpShuffle              = 1;
    prop->arch.hasFunnelShift              = 0; // TODO-hcc
    prop->arch.hasThreadFenceSystem        = 0; // TODO-hcc
    prop->arch.hasSyncThreadsExt           = 0; // TODO-hcc
    prop->arch.hasSurfaceFuncs             = 0; // TODO-hcc
    prop->arch.has3dGrid                   = 1;
    prop->arch.hasDynamicParallelism       = 0;

    prop->concurrentKernels = 1; // All ROCm hardware supports executing multiple kernels concurrently

    prop->canMapHostMemory = 1;  // All ROCm devices can map host memory
    prop->totalConstMem = 16384;
#if 0
    // TODO - code broken below since it always returns 1.
    // Are the flags part of the context or part of the device?
    if ( _device_flags | hipDeviceMapHost) {
        prop->canMapHostMemory = 1;
    } else {
        prop->canMapHostMemory = 0;
    }
#endif
    return e;
}


//=================================================================================================
// ihipCtx_t
//=================================================================================================
ihipCtx_t::ihipCtx_t(ihipDevice_t *device, unsigned deviceCnt, unsigned flags) :
    _ctxFlags(flags),
    _device(device),
    _criticalData(this, deviceCnt)
{
    //locked_reset();
    LockedAccessor_CtxCrit_t  crit(_criticalData);
	_defaultStream = new ihipStream_t(this, getDevice()->_acc.get_default_view(), hipStreamDefault);
	crit->addStream(_defaultStream);


		// Reset peer list to just me:
    crit->resetPeerWatchers(this);
    tprintf(DB_SYNC, "created ctx with defaultStream=%p (%s)\n", _defaultStream, ToString(_defaultStream).c_str());
};




ihipCtx_t::~ihipCtx_t()
{
    if (_defaultStream) {
        delete _defaultStream;
        _defaultStream = NULL;
    }
}
//Reset the device - this is called from hipDeviceReset.
//Device may be reset multiple times, and may be reset after init.
void ihipCtx_t::locked_reset()
{
    // Obtain mutex access to the device critical data, release by destructor
    LockedAccessor_CtxCrit_t  crit(_criticalData);


    //---
    //Wait for pending activity to complete?  TODO - check if this is required behavior:
    tprintf(DB_SYNC, "locked_reset waiting for activity to complete.\n");

    // Reset and remove streams:
    // Delete all created streams including the default one.
    for (auto streamI=crit->const_streams().begin(); streamI!=crit->const_streams().end(); streamI++) {
        ihipStream_t *stream = *streamI;
        (*streamI)->locked_wait();
        tprintf(DB_SYNC, " delete %s\n", ToString(stream).c_str());

        delete stream;
    }
    // Clear the list.
    crit->streams().clear();


    // Create a fresh default stream and add it:
    _defaultStream = new ihipStream_t(this, getDevice()->_acc.get_default_view(), hipStreamDefault);
    crit->addStream(_defaultStream);

#if 0
    // Reset peer list to just me:
    crit->resetPeerWatchers(this);

    // Reset and release all memory stored in the tracker:
    // Reset will remove peer mapping so don't need to do this explicitly.
    // FIXME - This is clearly a non-const action!  Is this a context reset or a device reset - maybe should reference count?
    ihipDevice_t *device = getWriteableDevice();
    device->_state = 0;
    am_memtracker_reset(device->_acc);
#endif
};


//---
std::string ihipCtx_t::toString() const
{
  std::ostringstream ss;
  ss << this;
  return ss.str();
};


//----




//=================================================================================================
// Utility functions, these are not part of the public HIP API
//=================================================================================================

//=================================================================================================






//   This called for submissions that are sent to the null/default stream.  This routine ensures
//   that this new command waits for activity in the other streams to complete before proceeding.
//
//   HIP_SYNC_NULL_STREAM=0 does all dependency resolutiokn on the GPU
//   HIP_SYNC_NULL_STREAM=1 s legacy non-optimal mode which conservatively waits on host.
//
//   If waitOnSelf is set, this additionally waits for the default stream to empty.
//   In new HIP_SYNC_NULL_STREAM=0 mode, this enqueues a marker which causes the default stream to wait for other
//   activity, but doesn't actually block the host.  If host blocking is desired, the caller should set syncHost.
//
//   syncToHost causes host to wait for the stream to finish.
//   Note HIP_SYNC_NULL_STREAM=1 path always sync to Host. 
void ihipCtx_t::locked_syncDefaultStream(bool waitOnSelf, bool syncHost)
{
    LockedAccessor_CtxCrit_t  crit(_criticalData);

    tprintf(DB_SYNC, "syncDefaultStream \n");

    // Vector of ops sent to each stream that will complete before ops sent to null stream:
    std::vector<hc::completion_future> depOps;

    for (auto streamI=crit->const_streams().begin(); streamI!=crit->const_streams().end(); streamI++) {
        ihipStream_t *stream = *streamI;

        // Don't wait for streams that have "opted-out" of syncing with NULL stream.
        // And - don't wait for the NULL stream, unless waitOnSelf specified.
        bool waitThisStream = (!(stream->_flags & hipStreamNonBlocking)) && 
                              (waitOnSelf || (stream != _defaultStream));

        if (HIP_SYNC_NULL_STREAM) {

            if (waitThisStream) {
                stream->locked_wait();
            }
        } else {
            if (waitThisStream) {
                LockedAccessor_StreamCrit_t streamCrit(stream->_criticalData);

                // The last marker will provide appropriate visibility:
                if (!streamCrit->_av.get_is_empty()) {
                    depOps.push_back(streamCrit->_av.create_marker(hc::accelerator_scope));
                    tprintf(DB_SYNC, "  push marker to wait for stream=%s\n", ToString(stream).c_str());
                } else {
                    tprintf(DB_SYNC, "  skipped stream=%s since it is empty\n", ToString(stream).c_str());
                }
            }
        }
    }

    
    // Enqueue a barrier to wait on all the barriers we sent above:
    if (!HIP_SYNC_NULL_STREAM && !depOps.empty()) {
        LockedAccessor_StreamCrit_t defaultStreamCrit(_defaultStream->_criticalData);
        tprintf(DB_SYNC, "  null-stream wait on %zu non-empty streams. sync_host=%d\n", depOps.size(), syncHost);
        hc::completion_future defaultCf = defaultStreamCrit->_av.create_blocking_marker(depOps.begin(), depOps.end(), hc::accelerator_scope);
        if (syncHost) {
            defaultCf.wait(); // TODO - account for active or blocking here.
        }
    }

    tprintf(DB_SYNC, "  syncDefaultStream depOps=%zu\n", depOps.size());

}


//---
void ihipCtx_t::locked_removeStream(ihipStream_t *s)
{
    LockedAccessor_CtxCrit_t  crit(_criticalData);

    crit->streams().remove(s);
}


//---
//Heavyweight synchronization that waits on all streams, ignoring hipStreamNonBlocking flag.
void ihipCtx_t::locked_waitAllStreams()
{
    LockedAccessor_CtxCrit_t  crit(_criticalData);

    tprintf(DB_SYNC, "waitAllStream\n");
    for (auto streamI=crit->const_streams().begin(); streamI!=crit->const_streams().end(); streamI++) {
        (*streamI)->locked_wait();
    }
}


std::string HIP_DB_string(unsigned db)
{
    std::string dbStr;
    bool first=true;
    for (int i=0; i<DB_MAX_FLAG; i++) {
        if (db & (1<<i)) {
            if (!first) {
                dbStr += "+";
            };
            dbStr += dbName[i]._color;
            dbStr += dbName[i]._shortName;
            dbStr += KNRM;
            first=false;
        };
    }

    return dbStr;
}

// Callback used to process HIP_DB input, supports either
// integer or flag names separated by +
std::string HIP_DB_callback(void *var_ptr, const char *envVarString)
{
    int * var_ptr_int = static_cast<int *> (var_ptr);

    std::string e(envVarString);
    trim(&e);
    if (!e.empty() && isdigit(e.c_str()[0])) {
        long int v = strtol(envVarString, NULL, 0);
        *var_ptr_int = (int) (v);
    } else {
        *var_ptr_int = 0;
        std::vector<std::string> tokens;
        tokenize(e, '+', &tokens);
        for (auto t=tokens.begin(); t!= tokens.end(); t++) {
            for (int i=0; i<DB_MAX_FLAG; i++) {
                if (!strcmp(t->c_str(), dbName[i]._shortName)) {
                    *var_ptr_int |= (1<<i);
                } // TODO - else throw error?
            }
        }
    }

    return HIP_DB_string(*var_ptr_int);;
}


// Callback used to process list of visible devices.
std::string HIP_VISIBLE_DEVICES_callback(void *var_ptr, const char *envVarString)
{
    // Parse the string stream of env and store the device ids to g_hip_visible_devices global variable
    std::string str = envVarString;
    std::istringstream ss(str);
    std::string device_id;
    // Clean up the defult value
    g_hip_visible_devices.clear();
    g_visible_device = true;
    // Read the visible device numbers
    while (std::getline(ss, device_id, ',')) {
        if (atoi(device_id.c_str()) >= 0) {
            g_hip_visible_devices.push_back(atoi(device_id.c_str()));
        } else { // Any device number after invalid number will not present
            break;
        }
    }

    std::string valueString;
    // Print out the number of ids
    for(int i=0;i<g_hip_visible_devices.size();i++) {
        valueString += std::to_string((g_hip_visible_devices[i]));
        valueString += ' ';
    }

    return valueString;
}


// TODO - change last arg to pointer.
void parseTrigger(std::string triggerString, std::vector<ProfTrigger> &profTriggers )
{
    std::vector<std::string> tidApiTokens;
    tokenize(std::string(triggerString), ',', &tidApiTokens);
    for (auto t=tidApiTokens.begin(); t != tidApiTokens.end(); t++) {
        std::vector<std::string> oneToken;
        //std::cout << "token=" << *t << "\n";
        tokenize(std::string(*t), '.', &oneToken);
        int tid = 1;
        uint64_t apiTrigger = 0;
        if (oneToken.size() == 1) {
            // the case with just apiNum
            apiTrigger = std::strtoull(oneToken[0].c_str(), nullptr, 0);
        } else if (oneToken.size() == 2) {
            // the case with tid.apiNum
            tid = std::strtoul(oneToken[0].c_str(), nullptr, 0);
            apiTrigger = std::strtoull(oneToken[1].c_str(), nullptr, 0);
        } else {
            throw ihipException(hipErrorRuntimeOther); // TODO -> bad env var?
        }

        if (tid > 10000) {
            throw ihipException(hipErrorRuntimeOther); // TODO -> bad env var?
        } else {
            profTriggers.resize(tid+1);
            //std::cout << "tid:" << tid << " add: " << apiTrigger << "\n";
            profTriggers[tid].add(apiTrigger);
        }
    }


    for (int tid=1; tid<profTriggers.size(); tid++) {
        profTriggers[tid].sort();
        profTriggers[tid].print(tid);
    }
}


void HipReadEnv()
{
    /*
     * Environment variables
     */
    g_hip_visible_devices.push_back(0); /* Set the default value of visible devices */
    READ_ENV_I(release, HIP_PRINT_ENV, 0,  "Print HIP environment variables.");
    //-- READ HIP_PRINT_ENV env first, since it has impact on later env var reading

    // TODO: In HIP/hcc, this variable blocks after both kernel commmands and data transfer.  Maybe should be bit-mask for each command type?
    READ_ENV_I(release, HIP_LAUNCH_BLOCKING, CUDA_LAUNCH_BLOCKING, "Make HIP kernel launches 'host-synchronous', so they block until any kernel launches. Alias: CUDA_LAUNCH_BLOCKING." );
    READ_ENV_S(release, HIP_LAUNCH_BLOCKING_KERNELS, 0, "Comma-separated list of kernel names to make host-synchronous, so they block until completed.");
    if (!HIP_LAUNCH_BLOCKING_KERNELS.empty()) {
        tokenize(HIP_LAUNCH_BLOCKING_KERNELS, ',', &g_hipLaunchBlockingKernels);
    }
    READ_ENV_I(release, HIP_API_BLOCKING, 0, "Make HIP APIs 'host-synchronous', so they block until completed.  Impacts hipMemcpyAsync, hipMemsetAsync." );
    
    READ_ENV_I(release, HIP_HIDDEN_FREE_MEM, 0, "Amount of memory to hide from the free memory reported by hipMemGetInfo, specified in MB. Impacts hipMemGetInfo." );

    READ_ENV_C(release, HIP_DB, 0,  "Print debug info.  Bitmask (HIP_DB=0xff) or flags separated by '+' (HIP_DB=api+sync+mem+copy)", HIP_DB_callback);
    if ((HIP_DB & (1<<DB_API))  && (HIP_TRACE_API == 0)) {
        // Set HIP_TRACE_API default before we read it, so it is printed correctly.
        HIP_TRACE_API = 1;
    }





    READ_ENV_I(release, HIP_TRACE_API, 0,  "Trace each HIP API call.  Print function name and return code to stderr as program executes.");
    READ_ENV_S(release, HIP_TRACE_API_COLOR, 0,  "Color to use for HIP_API.  None/Red/Green/Yellow/Blue/Magenta/Cyan/White");
    READ_ENV_I(release, HIP_PROFILE_API, 0,  "Add HIP API markers to ATP file generated with CodeXL. 0x1=short API name, 0x2=full API name including args.");
    READ_ENV_S(release, HIP_DB_START_API, 0,  "Comma-separated list of tid.api_seq_num for when to start debug and profiling.");
    READ_ENV_S(release, HIP_DB_STOP_API, 0,  "Comma-separated list of tid.api_seq_num for when to stop debug and profiling.");

    READ_ENV_C(release, HIP_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES, "Only devices whose index is present in the sequence are visible to HIP applications and they are enumerated in the order of sequence.", HIP_VISIBLE_DEVICES_callback );


    READ_ENV_I(release, HIP_WAIT_MODE, 0, "Force synchronization mode. 1= force yield, 2=force spin, 0=defaults specified in application");
    READ_ENV_I(release, HIP_FORCE_P2P_HOST, 0, "Force use of host/staging copy for peer-to-peer copies.1=always use copies, 2=always return false for hipDeviceCanAccessPeer");
    READ_ENV_I(release, HIP_FORCE_SYNC_COPY, 0, "Force all copies (even hipMemcpyAsync) to use sync copies");
    READ_ENV_I(release, HIP_FAIL_SOC, 0, "Fault on Sub-Optimal-Copy, rather than use a slower but functional implementation.  Bit 0x1=Fail on async copy with unpinned memory.  Bit 0x2=Fail peer copy rather than use staging buffer copy");

    READ_ENV_I(release, HIP_SYNC_HOST_ALLOC, 0, "Sync before and after all host memory allocations.  May help stability");
    READ_ENV_I(release, HIP_INIT_ALLOC, 0, "If not -1, initialize allocated memory to specified byte");
    READ_ENV_I(release, HIP_SYNC_NULL_STREAM, 0, "Synchronize on host for null stream submissions");
    READ_ENV_I(release, HIP_FORCE_NULL_STREAM,  0, "Force all stream allocations to secretly return the null stream");

    READ_ENV_I(release, HIP_SYNC_STREAM_WAIT, 0, "hipStreamWaitEvent will synchronize to host");


    READ_ENV_I(release, HIP_HOST_COHERENT, 0, "If set, all host memory will be allocated as fine-grained system memory.  This allows threadfence_system to work but prevents host memory from being cached on GPU which may have performance impact.");


    READ_ENV_I(release, HCC_OPT_FLUSH, 0, "When set, use agent-scope fence operations rather than system-scope fence operationsflush when possible. This flag controls both HIP and HCC behavior.");
    READ_ENV_I(release, HIP_EVENT_SYS_RELEASE, 0, "If set, event are created with hipEventReleaseToSystem by default.  If 0, events are created with hipEventReleaseToDevice by default.  The defaults can be overridden by specifying hipEventReleaseToSystem or hipEventReleaseToDevice flag when creating the event.");

    // Some flags have both compile-time and runtime flags - generate a warning if user enables the runtime flag but the compile-time flag is disabled.
    if (HIP_DB && !COMPILE_HIP_DB) {
        fprintf (stderr, "warning: env var HIP_DB=0x%x but COMPILE_HIP_DB=0.  (perhaps enable COMPILE_HIP_DB in src code before compiling?)\n", HIP_DB);
    }

    if (HIP_TRACE_API && !COMPILE_HIP_TRACE_API) {
        fprintf (stderr, "warning: env var HIP_TRACE_API=0x%x but COMPILE_HIP_TRACE_API=0.  (perhaps enable COMPILE_HIP_TRACE_API in src code before compiling?)\n", HIP_DB);
    }
    if (HIP_TRACE_API) {
        HIP_DB|=0x1;
    }

    if (HIP_PROFILE_API && !COMPILE_HIP_ATP_MARKER) {
        fprintf (stderr, "warning: env var HIP_PROFILE_API=0x%x but COMPILE_HIP_ATP_MARKER=0.  (perhaps enable COMPILE_HIP_ATP_MARKER in src code before compiling?)\n", HIP_PROFILE_API);
        HIP_PROFILE_API = 0;
    }

    if (HIP_DB) {
        fprintf (stderr, "HIP_DB=0x%x [%s]\n", HIP_DB, HIP_DB_string(HIP_DB).c_str());
    }

    std::transform(HIP_TRACE_API_COLOR.begin(),  HIP_TRACE_API_COLOR.end(), HIP_TRACE_API_COLOR.begin(), ::tolower);

    if (HIP_TRACE_API_COLOR == "none") {
        API_COLOR = "";
        API_COLOR_END = "";
    } else if (HIP_TRACE_API_COLOR == "red") {
        API_COLOR = KRED;
    } else if (HIP_TRACE_API_COLOR == "green") {
        API_COLOR = KGRN;
    } else if (HIP_TRACE_API_COLOR == "yellow") {
        API_COLOR = KYEL;
    } else if (HIP_TRACE_API_COLOR == "blue") {
        API_COLOR = KBLU;
    } else if (HIP_TRACE_API_COLOR == "magenta") {
        API_COLOR = KMAG;
    } else if (HIP_TRACE_API_COLOR == "cyan") {
        API_COLOR = KCYN;
    } else if (HIP_TRACE_API_COLOR == "white") {
        API_COLOR = KWHT;
    } else {
        fprintf (stderr, "warning: env var HIP_TRACE_API_COLOR=%s must be None/Red/Green/Yellow/Blue/Magenta/Cyan/White", HIP_TRACE_API_COLOR.c_str());
    };

    parseTrigger(HIP_DB_START_API, g_dbStartTriggers);
    parseTrigger(HIP_DB_STOP_API,  g_dbStopTriggers);
};



//---
//Function called one-time at initialization time to construct a table of all GPU devices.
//HIP/CUDA uses integer "deviceIds" - these are indexes into this table.
//AMP maintains a table of accelerators, but some are emulated - ie for debug or CPU.
//This function creates a vector with only the GPU accelerators.
//It is called with C++11 call_once, which provided thread-safety.
void ihipInit()
{

#if COMPILE_HIP_ATP_MARKER
    amdtInitializeActivityLogger();
    amdtScopedMarker("ihipInit", "HIP", NULL);
#endif


    HipReadEnv();


    /*
     * Build a table of valid compute devices.
     */
    auto accs = hc::accelerator::get_all();

    int deviceCnt = 0;
    for (int i=0; i<accs.size(); i++) {
        if (! accs[i].get_is_emulated()) {
            deviceCnt++;
        }
    };

    // Make sure the hip visible devices are within the deviceCnt range
    for (int i = 0; i < g_hip_visible_devices.size(); i++) {
        if(g_hip_visible_devices[i] >= deviceCnt){
            // Make sure any DeviceID after invalid DeviceID will be erased.
            g_hip_visible_devices.resize(i);
            break;
        }
    }

    hsa_status_t err = hsa_iterate_agents(findCpuAgent, &g_cpu_agent);
    if (err != HSA_STATUS_INFO_BREAK) {
        // didn't find a CPU.
        throw ihipException(hipErrorRuntimeOther);
    }

    g_deviceArray = new ihipDevice_t*  [deviceCnt];
    g_deviceCnt = 0;
    for (int i=0; i<accs.size(); i++) {
        // check if the device id is included in the HIP_VISIBLE_DEVICES env variable
        if (! accs[i].get_is_emulated()) {
            if (std::find(g_hip_visible_devices.begin(), g_hip_visible_devices.end(), (i-1)) == g_hip_visible_devices.end() && g_visible_device)
            {
                //If device is not in visible devices list, ignore
                continue;
            }
            g_deviceArray[g_deviceCnt] = new ihipDevice_t(g_deviceCnt, deviceCnt, accs[i]);
            g_deviceCnt++;
        }
    }

    g_allAgents = static_cast<hsa_agent_t*> (malloc((g_deviceCnt+1) * sizeof(hsa_agent_t)));
    g_allAgents[0] = g_cpu_agent;
    for (int i=0; i<g_deviceCnt; i++) {
        g_allAgents[i+1] = g_deviceArray[i]->_hsaAgent;
    }


    g_numLogicalThreads = std::thread::hardware_concurrency();

    // If HIP_VISIBLE_DEVICES is not set, make sure all devices are initialized
    if(!g_visible_device) {
        assert(deviceCnt == g_deviceCnt);
    }

    tprintf(DB_SYNC, "pid=%u %-30s g_numLogicalThreads=%u\n", getpid(), "<ihipInit>", g_numLogicalThreads);
}





//---
// Get the stream to use for a command submission.
//
// If stream==NULL synchronize appropriately with other streams and return the default av for the device.
// If stream is valid, return the AV to use.
hipStream_t ihipSyncAndResolveStream(hipStream_t stream)
{
    if (stream == hipStreamNull ) {
        // Submitting to NULL stream, call locked_syncDefaultStream to wait for all other streams:
        ihipCtx_t *ctx = ihipGetTlsDefaultCtx();
        tprintf(DB_SYNC, "ihipSyncAndResolveStream %s wait on default stream\n", ToString(stream).c_str());

#ifndef HIP_API_PER_THREAD_DEFAULT_STREAM
        ctx->locked_syncDefaultStream(false, false);
#endif
        return ctx->_defaultStream;
    } else {
        // Submitting to a "normal" stream, just wait for null stream:
        if (!(stream->_flags & hipStreamNonBlocking))  {
            if (HIP_SYNC_NULL_STREAM) {
                tprintf(DB_SYNC, "ihipSyncAndResolveStream %s host-wait on default stream\n", ToString(stream).c_str());
                stream->getCtx()->_defaultStream->locked_wait();
            } else {
                ihipStream_t *defaultStream = stream->getCtx()->_defaultStream;


                bool needGatherMarker = false; // used to gather together other markers.
                hc::completion_future dcf;
                {
                    LockedAccessor_StreamCrit_t defaultStreamCrit(defaultStream->criticalData());
                    // TODO - could call create_blocking_marker(queue) or uses existing marker.
                    if (!defaultStreamCrit->_av.get_is_empty()) {
                        needGatherMarker = true;

                        tprintf(DB_SYNC, "  %s adding marker to default %s for dependency\n", 
                                ToString(stream).c_str(), ToString(defaultStream).c_str());
                        dcf = defaultStreamCrit->_av.create_marker(hc::accelerator_scope);
                    } else {
                        tprintf(DB_SYNC, "  %s skipping marker since default stream is empty\n", ToString(stream).c_str());
                    }
                }

                if (needGatherMarker) {
                    // ensure any commands sent to this stream wait on the NULL stream before continuing
                    LockedAccessor_StreamCrit_t thisStreamCrit(stream->criticalData());
                    // TODO - could be "noret" version of create_blocking_marker
                    thisStreamCrit->_av.create_blocking_marker(dcf, hc::accelerator_scope);
                    tprintf(DB_SYNC, "  %s adding marker to wait for freshly recorded default-stream marker \n", 
                            ToString(stream).c_str());
                }
            }
        }

        return stream;
    }
}

void ihipPrintKernelLaunch(const char *kernelName, const grid_launch_parm *lp, const hipStream_t stream)
{

    if ((HIP_TRACE_API & (1<<TRACE_KCMD)) || HIP_PROFILE_API || (COMPILE_HIP_DB & HIP_TRACE_API)) {
        std::stringstream os;
        os  << tls_tidInfo.tid() << "." << tls_tidInfo.apiSeqNum()
            << " hipLaunchKernel '" << kernelName << "'"
            << " gridDim:"  << lp->grid_dim
            << " groupDim:" << lp->group_dim
            << " sharedMem:+" << lp->dynamic_group_mem_bytes
            << " " << *stream;

        if (COMPILE_HIP_DB && HIP_TRACE_API) {
            std::string fullStr;
            recordApiTrace(&fullStr, os.str());
        }

        if (HIP_PROFILE_API == 0x1) {
            std::string shortAtpString("hipLaunchKernel:");
            shortAtpString += kernelName;
            MARKER_BEGIN(shortAtpString.c_str(), "HIP");
        } else if (HIP_PROFILE_API == 0x2) {
            MARKER_BEGIN(os.str().c_str(), "HIP");
        }
    }
}

// Called just before a kernel is launched from hipLaunchKernel.
// Allows runtime to track some information about the stream.
hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, dim3 block, grid_launch_parm *lp, const char *kernelNameStr)
{
    HIP_INIT();
    stream = ihipSyncAndResolveStream(stream);
    lp->grid_dim.x = grid.x;
    lp->grid_dim.y = grid.y;
    lp->grid_dim.z = grid.z;
    lp->group_dim.x = block.x;
    lp->group_dim.y = block.y;
    lp->group_dim.z = block.z;
    lp->barrier_bit = barrier_bit_queue_default;
    lp->launch_fence = -1;

    auto crit = stream->lockopen_preKernelCommand();
    lp->av = &(crit->_av);
    lp->cf = nullptr;
    ihipPrintKernelLaunch(kernelNameStr, lp, stream);

    return (stream);
}


hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, dim3 block, grid_launch_parm *lp, const char *kernelNameStr)
{
    HIP_INIT();
    stream = ihipSyncAndResolveStream(stream);
    lp->grid_dim.x = grid;
    lp->grid_dim.y = 1;
    lp->grid_dim.z = 1;
    lp->group_dim.x = block.x;
    lp->group_dim.y = block.y;
    lp->group_dim.z = block.z;
    lp->barrier_bit = barrier_bit_queue_default;
    lp->launch_fence = -1;

    auto crit = stream->lockopen_preKernelCommand();
    lp->av = &(crit->_av);
    lp->cf = nullptr;
    ihipPrintKernelLaunch(kernelNameStr, lp, stream);
    return (stream);
}


hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, size_t block, grid_launch_parm *lp, const char *kernelNameStr)
{
    HIP_INIT();
    stream = ihipSyncAndResolveStream(stream);
    lp->grid_dim.x = grid.x;
    lp->grid_dim.y = grid.y;
    lp->grid_dim.z = grid.z;
    lp->group_dim.x = block;
    lp->group_dim.y = 1;
    lp->group_dim.z = 1;
    lp->barrier_bit = barrier_bit_queue_default;
    lp->launch_fence = -1;

    auto crit = stream->lockopen_preKernelCommand();
    lp->av = &(crit->_av);
    lp->cf = nullptr;
    ihipPrintKernelLaunch(kernelNameStr, lp, stream);
    return (stream);
}


hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, size_t block, grid_launch_parm *lp, const char *kernelNameStr)
{
    HIP_INIT();
    stream = ihipSyncAndResolveStream(stream);
    lp->grid_dim.x = grid;
    lp->grid_dim.y = 1;
    lp->grid_dim.z = 1;
    lp->group_dim.x = block;
    lp->group_dim.y = 1;
    lp->group_dim.z = 1;
    lp->barrier_bit = barrier_bit_queue_default;
    lp->launch_fence = -1;

    auto crit = stream->lockopen_preKernelCommand();
    lp->av = &(crit->_av);
    lp->cf = nullptr;


    ihipPrintKernelLaunch(kernelNameStr, lp, stream);
    return (stream);
}


//---
//Called after kernel finishes execution.
//This releases the lock on the stream.
void ihipPostLaunchKernel(const char *kernelName, hipStream_t stream, grid_launch_parm &lp)
{
    tprintf(DB_SYNC, "ihipPostLaunchKernel, unlocking stream\n");

    stream->lockclose_postKernelCommand(kernelName, lp.av);
    MARKER_END();
}

//=================================================================================================
// HIP API Implementation
//
// Implementor notes:
// _ All functions should call HIP_INIT_API as first action:
//    HIP_INIT_API(<function_arguments>);
//
// - ALl functions should use ihipLogStatus to return error code (not return error directly).
//=================================================================================================
//
//---

//-------------------------------------------------------------------------------------------------



const char *ihipErrorString(hipError_t hip_error)
{
    switch (hip_error) {
        case hipSuccess                         : return "hipSuccess";
        case hipErrorOutOfMemory                : return "hipErrorOutOfMemory";
        case hipErrorNotInitialized             : return "hipErrorNotInitialized";
        case hipErrorDeinitialized              : return "hipErrorDeinitialized";
        case hipErrorProfilerDisabled           : return "hipErrorProfilerDisabled";
        case hipErrorProfilerNotInitialized     : return "hipErrorProfilerNotInitialized";
        case hipErrorProfilerAlreadyStarted     : return "hipErrorProfilerAlreadyStarted";
        case hipErrorProfilerAlreadyStopped     : return "hipErrorProfilerAlreadyStopped";
        case hipErrorInvalidImage               : return "hipErrorInvalidImage";
        case hipErrorInvalidContext             : return "hipErrorInvalidContext";
        case hipErrorContextAlreadyCurrent      : return "hipErrorContextAlreadyCurrent";
        case hipErrorMapFailed                  : return "hipErrorMapFailed";
        case hipErrorUnmapFailed                : return "hipErrorUnmapFailed";
        case hipErrorArrayIsMapped              : return "hipErrorArrayIsMapped";
        case hipErrorAlreadyMapped              : return "hipErrorAlreadyMapped";
        case hipErrorNoBinaryForGpu             : return "hipErrorNoBinaryForGpu";
        case hipErrorAlreadyAcquired            : return "hipErrorAlreadyAcquired";
        case hipErrorNotMapped                  : return "hipErrorNotMapped";
        case hipErrorNotMappedAsArray           : return "hipErrorNotMappedAsArray";
        case hipErrorNotMappedAsPointer         : return "hipErrorNotMappedAsPointer";
        case hipErrorECCNotCorrectable          : return "hipErrorECCNotCorrectable";
        case hipErrorUnsupportedLimit           : return "hipErrorUnsupportedLimit";
        case hipErrorContextAlreadyInUse        : return "hipErrorContextAlreadyInUse";
        case hipErrorPeerAccessUnsupported      : return "hipErrorPeerAccessUnsupported";
        case hipErrorInvalidKernelFile          : return "hipErrorInvalidKernelFile";
        case hipErrorInvalidGraphicsContext     : return "hipErrorInvalidGraphicsContext";
        case hipErrorInvalidSource              : return "hipErrorInvalidSource";
        case hipErrorFileNotFound               : return "hipErrorFileNotFound";
        case hipErrorSharedObjectSymbolNotFound : return "hipErrorSharedObjectSymbolNotFound";
        case hipErrorSharedObjectInitFailed     : return "hipErrorSharedObjectInitFailed";
        case hipErrorOperatingSystem            : return "hipErrorOperatingSystem";
        case hipErrorSetOnActiveProcess         : return "hipErrorSetOnActiveProcess";
        case hipErrorInvalidHandle              : return "hipErrorInvalidHandle";
        case hipErrorNotFound                   : return "hipErrorNotFound";
        case hipErrorIllegalAddress             : return "hipErrorIllegalAddress";

        case hipErrorMissingConfiguration       : return "hipErrorMissingConfiguration";
        case hipErrorMemoryAllocation           : return "hipErrorMemoryAllocation";
        case hipErrorInitializationError        : return "hipErrorInitializationError";
        case hipErrorLaunchFailure              : return "hipErrorLaunchFailure";
        case hipErrorPriorLaunchFailure         : return "hipErrorPriorLaunchFailure";
        case hipErrorLaunchTimeOut              : return "hipErrorLaunchTimeOut";
        case hipErrorLaunchOutOfResources       : return "hipErrorLaunchOutOfResources";
        case hipErrorInvalidDeviceFunction      : return "hipErrorInvalidDeviceFunction";
        case hipErrorInvalidConfiguration       : return "hipErrorInvalidConfiguration";
        case hipErrorInvalidDevice              : return "hipErrorInvalidDevice";
        case hipErrorInvalidValue               : return "hipErrorInvalidValue";
        case hipErrorInvalidDevicePointer       : return "hipErrorInvalidDevicePointer";
        case hipErrorInvalidMemcpyDirection     : return "hipErrorInvalidMemcpyDirection";
        case hipErrorUnknown                    : return "hipErrorUnknown";
        case hipErrorInvalidResourceHandle      : return "hipErrorInvalidResourceHandle";
        case hipErrorNotReady                   : return "hipErrorNotReady";
        case hipErrorNoDevice                   : return "hipErrorNoDevice";
        case hipErrorPeerAccessAlreadyEnabled   : return "hipErrorPeerAccessAlreadyEnabled";

        case hipErrorPeerAccessNotEnabled       : return "hipErrorPeerAccessNotEnabled";
        case hipErrorRuntimeMemory              : return "hipErrorRuntimeMemory";
        case hipErrorRuntimeOther               : return "hipErrorRuntimeOther";
        case hipErrorHostMemoryAlreadyRegistered : return "hipErrorHostMemoryAlreadyRegistered";
        case hipErrorHostMemoryNotRegistered    : return "hipErrorHostMemoryNotRegistered";
        case hipErrorTbd                        : return "hipErrorTbd";
        default                                 : return "hipErrorUnknown";
    };
};




// Returns true if copyEngineCtx can see the memory allocated on dstCtx and srcCtx.
// The peer-list for a context controls which contexts have access to the memory allocated on that context.
// So we check dstCtx's and srcCtx's peerList to see if the both include thisCtx.
bool ihipStream_t::canSeeMemory(const ihipCtx_t *copyEngineCtx, const hc::AmPointerInfo *dstPtrInfo, const hc::AmPointerInfo *srcPtrInfo)
{

    // Make sure this is a device-to-device copy with all memory available to the requested copy engine
    //
    // TODO - pointer-info stores a deviceID not a context,may have some unusual side-effects here:
    if (dstPtrInfo->_sizeBytes == 0) {
        return false;
    } else {
        ihipCtx_t *dstCtx = ihipGetPrimaryCtx(dstPtrInfo->_appId);
        if (copyEngineCtx != dstCtx) {
            // Only checks peer list if contexts are different
            LockedAccessor_CtxCrit_t  ctxCrit(dstCtx->criticalData());
            //tprintf(DB_SYNC, "dstCrit lock succeeded\n");
            if (!ctxCrit->isPeerWatcher(copyEngineCtx)) {
                return false;
            };
        }
    }



    // TODO - pointer-info stores a deviceID not a context,may have some unusual side-effects here:
    if (srcPtrInfo->_sizeBytes == 0) {
        return false;
    } else {
        ihipCtx_t *srcCtx = ihipGetPrimaryCtx(srcPtrInfo->_appId);
        if (copyEngineCtx != srcCtx) {
            // Only checks peer list if contexts are different
            LockedAccessor_CtxCrit_t  ctxCrit(srcCtx->criticalData());
            //tprintf(DB_SYNC, "srcCrit lock succeeded\n");
            if (!ctxCrit->isPeerWatcher(copyEngineCtx)) {
                return false;
            };
        }
    }

    return true;
};


#define CASE_STRING(X)  case X: return #X ;break;

const char* hipMemcpyStr(unsigned memKind)
{
    switch (memKind) {
        CASE_STRING(hipMemcpyHostToHost);
        CASE_STRING(hipMemcpyHostToDevice);
        CASE_STRING(hipMemcpyDeviceToHost);
        CASE_STRING(hipMemcpyDeviceToDevice);
        CASE_STRING(hipMemcpyDefault);
        default : return ("unknown memcpyKind");
    };
}

const char* hcMemcpyStr(hc::hcCommandKind memKind)
{
    using namespace hc;
    switch (memKind) {
        CASE_STRING(hcMemcpyHostToHost);
        CASE_STRING(hcMemcpyHostToDevice);
        CASE_STRING(hcMemcpyDeviceToHost);
        CASE_STRING(hcMemcpyDeviceToDevice);
        //CASE_STRING(hcMemcpyDefault);
        default : return ("unknown memcpyKind");
    };
}



// Resolve hipMemcpyDefault to a known type.
unsigned ihipStream_t::resolveMemcpyDirection(bool srcInDeviceMem, bool dstInDeviceMem)
{
    hipMemcpyKind kind = hipMemcpyDefault;

    if( srcInDeviceMem  &&  dstInDeviceMem) { kind = hipMemcpyDeviceToDevice; }
    if( srcInDeviceMem  && !dstInDeviceMem) { kind = hipMemcpyDeviceToHost; }
    if(!srcInDeviceMem  && !dstInDeviceMem) { kind = hipMemcpyHostToHost; }
    if(!srcInDeviceMem  &&  dstInDeviceMem) { kind = hipMemcpyHostToDevice; }

    assert (kind != hipMemcpyDefault);
    return kind;
}


// hipMemKind must be "resolved" to a specific direction - cannot be default.
void ihipStream_t::resolveHcMemcpyDirection(unsigned hipMemKind,
                                            const hc::AmPointerInfo *dstPtrInfo,
                                            const hc::AmPointerInfo *srcPtrInfo,
                                            hc::hcCommandKind *hcCopyDir,
                                            ihipCtx_t **copyDevice,
                                            bool *forceUnpinnedCopy)
{
    // Ignore what the user tells us and always resolve the direction:
    // Some apps apparently rely on this.
    hipMemKind = resolveMemcpyDirection(srcPtrInfo->_isInDeviceMem, dstPtrInfo->_isInDeviceMem);


    switch (hipMemKind) {
        case hipMemcpyHostToHost:     *hcCopyDir = hc::hcMemcpyHostToHost; break;
        case hipMemcpyHostToDevice:   *hcCopyDir = hc::hcMemcpyHostToDevice; break;
        case hipMemcpyDeviceToHost:   *hcCopyDir = hc::hcMemcpyDeviceToHost; break;
        case hipMemcpyDeviceToDevice: *hcCopyDir = hc::hcMemcpyDeviceToDevice; break;
        default: throw ihipException(hipErrorRuntimeOther);
    };

    if (srcPtrInfo->_isInDeviceMem) {
        *copyDevice  = ihipGetPrimaryCtx(srcPtrInfo->_appId);
    } else if (dstPtrInfo->_isInDeviceMem) {
        *copyDevice  = ihipGetPrimaryCtx(dstPtrInfo->_appId);
    } else {
        *copyDevice = nullptr;
    }

    *forceUnpinnedCopy = false;
    if (canSeeMemory(*copyDevice, dstPtrInfo, srcPtrInfo)) {

        if (HIP_FORCE_P2P_HOST & 0x1) {
            *forceUnpinnedCopy = true;
            tprintf (DB_COPY, "Copy engine (dev:%d agent=0x%lx) can see src and dst but HIP_FORCE_P2P_HOST=0, forcing copy through staging buffers.\n",
                    *copyDevice ? (*copyDevice)->getDeviceNum() : -1, 
                    *copyDevice ? (*copyDevice)->getDevice()->_hsaAgent.handle : 0x0);

        } else {
            tprintf (DB_COPY, "Copy engine (dev:%d agent=0x%lx) can see src and dst.\n",
                    *copyDevice ? (*copyDevice)->getDeviceNum() : -1, 
                    *copyDevice ? (*copyDevice)->getDevice()->_hsaAgent.handle : 0x0);
        }
    } else {
        *forceUnpinnedCopy = true;
        tprintf (DB_COPY, "P2P: Copy engine(dev:%d agent=0x%lx) cannot see both host and device pointers - forcing copy with unpinned engine.\n",
                    *copyDevice ? (*copyDevice)->getDeviceNum() : -1, 
                    *copyDevice ? (*copyDevice)->getDevice()->_hsaAgent.handle : 0x0);
        if (HIP_FAIL_SOC & 0x2) {
            fprintf (stderr, "HIP_FAIL_SOC:  P2P: copy engine(dev:%d agent=0x%lx) cannot see both host and device pointers - forcing copy with unpinned engine.\n",
                    *copyDevice ? (*copyDevice)->getDeviceNum() : -1, 
                    *copyDevice ? (*copyDevice)->getDevice()->_hsaAgent.handle : 0x0);
            throw ihipException(hipErrorRuntimeOther);
        }
    }
}


void printPointerInfo(unsigned dbFlag, const char *tag, const void *ptr, const hc::AmPointerInfo &ptrInfo)
{
    tprintf (dbFlag, "  %s=%p baseHost=%p baseDev=%p sz=%zu home_dev=%d tracked=%d isDevMem=%d registered=%d\n",
             tag, ptr,
             ptrInfo._hostPointer, ptrInfo._devicePointer, ptrInfo._sizeBytes,
             ptrInfo._appId, ptrInfo._sizeBytes != 0, ptrInfo._isInDeviceMem, !ptrInfo._isAmManaged);
}


// the pointer-info as returned by HC refers to the allocation
// This routine modifies the pointer-info so it appears to refer to the specific ptr and sizeBytes. 
// TODO -remove this when HCC uses HSA pointer info functions directly.
void tailorPtrInfo(hc::AmPointerInfo *ptrInfo, const void * ptr, size_t sizeBytes)
{
    const char *ptrc = static_cast<const char *> (ptr);
    if (ptrInfo->_sizeBytes == 0) {
        // invalid ptrInfo, don't modify
        return;
    } else if (ptrInfo->_isInDeviceMem) {
        assert (ptrInfo->_devicePointer != nullptr);
        std::ptrdiff_t diff = ptrc - static_cast<const char*> (ptrInfo->_devicePointer);

        //TODO : assert-> runtime assert that only appears in debug mode
        assert (diff >= 0);
        assert (diff <= ptrInfo->_sizeBytes);

        ptrInfo->_devicePointer = const_cast<void*> (ptr);

        if (ptrInfo->_hostPointer != nullptr) {
            ptrInfo->_hostPointer = static_cast<char*>(ptrInfo->_hostPointer) + diff;
        }

    } else {

        assert (ptrInfo->_hostPointer != nullptr);
        std::ptrdiff_t diff = ptrc - static_cast<const char*> (ptrInfo->_hostPointer);

        //TODO : assert-> runtime assert that only appears in debug mode
        assert (diff >= 0);
        assert (diff <= ptrInfo->_sizeBytes);

        ptrInfo->_hostPointer = const_cast<void*>(ptr);

        if (ptrInfo->_devicePointer != nullptr) {
            ptrInfo->_devicePointer = static_cast<char*>(ptrInfo->_devicePointer) + diff;
        }
    }

    assert (sizeBytes <= ptrInfo->_sizeBytes);
    ptrInfo->_sizeBytes = sizeBytes;
};


bool getTailoredPtrInfo(hc::AmPointerInfo *ptrInfo, const void * ptr, size_t sizeBytes)
{
    bool tracked = (hc::am_memtracker_getinfo(ptrInfo, ptr) == AM_SUCCESS);

    if (tracked)  {
        tailorPtrInfo(ptrInfo, ptr, sizeBytes);
    }

    return tracked;
};


// TODO : For registered and host memory, if the portable flag is set, we need to recognize that and perform appropriate copy operation.
// What can happen now is that Portable memory is mapped into multiple devices but Peer access is not enabled. i
// The peer detection logic doesn't see that the memory is already mapped and so tries to use an unpinned copy algorithm.  If this is PinInPlace, then an error can occur.
// Need to track Portable flag correctly or use new RT functionality to query the peer status for the pointer.
//
// TODO - remove kind parm from here or use it below?
void ihipStream_t::locked_copySync(void* dst, const void* src, size_t sizeBytes, unsigned kind, bool resolveOn)
{
    ihipCtx_t *ctx = this->getCtx();
    const ihipDevice_t *device = ctx->getDevice();

    if (device == NULL) {
        throw ihipException(hipErrorInvalidDevice);
    }

    hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
    hc::AmPointerInfo dstPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);
    hc::AmPointerInfo srcPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
    hc::AmPointerInfo dstPtrInfo(NULL, NULL, 0, acc, 0, 0);
    hc::AmPointerInfo srcPtrInfo(NULL, NULL, 0, acc, 0, 0);
#endif
    bool dstTracked = getTailoredPtrInfo(&dstPtrInfo, dst, sizeBytes);
    bool srcTracked = getTailoredPtrInfo(&srcPtrInfo, src, sizeBytes);


    // Some code in HCC and in printPointerInfo uses _sizeBytes==0 as an indication ptr is not valid, so check it here:
    if (!dstTracked) {
        assert (dstPtrInfo._sizeBytes == 0);
    }
    if (!srcTracked) {
        assert (srcPtrInfo._sizeBytes == 0);
    }


    hc::hcCommandKind hcCopyDir;
    ihipCtx_t *copyDevice;
    bool forceUnpinnedCopy;
    resolveHcMemcpyDirection(kind, &dstPtrInfo, &srcPtrInfo, &hcCopyDir, &copyDevice, &forceUnpinnedCopy);

    {
        LockedAccessor_StreamCrit_t crit (_criticalData);
        tprintf (DB_COPY, "copySync copyDev:%d  dst=%p (phys_dev:%d, isDevMem:%d)  src=%p(phys_dev:%d, isDevMem:%d)   sz=%zu dir=%s forceUnpinnedCopy=%d\n",
                 copyDevice ? copyDevice->getDeviceNum():-1,
                 dst, dstPtrInfo._appId, dstPtrInfo._isInDeviceMem,
                 src, srcPtrInfo._appId, srcPtrInfo._isInDeviceMem,
                 sizeBytes, hcMemcpyStr(hcCopyDir), forceUnpinnedCopy);
        printPointerInfo(DB_COPY, "  dst", dst, dstPtrInfo);
        printPointerInfo(DB_COPY, "  src", src, srcPtrInfo);


        crit->_av.copy_ext(src, dst, sizeBytes, hcCopyDir, srcPtrInfo, dstPtrInfo, copyDevice ? &copyDevice->getDevice()->_acc : nullptr, forceUnpinnedCopy);
    }
}

void ihipStream_t::addSymbolPtrToTracker(hc::accelerator& acc, void* ptr, size_t sizeBytes) {
#if (__hcc_workweek__ >= 17332)
  hc::AmPointerInfo ptrInfo(NULL, ptr, ptr, sizeBytes, acc, true, false);
#else
  hc::AmPointerInfo ptrInfo(NULL, ptr, sizeBytes, acc, true, false);
#endif
  hc::am_memtracker_add(ptr, ptrInfo);
}

void ihipStream_t::lockedSymbolCopySync(hc::accelerator &acc, void* dst, void* src, size_t sizeBytes, size_t offset, unsigned kind)
{
  if(kind == hipMemcpyHostToHost){
    acc.memcpy_symbol(dst, (void*)src, sizeBytes, offset, Kalmar::hcMemcpyHostToHost);
  }
  if(kind == hipMemcpyHostToDevice){
    acc.memcpy_symbol(dst, (void*)src, sizeBytes, offset);
  }
  if(kind == hipMemcpyDeviceToDevice){
    acc.memcpy_symbol(dst, (void*)src, sizeBytes, offset, Kalmar::hcMemcpyDeviceToDevice);
  }
  if(kind == hipMemcpyDeviceToHost){
    acc.memcpy_symbol((void*) src, (void*)dst, sizeBytes, offset, Kalmar::hcMemcpyDeviceToHost);
  }
}

void ihipStream_t::lockedSymbolCopyAsync(hc::accelerator &acc, void* dst, void* src, size_t sizeBytes, size_t offset, unsigned kind)
{
    // TODO - review - this looks broken , should not be adding pointers to tracker dynamically:
  if(kind == hipMemcpyHostToDevice) {
#if (__hcc_workweek__ >= 17332)
    hc::AmPointerInfo srcPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
    hc::AmPointerInfo srcPtrInfo(NULL, NULL, 0, acc, 0, 0);
#endif
    bool srcTracked = (hc::am_memtracker_getinfo(&srcPtrInfo, src) == AM_SUCCESS);
    if(srcTracked) {
        addSymbolPtrToTracker(acc, dst, sizeBytes);
        locked_getAv()->copy_async((void*)src, dst, sizeBytes);
    } else {
        LockedAccessor_StreamCrit_t crit(_criticalData);
        this->wait(crit);
        acc.memcpy_symbol(dst, (void*)src, sizeBytes, offset);
    }
  }
  if(kind == hipMemcpyDeviceToHost) {
#if (__hcc_workweek__ >= 17332)
    hc::AmPointerInfo dstPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
    hc::AmPointerInfo dstPtrInfo(NULL, NULL, 0, acc, 0, 0);
#endif
    bool dstTracked = (hc::am_memtracker_getinfo(&dstPtrInfo, dst) == AM_SUCCESS);
    if(dstTracked) {
        addSymbolPtrToTracker(acc, src, sizeBytes);
        locked_getAv()->copy_async((void*)src, dst, sizeBytes);
    } else {
        LockedAccessor_StreamCrit_t crit(_criticalData);
        this->wait(crit);
        acc.memcpy_symbol((void*)src, (void*)dst, sizeBytes, offset, Kalmar::hcMemcpyDeviceToHost);
    }
  }
}


void ihipStream_t::locked_copyAsync(void* dst, const void* src, size_t sizeBytes, unsigned kind)
{

    const ihipCtx_t *ctx = this->getCtx();

    if ((ctx == nullptr) || (ctx->getDevice() == nullptr)) {
        tprintf (DB_COPY, "locked_copyAsync bad ctx or device\n");
        throw ihipException(hipErrorInvalidDevice);
    }

    if (kind == hipMemcpyHostToHost) {
        tprintf (DB_COPY, "locked_copyAsync: H2H with memcpy");

        // TODO - consider if we want to perhaps use the GPU SDMA engines anyway, to avoid the host-side sync here and keep everything flowing on the GPU.
        /* As this is a CPU op, we need to wait until all
        the commands in current stream are finished.
        */
        LockedAccessor_StreamCrit_t crit(_criticalData);
        this->wait(crit);

        memcpy(dst, src, sizeBytes);

    } else {

        hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo dstPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);
        hc::AmPointerInfo srcPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo dstPtrInfo(NULL, NULL, 0, acc, 0, 0);
        hc::AmPointerInfo srcPtrInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        bool dstTracked = getTailoredPtrInfo(&dstPtrInfo, dst, sizeBytes);
        bool srcTracked = getTailoredPtrInfo(&srcPtrInfo, src, sizeBytes);


        hc::hcCommandKind hcCopyDir;
        ihipCtx_t *copyDevice;
        bool forceUnpinnedCopy;
        resolveHcMemcpyDirection(kind, &dstPtrInfo, &srcPtrInfo, &hcCopyDir, &copyDevice, &forceUnpinnedCopy);
        tprintf (DB_COPY, "copyASync copyDev:%d  dst=%p (phys_dev:%d, isDevMem:%d)  src=%p(phys_dev:%d, isDevMem:%d)   sz=%zu dir=%s forceUnpinnedCopy=%d\n",
                 copyDevice ? copyDevice->getDeviceNum():-1,
                 dst, dstPtrInfo._appId, dstPtrInfo._isInDeviceMem,
                 src, srcPtrInfo._appId, srcPtrInfo._isInDeviceMem,
                 sizeBytes, hcMemcpyStr(hcCopyDir), forceUnpinnedCopy);
        printPointerInfo(DB_COPY, "  dst", dst, dstPtrInfo);
        printPointerInfo(DB_COPY, "  src", src, srcPtrInfo);

        // "tracked" really indicates if the pointer's virtual address is available in the GPU address space.
        // If both pointers are not tracked, we need to fall back to a sync copy.
        if (dstTracked && srcTracked && !forceUnpinnedCopy && copyDevice/*code below assumes this is !nullptr*/) {
            LockedAccessor_StreamCrit_t crit(_criticalData);

            // Perform fast asynchronous copy - we know copyDevice != NULL based on check above
            try {

                if (HIP_FORCE_SYNC_COPY) {
                    crit->_av.copy_ext      (src, dst, sizeBytes, hcCopyDir, srcPtrInfo, dstPtrInfo, &copyDevice->getDevice()->_acc, forceUnpinnedCopy);

                } else {
                    crit->_av.copy_async_ext(src, dst, sizeBytes, hcCopyDir, srcPtrInfo, dstPtrInfo, &copyDevice->getDevice()->_acc);
                }
            } catch (Kalmar::runtime_exception) {
                throw ihipException(hipErrorRuntimeOther);
            };


            if (HIP_API_BLOCKING) {
                tprintf(DB_SYNC, "%s LAUNCH_BLOCKING for completion of hipMemcpyAsync(sz=%zu)\n", ToString(this).c_str(), sizeBytes);
                this->wait(crit);
            }

        } else {
            if (HIP_FAIL_SOC & 0x1) {
                fprintf (stderr, "HIP_FAIL_SOC failed, async_copy requested but could not be completed since src or dst not accesible to copy agent\n");
                fprintf (stderr, "copyASync copyDev:%d  dst=%p (phys_dev:%d, isDevMem:%d)  src=%p(phys_dev:%d, isDevMem:%d)   sz=%zu dir=%s forceUnpinnedCopy=%d\n",
                         copyDevice ? copyDevice->getDeviceNum():-1,
                         dst, dstPtrInfo._appId, dstPtrInfo._isInDeviceMem,
                         src, srcPtrInfo._appId, srcPtrInfo._isInDeviceMem,
                         sizeBytes, hcMemcpyStr(hcCopyDir), forceUnpinnedCopy);
                fprintf (stderr, "  dst=%p baseHost=%p baseDev=%p sz=%zu home_dev=%d tracked=%d isDevMem=%d\n",
                         dst, dstPtrInfo._hostPointer, dstPtrInfo._devicePointer, dstPtrInfo._sizeBytes,
                         dstPtrInfo._appId, dstTracked, dstPtrInfo._isInDeviceMem);
                fprintf (stderr, "  src=%p baseHost=%p baseDev=%p sz=%zu home_dev=%d tracked=%d isDevMem=%d\n",
                         src, srcPtrInfo._hostPointer, srcPtrInfo._devicePointer, srcPtrInfo._sizeBytes,
                         srcPtrInfo._appId, srcTracked, srcPtrInfo._isInDeviceMem);
                throw ihipException(hipErrorRuntimeOther);
            }
            // Perform slow synchronous copy:
            LockedAccessor_StreamCrit_t crit(_criticalData);


            crit->_av.copy_ext(src, dst, sizeBytes, hcCopyDir, srcPtrInfo, dstPtrInfo, copyDevice ? &copyDevice->getDevice()->_acc : nullptr, forceUnpinnedCopy);
        }
    }
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//Profiler, really these should live elsewhere:
hipError_t hipProfilerStart()
{
    HIP_INIT_API();
#if COMPILE_HIP_ATP_MARKER
    amdtResumeProfiling(AMDT_ALL_PROFILING);
#endif

    return ihipLogStatus(hipSuccess);
};


hipError_t hipProfilerStop()
{
    HIP_INIT_API();
#if COMPILE_HIP_ATP_MARKER
    amdtStopProfiling(AMDT_ALL_PROFILING);
#endif

    return ihipLogStatus(hipSuccess);
};


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// HCC-specific accessor functions:

//---
hipError_t hipHccGetAccelerator(int deviceId, hc::accelerator *acc)
{
    HIP_INIT_API(deviceId, acc);

    const ihipDevice_t *device = ihipGetDevice(deviceId);
    hipError_t err;
    if (device == NULL) {
        err =  hipErrorInvalidDevice;
    } else {
        *acc = device->_acc;
        err = hipSuccess;
    }
    return ihipLogStatus(err);
}


//---
hipError_t hipHccGetAcceleratorView(hipStream_t stream, hc::accelerator_view **av)
{
    HIP_INIT_API(stream, av);

    if (stream == hipStreamNull ) {
        ihipCtx_t *device = ihipGetTlsDefaultCtx();
        stream = device->_defaultStream;
    }

    *av = stream->locked_getAv(); // TODO - review.

    hipError_t err = hipSuccess;
    return ihipLogStatus(err);
}

//// TODO - add identifier numbers for streams and devices to help with debugging.
//TODO - add a contect sequence number for debug. Print operator<< ctx:0.1 (device.ctx)
