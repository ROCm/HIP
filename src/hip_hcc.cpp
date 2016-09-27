/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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

#include "hip_runtime.h"
#include "hcc_detail/hip_hcc.h"
#include "hsa_ext_amd.h"
#include "hsakmt.h"

#include "hcc_detail/trace_helper.h"



//=================================================================================================
//Global variables:
//=================================================================================================
const int release = 1;

#define MEMCPY_D2H_STAGING_VS_PININPLACE_COPY_THRESHOLD    4194304
#define MEMCPY_H2D_DIRECT_VS_STAGING_COPY_THRESHOLD    65336
#define MEMCPY_H2D_STAGING_VS_PININPLACE_COPY_THRESHOLD    1048576

const char *API_COLOR = KGRN;
const char *API_COLOR_END = KNRM;

int HIP_LAUNCH_BLOCKING = 0;

int HIP_PRINT_ENV = 0;
int HIP_TRACE_API= 0;
std::string HIP_TRACE_API_COLOR("green");
int HIP_ATP_MARKER= 0;
int HIP_DB= 0;
int HIP_VISIBLE_DEVICES = 0; /* Contains a comma-separated sequence of GPU identifiers */
int HIP_NUM_KERNELS_INFLIGHT = 128;


std::once_flag hip_initialized;

// Array of pointers to devices.
ihipDevice_t **g_deviceArray;


bool g_visible_device = false;
unsigned g_deviceCnt;
std::vector<int> g_hip_visible_devices;
hsa_agent_t g_cpu_agent;



//=================================================================================================
// Thread-local storage:
//=================================================================================================

// This is the implicit context used by all HIP commands.
// It can be set by hipSetDevice or by the CTX manipulation commands:

thread_local hipError_t tls_lastHipError = hipSuccess;




//=================================================================================================
// Top-level "free" functions:
//=================================================================================================
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
//---
ihipStream_t::ihipStream_t(ihipCtx_t *ctx, hc::accelerator_view av, unsigned int flags) :
    _id(0), // will be set by add function.
    _flags(flags),
    _ctx(ctx),
    _criticalData(av)
{
    tprintf(DB_SYNC, " streamCreate: stream=%p\n", this);
};


//---
ihipStream_t::~ihipStream_t()
{
}


//Wait for all kernel and data copy commands in this stream to complete.
//This signature should be used in routines that already have locked the stream mutex
void ihipStream_t::wait(LockedAccessor_StreamCrit_t &crit, bool assertQueueEmpty)
{
    if (! assertQueueEmpty) {
        tprintf (DB_SYNC, "stream %p wait for queue-empty..\n", this);
        crit->_av.wait();
    }

    crit->_kernelCnt = 0;
}

//---
//Wait for all kernel and data copy commands in this stream to complete.
void ihipStream_t::locked_wait(bool assertQueueEmpty)
{
    LockedAccessor_StreamCrit_t crit(_criticalData);

    wait(crit, assertQueueEmpty);

};

// Causes current stream to wait for specified event to complete:
void ihipStream_t::locked_waitEvent(hipEvent_t event)
{
    LockedAccessor_StreamCrit_t crit(_criticalData);

    // TODO - check state of event here:
    crit->_av.create_blocking_marker(event->_marker);
}

// Create a marker in this stream.
// Save state in the event so it can track the status of the event.
void ihipStream_t::locked_recordEvent(hipEvent_t event)
{
    // Lock the stream to prevent simultaneous access
    LockedAccessor_StreamCrit_t crit(_criticalData);

    event->_marker = crit->_av.create_marker();
}

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

    if(crit->_kernelCnt > HIP_NUM_KERNELS_INFLIGHT){
       this->wait(crit);
       crit->_kernelCnt = 0;
    }
    crit->_kernelCnt++;

    return crit;
}


//---
// Must be called after kernel finishes, this releases the lock on the stream so other commands can submit.
void ihipStream_t::lockclose_postKernelCommand(hc::accelerator_view *av)
{

    if (HIP_LAUNCH_BLOCKING) {
        // TODO - fix this so it goes through proper stream::wait() call.
        av->wait();  // direct wait OK since we know the stream is locked. 
        tprintf(DB_SYNC, " %s LAUNCH_BLOCKING for kernel completion\n", ToString(this).c_str());
    }

    _criticalData.unlock(); // paired with lock from lockopen_preKernelCommand.
};




// Precursor: the stream is already locked,specifically so this routine can enqueue work into the specified av.
void ihipStream_t::launchModuleKernel(
                        hc::accelerator_view av,
                        hsa_signal_t signal,
                        uint32_t blockDimX,
                        uint32_t blockDimY,
                        uint32_t blockDimZ,
                        uint32_t gridDimX,
                        uint32_t gridDimY,
                        uint32_t gridDimZ,
                        uint32_t groupSegmentSize,
						uint32_t privateSegmentSize,
                        void *kernarg,
                        size_t kernSize,
                        uint64_t kernel){
    hsa_status_t status;
    void *kern;

    hsa_amd_memory_pool_t *pool = reinterpret_cast<hsa_amd_memory_pool_t*>(av.get_hsa_kernarg_region());
    status = hsa_amd_memory_pool_allocate(*pool, kernSize, 0, &kern);
    status = hsa_amd_agents_allow_access(1, (hsa_agent_t*)av.get_hsa_agent(), 0, kern);
    memcpy(kern, kernarg, kernSize);
    hsa_queue_t *Queue = (hsa_queue_t*)av.get_hsa_queue();
    const uint32_t queue_mask = Queue->size-1;
    uint32_t packet_index = hsa_queue_load_write_index_relaxed(Queue);
    hsa_kernel_dispatch_packet_t *dispatch_packet = &(((hsa_kernel_dispatch_packet_t*)(Queue->base_address))[packet_index & queue_mask]);

    dispatch_packet->completion_signal = signal;
    dispatch_packet->workgroup_size_x = blockDimX;
    dispatch_packet->workgroup_size_y = blockDimY;
    dispatch_packet->workgroup_size_z = blockDimZ;
    dispatch_packet->grid_size_x = blockDimX * gridDimX;
    dispatch_packet->grid_size_y = blockDimY * gridDimY;
    dispatch_packet->grid_size_z = blockDimZ * gridDimZ;
    dispatch_packet->group_segment_size = groupSegmentSize;
    dispatch_packet->private_segment_size = privateSegmentSize;
    dispatch_packet->kernarg_address = kern;
    dispatch_packet->kernel_object = kernel;
    uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

    uint16_t setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    uint32_t header32 = header | (setup << 16);

    __atomic_store_n((uint32_t*)(dispatch_packet), header32, __ATOMIC_RELEASE);

    hsa_queue_store_write_index_relaxed(Queue, packet_index + 1);
    hsa_signal_store_relaxed(Queue->doorbell_signal, packet_index);
}


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
bool ihipCtxCriticalBase_t<CtxMutex>::isPeer(const ihipCtx_t *peer)
{
    auto match = std::find(_peers.begin(), _peers.end(), peer);
    return (match != std::end(_peers));
}


template<>
bool ihipCtxCriticalBase_t<CtxMutex>::addPeer(ihipCtx_t *peer)
{
    auto match = std::find(_peers.begin(), _peers.end(), peer);
    if (match == std::end(_peers)) {
        // Not already a peer, let's update the list:
        _peers.push_back(peer);
        recomputePeerAgents();
        return true;
    }

    // If we get here - peer was already on list, silently ignore.
    return false;
}


template<>
bool ihipCtxCriticalBase_t<CtxMutex>::removePeer(ihipCtx_t *peer)
{
    auto match = std::find(_peers.begin(), _peers.end(), peer);
    if (match != std::end(_peers)) {
        // Found a valid peer, let's remove it.
        _peers.remove(peer);
        recomputePeerAgents();
        return true;
    } else {
        return false;
    }
}


template<>
void ihipCtxCriticalBase_t<CtxMutex>::resetPeers(ihipCtx_t *thisDevice)
{
    _peers.clear();
    _peerCnt = 0;
    addPeer(thisDevice); // peer-list always contains self agent.
}


template<>
void ihipCtxCriticalBase_t<CtxMutex>::printPeers(FILE *f) const
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
}
//=============================================================================

//=================================================================================================
// ihipDevice_t
//=================================================================================================
ihipDevice_t::ihipDevice_t(unsigned deviceId, unsigned deviceCnt, hc::accelerator &acc) :
    _deviceId(deviceId),
    _acc(acc)
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
    prop->sharedMemPerBlock = 0;
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
    err = hsa_agent_get_info(_hsaAgent, HSA_AGENT_INFO_NAME, &(prop->name));
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

    //uint64_t counterHz;
    //err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &counterHz);
    //DeviceErrorCheck(err);
    //prop->clockInstructionRate = counterHz / 1000;
    prop->clockInstructionRate = 100*1000; /* TODO-RT - hard-code until HSART has function to properly report clock */

    // Get Agent BDFID (bus/device/function ID)
    uint16_t bdf_id = 1;
    err = hsa_agent_get_info(_hsaAgent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf_id);
    DeviceErrorCheck(err);

    // BDFID is 16bit uint: [8bit - BusID | 5bit - Device ID | 3bit - Function/DomainID]
    // TODO/Clarify: cudaDeviceProp::pciDomainID how to report?
    // prop->pciDomainID =  bdf_id & 0x7;
    prop->pciDeviceID =  (bdf_id>>3) & 0x1F;
    prop->pciBusID =  (bdf_id>>8) & 0xFF;

    // Masquerade as a 3.0-level device. This will change as more HW functions are properly supported.
    // Application code should use the arch.has* to do detailed feature detection.
    prop->major = 2;
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

    HsaSystemProperties props;
    hsaKmtReleaseSystemProperties();
    if(HSAKMT_STATUS_SUCCESS == hsaKmtAcquireSystemProperties(&props)) {
        HsaNodeProperties node_prop = {0};
        if(HSAKMT_STATUS_SUCCESS == hsaKmtGetNodeProperties(node, &node_prop)) {
            uint32_t waves_per_cu = node_prop.MaxWavesPerSIMD;
            uint32_t simd_per_cu = node_prop.NumSIMDPerCU;
            prop-> maxThreadsPerMultiProcessor = prop->warpSize*waves_per_cu*simd_per_cu;
        }
    }


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
    prop->arch.hasDoubles                  = 1; // TODO - true for Fiji.
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
    _criticalData(deviceCnt)
{
    locked_reset();

    tprintf(DB_SYNC, "created ctx with defaultStream=%p\n", _defaultStream);
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
        tprintf(DB_SYNC, " delete stream=%p\n", stream);

        delete stream;
    }
    // Clear the list.
    crit->streams().clear();


    // Create a fresh default stream and add it:
    _defaultStream = new ihipStream_t(this, getDevice()->_acc.get_default_view(), hipStreamDefault);
    crit->addStream(_defaultStream);


    // Reset peer list to just me:
    crit->resetPeers(this);

    // Reset and release all memory stored in the tracker:
    // Reset will remove peer mapping so don't need to do this explicitly.
    // FIXME - This is clearly a non-const action!  Is this a context reset or a device reset - maybe should reference count?
    ihipDevice_t *device = getWriteableDevice();
    am_memtracker_reset(device->_acc);

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






// Implement "default" stream syncronization
//   This waits for all other streams to drain before continuing.
//   If waitOnSelf is set, this additionally waits for the default stream to empty.
void ihipCtx_t::locked_syncDefaultStream(bool waitOnSelf)
{
    LockedAccessor_CtxCrit_t  crit(_criticalData);

    tprintf(DB_SYNC, "syncDefaultStream\n");

    for (auto streamI=crit->const_streams().begin(); streamI!=crit->const_streams().end(); streamI++) {
        ihipStream_t *stream = *streamI;

        // Don't wait for streams that have "opted-out" of syncing with NULL stream.
        // And - don't wait for the NULL stream
        if (!(stream->_flags & hipStreamNonBlocking)) {

            if (waitOnSelf || (stream != _defaultStream)) {
                // TODO-hcc - use blocking or active wait here?
                // TODO-sync - cudaDeviceBlockingSync
                stream->locked_wait();
            }
        }
    }
}

//---
void ihipCtx_t::locked_addStream(ihipStream_t *s)
{
    LockedAccessor_CtxCrit_t  crit(_criticalData);

    crit->addStream(s);
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



//---
// Read environment variables.
void ihipReadEnv_I(int *var_ptr, const char *var_name1, const char *var_name2, const char *description)
{
    char * env = getenv(var_name1);

    // Check second name if first not defined, used to allow HIP_ or CUDA_ env vars.
    if ((env == NULL) && strcmp(var_name2, "0")) {
        env = getenv(var_name2);
    }

    // TODO: Refactor this code so it is a separate call rather than being part of ihipReadEnv_I, which should only read integers.
    // Check if the environment variable is either HIP_VISIBLE_DEVICES or CUDA_LAUNCH_BLOCKING, which
    // contains a sequence of comma-separated device IDs
    if (!(strcmp(var_name1,"HIP_VISIBLE_DEVICES") && strcmp(var_name2, "CUDA_VISIBLE_DEVICES")) && env){
        // Parse the string stream of env and store the device ids to g_hip_visible_devices global variable
        std::string str = env;
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
        // Print out the number of ids
        if (HIP_PRINT_ENV) {
            printf ("%-30s = ", var_name1);
            for(int i=0;i<g_hip_visible_devices.size();i++)
                printf ("%2d ", g_hip_visible_devices[i]);
            printf (": %s\n", description);
        }
    }
    else { // Parse environment variables with sigle value
        // Default is set when variable is initialized (at top of this file), so only override if we find
        // an environment variable.
        if (env) {
            long int v = strtol(env, NULL, 0);
            *var_ptr = (int) (v);
        }
        if (HIP_PRINT_ENV) {
            printf ("%-30s = %2d : %s\n", var_name1, *var_ptr, description);
        }
    }
}


void ihipReadEnv_S(std::string *var_ptr, const char *var_name1, const char *var_name2, const char *description)
{
    char * env = getenv(var_name1);

    // Check second name if first not defined, used to allow HIP_ or CUDA_ env vars.
    if ((env == NULL) && strcmp(var_name2, "0")) {
        env = getenv(var_name2);
    }

    if (env) {
        *var_ptr = env;
    }
    if (HIP_PRINT_ENV) {
        printf ("%-30s = %s : %s\n", var_name1, var_ptr->c_str(), description);
    }
}


#if defined (DEBUG)

#define READ_ENV_I(_build, _ENV_VAR, _ENV_VAR2, _description) \
    if ((_build == release) || (_build == debug) {\
        ihipReadEnv_I(&_ENV_VAR, #_ENV_VAR, #_ENV_VAR2, _description);\
    };
#define READ_ENV_S(_build, _ENV_VAR, _ENV_VAR2, _description) \
    if ((_build == release) || (_build == debug) {\
        ihipReadEnv_S(&_ENV_VAR, #_ENV_VAR, #_ENV_VAR2, _description);\
    };

#else

#define READ_ENV_I(_build, _ENV_VAR, _ENV_VAR2, _description) \
    if (_build == release) {\
        ihipReadEnv_I(&_ENV_VAR, #_ENV_VAR, #_ENV_VAR2, _description);\
    };

#define READ_ENV_S(_build, _ENV_VAR, _ENV_VAR2, _description) \
    if (_build == release) {\
        ihipReadEnv_S(&_ENV_VAR, #_ENV_VAR, #_ENV_VAR2, _description);\
    };

#endif



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
    /*
     * Environment variables
     */
    g_hip_visible_devices.push_back(0); /* Set the default value of visible devices */
    READ_ENV_I(release, HIP_PRINT_ENV, 0,  "Print HIP environment variables.");
    //-- READ HIP_PRINT_ENV env first, since it has impact on later env var reading

    // TODO: In HIP/hcc, this variable blocks after both kernel commmands and data transfer.  Maybe should be bit-mask for each command type?
    READ_ENV_I(release, HIP_LAUNCH_BLOCKING, CUDA_LAUNCH_BLOCKING, "Make HIP APIs 'host-synchronous', so they block until any kernel launches or data copy commands complete. Alias: CUDA_LAUNCH_BLOCKING." );
    READ_ENV_I(release, HIP_DB, 0,  "Print various debug info.  Bitmask, see hip_hcc.cpp for more information.");
    if ((HIP_DB & (1<<DB_API))  && (HIP_TRACE_API == 0)) {
        // Set HIP_TRACE_API default before we read it, so it is printed correctly.
        HIP_TRACE_API = 1;
    }

    READ_ENV_I(release, HIP_TRACE_API, 0,  "Trace each HIP API call.  Print function name and return code to stderr as program executes.");
    READ_ENV_S(release, HIP_TRACE_API_COLOR, 0,  "Color to use for HIP_API.  None/Red/Green/Yellow/Blue/Magenta/Cyan/White");
    READ_ENV_I(release, HIP_ATP_MARKER, 0,  "Add HIP function begin/end to ATP file generated with CodeXL");
    READ_ENV_I(release, HIP_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES, "Only devices whose index is present in the secquence are visible to HIP applications and they are enumerated in the order of secquence" );


    READ_ENV_I(release, HIP_NUM_KERNELS_INFLIGHT, 128, "Number of kernels per stream ");

    // Some flags have both compile-time and runtime flags - generate a warning if user enables the runtime flag but the compile-time flag is disabled.
    if (HIP_DB && !COMPILE_HIP_DB) {
        fprintf (stderr, "warning: env var HIP_DB=0x%x but COMPILE_HIP_DB=0.  (perhaps enable COMPILE_HIP_DB in src code before compiling?)", HIP_DB);
    }

    if (HIP_TRACE_API && !COMPILE_HIP_TRACE_API) {
        fprintf (stderr, "warning: env var HIP_TRACE_API=0x%x but COMPILE_HIP_TRACE_API=0.  (perhaps enable COMPILE_HIP_DB in src code before compiling?)", HIP_DB);
    }

    if (HIP_ATP_MARKER && !COMPILE_HIP_ATP_MARKER) {
        fprintf (stderr, "warning: env var HIP_ATP_MARKER=0x%x but COMPILE_HIP_ATP_MARKER=0.  (perhaps enable COMPILE_HIP_DB in src code before compiling?)", HIP_ATP_MARKER);
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

    // If HIP_VISIBLE_DEVICES is not set, make sure all devices are initialized
    if(!g_visible_device) {
        assert(deviceCnt == g_deviceCnt);
    }


    tprintf(DB_SYNC, "pid=%u %-30s\n", getpid(), "<ihipInit>");
}





//---
// Get the stream to use for a command submission.
//
// If stream==NULL synchronize appropriately with other streams and return the default av for the device.
// If stream is valid, return the AV to use.
hipStream_t ihipSyncAndResolveStream(hipStream_t stream)
{
    if (stream == hipStreamNull ) {
        ihipCtx_t *device = ihipGetTlsDefaultCtx();

#ifndef HIP_API_PER_THREAD_DEFAULT_STREAM
        device->locked_syncDefaultStream(false);
#endif
        return device->_defaultStream;
    } else {
        // ALl streams have to wait for legacy default stream to be empty:
        if (!(stream->_flags & hipStreamNonBlocking))  {
            tprintf(DB_SYNC, "stream %p wait default stream\n", stream);
            stream->getCtx()->_defaultStream->locked_wait();
        }

        return stream;
    }
}

void ihipPrintKernelLaunch(const char *kernelName, const grid_launch_parm *lp, const hipStream_t stream)
{
    if (HIP_ATP_MARKER || (COMPILE_HIP_DB && HIP_TRACE_API)) {
        std::stringstream os;
        os  << "<<hip-api: hipLaunchKernel '" << kernelName << "'"
            << " gridDim:"  << lp->grid_dim
            << " groupDim:" << lp->group_dim
            << " sharedMem:+" << lp->dynamic_group_mem_bytes
            << " " << *stream;


        if (COMPILE_HIP_DB && HIP_TRACE_API) {
            std::cerr << API_COLOR << os.str() << API_COLOR_END << std::endl;
        }
        SCOPED_MARKER(os.str().c_str(), "HIP", NULL);
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
void ihipPostLaunchKernel(hipStream_t stream, grid_launch_parm &lp)
{
    stream->lockclose_postKernelCommand(lp.av);
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


void ihipSetTs(hipEvent_t e)
{
    ihipEvent_t *eh = e;
    if (eh->_state == hipEventStatusRecorded) {
        // already recorded, done:
        return;
    } else {
        // TODO - use completion-future functions to obtain ticks and timestamps:
        hsa_signal_t *sig  = static_cast<hsa_signal_t*> (eh->_marker.get_native_handle());
        if (sig) {
            if (hsa_signal_load_acquire(*sig) == 0) {
                eh->_timestamp = eh->_marker.get_end_tick();
                eh->_state = hipEventStatusRecorded;
            }
        }
    }
}


// Returns true if thisCtx can see the memory allocated on dstCtx and srcCtx.
// The peer-list for a context controls which contexts have access to the memory allocated on that context.
// So we check dstCtx's and srcCtx's peerList to see if the booth include thisCtx. 
bool ihipStream_t::canSeePeerMemory(const ihipCtx_t *thisCtx, ihipCtx_t *dstCtx, ihipCtx_t *srcCtx)
{
    tprintf (DB_COPY1, "Checking if direct copy can be used.  thisCtx:%s;  dstCtx:%s ;  srcCtx:%s\n", 
            thisCtx->toString().c_str(), dstCtx->toString().c_str(), srcCtx->toString().c_str());

    // Use blocks to control scope of critical sections.
    {
        LockedAccessor_CtxCrit_t  ctxCrit(dstCtx->criticalData());
        tprintf(DB_SYNC, "dstCrit lock succeeded\n");
        if (!ctxCrit->isPeer(thisCtx)) {
            return false;
        };
    }

    {
        LockedAccessor_CtxCrit_t  ctxCrit(srcCtx->criticalData());
        tprintf(DB_SYNC, "srcCrit lock succeeded\n");
        if (!ctxCrit->isPeer(thisCtx)) {
            return false;
        };
    }

    return true;
};



// Resolve hipMemcpyDefault to a known type.
// TODO - review why is this so complicated, does this need srcTracked and dstTracked?
unsigned ihipStream_t::resolveMemcpyDirection(bool srcTracked, bool dstTracked, bool srcInDeviceMem, bool dstInDeviceMem)
{
    hipMemcpyKind kind = hipMemcpyDefault;
    if(!srcTracked && !dstTracked)
    {
        kind = hipMemcpyHostToHost;
    }
    if(!srcTracked && dstTracked)
    {
        if(dstInDeviceMem) { kind = hipMemcpyHostToDevice; }
        else{ kind = hipMemcpyHostToHost; }
    }
    if (srcTracked && !dstTracked) {
        if(srcInDeviceMem) { kind = hipMemcpyDeviceToHost; }
        else { kind = hipMemcpyHostToHost; }
    }
    if (srcTracked && dstTracked) {
        if(srcInDeviceMem && dstInDeviceMem) { kind = hipMemcpyDeviceToDevice; }
        if(srcInDeviceMem && !dstInDeviceMem) { kind = hipMemcpyDeviceToHost; }
        if(!srcInDeviceMem && !dstInDeviceMem) { kind = hipMemcpyHostToHost; }
        if(!srcInDeviceMem && dstInDeviceMem) { kind = hipMemcpyHostToDevice; }
    }

    assert (kind != hipMemcpyDefault);
    return kind;
}


// TODO - remove kind parm from here or use it below?
void ihipStream_t::copySync(LockedAccessor_StreamCrit_t &crit, void* dst, const void* src, size_t sizeBytes, unsigned kind, bool resolveOn)
{
    ihipCtx_t *ctx = this->getCtx();
    const ihipDevice_t *device = ctx->getDevice();

    if (device == NULL) {
        throw ihipException(hipErrorInvalidDevice);
    }

    hc::accelerator acc;
    hc::AmPointerInfo dstPtrInfo(NULL, NULL, 0, acc, 0, 0);
    hc::AmPointerInfo srcPtrInfo(NULL, NULL, 0, acc, 0, 0);
    bool dstTracked = (hc::am_memtracker_getinfo(&dstPtrInfo, dst) == AM_SUCCESS);
    bool srcTracked = (hc::am_memtracker_getinfo(&srcPtrInfo, src) == AM_SUCCESS);

    if (kind == hipMemcpyDefault) {
        kind = resolveMemcpyDirection(srcTracked, dstTracked, srcPtrInfo._isInDeviceMem, dstPtrInfo._isInDeviceMem);
    }
    hc::hcCommandKind hcCopyDir;
    switch (kind) {
        case hipMemcpyHostToHost:     hcCopyDir = hc::hcMemcpyHostToHost; break;
        case hipMemcpyHostToDevice:   hcCopyDir = hc::hcMemcpyHostToDevice; break;
        case hipMemcpyDeviceToHost:   hcCopyDir = hc::hcMemcpyDeviceToHost; break;
        case hipMemcpyDeviceToDevice: hcCopyDir = hc::hcMemcpyDeviceToDevice; break;
    };


    // If this is P2P accessi, we need to check to see if the copy agent (specified by the stream where the copy is enqueued) 
    // has peer access enabled to both the source and dest.  If this is true, then the copy agent can see both pointers 
    // and we can perform the access with the copy engine from the current stream.  If not true, then we will copy through the host. (forceHostCopyEngine=true).
    bool forceHostCopyEngine = false;
    if (hcCopyDir == hc::hcMemcpyDeviceToDevice) {
        if (!canSeePeerMemory(ctx, ihipGetPrimaryCtx(dstPtrInfo._appId), ihipGetPrimaryCtx(srcPtrInfo._appId))) {
            forceHostCopyEngine = true;
            tprintf (DB_COPY1, "Forcing use of host copy engine.\n");
        } else {
            tprintf (DB_COPY1, "Will use SDMA engine on streamDevice=%s.\n", ctx->toString().c_str());
        }
    };

    crit->_av.copy_ext(src, dst, sizeBytes, hcCopyDir, srcPtrInfo, dstPtrInfo, forceHostCopyEngine);
}


// Sync copy that acquires lock:
void ihipStream_t::locked_copySync(void* dst, const void* src, size_t sizeBytes, unsigned kind, bool resolveOn)
{
    LockedAccessor_StreamCrit_t crit (_criticalData);
    copySync(crit, dst, src, sizeBytes, kind, resolveOn);
}


void ihipStream_t::copyAsync(void* dst, const void* src, size_t sizeBytes, unsigned kind)
{
    LockedAccessor_StreamCrit_t crit(_criticalData);

    const ihipCtx_t *ctx = this->getCtx();

    if ((ctx == nullptr) || (ctx->getDevice() == nullptr)) {
        throw ihipException(hipErrorInvalidDevice);
    }

    if (kind == hipMemcpyHostToHost) {
        tprintf (DB_COPY2, "Asyc: H2H with memcpy");

        // TODO - consider if we want to perhaps use the GPU SDMA engines anyway, to avoid the host-side sync here and keep everything flowing on the GPU.
        /* As this is a CPU op, we need to wait until all
        the commands in current stream are finished.
        */
        this->wait(crit);

        memcpy(dst, src, sizeBytes);

    } else {
        bool trueAsync = true;

        hc::accelerator acc;
        hc::AmPointerInfo dstPtrInfo(NULL, NULL, 0, acc, 0, 0);
        hc::AmPointerInfo srcPtrInfo(NULL, NULL, 0, acc, 0, 0);
        bool dstTracked = (hc::am_memtracker_getinfo(&dstPtrInfo, dst) == AM_SUCCESS);
        bool srcTracked = (hc::am_memtracker_getinfo(&srcPtrInfo, src) == AM_SUCCESS);


        bool copyEngineCanSeeSrcAndDest = true;
        if (kind == hipMemcpyDeviceToDevice) {
            copyEngineCanSeeSrcAndDest = canSeePeerMemory(ctx, ihipGetPrimaryCtx(dstPtrInfo._appId), ihipGetPrimaryCtx(srcPtrInfo._appId));
        }




        // "tracked" really indicates if the pointer's virtual address is available in the GPU address space.
        // If both pointers are not tracked, we need to fall back to a sync copy.
        if (!dstTracked || !srcTracked || !copyEngineCanSeeSrcAndDest) {
            trueAsync = false;
        }


        if (trueAsync == true) {
            // Perform a synchronous copy:
            try {
                crit->_av.copy_async(src, dst, sizeBytes);
            } catch (Kalmar::runtime_exception) {
                throw ihipException(hipErrorRuntimeOther);
            }; 


            if (HIP_LAUNCH_BLOCKING) {
                tprintf(DB_SYNC, "LAUNCH_BLOCKING for completion of hipMemcpyAsync(%zu)\n", sizeBytes);
                this->wait(crit);
            } 
        } else {
            // Perform a synchronous copy:
            if (kind == hipMemcpyDefault) {
                kind = resolveMemcpyDirection(srcTracked, dstTracked, srcPtrInfo._isInDeviceMem, dstPtrInfo._isInDeviceMem);
            }
            copySync(crit, dst, src, sizeBytes, kind);
        }
    }
}

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

    *av = stream->locked_getAv();

    hipError_t err = hipSuccess;
    return ihipLogStatus(err);
}

//// TODO - add identifier numbers for streams and devices to help with debugging.
//TODO - add a contect sequence number for debug. Print operator<< ctx:0.1 (device.ctx)
