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

// TODO, re-org header order.
extern const char *ihipErrorString(hipError_t hip_error);
#include "hcc_detail/trace_helper.h"

const int release = 1;

#define MEMCPY_D2H_STAGING_VS_PININPLACE_COPY_THRESHOLD    4194304
#define MEMCPY_H2D_DIRECT_VS_STAGING_COPY_THRESHOLD    65336
#define MEMCPY_H2D_STAGING_VS_PININPLACE_COPY_THRESHOLD    1048576

int HIP_LAUNCH_BLOCKING = 0;

int HIP_PRINT_ENV = 0;
int HIP_TRACE_API= 0;
int HIP_ATP_MARKER= 0;
int HIP_DB= 0;
int HIP_STAGING_SIZE = 64;   /* size of staging buffers, in KB */
int HIP_STAGING_BUFFERS = 2;    // TODO - remove, two buffers should be enough.
int HIP_PININPLACE = 0;
int HIP_OPTIMAL_MEM_TRANSFER = 0; //ENV Variable to test different memory transfer logics
int HIP_H2D_MEM_TRANSFER_THRESHOLD_DIRECT_OR_STAGING = 0;
int HIP_H2D_MEM_TRANSFER_THRESHOLD_STAGING_OR_PININPLACE = 0;
int HIP_D2H_MEM_TRANSFER_THRESHOLD = 0;
int HIP_STREAM_SIGNALS = 2;  /* number of signals to allocate at stream creation */
int HIP_VISIBLE_DEVICES = 0; /* Contains a comma-separated sequence of GPU identifiers */


//---
// Chicken bits for disabling functionality to work around potential issues:
int HIP_DISABLE_HW_KERNEL_DEP = 0;
int HIP_DISABLE_HW_COPY_DEP = 0;

thread_local int tls_defaultDevice = 0;
thread_local hipError_t tls_lastHipError = hipSuccess;



//=================================================================================================
//Forward Declarations:
//=================================================================================================
bool ihipIsValidDevice(unsigned deviceIndex);

std::once_flag hip_initialized;
ihipDevice_t *g_devices;
bool g_visible_device = false;
unsigned g_deviceCnt;
std::vector<int> g_hip_visible_devices;
hsa_agent_t g_cpu_agent;



//=================================================================================================
// Implementation:
//=================================================================================================


//=================================================================================================
// ihipSignal_t:
//=================================================================================================
//
//---
ihipSignal_t::ihipSignal_t() :  _sig_id(0)
{
    if (hsa_signal_create(0/*value*/, 0, NULL, &_hsa_signal) != HSA_STATUS_SUCCESS) {
        throw ihipException(hipErrorRuntimeMemory);
    }
    //tprintf (DB_SIGNAL, "  allocated hsa_signal=%lu\n", (_hsa_signal.handle));
}

//---
ihipSignal_t::~ihipSignal_t()
{
    tprintf (DB_SIGNAL, "  destroy hsa_signal #%lu (#%lu)\n", (_hsa_signal.handle), _sig_id);
    if (hsa_signal_destroy(_hsa_signal) != HSA_STATUS_SUCCESS) {
       throw ihipException(hipErrorRuntimeOther);
    }
};



//=================================================================================================
// ihipStream_t:
//=================================================================================================
//---
ihipStream_t::ihipStream_t(unsigned device_index, hc::accelerator_view av, unsigned int flags) :
    _id(0), // will be set by add function.
    _av(av),
    _flags(flags),
    _device_index(device_index)
{
    tprintf(DB_SYNC, " streamCreate: stream=%p\n", this);
};


//---
ihipStream_t::~ihipStream_t()
{
}



//---
//TODO - this function is dangerous since it does not propertly account
//for younger commands which may be depending on the signals we are reclaiming.
//Will fix when we move to HCC management of copy signals.
void ihipStream_t::locked_reclaimSignals(SIGSEQNUM sigNum)
{
    LockedAccessor_StreamCrit_t crit(_criticalData);

    tprintf(DB_SIGNAL, "reclaim signal #%lu\n", sigNum);
    // Mark all signals older and including this one as available for re-allocation.
    crit->_oldest_live_sig_id = sigNum+1;
}


//---
void ihipStream_t::waitCopy(LockedAccessor_StreamCrit_t &crit, ihipSignal_t *signal)
{
    SIGSEQNUM sigNum = signal->_sig_id;
    tprintf(DB_SYNC, "waitCopy signal:#%lu\n", sigNum);

    hsa_signal_wait_acquire(signal->_hsa_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);


    tprintf(DB_SIGNAL, "waitCopy reclaim signal #%lu\n", sigNum);
    // Mark all signals older and including this one as available for reclaim
    if (sigNum > crit->_oldest_live_sig_id) {
        crit->_oldest_live_sig_id = sigNum+1; // TODO, +1 here seems dangerous.
    }

}

//Wait for all kernel and data copy commands in this stream to complete.
//This signature should be used in routines that already have locked the stream mutex
void ihipStream_t::wait(LockedAccessor_StreamCrit_t &crit, bool assertQueueEmpty)
{
    if (! assertQueueEmpty) {
        tprintf (DB_SYNC, "stream %p wait for queue-empty..\n", this);
        _av.wait();
    } else {
        assert (crit->_kernelCnt == 0);
    }
     
    if (crit->_last_copy_signal) {
        tprintf (DB_SYNC, "stream %p wait for lastCopy:#%lu...\n", this, lastCopySeqId(crit) );
        this->waitCopy(crit, crit->_last_copy_signal);
    }

    crit->_kernelCnt = 0;

    // Reset the stream to "empty" - next command will not set up an inpute dependency on any older signal.
    crit->_last_command_type = ihipCommandCopyH2D;
    crit->_last_copy_signal = NULL;
    crit->_signalCnt = 0;
}


//---
//Wait for all kernel and data copy commands in this stream to complete.
void ihipStream_t::locked_wait(bool assertQueueEmpty)
{
    LockedAccessor_StreamCrit_t crit(_criticalData);

    wait(crit, assertQueueEmpty);

};


// Recompute the peercnt and the packed _peerAgents whenever a peer is added or deleted.
// The packed _peerAgents can efficiently be used on each memory allocation.
template<> 
void ihipDeviceCriticalBase_t<DeviceMutex>::recomputePeerAgents()
{
    _peerCnt = 0;
    std::for_each (_peers.begin(), _peers.end(), [this](ihipDevice_t* device) {
        _peerAgents[_peerCnt++] = device->_hsa_agent;
    });
}


template<>
bool ihipDeviceCriticalBase_t<DeviceMutex>::isPeer(const ihipDevice_t *peer) 
{
    auto match = std::find(_peers.begin(), _peers.end(), peer);
    return (match != std::end(_peers));
}


template<>
bool ihipDeviceCriticalBase_t<DeviceMutex>::addPeer(ihipDevice_t *peer) 
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
bool ihipDeviceCriticalBase_t<DeviceMutex>::removePeer(ihipDevice_t *peer) 
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
void ihipDeviceCriticalBase_t<DeviceMutex>::resetPeers(ihipDevice_t *thisDevice)
{
    _peers.clear();
    _peerCnt = 0;
    addPeer(thisDevice); // peer-list always contains self agent.
}


template<>
void ihipDeviceCriticalBase_t<DeviceMutex>::addStream(ihipStream_t *stream)
{
    _streams.push_back(stream);
    stream->_id = incStreamId();
}

//-------------------------------------------------------------------------------------------------

//---
//Flavor that takes device index.
ihipDevice_t * getDevice(unsigned deviceIndex) 
{
    if (ihipIsValidDevice(deviceIndex)) {
        return &g_devices[deviceIndex];
    } else {
        return NULL;
    }
};


//---
ihipDevice_t * ihipStream_t::getDevice() const
{
    return ::getDevice(_device_index);
};

#define HIP_NUM_SIGNALS_PER_STREAM 32


//---
// Allocate a new signal from the signal pool.
// Returned signals have value of 0.
// Signals are intended for use in this stream and are always reclaimed "in-order".
ihipSignal_t *ihipStream_t::allocSignal(LockedAccessor_StreamCrit_t &crit)
{
    int numToScan = crit->_signalPool.size();

    crit->_signalCnt++;
    if(crit->_signalCnt == HIP_NUM_SIGNALS_PER_STREAM){
        this->wait(crit);
    }

    do {
        auto thisCursor = crit->_signalCursor;

        if (++crit->_signalCursor == crit->_signalPool.size()) {
            crit->_signalCursor = 0;
        }

        if (crit->_signalPool[thisCursor]._sig_id < crit->_oldest_live_sig_id) {
            SIGSEQNUM oldSigId = crit->_signalPool[thisCursor]._sig_id;
            crit->_signalPool[thisCursor]._index = thisCursor;
            crit->_signalPool[thisCursor]._sig_id  =  ++crit->_stream_sig_id;  // allocate it.
            tprintf(DB_SIGNAL, "allocatSignal #%lu at pos:%i (old sigId:%lu < oldest_live:%lu)\n",
                    crit->_signalPool[thisCursor]._sig_id,
                    thisCursor, oldSigId, crit->_oldest_live_sig_id);



            return &crit->_signalPool[thisCursor];
        }

    } while (--numToScan) ;

    assert(numToScan == 0);

    // Have to grow the pool:
    crit->_signalCursor = crit->_signalPool.size(); // set to the beginning of the new entries:
    if (crit->_signalCursor > 10000) {
        fprintf (stderr, "warning: signal pool size=%d, may indicate runaway number of inflight commands\n", crit->_signalCursor);
    }
    crit->_signalPool.resize(crit->_signalPool.size() * 2);
    tprintf (DB_SIGNAL, "grow signal pool to %zu entries, cursor=%d\n", crit->_signalPool.size(), crit->_signalCursor);
    return allocSignal(crit);  // try again,

    // Should never reach here.
    assert(0);
}


//---
void ihipStream_t::enqueueBarrier(hsa_queue_t* queue, ihipSignal_t *depSignal, ihipSignal_t *completionSignal)
{

    // Obtain the write index for the command queue
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    const uint32_t queueMask = queue->size - 1;

    // Define the barrier packet to be at the calculated queue index address
    hsa_barrier_and_packet_t* barrier = &(((hsa_barrier_and_packet_t*)(queue->base_address))[index&queueMask]);
    memset(barrier, 0, sizeof(hsa_barrier_and_packet_t));

    // setup header
    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    header |= 1 << HSA_PACKET_HEADER_BARRIER;
    //header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    //header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
    barrier->header = header;

    barrier->dep_signal[0].handle = depSignal ? depSignal->_hsa_signal.handle: 0;

    barrier->completion_signal.handle = completionSignal ? completionSignal->_hsa_signal.handle : 0;

    // TODO - check queue overflow, return error:
    // Increment write index and ring doorbell to dispatch the kernel
    hsa_queue_store_write_index_relaxed(queue, index+1);
    hsa_signal_store_relaxed(queue->doorbell_signal, index);
}


int HIP_NUM_KERNELS_INFLIGHT = 128;

//--
//When the commands in a stream change types (ie kernel command follows a data command,
//or data command follows a kernel command), then we need to add a barrier packet
//into the stream to mimic CUDA stream semantics.  (some hardware uses separate
//queues for data commands and kernel commands, and no implicit ordering is provided).
//
bool ihipStream_t::lockopen_preKernelCommand()
{
    LockedAccessor_StreamCrit_t crit(_criticalData, false/*no unlock at destruction*/);

    bool addedSync = false;

    if(crit->_kernelCnt > HIP_NUM_KERNELS_INFLIGHT){
        this->wait(crit);
       crit->_kernelCnt = 0;
    }
    crit->_kernelCnt++;
    // If switching command types, we need to add a barrier packet to synchronize things.
    if (crit->_last_command_type != ihipCommandKernel) {
        if (crit->_last_copy_signal) {
            addedSync = true;

            hsa_queue_t * q =  (hsa_queue_t*)_av.get_hsa_queue();
            if (HIP_DISABLE_HW_KERNEL_DEP == 0) {
                this->enqueueBarrier(q, crit->_last_copy_signal, NULL);
                tprintf (DB_SYNC, "stream %p switch %s to %s (barrier pkt inserted with wait on #%lu)\n",
                        this, ihipCommandName[crit->_last_command_type], ihipCommandName[ihipCommandKernel], crit->_last_copy_signal->_sig_id)

            } else if (HIP_DISABLE_HW_KERNEL_DEP>0) {
                    tprintf (DB_SYNC, "stream %p switch %s to %s (HOST wait for previous...)\n",
                            this, ihipCommandName[crit->_last_command_type], ihipCommandName[ihipCommandKernel]);
                    this->waitCopy(crit, crit->_last_copy_signal);
            } else if (HIP_DISABLE_HW_KERNEL_DEP==-1) {
                tprintf (DB_SYNC, "stream %p switch %s to %s (IGNORE dependency)\n",
                        this, ihipCommandName[crit->_last_command_type], ihipCommandName[ihipCommandKernel]);
            }
        }
        crit->_last_command_type = ihipCommandKernel;
    }

    return addedSync;
}


//---
// Must be called after kernel finishes, this releases the lock on the stream so other commands can submit.
void ihipStream_t::lockclose_postKernelCommand(hc::completion_future &kernelFuture)
{
    // We locked _criticalData in the lockopen_preKernelCommand() so OK to access here:
    _criticalData._last_kernel_future = kernelFuture;

    _criticalData.unlock(); // paired with lock from lockopen_preKernelCommand.
};



//---
// Called whenever a copy command is set to the stream.
// Examines the last command sent to this stream and returns a signal to wait on, if required.
int ihipStream_t::preCopyCommand(LockedAccessor_StreamCrit_t &crit, ihipSignal_t *copyCompletionSignal, hsa_signal_t *waitSignal, ihipCommand_t copyType)
{
    int needSync = 0;

    waitSignal->handle = 0;

    // If switching command types, we need to add a barrier packet to synchronize things.
    if (FORCE_SAMEDIR_COPY_DEP || (crit->_last_command_type != copyType)) {


        if (crit->_last_command_type == ihipCommandKernel) {
            tprintf (DB_SYNC, "stream %p switch %s to %s (async copy dep on prev kernel)\n",
                    this, ihipCommandName[crit->_last_command_type], ihipCommandName[copyType]);
            needSync = 1;
            ihipSignal_t *depSignal = allocSignal(crit);
            hsa_signal_store_relaxed(depSignal->_hsa_signal,1);
            this->enqueueBarrier(static_cast<hsa_queue_t*>(_av.get_hsa_queue()), NULL, depSignal);
            *waitSignal = depSignal->_hsa_signal;
        } else if (crit->_last_copy_signal) {
            needSync = 1;
            tprintf (DB_SYNC, "stream %p switch %s to %s (async copy dep on other copy #%lu)\n",
                    this, ihipCommandName[crit->_last_command_type], ihipCommandName[copyType], crit->_last_copy_signal->_sig_id);
            *waitSignal = crit->_last_copy_signal->_hsa_signal;
        }

        if (HIP_DISABLE_HW_COPY_DEP && needSync) {
            if (HIP_DISABLE_HW_COPY_DEP == -1) {
                tprintf (DB_SYNC, "IGNORE copy dependency\n")

            } else {
                tprintf (DB_SYNC, "HOST-wait for copy dependency\n")
                // do the wait here on the host, and disable the device-side command resolution.
                hsa_signal_wait_acquire(*waitSignal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
                needSync = 0;
            }
        }

        crit->_last_command_type = copyType;
    }

    crit->_last_copy_signal = copyCompletionSignal;

    return needSync;
}




//=================================================================================================
//
//Reset the device - this is called from hipDeviceReset.
//Device may be reset multiple times, and may be reset after init.
void ihipDevice_t::locked_reset()
{
    // Obtain mutex access to the device critical data, release by destructor
    LockedAccessor_DeviceCrit_t  crit(_criticalData);


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
    _default_stream = new ihipStream_t(_device_index, _acc.get_default_view(), hipStreamDefault);
    crit->addStream(_default_stream);


    // This resest peer list to just me:
    crit->resetPeers(this);

    // Reset and release all memory stored in the tracker:
    // Reset will remove peer mapping so don't need to do this explicitly.
    am_memtracker_reset(_acc);

};


//---
void ihipDevice_t::init(unsigned device_index, unsigned deviceCnt, hc::accelerator &acc, unsigned flags)
{
    _device_index = device_index;
    _device_flags = flags;
    _acc = acc;

    hsa_agent_t *agent = static_cast<hsa_agent_t*> (acc.get_hsa_agent());
    if (agent) {
        int err = hsa_agent_get_info(*agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &_compute_units);
        if (err != HSA_STATUS_SUCCESS) {
            _compute_units = 1;
        }

        _hsa_agent = *agent;
    } else {
        _hsa_agent.handle = static_cast<uint64_t> (-1);
    }

    getProperties(&_props);

    _criticalData.init(deviceCnt);

    locked_reset();


    tprintf(DB_SYNC, "created device with default_stream=%p\n", _default_stream);

    hsa_region_t *pinnedHostRegion;
    pinnedHostRegion = static_cast<hsa_region_t*>(_acc.get_hsa_am_system_region());
    _staging_buffer[0] = new StagingBuffer(_hsa_agent, *pinnedHostRegion, HIP_STAGING_SIZE*1024, HIP_STAGING_BUFFERS);
    _staging_buffer[1] = new StagingBuffer(_hsa_agent, *pinnedHostRegion, HIP_STAGING_SIZE*1024, HIP_STAGING_BUFFERS);

};




ihipDevice_t::~ihipDevice_t()
{
    if (_default_stream) {
        delete _default_stream;
        _default_stream = NULL;
    }

    for (int i=0; i<2; i++) {
        if (_staging_buffer[i]) {
            delete _staging_buffer[i];
            _staging_buffer[i] = NULL;
        }
    }
}

//----




//=================================================================================================
// Utility functions, these are not part of the public HIP API
//=================================================================================================

//=================================================================================================

#define DeviceErrorCheck(x) if (x != HSA_STATUS_SUCCESS) { return hipErrorInvalidDevice; }

#define ErrorCheck(x) error_check(x, __LINE__, __FILE__)

void error_check(hsa_status_t hsa_error_code, int line_num, std::string str) {
  if ((hsa_error_code != HSA_STATUS_SUCCESS)&& (hsa_error_code != HSA_STATUS_INFO_BREAK))  {
    printf("HSA reported error!\n In file: %s\nAt line: %d\n", str.c_str(),line_num);
  }
}

// CPU agent used for verification
hsa_agent_t cpu_agent_;
hsa_agent_t gpu_agent_;
int gpu_region_count;
// System region
hsa_amd_memory_pool_t sys_region_;
hsa_amd_memory_pool_t gpu_region_;

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

hsa_status_t FindCpuDevice(hsa_agent_t agent, void* data) {
    if (data == NULL) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    hsa_device_type_t hsa_device_type;
    hsa_status_t hsa_error_code =
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type);
    if (hsa_error_code != HSA_STATUS_SUCCESS) {
        return hsa_error_code;
    }

    if (hsa_device_type == HSA_DEVICE_TYPE_CPU) {
        *((hsa_agent_t*)data) = agent;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t GetDeviceRegion(hsa_amd_memory_pool_t region, void* data) {
    if (NULL == data) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    hsa_status_t err;
    hsa_amd_segment_t segment;
    uint32_t flag;

    err = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    ErrorCheck(err);
    if (HSA_AMD_SEGMENT_GLOBAL != segment) return HSA_STATUS_SUCCESS;
    err = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
    ErrorCheck(err);
    *((hsa_amd_memory_pool_t*)data) = region;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t FindGlobalRegion(hsa_amd_memory_pool_t region, void* data) {
    if (NULL == data) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    hsa_status_t err;
    hsa_amd_segment_t segment;
    uint32_t flag;
    err = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    ErrorCheck(err);

    err = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
    ErrorCheck(err);
    if ((HSA_AMD_SEGMENT_GLOBAL == segment) &&
        (flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
        *((hsa_amd_memory_pool_t*)data) = region;
    }
    return HSA_STATUS_SUCCESS;
}

void FindDeviceRegion()
{
    hsa_status_t err = hsa_iterate_agents(FindGpuDevice, &gpu_agent_);
    ErrorCheck(err);

    err = hsa_amd_agent_iterate_memory_pools(gpu_agent_, GetDeviceRegion, &gpu_region_);
    ErrorCheck(err);
}

void FindSystemRegion()
{
    hsa_status_t err = hsa_iterate_agents(FindCpuDevice, &cpu_agent_);
    ErrorCheck(err);

    err = hsa_amd_agent_iterate_memory_pools(cpu_agent_, FindGlobalRegion, &sys_region_);
    ErrorCheck(err);
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

// Internal version,
hipError_t ihipDevice_t::getProperties(hipDeviceProp_t* prop)
{
    hipError_t e = hipSuccess;
    hsa_status_t err;

    // Set some defaults in case we don't find the appropriate regions:
    prop->totalGlobalMem = 0;
    prop->totalConstMem = 0;
    prop->sharedMemPerBlock = 0;
    prop-> maxThreadsPerMultiProcessor = 0;
    prop->regsPerBlock = 0;

    if (_hsa_agent.handle == -1) {
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
    err = hsa_agent_get_info(_hsa_agent, HSA_AGENT_INFO_NAME, &(prop->name));
    DeviceErrorCheck(err);

    // Get agent node
    uint32_t node;
    err = hsa_agent_get_info(_hsa_agent, HSA_AGENT_INFO_NODE, &node);
    DeviceErrorCheck(err);

    // Get wavefront size
    err = hsa_agent_get_info(_hsa_agent, HSA_AGENT_INFO_WAVEFRONT_SIZE,&prop->warpSize);
    DeviceErrorCheck(err);

    // Get max total number of work-items in a workgroup
    err = hsa_agent_get_info(_hsa_agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &prop->maxThreadsPerBlock );
    DeviceErrorCheck(err);

    // Get max number of work-items of each dimension of a work-group
    uint16_t work_group_max_dim[3];
    err = hsa_agent_get_info(_hsa_agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, work_group_max_dim);
    DeviceErrorCheck(err);
    for( int i =0; i< 3 ; i++) {
        prop->maxThreadsDim[i]= work_group_max_dim[i];
    }

    hsa_dim3_t grid_max_dim;
    err = hsa_agent_get_info(_hsa_agent, HSA_AGENT_INFO_GRID_MAX_DIM, &grid_max_dim);
    DeviceErrorCheck(err);
    prop->maxGridSize[0]= (int) ((grid_max_dim.x == UINT32_MAX) ? (INT32_MAX) : grid_max_dim.x);
    prop->maxGridSize[1]= (int) ((grid_max_dim.y == UINT32_MAX) ? (INT32_MAX) : grid_max_dim.y);
    prop->maxGridSize[2]= (int) ((grid_max_dim.z == UINT32_MAX) ? (INT32_MAX) : grid_max_dim.z);

    // Get Max clock frequency
    err = hsa_agent_get_info(_hsa_agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY, &prop->clockRate);
    prop->clockRate *= 1000.0;   // convert Mhz to Khz.
    DeviceErrorCheck(err);

    //uint64_t counterHz;
    //err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &counterHz);
    //DeviceErrorCheck(err);
    //prop->clockInstructionRate = counterHz / 1000;
    prop->clockInstructionRate = 100*1000; /* TODO-RT - hard-code until HSART has function to properly report clock */

    // Get Agent BDFID (bus/device/function ID)
    uint16_t bdf_id = 1;
    err = hsa_agent_get_info(_hsa_agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf_id);
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
    err = hsa_agent_get_info(_hsa_agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &(prop->multiProcessorCount));
    DeviceErrorCheck(err);

    // TODO-hsart - this appears to return 0?
    uint32_t cache_size[4];
    err = hsa_agent_get_info(_hsa_agent, HSA_AGENT_INFO_CACHE_SIZE, cache_size);
    DeviceErrorCheck(err);
    prop->l2CacheSize = cache_size[1];

    /* Computemode for HSA Devices is always : cudaComputeModeDefault */
    prop->computeMode = 0;

    FindSystemRegion();
    FindDeviceRegion();
    int access=checkAccess(cpu_agent_, gpu_region_);
    if(0!= access){
        isLargeBar= 1;
    }
    else{
        isLargeBar=0;
    }


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
    err = hsa_agent_iterate_regions(_hsa_agent, get_region_info, prop);
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

    prop->concurrentKernels = 1; // All ROCR hardware supports executing multiple kernels concurrently
    if ( _device_flags | hipDeviceMapHost) {
        prop->canMapHostMemory = 1;
    } else {
        prop->canMapHostMemory = 0;
    }
    return e;
}


// Implement "default" stream syncronization
//   This waits for all other streams to drain before continuing.
//   If waitOnSelf is set, this additionally waits for the default stream to empty.
void ihipDevice_t::locked_syncDefaultStream(bool waitOnSelf)
{
    LockedAccessor_DeviceCrit_t  crit(_criticalData);

    tprintf(DB_SYNC, "syncDefaultStream\n");

    for (auto streamI=crit->const_streams().begin(); streamI!=crit->const_streams().end(); streamI++) {
        ihipStream_t *stream = *streamI;

        // Don't wait for streams that have "opted-out" of syncing with NULL stream.
        // And - don't wait for the NULL stream
        if (!(stream->_flags & hipStreamNonBlocking)) {

            if (waitOnSelf || (stream != _default_stream)) {
                // TODO-hcc - use blocking or active wait here?
                // TODO-sync - cudaDeviceBlockingSync
                stream->locked_wait();
            }
        }
    }
}

//---
void ihipDevice_t::locked_addStream(ihipStream_t *s)
{
    LockedAccessor_DeviceCrit_t  crit(_criticalData);

    crit->addStream(s);
}

//---
void ihipDevice_t::locked_removeStream(ihipStream_t *s)
{
    LockedAccessor_DeviceCrit_t  crit(_criticalData);

    crit->streams().remove(s);
}


//---
//Heavyweight synchronization that waits on all streams, ignoring hipStreamNonBlocking flag.
void ihipDevice_t::locked_waitAllStreams()
{
    LockedAccessor_DeviceCrit_t  crit(_criticalData);

    tprintf(DB_SYNC, "waitAllStream\n");
    for (auto streamI=crit->const_streams().begin(); streamI!=crit->const_streams().end(); streamI++) {
        (*streamI)->locked_wait();
    }
}



// Read environment variables.
void ihipReadEnv_I(int *var_ptr, const char *var_name1, const char *var_name2, const char *description)
{
    char * env = getenv(var_name1);

    // Check second name if first not defined, used to allow HIP_ or CUDA_ env vars.
    if ((env == NULL) && strcmp(var_name2, "0")) {
        env = getenv(var_name2);
    }

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
            }else// Any device number after invalid number will not present
                break;
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

#if defined (DEBUG)

#define READ_ENV_I(_build, _ENV_VAR, _ENV_VAR2, _description) \
    if ((_build == release) || (_build == debug) {\
        ihipReadEnv_I(&_ENV_VAR, #_ENV_VAR, #_ENV_VAR2, _description);\
    };

#else

#define READ_ENV_I(_build, _ENV_VAR, _ENV_VAR2, _description) \
    if (_build == release) {\
        ihipReadEnv_I(&_ENV_VAR, #_ENV_VAR, #_ENV_VAR2, _description);\
    };

#endif

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

    READ_ENV_I(release, HIP_LAUNCH_BLOCKING, CUDA_LAUNCH_BLOCKING, "Make HIP APIs 'host-synchronous', so they block until any kernel launches or data copy commands complete. Alias: CUDA_LAUNCH_BLOCKING." );
    READ_ENV_I(release, HIP_DB, 0,  "Print various debug info.  Bitmask, see hip_hcc.cpp for more information.");
    if ((HIP_DB & (1<<DB_API))  && (HIP_TRACE_API == 0)) {
        // Set HIP_TRACE_API default before we read it, so it is printed correctly.
        HIP_TRACE_API = 1;
    }

    READ_ENV_I(release, HIP_TRACE_API, 0,  "Trace each HIP API call.  Print function name and return code to stderr as program executes.");
    READ_ENV_I(release, HIP_ATP_MARKER, 0,  "Add HIP function begin/end to ATP file generated with CodeXL");
    READ_ENV_I(release, HIP_STAGING_SIZE, 0, "Size of each staging buffer (in KB)" );
    READ_ENV_I(release, HIP_STAGING_BUFFERS, 0, "Number of staging buffers to use in each direction. 0=use hsa_memory_copy.");
    READ_ENV_I(release, HIP_PININPLACE, 0, "For unpinned transfers, pin the memory in-place in chunks before doing the copy. Under development.");
    READ_ENV_I(release, HIP_OPTIMAL_MEM_TRANSFER, 0, "For optimal memory transfers for unpinned memory.Under testing.");
    READ_ENV_I(release, HIP_H2D_MEM_TRANSFER_THRESHOLD_DIRECT_OR_STAGING, 0, "Threshold value for H2D unpinned memory transfer decision between direct copy or staging buffer usage,Under testing.");
    READ_ENV_I(release, HIP_H2D_MEM_TRANSFER_THRESHOLD_STAGING_OR_PININPLACE, 0, "Threshold value for H2D unpinned memory transfer decision between staging buffer usage or pininplace usage .Under testing.");
    READ_ENV_I(release, HIP_D2H_MEM_TRANSFER_THRESHOLD, 0, "Threshold value for D2H unpinned memory transfer decision between staging buffer usage or pininplace usage .Under testing.");
    READ_ENV_I(release, HIP_STREAM_SIGNALS, 0, "Number of signals to allocate when new stream is created (signal pool will grow on demand)");
    READ_ENV_I(release, HIP_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES, "Only devices whose index is present in the secquence are visible to HIP applications and they are enumerated in the order of secquence" );

    READ_ENV_I(release, HIP_DISABLE_HW_KERNEL_DEP, 0, "Disable HW dependencies before kernel commands  - instead wait for dependency on host. -1 means ignore these dependencies. (debug mode)");
    READ_ENV_I(release, HIP_DISABLE_HW_COPY_DEP, 0, "Disable HW dependencies before copy commands  - instead wait for dependency on host. -1 means ifnore these dependencies (debug mode)");

    READ_ENV_I(release, HIP_NUM_KERNELS_INFLIGHT, 128, "Number of kernels per stream ");

    if (HIP_OPTIMAL_MEM_TRANSFER && !HIP_H2D_MEM_TRANSFER_THRESHOLD_DIRECT_OR_STAGING) {
        HIP_H2D_MEM_TRANSFER_THRESHOLD_DIRECT_OR_STAGING= MEMCPY_H2D_DIRECT_VS_STAGING_COPY_THRESHOLD;
        fprintf (stderr, "warning: env var HIP_OPTIMAL_MEM_TRANSFER=0x%x but HIP_H2D_MEM_TRANSFER_THRESHOLD_DIRECT_OR_STAGING=0.Using default value for this.\n", HIP_OPTIMAL_MEM_TRANSFER);
    }

    if (HIP_OPTIMAL_MEM_TRANSFER && !HIP_H2D_MEM_TRANSFER_THRESHOLD_STAGING_OR_PININPLACE) {
        HIP_H2D_MEM_TRANSFER_THRESHOLD_STAGING_OR_PININPLACE= MEMCPY_H2D_STAGING_VS_PININPLACE_COPY_THRESHOLD;
        fprintf (stderr, "warning: env var HIP_OPTIMAL_MEM_TRANSFER=0x%x but HIP_H2D_MEM_TRANSFER_THRESHOLD_STAGING_OR_PININPLACE=0.Using default value for this.\n", HIP_OPTIMAL_MEM_TRANSFER);
    }

    if (HIP_OPTIMAL_MEM_TRANSFER && !HIP_D2H_MEM_TRANSFER_THRESHOLD) {
        HIP_D2H_MEM_TRANSFER_THRESHOLD= MEMCPY_D2H_STAGING_VS_PININPLACE_COPY_THRESHOLD;
        fprintf (stderr, "warning: env var HIP_OPTIMAL_MEM_TRANSFER=0x%x but HIP_D2H_MEM_TRANSFER_THRESHOLD=0.Using default value for this.\n", HIP_OPTIMAL_MEM_TRANSFER);
    }
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

    g_devices = new ihipDevice_t[deviceCnt];
    g_deviceCnt = 0;
    for (int i=0; i<accs.size(); i++) {
        // check if the device id is included in the HIP_VISIBLE_DEVICES env variable
        if (! accs[i].get_is_emulated()) {
            if (std::find(g_hip_visible_devices.begin(), g_hip_visible_devices.end(), (i-1)) == g_hip_visible_devices.end() && g_visible_device)
            {
                //If device is not in visible devices list, ignore
                continue;
            }
            g_devices[g_deviceCnt].init(g_deviceCnt, deviceCnt, accs[i], hipDeviceMapHost);
            g_deviceCnt++;
        }
    }

    // If HIP_VISIBLE_DEVICES is not set, make sure all devices are initialized
    if(!g_visible_device) {
        assert(deviceCnt == g_deviceCnt);
    }


    hsa_status_t err = hsa_iterate_agents(findCpuAgent, &g_cpu_agent);
    if (err != HSA_STATUS_INFO_BREAK) {
        // didn't find a CPU.
        throw ihipException(hipErrorRuntimeOther);
    }


    tprintf(DB_SYNC, "pid=%u %-30s\n", getpid(), "<ihipInit>");
}


bool ihipIsValidDevice(unsigned deviceIndex)
{
    // deviceIndex is unsigned so always > 0
    return (deviceIndex < g_deviceCnt);
}

//---
ihipDevice_t *ihipGetTlsDefaultDevice()
{
    // If this is invalid, the TLS state is corrupt.
    // This can fire if called before devices are initialized.
    // TODO - consider replacing assert with error code
    assert (ihipIsValidDevice(tls_defaultDevice));

    return &g_devices[tls_defaultDevice];
}


//---
ihipDevice_t *ihipGetDevice(int deviceId)
{
    if ((deviceId >= 0) && (deviceId < g_deviceCnt)) {
        return &g_devices[deviceId];
    } else {
        return NULL;
    }

}

//---
// Get the stream to use for a command submission.
//
// If stream==NULL synchronize appropriately with other streams and return the default av for the device.
// If stream is valid, return the AV to use.
hipStream_t ihipSyncAndResolveStream(hipStream_t stream)
{
    if (stream == hipStreamNull ) {
        ihipDevice_t *device = ihipGetTlsDefaultDevice();

#ifndef HIP_API_PER_THREAD_DEFAULT_STREAM
        device->locked_syncDefaultStream(false);
#endif
        return device->_default_stream;
    } else {
        // Have to wait for legacy default stream to be empty:
        if (!(stream->_flags & hipStreamNonBlocking))  {
            tprintf(DB_SYNC, "stream %p wait default stream\n", stream);
            stream->getDevice()->_default_stream->locked_wait();
        }

        return stream;
    }
}


// TODO - data-up to data-down:
// Called just before a kernel is launched from hipLaunchKernel.
// Allows runtime to track some information about the stream.
hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, dim3 block, grid_launch_parm *lp)
{
    HIP_INIT_API(stream, grid, block, lp);
    stream = ihipSyncAndResolveStream(stream);
#if USE_GRID_LAUNCH_20 
    lp->grid_dim.x = grid.x;
    lp->grid_dim.y = grid.y;
    lp->grid_dim.z = grid.z;
    lp->group_dim.x = block.x;
    lp->group_dim.y = block.y;
    lp->group_dim.z = block.z;
    lp->barrier_bit = barrier_bit_queue_default;
    lp->launch_fence = -1;
#else
    lp->gridDim.x = grid.x;
    lp->gridDim.y = grid.y;
    lp->gridDim.z = grid.z;
    lp->groupDim.x = block.x;
    lp->groupDim.y = block.y;
    lp->groupDim.z = block.z;
#endif
    stream->lockopen_preKernelCommand();
//    *av = &stream->_av;
    lp->av = &stream->_av;
    lp->cf = new hc::completion_future;
//    lp->av = static_cast<void*>(av);
//    lp->cf = static_cast<void*>(malloc(sizeof(hc::completion_future)));
    return (stream);
}
hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, dim3 block, grid_launch_parm *lp)
{
    HIP_INIT_API(stream, grid, block, lp);
    stream = ihipSyncAndResolveStream(stream);
#if USE_GRID_LAUNCH_20 
    lp->grid_dim.x = grid;
    lp->grid_dim.y = 1;
    lp->grid_dim.z = 1;
    lp->group_dim.x = block.x;
    lp->group_dim.y = block.y;
    lp->group_dim.z = block.z;
    lp->barrier_bit = barrier_bit_queue_default;
    lp->launch_fence = -1;
#else
    lp->gridDim.x = grid;
    lp->gridDim.y = 1;
    lp->gridDim.z = 1;
    lp->groupDim.x = block.x;
    lp->groupDim.y = block.y;
    lp->groupDim.z = block.z;
#endif
    stream->lockopen_preKernelCommand();
//    *av = &stream->_av;
    lp->av = &stream->_av;
    lp->cf = new hc::completion_future;
//    lp->av = static_cast<void*>(av);
//    lp->cf = static_cast<void*>(malloc(sizeof(hc::completion_future)));
    return (stream);
}

hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, size_t block, grid_launch_parm *lp)
{
    HIP_INIT_API(stream, grid, block, lp);
    stream = ihipSyncAndResolveStream(stream);
#if USE_GRID_LAUNCH_20 
    lp->grid_dim.x = grid.x;
    lp->grid_dim.y = grid.y;
    lp->grid_dim.z = grid.z;
    lp->group_dim.x = block;
    lp->group_dim.y = 1;
    lp->group_dim.z = 1;
    lp->barrier_bit = barrier_bit_queue_default;
    lp->launch_fence = -1;
#else
    lp->gridDim.x = grid.x;
    lp->gridDim.y = grid.y;
    lp->gridDim.z = grid.z;
    lp->groupDim.x = block;
    lp->groupDim.y = 1;
    lp->groupDim.z = 1;
#endif
    stream->lockopen_preKernelCommand();
//    *av = &stream->_av;
    lp->av = &stream->_av;
    lp->cf = new hc::completion_future;
//    lp->av = static_cast<void*>(av);
//    lp->cf = static_cast<void*>(malloc(sizeof(hc::completion_future)));
    return (stream);
}

hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, size_t block, grid_launch_parm *lp)
{
    HIP_INIT_API(stream, grid, block, lp);
    stream = ihipSyncAndResolveStream(stream);
#if USE_GRID_LAUNCH_20 
    lp->grid_dim.x = grid;
    lp->grid_dim.y = 1;
    lp->grid_dim.z = 1;
    lp->group_dim.x = block;
    lp->group_dim.y = 1;
    lp->group_dim.z = 1;
    lp->barrier_bit = barrier_bit_queue_default;
    lp->launch_fence = -1;
#else
    lp->gridDim.x = grid;
    lp->gridDim.y = 1;
    lp->gridDim.z = 1;
    lp->groupDim.x = block;
    lp->groupDim.y = 1;
    lp->groupDim.z = 1;
#endif
    stream->lockopen_preKernelCommand();
//    *av = &stream->_av;
    lp->av = &stream->_av;
    lp->cf = new hc::completion_future;
//    lp->av = static_cast<void*>(av);
//    lp->cf = static_cast<void*>(malloc(sizeof(hc::completion_future)));
    return (stream);
}


//---
//Called after kernel finishes execution.
void ihipPostLaunchKernel(hipStream_t stream, grid_launch_parm &lp)
{
//    stream->lockclose_postKernelCommand(cf);
    stream->lockclose_postKernelCommand(*lp.cf);
    if (HIP_LAUNCH_BLOCKING) {
        tprintf(DB_SYNC, " stream:%p LAUNCH_BLOCKING for kernel completion\n", stream);
    }
}


//
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
        case hipErrorMemoryAllocation           : return "hipErrorMemoryAllocation";
        case hipErrorLaunchOutOfResources       : return "hipErrorLaunchOutOfResources";
        case hipErrorInvalidValue               : return "hipErrorInvalidValue";
        case hipErrorInvalidResourceHandle      : return "hipErrorInvalidResourceHandle";
        case hipErrorInvalidDevice              : return "hipErrorInvalidDevice";
        case hipErrorInvalidMemcpyDirection     : return "hipErrorInvalidMemcpyDirection";
        case hipErrorNoDevice                   : return "hipErrorNoDevice";
        case hipErrorNotReady                   : return "hipErrorNotReady";
        case hipErrorPeerAccessNotEnabled       : return "hipErrorPeerAccessNotEnabled";
        case hipErrorPeerAccessAlreadyEnabled   : return "hipErrorPeerAccessAlreadyEnabled";

        case hipErrorRuntimeMemory              : return "hipErrorRuntimeMemory";
        case hipErrorRuntimeOther               : return "hipErrorRuntimeOther";
        case hipErrorUnknown                    : return "hipErrorUnknown";
        case hipErrorTbd                        : return "hipErrorTbd";
        default                                 : return "hipErrorUnknown";
    };
};


void ihipSetTs(hipEvent_t e)
{
    ihipEvent_t *eh = e._handle;
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



// Resolve hipMemcpyDefault to a known type.
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


// Setup the copyCommandType and the copy agents (for hsa_amd_memory_async_copy)
// srcPhysAcc is the physical location of the src data.  For many copies this is the 
void ihipStream_t::setAsyncCopyAgents(unsigned kind, ihipCommand_t *commandType, hsa_agent_t *srcAgent, hsa_agent_t *dstAgent)
{
    // current* represents the device associated with the specified stream.
    ihipDevice_t *streamDevice = this->getDevice();
    hsa_agent_t streamAgent = streamDevice->_hsa_agent;

    // ROCR runtime logic is :
    // -    If both src and dst are cpu agent, launch thread and memcpy.  We want to avoid this.
    // -    If either/both src or dst is a gpu agent, use the first gpu agent’s DMA engine to perform the copy.

    switch (kind) {
        //case hipMemcpyHostToHost     : *commandType = ihipCommandCopyH2H; *srcAgent=streamAgent; *dstAgent=streamAgent; break;  // TODO - enable me, for async copy use SDMA.
        case hipMemcpyHostToHost     : *commandType = ihipCommandCopyH2H; *srcAgent=g_cpu_agent; *dstAgent=g_cpu_agent; break;
        case hipMemcpyHostToDevice   : *commandType = ihipCommandCopyH2D; *srcAgent=g_cpu_agent; *dstAgent=streamAgent; break;
        case hipMemcpyDeviceToHost   : *commandType = ihipCommandCopyD2H; *srcAgent=streamAgent; *dstAgent=g_cpu_agent; break;
        case hipMemcpyDeviceToDevice : *commandType = ihipCommandCopyD2D; *srcAgent=streamAgent; *dstAgent=streamAgent; break;
        default: throw ihipException(hipErrorInvalidMemcpyDirection);
    };
}


void ihipStream_t::copySync(LockedAccessor_StreamCrit_t &crit, void* dst, const void* src, size_t sizeBytes, unsigned kind)
{
    ihipDevice_t *device = this->getDevice();
    if (device == NULL) {
        throw ihipException(hipErrorInvalidDevice);
    }

    hc::accelerator acc;
    hc::AmPointerInfo dstPtrInfo(NULL, NULL, 0, acc, 0, 0);
    hc::AmPointerInfo srcPtrInfo(NULL, NULL, 0, acc, 0, 0);

    bool dstTracked = (hc::am_memtracker_getinfo(&dstPtrInfo, dst) == AM_SUCCESS);
    bool srcTracked = (hc::am_memtracker_getinfo(&srcPtrInfo, src) == AM_SUCCESS);
    bool srcInDeviceMem = srcPtrInfo._isInDeviceMem;
    bool dstInDeviceMem = dstPtrInfo._isInDeviceMem;

    // Resolve default to a specific Kind so we know which algorithm to use:
    if (kind == hipMemcpyDefault) {
        kind = resolveMemcpyDirection(srcTracked, dstTracked, srcInDeviceMem, dstInDeviceMem);
    };

    hsa_signal_t depSignal;

    bool copyEngineCanSeeSrcAndDest = false;
    if (kind == hipMemcpyDeviceToDevice) {
#if USE_PEER_TO_PEER>=2
        // TODO - consider refactor.  Do we need to support simul access of enable/disable peers with access?
        LockedAccessor_DeviceCrit_t  dcrit(device->criticalData());
        if (dcrit->isPeer(::getDevice(dstPtrInfo._appId)) && (dcrit->isPeer(::getDevice(srcPtrInfo._appId)))) {
            copyEngineCanSeeSrcAndDest = true;
        }
#endif
    }

    if (kind == hipMemcpyHostToDevice) {
        int depSignalCnt = preCopyCommand(crit, NULL, &depSignal, ihipCommandCopyH2D);
        if(!srcTracked){
            if (HIP_STAGING_BUFFERS) {
                tprintf(DB_COPY1, "D2H && !dstTracked: staged copy H2D dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);
                if(HIP_OPTIMAL_MEM_TRANSFER)
                {
                    if((device->isLargeBar)&&(sizeBytes < HIP_H2D_MEM_TRANSFER_THRESHOLD_DIRECT_OR_STAGING)){
                        memcpy(dst,src,sizeBytes);
                        std::atomic_thread_fence(std::memory_order_release);
                    }
                    else{
                        if(sizeBytes > HIP_H2D_MEM_TRANSFER_THRESHOLD_STAGING_OR_PININPLACE){
                        //if (HIP_PININPLACE) {
                            device->_staging_buffer[0]->CopyHostToDevicePinInPlace(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                        } else {
                            device->_staging_buffer[0]->CopyHostToDevice(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                        }
                        // The copy waits for inputs and then completes before returning so can reset queue to empty:
                        this->wait(crit, true);
                    }
                }
                else {
                    if (HIP_PININPLACE) {
                        device->_staging_buffer[0]->CopyHostToDevicePinInPlace(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                    } else {
                        device->_staging_buffer[0]->CopyHostToDevice(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                    }
               }
            }
            else {
                // TODO - remove, slow path.
                tprintf(DB_COPY1, "H2D && ! srcTracked: am_copy dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);
#if USE_AV_COPY
                _av.copy(src,dst,sizeBytes);
#else
                hc::am_copy(dst, src, sizeBytes);
#endif
            }
        } else {
            // This is H2D copy, and source is pinned host memory : we can copy directly w/o using staging buffer.
            hsa_agent_t dstAgent = *(static_cast<hsa_agent_t*>(dstPtrInfo._acc.get_hsa_agent()));
            hsa_agent_t srcAgent = *(static_cast<hsa_agent_t*>(srcPtrInfo._acc.get_hsa_agent()));

            ihipSignal_t *ihipSignal = allocSignal(crit);
            hsa_signal_t copyCompleteSignal = ihipSignal->_hsa_signal;

            hsa_signal_store_relaxed(copyCompleteSignal, 1);
            void *devPtrSrc = srcPtrInfo._devicePointer;
            tprintf(DB_COPY1, "HSA Async_copy dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);

            hsa_status_t hsa_status = hsa_amd_memory_async_copy(dst, dstAgent, devPtrSrc, g_cpu_agent, sizeBytes, depSignalCnt, depSignalCnt ? &depSignal:0x0, copyCompleteSignal);

        // This is sync copy, so let's wait for copy right here:
            if (hsa_status == HSA_STATUS_SUCCESS) {
                waitCopy(crit, ihipSignal); // wait for copy, and return to pool.
            } else {
                throw ihipException(hipErrorInvalidValue);
            }
        }
    } else if (kind == hipMemcpyDeviceToHost) {
        int depSignalCnt = preCopyCommand(crit, NULL, &depSignal, ihipCommandCopyD2H);
        if (!dstTracked){
            if (HIP_STAGING_BUFFERS) {
                tprintf(DB_COPY1, "D2H && !dstTracked: staged copy D2H dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);
                //printf ("staged-copy- read dep signals\n");
                if(HIP_OPTIMAL_MEM_TRANSFER)
                {
                    if(sizeBytes> HIP_D2H_MEM_TRANSFER_THRESHOLD){
                        device->_staging_buffer[1]->CopyDeviceToHostPinInPlace(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                    }else {
                         //printf ("staged-copy- read dep signals\n");
                         device->_staging_buffer[1]->CopyDeviceToHost(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                    }
                }else
                {
                    device->_staging_buffer[1]->CopyDeviceToHost(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                }
                // The copy completes before returning so can reset queue to empty:
                this->wait(crit, true);

            } else {
            // TODO - remove, slow path.
                tprintf(DB_COPY1, "D2H && !dstTracked: am_copy dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);
#if USE_AV_COPY
                _av.copy(src, dst, sizeBytes);
#else
                hc::am_copy(dst, src, sizeBytes);
#endif
            }
        } else {
            // This is D2H copy, and destination is pinned host memory : we can copy directly w/o using staging buffer.
            hsa_agent_t dstAgent = *(static_cast<hsa_agent_t*>(dstPtrInfo._acc.get_hsa_agent()));
            hsa_agent_t srcAgent = *(static_cast<hsa_agent_t*>(srcPtrInfo._acc.get_hsa_agent()));

            ihipSignal_t *ihipSignal = allocSignal(crit);
            hsa_signal_t copyCompleteSignal = ihipSignal->_hsa_signal;

            hsa_signal_store_relaxed(copyCompleteSignal, 1);
            void *devPtrDst = dstPtrInfo._devicePointer;
            tprintf(DB_COPY1, "HSA Async_copy dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);

            hsa_status_t hsa_status = hsa_amd_memory_async_copy(devPtrDst, g_cpu_agent, src, srcAgent, sizeBytes, depSignalCnt, depSignalCnt ? &depSignal:0x0, copyCompleteSignal);

        // This is sync copy, so let's wait for copy right here:
            if (hsa_status == HSA_STATUS_SUCCESS) {
                waitCopy(crit, ihipSignal); // wait for copy, and return to pool.
            } else {
                throw ihipException(hipErrorInvalidValue);
            }
        }
    } else if (kind == hipMemcpyHostToHost)  {
        int depSignalCnt = preCopyCommand(crit, NULL, &depSignal, ihipCommandCopyH2H);

        if (depSignalCnt) {
            // host waits before doing host memory copy.
            hsa_signal_wait_acquire(depSignal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
        }
        memcpy(dst, src, sizeBytes);
    } else if ((kind == hipMemcpyDeviceToDevice) && !copyEngineCanSeeSrcAndDest)  {
        int depSignalCnt = preCopyCommand(crit, NULL, &depSignal, ihipCommandCopyP2P);
        if (HIP_STAGING_BUFFERS) {
            tprintf(DB_COPY1, "P2P but engine can't see both pointers: staged copy P2P dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);
            //printf ("staged-copy- read dep signals\n");
            hsa_agent_t dstAgent = * (static_cast<hsa_agent_t*> (dstPtrInfo._acc.get_hsa_agent()));
            hsa_agent_t srcAgent = * (static_cast<hsa_agent_t*> (srcPtrInfo._acc.get_hsa_agent()));

            device->_staging_buffer[1]->CopyPeerToPeer(dst, dstAgent, src, srcAgent, sizeBytes, depSignalCnt ? &depSignal : NULL);

            // The copy completes before returning so can reset queue to empty:
            this->wait(crit, true);

        } else {
            assert(0); // currently no fallback for this path.
        } 
        
    } else {
        // If not special case - these can all be handled by the hsa async copy:
        ihipCommand_t commandType;
        hsa_agent_t srcAgent, dstAgent;
        setAsyncCopyAgents(kind, &commandType, &srcAgent, &dstAgent);

        int depSignalCnt = preCopyCommand(crit, NULL, &depSignal, commandType);

        // Get a completion signal:
        ihipSignal_t *ihipSignal = allocSignal(crit);
        hsa_signal_t copyCompleteSignal = ihipSignal->_hsa_signal;

        hsa_signal_store_relaxed(copyCompleteSignal, 1);

        tprintf(DB_COPY1, "HSA Async_copy dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);

        hsa_status_t hsa_status = hsa_amd_memory_async_copy(dst, dstAgent, src, srcAgent, sizeBytes, depSignalCnt, depSignalCnt ? &depSignal:0x0, copyCompleteSignal);

        // This is sync copy, so let's wait for copy right here:
        if (hsa_status == HSA_STATUS_SUCCESS) {
            waitCopy(crit, ihipSignal); // wait for copy, and return to pool.
        } else {
            throw ihipException(hipErrorInvalidValue);
        }
    }

}


// Sync copy that acquires lock:
void ihipStream_t::locked_copySync(void* dst, const void* src, size_t sizeBytes, unsigned kind)
{
    LockedAccessor_StreamCrit_t crit (_criticalData);
    copySync(crit, dst, src, sizeBytes, kind);
}



void ihipStream_t::copyAsync(void* dst, const void* src, size_t sizeBytes, unsigned kind)
{
    LockedAccessor_StreamCrit_t crit(_criticalData);

    ihipDevice_t *device = this->getDevice();

    if (device == NULL) {
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


        // "tracked" really indicates if the pointer's virtual address is available in the GPU address space.
        // If both pointers are not tracked, we need to fall back to a sync copy.
        if (!dstTracked || !srcTracked) {
            trueAsync = false;
        }

        if (kind == hipMemcpyDefault) {
            bool srcInDeviceMem = (srcTracked && srcPtrInfo._isInDeviceMem);
            bool dstInDeviceMem = (dstTracked && dstPtrInfo._isInDeviceMem);
            kind = resolveMemcpyDirection(srcTracked, dstTracked, srcPtrInfo._isInDeviceMem, dstPtrInfo._isInDeviceMem);
        }


        ihipSignal_t *ihip_signal = allocSignal(crit);
        hsa_signal_store_relaxed(ihip_signal->_hsa_signal, 1);


        if(trueAsync == true){

            ihipCommand_t commandType;
            hsa_agent_t srcAgent, dstAgent;
            setAsyncCopyAgents(kind, &commandType, &srcAgent, &dstAgent);

            hsa_signal_t depSignal;
            int depSignalCnt = preCopyCommand(crit, ihip_signal, &depSignal, commandType);

            tprintf (DB_SYNC, " copy-async, waitFor=%lu completion=#%lu(%lu)\n", depSignalCnt? depSignal.handle:0x0, ihip_signal->_sig_id, ihip_signal->_hsa_signal.handle);

            hsa_status_t hsa_status = hsa_amd_memory_async_copy(dst, dstAgent, src, srcAgent, sizeBytes, depSignalCnt, depSignalCnt ? &depSignal:0x0, ihip_signal->_hsa_signal);


            if (hsa_status == HSA_STATUS_SUCCESS) {
                if (HIP_LAUNCH_BLOCKING) {
                    tprintf(DB_SYNC, "LAUNCH_BLOCKING for completion of hipMemcpyAsync(%zu)\n", sizeBytes);
                    this->wait(crit);
                }
            } else {
                // This path can be hit if src or dst point to unpinned host memory.
                // TODO-stream - does async-copy fall back to sync if input pointers are not pinned?
                throw ihipException(hipErrorInvalidValue);
            }
        } else {
            copySync(crit, dst, src, sizeBytes, kind);
        }
    }
}

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// HCC-specific accessor functions:

/**
 * @return #hipSuccess, #hipErrorInvalidDevice
 */
//---
hipError_t hipHccGetAccelerator(int deviceId, hc::accelerator *acc)
{
    HIP_INIT_API(deviceId, acc);

    ihipDevice_t *d = ihipGetDevice(deviceId);
    hipError_t err;
    if (d == NULL) {
        err =  hipErrorInvalidDevice;
    } else {
        *acc = d->_acc;
        err = hipSuccess;
    }
    return ihipLogStatus(err);
}


/**
 * @return #hipSuccess
 */
//---
hipError_t hipHccGetAcceleratorView(hipStream_t stream, hc::accelerator_view **av)
{
    HIP_INIT_API(stream, av);

    if (stream == hipStreamNull ) {
        ihipDevice_t *device = ihipGetTlsDefaultDevice();
        stream = device->_default_stream;
    }

    *av = &(stream->_av);

    hipError_t err = hipSuccess;
    return ihipLogStatus(err);
}

// TODO - review signal / error reporting code.
// TODO - describe naming convention. ihip _.  No accessors.  No early returns from functions. Set status to success at top, only set error codes in implementation.  No tabs.
//        Caps convention _ or camelCase
//        if { }
//        Should use ihip* data structures inside code rather than app-facing hip.  For example, use ihipDevice_t (rather than hipDevice_t), ihipStream_t (rather than hipStream_t).
//        locked_
// TODO - describe MT strategy
//
//// TODO - add identifier numbers for streams and devices to help with debugging.

#if ONE_OBJECT_FILE
#include "staging_buffer.cpp"
#endif
