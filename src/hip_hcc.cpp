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
int HIP_STAGING_SIZE = 64;   /* size of staging buffers, in KB */
int HIP_STAGING_BUFFERS = 2;    // TODO - remove, two buffers should be enough.
int HIP_PININPLACE = 0;
int HIP_OPTIMAL_MEM_TRANSFER = 0; //ENV Variable to test different memory transfer logics
int HIP_H2D_MEM_TRANSFER_THRESHOLD_DIRECT_OR_STAGING = 0;
int HIP_H2D_MEM_TRANSFER_THRESHOLD_STAGING_OR_PININPLACE = 0;
int HIP_D2H_MEM_TRANSFER_THRESHOLD = 0;
int HIP_STREAM_SIGNALS = 32;  /* number of signals to allocate at stream creation */
int HIP_VISIBLE_DEVICES = 0; /* Contains a comma-separated sequence of GPU identifiers */


//---
// Chicken bits for disabling functionality to work around potential issues:
int HIP_DISABLE_HW_KERNEL_DEP = 0;
int HIP_DISABLE_HW_COPY_DEP = 0;





std::once_flag hip_initialized;

// Array of pointers to devices.
ihipDevice_t **g_deviceArray;


bool g_visible_device = false;
unsigned g_deviceCnt;
std::vector<int> g_hip_visible_devices;
hsa_agent_t g_cpu_agent;


// TODO, remove these if possible:
hsa_agent_t gpu_agent_;
hsa_amd_memory_pool_t gpu_pool_;

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
// ihipSignal_t:
//=================================================================================================
//
//---
ihipSignal_t::ihipSignal_t() :  _sigId(0)
{
    if (hsa_signal_create(0/*value*/, 0, NULL, &_hsaSignal) != HSA_STATUS_SUCCESS) {
        throw ihipException(hipErrorRuntimeMemory);
    }
    //tprintf (DB_SIGNAL, "  allocated hsa_signal=%lu\n", (_hsaSignal.handle));
}

//---
ihipSignal_t::~ihipSignal_t()
{
    tprintf (DB_SIGNAL, "  destroy hsa_signal #%lu (#%lu)\n", (_hsaSignal.handle), _sigId);
    if (hsa_signal_destroy(_hsaSignal) != HSA_STATUS_SUCCESS) {
       throw ihipException(hipErrorRuntimeOther);
    }
};



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
    SIGSEQNUM sigNum = signal->_sigId;
    tprintf(DB_SYNC, "waitCopy signal:#%lu\n", sigNum);

    hsa_signal_wait_acquire(signal->_hsaSignal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);


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
//        crit->_av.wait();
        waitOnAllCFs(crit);
    }

    if (crit->_last_copy_signal) {
        tprintf (DB_SYNC, "stream %p wait for lastCopy:#%lu...\n", this, lastCopySeqId(crit) );
        this->waitCopy(crit, crit->_last_copy_signal);
    }

    crit->_kernelCnt = 0;

    // Reset the stream to "empty" - next command will not set up an inpute dependency on any older signal.
    crit->_last_command_type = ihipCommandCopyH2D;
    crit->_last_copy_signal = NULL;
//    crit->_signalCnt = 0;
}

void ihipStream_t::addCFtoStream(LockedAccessor_StreamCrit_t &crit, hc::completion_future *cf)
{
    crit->_cfs.push_back(cf);
}

void ihipStream_t::waitOnAllCFs(LockedAccessor_StreamCrit_t &crit)
{
    for(uint32_t i=0;i<crit->_cfs.size();i++){
        if(crit->_cfs[i] != NULL){
            crit->_cfs[i]->wait();
            delete crit->_cfs[i];
        }
    }
    crit->_cfs.clear();
}

//---
//Wait for all kernel and data copy commands in this stream to complete.
void ihipStream_t::locked_wait(bool assertQueueEmpty)
{
    LockedAccessor_StreamCrit_t crit(_criticalData);

    wait(crit, assertQueueEmpty);

};

#if USE_AV_COPY
// Causes current stream to wait for specified event to complete:
void ihipStream_t::locked_waitEvent(hipEvent_t event)
{
    LockedAccessor_StreamCrit_t crit(_criticalData);

    // TODO - check state of event here:
    crit->_av.create_blocking_marker(event->_marker);
}
#endif

// Create a marker in this stream.
// Save state in the event so it can track the status of the event.
void ihipStream_t::locked_recordEvent(hipEvent_t event)
{
    // Lock the stream to prevent simultaneous access
    LockedAccessor_StreamCrit_t crit(_criticalData);

    event->_marker = crit->_av.create_marker();
    event->_copySeqId = lastCopySeqId(crit);
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


#define HIP_NUM_SIGNALS_PER_STREAM 32


//---
// Allocate a new signal from the signal pool.
// Returned signals have value of 0.
// Signals are intended for use in this stream and are always reclaimed "in-order".
ihipSignal_t *ihipStream_t::allocSignal(LockedAccessor_StreamCrit_t &crit)
{
    int numToScan = crit->_signalPool.size();
    crit->_signalCnt++;
    if(crit->_signalCnt == HIP_STREAM_SIGNALS){
        this->wait(crit);
        crit->_signalCnt = 0;
    }

    return &crit->_signalPool[crit->_signalCnt];

    do {
        auto thisCursor = crit->_signalCursor;

        if (++crit->_signalCursor == crit->_signalPool.size()) {
            crit->_signalCursor = 0;
        }

        if (crit->_signalPool[thisCursor]._sigId < crit->_oldest_live_sig_id) {
            SIGSEQNUM oldSigId = crit->_signalPool[thisCursor]._sigId;
            crit->_signalPool[thisCursor]._index = thisCursor;
            crit->_signalPool[thisCursor]._sigId  =  ++crit->_streamSigId;  // allocate it.
            tprintf(DB_SIGNAL, "allocatSignal #%lu at pos:%i (old sigId:%lu < oldest_live:%lu)\n",
                    crit->_signalPool[thisCursor]._sigId,
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

    barrier->dep_signal[0].handle = depSignal ? depSignal->_hsaSignal.handle: 0;

    barrier->completion_signal.handle = completionSignal ? completionSignal->_hsaSignal.handle : 0;

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
LockedAccessor_StreamCrit_t ihipStream_t::lockopen_preKernelCommand()
{
    LockedAccessor_StreamCrit_t crit(_criticalData, false/*no unlock at destruction*/);


    if(crit->_kernelCnt > HIP_NUM_KERNELS_INFLIGHT){
        this->wait(crit);
       crit->_kernelCnt = 0;
    }
    crit->_kernelCnt++;
    // If switching command types, we need to add a barrier packet to synchronize things.
    if (crit->_last_command_type != ihipCommandKernel) {
        if (crit->_last_copy_signal) {

            hsa_queue_t * q =  (hsa_queue_t*) (crit->_av.get_hsa_queue());
            if (HIP_DISABLE_HW_KERNEL_DEP == 0) {
                this->enqueueBarrier(q, crit->_last_copy_signal, NULL);
                tprintf (DB_SYNC, "stream %p switch %s to %s (barrier pkt inserted with wait on #%lu)\n",
                        this, ihipCommandName[crit->_last_command_type], ihipCommandName[ihipCommandKernel], crit->_last_copy_signal->_sigId)

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

    return crit;
}


//---
// Must be called after kernel finishes, this releases the lock on the stream so other commands can submit.
void ihipStream_t::lockclose_postKernelCommand(hc::completion_future &kernelFuture)
{
    // We locked _criticalData in the lockopen_preKernelCommand() so OK to access here:
    _criticalData._last_kernel_future = kernelFuture;

    if (HIP_LAUNCH_BLOCKING) {
        kernelFuture.wait();
        tprintf(DB_SYNC, " %s LAUNCH_BLOCKING for kernel completion\n", ToString(this).c_str());
    }

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
            hsa_signal_store_relaxed(depSignal->_hsaSignal,1);
            this->enqueueBarrier(static_cast<hsa_queue_t*>(crit->_av.get_hsa_queue()), NULL, depSignal);
            *waitSignal = depSignal->_hsaSignal;
        } else if (crit->_last_copy_signal) {
            needSync = 1;
            tprintf (DB_SYNC, "stream %p switch %s to %s (async copy dep on other copy #%lu)\n",
                    this, ihipCommandName[crit->_last_command_type], ihipCommandName[copyType], crit->_last_copy_signal->_sigId);
            *waitSignal = crit->_last_copy_signal->_hsaSignal;
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

    _stagingBuffer[0] = new UnpinnedCopyEngine(_hsaAgent,g_cpu_agent, HIP_STAGING_SIZE*1024, HIP_STAGING_BUFFERS,HIP_H2D_MEM_TRANSFER_THRESHOLD_DIRECT_OR_STAGING,HIP_H2D_MEM_TRANSFER_THRESHOLD_STAGING_OR_PININPLACE,HIP_D2H_MEM_TRANSFER_THRESHOLD);
    _stagingBuffer[1] = new UnpinnedCopyEngine(_hsaAgent,g_cpu_agent, HIP_STAGING_SIZE*1024, HIP_STAGING_BUFFERS,HIP_H2D_MEM_TRANSFER_THRESHOLD_DIRECT_OR_STAGING,HIP_H2D_MEM_TRANSFER_THRESHOLD_STAGING_OR_PININPLACE,HIP_D2H_MEM_TRANSFER_THRESHOLD);

    _primaryCtx = new ihipCtx_t(this, deviceCnt, hipDeviceMapHost);
}


ihipDevice_t::~ihipDevice_t()
{
    for (int i=0; i<2; i++) {
        if (_stagingBuffer[i]) {
            delete _stagingBuffer[i];
            _stagingBuffer[i] = NULL;
        }
    }
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

void FindDevicePool()
{
    hsa_status_t err = hsa_iterate_agents(FindGpuDevice, &gpu_agent_);
    ErrorCheck(err);

    err = hsa_amd_agent_iterate_memory_pools(gpu_agent_, GetDevicePool, &gpu_pool_);
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

    FindDevicePool();
    int access=checkAccess(g_cpu_agent, gpu_pool_);
    if (0!= access){
        _isLargeBar= 1;
    } else {
        _isLargeBar=0;
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

    READ_ENV_I(release, HIP_LAUNCH_BLOCKING, CUDA_LAUNCH_BLOCKING, "Make HIP APIs 'host-synchronous', so they block until any kernel launches or data copy commands complete. Alias: CUDA_LAUNCH_BLOCKING." );
    READ_ENV_I(release, HIP_DB, 0,  "Print various debug info.  Bitmask, see hip_hcc.cpp for more information.");
    if ((HIP_DB & (1<<DB_API))  && (HIP_TRACE_API == 0)) {
        // Set HIP_TRACE_API default before we read it, so it is printed correctly.
        HIP_TRACE_API = 1;
    }

    READ_ENV_I(release, HIP_TRACE_API, 0,  "Trace each HIP API call.  Print function name and return code to stderr as program executes.");
    READ_ENV_S(release, HIP_TRACE_API_COLOR, 0,  "Color to use for HIP_API.  None/Red/Green/Yellow/Blue/Magenta/Cyan/White");
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

// TODO - data-up to data-down:
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
    lp->cf = new hc::completion_future;
    stream->addCFtoStream(crit, lp->cf);
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
    lp->cf = new hc::completion_future;
    stream->addCFtoStream(crit, lp->cf);
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
    lp->cf = new hc::completion_future;
    stream->addCFtoStream(crit, lp->cf);
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
    lp->cf = new hc::completion_future; // TODO, is this necessary?

    stream->addCFtoStream(crit, lp->cf);

    ihipPrintKernelLaunch(kernelNameStr, lp, stream);
    return (stream);
}


//---
//Called after kernel finishes execution.
//This releases the lock on the stream.
void ihipPostLaunchKernel(hipStream_t stream, grid_launch_parm &lp)
{
    stream->lockclose_postKernelCommand(*(lp.cf));
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
    const ihipDevice_t *streamDevice = this->getDevice();
    hsa_agent_t streamAgent = streamDevice->_hsaAgent;

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
    bool srcInDeviceMem = srcPtrInfo._isInDeviceMem;
    bool dstInDeviceMem = dstPtrInfo._isInDeviceMem;

    // Resolve default to a specific Kind so we know which algorithm to use:
    if (kind == hipMemcpyDefault && resolveOn) {
        kind = resolveMemcpyDirection(srcTracked, dstTracked, srcInDeviceMem, dstInDeviceMem);
    };

    hsa_signal_t depSignal;

    bool copyEngineCanSeeSrcAndDest = false;
    if (kind == hipMemcpyDeviceToDevice) {
#if USE_PEER_TO_PEER>=2
        // Lock to prevent another thread from modifying peer list while we are trying to look at it.
        LockedAccessor_CtxCrit_t  dcrit(ctx->criticalData());
        // FIXME - this assumes peer access only from primary context.
        // Would need to change the tracker to store a void * parameter that we could map to the ctx where the pointer is allocated.
        if (dcrit->isPeer(ihipGetPrimaryCtx(dstPtrInfo._appId)) && (dcrit->isPeer(ihipGetPrimaryCtx(srcPtrInfo._appId)))) {
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
                    device->_stagingBuffer[0]->CopyHostToDevice(1,device->_isLargeBar,dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                }
                else {
                    if (HIP_PININPLACE) {
                        device->_stagingBuffer[0]->CopyHostToDevicePinInPlace(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                    } else {
                        device->_stagingBuffer[0]->CopyHostToDevice(0,0,dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                    }
               }
               // The copy waits for inputs and then completes before returning so can reset queue to empty:
               this->wait(crit, true);
            }
            else {
                // TODO - remove, slow path.
                tprintf(DB_COPY1, "H2D && ! srcTracked: am_copy dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);
#if USE_AV_COPY
                crit->_av.copy(src,dst,sizeBytes);
#else
                hc::am_copy(dst, src, sizeBytes);
#endif
            }
        } else {
            // This is H2D copy, and source is pinned host memory : we can copy directly w/o using staging buffer.
            hsa_agent_t dstAgent = *(static_cast<hsa_agent_t*>(dstPtrInfo._acc.get_hsa_agent()));
            hsa_agent_t srcAgent = *(static_cast<hsa_agent_t*>(srcPtrInfo._acc.get_hsa_agent()));

            ihipSignal_t *ihipSignal = allocSignal(crit);
            hsa_signal_t copyCompleteSignal = ihipSignal->_hsaSignal;

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
                    //printf ("staged-copy- read dep signals\n");
                    device->_stagingBuffer[1]->CopyDeviceToHost(1,dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                }
                else
                {
                    device->_stagingBuffer[1]->CopyDeviceToHost(0,dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                }
                // The copy completes before returning so can reset queue to empty:
                this->wait(crit, true);

            } else {
            // TODO - remove, slow path.
                tprintf(DB_COPY1, "D2H && !dstTracked: am_copy dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);
#if USE_AV_COPY
                crit->_av.copy(src, dst, sizeBytes);
#else
                hc::am_copy(dst, src, sizeBytes);
#endif
            }
        } else {
            // This is D2H copy, and destination is pinned host memory : we can copy directly w/o using staging buffer.
            hsa_agent_t dstAgent = *(static_cast<hsa_agent_t*>(dstPtrInfo._acc.get_hsa_agent()));
            hsa_agent_t srcAgent = *(static_cast<hsa_agent_t*>(srcPtrInfo._acc.get_hsa_agent()));

            ihipSignal_t *ihipSignal = allocSignal(crit);
            hsa_signal_t copyCompleteSignal = ihipSignal->_hsaSignal;

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

            device->_stagingBuffer[1]->CopyPeerToPeer(dst, dstAgent, src, srcAgent, sizeBytes, depSignalCnt ? &depSignal : NULL);

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
        hsa_signal_t copyCompleteSignal = ihipSignal->_hsaSignal;

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
        hsa_signal_store_relaxed(ihip_signal->_hsaSignal, 1);


        if(trueAsync == true){

            ihipCommand_t commandType;
            hsa_agent_t srcAgent, dstAgent;
            setAsyncCopyAgents(kind, &commandType, &srcAgent, &dstAgent);

            hsa_signal_t depSignal;
            int depSignalCnt = preCopyCommand(crit, ihip_signal, &depSignal, commandType);

            tprintf (DB_SYNC, " copy-async, waitFor=%lu completion=#%lu(%lu)\n", depSignalCnt? depSignal.handle:0x0, ihip_signal->_sigId, ihip_signal->_hsaSignal.handle);

            hsa_status_t hsa_status = hsa_amd_memory_async_copy(dst, dstAgent, src, srcAgent, sizeBytes, depSignalCnt, depSignalCnt ? &depSignal:0x0, ihip_signal->_hsaSignal);


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

// TODO - review signal / error reporting code.
// TODO - describe naming convention. ihip _.  No accessors.  No early returns from functions. Set status to success at top, only set error codes in implementation.  No tabs.
//        Caps convention _ or camelCase
//        if { }
//        Should use ihip* data structures inside code rather than app-facing hip.  For example, use ihipCtx_t (rather than hipDevice_t), ihipStream_t (rather than hipStream_t).
//        locked_
// TODO - describe MT strategy
//
//// TODO - add identifier numbers for streams and devices to help with debugging.

#if ONE_OBJECT_FILE
#include "unpinned_copy_engine.cpp"
#endif
