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

#include <thread>
#include <mutex>
#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Stream
//
#if defined(__HCC__) && (__hcc_minor__ < 3)
enum queue_priority
{
    priority_high = 0,
    priority_normal = 0,
    priority_low = 0
};
#else
enum queue_priority
{
    priority_high = Kalmar::priority_high,
    priority_normal = Kalmar::priority_normal,
    priority_low = Kalmar::priority_low
};
#endif

//---
hipError_t ihipStreamCreate(TlsData *tls, hipStream_t* stream, unsigned int flags, int priority) {
    ihipCtx_t* ctx = ihipGetTlsDefaultCtx();

    hipError_t e = hipSuccess;

    if (ctx) {
        if (HIP_FORCE_NULL_STREAM) {
            *stream = 0;
        } else if( NULL == stream ){
            e = hipErrorInvalidValue;
        } else {
            hc::accelerator acc = ctx->getWriteableDevice()->_acc;

            // TODO - se try-catch loop to detect memory exception?
            //
            // Note this is an execute_any_order queue, 
            // CUDA stream behavior is that all kernels submitted will automatically
            // wait for prev to complete, this behaviour will be mainatined by 
            // hipModuleLaunchKernel. execute_any_order will help 
	    // hipExtModuleLaunchKernel , which uses a special flag

            {
                // Obtain mutex access to the device critical data, release by destructor
                LockedAccessor_CtxCrit_t ctxCrit(ctx->criticalData());

#if defined(__HCC__) && (__hcc_minor__ < 3)
                auto istream = new ihipStream_t(ctx, acc.create_view(), flags);
#else
                auto istream = new ihipStream_t(ctx, acc.create_view(Kalmar::execute_any_order, Kalmar::queuing_mode_automatic, (Kalmar::queue_priority)priority), flags);
#endif

                ctxCrit->addStream(istream);
                *stream = istream;
            }
            tprintf(DB_SYNC, "hipStreamCreate, %s\n", ToString(*stream).c_str());
        }

    } else {
        e = hipErrorInvalidDevice;
    }

    return e;
}


//---
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
    HIP_INIT_API(hipStreamCreateWithFlags, stream, flags);
    if(flags == hipStreamDefault || flags == hipStreamNonBlocking)
        return ihipLogStatus(ihipStreamCreate(tls, stream, flags, priority_normal));
    else
        return ihipLogStatus(hipErrorInvalidValue);
}

//---
hipError_t hipStreamCreate(hipStream_t* stream) {
    HIP_INIT_API(hipStreamCreate, stream);

    return ihipLogStatus(ihipStreamCreate(tls, stream, hipStreamDefault, priority_normal));
}

//---
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
    HIP_INIT_API(hipStreamCreateWithPriority, stream, flags, priority);

    // clamp priority to range [priority_high:priority_low]
    priority = (priority < priority_high ? priority_high : (priority > priority_low ? priority_low : priority));
    return ihipLogStatus(ihipStreamCreate(tls, stream, flags, priority));
}

//---
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    HIP_INIT_API(hipDeviceGetStreamPriorityRange, leastPriority, greatestPriority);

    if (leastPriority != NULL) *leastPriority = priority_low;
    if (greatestPriority != NULL) *greatestPriority = priority_high;
    return ihipLogStatus(hipSuccess);
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
    HIP_INIT_SPECIAL_API(hipStreamWaitEvent, TRACE_SYNC, stream, event, flags);

    hipError_t e = hipSuccess;

    if (event == nullptr) {
        e = hipErrorInvalidResourceHandle;

    } else {
        auto ecd = event->locked_copyCrit(); 
        if ((ecd._state != hipEventStatusUnitialized) && (ecd._state != hipEventStatusCreated)) {
            if (HIP_SYNC_STREAM_WAIT || (HIP_SYNC_NULL_STREAM && (stream == 0))) {
                // conservative wait on host for the specified event to complete:
                // return _stream->locked_eventWaitComplete(this, waitMode);
                //
                ecd.marker().wait((event->_flags & hipEventBlockingSync) ? hc::hcWaitModeBlocked
                                                                         : hc::hcWaitModeActive);
            } else {
                stream = ihipSyncAndResolveStream(stream);
                // This will use create_blocking_marker to wait on the specified queue.
                stream->locked_streamWaitEvent(ecd);
            }
        }
    }  // else event not recorded, return immediately and don't create marker.

    return ihipLogStatus(e);
};


//---
hipError_t hipStreamQuery(hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipStreamQuery, TRACE_QUERY, stream);

    // Use default stream if 0 specified:
    if (stream == hipStreamNull) {
        ihipCtx_t* device = ihipGetTlsDefaultCtx();
        stream = device->_defaultStream;
    }

    bool isEmpty = 0;

    {
        LockedAccessor_StreamCrit_t crit(stream->_criticalData);
        isEmpty = crit->_av.get_is_empty();
    }

    hipError_t e = isEmpty ? hipSuccess : hipErrorNotReady;

    return ihipLogStatus(e);
}


//---
hipError_t hipStreamSynchronize(hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipStreamSynchronize, TRACE_SYNC, stream);

    return ihipLogStatus(ihipStreamSynchronize(tls, stream));
}


//---
/**
 * @return #hipSuccess, #hipErrorInvalidResourceHandle
 */
hipError_t hipStreamDestroy(hipStream_t stream) {
    HIP_INIT_API(hipStreamDestroy, stream);

    hipError_t e = hipSuccess;

    //--- Drain the stream:
    if (stream == NULL) {
        if (!HIP_FORCE_NULL_STREAM) {
            e = hipErrorInvalidResourceHandle;
        }
    } else {
        stream->locked_wait();

        ihipCtx_t* ctx = stream->getCtx();

        if (ctx) {
            ctx->locked_removeStream(stream);
            delete stream;
        } else {
            e = hipErrorInvalidResourceHandle;
        }
    }

    return ihipLogStatus(e);
}


//---
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
    HIP_INIT_API(hipStreamGetFlags, stream, flags);

    if (flags == NULL) {
        return ihipLogStatus(hipErrorInvalidValue);
    } else if (stream == hipStreamNull) {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    } else {
        *flags = stream->_flags;
        return ihipLogStatus(hipSuccess);
    }
}


//--
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
    HIP_INIT_API(hipStreamGetPriority, stream, priority);

    if (priority == NULL) {
        return ihipLogStatus(hipErrorInvalidValue);
    } else if (stream == hipStreamNull) {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    } else {
#if defined(__HCC__) && (__hcc_minor__ < 3)
        *priority = 0;
#else
        LockedAccessor_StreamCrit_t crit(stream->criticalData());
        *priority = crit->_av.get_queue_priority();
#endif
        return ihipLogStatus(hipSuccess);
    }
}


//---
hipError_t setCallbackPacket(hsa_queue_t* queue,
                             uint64_t& index, uint64_t& nextIndex,
                             hsa_barrier_and_packet_t** barrier1,
                             hsa_barrier_and_packet_t** barrier2){

    if(queue == nullptr ||
       barrier1 == nullptr || barrier2 == nullptr)
            return hipErrorInvalidValue;

    uint64_t tempIndex = 0;
    uint32_t mask = queue->size - 1;
    hsa_barrier_and_packet_t* tempBarrier;

    // Check for empty packets
    do{
        tempIndex = hsa_queue_load_write_index_scacquire(queue);
        tempBarrier = &(((hsa_barrier_and_packet_t*)(queue->base_address))[tempIndex & mask]);
    }while(!(tempBarrier->header & HSA_PACKET_TYPE_INVALID));

    // Reserve two packets for two barriers
    index = hsa_queue_add_write_index_scacquire(queue, 2);

    if(index > mask)
    {
        index = 0;
        nextIndex = 1;
    }
    else if(index == mask)
        nextIndex = 0;
    else
        nextIndex = index + 1;

    tempBarrier = new hsa_barrier_and_packet_t;
    memset(tempBarrier, 0, sizeof(hsa_barrier_and_packet_t));
    tempBarrier->header = HSA_PACKET_TYPE_INVALID;

    // Barrier 1
    *barrier1 = &(((hsa_barrier_and_packet_t*)(queue->base_address))[index & mask]);
    memcpy(*barrier1,tempBarrier,sizeof(hsa_barrier_and_packet_t));

    // Barrier 2
    *barrier2 = &(((hsa_barrier_and_packet_t*)(queue->base_address))[nextIndex & mask]);
    memcpy(*barrier2,tempBarrier,sizeof(hsa_barrier_and_packet_t));

    delete tempBarrier;

    return hipSuccess;
}

hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags) {

    HIP_INIT_API(hipStreamAddCallback, stream, callback, userData, flags);
    hipError_t e = hipSuccess;

    if(stream == hipStreamNull)
    {
        ihipCtx_t* device = ihipGetTlsDefaultCtx();
        stream = device->_defaultStream;
    }

    // 1. Lock the queue
    hsa_queue_t* lockedQ = static_cast<hsa_queue_t*> (stream->criticalData()._av.acquire_locked_hsa_queue());

    if(lockedQ == nullptr)
    {
        // No queue attached to stream hence exiting early
        return ihipLogStatus(hipErrorMissingConfiguration);
    }

    // 2. Allocate a singals
    hsa_signal_t signal;
    hsa_status_t status = hsa_signal_create(1, 0, NULL, &signal);

    if(status != HSA_STATUS_SUCCESS)
    {
        return ihipLogStatus(hipErrorInvalidValue);
    }

    hsa_signal_t depSignal;
    status = hsa_signal_create(1, 0, NULL, &depSignal);

    if(status != HSA_STATUS_SUCCESS)
    {
        return ihipLogStatus(hipErrorInvalidValue);
    }

    // 3. Store callback details
    ihipStreamCallback_t* cb = new ihipStreamCallback_t(stream, callback, userData);
    if(cb == nullptr)
    {
        return ihipLogStatus(hipErrorMemoryAllocation);
    }
    cb->_signal = depSignal;

    // 4. Create barrier packets
    uint64_t index ;
    uint64_t nextIndex;

    hsa_barrier_and_packet_t* barrier;
    hsa_barrier_and_packet_t* depBarrier;

    setCallbackPacket(lockedQ, index, nextIndex, &barrier, &depBarrier);

    barrier->completion_signal = signal;

    depBarrier->dep_signal[0] = depSignal;

    uint16_t header = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE)| 1 << HSA_PACKET_HEADER_BARRIER;

    // 5. Update packet header,
    // Intentionally updated second barrier header before first in order to avoid race
    depBarrier->header = header;
    barrier->header = header;

    // 6. Trigger the doorbell
    nextIndex = nextIndex + 1;
    hsa_queue_store_write_index_screlease(lockedQ, nextIndex);
    hsa_signal_store_relaxed(lockedQ->doorbell_signal, index+1);

    // 7. Release queue
    stream->criticalData()._av.release_locked_hsa_queue();

    // 8. Register signal callback
    hsa_amd_signal_async_handler(signal, HSA_SIGNAL_CONDITION_EQ, 0, ihipStreamCallbackHandler, cb);

    return ihipLogStatus(e);
}
