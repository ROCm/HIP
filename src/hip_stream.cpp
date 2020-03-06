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
#if defined(__HCC__) && (__hcc_major__ < 3) && (__hcc_minor__ < 3)
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

#if defined(__HCC__) && (__hcc_major__ < 3) && (__hcc_minor__ < 3)
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
        e = hipErrorInvalidHandle;

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
        isEmpty = stream->is_empty(crit);
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
 * @return #hipSuccess, #hipErrorInvalidHandle
 */
hipError_t hipStreamDestroy(hipStream_t stream) {
    HIP_INIT_API(hipStreamDestroy, stream);

    hipError_t e = hipSuccess;

    //--- Drain the stream:
    if (stream == NULL) {
        if (!HIP_FORCE_NULL_STREAM) {
            e = hipErrorInvalidHandle;
        }
    } else {
        stream->locked_wait();

        ihipCtx_t* ctx = stream->getCtx();

        if (ctx) {
            ctx->locked_removeStream(stream);
            delete stream;
        } else {
            e = hipErrorInvalidHandle;
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
        return ihipLogStatus(hipErrorInvalidHandle);
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
        return ihipLogStatus(hipErrorInvalidHandle);
    } else {
#if defined(__HCC__) && (__hcc_major__ < 3) && (__hcc_minor__ < 3)
        *priority = 0;
#else
        LockedAccessor_StreamCrit_t crit(stream->criticalData());
        *priority = crit->_av.get_queue_priority();
#endif
        return ihipLogStatus(hipSuccess);
    }
}


//---
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags) {
    HIP_INIT_API(hipStreamAddCallback, stream, callback, userData, flags);

    auto stream_original{stream};
    stream = ihipSyncAndResolveStream(stream);

    if (!stream) return hipErrorInvalidValue;

    LockedAccessor_StreamCrit_t cs{stream->criticalData()};

    // stream is locked, now lock underlying HSA queue using unique_ptr RAII
    auto av{cs->_av};
    auto d{[=](hsa_queue_t*) mutable { av.release_locked_hsa_queue(); }};
    std::unique_ptr<hsa_queue_t, decltype(d)> q{
        static_cast<hsa_queue_t*>(av.acquire_locked_hsa_queue()),
        std::move(d)};

    // get first packet
    auto b_index{hsa_queue_add_write_index_relaxed(q.get(), 1) % q->size};
    auto b{static_cast<hsa_barrier_or_packet_t*>(q->base_address) + b_index};
    assert(b->header == HSA_PACKET_TYPE_INVALID);

    // get second packet
    auto c_index{hsa_queue_add_write_index_relaxed(q.get(), 1) % q->size};
    auto c{static_cast<hsa_barrier_or_packet_t*>(q->base_address) + c_index};
    assert(c->header == HSA_PACKET_TYPE_INVALID);

    // busy wait for rooom in the HSA queue; this should be rare
    // TODO exponential backoff, limit number of attempts to indicate an error
    while(c_index - hsa_queue_load_read_index_scacquire(q.get()) >= q->size);

    // zero out all but the packet headers, which should be HSA_PACKET_TYPE_INVALID
    static constexpr size_t aql_header_size = 16;
    static constexpr size_t memset_size = sizeof(hsa_barrier_or_packet_t) - aql_header_size;
    memset(reinterpret_cast<char*>(b)+aql_header_size, 0, memset_size);
    memset(reinterpret_cast<char*>(c)+aql_header_size, 0, memset_size);

    // create signal in first packet, initialized to 2
    hsa_signal_create(2, 0, nullptr, static_cast<hsa_signal_t*>(&b->completion_signal));

    // append signal to stream so we can cleanup later
    cs->_pending_callbacks.push_back(b->completion_signal);

    // second packet depends on first packet's signal reaching 0
    c->dep_signal[0] = b->completion_signal;

    // create callback that can be passed to hsa_amd_signal_async_handler
    // this function will call the user's callback, then sets first packet's signal to 0 to indicate completion
    auto sgn{b->completion_signal};
    auto t{new std::function<void()>{[=]() {
        callback(stream_original, hipSuccess, userData);
        hsa_signal_store_relaxed(sgn, 0);
    }}};

    // register above callback with HSA runtime to be called when first packet's signal
    // is decremented from 2 to 1 by CP
    hsa_amd_signal_async_handler(b->completion_signal, HSA_SIGNAL_CONDITION_EQ, 1,
        [](hsa_signal_value_t x, void* p) {
            (*static_cast<decltype(t)>(p))();
            delete static_cast<decltype(t)>(p);
            return false;
        }, t);

    // atomically write the headers of both packets into the HSA queue
    __atomic_store_n(
            &b->header,
            (HSA_PACKET_TYPE_BARRIER_OR << HSA_PACKET_HEADER_TYPE) |
            (1 << HSA_PACKET_HEADER_BARRIER) |
            (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
            (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE),
            __ATOMIC_RELAXED);
    __atomic_store_n(
            &c->header,
            (HSA_PACKET_TYPE_BARRIER_OR << HSA_PACKET_HEADER_TYPE) |
            (1 << HSA_PACKET_HEADER_BARRIER) |
            (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
            (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE),
            __ATOMIC_RELAXED);

    // ring queue's doorbell
    hsa_signal_store_relaxed(q->doorbell_signal, c_index);

    return ihipLogStatus(hipSuccess);
}
