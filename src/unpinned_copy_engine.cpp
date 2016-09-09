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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hc_am.hpp>

#include "hsa_ext_amd.h"

#include "hcc_detail/unpinned_copy_engine.h"

#ifdef HIP_HCC
#include "hcc_detail/hip_runtime.h"
#include "hcc_detail/hip_hcc.h"
#define THROW_ERROR(e) throw ihipException(e)
#else
#define THROW_ERROR(e) throw
#define tprintf(trace_level, ...)
#endif

void errorCheck(hsa_status_t hsa_error_code, int line_num, std::string str) {
  if ((hsa_error_code != HSA_STATUS_SUCCESS)&& (hsa_error_code != HSA_STATUS_INFO_BREAK))  {
    printf("HSA reported error!\n In file: %s\nAt line: %d\n", str.c_str(),line_num);
  }
}

#define ErrorCheck(x) errorCheck(x, __LINE__, __FILE__)
hsa_amd_memory_pool_t sys_pool_;

hsa_status_t findGlobalPool(hsa_amd_memory_pool_t pool, void* data) {
    if (NULL == data) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    hsa_status_t err;
    hsa_amd_segment_t segment;
    uint32_t flag;
    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    ErrorCheck(err);

    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
    ErrorCheck(err);
    if ((HSA_AMD_SEGMENT_GLOBAL == segment) &&
        (flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)) {
        *((hsa_amd_memory_pool_t*)data) = pool;
    }
    return HSA_STATUS_SUCCESS;
}

//-------------------------------------------------------------------------------------------------
UnpinnedCopyEngine::UnpinnedCopyEngine(hsa_agent_t hsaAgent, hsa_agent_t cpuAgent, size_t bufferSize, int numBuffers, int thresholdH2DDirectStaging,int thresholdH2DStagingPinInPlace,int thresholdD2H) :
    _hsaAgent(hsaAgent),
    _cpuAgent(cpuAgent),
    _bufferSize(bufferSize),
    _numBuffers(numBuffers > _max_buffers ? _max_buffers : numBuffers),
    _hipH2DTransferThresholdDirectOrStaging(thresholdH2DDirectStaging),
    _hipH2DTransferThresholdStagingOrPininplace(thresholdH2DStagingPinInPlace),
    _hipD2HTransferThreshold(thresholdD2H)
{
    hsa_status_t err = hsa_amd_agent_iterate_memory_pools(_cpuAgent, findGlobalPool, &sys_pool_);
    ErrorCheck(err);
    for (int i=0; i<_numBuffers; i++) {
        // TODO - experiment with alignment here.
        err = hsa_amd_memory_pool_allocate(sys_pool_, _bufferSize, 0, (void**)(&_pinnedStagingBuffer[i]));
        ErrorCheck(err);

        if ((err != HSA_STATUS_SUCCESS) || (_pinnedStagingBuffer[i] == NULL)) {
            THROW_ERROR(hipErrorMemoryAllocation);
        }

        err = hsa_amd_agents_allow_access(1, &hsaAgent, NULL, _pinnedStagingBuffer[i]);
        ErrorCheck(err);

        hsa_signal_create(0, 0, NULL, &_completionSignal[i]);
        hsa_signal_create(0, 0, NULL, &_completionSignal2[i]);
    }

};


//---
UnpinnedCopyEngine::~UnpinnedCopyEngine()
{
    for (int i=0; i<_numBuffers; i++) {
        if (_pinnedStagingBuffer[i]) {
            hsa_amd_memory_pool_free(_pinnedStagingBuffer[i]);
            _pinnedStagingBuffer[i] = NULL;
        }
        hsa_signal_destroy(_completionSignal[i]);
        hsa_signal_destroy(_completionSignal2[i]);
    }
}



//---
//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from host CPU.
//IN: src - src pointer for copy.  Must be accessible from agent this buffer is associated with (via _hsaAgent)
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void UnpinnedCopyEngine::CopyHostToDevicePinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    std::lock_guard<std::mutex> l (_copyLock);

    const char *srcp = static_cast<const char*> (src);
    char *dstp = static_cast<char*> (dst);

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completionSignal[i], 0);
    }

    if (sizeBytes >= UINT64_MAX/2) {
        THROW_ERROR (hipErrorInvalidValue);
    }
    int bufferIndex = 0;

    size_t theseBytes= sizeBytes;
    //tprintf (DB_COPY2, "H2D: waiting... on completion signal handle=%lu\n", _completionSignal[bufferIndex].handle);
    //hsa_signal_wait_acquire(_completionSignal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

    //void * masked_srcp = (void*) ((uintptr_t)srcp & (uintptr_t)(~0x3f)) ; // TODO
    void *locked_srcp;
    //hsa_status_t hsa_status = hsa_amd_memory_lock(masked_srcp, theseBytes, &_hsaAgent, 1, &locked_srcp);
    hsa_status_t hsa_status = hsa_amd_memory_lock(const_cast<char*> (srcp), theseBytes, &_hsaAgent, 1, &locked_srcp);
    //tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: pin-in-place:%p+%zu bufferIndex[%d]\n", bytesRemaining, srcp, theseBytes, bufferIndex);
    //printf ("status=%x srcp=%p, masked_srcp=%p, locked_srcp=%p\n", hsa_status, srcp, masked_srcp, locked_srcp);

    if (hsa_status != HSA_STATUS_SUCCESS) {
        THROW_ERROR (hipErrorRuntimeMemory);
    }

    hsa_signal_store_relaxed(_completionSignal[bufferIndex], 1);

    hsa_status = hsa_amd_memory_async_copy(dstp, _hsaAgent, locked_srcp, _cpuAgent, theseBytes, waitFor ? 1:0, waitFor, _completionSignal[bufferIndex]);
    //tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: async_copy %zu bytes %p to %p status=%x\n", bytesRemaining, theseBytes, _pinnedStagingBuffer[bufferIndex], dstp, hsa_status);

    if (hsa_status != HSA_STATUS_SUCCESS) {
        THROW_ERROR (hipErrorRuntimeMemory);
    }
    tprintf (DB_COPY2, "H2D: waiting... on completion signal handle=%lu\n", _completionSignal[bufferIndex].handle);
    hsa_signal_wait_acquire(_completionSignal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    hsa_amd_memory_unlock(const_cast<char*> (srcp));
    // Assume subsequent commands are dependent on previous and don't need dependency after first copy submitted, HIP_ONESHOT_COPY_DEP=1
    waitFor = NULL;
}


// Copy using simple memcpy.  Only works on large-bar systems.
void UnpinnedCopyEngine::CopyHostToDeviceMemcpy(int isLargeBar, void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    if (!isLargeBar) {
        THROW_ERROR (hipErrorInvalidValue);
    }

    memcpy(dst,src,sizeBytes);
    std::atomic_thread_fence(std::memory_order_release);
};



void UnpinnedCopyEngine::CopyHostToDevice(UnpinnedCopyEngine::CopyMode copyMode, int isLargeBar,void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    if (copyMode == ChooseBest) {
        if (isLargeBar && (sizeBytes < _hipH2DTransferThresholdDirectOrStaging)) {
            copyMode = UseMemcpy;
        } else if (sizeBytes > _hipH2DTransferThresholdStagingOrPininplace) {
            copyMode = UsePinInPlace;
        } else {
            copyMode = UseStaging;
        }
    }

    if (copyMode == UseMemcpy) {



	} else if (copyMode == UsePinInPlace) {
        CopyHostToDevicePinInPlace(dst, src, sizeBytes, waitFor);

	} else if (copyMode == UseStaging) {
        CopyHostToDeviceStaging(dst, src, sizeBytes, waitFor);

    } else {
        // Unknown copy mode.
        THROW_ERROR(hipErrorInvalidValue);
    }
}


//---
//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from host CPU.
//IN: src - src pointer for copy.  Must be accessible from agent this buffer is associated with (via _hsaAgent)
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void UnpinnedCopyEngine::CopyHostToDeviceStaging(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
	{
        std::lock_guard<std::mutex> l (_copyLock);

        const char *srcp = static_cast<const char*> (src);
        char *dstp = static_cast<char*> (dst);

        for (int i=0; i<_numBuffers; i++) {
            hsa_signal_store_relaxed(_completionSignal[i], 0);
        }

        if (sizeBytes >= UINT64_MAX/2) {
            THROW_ERROR (hipErrorInvalidValue);
        }
        int bufferIndex = 0;
        for (int64_t bytesRemaining=sizeBytes; bytesRemaining>0 ;  bytesRemaining -= _bufferSize) {

            size_t theseBytes = (bytesRemaining > _bufferSize) ? _bufferSize : bytesRemaining;

            tprintf (DB_COPY2, "H2D: waiting... on completion signal handle=%lu\n", _completionSignal[bufferIndex].handle);
            hsa_signal_wait_acquire(_completionSignal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

            tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: copy %zu bytes %p to stagingBuf[%d]:%p\n", bytesRemaining, theseBytes, srcp, bufferIndex, _pinnedStagingBuffer[bufferIndex]);
            // TODO - use uncached memcpy, someday.
            memcpy(_pinnedStagingBuffer[bufferIndex], srcp, theseBytes);


            hsa_signal_store_relaxed(_completionSignal[bufferIndex], 1);
            hsa_status_t hsa_status = hsa_amd_memory_async_copy(dstp, _hsaAgent, _pinnedStagingBuffer[bufferIndex], _cpuAgent, theseBytes, waitFor ? 1:0, waitFor, _completionSignal[bufferIndex]);
            tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: async_copy %zu bytes %p to %p status=%x\n", bytesRemaining, theseBytes, _pinnedStagingBuffer[bufferIndex], dstp, hsa_status);
            if (hsa_status != HSA_STATUS_SUCCESS) {
                THROW_ERROR ((hipErrorRuntimeMemory));
            }

            srcp += theseBytes;
            dstp += theseBytes;
            if (++bufferIndex >= _numBuffers) {
                bufferIndex = 0;
            }

            // Assume subsequent commands are dependent on previous and don't need dependency after first copy submitted, HIP_ONESHOT_COPY_DEP=1
            waitFor = NULL;
        }


        for (int i=0; i<_numBuffers; i++) {
            hsa_signal_wait_acquire(_completionSignal[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
        }
	}
}


void UnpinnedCopyEngine::CopyDeviceToHostPinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    std::lock_guard<std::mutex> l (_copyLock);

    const char *srcp = static_cast<const char*> (src);
    char *dstp = static_cast<char*> (dst);

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completionSignal[i], 0);
    }

    if (sizeBytes >= UINT64_MAX/2) {
        THROW_ERROR (hipErrorInvalidValue);
    }
    int bufferIndex = 0;
    size_t theseBytes= sizeBytes;
    void *locked_destp;

    hsa_status_t hsa_status = hsa_amd_memory_lock(const_cast<char*> (dstp), theseBytes, &_hsaAgent, 1, &locked_destp);


    if (hsa_status != HSA_STATUS_SUCCESS) {
        THROW_ERROR (hipErrorRuntimeMemory);
    }

    hsa_signal_store_relaxed(_completionSignal[bufferIndex], 1);

    hsa_status = hsa_amd_memory_async_copy(locked_destp,_cpuAgent , srcp, _hsaAgent, theseBytes, waitFor ? 1:0, waitFor, _completionSignal[bufferIndex]);

    if (hsa_status != HSA_STATUS_SUCCESS) {
        THROW_ERROR (hipErrorRuntimeMemory);
    }
    tprintf (DB_COPY2, "D2H: waiting... on completion signal handle=%lu\n", _completionSignal[bufferIndex].handle);
    hsa_signal_wait_acquire(_completionSignal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    hsa_amd_memory_unlock(const_cast<char*> (dstp));

    // Assume subsequent commands are dependent on previous and don't need dependency after first copy submitted, HIP_ONESHOT_COPY_DEP=1
    waitFor = NULL;
}


void UnpinnedCopyEngine::CopyDeviceToHost(CopyMode copyMode ,void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    if (copyMode == ChooseBest) {
        if (sizeBytes > _hipD2HTransferThreshold) {
            copyMode = UsePinInPlace;
        } else {
            copyMode = UseStaging;
        }
    }


	if (copyMode == UsePinInPlace) {
        CopyDeviceToHostPinInPlace(dst, src, sizeBytes, waitFor);
    } if (copyMode == UseStaging) { 
        CopyDeviceToHostStaging(dst, src, sizeBytes, waitFor);
    } else {
        // Unknown copy mode.
        THROW_ERROR(hipErrorInvalidValue);
    }
}

//---
//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from agent this buffer is associated with (via _hsaAgent).
//IN: src - src pointer for copy.  Must be accessible from host CPU.
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void UnpinnedCopyEngine::CopyDeviceToHostStaging(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    {
        std::lock_guard<std::mutex> l (_copyLock);

        const char *srcp0 = static_cast<const char*> (src);
        char *dstp1 = static_cast<char*> (dst);

        for (int i=0; i<_numBuffers; i++) {
            hsa_signal_store_relaxed(_completionSignal[i], 0);
        }

        if (sizeBytes >= UINT64_MAX/2) {
            THROW_ERROR (hipErrorInvalidValue);
        }

        int64_t bytesRemaining0 = sizeBytes; // bytes to copy from dest into staging buffer.
        int64_t bytesRemaining1 = sizeBytes; // bytes to copy from staging buffer into final dest

        while (bytesRemaining1 > 0)
        {
            // First launch the async copies to copy from dest to host
            for (int bufferIndex = 0; (bytesRemaining0>0) && (bufferIndex < _numBuffers);  bytesRemaining0 -= _bufferSize, bufferIndex++) {

                size_t theseBytes = (bytesRemaining0 > _bufferSize) ? _bufferSize : bytesRemaining0;

                tprintf (DB_COPY2, "D2H: bytesRemaining0=%zu  async_copy %zu bytes src:%p to staging:%p\n", bytesRemaining0, theseBytes, srcp0, _pinnedStagingBuffer[bufferIndex]);
                hsa_signal_store_relaxed(_completionSignal[bufferIndex], 1);
                hsa_status_t hsa_status = hsa_amd_memory_async_copy(_pinnedStagingBuffer[bufferIndex], _cpuAgent, srcp0, _hsaAgent, theseBytes, waitFor ? 1:0, waitFor, _completionSignal[bufferIndex]);
                if (hsa_status != HSA_STATUS_SUCCESS) {
                    THROW_ERROR (hipErrorRuntimeMemory);
                }

                srcp0 += theseBytes;


                // Assume subsequent commands are dependent on previous and don't need dependency after first copy submitted, HIP_ONESHOT_COPY_DEP=1
                waitFor = NULL;
            }

            // Now unload the staging buffers:
            for (int bufferIndex=0; (bytesRemaining1>0) && (bufferIndex < _numBuffers);  bytesRemaining1 -= _bufferSize, bufferIndex++) {

                size_t theseBytes = (bytesRemaining1 > _bufferSize) ? _bufferSize : bytesRemaining1;

                tprintf (DB_COPY2, "D2H: wait_completion[%d] bytesRemaining=%zu\n", bufferIndex, bytesRemaining1);
                hsa_signal_wait_acquire(_completionSignal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

                tprintf (DB_COPY2, "D2H: bytesRemaining1=%zu copy %zu bytes stagingBuf[%d]:%p to dst:%p\n", bytesRemaining1, theseBytes, bufferIndex, _pinnedStagingBuffer[bufferIndex], dstp1);
                memcpy(dstp1, _pinnedStagingBuffer[bufferIndex], theseBytes);

                dstp1 += theseBytes;
            }
		}
    }
}


//---
//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from agent this buffer is associated with (via _hsaAgent).
//IN: src - src pointer for copy.  Must be accessible from host CPU.
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void UnpinnedCopyEngine::CopyPeerToPeer(void* dst, hsa_agent_t dstAgent, const void* src, hsa_agent_t srcAgent, size_t sizeBytes, hsa_signal_t *waitFor)
{
    std::lock_guard<std::mutex> l (_copyLock);

    const char *srcp0 = static_cast<const char*> (src);
    char *dstp1 = static_cast<char*> (dst);

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completionSignal[i], 0);
        hsa_signal_store_relaxed(_completionSignal2[i], 0);
    }

    if (sizeBytes >= UINT64_MAX/2) {
        THROW_ERROR (hipErrorInvalidValue);
    }

    int64_t bytesRemaining0 = sizeBytes; // bytes to copy from dest into staging buffer.
    int64_t bytesRemaining1 = sizeBytes; // bytes to copy from staging buffer into final dest

    while (bytesRemaining1 > 0) {
        // First launch the async copies to copy from dest to host
        for (int bufferIndex = 0; (bytesRemaining0>0) && (bufferIndex < _numBuffers);  bytesRemaining0 -= _bufferSize, bufferIndex++) {

            size_t theseBytes = (bytesRemaining0 > _bufferSize) ? _bufferSize : bytesRemaining0;

            // Wait to make sure we are not overwriting a buffer before it has been drained:
            hsa_signal_wait_acquire(_completionSignal2[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

            tprintf (DB_COPY2, "P2P: bytesRemaining0=%zu  async_copy %zu bytes src:%p to staging:%p\n", bytesRemaining0, theseBytes, srcp0, _pinnedStagingBuffer[bufferIndex]);
            hsa_signal_store_relaxed(_completionSignal[bufferIndex], 1);
            hsa_status_t hsa_status = hsa_amd_memory_async_copy(_pinnedStagingBuffer[bufferIndex], _cpuAgent, srcp0, srcAgent, theseBytes, waitFor ? 1:0, waitFor, _completionSignal[bufferIndex]);
            if (hsa_status != HSA_STATUS_SUCCESS) {
                THROW_ERROR (hipErrorRuntimeMemory);
            }

            srcp0 += theseBytes;


            // Assume subsequent commands are dependent on previous and don't need dependency after first copy submitted, HIP_ONESHOT_COPY_DEP=1
            waitFor = NULL;
        }

        // Now unload the staging buffers:
        for (int bufferIndex=0; (bytesRemaining1>0) && (bufferIndex < _numBuffers);  bytesRemaining1 -= _bufferSize, bufferIndex++) {

            size_t theseBytes = (bytesRemaining1 > _bufferSize) ? _bufferSize : bytesRemaining1;

            tprintf (DB_COPY2, "P2P: wait_completion[%d] bytesRemaining=%zu\n", bufferIndex, bytesRemaining1);

            bool hostWait = 0; // TODO - remove me

            if (hostWait) {
                // Host-side wait, should not be necessary:
                hsa_signal_wait_acquire(_completionSignal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
            }

            tprintf (DB_COPY2, "P2P: bytesRemaining1=%zu copy %zu bytes stagingBuf[%d]:%p to device:%p\n", bytesRemaining1, theseBytes, bufferIndex, _pinnedStagingBuffer[bufferIndex], dstp1);
            hsa_signal_store_relaxed(_completionSignal2[bufferIndex], 1);
            hsa_status_t hsa_status = hsa_amd_memory_async_copy(dstp1, dstAgent, _pinnedStagingBuffer[bufferIndex], _cpuAgent /*not used*/, theseBytes,
                                      hostWait ? 0:1, hostWait ? NULL : &_completionSignal[bufferIndex],
                                      _completionSignal2[bufferIndex]);

            dstp1 += theseBytes;
        }
    }


    // Wait for the staging-buffer to dest copies to complete:
    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_wait_acquire(_completionSignal2[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    }
}
