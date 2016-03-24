
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Memory
//
//
//

//---
/**
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
 */
hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, void* ptr)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    hc::accelerator acc;
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
    am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
    if (status == AM_SUCCESS) {

        attributes->memoryType    = amPointerInfo._isInDeviceMem ? hipMemoryTypeDevice: hipMemoryTypeHost;
        attributes->hostPointer   = amPointerInfo._hostPointer;
        attributes->devicePointer = amPointerInfo._devicePointer;
        attributes->isManaged     = 0;
        if(attributes->memoryType == hipMemoryTypeHost){
            attributes->hostPointer = ptr;
        }
        if(attributes->memoryType == hipMemoryTypeDevice){
            attributes->devicePointer = ptr;
        }
        attributes->allocationFlags = amPointerInfo._appAllocationFlags;
        attributes->device          = amPointerInfo._appId;

        if (attributes->device < 0) {
            e = hipErrorInvalidDevice;
        }


    } else {
        attributes->memoryType    = hipMemoryTypeDevice;
        attributes->hostPointer   = 0;
        attributes->devicePointer = 0;
        attributes->device        = -1;
        attributes->isManaged     = 0;
        attributes->allocationFlags = 0;

        e = hipErrorUnknown; // TODO - should be hipErrorInvalidValue ?
    }

    return ihipLogStatus(e);
}


/**
 * @returns #hipSuccess,
 * @returns #hipErrorInvalidValue if flags are not 0
 * @returns #hipErrorMemoryAllocation if hostPointer is not a tracked allocation.
 */
hipError_t hipHostGetDevicePointer(void **devicePointer, void *hostPointer, unsigned flags)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    // Flags must be 0:
    if (flags != 0) {
        e = hipErrorInvalidValue;
    } else {
        hc::accelerator acc;
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, hostPointer);
        if (status == AM_SUCCESS) {
            *devicePointer = amPointerInfo._devicePointer;
        } else {
            e = hipErrorMemoryAllocation;
            *devicePointer = NULL;
        }
    }

    return ihipLogStatus(e);
}



// kernel for launching memcpy operations:
template <typename T>
hc::completion_future
ihipMemcpyKernel(hipStream_t stream, T * c, const T * a, size_t sizeBytes)
{
    int wg = std::min((unsigned)8, stream->getDevice()->_compute_units);
    const int threads_per_wg = 256;

    int threads = wg * threads_per_wg;
    if (threads > sizeBytes) {
        threads = ((sizeBytes + threads_per_wg - 1) / threads_per_wg) * threads_per_wg;
    }


    hc::extent<1> ext(threads);
    auto ext_tile = ext.tile(threads_per_wg);

    hc::completion_future cf =
    hc::parallel_for_each(
            stream->_av,
            ext_tile,
            [=] (hc::tiled_index<1> idx)
            __attribute__((hc))
    {
        int offset = amp_get_global_id(0);
        // TODO-HCC - change to hc_get_local_size()
        int stride = amp_get_local_size(0) * hc_get_num_groups(0) ;

        for (int i=offset; i<sizeBytes; i+=stride) {
            c[i] = a[i];
        }
    });

    return cf;
}


// kernel for launching memset operations:
template <typename T>
hc::completion_future
ihipMemsetKernel(hipStream_t stream, T * ptr, T val, size_t sizeBytes)
{
    int wg = std::min((unsigned)8, stream->getDevice()->_compute_units);
    const int threads_per_wg = 256;

    int threads = wg * threads_per_wg;
    if (threads > sizeBytes) {
        threads = ((sizeBytes + threads_per_wg - 1) / threads_per_wg) * threads_per_wg;
    }


    hc::extent<1> ext(threads);
    auto ext_tile = ext.tile(threads_per_wg);

    hc::completion_future cf =
    hc::parallel_for_each(
            stream->_av,
            ext_tile,
            [=] (hc::tiled_index<1> idx)
            __attribute__((hc))
    {
        int offset = amp_get_global_id(0);
        // TODO-HCC - change to hc_get_local_size()
        int stride = amp_get_local_size(0) * hc_get_num_groups(0) ;

        for (int i=offset; i<sizeBytes; i+=stride) {
            ptr[i] = val;
        }
    });

    return cf;

}

//---
/**
 * @returns #hipSuccess #hipErrorMemoryAllocation
 */
hipError_t hipMalloc(void** ptr, size_t sizeBytes)
{
    HIP_INIT_API(ptr, sizeBytes);

    hipError_t  hip_status = hipSuccess;

	auto device = ihipGetTlsDefaultDevice();

    if (device) {
        const unsigned am_flags = 0;
        *ptr = hc::am_alloc(sizeBytes, device->_acc, am_flags);

        if (sizeBytes && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        } else {
            hc::am_memtracker_update(*ptr, device->_device_index, 0);
        }
    } else {
        hip_status = hipErrorMemoryAllocation;
    }

    return ihipLogStatus(hip_status);
}


hipError_t hipMallocHost(void** ptr, size_t sizeBytes)
{
    HIP_INIT_API(ptr, sizeBytes);

    hipError_t  hip_status = hipSuccess;

    const unsigned am_flags = amHostPinned;
	auto device = ihipGetTlsDefaultDevice();

    if (device) {
        *ptr = hc::am_alloc(sizeBytes, device->_acc, am_flags);
        if (sizeBytes && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        } else {
            hc::am_memtracker_update(*ptr, device->_device_index, 0);
        }

        tprintf (DB_MEM, "  %s: pinned ptr=%p\n", __func__, *ptr);
    }

    return ihipLogStatus(hip_status);
}


hipError_t hipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags)
{
    HIP_INIT_API(ptr, sizeBytes, flags);

    hipError_t hip_status = hipSuccess;

    auto device = ihipGetTlsDefaultDevice();

    if(device){
        if(flags == hipHostMallocDefault){
            *ptr = hc::am_alloc(sizeBytes, device->_acc, amHostPinned);
            if(sizeBytes && (*ptr == NULL)){
                hip_status = hipErrorMemoryAllocation;
            }else{
                hc::am_memtracker_update(*ptr, device->_device_index, 0);
            }
            tprintf(DB_MEM, " %s: pinned ptr=%p\n", __func__, *ptr);
        } else if(flags & hipHostMallocMapped){
            *ptr = hc::am_alloc(sizeBytes, device->_acc, amHostPinned);
            if(sizeBytes && (*ptr == NULL)){
                hip_status = hipErrorMemoryAllocation;
            }else{
                hc::am_memtracker_update(*ptr, device->_device_index, flags);
            }
            tprintf(DB_MEM, " %s: pinned ptr=%p\n", __func__, *ptr);
        }
    }
    return ihipLogStatus(hip_status);
}


// TODO - remove me, this is deprecated.
hipError_t hipHostAlloc(void** ptr, size_t sizeBytes, unsigned int flags)
{
    return hipHostMalloc(ptr, sizeBytes, flags);
};


hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr)
{
    HIP_INIT_API(flagsPtr, hostPtr);

	hipError_t hip_status = hipSuccess;

	hc::accelerator acc;
	hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
	am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, hostPtr);
	if(status == AM_SUCCESS){
		*flagsPtr = amPointerInfo._appAllocationFlags;
		if(*flagsPtr == 0){
			hip_status = hipErrorInvalidValue;
		}
		else{
			hip_status = hipSuccess;
		}
		tprintf(DB_MEM, " %s: host ptr=%p\n", __func__, hostPtr);
	}else{
		hip_status = hipErrorInvalidValue;
	}
	return ihipLogStatus(hip_status);
}

hipError_t hipHostRegister(void *hostPtr, size_t sizeBytes, unsigned int flags)
{
    HIP_INIT_API(hostPtr, sizeBytes, flags);

	hipError_t hip_status = hipSuccess;

	auto device = ihipGetTlsDefaultDevice();
	void* srcPtr;
	if(hostPtr == NULL){
		return ihipLogStatus(hipErrorInvalidValue);
	}
	if(device){
	if(flags == hipHostRegisterDefault){
		hsa_status_t hsa_status = hsa_amd_memory_lock(hostPtr, sizeBytes, &device->_hsa_agent, 1, &srcPtr);
		if(hsa_status == HSA_STATUS_SUCCESS){
			hip_status = hipSuccess;	
		}else{
			hip_status = hipErrorMemoryAllocation;
		}
	}
	else if (flags | hipHostRegisterMapped){
		hsa_status_t hsa_status = hsa_amd_memory_lock(hostPtr, sizeBytes, &device->_hsa_agent, 1, &srcPtr);
		//TODO: Added feature for actual host pointer being tracked
		if(hsa_status != HSA_STATUS_SUCCESS){
			hip_status = hipErrorMemoryAllocation;
		}
	}
	}
	return ihipLogStatus(hip_status);
}

hipError_t hipHostUnregister(void *hostPtr)
{
    HIP_INIT_API(hostPtr);

	hipError_t hip_status = hipSuccess;
	if(hostPtr == NULL){
		hip_status = hipErrorInvalidValue;
	}else{
	hsa_status_t hsa_status = hsa_amd_memory_unlock(hostPtr);
	if(hsa_status != HSA_STATUS_SUCCESS){
		hip_status = hipErrorInvalidValue;
// TODO: Add a different return error. This is not true
	}
	}
	return ihipLogStatus(hip_status);
}


//---
hipError_t hipMemcpyToSymbol(const char* symbolName, const void *src, size_t count, size_t offset, hipMemcpyKind kind)
{
    HIP_INIT_API(symbolName, src, count, offset, kind);

#ifdef USE_MEMCPYTOSYMBOL
	if(kind != hipMemcpyHostToDevice)
	{
		return ihipLogStatus(hipErrorInvalidValue);
	}
	auto device = ihipGetTlsDefaultDevice();

    //hsa_signal_t depSignal;
    //int depSignalCnt = device._default_stream->preCopyCommand(NULL, &depSignal, ihipCommandCopyH2D);
    assert(0);  // Need to properly synchronize the copy - do something with depSignal if != NULL.

	device->_acc.memcpy_symbol(symbolName, (void*) src,count, offset);
#endif
    return ihipLogStatus(hipSuccess);
}


// Resolve hipMemcpyDefault to a known type.
unsigned ihipStream_t::resolveMemcpyDirection(bool srcInDeviceMem, bool dstInDeviceMem)
{
    hipMemcpyKind kind = hipMemcpyDefault;

    if (!srcInDeviceMem && !dstInDeviceMem) {
        kind = hipMemcpyHostToHost;
    } else if (!srcInDeviceMem && dstInDeviceMem) {
        kind = hipMemcpyHostToDevice;
    } else if (srcInDeviceMem && !dstInDeviceMem) {
        kind = hipMemcpyDeviceToHost;
    } else if (srcInDeviceMem &&  dstInDeviceMem) {
        kind = hipMemcpyDeviceToDevice;
    }

    assert (kind != hipMemcpyDefault);

    return kind;
}


// Setup the copyCommandType and the copy agents (for hsa_amd_memory_async_copy)
void ihipStream_t::setCopyAgents(unsigned kind, ihipCommand_t *commandType, hsa_agent_t *srcAgent, hsa_agent_t *dstAgent)
{
    ihipDevice_t *device = this->getDevice();
    hsa_agent_t deviceAgent = device->_hsa_agent;

    switch (kind) {
        case hipMemcpyHostToHost     : *commandType = ihipCommandCopyH2H; *srcAgent=g_cpu_agent; *dstAgent=g_cpu_agent; break;
        case hipMemcpyHostToDevice   : *commandType = ihipCommandCopyH2D; *srcAgent=g_cpu_agent; *dstAgent=deviceAgent; break;
        case hipMemcpyDeviceToHost   : *commandType = ihipCommandCopyD2H; *srcAgent=deviceAgent; *dstAgent=g_cpu_agent; break;
        case hipMemcpyDeviceToDevice : *commandType = ihipCommandCopyD2D; *srcAgent=deviceAgent; *dstAgent=deviceAgent; break;
        default: throw ihipException(hipErrorInvalidMemcpyDirection);
    };
}


void ihipStream_t::copySync(void* dst, const void* src, size_t sizeBytes, unsigned kind)
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


    // Resolve default to a specific Kind so we know which algorithm to use:
    if (kind == hipMemcpyDefault) {
        bool srcInDeviceMem = (srcTracked && srcPtrInfo._isInDeviceMem);
        bool dstInDeviceMem = (dstTracked && dstPtrInfo._isInDeviceMem);
        kind = resolveMemcpyDirection(srcInDeviceMem, dstInDeviceMem);
    };

    hsa_signal_t depSignal;

    if ((kind == hipMemcpyHostToDevice) && (!srcTracked)) {
        int depSignalCnt = preCopyCommand(NULL, &depSignal, ihipCommandCopyH2D);
        if (HIP_STAGING_BUFFERS) {
            tprintf(DB_COPY1, "D2H && !dstTracked: staged copy H2D dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);

            if (HIP_PININPLACE) {
                device->_staging_buffer[0]->CopyHostToDevicePinInPlace(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
            } else  {
                device->_staging_buffer[0]->CopyHostToDevice(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
            }

            // The copy waits for inputs and then completes before returning so can reset queue to empty:
            this->wait(true);
        } else {
            // TODO - remove, slow path.
            tprintf(DB_COPY1, "H2D && ! srcTracked: am_copy dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);
#if USE_AV_COPY
            _av.copy(src,dst,sizeBytes);
#else
            hc::am_copy(dst, src, sizeBytes);
#endif
        }
    } else if ((kind == hipMemcpyDeviceToHost) && (!dstTracked)) {
        int depSignalCnt = preCopyCommand(NULL, &depSignal, ihipCommandCopyD2H);
        if (HIP_STAGING_BUFFERS) {
            tprintf(DB_COPY1, "D2H && !dstTracked: staged copy D2H dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);
            //printf ("staged-copy- read dep signals\n");
            device->_staging_buffer[1]->CopyDeviceToHost(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);

            // The copy waits for inputs and then completes before returning so can reset queue to empty:
            this->wait(true);

        } else {
            // TODO - remove, slow path.
            tprintf(DB_COPY1, "D2H && !dstTracked: am_copy dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);
#if USE_AV_COPY
            _av.copy(src, dst, sizeBytes);
#else
            hc::am_copy(dst, src, sizeBytes);
#endif
        }
    } else if (kind == hipMemcpyHostToHost)  { 
        int depSignalCnt = preCopyCommand(NULL, &depSignal, ihipCommandCopyH2H);

        if (depSignalCnt) {
            // host waits before doing host memory copy.
            hsa_signal_wait_acquire(depSignal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
        }
        tprintf(DB_COPY1, "H2H memcpy dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);
        memcpy(dst, src, sizeBytes);

    } else {
        // If not special case - these can all be handled by the hsa async copy:
        ihipCommand_t commandType;
        hsa_agent_t srcAgent, dstAgent;
        setCopyAgents(kind, &commandType, &srcAgent, &dstAgent);

        int depSignalCnt = preCopyCommand(NULL, &depSignal, commandType);

        // Get a completion signal:
        ihipSignal_t *ihipSignal = allocSignal();
        hsa_signal_t copyCompleteSignal = ihipSignal->_hsa_signal;

        hsa_signal_store_relaxed(copyCompleteSignal, 1);

        tprintf(DB_COPY1, "HSA Async_copy dst=%p src=%p sz=%zu\n", dst, src, sizeBytes);

        hsa_status_t hsa_status = hsa_amd_memory_async_copy(dst, dstAgent, src, srcAgent, sizeBytes, depSignalCnt, depSignalCnt ? &depSignal:0x0, copyCompleteSignal);

        // This is sync copy, so let's wait for copy right here:
        if (hsa_status == HSA_STATUS_SUCCESS) {
            waitCopy(ihipSignal); // wait for copy, and return to pool.
        } else {
            throw ihipException(hipErrorInvalidValue);
        }
    }
}




void ihipStream_t::copyAsync(void* dst, const void* src, size_t sizeBytes, unsigned kind)
{
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
        this->wait();

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
            kind = resolveMemcpyDirection(srcInDeviceMem, dstInDeviceMem);
        }



        ihipSignal_t *ihip_signal = allocSignal();
        hsa_signal_store_relaxed(ihip_signal->_hsa_signal, 1);


        if(trueAsync == true){

            ihipCommand_t commandType;
            hsa_agent_t srcAgent, dstAgent;
            setCopyAgents(kind, &commandType, &srcAgent, &dstAgent);

            hsa_signal_t depSignal;
            int depSignalCnt = preCopyCommand(ihip_signal, &depSignal, commandType);

            tprintf (DB_SYNC, " copy-async, waitFor=%lu completion=#%lu(%lu)\n", depSignalCnt? depSignal.handle:0x0, ihip_signal->_sig_id, ihip_signal->_hsa_signal.handle);

            hsa_status_t hsa_status = hsa_amd_memory_async_copy(dst, dstAgent, src, srcAgent, sizeBytes, depSignalCnt, depSignalCnt ? &depSignal:0x0, ihip_signal->_hsa_signal);


            if (hsa_status == HSA_STATUS_SUCCESS) {
                if (HIP_LAUNCH_BLOCKING) {
                    tprintf(DB_SYNC, "LAUNCH_BLOCKING for completion of hipMemcpyAsync(%zu)\n", sizeBytes);
                    this->wait();
                }
            } else {
                // This path can be hit if src or dst point to unpinned host memory.
                // TODO-stream - does async-copy fall back to sync if input pointers are not pinned?
                throw ihipException(hipErrorInvalidValue);
            }
        } else {
            copySync(dst, src, sizeBytes, kind);
        }
    }
}


//---
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
{
    HIP_INIT_API(dst, src, sizeBytes, kind);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {
        stream->copySync(dst, src, sizeBytes, kind);
    }
    catch (ihipException ex) {
        e = ex._code;
    }


    if (HIP_LAUNCH_BLOCKING) {
        tprintf(DB_SYNC, "LAUNCH_BLOCKING for completion of hipMemcpy\n");
        stream->wait();
    }

    return ihipLogStatus(e);
}


/**
 * @result #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidMemcpyDirection, 
 * @result #hipErrorInvalidValue : If dst==NULL or src==NULL, or other bad argument.
 * @warning on HCC hipMemcpyAsync does not support overlapped H2D and D2H copies.
 * @warning on HCC hipMemcpyAsync requires that any host pointers are pinned (ie via the hipMallocHost call).
 */
//---
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
{
    HIP_INIT_API(dst, src, sizeBytes, kind, stream);

    hipError_t e = hipSuccess;

    stream = ihipSyncAndResolveStream(stream);


    if ((dst == NULL) || (src == NULL)) {
        e= hipErrorInvalidValue;
    } else if (stream) {
        try {
            stream->copyAsync(dst, src, sizeBytes, kind);
        }
        catch (ihipException ex) {
            e = ex._code;
        }
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}


// TODO-sync: function is async unless target is pinned host memory - then these are fully sync.
/** @return #hipErrorInvalidValue
 */
hipError_t hipMemsetAsync(void* dst, int  value, size_t sizeBytes, hipStream_t stream )
{
    HIP_INIT_API(dst, value, sizeBytes, stream);

    hipError_t e = hipSuccess;

    stream =  ihipSyncAndResolveStream(stream);
    stream->preKernelCommand();

    if (stream) {

        hc::completion_future cf ;

        if ((sizeBytes & 0x3) == 0) {
            // use a faster word-per-workitem copy:
            try {
                value = value & 0xff;
                unsigned value32 = (value << 24) | (value << 16) | (value << 8) | (value) ;
                cf = ihipMemsetKernel<unsigned> (stream, static_cast<unsigned*> (dst), value32, sizeBytes/sizeof(unsigned));
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        } else {
            // use a slow byte-per-workitem copy:
            try {
                cf = ihipMemsetKernel<char> (stream, static_cast<char*> (dst), value, sizeBytes);
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        }

        stream->postKernelCommand(cf);


        if (HIP_LAUNCH_BLOCKING) {
            tprintf (DB_SYNC, "'%s' LAUNCH_BLOCKING wait for completion [stream:%p].\n", __func__, (void*)stream);
            cf.wait();
            tprintf (DB_SYNC, "'%s' LAUNCH_BLOCKING completed [stream:%p].\n", __func__, (void*)stream);
        }
    } else {
        e = hipErrorInvalidValue;
    }


    return ihipLogStatus(e);
};


hipError_t hipMemset(void* dst, int  value, size_t sizeBytes )
{
    HIP_INIT_API(dst, value, sizeBytes);

    // TODO - call an ihip memset so HIP_TRACE is correct.
    return hipMemsetAsync(dst, value, sizeBytes, hipStreamNull);
}


/*
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue (if free != NULL due to bug)S
 * @warning On HCC, the free memory only accounts for memory allocated by this process and may be optimistic.
 */
hipError_t hipMemGetInfo  (size_t *free, size_t *total)
{
    HIP_INIT_API(free, total);

    hipError_t e = hipSuccess;

    ihipDevice_t * hipDevice = ihipGetTlsDefaultDevice();
    if (hipDevice) {
        if (total) {
            *total = hipDevice->_props.totalGlobalMem;
        }

        if (free) {
            // TODO - replace with kernel-level for reporting free memory:
            size_t deviceMemSize, hostMemSize, userMemSize;
            hc::am_memtracker_sizeinfo(hipDevice->_acc, &deviceMemSize, &hostMemSize, &userMemSize);
            printf ("deviceMemSize=%zu\n", deviceMemSize);
        
            *free =  hipDevice->_props.totalGlobalMem - deviceMemSize;
        }

    } else {
        e = hipErrorInvalidDevice;
    }

    return ihipLogStatus(e);
}


//---
hipError_t hipFree(void* ptr)
{
    HIP_INIT_API(ptr);

    hipError_t hipStatus = hipErrorInvalidDevicePointer;

   // Synchronize to ensure all work has finished.
    ihipGetTlsDefaultDevice()->waitAllStreams(); // ignores non-blocking streams, this waits for all activity to finish.

    if (ptr) {
        hc::accelerator acc;
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if(status == AM_SUCCESS){
            if(amPointerInfo._hostPointer == NULL){
                hc::am_free(ptr);
                hipStatus = hipSuccess;
            }
        }
    }

    return ihipLogStatus(hipStatus);
}


hipError_t hipHostFree(void* ptr)
{
    HIP_INIT_API(ptr);

    // TODO - ensure this pointer was created by hipMallocHost and not hipMalloc
    std::call_once(hip_initialized, ihipInit);

    hipError_t hipStatus = hipErrorInvalidDevicePointer;
    if (ptr) {
        hc::accelerator acc;
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if(status == AM_SUCCESS){
            if(amPointerInfo._hostPointer == ptr){
                hc::am_free(ptr);
                hipStatus = hipSuccess;
            }
        }
    }

    return ihipLogStatus(hipStatus);
};


// TODO - deprecated function.
hipError_t hipFreeHost(void* ptr)
{
    return hipHostFree(ptr);
}


