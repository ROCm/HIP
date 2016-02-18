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
#include <list>
#include <sys/types.h>
#include <unistd.h>
#include <hc.hpp>
#include <hc_am.hpp>

#include "hip_runtime.h"

#include "hsa_ext_amd.h"

#define USE_PINNED_HOST (__hcc_workweek__ >= 1601)

//#define USE_ASYNC_COPY 

#define INLINE static inline

//---
// Environment variables:
// TODO-HCC - map this to the HC instruction that uses HSAIL to get the wave size.
int warpSize = 64;

// Intended to distinguish whether an environment variable should be visible only in debug mode, or in debug+release.
//static const int debug = 0;
static const int release = 1;

int HIP_PRINT_ENV = 0;
int HIP_TRACE_API= 0;
int HIP_LAUNCH_BLOCKING = 0;

#define TRACE_API  0x1 /* trace API calls and return values */
#define TRACE_SYNC 0x2 /* trace synchronization pieces */
#define TRACE_MEM  0x4 /* trace memory allocation / deallocation */

#define tprintf(trace_level, ...) {\
    if (HIP_TRACE_API & trace_level) {\
        fprintf (stderr, "hiptrace%d: ", trace_level); \
        fprintf (stderr, __VA_ARGS__);\
    }\
}

const hipStream_t hipStreamNull = 0x0;

struct ihipDevice_t;


enum ihipCommand_t {
    ihipCommandKernel,
    ihipCommandData,
};

// Internal stream structure.
struct ihipStream_t {
    unsigned             _device_index;
    hc::accelerator_view _av;
    unsigned            _flags;
    ihipCommand_t        _last_command;

    //ihipStream_t() : _av(){ };
    ihipStream_t(unsigned device_index, hc::accelerator_view av, unsigned int flags) :
        _device_index(device_index), _av(av), _flags(flags), _last_command(ihipCommandKernel)
         {};
} ;



//----
// Internal event structure:
enum hipEventStatus_t {
   hipEventStatusUnitialized = 0, // event is unutilized, must be "Created" before use.
   hipEventStatusCreated     = 1,
   hipEventStatusRecording   = 2, // event has been enqueued to record something.
   hipEventStatusRecorded    = 3, // event has been recorded - timestamps are valid.
} ;


// internal hip event structure.
struct ihipEvent_t {
    hipEventStatus_t       _state;

    hipStream_t           _stream;  // Stream where the event is recorded, or NULL if all streams.
    unsigned              _flags;

    hc::completion_future _marker;
    uint64_t              _timestamp;  // store timestamp, may be set on host or by marker.
} ;


struct ihipDevice_t
{
    unsigned                _device_index; // index into g_devices.

    hipDeviceProp_t         _props;        // saved device properties.
    hc::accelerator         _acc;
    hsa_agent_t             _hsa_agent;    // hsa agent handle

    // The NULL stream is used if no other stream is specified.
    // NULL has special synchronization properties with other streams.
    ihipStream_t            *_null_stream;

    std::list<ihipStream_t*> _streams;   // streams associated with this device.

    unsigned                _compute_units;

public:
    ihipDevice_t(unsigned device_index, hc::accelerator acc);
    hipError_t getProperties(hipDeviceProp_t* prop);

    // TODO- create a copy constructor.
    //~ihipDevice_t();
};


//=================================================================================================
ihipDevice_t::ihipDevice_t(unsigned device_index, hc::accelerator acc)
    :   _device_index(device_index),
        _acc(acc)
{
    hsa_agent_t *agent = static_cast<hsa_agent_t*> (acc.get_default_view().get_hsa_agent());
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

    _null_stream = new ihipStream_t(device_index, acc.get_default_view(), hipStreamDefault);
    this->_streams.push_back(_null_stream);
    tprintf(TRACE_SYNC, "created device with null_stream=%p\n", _null_stream);
};

#if 0
ihipDevice_t::~ihipDevice_t()
{
    if (_null_stream) {
        delete _null_stream;
        _null_stream = NULL;
    }
}
#endif

//----

//=================================================================================================
//TLS - must be initialized here.
thread_local hipError_t tls_lastHipError = hipSuccess;
thread_local int tls_defaultDevice = 0;

// Global initialization.
std::once_flag hip_initialized;
std::vector<ihipDevice_t> g_devices; // Vector of all non-emulated (ie GPU) accelerators in the system.

//=================================================================================================



//=================================================================================================
// Utility functions, these are not part of the public HIP API
//=================================================================================================

//=================================================================================================

#define DeviceErrorCheck(x) if (x != HSA_STATUS_SUCCESS) { return hipErrorInvalidDevice; }

#define ErrorCheck(x) error_check(x, __LINE__, __FILE__)

void error_check(hsa_status_t hsa_error_code, int line_num, std::string str) {
  if (hsa_error_code != HSA_STATUS_SUCCESS) {
    printf("HSA reported error!\n In file: %s\nAt line: %d\n", str.c_str(),line_num);
  }
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

    /* Computemode for HSA Devices is always : cudaComputeModeDefault :/
                             Default compute mode (Multiple threads can use cudaSetDevice() with this device)  */
    prop->computeMode = 0;

    // Get Max Threads Per Multiprocessor
/*
    HsaSystemProperties props;
    hsaKmtReleaseSystemProperties();
    if(HSAKMT_STATUS_SUCCESS == hsaKmtAcquireSystemProperties(&props)) {
        HsaNodeProperties node_prop = {0};
        if(HSAKMT_STATUS_SUCCESS == hsaKmtGetNodeProperties(node, &node_prop)) {
            uint32_t waves_per_cu = node_prop.MaxWavesPerSIMD;
            prop-> maxThreadsPerMultiProcessor = prop->warpsize*waves_per_cu;
        }
    }
*/

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
    prop->arch.hasSharedInt32Atomics       = 0; // TODO-hcc-atomics
    prop->arch.hasSharedFloatAtomicExch    = 0; // TODO-hcc-atomics
    prop->arch.hasFloatAtomicAdd           = 0;
    prop->arch.hasGlobalInt64Atomics       = 1;
    prop->arch.hasSharedInt64Atomics       = 0; // TODO-hcc-atomics
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
    return e;
}

#define ihipLogStatus(_hip_status) \
    ({\
        tls_lastHipError = _hip_status;\
        \
        if (HIP_TRACE_API & TRACE_API) {\
            fprintf(stderr, "hiptrace1: %-30s ret=%2d\n", __func__,  _hip_status);\
        }\
        _hip_status;\
    })



// Read environment variables.
void ihipReadEnv_I(int *var_ptr, const char *var_name1, const char *var_name2, const char *description)
{
    char * env = getenv(var_name1);

    // Check second name if first not defined, used to allow HIP_ or CUDA_ env vars.
    if ((env == NULL) && strcmp(var_name2, "0")) {
        env = getenv(var_name2);
    }

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



//---
//Function called one-time at initialization time to construct a table of all GPU devices.
//HIP/CUDA uses integer "deviceIds" - these are indexes into this table.
//AMP maintains a table of accelerators, but some are emulated - ie for debug or CPU.
//This function creates a vector with only the GPU accelerators.
//It is called with C++11 call_once, which provided thread-safety.
void ihipInit()
{

    /*
     * Build a table of valid compute devices.
     */
    auto accs = hc::accelerator::get_all();
    g_devices.reserve(accs.size());
    for (int i=0; i<accs.size(); i++) {
        if (! accs[i].get_is_emulated()) {
           g_devices.emplace_back(ihipDevice_t(g_devices.size(), accs[i]));
        }
    }

    /*
     * Environment variables
     */
    READ_ENV_I(release, HIP_PRINT_ENV, 0,  "Print HIP environment variables.");
    READ_ENV_I(release, HIP_TRACE_API, 0,  "Trace each HIP API call.  Print function name and return code to stderr as program executes.");
    READ_ENV_I(release, HIP_LAUNCH_BLOCKING, CUDA_LAUNCH_BLOCKING, "Make HIP APIs 'host-synchronous', so they block until any kernel launches or data copy commands complete. Alias: CUDA_LAUNCH_BLOCKING." );

    tprintf(TRACE_API, "pid=%u %-30s\n", getpid(), "<ihipInit>");

}

INLINE bool ihipIsValidDevice(unsigned deviceIndex)
{
    // deviceIndex is unsigned so always > 0
    return (deviceIndex < g_devices.size());
}


//---
INLINE ihipDevice_t *ihipGetTlsDefaultDevice()
{
    // If this is invalid, the TLS state is corrupt.
    // This can fire if called before devices are initialized.
    // TODO - consider replacing assert with error code 
    assert (ihipIsValidDevice(tls_defaultDevice));

    return &g_devices[tls_defaultDevice];
}


//---
INLINE ihipDevice_t *ihipGetDevice(int deviceId)
{
    if ((deviceId >= 0) && (deviceId < g_devices.size())) {
        return &g_devices[deviceId];
    } else {
        return NULL;
    }

}


//---
//Heavyweight synchronization that waits on all streams, ignoring hipStreamNonBlocking flag.
static inline void ihipWaitAllStreams(ihipDevice_t *device)
{
    tprintf(TRACE_SYNC, "waitAllStream\n");
    for (auto streamI=device->_streams.begin(); streamI!=device->_streams.end(); streamI++) {
        (*streamI)->_av.wait();
    }
}




inline void ihipWaitNullStream(ihipDevice_t *device)
{
    tprintf(TRACE_SYNC, "waitNullStream\n");

    for (auto streamI=device->_streams.begin(); streamI!=device->_streams.end(); streamI++) {
        ihipStream_t *stream = *streamI;
        if (!(stream->_flags & hipStreamNonBlocking)) {
            // TODO-hcc - use blocking or active wait here?
            // TODO-sync - cudaDeviceBlockingSync
            stream->_av.wait();
        }
    }
}


//---
// Get the stream to use for a command submission.
//
// If stream==NULL synchronize appropriately with other streams and return the default av for the device.
// If stream is valid, return the AV to use.
inline hipStream_t ihipSyncAndResolveStream(hipStream_t stream)
{
    if (stream == hipStreamNull ) {
        ihipDevice_t *device = ihipGetTlsDefaultDevice();
        ihipWaitNullStream(device);

        return device->_null_stream;
    } else {
        return stream;
    }
}

#if 0
inline hsa_status_t
HSABarrier::enqueueBarrier(hsa_queue_t* queue) {
    hsa_status_t status = HSA_STATUS_SUCCESS;

    hc::completion_future marker = stream->_av.create_marker();

    // Create a signal to wait for the barrier to finish.
    std::pair<hsa_signal_t, int> ret = Kalmar::ctx.getSignal();
    signal = ret.first;
    signalIndex = ret.second;

    // Obtain the write index for the command queue
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    const uint32_t queueMask = queue->size - 1;

    // Define the barrier packet to be at the calculated queue index address
    hsa_barrier_and_packet_t* barrier = &(((hsa_barrier_and_packet_t*)(queue->base_address))[index&queueMask]);
    memset(barrier, 0, sizeof(hsa_barrier_and_packet_t));

    // setup header
    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    header |= 1 << HSA_PACKET_HEADER_BARRIER;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
    barrier->header = header;

    barrier->completion_signal = signal;

    // Increment write index and ring doorbell to dispatch the kernel
    hsa_queue_store_write_index_relaxed(queue, index+1);
    hsa_signal_store_relaxed(queue->doorbell_signal, index);

    isDispatched = true;

    return status;
}
#endif

//--
//When the commands in a stream change types (ie kernel command follows a data command,
//or data command follows a kernel command), then we need to add a barrier packet
//into the stream to mimic CUDA stream semantics.  (some hardware uses separate
//queues for data commands and kernel commands, and no implicit ordering is provided).
//
inline bool ihipCheckCommandSwitchSync(hipStream_t stream, ihipCommand_t new_command, hc::completion_future *marker)
{
    bool addedSync = false;
    // If switching command types, we need to add a barrier packet to synchronize things.
    if (stream->_last_command != new_command) {
        addedSync = true;
        *marker = stream->_av.create_marker();

        tprintf (TRACE_SYNC, "stream %p switch to %s (barrier pkt inserted)\n", (void*)stream, new_command == ihipCommandKernel ? "Kernel" : "Data");
        stream->_last_command = new_command;
    }

    return addedSync;
}


// Called just before a kernel is launched from hipLaunchKernel.
// Allows runtime to track some information about the stream.
hc::accelerator_view *ihipLaunchKernel(hipStream_t stream)
{

    stream = ihipSyncAndResolveStream(stream);

    hc::completion_future marker;
    ihipCheckCommandSwitchSync(stream, ihipCommandKernel, &marker);

    return &(stream->_av);
}


//
//=================================================================================================
// HIP API Implementation
//
// Implementor notes:
// _ All functions should call ihipInit as first action:
//    std::call_once(hip_initialized, ihipInit);
//
// - ALl functions should use ihipLogStatus to return error code (not return error directly).
//=================================================================================================
//
//---


//-------------------------------------------------------------------------------------------------
//Devices
//-------------------------------------------------------------------------------------------------
//---
/**
 * @return  #hipSuccess
 */
hipError_t hipGetDevice(int *device)
{
    std::call_once(hip_initialized, ihipInit);

    *device = tls_defaultDevice;
    return ihipLogStatus(hipSuccess);
}


//---
/**
 * @return  #hipSuccess, #hipErrorNoDevice
 */
hipError_t hipGetDeviceCount(int *count)
{
    std::call_once(hip_initialized, ihipInit);

    *count = g_devices.size();

    if (*count > 0) {
        return ihipLogStatus(hipSuccess);
    } else {
        return ihipLogStatus(hipErrorNoDevice);
    }
}


//---
/**
 * @returns #hipSuccess
 */
hipError_t hipDeviceSetCacheConfig ( hipFuncCache cacheConfig )
{
    std::call_once(hip_initialized, ihipInit);

    // Nop, AMD does not support variable cache configs.

    return ihipLogStatus(hipSuccess);
}


//---
/**
 * @returns #hipSuccess
 */
hipError_t hipDeviceGetCacheConfig ( hipFuncCache *cacheConfig )
{
    std::call_once(hip_initialized, ihipInit);

    *cacheConfig = hipFuncCachePreferNone;

    return ihipLogStatus(hipSuccess);
}


//---
/**
 * @returns #hipSuccess
 */
hipError_t hipFuncSetCacheConfig ( hipFuncCache cacheConfig )
{
    std::call_once(hip_initialized, ihipInit);

    // Nop, AMD does not support variable cache configs.

    return ihipLogStatus(hipSuccess);
}



//---
/**
 * @returns #hipSuccess
 */
hipError_t hipDeviceSetSharedMemConfig ( hipSharedMemConfig config )
{
    std::call_once(hip_initialized, ihipInit);

    // Nop, AMD does not support variable shared mem configs.

    return ihipLogStatus(hipSuccess);
}



//---
/**
 * @returns #hipSuccess
 */
hipError_t hipDeviceGetSharedMemConfig ( hipSharedMemConfig * pConfig )
{
    std::call_once(hip_initialized, ihipInit);

    *pConfig = hipSharedMemBankSizeFourByte;

    return ihipLogStatus(hipSuccess);
}

//---
/**
 * @return #hipSuccess, #hipErrorInvalidDevice
 */
hipError_t hipSetDevice(int device)
{
    std::call_once(hip_initialized, ihipInit);

    if ((device < 0) || (device > g_devices.size())) {
        return ihipLogStatus(hipErrorInvalidDevice);
    } else {
        tls_defaultDevice = device;
        return ihipLogStatus(hipSuccess);
    }
}


//---
/**
 * @return #hipSuccess
 */
hipError_t hipDeviceSynchronize(void)
{
    std::call_once(hip_initialized, ihipInit);

    ihipWaitAllStreams(ihipGetTlsDefaultDevice()); // ignores non-blocking streams, this waits for all activity to finish.

    return ihipLogStatus(hipSuccess);
}


//---
/**
 * @return @ref hipSuccess
 * @bug On HCC, hipDeviceReset is a nop and does not reset the device state.
 */
hipError_t hipDeviceReset(void)
{
    std::call_once(hip_initialized, ihipInit);

    // TODO-HCC
    // This function needs some support from HSART and KFD.
    // It should destroy and clean up all resources allocated with the default device in the current process.
    // and needs to destroy all queues as well.
    //

    return ihipLogStatus(hipSuccess);
}

/**
 *
 */
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    ihipDevice_t * hipDevice = ihipGetDevice(device);
    hipDeviceProp_t *prop = &hipDevice->_props;
    if (hipDevice) {
        switch (attr) {
        case hipDeviceAttributeMaxThreadsPerBlock:
            *pi = prop->maxThreadsPerBlock; break;
        case hipDeviceAttributeMaxBlockDimX:
            *pi = prop->maxThreadsDim[0]; break;
        case hipDeviceAttributeMaxBlockDimY:
            *pi = prop->maxThreadsDim[1]; break;
        case hipDeviceAttributeMaxBlockDimZ:
            *pi = prop->maxThreadsDim[2]; break;
        case hipDeviceAttributeMaxGridDimX:
            *pi = prop->maxGridSize[0]; break;
        case hipDeviceAttributeMaxGridDimY:
            *pi = prop->maxGridSize[1]; break;
        case hipDeviceAttributeMaxGridDimZ:
            *pi = prop->maxGridSize[2]; break;
        case hipDeviceAttributeMaxSharedMemoryPerBlock:
            *pi = prop->sharedMemPerBlock; break;
        case hipDeviceAttributeTotalConstantMemory:
            *pi = prop->totalConstMem; break;
        case hipDeviceAttributeWarpSize:
            *pi = prop->warpSize; break;
        case hipDeviceAttributeMaxRegistersPerBlock:
            *pi = prop->regsPerBlock; break;
        case hipDeviceAttributeClockRate:
            *pi = prop->clockRate; break;
        case hipDeviceAttributeMemoryClockRate:
            *pi = prop->memoryClockRate; break;
        case hipDeviceAttributeMemoryBusWidth:
            *pi = prop->memoryBusWidth; break;
        case hipDeviceAttributeMultiprocessorCount:
            *pi = prop->multiProcessorCount; break;
        case hipDeviceAttributeComputeMode:
            *pi = prop->computeMode; break;
        case hipDeviceAttributeL2CacheSize:
            *pi = prop->l2CacheSize; break;
        case hipDeviceAttributeMaxThreadsPerMultiProcessor:
            *pi = prop->maxThreadsPerMultiProcessor; break;
        case hipDeviceAttributeComputeCapabilityMajor:
            *pi = prop->major; break;
        case hipDeviceAttributeComputeCapabilityMinor:
            *pi = prop->minor; break;
        case hipDeviceAttributePciBusId:
            *pi = prop->pciBusID; break;
        case hipDeviceAttributeConcurrentKernels:
            *pi = prop->concurrentKernels; break;
        case hipDeviceAttributePciDeviceId:
            *pi = prop->pciDeviceID; break;
        case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
            *pi = prop->maxSharedMemoryPerMultiProcessor; break;
        default:
            e = hipErrorInvalidValue; break;
        }
    } else {
        e = hipErrorInvalidDevice;
    }
    return ihipLogStatus(e);
}


/**
 * @return #hipSuccess, #hipErrorInvalidDevice
 * @bug HCC always returns 0 for maxThreadsPerMultiProcessor
 * @bug HCC always returns 0 for regsPerBlock
 * @bug HCC always returns 0 for l2CacheSize
 */
hipError_t hipDeviceGetProperties(hipDeviceProp_t* props, int device)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e;

    ihipDevice_t * hipDevice = ihipGetDevice(device);
    if (hipDevice) {
        // copy saved props
        *props = hipDevice->_props;
        e = hipSuccess;
    } else {
        e = hipErrorInvalidDevice;
    }

    return ihipLogStatus(e);
}




//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Error Handling
//---
/**
 * @returns return code from last HIP called from the active host thread.
 */
hipError_t hipGetLastError()
{
    std::call_once(hip_initialized, ihipInit);

    // Return last error, but then reset the state:
    return tls_lastHipError;
    ihipLogStatus(hipSuccess);
}

hipError_t hipPeakAtLastError()
{
    std::call_once(hip_initialized, ihipInit);

    return tls_lastHipError;
    ihipLogStatus(tls_lastHipError);
}


//---
const char *hipGetErrorName(hipError_t hip_error)
{
    std::call_once(hip_initialized, ihipInit);

    switch (hip_error) {
        case hipSuccess                     : return "hipSuccess";
        case hipErrorMemoryAllocation       : return "hipErrorMemoryAllocation";
        case hipErrorMemoryFree             : return "hipErrorMemoryFree";
        case hipErrorUnknownSymbol          : return "hipErrorUnknownSymbol";
        case hipErrorOutOfResources         : return "hipErrorOutOfResources";
        case hipErrorInvalidValue           : return "hipErrorInvalidValue";
        case hipErrorInvalidResourceHandle  : return "hipErrorInvalidResourceHandle";
        case hipErrorInvalidDevice          : return "hipErrorInvalidDevice";
        case hipErrorNoDevice               : return "hipErrorNoDevice";
        case hipErrorNotReady               : return "hipErrorNotReady";
        case hipErrorUnknown                : return "hipErrorUnknown";
        case hipErrorTbd                    : return "hipErrorTbd";
        default                             : return "hipErrorUnknown";
    };
}


/**
 * @warning : hipGetErrorString returns string from hipGetErrorName
 */

//---
const char *hipGetErrorString(hipError_t hip_error)
{
    std::call_once(hip_initialized, ihipInit);

    // TODO - return a message explaining the error.
    // TODO - This should be set up to return the same string reported in the the doxygen comments, somehow.
    return hipGetErrorName(hip_error);
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Stream
//

//---
hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags)
{
    std::call_once(hip_initialized, ihipInit);

    ihipDevice_t *device = ihipGetTlsDefaultDevice();
    hc::accelerator acc = device->_acc;

    // TODO - se try-catch loop to detect memory exception?
    //
    //
    //Note this is an execute_in_order queue, so all kernels submitted will atuomatically wait for prev to complete:
    //This matches CUDA stream behavior:

    auto istream = new ihipStream_t(device->_device_index, acc.create_view(), flags);
    device->_streams.push_back(istream);
    *stream = istream;
    tprintf(TRACE_SYNC, "hipStreamCreate, stream=%p\n", *stream);

    return ihipLogStatus(hipSuccess);
}

/**
 * @bug This function conservatively waits for all work in the specified stream to complete.
 */
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags)
{

    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    {
        // Super-conservative version of this - TODO - remove me:
        stream->_av.wait();
        e = hipSuccess;
    }

    return ihipLogStatus(e);
};


hipError_t hipStreamSynchronize(hipStream_t stream)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    if (stream == NULL) {
        ihipDevice_t *device = ihipGetTlsDefaultDevice();
        ihipWaitNullStream(device);
    } else {
        stream->_av.wait();
        e = hipSuccess;
    }


    return ihipLogStatus(e);
};


//---
/**
 * @return #hipSuccess, #hipErrorInvalidResourceHandle
 */
hipError_t hipStreamDestroy(hipStream_t stream)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    if (ihipIsValidDevice(stream->_device_index)) {

        ihipDevice_t *device = &g_devices[stream->_device_index];

        device->_streams.remove(stream);

        delete stream;

        e = hipSuccess;
    } else {
        e = hipErrorInvalidResourceHandle;
    }

    return ihipLogStatus(hipSuccess);
}


//---
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags)
{
    std::call_once(hip_initialized, ihipInit);

    if (flags == NULL) {
        return ihipLogStatus(hipErrorInvalidValue);
    } else if (stream == NULL) {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    } else {
        *flags = stream->_flags;
        return ihipLogStatus(hipSuccess);
    }
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Events
//---
/**
 * @warning : flags must be 0.
 */
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags)
{
    // TODO - support hipEventDefault, hipEventBlockingSync, hipEventDisableTiming
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    if (flags == 0) {
        ihipEvent_t *eh = event->_handle = new ihipEvent_t();

        eh->_state  = hipEventStatusCreated;
        eh->_stream = NULL;
        eh->_flags  = flags;
    } else {
        e = hipErrorInvalidValue;
    }


    return ihipLogStatus(e);
}


//---
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)
{
    std::call_once(hip_initialized, ihipInit);

    ihipEvent_t *eh = event._handle;
    if (eh && eh->_state != hipEventStatusUnitialized)   {
        eh->_stream = stream;

        if (stream == NULL) {
            // If stream == NULL, wait on all queues.
            // This matches behavior described in CUDA 7 RT APIs, which say that "This function uses standard default stream semantics".
            // TODO-HCC fix this - is CUDA this conservative or still uses device timestamps?
            ihipDevice_t *device = ihipGetTlsDefaultDevice();
            ihipWaitNullStream(device);


            eh->_timestamp = hc::get_system_ticks();
            eh->_state = hipEventStatusRecorded;
            return ihipLogStatus(hipSuccess);
        } else {
            eh->_state  = hipEventStatusRecording;
            // Clear timestamps
            eh->_timestamp = 0;
            eh->_marker = stream->_av.create_marker();

            return ihipLogStatus(hipSuccess);
        }
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
}


//---
hipError_t hipEventDestroy(hipEvent_t event)
{
    std::call_once(hip_initialized, ihipInit);

    event._handle->_state  = hipEventStatusUnitialized;

    delete event._handle;
    event._handle = NULL;

    // TODO - examine return additional error codes
    return ihipLogStatus(hipSuccess);
}


//---
hipError_t hipEventSynchronize(hipEvent_t event)
{
    std::call_once(hip_initialized, ihipInit);

    ihipEvent_t *eh = event._handle;

    if (eh) {
        if (eh->_state == hipEventStatusUnitialized) {
            return ihipLogStatus(hipErrorInvalidResourceHandle);
        } else if (eh->_state == hipEventStatusCreated ) {
            // Created but not actually recorded on any device:
            return ihipLogStatus(hipSuccess);
        } else if (eh->_stream == NULL) {
            ihipDevice_t *device = ihipGetTlsDefaultDevice();
            ihipWaitNullStream(device);
            return ihipLogStatus(hipSuccess);
        } else {
#if __hcc_workweek__ >= 16033
            eh->_marker.wait((eh->_flags & hipEventBlockingSync) ? hc::hcWaitModeBlocked : hc::hcWaitModeActive);
#else
            eh->_marker.wait();
#endif
            return ihipLogStatus(hipSuccess);
        }
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
}


void ihipSetTs(hipEvent_t e)
{
    ihipEvent_t *eh = e._handle;
    if (eh->_state == hipEventStatusRecorded) {
        // already recorded, done:
        return;
    } else {
        // Test this code:
        hsa_signal_t *sig  = static_cast<hsa_signal_t*> (eh->_marker.get_native_handle());
        if (sig) {
            if (hsa_signal_load_acquire(*sig) == 0) {
                eh->_timestamp = eh->_marker.get_end_tick();
                eh->_state = hipEventStatusRecorded;
            }
        }
    }
}


//---
hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop)
{
    std::call_once(hip_initialized, ihipInit);

    ihipEvent_t *start_eh = start._handle;
    ihipEvent_t *stop_eh = stop._handle;

    ihipSetTs(start);
    ihipSetTs(stop);

    hipError_t status = hipSuccess;
    *ms = 0.0f;

    if (start_eh && stop_eh) {
        if ((start_eh->_state == hipEventStatusRecorded) && (stop_eh->_state == hipEventStatusRecorded)) {
            // Common case, we have good information for both events.

            int64_t tickDiff = (stop_eh->_timestamp - start_eh->_timestamp);

            // TODO-move this to a variable saved with each agent.
            uint64_t freqHz;
            hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freqHz);
            if (freqHz) {
                *ms = ((double)(tickDiff) /  (double)(freqHz)) * 1000.0f;
                status = hipSuccess;
            } else {
                * ms = 0.0f;
                status = hipErrorInvalidValue;
            }


        } else if ((start_eh->_state == hipEventStatusRecording) ||
                   (stop_eh->_state  == hipEventStatusRecording)) {
            status = hipErrorNotReady;
        } else if ((start_eh->_state == hipEventStatusUnitialized) ||
                   (stop_eh->_state  == hipEventStatusUnitialized)) {
            status = hipErrorInvalidResourceHandle;
        }
    }

    return ihipLogStatus(status);
}


//---
hipError_t hipEventQuery(hipEvent_t event)
{
    std::call_once(hip_initialized, ihipInit);

    ihipEvent_t *eh = event._handle;

    // TODO-stream - need to read state of signal here:  The event may have become ready after recording..
    // TODO-HCC - use get_hsa_signal here.

    if (eh->_state == hipEventStatusRecording) {
        return ihipLogStatus(hipErrorNotReady);
    } else {
        return ihipLogStatus(hipSuccess);
    }
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Memory
//
//

// kernel for launching memcpy operations:
template <typename T>
hc::completion_future
ihipMemcpyKernel(hipStream_t stream, T * c, const T * a, size_t sizeBytes)
{
    int wg = std::min((unsigned)8, g_devices[stream->_device_index]._compute_units);
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
    int wg = std::min((unsigned)8, g_devices[stream->_device_index]._compute_units);
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
hipError_t hipMalloc(void** ptr, size_t sizeBytes)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t  hip_status = hipSuccess;

    const unsigned am_flags = 0;
    *ptr = hc::am_alloc(sizeBytes, ihipGetTlsDefaultDevice()->_acc, am_flags);

    if (*ptr == NULL) {
        hip_status = hipErrorMemoryAllocation;
    } else {
        hip_status = hipSuccess;
    }

    ihipLogStatus(hip_status);

    return hip_status;
}


hipError_t hipMallocHost(void** ptr, size_t sizeBytes)
{
    std::call_once(hip_initialized, ihipInit);

#if USE_PINNED_HOST

    const unsigned am_flags = amHostPinned;

    *ptr = hc::am_alloc(sizeBytes, ihipGetTlsDefaultDevice()->_acc, am_flags);
    hipError_t  hip_status = hipSuccess;
    if (*ptr == NULL) {
        hip_status = hipErrorMemoryAllocation;
    } else {
        hip_status = hipSuccess;
    }

    tprintf (TRACE_MEM, "  %s: pinned ptr=%p\n", __func__, *ptr);

    ihipLogStatus(hip_status);

    return hip_status;

#else
    // TODO-hcc remove-me

    // This code only works on Kaveri:
    *ptr = malloc(sizeBytes); // TODO - call am_alloc for device memory, this will only on KV HSA.
    if (*ptr != NULL) {
        //TODO-hsart : need memory pin APIs to implement this correctly.
        // FOr now do our best to allocate the memory, but return an error since
        // the returned pointer can only be used on the HOST not the GPU.
        return ihipLogStatus(hipErrorMemoryAllocation);
    } else {
        return ihipLogStatus(hipErrorMemoryAllocation);
    }
#endif
}

hipError_t hipMemcpyToSymbol(const char* symbolName, const void *src, size_t count, size_t offset, hipMemcpyKind kind)
{
#ifdef USE_MEMCPYTOSYMBOL
    if(kind != hipMemcpyHostToDevice) {
        return ihipLogStatus(hipErrorInvalidValue);
    }
    auto device = ihipGetTlsDefaultDevice();
    hc::completion_future marker;
    ihipCheckCommandSwitchSync(device._null_stream, ihipCommandData, &marker);
    device->_acc.memcpy_symbol(symbolName, (void*) src,count, offset);
#endif
    return ihipLogStatus(hipSuccess);
}


//---
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
{
    std::call_once(hip_initialized, ihipInit);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;
    ihipCheckCommandSwitchSync(stream, ihipCommandData, &marker);

    hipError_t e = hipSuccess;

#ifdef USE_ASYNC_COPY
    if (ihipIsValidDevice(stream->_device_index)) {

        ihipDevice_t *device = &g_devices[stream->_device_index];

        hsa_signal_t completion_signal; // init/obtain from pool.

        hsa_status_t hsa_status = hsa_amd_memory_async_copy(dst, src, size, device->_hsa_agent, 0, NULL, &completion_signal);

        e = (hsa_status == HSA_STATUS_SUCCESS) ? hipSuccess : hipErrorTbd;
    } else {
        e = hipErrorInvalidResourceHandle;
    }
        

#else
    // TODO-hsart - what synchronization does hsa_copy provide?
    hc::am_copy(dst, src, sizeBytes);
    e = hipSuccess;
#endif

    // TODO - when am_copy becomes async, and we have HIP_LAUNCH_BLOCKING set, then we would wait for copy operation to complete here.

    return ihipLogStatus(e);
}


//---
/*
 * @warning on HCC hipMemcpyAsync uses a synchronous copy.
 */
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    stream =  ihipSyncAndResolveStream(stream);

    hc::completion_future marker;
    ihipCheckCommandSwitchSync(stream, ihipCommandData, &marker);

    // Dispatch async memory copy to synchronize with items in the specified stream.

    // Async - need to set up dependency on the last command queued to the device?

    // TODO-hsart This routine needs to ensure that dst and src are mapped on the GPU.
    // This is a synchronous copy - remove and replace with code below when we have appropriate LOCK APIs.
    hc::am_copy(dst, src, sizeBytes);

#if 0

    hipStream_t s =ihipGetStream(stream);


    if (s) {
        hc::completion_future cf = ihipMemcpyKernel<char> (s, static_cast<char*> (dst), static_cast<const char*> (src), sizeBytes);

        //cf.wait();

        e =  hipSuccess;
    } else {
        e = hipErrorInvalidValue;
    }
#endif

    // TODO - if am_copy becomes async, and we have HIP_LAUNCH_BLOCKING set, then we would wait for copy operation to complete here.

    return ihipLogStatus(e);
}


// TODO-sync: function is async unless target is pinned host memory - then these are fully sync.
hipError_t hipMemsetAsync(void* dst, int  value, size_t sizeBytes, hipStream_t stream )
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    stream =  ihipSyncAndResolveStream(stream);
    hc::completion_future marker;
    ihipCheckCommandSwitchSync(stream, ihipCommandData, &marker);


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


    if (HIP_LAUNCH_BLOCKING) {
        tprintf (TRACE_SYNC, "'%s' LAUNCH_BLOCKING wait for completion [stream:%p].\n", __func__, (void*)stream);
        cf.wait();
        tprintf (TRACE_SYNC, "'%s' LAUNCH_BLOCKING completed [stream:%p].\n", __func__, (void*)stream);
    }


    return ihipLogStatus(e);
};


hipError_t hipMemset(void* dst, int  value, size_t sizeBytes )
{
    return hipMemsetAsync(dst, value, sizeBytes, hipStreamNull);
}


/*
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue (if free != NULL due to bug)
 * @bug - on hcc free always returns 50% of peak regardless of current allocations.  hipMemGetInfo returns hipErrorInvalidValue to indicate this.
 */
hipError_t hipMemGetInfo  (   size_t *    free, size_t *    total    )
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    ihipDevice_t * hipDevice = ihipGetTlsDefaultDevice();
    if (hipDevice) {
        if (total) {
            *total = hipDevice->_props.totalGlobalMem;
        }

        if (free) {
            *free =  hipDevice->_props.totalGlobalMem  * 0.5; // TODO
            e=hipErrorInvalidValue;
        }

    } else {
        e = hipErrorInvalidDevice;
    }

    // TODO-runtime  - when we fix the 50% bug.
    //return ihipLogStatus(hipErrorSuccess);
    return ihipLogStatus(hipErrorInvalidValue);
}


//---
hipError_t hipFree(void* ptr)
{
    std::call_once(hip_initialized, ihipInit);


   // Synchronize to ensure all work has finished.
    ihipWaitAllStreams(ihipGetTlsDefaultDevice());

    if (ptr) {
        hc::am_free(ptr);
    }

    return ihipLogStatus(hipSuccess);
}


hipError_t hipFreeHost(void* ptr)
{
    std::call_once(hip_initialized, ihipInit);

    if (ptr) {
#if USE_PINNED_HOST
        tprintf (TRACE_MEM, "  %s: %p\n", __func__, ptr);
        hc::am_free(ptr);
#else
        free(ptr);
#endif
    }

    return ihipLogStatus(hipSuccess);
};



/**
 * @warning HCC returns 0 in *canAccessPeer ; Need to update this function when RT supports P2P
 */
//---
hipError_t hipDeviceCanAccessPeer ( int* canAccessPeer, int  device, int  peerDevice )
{
    std::call_once(hip_initialized, ihipInit);
    *canAccessPeer = false;
    return ihipLogStatus(hipSuccess);
}


/**
 * @warning Need to update this function when RT supports P2P
 */
//---
hipError_t  hipDeviceDisablePeerAccess ( int  peerDevice )
{
    std::call_once(hip_initialized, ihipInit);
    // TODO-p2p
    return ihipLogStatus(hipSuccess);
};


/**
 * @warning Need to update this function when RT supports P2P
 */
//---
hipError_t  hipDeviceEnablePeerAccess ( int  peerDevice, unsigned int  flags )
{
    std::call_once(hip_initialized, ihipInit);
    // TODO-p2p
    return ihipLogStatus(hipSuccess);
}


//---
hipError_t hipMemcpyPeer ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t sizeBytes )
{
    std::call_once(hip_initialized, ihipInit);
    // HCC has a unified memory architecture so device specifiers are not required.
    return hipMemcpy(dst, src, sizeBytes, hipMemcpyDefault);
};


/**
 * @bug This function uses a synchronous copy
 */
//---
hipError_t hipMemcpyPeerAsync ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t sizeBytes, hipStream_t stream )
{
    std::call_once(hip_initialized, ihipInit);
    // HCC has a unified memory architecture so device specifiers are not required.
    return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream);
};


/**
 * @return #hipSuccess
 */
//---
hipError_t hipDriverGetVersion(int *driverVersion)
{
    std::call_once(hip_initialized, ihipInit);
    *driverVersion = 4;
    return ihipLogStatus(hipSuccess);
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
    std::call_once(hip_initialized, ihipInit);

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
    std::call_once(hip_initialized, ihipInit);

    if (stream == hipStreamNull ) {
        ihipDevice_t *device = ihipGetTlsDefaultDevice();
        stream = device->_null_stream;
    }

    *av = &(stream->_av);

    hipError_t err = hipSuccess;
    return ihipLogStatus(err);
}
