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
#include <deque>

#include <hc.hpp>
#include <hc_am.hpp>

#include "hip_runtime.h"

#include "hsa_ext_amd.h"



#define USE_AM_TRACKER 1  /* >0 = use new AM memory tracker features. */
#define USE_ROCR_V2    1  /* use the ROCR v2 async copy API with dst and src agents */

#if (USE_AM_TRACKER) and (__hcc_workweek__ < 16074)
#error (USE_AM_TRACKER requries HCC version of 16074 or newer)
#endif


#if (USE_ROCR_V2) and (USE_AM_TRACKER == 0)
#error (USE_ROCR_V2 requires USE_AM_TRACKER>0)
#endif


#define INLINE static inline

//---
// Environment variables:

// Intended to distinguish whether an environment variable should be visible only in debug mode, or in debug+release.
//static const int debug = 0;
static const int release = 1;


int HIP_LAUNCH_BLOCKING = 0;

int HIP_PRINT_ENV = 0;
int HIP_TRACE_API= 0;
int HIP_STAGING_SIZE = 64;   /* size of staging buffers, in KB */
int HIP_STAGING_BUFFERS = 2;    // TODO - remove, two buffers should be enough.
int HIP_PININPLACE = 0;  
int HIP_STREAM_SIGNALS = 2;  /* number of signals to allocate at stream creation */


//---
// Chicken bits for disabling functionality to work around potential issues:
int HIP_DISABLE_HW_KERNEL_DEP = 1;
int HIP_DISABLE_HW_COPY_DEP = 1;

int HIP_DISABLE_BIDIR_MEMCPY = 0;
int HIP_ONESHOT_COPY_DEP     = 1;  // TODO - setting this =1  is a good thing, reduces input deps on 


//---
//Debug flags:
#define TRACE_API    0x01 /* trace API calls and return values */
#define TRACE_SYNC   0x02 /* trace synchronization pieces */
#define TRACE_MEM    0x04 /* trace memory allocation / deallocation */
#define TRACE_COPY2  0x08 /* trace memory copy commands. Detailed. */
#define TRACE_SIGNAL 0x10 /* trace signal pool commands */

#define tprintf(trace_level, ...) {\
    if (HIP_TRACE_API & trace_level) {\
        fprintf (stderr, "hiptrace%x: ", trace_level); \
        fprintf (stderr, __VA_ARGS__);\
    }\
}

const hipStream_t hipStreamNull = 0x0;

struct ihipDevice_t;


enum ihipCommand_t {
    ihipCommandKernel,
    ihipCommandCopyH2D,
    ihipCommandCopyD2H,
};

const char* ihipCommandName[] = {
    "Kernel", "CopyH2D", "CopyD2H"
};



typedef uint64_t SIGSEQNUM;

//---
// Small wrapper around signals.
// Designed to be used from stream.
// TODO-someday refactor this class so it can be stored in a vector<>  
// we already store the index here so we can use for garbage collection.
struct ihipSignal_t {
    hsa_signal_t   _hsa_signal; // hsa signal handle
    int            _index;      // Index in pool, used for garbage collection.
    SIGSEQNUM      _sig_id;     // unique sequentially increasing ID.

    ihipSignal_t(); 
    ~ihipSignal_t();

    inline void release();
};


// Used to remove lock, for performance or stimulating bugs.
class FakeMutex
{
  public:
    void lock() {  }
    bool try_lock() {return true; }
    void unlock() { }
};



// Internal stream structure.
class ihipStream_t {
public:

    ihipStream_t(unsigned device_index, hc::accelerator_view av, unsigned int flags);
    ~ihipStream_t();

    inline void                 reclaimSignals(SIGSEQNUM sigNum);
    inline void                 waitAndReclaimOlder(ihipSignal_t *signal);
    inline void                 wait();

    inline ihipDevice_t *       getDevice() const;

    ihipSignal_t *              getSignal() ;

    inline bool                 preKernelCommand();
    inline void                 postKernelCommand(hc::completion_future &kernel_future);
    inline int                  copyCommand(ihipSignal_t *lastCopy, hsa_signal_t *waitSignal, ihipCommand_t copyType);

    inline void                 resetToEmpty();

    inline SIGSEQNUM            lastCopySeqId() { return _last_copy_signal ? _last_copy_signal->_sig_id : 0; };
    std::mutex &                mutex() {return _mutex;};

    //---
    hc::accelerator_view _av;
    unsigned            _flags;
private:
    void                        enqueueBarrier(hsa_queue_t* queue, ihipSignal_t *depSignal);

    unsigned                    _device_index;
    ihipCommand_t               _last_command_type;  // type of the last command

    // signal of last copy command sent to the stream.  
    // May be NULL, indicating the previous command has completley finished and future commands don't need to create a dependency.
    // Copy can be either H2D or D2H.
    ihipSignal_t                *_last_copy_signal;  
    hc::completion_future       _last_kernel_future;  // Completion future of last kernel command sent to GPU.
    
    int                         _signalCursor;
    
    SIGSEQNUM                   _stream_sig_id;      // Monotonically increasing unique signal id.
    SIGSEQNUM                   _oldest_live_sig_id; // oldest live seq_id, anything < this can be allocated.
    std::deque<ihipSignal_t>    _signalPool;   // Pool of signals for use by this stream.

    std::mutex                  _mutex;
};



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

    SIGSEQNUM             _copy_seq_id;
} ;


//-------------------------------------------------------------------------------------------------
struct StagingBuffer {

    static const int _max_buffers = 4;

    StagingBuffer(ihipDevice_t *device, size_t bufferSize, int numBuffers) ;
    ~StagingBuffer();

    void CopyHostToDevice(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);
    void CopyHostToDevicePinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);

    void CopyDeviceToHost   (void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);
    void CopyDeviceToHostPinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);


private:
    ihipDevice_t    *_device;
    size_t          _bufferSize;  // Size of the buffers.
    int             _numBuffers;

    char            *_pinnedStagingBuffer[_max_buffers];
    hsa_signal_t     _completion_signal[_max_buffers];
};



//-------------------------------------------------------------------------------------------------
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

    hsa_signal_t             _copy_signal;       // signal to use for synchronous memcopies
    std::mutex               _copy_lock[2];      // mutex for each direction.
    StagingBuffer           *_staging_buffer[2]; // one buffer for each direction.

public:
    void reset();
    void init(unsigned device_index, hc::accelerator acc);
    hipError_t getProperties(hipDeviceProp_t* prop);

    ~ihipDevice_t();
};


//=================================================================================================
// Global Data Structures:
//=================================================================================================
//TLS - must be initialized here.
thread_local hipError_t tls_lastHipError = hipSuccess;
thread_local int tls_defaultDevice = 0;

// Global initialization.
std::once_flag hip_initialized;
ihipDevice_t *g_devices; // Array of all non-emulated (ie GPU) accelerators in the system.
unsigned g_deviceCnt;
//=================================================================================================


//=================================================================================================
//Forward Declarations:
//=================================================================================================
INLINE bool ihipIsValidDevice(unsigned deviceIndex);

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
      throw;
  }
  tprintf (TRACE_SIGNAL, "  allocated hsa_signal=%lu\n", (_hsa_signal.handle));
}

//---
ihipSignal_t::~ihipSignal_t() 
{
    tprintf (TRACE_SIGNAL, "  destroy hsa_signal #%lu (#%lu)\n", (_hsa_signal.handle), _sig_id);
    if (hsa_signal_destroy(_hsa_signal) != HSA_STATUS_SUCCESS) {
        throw; // TODO
    }
};



//=================================================================================================
// ihipStream_t:
//=================================================================================================
//---
ihipStream_t::ihipStream_t(unsigned device_index, hc::accelerator_view av, unsigned int flags) :
    _av(av), 
    _flags(flags), 
    _device_index(device_index), 
    _last_copy_signal(0),
    _signalCursor(0),
    _stream_sig_id(0),
    _oldest_live_sig_id(1)
{
    tprintf(TRACE_SYNC, " streamCreate: stream=%p\n", this);
    _signalPool.resize(HIP_STREAM_SIGNALS > 0 ? HIP_STREAM_SIGNALS : 1);

    resetToEmpty();
};


//---
ihipStream_t::~ihipStream_t()
{
    _signalPool.clear();
}


//---
// Reset the stream to "empty" - next command will not set up an inpute dependency on any older signal.
void ihipStream_t::resetToEmpty() 
{
    _last_command_type = ihipCommandCopyH2D;
    _last_copy_signal = NULL;
}

//---
void ihipStream_t::reclaimSignals(SIGSEQNUM sigNum)
{
    tprintf(TRACE_SIGNAL, "reclaim signal #%lu\n", sigNum);
    // Mark all signals older and including this one as available for 
    _oldest_live_sig_id = sigNum+1;
}


//---
void ihipStream_t::waitAndReclaimOlder(ihipSignal_t *signal) 
{
    hsa_signal_wait_acquire(_last_copy_signal->_hsa_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

    reclaimSignals(_last_copy_signal->_sig_id); 

}


//---
//Wait for all queues kernels in the associated accelerator_view to complete.
void ihipStream_t::wait() 
{
    tprintf (TRACE_SYNC, "stream %p wait for queue-empty and lastCopy:#%lu...\n", this, _last_copy_signal ? _last_copy_signal->_sig_id: 0x0 );
    _av.wait();
    if (_last_copy_signal) {
        this->waitAndReclaimOlder(_last_copy_signal);
    }

    resetToEmpty();
};


//---
inline ihipDevice_t * ihipStream_t::getDevice() const 
{ 
    if (ihipIsValidDevice(_device_index)) {
        return &g_devices[_device_index]; 
    } else {
        return NULL;
    }
};


//---
// Allocate a new signal from the signal pool.
// Returned signals have value of 0.
// Signals are intended for use in this stream and are always reclaimed "in-order".
ihipSignal_t *ihipStream_t::getSignal() 
{
    int numToScan = _signalPool.size();
    do {
        auto thisCursor = _signalCursor;
        if (++_signalCursor == _signalPool.size()) {
            _signalCursor = 0;
        }

        if (_signalPool[thisCursor]._sig_id < _oldest_live_sig_id) {
            _signalPool[thisCursor]._index = thisCursor;
            _signalPool[thisCursor]._sig_id  =  ++_stream_sig_id;  // allocate it.


            return &_signalPool[thisCursor];
        }

    } while (--numToScan) ;

    assert(numToScan == 0);

    // Have to grow the pool:
    _signalCursor = _signalPool.size(); // set to the beginning of the new entries:
    _signalPool.resize(_signalPool.size() * 2);
    tprintf (TRACE_SIGNAL, "grow signal pool to %zu entries, cursor=%d\n", _signalPool.size(), _signalCursor);
    return getSignal();  // try again, 

    // Should never reach here.
    assert(0);
}


//---
void ihipStream_t::enqueueBarrier(hsa_queue_t* queue, ihipSignal_t *depSignal) 
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

    barrier->dep_signal[0] = depSignal->_hsa_signal;

    barrier->completion_signal.handle = 0;

    // TODO - check queue overflow, return error:
    // Increment write index and ring doorbell to dispatch the kernel
    hsa_queue_store_write_index_relaxed(queue, index+1);
    hsa_signal_store_relaxed(queue->doorbell_signal, index);
}


//--
//When the commands in a stream change types (ie kernel command follows a data command,
//or data command follows a kernel command), then we need to add a barrier packet
//into the stream to mimic CUDA stream semantics.  (some hardware uses separate
//queues for data commands and kernel commands, and no implicit ordering is provided).
//
inline bool ihipStream_t::preKernelCommand()
{
    _mutex.lock(); // will be unlocked in postKernelCommand

    bool addedSync = false;
    // If switching command types, we need to add a barrier packet to synchronize things.
    if (_last_command_type != ihipCommandKernel) {
        if (_last_copy_signal) {
            addedSync = true;

            hsa_queue_t * q =  (hsa_queue_t*)_av.get_hsa_queue();
            if (! HIP_DISABLE_HW_KERNEL_DEP) {
                this->enqueueBarrier(q, _last_copy_signal);
                tprintf (TRACE_SYNC, "stream %p switch %s to %s (barrier pkt inserted with wait on #%lu)\n", 
                        this, ihipCommandName[_last_command_type], ihipCommandName[ihipCommandKernel], _last_copy_signal->_sig_id)

            } else {
                tprintf (TRACE_SYNC, "stream %p switch %s to %s (wait for previous...)\n", 
                        this, ihipCommandName[_last_command_type], ihipCommandName[ihipCommandKernel]);
                this->waitAndReclaimOlder(_last_copy_signal);
            }
        }
        _last_command_type = ihipCommandKernel;
    }

    return addedSync;
}


//---
inline void ihipStream_t::postKernelCommand(hc::completion_future &kernelFuture)
{
    _last_kernel_future = kernelFuture;

    _mutex.unlock();
};



//---
// Called whenever a copy command is set to the stream.
// Examines the last command sent to this stream and returns a signal to wait on, if required.
inline int ihipStream_t::copyCommand(ihipSignal_t *lastCopy, hsa_signal_t *waitSignal, ihipCommand_t copyType)
{
    int needSync = 0;

    waitSignal->handle = 0;
    // If switching command types, we need to add a barrier packet to synchronize things.
    if (_last_command_type != copyType) {


        if (_last_command_type == ihipCommandKernel) {
            tprintf (TRACE_SYNC, "stream %p switch %s to %s (async copy dep on prev kernel)\n", 
                    this, ihipCommandName[_last_command_type], ihipCommandName[copyType]);
            needSync = 1;
            hsa_signal_t *hsaSignal = (static_cast<hsa_signal_t*> (_last_kernel_future.get_native_handle()));
            if (hsaSignal) {
                *waitSignal = * hsaSignal;
            }
        } else if (_last_copy_signal) {
            needSync = 1;
            tprintf (TRACE_SYNC, "stream %p switch %s to %s (async copy dep on other copy #%lu)\n", 
                    this, ihipCommandName[_last_command_type], ihipCommandName[copyType], _last_copy_signal->_sig_id);
            *waitSignal = _last_copy_signal->_hsa_signal;
        }

        if (HIP_DISABLE_HW_COPY_DEP && needSync) {
            // do the wait here on the host, and disable the device-side command resolution.
            hsa_signal_wait_acquire(*waitSignal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
            needSync = 0;
        }

        _last_command_type = copyType;
    }

    _last_copy_signal = lastCopy;

    return needSync;
}


//=================================================================================================
//
//Reset the device - this is called from hipDeviceReset.  
//Device may be reset multiple times, and may be reset after init.
void ihipDevice_t::reset() 
{
    _staging_buffer[0] = new StagingBuffer(this, HIP_STAGING_SIZE*1024, HIP_STAGING_BUFFERS);
    _staging_buffer[1] = new StagingBuffer(this, HIP_STAGING_SIZE*1024, HIP_STAGING_BUFFERS);
};


//---
void ihipDevice_t::init(unsigned device_index, hc::accelerator acc)
{
    _device_index = device_index;
    _acc = acc;
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

    hsa_signal_create(0, 0, NULL, &_copy_signal);

    this->reset();
};


ihipDevice_t::~ihipDevice_t()
{
    if (_null_stream) {
        delete _null_stream;
        _null_stream = NULL;
    }

    for (int i=0; i<2; i++) {
        if (_staging_buffer[i]) {
            delete _staging_buffer[i];
        }
    }
    hsa_signal_destroy(_copy_signal);
}

//----




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

#ifdef USE_ROCR_20
    // Get Max memory clock frequency
    err = hsa_region_get_info(*am_region, (hsa_region_info_t)HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY, &prop->memoryClockRate);
    DeviceErrorCheck(err);
    prop->memoryClockRate *= 1000.0;   // convert Mhz to Khz.

    // Get global memory bus width in bits
    err = hsa_region_get_info(*am_region, (hsa_region_info_t)HSA_AMD_REGION_INFO_BUS_WIDTH, &prop->memoryBusWidth);
    DeviceErrorCheck(err);
#endif

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
     * Environment variables
     */
    READ_ENV_I(release, HIP_PRINT_ENV, 0,  "Print HIP environment variables.");  
    //-- READ HIP_PRINT_ENV env first, since it has impact on later env var reading

    READ_ENV_I(release, HIP_LAUNCH_BLOCKING, CUDA_LAUNCH_BLOCKING, "Make HIP APIs 'host-synchronous', so they block until any kernel launches or data copy commands complete. Alias: CUDA_LAUNCH_BLOCKING." );
    READ_ENV_I(release, HIP_TRACE_API, 0,  "Trace each HIP API call.  Print function name and return code to stderr as program executes.");
    READ_ENV_I(release, HIP_STAGING_SIZE, 0, "Size of each staging buffer (in KB)" );
    READ_ENV_I(release, HIP_STAGING_BUFFERS, 0, "Number of staging buffers to use in each direction. 0=use hsa_memory_copy.");
    READ_ENV_I(release, HIP_PININPLACE, 0, "For unpinned transfers, pin the memory in-place in chunks before doing the copy");
    READ_ENV_I(release, HIP_STREAM_SIGNALS, 0, "Number of signals to allocate when new stream is created (signal pool will grow on demand)");

    READ_ENV_I(release, HIP_DISABLE_HW_KERNEL_DEP, 0, "Disable HW dependencies before kernel commands  - instead wait for dependency on host.");
    READ_ENV_I(release, HIP_DISABLE_HW_COPY_DEP, 0, "Disable HW dependencies before copy commands  - instead wait for dependency on host.");
    READ_ENV_I(release, HIP_DISABLE_BIDIR_MEMCPY, 0, "Disable simultaneous H2D memcpy and D2H memcpy to same device");
    READ_ENV_I(release, HIP_ONESHOT_COPY_DEP, 0, "If set, only set the copy input dependency for the first copy command in a staged copy.  If clear, set the dep for each copy.");

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

    g_devices = new ihipDevice_t[deviceCnt];
    g_deviceCnt = 0;
    for (int i=0; i<accs.size(); i++) {
        if (! accs[i].get_is_emulated()) {
           g_devices[g_deviceCnt].init(g_deviceCnt, accs[i]);
           g_deviceCnt++;
        }
    }

    assert(deviceCnt == g_deviceCnt);


    tprintf(TRACE_API, "pid=%u %-30s\n", getpid(), "<ihipInit>");

}

INLINE bool ihipIsValidDevice(unsigned deviceIndex)
{
    // deviceIndex is unsigned so always > 0
    return (deviceIndex < g_deviceCnt);
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
    if ((deviceId >= 0) && (deviceId < g_deviceCnt)) {
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
        (*streamI)->wait();
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
            stream->wait();
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







// TODO - data-up to data-down:
// Called just before a kernel is launched from hipLaunchKernel.
// Allows runtime to track some information about the stream.
hipStream_t ihipPreLaunchKernel(hipStream_t stream, hc::accelerator_view **av)
{
    stream = ihipSyncAndResolveStream(stream);

    stream->preKernelCommand();

    *av = &stream->_av;

    return (stream);
}


//---
//Called after kernel finishes execution.
void ihipPostLaunchKernel(hipStream_t stream, hc::completion_future &kernelFuture)
{
    stream->postKernelCommand(kernelFuture);
    if (HIP_LAUNCH_BLOCKING) {
        tprintf(TRACE_SYNC, " stream:%p LAUNCH_BLOCKING for kernel completion\n", stream);
    }
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

    *count = g_deviceCnt;

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

    if ((device < 0) || (device >= g_deviceCnt)) {
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
 */
hipError_t hipDeviceReset(void)
{
    std::call_once(hip_initialized, ihipInit);

    ihipDevice_t *device = ihipGetTlsDefaultDevice();

    // TODO-HCC
    // This function currently does a user-level cleanup of known resources.
    // It could benefit from KFD support to perform a more "nuclear" clean that would include any associated kernel resources and page table entries.


    //---
    //Wait for pending activity to complete?
    //TODO - check if this is required behavior:
    for (auto streamI=device->_streams.begin(); streamI!=device->_streams.end(); streamI++) {
        ihipStream_t *stream = *streamI;
        stream->wait();
    }

    // Reset and remove streams:
    device->_streams.clear();


#if USE_AM_TRACKER
    if (device) {
        am_memtracker_reset(device->_acc);
        device->reset(); // re-allocate required resources.
    }
#endif

	// TODO - reset all streams on the device.

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
#ifdef USE_ROCR_20
        case hipDeviceAttributeMemoryClockRate:
            *pi = prop->memoryClockRate; break;
        case hipDeviceAttributeMemoryBusWidth:
            *pi = prop->memoryBusWidth; break;
#endif
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
        case hipDeviceAttributeIsMultiGpuBoard:
            *pi = prop->isMultiGpuBoard; break;
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
        case hipErrorInvalidMemcpyDirection : return "hipErrorInvalidMemcpyDirection";
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
        stream->wait();
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
        stream->wait();
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

    //--- Drain the stream:
    if (stream == NULL) {
        ihipDevice_t *device = ihipGetTlsDefaultDevice();
        ihipWaitNullStream(device);
    } else {
        stream->wait();
        e = hipSuccess;
    }

    ihipDevice_t *device = stream->getDevice();

    if (device) {
        device->_streams.remove(stream);
        delete stream;
    } else {
        e = hipErrorInvalidResourceHandle;
    }

    return ihipLogStatus(e);
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
        eh->_timestamp  = 0;
        eh->_copy_seq_id  = 0;
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
            eh->_copy_seq_id = stream->lastCopySeqId();

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
            eh->_stream->reclaimSignals(eh->_copy_seq_id);

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
//

//---
/**
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
 */
hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, void* ptr) 
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

#if USE_AM_TRACKER
    hc::accelerator acc;
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
    am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
    if (status == AM_SUCCESS) {

        attributes->memoryType    = amPointerInfo._isInDeviceMem ? hipMemoryTypeDevice: hipMemoryTypeHost;
        attributes->hostPointer   = amPointerInfo._hostPointer;
        attributes->devicePointer = amPointerInfo._devicePointer;
        attributes->isManaged     = 0;

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

        e = hipErrorInvalidValue;
    }
#else
    e = hipErrorInvalidValue;
#endif

    return ihipLogStatus(e);
}


#if USE_AM_TRACKER
// TODO - test this function:
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
    if (flags == 0) {
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
#endif



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
    std::call_once(hip_initialized, ihipInit);

    hipError_t  hip_status = hipSuccess;

	auto device = ihipGetTlsDefaultDevice();

    if (device) {
        const unsigned am_flags = 0;
        *ptr = hc::am_alloc(sizeBytes, device->_acc, am_flags);

        if (sizeBytes && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        } else {
#if USE_AM_TRACKER
            hc::am_memtracker_update(*ptr, device->_device_index, 0);
#endif
        }
    } else {
        hip_status = hipErrorMemoryAllocation;
    }

    return ihipLogStatus(hip_status);
}


hipError_t hipMallocHost(void** ptr, size_t sizeBytes)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t  hip_status = hipSuccess;

    const unsigned am_flags = amHostPinned;
	auto device = ihipGetTlsDefaultDevice();

    if (device) {
        *ptr = hc::am_alloc(sizeBytes, device->_acc, am_flags);
        if (sizeBytes && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        } else {
#if USE_AM_TRACKER
            hc::am_memtracker_update(*ptr, device->_device_index, 0);
#endif
        }

        tprintf (TRACE_MEM, "  %s: pinned ptr=%p\n", __func__, *ptr);
    }

    return ihipLogStatus(hip_status);
}

//---
hipError_t hipMemcpyToSymbol(const char* symbolName, const void *src, size_t count, size_t offset, hipMemcpyKind kind)
{
    std::call_once(hip_initialized, ihipInit);

#ifdef USE_MEMCPYTOSYMBOL
	if(kind != hipMemcpyHostToDevice)
	{
		return ihipLogStatus(hipErrorInvalidValue);
	}
	auto device = ihipGetTlsDefaultDevice();

    //hsa_signal_t depSignal;
    //int depSignalCnt = device._null_stream->copyCommand(NULL, &depSignal, ihipCommandCopyH2D);
    assert(0);  // Need to properly synchronize the copy - do something with depSignal if != NULL.

	device->_acc.memcpy_symbol(symbolName, (void*) src,count, offset);
#endif
    return ihipLogStatus(hipSuccess);
}


//-------------------------------------------------------------------------------------------------
StagingBuffer::StagingBuffer(ihipDevice_t *device, size_t bufferSize, int numBuffers) :
    _device(device),
    _bufferSize(bufferSize),
    _numBuffers(numBuffers > _max_buffers ? _max_buffers : numBuffers)
{
    

    
    for (int i=0; i<_numBuffers; i++) {
        // TODO - experiment with alignment here.
        _pinnedStagingBuffer[i] = hc::am_alloc(_bufferSize, device->_acc, amHostPinned);
        if (_pinnedStagingBuffer[i] == NULL) {
            throw;
        }
        hsa_signal_create(0, 0, NULL, &_completion_signal[i]);
    }
};

//---
StagingBuffer::~StagingBuffer()
{
    for (int i=0; i<_numBuffers; i++) {
        if (_pinnedStagingBuffer[i]) {
            hc::am_free(_pinnedStagingBuffer[i]);
            _pinnedStagingBuffer[i] = NULL;
        }
        hsa_signal_destroy(_completion_signal[i]);
    }
}



//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from host CPU.
//IN: src - src pointer for copy.  Must be accessible from agent this buffer is associated with (via _device)
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void StagingBuffer::CopyHostToDevicePinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor) 
{
    const char *srcp = static_cast<const char*> (src); 
    char *dstp = static_cast<char*> (dst); 

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completion_signal[i], 0);
    }

    assert(sizeBytes < UINT64_MAX/2); // TODO
    int bufferIndex = 0;
    for (int64_t bytesRemaining=sizeBytes; bytesRemaining>0 ;  bytesRemaining -= _bufferSize) {

        size_t theseBytes = (bytesRemaining > _bufferSize) ? _bufferSize : bytesRemaining;

        tprintf (TRACE_COPY2, "H2D: waiting... on completion signal handle=%lu\n", _completion_signal[bufferIndex].handle);
        hsa_signal_wait_acquire(_completion_signal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE); 

        tprintf (TRACE_COPY2, "H2D: bytesRemaining=%zu: pin-in-place:%p+%zu bufferIndex[%d]\n", bytesRemaining, srcp, theseBytes, bufferIndex);


        memcpy(_pinnedStagingBuffer[bufferIndex], srcp, theseBytes);
        void *locked_srcp;
        hsa_status_t hsa_status = hsa_amd_memory_lock(const_cast<char *> (srcp), theseBytes, &_device->_hsa_agent, 1, &locked_srcp);

        assert (hsa_status == HSA_STATUS_SUCCESS);

        hsa_signal_store_relaxed(_completion_signal[bufferIndex], 1);

#if USE_ROCR_V2
        hsa_status = hsa_amd_memory_async_copy(dstp, _device->_hsa_agent, locked_srcp, _device->_hsa_agent, theseBytes, waitFor ? 1:0, waitFor, _completion_signal[bufferIndex]);
#else
        assert(0);
#endif
        tprintf (TRACE_COPY2, "H2D: bytesRemaining=%zu: async_copy %zu bytes %p to %p status=%x\n", bytesRemaining, theseBytes, _pinnedStagingBuffer[bufferIndex], dstp, hsa_status);

        assert(hsa_status == HSA_STATUS_SUCCESS); // TODO - throw 

        srcp += theseBytes;
        dstp += theseBytes;
        if (++bufferIndex >= _numBuffers) {
            bufferIndex = 0;
        }

        if (HIP_ONESHOT_COPY_DEP) {
            waitFor = NULL; // TODO - don't need dependency after first copy submitted?
        }
    }

    // TODO - 
    printf ("unpin the memory\n");


    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_wait_acquire(_completion_signal[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE); 
    }
}




//---
//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from host CPU.
//IN: src - src pointer for copy.  Must be accessible from agent this buffer is associated with (via _device)
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void StagingBuffer::CopyHostToDevice(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor) 
{
    const char *srcp = static_cast<const char*> (src); 
    char *dstp = static_cast<char*> (dst); 

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completion_signal[i], 0);
    }

    assert(sizeBytes < UINT64_MAX/2); // TODO
    int bufferIndex = 0;
    for (int64_t bytesRemaining=sizeBytes; bytesRemaining>0 ;  bytesRemaining -= _bufferSize) {

        size_t theseBytes = (bytesRemaining > _bufferSize) ? _bufferSize : bytesRemaining;

        tprintf (TRACE_COPY2, "H2D: waiting... on completion signal handle=%lu\n", _completion_signal[bufferIndex].handle);
        hsa_signal_wait_acquire(_completion_signal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE); 

        tprintf (TRACE_COPY2, "H2D: bytesRemaining=%zu: copy %zu bytes %p to stagingBuf[%d]:%p\n", bytesRemaining, theseBytes, srcp, bufferIndex, _pinnedStagingBuffer[bufferIndex]);
        // TODO - use uncached memcpy, someday.
        memcpy(_pinnedStagingBuffer[bufferIndex], srcp, theseBytes);


        hsa_signal_store_relaxed(_completion_signal[bufferIndex], 1);

#if USE_ROCR_V2
        hsa_status_t hsa_status = hsa_amd_memory_async_copy(dstp, _device->_hsa_agent, _pinnedStagingBuffer[bufferIndex], _device->_hsa_agent, theseBytes, waitFor ? 1:0, waitFor, _completion_signal[bufferIndex]);
#else
        hsa_status_t hsa_status = hsa_amd_memory_async_copy(dstp, _pinnedStagingBuffer[bufferIndex], theseBytes, _device->_hsa_agent, 0, NULL, _completion_signal[bufferIndex]);
#endif
        tprintf (TRACE_COPY2, "H2D: bytesRemaining=%zu: async_copy %zu bytes %p to %p status=%x\n", bytesRemaining, theseBytes, _pinnedStagingBuffer[bufferIndex], dstp, hsa_status);

        assert(hsa_status == HSA_STATUS_SUCCESS); // TODO - throw 

        srcp += theseBytes;
        dstp += theseBytes;
        if (++bufferIndex >= _numBuffers) {
            bufferIndex = 0;
        }

        if (HIP_ONESHOT_COPY_DEP) {
            waitFor = NULL; // TODO - don't need dependency after first copy submitted?
        }
    }


    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_wait_acquire(_completion_signal[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE); 
    }
}

//---
//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from agent this buffer is assocaited with (via _device).
//IN: src - src pointer for copy.  Must be accessible from host CPU.
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void StagingBuffer::CopyDeviceToHost(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor) 
{
    const char *srcp0 = static_cast<const char*> (src); 
    char *dstp1 = static_cast<char*> (dst); 

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completion_signal[i], 0);
    }

    assert(sizeBytes < UINT64_MAX/2); // TODO

    int64_t bytesRemaining0 = sizeBytes; // bytes to copy from dest into staging buffer.
    int64_t bytesRemaining1 = sizeBytes; // bytes to copy from staging buffer into final dest

    while (bytesRemaining1 > 0) {
        // First launch the async copies to copy from dest to host
        for (int bufferIndex = 0; (bytesRemaining0>0) && (bufferIndex < _numBuffers);  bytesRemaining0 -= _bufferSize, bufferIndex++) {

            size_t theseBytes = (bytesRemaining0 > _bufferSize) ? _bufferSize : bytesRemaining0;

            tprintf (TRACE_COPY2, "D2H: bytesRemaining0=%zu  async_copy %zu bytes src:%p to staging:%p\n", bytesRemaining0, theseBytes, srcp0, _pinnedStagingBuffer[bufferIndex]);
            hsa_signal_store_relaxed(_completion_signal[bufferIndex], 1);
#if USE_ROCR_V2
            hsa_status_t hsa_status = hsa_amd_memory_async_copy(_pinnedStagingBuffer[bufferIndex], _device->_hsa_agent, srcp0, _device->_hsa_agent, theseBytes, waitFor ? 1:0, waitFor, _completion_signal[bufferIndex]);
#else
            hsa_status_t hsa_status = hsa_amd_memory_async_copy(_pinnedStagingBuffer[bufferIndex], srcp0, theseBytes, _device->_hsa_agent, 0, NULL, _completion_signal[bufferIndex]);
#endif
            assert(hsa_status == HSA_STATUS_SUCCESS); // TODO - throw 

            srcp0 += theseBytes;

           
            if (HIP_ONESHOT_COPY_DEP) {
                waitFor = NULL; // TODO - don't need dependency after first copy submitted?
            }
        }

        // Now unload the staging buffers:
        for (int bufferIndex=0; (bytesRemaining1>0) && (bufferIndex < _numBuffers);  bytesRemaining1 -= _bufferSize, bufferIndex++) {

            size_t theseBytes = (bytesRemaining1 > _bufferSize) ? _bufferSize : bytesRemaining1;

            tprintf (TRACE_COPY2, "D2H: wait_completion[%d] bytesRemaining=%zu\n", bufferIndex, bytesRemaining1);
            hsa_signal_wait_acquire(_completion_signal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE); 

            tprintf (TRACE_COPY2, "D2H: bytesRemaining1=%zu copy %zu bytes stagingBuf[%d]:%p to dst:%p\n", bytesRemaining1, theseBytes, bufferIndex, _pinnedStagingBuffer[bufferIndex], dstp1);
            memcpy(dstp1, _pinnedStagingBuffer[bufferIndex], theseBytes);

            dstp1 += theseBytes;
        }
    }


    //for (int i=0; i<_numBuffers; i++) {
    //    hsa_signal_wait_acquire(_completion_signal[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE); 
    //}
}




#if USE_AM_TRACKER
void ihipSyncCopy(ihipStream_t *stream, void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
{
    ihipDevice_t *device = stream->getDevice();

    if (device == NULL) {
        throw;
    }

    hc::accelerator acc;
    hc::AmPointerInfo dstPtrInfo(NULL, NULL, 0, acc, 0, 0);
    hc::AmPointerInfo srcPtrInfo(NULL, NULL, 0, acc, 0, 0);

    bool dstNotTracked = (hc::am_memtracker_getinfo(&dstPtrInfo, dst) != AM_SUCCESS);
    bool srcNotTracked = (hc::am_memtracker_getinfo(&srcPtrInfo, src) != AM_SUCCESS);


    // Resolve default to a specific Kind so we know which algorithm to use:
    if (kind == hipMemcpyDefault) {
        bool dstIsHost = (dstNotTracked || !dstPtrInfo._isInDeviceMem);
        bool srcIsHost = (srcNotTracked || !srcPtrInfo._isInDeviceMem);
        if (srcIsHost && !dstIsHost) {
            kind = hipMemcpyHostToDevice;
        } else if (!srcIsHost && dstIsHost) {
            kind = hipMemcpyDeviceToHost;
        } else if (srcIsHost && dstIsHost) {
            kind = hipMemcpyHostToHost;
        } else if (srcIsHost && dstIsHost) {
            kind = hipMemcpyDeviceToDevice;
        }
    }


    if ((kind == hipMemcpyHostToDevice) && (srcNotTracked)) {
        if (HIP_STAGING_BUFFERS) {
            std::lock_guard<std::mutex> l (device->_copy_lock[0]);
            //printf ("staged-copy- read dep signals\n");

            hsa_signal_t depSignal;
            int depSignalCnt = stream->copyCommand(NULL, &depSignal, ihipCommandCopyH2D);

            if (HIP_PININPLACE) {
                device->_staging_buffer[0]->CopyHostToDevicePinInPlace(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL); 
            } else  {
                device->_staging_buffer[0]->CopyHostToDevice(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL); 
            }

            // The copy waits for inputs and then completes before returning.
            stream->resetToEmpty();
        } else {
            // TODO - remove, slow path.
            hc::am_copy(dst, src, sizeBytes);
        }
    } else if ((kind == hipMemcpyDeviceToHost) && (dstNotTracked)) {
        if (HIP_STAGING_BUFFERS) {
            std::lock_guard<std::mutex> l (device->_copy_lock[HIP_DISABLE_BIDIR_MEMCPY ? 0:1]);
            //printf ("staged-copy- read dep signals\n");
            hsa_signal_t depSignal;
            int depSignalCnt = stream->copyCommand(NULL, &depSignal, ihipCommandCopyD2H);
            device->_staging_buffer[1]->CopyDeviceToHost(dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL); 
        } else {
            // TODO - remove, slow path.
            hc::am_copy(dst, src, sizeBytes);
        }
    } else if (kind == hipMemcpyHostToHost)  { // TODO-refactor.
        memcpy(dst, src, sizeBytes); 

    } else {
        ihipCommand_t copyType;
        if ((kind == hipMemcpyHostToDevice) || (kind == hipMemcpyDeviceToDevice)) {
            copyType = ihipCommandCopyH2D;
        } else if (kind == hipMemcpyDeviceToHost) {
            copyType = ihipCommandCopyD2H;
        } else {
            // TODO - return error condition:
            //e = hipErrorInvalidMemcpyDirection;
            copyType = ihipCommandCopyD2H;
        }

        device->_copy_lock[HIP_DISABLE_BIDIR_MEMCPY? 0:1].lock();

        hsa_signal_store_relaxed(device->_copy_signal, 1);

        hsa_signal_t depSignal;
        int depSignalCnt = stream->copyCommand(NULL, &depSignal, copyType);

#if USE_ROCR_V2
        hsa_status_t hsa_status = hsa_amd_memory_async_copy(dst, device->_hsa_agent, src, device->_hsa_agent, sizeBytes, depSignalCnt, depSignalCnt ? &depSignal:0x0, device->_copy_signal);
#else
        hsa_status_t hsa_status = hsa_amd_memory_async_copy(dst, src, sizeBytes, device->_hsa_agent, 0, NULL, device->_copy_signal);
#endif

        if (hsa_status == HSA_STATUS_SUCCESS) {
            hsa_signal_wait_relaxed(device->_copy_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE); 
        }

        device->_copy_lock[HIP_DISABLE_BIDIR_MEMCPY ? 0:1].unlock();

    }
}
#endif


//---
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
{
    std::call_once(hip_initialized, ihipInit);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

#if USE_AM_TRACKER
    try {
        ihipSyncCopy(stream, dst, src, sizeBytes, kind);
    } 
    catch (...) {
        e = hipErrorInvalidResourceHandle;
    }


#else
    hc::am_copy(dst, src, sizeBytes);
    e = hipSuccess;
#endif

    return ihipLogStatus(e);
}


#if USE_AM_TRACKER==0
/**
 * @warning on HCC hipMemcpyAsync uses a synchronous copy.
 */
#endif
/**
 * @result #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidMemcpyDirection, #hipErrorInvalidValue
 */
//---
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    stream = ihipSyncAndResolveStream(stream);

#if USE_AM_TRACKER
    if (stream) {
        ihipDevice_t *device = stream->getDevice();

        if (device == NULL) {
            e = hipErrorInvalidDevice;

        } else if (kind == hipMemcpyDefault) {
            e = hipErrorInvalidMemcpyDirection;

        } else if (kind == hipMemcpyHostToHost) {
            tprintf (TRACE_COPY2, "H2H copy with memcpy");

            memcpy(dst, src, sizeBytes);

        } else {
            ihipSignal_t *ihip_signal = stream->getSignal();
            hsa_signal_store_relaxed(ihip_signal->_hsa_signal, 1);

            ihipCommand_t copyType;
            if ((kind == hipMemcpyHostToDevice) || (kind == hipMemcpyDeviceToDevice)) {
                copyType = ihipCommandCopyH2D;
            } else if (kind == hipMemcpyDeviceToHost) {
                copyType = ihipCommandCopyD2H;
            } else {
                e = hipErrorInvalidMemcpyDirection;
                copyType = ihipCommandCopyD2H;
            }

#if USE_ROCR_V2
            hsa_signal_t depSignal;
            int depSignalCnt = stream->copyCommand(ihip_signal, &depSignal, copyType);

            tprintf (TRACE_SYNC, " copy-async, waitFor=%lu completion=#%lu(%lu)\n", depSignalCnt? depSignal.handle:0x0, ihip_signal->_sig_id, ihip_signal->_hsa_signal.handle);

            hsa_status_t hsa_status = hsa_amd_memory_async_copy(dst, device->_hsa_agent, src, device->_hsa_agent, sizeBytes, depSignalCnt, depSignalCnt ? &depSignal:0x0, ihip_signal->_hsa_signal);
#else
            hsa_status_t hsa_status = hsa_amd_memory_async_copy(dst, src, sizeBytes, device->_hsa_agent, 0, NULL, ihip_signal->_hsa_signal);
#endif


            if (hsa_status == HSA_STATUS_SUCCESS) {
                // TODO-stream - fix release-signal calls here.
                if (HIP_LAUNCH_BLOCKING) {
                    tprintf(TRACE_SYNC, "LAUNCH_BLOCKING for completion of hipMemcpyAsync(%zu)\n", sizeBytes);
                    stream->wait();
                }  
            } else {
                // This path can be hit if src or dst point to unpinned host memory.
                // TODO-stream - does async-copy fall back to sync if input pointers are not pinned?
                e = hipErrorInvalidValue;
            }
        }
    } else {
        e = hipErrorInvalidValue;
    }
#else
    // TODO-hsart This routine needs to ensure that dst and src are mapped on the GPU.
    // This is a synchronous copy - remove and replace with code below when we have appropriate LOCK APIs.
    hc::am_copy(dst, src, sizeBytes);
#endif


    return ihipLogStatus(e);
}


// TODO-sync: function is async unless target is pinned host memory - then these are fully sync.
/** @return #hipErrorInvalidValue
 */
hipError_t hipMemsetAsync(void* dst, int  value, size_t sizeBytes, hipStream_t stream )
{
    std::call_once(hip_initialized, ihipInit);

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
            tprintf (TRACE_SYNC, "'%s' LAUNCH_BLOCKING wait for completion [stream:%p].\n", __func__, (void*)stream);
            cf.wait();
            tprintf (TRACE_SYNC, "'%s' LAUNCH_BLOCKING completed [stream:%p].\n", __func__, (void*)stream);
        } 
    } else {
        e = hipErrorInvalidValue;
    }


    return ihipLogStatus(e);
};


hipError_t hipMemset(void* dst, int  value, size_t sizeBytes )
{
    std::call_once(hip_initialized, ihipInit);

    // TODO - call an ihip memset so HIP_TRACE is correct.
    return hipMemsetAsync(dst, value, sizeBytes, hipStreamNull);
}


/*
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue (if free != NULL due to bug)S
 * @warning On HCC, the free memory only accounts for memory allocated by this process and may be optimistic.
 */
hipError_t hipMemGetInfo  (size_t *free, size_t *total)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    ihipDevice_t * hipDevice = ihipGetTlsDefaultDevice();
    if (hipDevice) {
        if (total) {
            *total = hipDevice->_props.totalGlobalMem;
        }

        if (free) {
#if USE_AM_TRACKER
            // TODO - replace with kernel-level for reporting free memory:
            size_t deviceMemSize, hostMemSize, userMemSize;
            hc::am_memtracker_sizeinfo(hipDevice->_acc, &deviceMemSize, &hostMemSize, &userMemSize);
            *free =  hipDevice->_props.totalGlobalMem - deviceMemSize;
#else
            *free =  hipDevice->_props.totalGlobalMem * 0.5; // TODO
            e=hipErrorInvalidValue;
#endif
        }

    } else {
        e = hipErrorInvalidDevice;
    }

    return ihipLogStatus(e);
}


//---
hipError_t hipFree(void* ptr)
{
    // TODO - ensure this pointer was created by hipMalloc and not hipMallocHost
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
    // TODO - ensure this pointer was created by hipMallocHost and not hipMalloc
    std::call_once(hip_initialized, ihipInit);

    if (ptr) {
        tprintf (TRACE_MEM, "  %s: %p\n", __func__, ptr);
        hc::am_free(ptr);
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

// TODO - review signal / error reporting code.
