#ifndef HIP_INLINE_H
#define HIP_INLINE_H

#include "trace_helper.h"

#define INLINE static inline
extern ihipDevice_t *g_devices;
extern thread_local int tls_defaultDevice;
extern const hipStream_t hipStreamNull;

INLINE bool ihipIsValidDevice(unsigned deviceIndex)
{
    // deviceIndex is unsigned so always > 0
    return (deviceIndex < g_deviceCnt);
}

/*// check if the device ID is set as visible*/
//INLINE bool ihipIsVisibleDevice(unsigned deviceIndex)
//{
    //return std::find(g_hip_visible_devices.begin(), g_hip_visible_devices.end(),
            //(int)deviceIndex) != g_hip_visible_devices.end();
/*}*/

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

inline hipStream_t ihipSyncAndResolveStream(hipStream_t stream)
{
    if (stream == hipStreamNull ) {
        ihipDevice_t *device = ihipGetTlsDefaultDevice();

#ifndef HIP_API_PER_THREAD_DEFAULT_STREAM
        device->syncDefaultStream(false);
#endif
        return device->_default_stream;
    } else {
        // Have to wait for legacy default stream to be empty:
        if (!(stream->_flags & hipStreamNonBlocking))  {
            tprintf(DB_SYNC, "stream %p wait default stream\n", stream);
            stream->getDevice()->_default_stream->wait();
        }
        
        return stream;
    }
}

#endif
