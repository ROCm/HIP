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

#include "hip_runtime.h"
#include "hcc_detail/hip_hcc.h"
#include "hcc_detail/trace_helper.h"
#define USE_PEER_TO_PEER 1

/**
 * @warning HCC returns 0 in *canAccessPeer ; Need to update this function when RT supports P2P
 */
//---
hipError_t hipDeviceCanAccessPeer ( int* canAccessPeer, int  deviceId, int peerDeviceId)
{
    HIP_INIT_API(canAccessPeer, deviceId, peerDeviceId);

    hipError_t err = hipSuccess;

#if USE_PEER_TO_PEER
    auto device = ihipGetDevice(deviceId);
    auto peerDevice = ihipGetDevice(peerDeviceId);

    if ((device != NULL) && (peerDevice != NULL)) {
#if USE_PEER_TO_PEER==2
        *canAccessPeer = peerDevice->_acc.is_peer(device->_acc);
#else
        *canAccessPeer = 0;
#endif

    } else {
        *canAccessPeer = false;
        err = hipErrorInvalidDevice;
    }


#else
    *canAccessPeer = false;
#endif
    return ihipLogStatus(err);
}


/**
 * warning Need to update this function when RT supports P2P
 */
//---
hipError_t  hipDeviceDisablePeerAccess ( int  peerDevice )
{
    HIP_INIT_API(peerDevice);

    // TODO-p2p
    return ihipLogStatus(hipSuccess);
};


/**
 * @warning Need to update this function when RT supports P2P
 */
//---
 // Enable registering memory on peerDevice for direct access from the current device.
hipError_t  hipDeviceEnablePeerAccess (int peerDeviceId, unsigned int flags)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t err = hipSuccess;
#if USE_PEER_TO_PEER
    if (flags != 0) {
        err = hipErrorInvalidValue;
    } else {
        auto peerDevice = ihipGetDevice(peerDeviceId);
        if (peerDevice != NULL) {

        } else {
            err = hipErrorInvalidDevice;
        }
    }
#endif

    return ihipLogStatus(err);
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
    HIP_INIT_API(driverVersion);

    if (driverVersion) {
        *driverVersion = 4;
    }

    return ihipLogStatus(hipSuccess);
}


