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

#include "hip/hip_runtime.h"
#include "hip/hcc_detail/hip_hcc.h"
#include "hip/hcc_detail/trace_helper.h"


// Peer access functions.
// There are two flavors:
//   - one where contexts are specified with hipCtx_t type.
//   - one where contexts are specified with integer deviceIds, that are mapped to the primary context for that device.
// The implementation contains a set of internal ihip* functions which operate on contexts.  Then the 
// public APIs are thin wrappers which call into this internal implementations.
// TODO - actually not yet - currently the integer deviceId flavors just call the context APIs.  need to fix.

/**
 * HCC returns 0 in *canAccessPeer ; Need to update this function when RT supports P2P
 */
//---
hipError_t hipDeviceCanAccessPeer (int* canAccessPeer, hipCtx_t thisCtx, hipCtx_t peerCtx)
{
    HIP_INIT_API(canAccessPeer, thisCtx, peerCtx);

    hipError_t err = hipSuccess;


    if ((thisCtx != NULL) && (peerCtx != NULL)) {
        if (thisCtx == peerCtx) {
            *canAccessPeer = 0;
        } else {
             *canAccessPeer = peerCtx->getDevice()->_acc.get_is_peer(thisCtx->getDevice()->_acc);
        }

    } else {
        *canAccessPeer = 0;
        err = hipErrorInvalidDevice;
    }


    return ihipLogStatus(err);
}


//---
// Disable visibility of this device into memory allocated on peer device.
// Remove this device from peer device peerlist.
hipError_t ihipDisablePeerAccess (hipCtx_t peerCtx)
{
    hipError_t err = hipSuccess;

    auto thisCtx = ihipGetTlsDefaultCtx();
    if ((thisCtx != NULL) && (peerCtx != NULL)) {
        bool canAccessPeer =  peerCtx->getDevice()->_acc.get_is_peer(thisCtx->getDevice()->_acc);

        if (! canAccessPeer) {
            err = hipErrorInvalidDevice;  // P2P not allowed between these devices.
        } else if (thisCtx == peerCtx)  {
            err = hipErrorInvalidDevice;  // Can't disable peer access to self.
        } else {
            LockedAccessor_CtxCrit_t peerCrit(peerCtx->criticalData());
            bool changed = peerCrit->removePeer(thisCtx);
            if (changed) {
                // Update the peers for all memory already saved in the tracker:
                am_memtracker_update_peers(peerCtx->getDevice()->_acc, peerCrit->peerCnt(), peerCrit->peerAgents());
            } else {
                err = hipErrorPeerAccessNotEnabled; // never enabled P2P access.
            }
        } 
    } else {
        err = hipErrorInvalidDevice;
    }

    return err;
};


//---
// Allow the current device to see all memory allocated on peerCtx.
// This should add this device to the peer-device peer list.
hipError_t ihipEnablePeerAccess (hipCtx_t peerCtx, unsigned int flags)
{

    hipError_t err = hipSuccess;
    if (flags != 0) {
        err = hipErrorInvalidValue;
    } else {
        auto thisCtx = ihipGetTlsDefaultCtx();
        if (thisCtx == peerCtx)  {
            err = hipErrorInvalidDevice;  // Can't enable peer access to self.
        } else if ((thisCtx != NULL) && (peerCtx != NULL)) {
            LockedAccessor_CtxCrit_t peerCrit(peerCtx->criticalData());
            // Add thisCtx to peerCtx's access list so that new allocations on peer will be made visible to this device:
            bool isNewPeer = peerCrit->addPeer(thisCtx);
            if (isNewPeer) {
                am_memtracker_update_peers(peerCtx->getDevice()->_acc, peerCrit->peerCnt(), peerCrit->peerAgents());
            } else {
                err = hipErrorPeerAccessAlreadyEnabled;
            }
        } else {
            err = hipErrorInvalidDevice;
        }
    }

    return err;
}


//---
hipError_t hipMemcpyPeer (void* dst, hipCtx_t dstCtx, const void* src, hipCtx_t srcCtx, size_t sizeBytes)
{
    HIP_INIT_API(dst, dstCtx, src, srcCtx, sizeBytes);

    // TODO - move to ihip memory copy implementaion.
    // HCC has a unified memory architecture so device specifiers are not required.
    return hipMemcpy(dst, src, sizeBytes, hipMemcpyDefault);
};


//---
hipError_t hipMemcpyPeerAsync (void* dst, hipCtx_t dstDevice, const void* src, hipCtx_t srcDevice, size_t sizeBytes, hipStream_t stream)
{
    HIP_INIT_API(dst, dstDevice, src, srcDevice, sizeBytes, stream);

    // TODO - move to ihip memory copy implementaion.
    // HCC has a unified memory architecture so device specifiers are not required.
    return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream);
};



//=============================================================================
// These are the flavors that accept integer deviceIDs.
// Implementations map these to primary contexts and call the internal functions above.
//=============================================================================

hipError_t hipDeviceCanAccessPeer (int* canAccessPeer, int deviceId, int peerDeviceId)
{
    HIP_INIT_API(canAccessPeer, deviceId, peerDeviceId);
    return hipDeviceCanAccessPeer(canAccessPeer, ihipGetPrimaryCtx(deviceId), ihipGetPrimaryCtx(peerDeviceId));
}


hipError_t hipDeviceDisablePeerAccess (int peerDeviceId)
{
    HIP_INIT_API(peerDeviceId);

    return ihipLogStatus(ihipDisablePeerAccess(ihipGetPrimaryCtx(peerDeviceId)));
}


hipError_t hipDeviceEnablePeerAccess (int peerDeviceId, unsigned int flags)
{
    HIP_INIT_API(peerDeviceId, flags);

    return ihipLogStatus(ihipEnablePeerAccess(ihipGetPrimaryCtx(peerDeviceId), flags));
}


hipError_t hipMemcpyPeer (void* dst, int  dstDevice, const void* src, int  srcDevice, size_t sizeBytes)
{
    HIP_INIT_API(dst, dstDevice, src, srcDevice, sizeBytes);
    return hipMemcpyPeer(dst, ihipGetPrimaryCtx(dstDevice), src, ihipGetPrimaryCtx(srcDevice), sizeBytes);
}


hipError_t hipMemcpyPeerAsync (void* dst, int  dstDevice, const void* src, int  srcDevice, size_t sizeBytes, hipStream_t stream)
{
    HIP_INIT_API(dst, dstDevice, src, srcDevice, sizeBytes, stream);
    return hipMemcpyPeerAsync(dst, ihipGetPrimaryCtx(dstDevice), src, ihipGetPrimaryCtx(srcDevice), sizeBytes, stream);
}

hipError_t hipCtxEnablePeerAccess (hipCtx_t peerCtx, unsigned int flags)
{
    HIP_INIT_API(peerCtx, flags);

    return ihipLogStatus(ihipEnablePeerAccess(peerCtx, flags));
}

hipError_t hipCtxDisablePeerAccess (hipCtx_t peerCtx)
{
    HIP_INIT_API(peerCtx);

    return ihipLogStatus(ihipDisablePeerAccess(peerCtx));
}
