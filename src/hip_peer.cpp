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

#include <hc_am.hpp>

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"


// Peer access functions.
// There are two flavors:
//   - one where contexts are specified with hipCtx_t type.
//   - one where contexts are specified with integer deviceIds, that are mapped to the primary
//   context for that device.
// The implementation contains a set of internal ihip* functions which operate on contexts.  Then
// the public APIs are thin wrappers which call into this internal implementations.
// TODO - actually not yet - currently the integer deviceId flavors just call the context APIs. need
// to fix.


hipError_t ihipDeviceCanAccessPeer(int* canAccessPeer, hipCtx_t thisCtx, hipCtx_t peerCtx) {
    hipError_t err = hipSuccess;


    if ((thisCtx != NULL) && (peerCtx != NULL)) {
        if (thisCtx == peerCtx) {
            *canAccessPeer = 0;
            tprintf(DB_MEM, "Can't be peer to self. (this=%s, peer=%s)\n",
                    thisCtx->toString().c_str(), peerCtx->toString().c_str());
        } else if (HIP_FORCE_P2P_HOST & 0x2) {
            *canAccessPeer = false;
            tprintf(DB_MEM,
                    "HIP_FORCE_P2P_HOST denies peer access this=%s peer=%s  canAccessPeer=%d\n",
                    thisCtx->toString().c_str(), peerCtx->toString().c_str(), *canAccessPeer);
        } else {
            *canAccessPeer = peerCtx->getDevice()->_acc.get_is_peer(thisCtx->getDevice()->_acc);
            tprintf(DB_MEM, "deviceCanAccessPeer this=%s peer=%s  canAccessPeer=%d\n",
                    thisCtx->toString().c_str(), peerCtx->toString().c_str(), *canAccessPeer);
        }

    } else {
        *canAccessPeer = 0;
        err = hipErrorInvalidDevice;
    }


    return err;
}


/**
 * HCC returns 0 in *canAccessPeer ; Need to update this function when RT supports P2P
 */
//---
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, hipCtx_t thisCtx, hipCtx_t peerCtx) {
    HIP_INIT_API(hipDeviceCanAccessPeer2, canAccessPeer, thisCtx, peerCtx);

    return ihipLogStatus(ihipDeviceCanAccessPeer(canAccessPeer, thisCtx, peerCtx));
}


//---
// Disable visibility of this device into memory allocated on peer device.
// Remove this device from peer device peerlist.
hipError_t ihipDisablePeerAccess(hipCtx_t peerCtx) {
    hipError_t err = hipSuccess;

    auto thisCtx = ihipGetTlsDefaultCtx();
    if ((thisCtx != NULL) && (peerCtx != NULL)) {
        bool canAccessPeer = peerCtx->getDevice()->_acc.get_is_peer(thisCtx->getDevice()->_acc);

        if (!canAccessPeer) {
            err = hipErrorInvalidDevice;  // P2P not allowed between these devices.
        } else if (thisCtx == peerCtx) {
            err = hipErrorInvalidDevice;  // Can't disable peer access to self.
        } else {
            LockedAccessor_CtxCrit_t peerCrit(peerCtx->criticalData());
            bool changed = peerCrit->removePeerWatcher(peerCtx, thisCtx);
            if (changed) {
                tprintf(DB_MEM, "device %s disable access to memory allocated on peer:%s\n",
                        thisCtx->toString().c_str(), peerCtx->toString().c_str());
                // Update the peers for all memory already saved in the tracker:
                am_memtracker_update_peers(peerCtx->getDevice()->_acc, peerCrit->peerCnt(),
                                           peerCrit->peerAgents());
            } else {
                err = hipErrorPeerAccessNotEnabled;  // never enabled P2P access.
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
hipError_t ihipEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
    hipError_t err = hipSuccess;
    if (flags != 0) {
        err = hipErrorInvalidValue;
    } else {
        auto thisCtx = ihipGetTlsDefaultCtx();
        if (thisCtx == peerCtx) {
            err = hipErrorInvalidDevice;  // Can't enable peer access to self.
        } else if ((thisCtx != NULL) && (peerCtx != NULL)) {
            LockedAccessor_CtxCrit_t peerCrit(peerCtx->criticalData());
            // Add thisCtx to peerCtx's access list so that new allocations on peer will be made
            // visible to this device:
            bool isNewPeer = peerCrit->addPeerWatcher(peerCtx, thisCtx);
            if (isNewPeer) {
                tprintf(DB_MEM, "device=%s can now see all memory allocated on peer=%s\n",
                        thisCtx->toString().c_str(), peerCtx->toString().c_str());
                am_memtracker_update_peers(peerCtx->getDevice()->_acc, peerCrit->peerCnt(),
                                           peerCrit->peerAgents());
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
hipError_t hipMemcpyPeer(void* dst, hipCtx_t dstCtx, const void* src, hipCtx_t srcCtx,
                         size_t sizeBytes) {
    HIP_INIT_API(hipMemcpyPeer2, dst, dstCtx, src, srcCtx, sizeBytes);

    // TODO - move to ihip memory copy implementaion.
    // HCC has a unified memory architecture so device specifiers are not required.
    return ihipLogStatus(hipMemcpy(dst, src, sizeBytes, hipMemcpyDefault));
};


//---
hipError_t hipMemcpyPeerAsync(void* dst, hipCtx_t dstDevice, const void* src, hipCtx_t srcDevice,
                              size_t sizeBytes, hipStream_t stream) {
    HIP_INIT_API(hipMemcpyPeerAsync2, dst, dstDevice, src, srcDevice, sizeBytes, stream);

    // TODO - move to ihip memory copy implementaion.
    // HCC has a unified memory architecture so device specifiers are not required.
    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
};


//=============================================================================
// These are the flavors that accept integer deviceIDs.
// Implementations map these to primary contexts and call the internal functions above.
//=============================================================================

hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId) {
    HIP_INIT_API(hipDeviceCanAccessPeer, canAccessPeer, deviceId, peerDeviceId);
    return ihipLogStatus(ihipDeviceCanAccessPeer(canAccessPeer, ihipGetPrimaryCtx(deviceId),
                                                 ihipGetPrimaryCtx(peerDeviceId)));
}


hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
    HIP_INIT_API(hipDeviceDisablePeerAccess, peerDeviceId);

    return ihipLogStatus(ihipDisablePeerAccess(ihipGetPrimaryCtx(peerDeviceId)));
}


hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
    HIP_INIT_API(hipDeviceEnablePeerAccess, peerDeviceId, flags);

    return ihipLogStatus(ihipEnablePeerAccess(ihipGetPrimaryCtx(peerDeviceId), flags));
}


hipError_t hipMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice,
                         size_t sizeBytes) {
    HIP_INIT_API(hipMemcpyPeer, dst, dstDevice, src, srcDevice, sizeBytes);
    return ihipLogStatus(hipMemcpyPeer(dst, ihipGetPrimaryCtx(dstDevice), src,
                                       ihipGetPrimaryCtx(srcDevice), sizeBytes));
}


hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice,
                              size_t sizeBytes, hipStream_t stream) {
    HIP_INIT_API(hipMemcpyPeerAsync, dst, dstDevice, src, srcDevice, sizeBytes, stream);
    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
}

hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
    HIP_INIT_API(hipCtxEnablePeerAccess, peerCtx, flags);

    return ihipLogStatus(ihipEnablePeerAccess(peerCtx, flags));
}

hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
    HIP_INIT_API(hipCtxDisablePeerAccess, peerCtx);

    return ihipLogStatus(ihipDisablePeerAccess(peerCtx));
}
