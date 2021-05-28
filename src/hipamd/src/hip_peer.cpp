/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>

#include "hip_internal.hpp"

hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, hipCtx_t thisCtx, hipCtx_t peerCtx) {
  HIP_INIT_API(NONE, canAccessPeer, thisCtx, peerCtx);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipMemcpyPeer(void* dst, hipCtx_t dstCtx, const void* src, hipCtx_t srcCtx,
                         size_t sizeBytes) {
  HIP_INIT_API(NONE, dst, dstCtx, src, srcCtx, sizeBytes);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipMemcpyPeerAsync(void* dst, hipCtx_t dstDevice, const void* src, hipCtx_t srcDevice,
                              size_t sizeBytes, hipStream_t stream) {
  HIP_INIT_API(NONE, dst, dstDevice, src, srcDevice, sizeBytes, stream);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t canAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId){
  amd::Device* device = nullptr;
  amd::Device* peer_device = nullptr;
  if (canAccessPeer == nullptr) {
    return hipErrorInvalidValue;
  }
  /* Peer cannot be self */
  if (deviceId == peerDeviceId) {
    *canAccessPeer = 0;
    return hipSuccess;
  }
  /* Cannot exceed the max number of devices */
  if (static_cast<size_t>(deviceId) >= g_devices.size()
       || static_cast<size_t>(peerDeviceId) >= g_devices.size()) {
    return hipErrorInvalidDevice;
  }
  device = g_devices[deviceId]->devices()[0];
  peer_device = g_devices[peerDeviceId]->devices()[0];
  *canAccessPeer = static_cast<int>(std::find(device->p2pDevices_.begin(),
                                              device->p2pDevices_.end(), as_cl(peer_device))
                                              != device->p2pDevices_.end());
  return hipSuccess;
}

hipError_t findLinkInfo(int device1, int device2,
                        std::vector<amd::Device::LinkAttrType>* link_attrs) {

  amd::Device* amd_dev_obj1 = nullptr;
  amd::Device* amd_dev_obj2 = nullptr;
  const int numDevices = static_cast<int>(g_devices.size());

  if ((device1 < 0) || (device1 >= numDevices) || (device2 < 0) || (device2 >= numDevices)) {
    return hipErrorInvalidDevice;
  }

  amd_dev_obj1 = g_devices[device1]->devices()[0];
  amd_dev_obj2 = g_devices[device2]->devices()[0];

  if (!amd_dev_obj1->findLinkInfo(*amd_dev_obj2, link_attrs)) {
    return hipErrorInvalidHandle;
  }

  return hipSuccess;
}

hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2,
                                        uint32_t* linktype, uint32_t* hopcount) {
  HIP_INIT_API(hipExtGetLinkTypeAndHopCount, device1, device2, linktype, hopcount);

  if (linktype == nullptr || hopcount == nullptr ||
      device1 == device2  || device1 < 0 || device2 < 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Fill out the list of LinkAttributes
  std::vector<amd::Device::LinkAttrType> link_attrs;
  link_attrs.push_back(std::make_pair(amd::Device::LinkAttribute::kLinkLinkType, 0));
  link_attrs.push_back(std::make_pair(amd::Device::LinkAttribute::kLinkHopCount, 0));

  HIP_RETURN_ONFAIL(findLinkInfo(device1, device2, &link_attrs));

  *linktype = static_cast<uint32_t>(link_attrs[0].second);
  *hopcount = static_cast<uint32_t>(link_attrs[1].second);

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr,
                                    int srcDevice, int dstDevice) {
  HIP_INIT_API(hipDeviceGetP2PAttribute, value, attr, srcDevice, dstDevice);

  if (value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (srcDevice == dstDevice || srcDevice >= static_cast<int>(g_devices.size())
      || dstDevice >= static_cast<int>(g_devices.size())) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  std::vector<amd::Device::LinkAttrType> link_attrs;

  switch (attr) {
    case hipDevP2PAttrPerformanceRank : {
      link_attrs.push_back(std::make_pair(amd::Device::LinkAttribute::kLinkLinkType, 0));
      break;
    }
    case hipDevP2PAttrAccessSupported : {
      HIP_RETURN_ONFAIL(canAccessPeer(value, srcDevice, dstDevice));
      break;
    }
    case hipDevP2PAttrNativeAtomicSupported : {
      link_attrs.push_back(std::make_pair(amd::Device::LinkAttribute::kLinkLinkType, 0));
      break;
    }
    case hipDevP2PAttrHipArrayAccessSupported : {
      hipDeviceProp_t srcDeviceProp;
      hipDeviceProp_t dstDeviceProp;
      HIP_RETURN_ONFAIL(hipGetDeviceProperties(&srcDeviceProp, srcDevice));
      HIP_RETURN_ONFAIL(hipGetDeviceProperties(&dstDeviceProp, dstDevice));

      // Linear layout access is supported if P2P is enabled
      // Opaque Images are supported only on homogeneous systems
      // Might have more conditions to check, in future.
      if (srcDeviceProp.gcnArch == dstDeviceProp.gcnArch) {
        HIP_RETURN_ONFAIL(canAccessPeer(value, srcDevice, dstDevice));
      } else {
        value = 0;
      }
      break;
    }
    default : {
      LogPrintfError("Invalid attribute attr: %d ", attr);
      HIP_RETURN(hipErrorInvalidValue);
      break;
    }
  }

  if ((attr != hipDevP2PAttrAccessSupported) && (attr != hipDevP2PAttrHipArrayAccessSupported)) {
    HIP_RETURN_ONFAIL(findLinkInfo(srcDevice, dstDevice, &link_attrs));
    *value = static_cast<int>(link_attrs[0].second);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceCanAccessPeer(int* canAccess, int deviceId, int peerDeviceId) {
  HIP_INIT_API(hipDeviceCanAccessPeer, canAccess, deviceId, peerDeviceId);
  HIP_RETURN(canAccessPeer(canAccess, deviceId, peerDeviceId));
}

hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
  HIP_INIT_API(hipDeviceDisablePeerAccess, peerDeviceId);
  int deviceId = hip::getCurrentDevice()->deviceId();
  int canAccess = 0;
  if ((hipSuccess != canAccessPeer(&canAccess, deviceId, peerDeviceId)) || (canAccess == 0)) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  amd::Device* device = g_devices[deviceId]->devices()[0];
  amd::Device* peer_device = g_devices[peerDeviceId]->devices()[0];
  peer_device->disableP2P(device);

  HIP_RETURN(hip::getCurrentDevice()->DisablePeerAccess(peerDeviceId));
}

hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  HIP_INIT_API(hipDeviceEnablePeerAccess, peerDeviceId, flags);
  int deviceId = hip::getCurrentDevice()->deviceId();
  int canAccess = 0;
  if (flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if ((hipSuccess != canAccessPeer(&canAccess, deviceId, peerDeviceId)) || (canAccess == 0)) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  amd::Device* device = g_devices[deviceId]->asContext()->devices()[0];
  amd::Device* peer_device = g_devices[peerDeviceId]->asContext()->devices()[0];
  peer_device->enableP2P(device);

  HIP_RETURN(hip::getCurrentDevice()->EnablePeerAccess(peerDeviceId));
}

hipError_t hipMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice,
                         size_t sizeBytes) {
  HIP_INIT_API(hipMemcpyPeer, dst, dstDevice, src, srcDevice, sizeBytes);

  if (srcDevice >= static_cast<int>(g_devices.size()) ||
      dstDevice >= static_cast<int>(g_devices.size()) ||
      srcDevice < 0 || dstDevice < 0) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  HIP_RETURN(hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToDevice));
}

hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice,
                              size_t sizeBytes, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyPeerAsync, dst, dstDevice, src, srcDevice, sizeBytes, stream);

  if (srcDevice >= static_cast<int>(g_devices.size()) ||
      dstDevice >= static_cast<int>(g_devices.size()) ||
      srcDevice < 0 || dstDevice < 0) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  HIP_RETURN(hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream));
}

hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
  HIP_INIT_API(hipCtxEnablePeerAccess, peerCtx, flags);

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
  HIP_INIT_API(hipCtxDisablePeerAccess, peerCtx);

  HIP_RETURN(hipSuccess);
}
