
/**
 * @warning HCC returns 0 in *canAccessPeer ; Need to update this function when RT supports P2P
 */
//---
hipError_t hipDeviceCanAccessPeer ( int* canAccessPeer, int  device, int  peerDevice )
{
    HIP_INIT_API(canAccessPeer, device, peerDevice);

    *canAccessPeer = false;
    return ihipLogStatus(hipSuccess);
}


/**
 * @warning Need to update this function when RT supports P2P
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
