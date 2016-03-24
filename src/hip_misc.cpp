
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
        stream = device->_default_stream;
    }

    *av = &(stream->_av);

    hipError_t err = hipSuccess;
    return ihipLogStatus(err);
}
