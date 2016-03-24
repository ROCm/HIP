//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Error Handling
//---
/**
 * @returns return code from last HIP called from the active host thread.
 */
hipError_t hipGetLastError()
{
    HIP_INIT_API();

    // Return last error, but then reset the state:
    hipError_t e = ihipLogStatus(tls_lastHipError);
    tls_lastHipError = hipSuccess;
    return e;
}


//---
hipError_t hipPeakAtLastError()
{
    HIP_INIT_API();

   
    // peak at last error, but don't reset it. 
    return ihipLogStatus(tls_lastHipError);
}


const char *ihipErrorString(hipError_t hip_error)
{
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
        case hipErrorRuntimeMemory          : return "hipErrorRuntimeMemory";
        case hipErrorRuntimeOther           : return "hipErrorRuntimeOther";
        case hipErrorUnknown                : return "hipErrorUnknown";
        case hipErrorTbd                    : return "hipErrorTbd";
        default                             : return "hipErrorUnknown";
    };
};



//---
const char *hipGetErrorName(hipError_t hip_error)
{
    HIP_INIT_API(hip_error);

    return ihipErrorString(hip_error);
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
