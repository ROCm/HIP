#include "hip_runtime.h"
#include "hcc_detail/hip_hcc.h"
#include "hcc_detail/trace_helper.h"

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
