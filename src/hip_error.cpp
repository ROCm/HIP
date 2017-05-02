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

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Error Handling
//---

hipError_t hipGetLastError()
{
    HIP_INIT_API();

    // Return last error, but then reset the state:
    hipError_t e = ihipLogStatus(tls_lastHipError);
    tls_lastHipError = hipSuccess;
    return e;
}

hipError_t hipPeekAtLastError()
{
    HIP_INIT_API();

    // peek at last error, but don't reset it.
    return ihipLogStatus(tls_lastHipError);
}

const char *hipGetErrorName(hipError_t hip_error)
{
    HIP_INIT_API(hip_error);

    return ihipErrorString(hip_error);
}

const char *hipGetErrorString(hipError_t hip_error)
{
    HIP_INIT_API(hip_error);

    // TODO - return a message explaining the error.
    // TODO - This should be set up to return the same string reported in the the doxygen comments, somehow.
    return hipGetErrorName(hip_error);
}
