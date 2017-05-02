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

#include "hip/hcc_detail/hip_ldg.h"
#include "hip/hcc_detail/hip_vector_types.h"

__device__ char                 __ldg(const char* ptr)
{
    return *ptr;
}

__device__ char2                __ldg(const char2* ptr)
{
    return *ptr;
}

__device__ char4                __ldg(const char4* ptr)
{
    return *ptr;
}

__device__ signed char          __ldg(const signed char* ptr)
{
    return ptr[0];
}

__device__ unsigned char        __ldg(const unsigned char* ptr)
{
    return ptr[0];
}

__device__ short                __ldg(const short* ptr)
{
    return ptr[0];
}

__device__ short2               __ldg(const short2* ptr)
{
    return ptr[0];
}

__device__ short4               __ldg(const short4* ptr)
{
    return ptr[0];
}

__device__ unsigned short       __ldg(const unsigned short* ptr)
{
    return ptr[0];
}

__device__ int                  __ldg(const int* ptr)
{
    return ptr[0];
}

__device__ int2                 __ldg(const int2* ptr)
{
    return ptr[0];
}

__device__ int4                 __ldg(const int4* ptr)
{
    return ptr[0];
}

__device__ unsigned int         __ldg(const unsigned int* ptr)
{
    return ptr[0];
}


__device__ long                 __ldg(const long* ptr)
{
    return ptr[0];
}

__device__ unsigned long        __ldg(const unsigned long* ptr)
{
    return ptr[0];
}

__device__ long long            __ldg(const long long* ptr)
{
    return ptr[0];
}

__device__ longlong2            __ldg(const longlong2* ptr)
{
    return ptr[0];
}

__device__ unsigned long long   __ldg(const unsigned long long* ptr)
{
    return ptr[0];
}

__device__ uchar2               __ldg(const uchar2* ptr)
{
    return ptr[0];
}

__device__ uchar4               __ldg(const uchar4* ptr)
{
    return ptr[0];
}

__device__ ushort2              __ldg(const ushort2* ptr)
{
    return ptr[0];
}

__device__ uint2                __ldg(const uint2* ptr)
{
    return ptr[0];
}

__device__ uint4                __ldg(const uint4* ptr)
{
    return ptr[0];
}

__device__ ulonglong2           __ldg(const ulonglong2* ptr)
{
    return ptr[0];
}

__device__ float                __ldg(const float* ptr)
{
    return ptr[0];
}

__device__ float2               __ldg(const float2* ptr)
{
    return ptr[0];
}

__device__ float4               __ldg(const float4* ptr)
{
    return ptr[0];
}

__device__ double               __ldg(const double* ptr)
{
    return ptr[0];
}

__device__ double2              __ldg(const double2* ptr)
{
    return ptr[0];
}
