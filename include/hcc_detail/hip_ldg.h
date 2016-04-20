#ifndef HIP_LDG_H
#define HIP_LDG_H

#if __HCC__
#include"hip_vector_types.h"
#include"host_defines.h"
__device__ char            __ldg(const char* );
__device__ signed char     __ldg(const signed char* );
__device__ short           __ldg(const short* );
__device__ int             __ldg(const int* );
__device__ long            __ldg(const long* );
__device__ long long       __ldg(const long long* );
__device__ char2           __ldg(const char2* );
__device__ char4           __ldg(const char4* );
__device__ short2          __ldg(const short2* );
__device__ short4          __ldg(const short4* );
__device__ int2            __ldg(const int2* );
__device__ int4            __ldg(const int4* );
__device__ longlong2       __ldg(const longlong2* );
__device__ unsigned char   __ldg(const unsigned char* );
__device__ unsigned short  __ldg(const unsigned short* );
__device__ unsigned int    __ldg(const unsigned int* );
__device__ unsigned long   __ldg(const unsigned long* );
__device__ unsigned long long __ldg(const unsigned long long* );
__device__ uchar2          __ldg(const uchar2* );
__device__ uchar4          __ldg(const uchar4* );
__device__ ushort2         __ldg(const ushort2* );
__device__ uint2           __ldg(const uint2* );
__device__ uint4           __ldg(const uint4* );
__device__ ulonglong2      __ldg(const ulonglong2* );
__device__ float           __ldg(const float* );
__device__ double          __ldg(const double* );
__device__ float2          __ldg(const float2* );
__device__ float4          __ldg(const float4* );
__device__ double2         __ldg(const double2* );

#endif

#endif
