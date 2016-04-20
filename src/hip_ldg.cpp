#include"hcc_detail/hip_ldg.h"

__device__ char __ldg(const char* ptr)
{
    return *ptr;
}

__device__ signed char __ldg(const signed char* ptr)
{
    return ptr[0];
}

__device__ short __ldg(const short* ptr)
{
    return ptr[0];
}

__device__ int __ldg(const int* ptr)
{
    return ptr[0];
}

__device__ long long __ldg(const long long* ptr)
{
    return ptr[0];
}


__device__ int2 __ldg(const int2* ptr)
{
    return ptr[0];
}

__device__ int4 __ldg(const int4* ptr)
{
    return ptr[0];
}

__device__ float __ldg(const float* ptr)
{
    return ptr[0];
}
