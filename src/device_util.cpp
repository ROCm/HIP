#include"hip_runtime.h"
#include<hc.hpp>
#include<grid_launch.h>

const int warpSize = 64;

__device__ long long int clock64() { return (long long int)hc::__clock_u64(); };
__device__ clock_t clock() { return (clock_t)hc::__clock_u64(); };


//atomicAdd()
__device__  int atomicAdd(int* address, int val)
{
	return hc::atomic_fetch_add(address,val);
}
__device__  unsigned int atomicAdd(unsigned int* address,
                       unsigned int val)
{
   return hc::atomic_fetch_add(address,val);
}
__device__  unsigned long long int atomicAdd(unsigned long long int* address,
                                 unsigned long long int val)
{
 return (long long int)hc::atomic_fetch_add((uint64_t*)address,(uint64_t)val);
}
__device__  float atomicAdd(float* address, float val)
{
	return hc::atomic_fetch_add(address,val);
}

//atomicSub()
__device__  int atomicSub(int* address, int val)
{
	return hc::atomic_fetch_sub(address,val);
}
__device__  unsigned int atomicSub(unsigned int* address,
                       unsigned int val)
{
   return hc::atomic_fetch_sub(address,val);
}

//atomicExch()
__device__  int atomicExch(int* address, int val)
{
	return hc::atomic_exchange(address,val);
}
__device__  unsigned int atomicExch(unsigned int* address,
                        unsigned int val)
{
	return hc::atomic_exchange(address,val);
}
__device__  unsigned long long int atomicExch(unsigned long long int* address,
                                  unsigned long long int val)
{
	return (long long int)hc::atomic_exchange((uint64_t*)address,(uint64_t)val);
}
__device__  float atomicExch(float* address, float val)
{
	return hc::atomic_exchange(address,val);
}

//atomicMin()
__device__  int atomicMin(int* address, int val)
{
	return hc::atomic_fetch_min(address,val);
}
__device__  unsigned int atomicMin(unsigned int* address,
                       unsigned int val)
{
	return hc::atomic_fetch_min(address,val);
}
__device__  unsigned long long int atomicMin(unsigned long long int* address,
                                 unsigned long long int val)
{
	return (long long int)hc::atomic_fetch_min((uint64_t*)address,(uint64_t)val);
}

//atomicMax()
__device__  int atomicMax(int* address, int val)
{
	return hc::atomic_fetch_max(address,val);
}
__device__  unsigned int atomicMax(unsigned int* address,
                       unsigned int val)
{
	return hc::atomic_fetch_max(address,val);
}
__device__  unsigned long long int atomicMax(unsigned long long int* address,
                                 unsigned long long int val)
{
	return (long long int)hc::atomic_fetch_max((uint64_t*)address,(uint64_t)val);
}

//atomicCAS()
__device__  int atomicCAS(int* address, int compare, int val)
{
	hc::atomic_compare_exchange(address,&compare,val);
	return *address;
}
__device__  unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val)
{
	hc::atomic_compare_exchange(address,&compare,val);
	return *address;
}
__device__  unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val)
{
	hc::atomic_compare_exchange((uint64_t*)address,(uint64_t*)&compare,(uint64_t)val);
	return *address;
}

//atomicAnd()
__device__  int atomicAnd(int* address, int val)
{
	return hc::atomic_fetch_and(address,val);
}
__device__  unsigned int atomicAnd(unsigned int* address,
                       unsigned int val)
{
	return hc::atomic_fetch_and(address,val);
}
__device__  unsigned long long int atomicAnd(unsigned long long int* address,
                                 unsigned long long int val)
{
	return (long long int)hc::atomic_fetch_and((uint64_t*)address,(uint64_t)val);
}

//atomicOr()
__device__  int atomicOr(int* address, int val)
{
	return hc::atomic_fetch_or(address,val);
}
__device__  unsigned int atomicOr(unsigned int* address,
                      unsigned int val)
{
	return hc::atomic_fetch_or(address,val);
}
__device__  unsigned long long int atomicOr(unsigned long long int* address,
                                unsigned long long int val)
{
	return (long long int)hc::atomic_fetch_or((uint64_t*)address,(uint64_t)val);
}

//atomicXor()
__device__  int atomicXor(int* address, int val)
{
	return hc::atomic_fetch_xor(address,val);
}
__device__  unsigned int atomicXor(unsigned int* address,
                       unsigned int val)
{
	return hc::atomic_fetch_xor(address,val);
}
__device__  unsigned long long int atomicXor(unsigned long long int* address,
                                 unsigned long long int val)
{
	return (long long int)hc::atomic_fetch_xor((uint64_t*)address,(uint64_t)val);
}




__device__ unsigned int test__popc(unsigned int input)
{
    return hc::__popcount_u32_b32(input);
}

// integer intrinsic function __poc __clz __ffs __brev
__device__ unsigned int __popc( unsigned int input)
{
    return hc::__popcount_u32_b32(input);
}

__device__ unsigned int test__popc(unsigned int input);

__device__ unsigned int __popcll( unsigned long long int input)
{
	return hc::__popcount_u32_b64(input);
}

__device__ unsigned int __clz(unsigned int input)
{
	return hc::__firstbit_u32_u32( input);
}

__device__ unsigned int __clzll(unsigned long long int input)
{
	return hc::__firstbit_u32_u64( input);
}

__device__ unsigned int __clz(int input)
{
	return hc::__firstbit_u32_s32(  input);
}

__device__ unsigned int __clzll(long long int input)
{
	return hc::__firstbit_u32_s64( input);
}

__device__ unsigned int __ffs(unsigned int input)
{
	return hc::__lastbit_u32_u32( input)+1;
}

__device__ unsigned int __ffsll(unsigned long long int input)
{
	return hc::__lastbit_u32_u64( input)+1;
}

__device__ unsigned int __ffs(int input)
{
	return hc::__lastbit_u32_s32( input)+1;
}

__device__ unsigned int __ffsll(long long int input)
{
	return hc::__lastbit_u32_s64( input)+1;
}

__device__ unsigned int __brev( unsigned int input)
{
	return hc::__bitrev_b32( input);
}

__device__ unsigned long long int __brevll( unsigned long long int input)
{
	return hc::__bitrev_b64( input);
}

// warp vote function __all __any __ballot
__device__ int __all(  int input)
{
	return hc::__all( input);
}


__device__ int __any( int input)
{
	if( hc::__any( input)!=0) return 1;
	else return 0;
}

__device__ unsigned long long int __ballot( int input)
{
	return hc::__ballot( input);
}

// warp shuffle functions
__device__ int __shfl(int input, int lane, int width)
{
  return hc::__shfl(input,lane,width);
}

__device__  int __shfl_up(int input, unsigned int lane_delta, int width)
{
  return hc::__shfl_up(input,lane_delta,width);
}

__device__  int __shfl_down(int input, unsigned int lane_delta, int width)
{
  return hc::__shfl_down(input,lane_delta,width);
}

__device__  int __shfl_xor(int input, int lane_mask, int width)
{
  return hc::__shfl_xor(input,lane_mask,width);
}

__device__  float __shfl(float input, int lane, int width)
{
  return hc::__shfl(input,lane,width);
}

__device__  float __shfl_up(float input, unsigned int lane_delta, int width)
{
  return hc::__shfl_up(input,lane_delta,width);
}

__device__  float __shfl_down(float input, unsigned int lane_delta, int width)
{
  return hc::__shfl_down(input,lane_delta,width);
}

__device__  float __shfl_xor(float input, int lane_mask, int width)
{
  return hc::__shfl_xor(input,lane_mask,width);
}



//TODO - add a couple fast math operations here, the set here will grow :
__device__  float __cosf(float x) {return hc::fast_math::cosf(x); };
__device__  float __expf(float x) {return hc::fast_math::expf(x); };
__device__  float __frsqrt_rn(float x) {return hc::fast_math::rsqrt(x); };
__device__  float __fsqrt_rd(float x) {return hc::fast_math::sqrt(x); };
__device__  float __fsqrt_rn(float x) {return hc::fast_math::sqrt(x); };
__device__  float __fsqrt_ru(float x) {return hc::fast_math::sqrt(x); };
__device__  float __fsqrt_rz(float x) {return hc::fast_math::sqrt(x); };
__device__  float __log10f(float x) {return hc::fast_math::log10f(x); };
__device__  float __log2f(float x) {return hc::fast_math::log2f(x); };
__device__  float __logf(float x) {return hc::fast_math::logf(x); };
__device__  float __powf(float base, float exponent) {return hc::fast_math::powf(base, exponent); };
__device__  void __sincosf(float x, float *s, float *c) {return hc::fast_math::sincosf(x, s, c); };
__device__  float __sinf(float x) {return hc::fast_math::sinf(x); };
__device__  float __tanf(float x) {return hc::fast_math::tanf(x); };
__device__  float __dsqrt_rd(double x) {return hc::fast_math::sqrt(x); };
__device__  float __dsqrt_rn(double x) {return hc::fast_math::sqrt(x); };
__device__  float __dsqrt_ru(double x) {return hc::fast_math::sqrt(x); };
__device__  float __dsqrt_rz(double x) {return hc::fast_math::sqrt(x); };


