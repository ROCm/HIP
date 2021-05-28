

#include "amd_device_functions.h"

#if __has_builtin(__hip_atomic_compare_exchange_strong)

#if !__HIP_DEVICE_COMPILE__
//TODO: Remove this after compiler pre-defines the following Macros.
#define __HIP_MEMORY_SCOPE_SINGLETHREAD 1
#define __HIP_MEMORY_SCOPE_WAVEFRONT 2
#define __HIP_MEMORY_SCOPE_WORKGROUP 3
#define __HIP_MEMORY_SCOPE_AGENT 4
#define __HIP_MEMORY_SCOPE_SYSTEM 5
#endif

__device__
inline
int atomicCAS(int* address, int compare, int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
    return compare;
}

__device__
inline
int atomicCAS_system(int* address, int compare, int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
    return compare;
}

__device__
inline
unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__
inline
unsigned int atomicCAS_system(unsigned int* address, unsigned int compare, unsigned int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  return compare;
}

__device__
inline
unsigned long long atomicCAS(unsigned long long* address, unsigned long long compare,
                             unsigned long long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__
inline
unsigned long long atomicCAS_system(unsigned long long* address, unsigned long long compare,
                                    unsigned long long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  return compare;
}

__device__
inline
int atomicAdd(int* address, int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicAdd_system(int* address, int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicAdd(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicAdd_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicAdd(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicAdd_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
float atomicAdd(float* address, float val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
float atomicAdd_system(float* address, float val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

#if !defined(__HIPCC_RTC__)
DEPRECATED("use atomicAdd instead")
#endif // !defined(__HIPCC_RTC__)
__device__
inline
void atomicAddNoRet(float* address, float val)
{
    __ockl_atomic_add_noret_f32(address, val);
}

__device__
inline
double atomicAdd(double* address, double val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
double atomicAdd_system(double* address, double val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicSub(int* address, int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicSub_system(int* address, int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicSub(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicSub_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicExch(int* address, int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicExch_system(int* address, int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicExch(unsigned int* address, unsigned int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicExch_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicExch(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
float atomicExch(float* address, float val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
float atomicExch_system(float* address, float val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicMin(int* address, int val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicMin_system(int* address, int val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicMin(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicMin_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicMin(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicMin_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicMax(int* address, int val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicMax_system(int* address, int val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicMax(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicMax_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicMax(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicMax_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicInc(unsigned int* address, unsigned int val)
{
    __device__
    extern
    unsigned int __builtin_amdgcn_atomic_inc(
        unsigned int*,
        unsigned int,
        unsigned int,
        unsigned int,
        bool) __asm("llvm.amdgcn.atomic.inc.i32.p0i32");

    return __builtin_amdgcn_atomic_inc(
        address, val, __ATOMIC_RELAXED, 1 /* Device scope */, false);
}

__device__
inline
unsigned int atomicDec(unsigned int* address, unsigned int val)
{
    __device__
    extern
    unsigned int __builtin_amdgcn_atomic_dec(
        unsigned int*,
        unsigned int,
        unsigned int,
        unsigned int,
        bool) __asm("llvm.amdgcn.atomic.dec.i32.p0i32");

    return __builtin_amdgcn_atomic_dec(
        address, val, __ATOMIC_RELAXED, 1 /* Device scope */, false);
}

__device__
inline
int atomicAnd(int* address, int val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicAnd_system(int* address, int val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicAnd(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicAnd_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__
inline
unsigned long long atomicAnd(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicAnd_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicOr(int* address, int val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicOr_system(int* address, int val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicOr(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicOr_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicOr(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicOr_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicXor(int* address, int val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicXor_system(int* address, int val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicXor(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicXor_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicXor(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicXor_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

#else

__device__
inline
int atomicCAS(int* address, int compare, int val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}
__device__
inline
unsigned int atomicCAS(
    unsigned int* address, unsigned int compare, unsigned int val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}
__device__
inline
unsigned long long atomicCAS(
    unsigned long long* address,
    unsigned long long compare,
    unsigned long long val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}

__device__
inline
int atomicAdd(int* address, int val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicAdd(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicAdd(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
float atomicAdd(float* address, float val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}

#if !defined(__HIPCC_RTC__)
DEPRECATED("use atomicAdd instead")
#endif // !defined(__HIPCC_RTC__)
__device__
inline
void atomicAddNoRet(float* address, float val)
{
    __ockl_atomic_add_noret_f32(address, val);
}

__device__
inline
double atomicAdd(double* address, double val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicSub(int* address, int val)
{
    return __atomic_fetch_sub(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicSub(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_sub(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicExch(int* address, int val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicExch(unsigned int* address, unsigned int val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicExch(unsigned long long* address, unsigned long long val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
float atomicExch(float* address, float val)
{
    return __uint_as_float(__atomic_exchange_n(
        reinterpret_cast<unsigned int*>(address),
        __float_as_uint(val),
        __ATOMIC_RELAXED));
}

__device__
inline
int atomicMin(int* address, int val)
{
    return __atomic_fetch_min(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicMin(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_min(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicMin(
    unsigned long long* address, unsigned long long val)
{
    unsigned long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (val < tmp) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) { tmp = tmp1; continue; }

        tmp = atomicCAS(address, tmp, val);
    }

    return tmp;
}

__device__
inline
int atomicMax(int* address, int val)
{
    return __atomic_fetch_max(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicMax(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_max(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicMax(
    unsigned long long* address, unsigned long long val)
{
    unsigned long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (tmp < val) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) { tmp = tmp1; continue; }

        tmp = atomicCAS(address, tmp, val);
    }

    return tmp;
}

__device__
inline
unsigned int atomicInc(unsigned int* address, unsigned int val)
{
    __device__
    extern
    unsigned int __builtin_amdgcn_atomic_inc(
        unsigned int*,
        unsigned int,
        unsigned int,
        unsigned int,
        bool) __asm("llvm.amdgcn.atomic.inc.i32.p0i32");

    return __builtin_amdgcn_atomic_inc(
        address, val, __ATOMIC_RELAXED, 1 /* Device scope */, false);
}

__device__
inline
unsigned int atomicDec(unsigned int* address, unsigned int val)
{
    __device__
    extern
    unsigned int __builtin_amdgcn_atomic_dec(
        unsigned int*,
        unsigned int,
        unsigned int,
        unsigned int,
        bool) __asm("llvm.amdgcn.atomic.dec.i32.p0i32");

    return __builtin_amdgcn_atomic_dec(
        address, val, __ATOMIC_RELAXED, 1 /* Device scope */, false);
}

__device__
inline
int atomicAnd(int* address, int val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicAnd(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicAnd(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicOr(int* address, int val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicOr(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicOr(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicXor(int* address, int val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicXor(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicXor(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}

#endif
