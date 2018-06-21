#pragma once

#include "device_functions.h"

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
    unsigned int* uaddr{reinterpret_cast<unsigned int*>(address)};
    unsigned int old{__atomic_load_n(uaddr, __ATOMIC_RELAXED)};
    unsigned int r;

    do {
        r = old;
        old = atomicCAS(uaddr, r, __float_as_uint(val + __uint_as_float(r)));
    } while (r != old);

    return __uint_as_float(r);
}
__device__
inline
double atomicAdd(double* address, double val)
{
    unsigned long long* uaddr{reinterpret_cast<unsigned long long*>(address)};
    unsigned long long old{__atomic_load_n(uaddr, __ATOMIC_RELAXED)};
    unsigned long long r;

    do {
        r = old;
        old = atomicCAS(
            uaddr, r, __double_as_longlong(val + __longlong_as_double(r)));
    } while (r != old);

    return __longlong_as_double(r);
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
unsigned int atomicExch(unsigned long long* address, unsigned long long val)
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
    return __sync_fetch_and_min(address, val);
}
__device__
inline
unsigned int atomicMin(unsigned int* address, unsigned int val)
{
    return __sync_fetch_and_umin(address, val);
}
__device__
inline
unsigned long long atomicMin(
    unsigned long long* address, unsigned long long val)
{
    unsigned long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (val < tmp) { tmp = atomicCAS(address, tmp, val); }

    return tmp;
}

__device__
inline
int atomicMax(int* address, int val)
{
    return __sync_fetch_and_max(address, val);
}
__device__
inline
unsigned int atomicMax(unsigned int* address, unsigned int val)
{
    return __sync_fetch_and_umax(address, val);
}
__device__
inline
unsigned long long atomicMax(
    unsigned long long* address, unsigned long long val)
{
    unsigned long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (tmp < val) { tmp = atomicCAS(address, tmp, val); }

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

// TODO: add scoped atomics i.e. atomic{*}_system && atomic{*}_block.
