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

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COMPLEX_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COMPLEX_H

#include "hip/hcc_detail/hip_vector_types.h"

// TODO: Clang has a bug which allows device functions to call std functions
// when std functions are introduced into default namespace by using statement.
// math.h may be included after this bug is fixed.
#if __cplusplus
#include <cmath>
#else
#include "math.h"
#endif

#if __cplusplus
#define COMPLEX_NEG_OP_OVERLOAD(type)                                                              \
    __device__ __host__ static inline type operator-(const type& op) {                             \
        type ret;                                                                                  \
        ret.CmplxNum.x = -op.CmplxNum.x;                                                           \
        ret.CmplxNum.y = -op.CmplxNum.y;                                                           \
        return ret;                                                                                \
    }

#define COMPLEX_EQ_OP_OVERLOAD(type)                                                               \
    __device__ __host__ static inline bool operator==(const type& lhs, const type& rhs) {          \
        return lhs.CmplxNum.x == rhs.CmplxNum.x && lhs.CmplxNum.y == rhs.CmplxNum.y;               \
    }

#define COMPLEX_NE_OP_OVERLOAD(type)                                                               \
    __device__ __host__ static inline bool operator!=(const type& lhs, const type& rhs) {          \
        return !(lhs == rhs);                                                                      \
    }

#define COMPLEX_ADD_OP_OVERLOAD(type)                                                              \
    __device__ __host__ static inline type operator+(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.CmplxNum.x = lhs.CmplxNum.x + rhs.CmplxNum.x;                                          \
        ret.CmplxNum.y = lhs.CmplxNum.y + rhs.CmplxNum.y;                                          \
        return ret;                                                                                \
    }

#define COMPLEX_SUB_OP_OVERLOAD(type)                                                              \
    __device__ __host__ static inline type operator-(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.CmplxNum.x = lhs.CmplxNum.x - rhs.CmplxNum.x;                                          \
        ret.CmplxNum.y = lhs.CmplxNum.y - rhs.CmplxNum.y;                                          \
        return ret;                                                                                \
    }

#define COMPLEX_MUL_OP_OVERLOAD(type)                                                              \
    __device__ __host__ static inline type operator*(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.CmplxNum.x = lhs.CmplxNum.x * rhs.CmplxNum.x - lhs.CmplxNum.y * rhs.CmplxNum.y;        \
        ret.CmplxNum.y = lhs.CmplxNum.x * rhs.CmplxNum.y + lhs.CmplxNum.y * rhs.CmplxNum.x;        \
        return ret;                                                                                \
    }

#define COMPLEX_DIV_OP_OVERLOAD(type)                                                              \
    __device__ __host__ static inline type operator/(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.CmplxNum.x = (lhs.CmplxNum.x * rhs.CmplxNum.x + lhs.CmplxNum.y * rhs.CmplxNum.y);      \
        ret.CmplxNum.y = (rhs.CmplxNum.x * lhs.CmplxNum.y - lhs.CmplxNum.x * rhs.CmplxNum.y);      \
        ret.CmplxNum.x = ret.CmplxNum.x / (rhs.CmplxNum.x * rhs.CmplxNum.x + rhs.CmplxNum.y * rhs.CmplxNum.y);                                           \
        ret.CmplxNum.y = ret.CmplxNum.y / (rhs.CmplxNum.x * rhs.CmplxNum.x + rhs.CmplxNum.y * rhs.CmplxNum.y);                                           \
        return ret;                                                                                \
    }

#define COMPLEX_ADD_PREOP_OVERLOAD(type)                                                           \
    __device__ __host__ static inline type& operator+=(type& lhs, const type& rhs) {               \
        lhs.CmplxNum.x += rhs.CmplxNum.x;                                                          \
        lhs.CmplxNum.y += rhs.CmplxNum.y;                                                          \
        return lhs;                                                                                \
    }

#define COMPLEX_SUB_PREOP_OVERLOAD(type)                                                           \
    __device__ __host__ static inline type& operator-=(type& lhs, const type& rhs) {               \
        lhs.CmplxNum.x -= rhs.CmplxNum.x;                                                          \
        lhs.CmplxNum.y -= rhs.CmplxNum.y;                                                          \
        return lhs;                                                                                \
    }

#define COMPLEX_MUL_PREOP_OVERLOAD(type)                                                           \
    __device__ __host__ static inline type& operator*=(type& lhs, const type& rhs) {               \
        lhs = lhs * rhs;                                                                           \
        return lhs;                                                                                \
    }

#define COMPLEX_DIV_PREOP_OVERLOAD(type)                                                           \
    __device__ __host__ static inline type& operator/=(type& lhs, const type& rhs) {               \
        lhs = lhs / rhs;                                                                           \
        return lhs;                                                                                \
    }

#define COMPLEX_SCALAR_PRODUCT(type, type1)                                                        \
    __device__ __host__ static inline type operator*(const type& lhs, type1 rhs) {                 \
        type ret;                                                                                  \
        ret.CmplxNum.x = lhs.CmplxNum.x * rhs;                                                     \
        ret.CmplxNum.y = lhs.CmplxNum.y * rhs;                                                     \
        return ret;                                                                                \
    }

#endif

typedef struct hipFloatComplex
{
   float2 CmplxNum;
};

__device__ __host__ static inline float hipCrealf(hipFloatComplex z) { return z.CmplxNum.x; }

__device__ __host__ static inline float hipCimagf(hipFloatComplex z) { return z.CmplxNum.y; }

__device__ __host__ static inline hipFloatComplex make_hipFloatComplex(float a, float b) {
    hipFloatComplex z;
    z.CmplxNum.x = a;
    z.CmplxNum.y = b;
    return z;
}

__device__ __host__ static inline hipFloatComplex hipConjf(hipFloatComplex z) {
    hipFloatComplex ret;
    ret.CmplxNum.x = z.CmplxNum.x;
    ret.CmplxNum.y = -z.CmplxNum.y;
    return ret;
}

__device__ __host__ static inline float hipCsqabsf(hipFloatComplex z) {
    return z.CmplxNum.x * z.CmplxNum.x + z.CmplxNum.y * z.CmplxNum.y;
}

__device__ __host__ static inline hipFloatComplex hipCaddf(hipFloatComplex p, hipFloatComplex q) {
    return make_hipFloatComplex(p.CmplxNum.x + q.CmplxNum.x, p.CmplxNum.y + q.CmplxNum.y);
}

__device__ __host__ static inline hipFloatComplex hipCsubf(hipFloatComplex p, hipFloatComplex q) {
    return make_hipFloatComplex(p.CmplxNum.x - q.CmplxNum.x, p.CmplxNum.y - q.CmplxNum.y);
}

__device__ __host__ static inline hipFloatComplex hipCmulf(hipFloatComplex p, hipFloatComplex q) {
    return make_hipFloatComplex(p.CmplxNum.x * q.CmplxNum.x - p.CmplxNum.y * q.CmplxNum.y, p.CmplxNum.y * q.CmplxNum.x + p.CmplxNum.x * q.CmplxNum.y);
}

__device__ __host__ static inline hipFloatComplex hipCdivf(hipFloatComplex p, hipFloatComplex q) {
    float sqabs = hipCsqabsf(q);
    hipFloatComplex ret;
    ret.CmplxNum.x = (p.CmplxNum.x * q.CmplxNum.x + p.CmplxNum.y * q.CmplxNum.y) / sqabs;
    ret.CmplxNum.y = (p.CmplxNum.y * q.CmplxNum.x - p.CmplxNum.x * q.CmplxNum.y) / sqabs;
    return ret;
}

__device__ __host__ static inline float hipCabsf(hipFloatComplex z) { return sqrtf(hipCsqabsf(z)); }


typedef struct hipDoubleComplex
{
    double2 CmplxNum; 
};

//typedef double2 hipDoubleComplex;

__device__ __host__ static inline double hipCreal(hipDoubleComplex z) { return z.CmplxNum.x; }

__device__ __host__ static inline double hipCimag(hipDoubleComplex z) { return z.CmplxNum.y; }

__device__ __host__ static inline hipDoubleComplex make_hipDoubleComplex(double a, double b) {
    hipDoubleComplex z;
    z.CmplxNum.x = a;
    z.CmplxNum.y = b;
    return z;
}

__device__ __host__ static inline hipDoubleComplex hipConj(hipDoubleComplex z) {
    hipDoubleComplex ret;
    ret.CmplxNum.x = z.CmplxNum.x;
    ret.CmplxNum.y = z.CmplxNum.y;
    return ret;
}

__device__ __host__ static inline double hipCsqabs(hipDoubleComplex z) {
    return z.CmplxNum.x * z.CmplxNum.x + z.CmplxNum.y * z.CmplxNum.y;
}

__device__ __host__ static inline hipDoubleComplex hipCadd(hipDoubleComplex p, hipDoubleComplex q) {
    return make_hipDoubleComplex(p.CmplxNum.x + q.CmplxNum.x, p.CmplxNum.y + q.CmplxNum.y);
}

__device__ __host__ static inline hipDoubleComplex hipCsub(hipDoubleComplex p, hipDoubleComplex q) {
    return make_hipDoubleComplex(p.CmplxNum.x - q.CmplxNum.x, p.CmplxNum.y - q.CmplxNum.y);
}

__device__ __host__ static inline hipDoubleComplex hipCmul(hipDoubleComplex p, hipDoubleComplex q) {
    return make_hipDoubleComplex(p.CmplxNum.x * q.CmplxNum.x - p.CmplxNum.y * q.CmplxNum.y, p.CmplxNum.y * q.CmplxNum.x + p.CmplxNum.x * q.CmplxNum.y);
}

__device__ __host__ static inline hipDoubleComplex hipCdiv(hipDoubleComplex p, hipDoubleComplex q) {
    double sqabs = hipCsqabs(q);
    hipDoubleComplex ret;
    ret.CmplxNum.x = (p.CmplxNum.x * q.CmplxNum.x + p.CmplxNum.y * q.CmplxNum.y) / sqabs;
    ret.CmplxNum.y = (p.CmplxNum.y * q.CmplxNum.x - p.CmplxNum.x * q.CmplxNum.y) / sqabs;
    return ret;
}

__device__ __host__ static inline double hipCabs(hipDoubleComplex z) { return sqrtf(hipCsqabs(z)); }


#if __cplusplus

COMPLEX_NEG_OP_OVERLOAD(hipFloatComplex)
COMPLEX_EQ_OP_OVERLOAD(hipFloatComplex)
COMPLEX_NE_OP_OVERLOAD(hipFloatComplex)
COMPLEX_ADD_OP_OVERLOAD(hipFloatComplex)
COMPLEX_SUB_OP_OVERLOAD(hipFloatComplex)
COMPLEX_MUL_OP_OVERLOAD(hipFloatComplex)
COMPLEX_DIV_OP_OVERLOAD(hipFloatComplex)
COMPLEX_ADD_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_SUB_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_MUL_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_DIV_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned short)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed short)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned int)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed int)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, float)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned long)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed long)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, double)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed long long)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned long long)

COMPLEX_NEG_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_EQ_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_NE_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_ADD_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_SUB_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_MUL_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_DIV_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_ADD_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_SUB_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_MUL_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_DIV_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned short)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed short)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned int)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed int)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, float)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned long)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed long)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, double)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed long long)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned long long)

#endif


typedef hipFloatComplex hipComplex;

__device__ __host__ static inline hipComplex make_hipComplex(float x, float y) {
    return make_hipFloatComplex(x, y);
}

__device__ __host__ static inline hipFloatComplex hipComplexDoubleToFloat(hipDoubleComplex z) {
    return make_hipFloatComplex((float)z.CmplxNum.x, (float)z.CmplxNum.y);
}

__device__ __host__ static inline hipDoubleComplex hipComplexFloatToDouble(hipFloatComplex z) {
    return make_hipDoubleComplex((double)z.CmplxNum.x, (double)z.CmplxNum.y);
}

__device__ __host__ static inline hipComplex hipCfmaf(hipComplex p, hipComplex q, hipComplex r) {
    float real = (p.CmplxNum.x * q.CmplxNum.x) + r.CmplxNum.x;
    float imag = (q.CmplxNum.x * p.CmplxNum.y) + r.CmplxNum.y;

    real = -(p.CmplxNum.y * q.CmplxNum.y) + real;
    imag = (p.CmplxNum.x * q.CmplxNum.y) + imag;

    return make_hipComplex(real, imag);
}

__device__ __host__ static inline hipDoubleComplex hipCfma(hipDoubleComplex p, hipDoubleComplex q,
                                                           hipDoubleComplex r) {
    double real = (p.CmplxNum.x * q.CmplxNum.x) + r.CmplxNum.x;
    double imag = (q.CmplxNum.x * p.CmplxNum.y) + r.CmplxNum.y;

    real = -(p.CmplxNum.y * q.CmplxNum.y) + real;
    imag = (p.CmplxNum.x * q.CmplxNum.x) + imag;

    return make_hipDoubleComplex(real, imag);
}

#endif //HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COMPLEX_H
