/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#ifndef HIPCOMPLEX_H
#define HIPCOMPLEX_H

typedef struct{
    float x;
    float y;
}hipFloatComplex;

__device__ static inline float hipCrealf(hipFloatComplex z){
    return z.x;
}

__device__ static inline float hipCimagf(hipFloatComplex z){
    return z.y;
}

__device__ static inline hipFloatComplex make_hipFloatComplex(float a, float b){
    hipFloatComplex z;
    z.x = a;
    z.y = b;
    return z;
}

__device__ static inline hipFloatComplex hipConjf(hipFloatComplex z){
    hipFloatComplex ret;
    ret.x = z.x;
    ret.y = -z.y;
    return ret;
}

__device__ static inline float hipCsqabsf(hipFloatComplex z){
    return z.x * z.x + z.y * z.y;
}

__device__ static inline hipFloatComplex hipCaddf(hipFloatComplex p, hipFloatComplex q){
    return make_hipFloatComplex(p.x + q.x, p.y + q.y);
}

__device__ static inline hipFloatComplex hipCsubf(hipFloatComplex p, hipFloatComplex q){
    return make_hipFloatComplex(p.x - q.x, p.y - q.y);
}

__device__ static inline hipFloatComplex hipCmulf(hipFloatComplex p, hipFloatComplex q){
    return make_hipFloatComplex(p.x * q.x - p.y * q.y, p.y * q.x + p.x * q.y);
}

__device__ static inline hipFloatComplex hipCdivf(hipFloatComplex p, hipFloatComplex q){
    float sqabs = hipCsqabsf(q);
    hipFloatComplex ret;
    ret.x = (p.x * q.x + p.y * q.y)/sqabs;
    ret.y = (p.y * q.x - p.x * q.y)/sqabs;
    return ret;
}

__device__ static inline float hipCabsf(hipFloatComplex z){
    return sqrtf(hipCsqabsf(z));
}


typedef struct{
    double x;
    double y;
}hipDoubleComplex;

__device__ static inline double hipCreal(hipDoubleComplex z){
    return z.x;
}

__device__ static inline double hipCimag(hipDoubleComplex z){
    return z.y;
}

__device__ static inline hipDoubleComplex make_hipDoubleComplex(double a, double b){
    hipDoubleComplex z;
    z.x = a;
    z.y = b;
    return z;
}

__device__ static inline hipDoubleComplex hipConj(hipDoubleComplex z){
    hipDoubleComplex ret;
    ret.x = z.x;
    ret.y = z.y;
    return ret;
}

__device__ static inline double hipCsqabs(hipDoubleComplex z){
    return z.x * z.x + z.y * z.y;
}

__device__ static inline hipDoubleComplex hipCadd(hipDoubleComplex p, hipDoubleComplex q){
    return make_hipDoubleComplex(p.x + q.x, p.y + q.y);
}

__device__ static inline hipDoubleComplex hipCsub(hipDoubleComplex p, hipDoubleComplex q){
    return make_hipDoubleComplex(p.x - q.x, p.y - q.y);
}

__device__ static inline hipDoubleComplex hipCmul(hipDoubleComplex p, hipDoubleComplex q){
    return make_hipDoubleComplex(p.x * q.x - p.y * q.y, p.y * q.x + p.x * q.y);
}

__device__ static inline hipDoubleComplex hipCdiv(hipDoubleComplex p, hipDoubleComplex q){
    double sqabs = hipCsqabs(q);
    hipDoubleComplex ret;
    ret.x = (p.x * q.x + p.y * q.y)/sqabs;
    ret.y = (p.y * q.x - p.x * q.y)/sqabs;
    return ret;
}

__device__ static inline double hipCabs(hipDoubleComplex z){
    return sqrtf(hipCsqabs(z));
}

typedef hipFloatComplex hipComplex;

__device__ static inline hipComplex make_hipComplex(float x,
                                            float y){
    return make_hipFloatComplex(x, y);
}

__device__ static inline hipFloatComplex hipComplexDoubleToFloat
(hipDoubleComplex z){
    return make_hipFloatComplex((float)z.x, (float)z.y);
}

__device__ static inline hipDoubleComplex hipComplexFloatToDouble
(hipFloatComplex z){
    return make_hipDoubleComplex((double)z.x, (double)z.y);
}

__device__ static inline hipComplex hipCfmaf(hipComplex p, hipComplex q, hipComplex r){
    float real = (p.x * q.x) + r.x;
    float imag = (q.x * p.y) + r.y;

    real = -(p.y * q.y) + real;
    imag = (p.x * q.y) + imag;

    return make_hipComplex(real, imag);
}

__device__ static inline hipDoubleComplex hipCfma(hipDoubleComplex p, hipDoubleComplex q, hipDoubleComplex r){
    float real = (p.x * q.x) + r.x;
    float imag = (q.x * p.y) + r.y;

    real = -(p.y * q.y) + real;
    imag = (p.x * q.y) + imag;

    return make_hipDoubleComplex(real, imag);
}

#endif
