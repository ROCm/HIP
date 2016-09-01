#ifndef HIPCOMPLEX_H
#define HIPCOMPLEX_H

#include"cuComplex.h"

typedef cuFloatComplex hipFloatComplex;

__device__ __host__ static inline float hipCrealf(hipFloatComplex z){
    return cuCrealf(z);
}

__device__ __host__ static inline float hipCimagf(hipFloatComplex z){
    return cuCimagf(z);
}

__device__ __host__ static inline hipFloatComplex make_hipFloatComplex(float a, float b){
    return make_cuFloatComplex(a, b);
}

__device__ __host__ static inline hipFloatComplex hipConjf(hipFloatComplex z){
    return cuConjf(z);
}

__device__ __host__ static inline float hipCsqabsf(hipFloatComplex z){
    return cuCabsf(z) * cuCabsf(z);
}

__device__ __host__ static inline hipFloatComplex hipCaddf(hipFloatComplex p, hipFloatComplex q){
    return cuCaddf(p, q);
}

__device__ __host__ static inline hipFloatComplex hipCsubf(hipFloatComplex p, hipFloatComplex q){
    return cuCsubf(p, q);
}

__device__ __host__ static inline hipFloatComplex hipCmulf(hipFloatComplex p, hipFloatComplex q){
    return cuCmulf(p, q);
}

__device__ __host__ static inline hipFloatComplex hipCdivf(hipFloatComplex p, hipFloatComplex q){
    return cuCdivf(p, q);
}

__device__ __host__ static inline float hipCabsf(hipFloatComplex z){
    return cuCabsf(p, q);
}

typedef cuDoubleComplex hipDoubleComplex;

__device__ __host__ static inline double hipCreal(hipDoubleComplex z){
    return cuCreal(z);
}

__device__ __host__ static inline double hipCimag(hipDoubleComplex z){
    return cuCimag(z);
}

__device__ __host__ static inline hipDoubleComplex make_hipDoubleComplex(double a, double b){
    return make_cuDoubleComplex(a, b);
}

__device__ __host__ static inline hipDoubleComplex hipConj(hipDoubleComplex z){
    return cuConj(z);
}

__device__ __host__ static inline hipDoubleComplex hipCsqabs(hipDoubleComplex z){
    return cuCabs(z) * cuCabs(z);
}

__device__ __host__ static inline hipDoubleComplex hipCadd(hipDoubleComplex p, hipDoubleComplex q){
    return cuCadd(p, q);
}

__device__ __host__ static inline hipDoubleComplex hipCsub(hipDoubleComplex p, hipDoubleComplex q){
    return cuCsub(p, q);
}

__device__ __host__ static inline hipDoubleComplex hipCdiv(hipDoubleComplex p, hipDoubleComplex q){
    return cuCdiv(p, q);
}

__device__ __host__ static inline double hipCabs(hipDoubleComplex z){
    return cuCabs(z);
}

typedef cuFloatComplex hipComplex;

__device__ __host__ static inline hipComplex make_Complex(float x, float y){
    return make_cuComplex(x, y);
}

__device__ __host__ static inline hipFloatComplex hipComplexDoubleToFloat(hipDoubleComplex z){
    return cuComplexDoubleToFloat(z);
}

__device__ __host__ static inline hipDoubleComplex hipComplexFloatToDouble(hipFloatComplex z){
    return cuComplexFloatToDouble(z);
}

__device__ __host__ static inline hipComplex hipCfmaf(hipComplex p, hipComplex q, hipComplex r){
    return cuCfmaf(p, q, r);
}

__device__ __host__ static inline hipDoubleComplex hipCfma(hipComplex p, hipComplex q, hipComplex r){
    return cuCfma(p, q, r);
}

#endif
