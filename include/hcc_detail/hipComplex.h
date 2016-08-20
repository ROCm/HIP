#ifndef HIPCOMPLEX_H
#define HIPCOMPLEX_H


typedef struct{
    float x;
    float y;
}hipFloatComplex;

__device__ inline float hipCrealf(hipFloatComplex z){
    return z.x;
}

__device__ inline float hipCimagf(hipFloatComplex z){
    return z.y;
}

__device__ inline hipFloatComplex make_hipFloatComplex(float a, float b){
    hipFloatComplex z;
    z.x = a;
    z.y = b;
    return z;
}

__device__ inline hipFloatComplex hipConjf(hipFloatComplex z){
    hipFloatComplex ret;
    ret.x = z.x;
    ret.y = z.y;
    return ret;
}

__device__ inline float hipCsqabsf(hipFloatComplex z){
    return z.x * z.x + z.y * z.y;
}

__device__ hipFloatComplex hipCaddf(hipFloatComplex p, hipFloatComplex q){
    return make_hipFloatComplex(p.x + q.x, p.y + q.y);
}

__device__ hipFloatComplex hipCsubf(hipFloatComplex p, hipFloatComplex q){
    return make_hipFloatComplex(p.x - q.x, p.y - q.y);
}

__device__ hipFloatComplex hipCmulf(hipFloatComplex p, hipFloatComplex q){
    return make_hipFloatComplex(p.x * q.x - p.y * q.y, p.y * q.x + p.x * q.y);
}

__device__ hipFloatComplex hipCdivf(hipFloatComplex p, hipFloatComplex q){
    float sqabs = hipCsqabsf(q);
    hipFloatComplex ret;
    ret.x = (p.x * q.x + p.y * q.y)/sqabs;
    ret.y = (p.y * q.x - p.x * q.y)/sqabs;
    return ret;
}

__device__ float hipCabsf(hipFloatComplex z){
    return sqrtf(hipCsqabsf(z));
}


typedef struct{
    double x;
    double y;
}hipDoubleComplex;

__device__ inline double hipCreal(hipDoubleComplex z){
    return z.x;
}

__device__ inline double hipCimag(hipDoubleComplex z){
    return z.y;
}

__device__ inline hipDoubleComplex make_hipDoubleComplex(double a, double b){
    hipDoubleComplex z;
    z.x = a;
    z.y = b;
    return z;
}

__device__ inline hipDoubleComplex hipConj(hipDoubleComplex z){
    hipDoubleComplex ret;
    ret.x = z.x;
    ret.y = z.y;
    return ret;
}

__device__ inline double hipCsqabs(hipDoubleComplex z){
    return z.x * z.x + z.y * z.y;
}

__device__ hipDoubleComplex hipCadd(hipDoubleComplex p, hipDoubleComplex q){
    return make_hipDoubleComplex(p.x + q.x, p.y + q.y);
}

__device__ hipDoubleComplex hipCsub(hipDoubleComplex p, hipDoubleComplex q){
    return make_hipDoubleComplex(p.x - q.x, p.y - q.y);
}

__device__ hipDoubleComplex hipCmul(hipDoubleComplex p, hipDoubleComplex q){
    return make_hipDoubleComplex(p.x * q.x - p.y * q.y, p.y * q.x + p.x * q.y);
}

__device__ hipDoubleComplex hipCdiv(hipDoubleComplex p, hipDoubleComplex q){
    double sqabs = hipCsqabs(q);
    hipDoubleComplex ret;
    ret.x = (p.x * q.x + p.y * q.y)/sqabs;
    ret.y = (p.y * q.x - p.x * q.y)/sqabs;
    return ret;
}

__device__ inline double hipCabs(hipDoubleComplex z){
    return sqrtf(hipCsqabs(z));
}

#endif
