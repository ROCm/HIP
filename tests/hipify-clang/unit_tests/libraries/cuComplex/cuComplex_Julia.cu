// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK: #include "hip/hip_complex.h"
#include "cuComplex.h"

#define TYPEFLOAT
#define DIMX 100
#define DIMY 40
#define moveX 2
#define moveY 1

#define MAXITERATIONS 10

#ifdef TYPEFLOAT
#define TYPE float
// CHECK: #define cTYPE hipFloatComplex
#define cTYPE cuFloatComplex
// CHECK: #define cMakecuComplex(re,i) make_hipFloatComplex(re,i)
#define cMakecuComplex(re,i) make_cuFloatComplex(re,i)
#endif
#ifdef TYPEDOUBLE
// CHECK: #define TYPE hipDoubleComplex
#define TYPE cuDoubleComplex
// CHECK: #define cMakecuComplex(re,i) make_hipDoubleComplex(re,i)
#define cMakecuComplex(re,i) make_cuDoubleComplex(re,i)
#endif

__device__ cTYPE juliaFunctor(cTYPE p, cTYPE c) {
  // CHECK: return hipCaddf(hipCmulf(p, p), c);
  return cuCaddf(cuCmulf(p, p), c);
}

__device__ cTYPE convertToComplex(int x, int y, float zoom) {
  TYPE jx = 1.5 * (x - DIMX / 2) / (0.5 * zoom * DIMX) + moveX;
  TYPE jy = (y - DIMY / 2) / (0.5 * zoom * DIMY) + moveY;
  return cMakecuComplex(jx, jy);
}

__device__ int evolveComplexPoint(cTYPE p, cTYPE c) {
  int it = 1;
  // CHECK: while (it <= MAXITERATIONS && hipCabsf(p) <= 4) {
  while (it <= MAXITERATIONS && cuCabsf(p) <= 4) {
    p = juliaFunctor(p, c);
    it++;
  }
  return it;
}

__global__ void computeJulia(int* data, cTYPE c, float zoom) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i<DIMX && j<DIMY) {
    cTYPE p = convertToComplex(i, j, zoom);
    data[i*DIMY + j] = evolveComplexPoint(p, c);
  }
}
