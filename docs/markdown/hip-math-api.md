# HIP MATH APIs Documentation 
HIP supports most of the device functions supported by CUDA. Way to find the unsupported one is to search for the function and check its description
Note: This document is not human generated. Any changes to this file will be discarded. Please make changes to Python3 script docs/markdown/device_md_gen.py

## For Developers 
If you add or fixed a device function, make sure to add a signature of the function and definition later.
For example, if you want to add `__device__ float __dotf(float4, float4)`, which does a dot product on 4 float vector components 
The way to add to the header is, 
```cpp 
__device__ static float __dotf(float4, float4); 
/*Way down in the file....*/
__device__ static inline float __dotf(float4 x, float4 y) { 
 /*implementation*/
}
```

This helps python script to add the device function newly declared into markdown documentation (as it looks at functions with `;` at the end and `__device__` at the beginning)

The next step would be to add Description to  `deviceFuncDesc` dictionary in python script.
From the above example, it can be writtern as,
`deviceFuncDesc['__dotf'] = 'This functions takes 2 4 component float vector and outputs dot product across them'`

### acosf
```cpp 
__device__ float acosf(float x);

```
**Description:**  This function returns floating point of arc cosine from a floating point input


### acoshf
```cpp 
__device__ float acoshf(float x);

```
**Description:**  Supported


### asinf
```cpp 
__device__ float asinf(float x);

```
**Description:**  Supported


### asinhf
```cpp 
__device__ float asinhf(float x);

```
**Description:**  Supported


### atan2f
```cpp 
__device__ float atan2f(float y, float x);

```
**Description:**  Supported


### atanf
```cpp 
__device__ float atanf(float x);

```
**Description:**  Supported


### atanhf
```cpp 
__device__ float atanhf(float x);

```
**Description:**  Supported


### cbrtf
```cpp 
__device__ float cbrtf(float x);

```
**Description:**  Supported


### ceilf
```cpp 
__device__ float ceilf(float x);

```
**Description:**  Supported


### copysignf
```cpp 
__device__ float copysignf(float x, float y);

```
**Description:**  Supported


### cosf
```cpp 
__device__ float cosf(float x);

```
**Description:**  Supported


### coshf
```cpp 
__device__ float coshf(float x);

```
**Description:**  Supported


### cospif
```cpp 
__device__ float cospif(float x);

```
**Description:**  Supported


### cyl_bessel_i0f
```cpp 
//__device__ float cyl_bessel_i0f(float x);

```
**Description:**  **NOT Supported**


### cyl_bessel_i1f
```cpp 
//__device__ float cyl_bessel_i1f(float x);

```
**Description:**  **NOT Supported**


### erfcf
```cpp 
__device__ float erfcf(float x);

```
**Description:**  Supported


### erfcinvf
```cpp 
__device__  float erfcinvf(float y);

```
**Description:**  Supported


### erfcxf
```cpp 
__device__ float erfcxf(float x);

```
**Description:**  Supported


### erff
```cpp 
__device__ float erff(float x);

```
**Description:**  Supported


### erfinvf
```cpp 
__device__ float erfinvf(float y);

```
**Description:**  Supported


### exp10f
```cpp 
__device__ float exp10f(float x);

```
**Description:**  Supported


### exp2f
```cpp 
__device__ float exp2f(float x);

```
**Description:**  Supported


### expf
```cpp 
__device__ float expf(float x);

```
**Description:**  Supported


### expm1f
```cpp 
__device__ float expm1f(float x);

```
**Description:**  Supported


### fabsf
```cpp 
__device__ float fabsf(float x);

```
**Description:**  Supported


### fdimf
```cpp 
__device__ float fdimf(float x, float y);

```
**Description:**  Supported


### fdividef
```cpp 
__device__ float fdividef(float x, float y);

```
**Description:**  Supported


### floorf
```cpp 
__device__ float floorf(float x);

```
**Description:**  Supported


### fmaf
```cpp 
__device__ float fmaf(float x, float y, float z);

```
**Description:**  Supported


### fmaxf
```cpp 
__device__ float fmaxf(float x, float y);

```
**Description:**  Supported


### fminf
```cpp 
__device__ float fminf(float x, float y);

```
**Description:**  Supported


### fmodf
```cpp 
__device__ float fmodf(float x, float y);

```
**Description:**  Supported


### frexpf
```cpp 
//__device__ float frexpf(float x, int* nptr);

```
**Description:**  **NOT Supported**


### hypotf
```cpp 
__device__ float hypotf(float x, float y);

```
**Description:**  Supported


### ilogbf
```cpp 
__device__ float ilogbf(float x);

```
**Description:**  Supported


### isfinite
```cpp 
__device__ int isfinite(float a);

```
**Description:**  Supported


### isinf
```cpp 
__device__ unsigned isinf(float a);

```
**Description:**  Supported


### isnan
```cpp 
__device__ unsigned isnan(float a);

```
**Description:**  Supported


### j0f
```cpp 
__device__ float j0f(float x);

```
**Description:**  Supported


### j1f
```cpp 
__device__ float j1f(float x);

```
**Description:**  Supported


### jnf
```cpp 
__device__ float jnf(int n, float x);

```
**Description:**  Supported


### ldexpf
```cpp 
__device__ float ldexpf(float x, int exp);

```
**Description:**  Supported


### lgammaf
```cpp 
//__device__ float lgammaf(float x);

```
**Description:**  **NOT Supported**


### llrintf
```cpp 
__device__ long long int llrintf(float x);

```
**Description:**  Supported


### llroundf
```cpp 
__device__ long long int llroundf(float x);

```
**Description:**  Supported


### log10f
```cpp 
__device__ float log10f(float x);

```
**Description:**  Supported


### log1pf
```cpp 
__device__ float log1pf(float x);

```
**Description:**  Supported


### logbf
```cpp 
__device__ float logbf(float x);

```
**Description:**  Supported


### lrintf
```cpp 
__device__ long int lrintf(float x);

```
**Description:**  Supported


### lroundf
```cpp 
__device__ long int lroundf(float x);

```
**Description:**  Supported


### modff
```cpp 
//__device__ float modff(float x, float *iptr);

```
**Description:**  **NOT Supported**


### nanf
```cpp 
__device__ float nanf(const char* tagp);

```
**Description:**  Supported


### nearbyintf
```cpp 
__device__ float nearbyintf(float x);

```
**Description:**  Supported


### nextafterf
```cpp 
//__device__ float nextafterf(float x, float y);

```
**Description:**  **NOT Supported**


### norm3df
```cpp 
__device__ float norm3df(float a, float b, float c);

```
**Description:**  Supported


### norm4df
```cpp 
__device__ float norm4df(float a, float b, float c, float d);

```
**Description:**  Supported


### normcdff
```cpp 
__device__ float normcdff(float y);

```
**Description:**  Supported


### normcdfinvf
```cpp 
__device__ float normcdfinvf(float y);

```
**Description:**  Supported


### normf
```cpp 
__device__ float normf(int dim, const float *a);

```
**Description:**  Supported


### powf
```cpp 
__device__ float powf(float x, float y);

```
**Description:**  Supported


### rcbrtf
```cpp 
__device__ float rcbrtf(float x);

```
**Description:**  Supported


### remainderf
```cpp 
__device__ float remainderf(float x, float y);

```
**Description:**  Supported


### remquof
```cpp 
__device__ float remquof(float x, float y, int *quo);

```
**Description:**  Supported


### rhypotf
```cpp 
__device__ float rhypotf(float x, float y);

```
**Description:**  Supported


### rintf
```cpp 
__device__ float rintf(float x);

```
**Description:**  Supported


### rnorm3df
```cpp 
__device__ float rnorm3df(float a, float b, float c);

```
**Description:**  Supported


### rnorm4df
```cpp 
__device__ float rnorm4df(float a, float b, float c, float d);

```
**Description:**  Supported


### rnormf
```cpp 
__device__ float rnormf(int dim, const float* a);

```
**Description:**  Supported


### roundf
```cpp 
__device__ float roundf(float x);

```
**Description:**  Supported


### rsqrtf
```cpp 
__device__ float rsqrtf(float x);

```
**Description:**  Supported


### scalblnf
```cpp 
__device__ float scalblnf(float x, long int n);

```
**Description:**  Supported


### scalbnf
```cpp 
__device__ float scalbnf(float x, int n);

```
**Description:**  Supported


### signbit
```cpp 
__device__ int signbit(float a);

```
**Description:**  Supported


### sincosf
```cpp 
__device__ void sincosf(float x, float *sptr, float *cptr);

```
**Description:**  Supported


### sincospif
```cpp 
__device__ void sincospif(float x, float *sptr, float *cptr);

```
**Description:**  Supported


### sinf
```cpp 
__device__ float sinf(float x);

```
**Description:**  Supported


### sinhf
```cpp 
__device__ float sinhf(float x);

```
**Description:**  Supported


### sinpif
```cpp 
__device__ float sinpif(float x);

```
**Description:**  Supported


### sqrtf
```cpp 
__device__ float sqrtf(float x);

```
**Description:**  Supported


### tanf
```cpp 
__device__ float tanf(float x);

```
**Description:**  Supported


### tanhf
```cpp 
__device__ float tanhf(float x);

```
**Description:**  Supported


### tgammaf
```cpp 
__device__ float tgammaf(float x);

```
**Description:**  Supported


### truncf
```cpp 
__device__ float truncf(float x);

```
**Description:**  Supported


### y0f
```cpp 
__device__ float y0f(float x);

```
**Description:**  Supported


### y1f
```cpp 
__device__ float y1f(float x);

```
**Description:**  Supported


### ynf
```cpp 
__device__ float ynf(int n, float x);

```
**Description:**  Supported


### acos
```cpp 
__device__ double acos(double x);

```
**Description:**  Supported


### acosh
```cpp 
__device__ double acosh(double x);

```
**Description:**  Supported


### asin
```cpp 
__device__ double asin(double x);

```
**Description:**  Supported


### asinh
```cpp 
__device__ double asinh(double x);

```
**Description:**  Supported


### atan
```cpp 
__device__ double atan(double x);

```
**Description:**  Supported


### atan2
```cpp 
__device__ double atan2(double y, double x);

```
**Description:**  Supported


### atanh
```cpp 
__device__ double atanh(double x);

```
**Description:**  Supported


### cbrt
```cpp 
__device__ double cbrt(double x);

```
**Description:**  Supported


### ceil
```cpp 
__device__ double ceil(double x);

```
**Description:**  Supported


### copysign
```cpp 
__device__ double copysign(double x, double y);

```
**Description:**  Supported


### cos
```cpp 
__device__ double cos(double x);

```
**Description:**  Supported


### cosh
```cpp 
__device__ double cosh(double x);

```
**Description:**  Supported


### cospi
```cpp 
__device__ double cospi(double x);

```
**Description:**  Supported


### cyl_bessel_i0
```cpp 
//__device__ double cyl_bessel_i0(double x);

```
**Description:**  **NOT Supported**


### cyl_bessel_i1
```cpp 
//__device__ double cyl_bessel_i1(double x);

```
**Description:**  **NOT Supported**


### erf
```cpp 
__device__ double erf(double x);

```
**Description:**  Supported


### erfc
```cpp 
__device__ double erfc(double x);

```
**Description:**  Supported


### erfcinv
```cpp 
__device__ double erfcinv(double y);

```
**Description:**  Supported


### erfcx
```cpp 
__device__ double erfcx(double x);

```
**Description:**  Supported


### erfinv
```cpp 
__device__ double erfinv(double x);

```
**Description:**  Supported


### exp
```cpp 
__device__ double exp(double x);

```
**Description:**  Supported


### exp10
```cpp 
__device__ double exp10(double x);

```
**Description:**  Supported


### exp2
```cpp 
__device__ double exp2(double x);

```
**Description:**  Supported


### expm1
```cpp 
__device__ double expm1(double x);

```
**Description:**  Supported


### fabs
```cpp 
__device__ double fabs(double x);

```
**Description:**  Supported


### fdim
```cpp 
__device__ double fdim(double x, double y);

```
**Description:**  Supported


### floor
```cpp 
__device__ double floor(double x);

```
**Description:**  Supported


### fma
```cpp 
__device__ double fma(double x, double y, double z);

```
**Description:**  Supported


### fmax
```cpp 
__device__ double fmax(double x, double y);

```
**Description:**  Supported


### fmin
```cpp 
__device__ double fmin(double x, double y);

```
**Description:**  Supported


### fmod
```cpp 
__device__ double fmod(double x, double y);

```
**Description:**  Supported


### frexp
```cpp 
//__device__ double frexp(double x, int *nptr);

```
**Description:**  **NOT Supported**


### hypot
```cpp 
__device__ double hypot(double x, double y);

```
**Description:**  Supported


### ilogb
```cpp 
__device__ double ilogb(double x);

```
**Description:**  Supported


### isfinite
```cpp 
__device__ int isfinite(double x);

```
**Description:**  Supported


### isinf
```cpp 
__device__ unsigned isinf(double x);

```
**Description:**  Supported


### isnan
```cpp 
__device__ unsigned isnan(double x);

```
**Description:**  Supported


### j0
```cpp 
__device__ double j0(double x);

```
**Description:**  Supported


### j1
```cpp 
__device__ double j1(double x);

```
**Description:**  Supported


### jn
```cpp 
__device__ double jn(int n, double x);

```
**Description:**  Supported


### ldexp
```cpp 
__device__ double ldexp(double x, int exp);

```
**Description:**  Supported


### lgamma
```cpp 
__device__ double lgamma(double x);

```
**Description:**  Supported


### llrint
```cpp 
__device__ long long llrint(double x);

```
**Description:**  Supported


### llround
```cpp 
__device__ long long llround(double x);

```
**Description:**  Supported


### log
```cpp 
__device__ double log(double x);

```
**Description:**  Supported


### log10
```cpp 
__device__ double log10(double x);

```
**Description:**  Supported


### log1p
```cpp 
__device__ double log1p(double x);

```
**Description:**  Supported


### log2
```cpp 
__device__ double log2(double x);

```
**Description:**  Supported


### logb
```cpp 
__device__ double logb(double x);

```
**Description:**  Supported


### lrint
```cpp 
__device__ long int lrint(double x);

```
**Description:**  Supported


### lround
```cpp 
__device__ long int lround(double x);

```
**Description:**  Supported


### modf
```cpp 
//__device__ double modf(double x, double *iptr);

```
**Description:**  **NOT Supported**


### nan
```cpp 
__device__ double nan(const char* tagp);

```
**Description:**  Supported


### nearbyint
```cpp 
__device__ double nearbyint(double x);

```
**Description:**  Supported


### nextafter
```cpp 
__device__ double nextafter(double x, double y);

```
**Description:**  Supported


### norm
```cpp 
__device__ double norm(int dim, const double* t);

```
**Description:**  Supported


### norm3d
```cpp 
__device__ double norm3d(double a, double b, double c);

```
**Description:**  Supported


### norm4d
```cpp 
__device__ double norm4d(double a, double b, double c, double d);

```
**Description:**  Supported


### normcdf
```cpp 
__device__ double normcdf(double y);

```
**Description:**  Supported


### normcdfinv
```cpp 
__device__ double normcdfinv(double y);

```
**Description:**  Supported


### pow
```cpp 
__device__ double pow(double x, double y);

```
**Description:**  Supported


### rcbrt
```cpp 
__device__ double rcbrt(double x);

```
**Description:**  Supported


### remainder
```cpp 
__device__ double remainder(double x, double y);

```
**Description:**  Supported


### remquo
```cpp 
//__device__ double remquo(double x, double y, int *quo);

```
**Description:**  **NOT Supported**


### rhypot
```cpp 
__device__ double rhypot(double x, double y);

```
**Description:**  Supported


### rint
```cpp 
__device__ double rint(double x);

```
**Description:**  Supported


### rnorm
```cpp 
__device__ double rnorm(int dim, const double* t);

```
**Description:**  Supported


### rnorm3d
```cpp 
__device__ double rnorm3d(double a, double b, double c);

```
**Description:**  Supported


### rnorm4d
```cpp 
__device__ double rnorm4d(double a, double b, double c, double d);

```
**Description:**  Supported


### round
```cpp 
__device__ double round(double x);

```
**Description:**  Supported


### rsqrt
```cpp 
__device__ double rsqrt(double x);

```
**Description:**  Supported


### scalbln
```cpp 
__device__ double scalbln(double x, long int n);

```
**Description:**  Supported


### scalbn
```cpp 
__device__ double scalbn(double x, int n);

```
**Description:**  Supported


### signbit
```cpp 
__device__ int signbit(double a);

```
**Description:**  Supported


### sin
```cpp 
__device__ double sin(double a);

```
**Description:**  Supported


### sincos
```cpp 
__device__ void sincos(double x, double *sptr, double *cptr);

```
**Description:**  Supported


### sincospi
```cpp 
__device__ void sincospi(double x, double *sptr, double *cptr);

```
**Description:**  Supported


### sinh
```cpp 
__device__ double sinh(double x);

```
**Description:**  Supported


### sinpi
```cpp 
__device__ double sinpi(double x);

```
**Description:**  Supported


### sqrt
```cpp 
__device__ double sqrt(double x);

```
**Description:**  Supported


### tan
```cpp 
__device__ double tan(double x);

```
**Description:**  Supported


### tanh
```cpp 
__device__ double tanh(double x);

```
**Description:**  Supported


### tgamma
```cpp 
__device__ double tgamma(double x);

```
**Description:**  Supported


### trunc
```cpp 
__device__ double trunc(double x);

```
**Description:**  Supported


### y0
```cpp 
__device__ double y0(double x);

```
**Description:**  Supported


### y1
```cpp 
__device__ double y1(double y);

```
**Description:**  Supported


### yn
```cpp 
__device__ double yn(int n, double x);

```
**Description:**  Supported


### __cosf
```cpp 
__device__  float __cosf(float x);

```
**Description:**  Supported


### __exp10f
```cpp 
__device__  float __exp10f(float x);

```
**Description:**  Supported


### __expf
```cpp 
__device__  float __expf(float x);

```
**Description:**  Supported


### __fadd_rd
```cpp 
__device__ static  float __fadd_rd(float x, float y);

```
**Description:**  Supported


### __fadd_rn
```cpp 
__device__ static  float __fadd_rn(float x, float y);

```
**Description:**  Supported


### __fadd_ru
```cpp 
__device__ static  float __fadd_ru(float x, float y);

```
**Description:**  Supported


### __fadd_rz
```cpp 
__device__ static  float __fadd_rz(float x, float y);

```
**Description:**  Supported


### __fdiv_rd
```cpp 
__device__ static  float __fdiv_rd(float x, float y);

```
**Description:**  Supported


### __fdiv_rn
```cpp 
__device__ static  float __fdiv_rn(float x, float y);

```
**Description:**  Supported


### __fdiv_ru
```cpp 
__device__ static  float __fdiv_ru(float x, float y);

```
**Description:**  Supported


### __fdiv_rz
```cpp 
__device__ static  float __fdiv_rz(float x, float y);

```
**Description:**  Supported


### __fdividef
```cpp 
__device__ static  float __fdividef(float x, float y);

```
**Description:**  Supported


### __fmaf_rd
```cpp 
__device__  float __fmaf_rd(float x, float y, float z);

```
**Description:**  Supported


### __fmaf_rn
```cpp 
__device__  float __fmaf_rn(float x, float y, float z);

```
**Description:**  Supported


### __fmaf_ru
```cpp 
__device__  float __fmaf_ru(float x, float y, float z);

```
**Description:**  Supported


### __fmaf_rz
```cpp 
__device__  float __fmaf_rz(float x, float y, float z);

```
**Description:**  Supported


### __fmul_rd
```cpp 
__device__ static  float __fmul_rd(float x, float y);

```
**Description:**  Supported


### __fmul_rn
```cpp 
__device__ static  float __fmul_rn(float x, float y);

```
**Description:**  Supported


### __fmul_ru
```cpp 
__device__ static  float __fmul_ru(float x, float y);

```
**Description:**  Supported


### __fmul_rz
```cpp 
__device__ static  float __fmul_rz(float x, float y);

```
**Description:**  Supported


### __frcp_rd
```cpp 
__device__  float __frcp_rd(float x);

```
**Description:**  Supported


### __frcp_rn
```cpp 
__device__  float __frcp_rn(float x);

```
**Description:**  Supported


### __frcp_ru
```cpp 
__device__  float __frcp_ru(float x);

```
**Description:**  Supported


### __frcp_rz
```cpp 
__device__  float __frcp_rz(float x);

```
**Description:**  Supported


### __frsqrt_rn
```cpp 
__device__  float __frsqrt_rn(float x);

```
**Description:**  Supported


### __fsqrt_rd
```cpp 
__device__  float __fsqrt_rd(float x);

```
**Description:**  Supported


### __fsqrt_rn
```cpp 
__device__  float __fsqrt_rn(float x);

```
**Description:**  Supported


### __fsqrt_ru
```cpp 
__device__  float __fsqrt_ru(float x);

```
**Description:**  Supported


### __fsqrt_rz
```cpp 
__device__  float __fsqrt_rz(float x);

```
**Description:**  Supported


### __fsub_rd
```cpp 
__device__ static  float __fsub_rd(float x, float y);

```
**Description:**  Supported


### __fsub_rn
```cpp 
__device__ static  float __fsub_rn(float x, float y);

```
**Description:**  Supported


### __fsub_ru
```cpp 
__device__ static  float __fsub_ru(float x, float y);

```
**Description:**  Supported


### __log10f
```cpp 
__device__  float __log10f(float x);

```
**Description:**  Supported


### __log2f
```cpp 
__device__  float __log2f(float x);

```
**Description:**  Supported


### __logf
```cpp 
__device__  float __logf(float x);

```
**Description:**  Supported


### __powf
```cpp 
__device__  float __powf(float base, float exponent);

```
**Description:**  Supported


### __saturatef
```cpp 
__device__ static  float __saturatef(float x);

```
**Description:**  Supported


### __sincosf
```cpp 
__device__  void __sincosf(float x, float *s, float *c);

```
**Description:**  Supported


### __sinf
```cpp 
__device__  float __sinf(float x);

```
**Description:**  Supported


### __tanf
```cpp 
__device__  float __tanf(float x);

```
**Description:**  Supported


### __dadd_rd
```cpp 
__device__ static  double __dadd_rd(double x, double y);

```
**Description:**  Supported


### __dadd_rn
```cpp 
__device__ static  double __dadd_rn(double x, double y);

```
**Description:**  Supported


### __dadd_ru
```cpp 
__device__ static  double __dadd_ru(double x, double y);

```
**Description:**  Supported


### __dadd_rz
```cpp 
__device__ static  double __dadd_rz(double x, double y);

```
**Description:**  Supported


### __ddiv_rd
```cpp 
__device__ static  double __ddiv_rd(double x, double y);

```
**Description:**  Supported


### __ddiv_rn
```cpp 
__device__ static  double __ddiv_rn(double x, double y);

```
**Description:**  Supported


### __ddiv_ru
```cpp 
__device__ static  double __ddiv_ru(double x, double y);

```
**Description:**  Supported


### __ddiv_rz
```cpp 
__device__ static  double __ddiv_rz(double x, double y);

```
**Description:**  Supported


### __dmul_rd
```cpp 
__device__ static  double __dmul_rd(double x, double y);

```
**Description:**  Supported


### __dmul_rn
```cpp 
__device__ static  double __dmul_rn(double x, double y);

```
**Description:**  Supported


### __dmul_ru
```cpp 
__device__ static  double __dmul_ru(double x, double y);

```
**Description:**  Supported


### __dmul_rz
```cpp 
__device__ static  double __dmul_rz(double x, double y);

```
**Description:**  Supported


### __drcp_rd
```cpp 
__device__  double __drcp_rd(double x);

```
**Description:**  Supported


### __drcp_rn
```cpp 
__device__  double __drcp_rn(double x);

```
**Description:**  Supported


### __drcp_ru
```cpp 
__device__  double __drcp_ru(double x);

```
**Description:**  Supported


### __drcp_rz
```cpp 
__device__  double __drcp_rz(double x);

```
**Description:**  Supported


### __dsqrt_rd
```cpp 
__device__  double __dsqrt_rd(double x);

```
**Description:**  Supported


### __dsqrt_rn
```cpp 
__device__  double __dsqrt_rn(double x);

```
**Description:**  Supported


### __dsqrt_ru
```cpp 
__device__  double __dsqrt_ru(double x);

```
**Description:**  Supported


### __dsqrt_rz
```cpp 
__device__  double __dsqrt_rz(double x);

```
**Description:**  Supported


### __dsub_rd
```cpp 
__device__ static  double __dsub_rd(double x, double y);

```
**Description:**  Supported


### __dsub_rn
```cpp 
__device__ static  double __dsub_rn(double x, double y);

```
**Description:**  Supported


### __dsub_ru
```cpp 
__device__ static  double __dsub_ru(double x, double y);

```
**Description:**  Supported


### __dsub_rz
```cpp 
__device__ static  double __dsub_rz(double x, double y);

```
**Description:**  Supported


### __fma_rd
```cpp 
__device__  double __fma_rd(double x, double y, double z);

```
**Description:**  Supported


### __fma_rn
```cpp 
__device__  double __fma_rn(double x, double y, double z);

```
**Description:**  Supported


### __fma_ru
```cpp 
__device__  double __fma_ru(double x, double y, double z);

```
**Description:**  Supported


### __fma_rz
```cpp 
__device__  double __fma_rz(double x, double y, double z);

```
**Description:**  Supported


### __brev
```cpp 
__device__ unsigned int __brev( unsigned int x);

```
**Description:**  Supported


### __brevll
```cpp 
__device__ unsigned long long int __brevll( unsigned long long int x);

```
**Description:**  Supported


### __byte_perm
```cpp 
__device__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);

```
**Description:**  Supported


### __clz
```cpp 
__device__ unsigned int __clz(int x);

```
**Description:**  Supported


### __clzll
```cpp 
__device__ unsigned int __clzll(long long int x);

```
**Description:**  Supported


### __ffs
```cpp 
__device__ unsigned int __ffs(int x);

```
**Description:**  Supported


### __ffsll
```cpp 
__device__ unsigned int __ffsll(long long int x);

```
**Description:**  Supported


### __hadd
```cpp 
__device__ static unsigned int __hadd(int x, int y);

```
**Description:**  Supported


### __mul24
```cpp 
__device__ static int __mul24(int x, int y);

```
**Description:**  Supported


### __mul64hi
```cpp 
__device__ long long int __mul64hi(long long int x, long long int y);

```
**Description:**  Supported


### __mulhi
```cpp 
__device__ static int __mulhi(int x, int y);

```
**Description:**  Supported


### __popc
```cpp 
__device__ unsigned int __popc(unsigned int x);

```
**Description:**  Supported


### __popcll
```cpp 
__device__ unsigned int __popcll(unsigned long long int x);

```
**Description:**  Supported


### __rhadd
```cpp 
__device__ static int __rhadd(int x, int y);

```
**Description:**  Supported


### __sad
```cpp 
__device__ static unsigned int __sad(int x, int y, int z);

```
**Description:**  Supported


### __uhadd
```cpp 
__device__ static unsigned int __uhadd(unsigned int x, unsigned int y);

```
**Description:**  Supported


### __umul24
```cpp 
__device__ static int __umul24(unsigned int x, unsigned int y);

```
**Description:**  Supported


### __umul64hi
```cpp 
__device__ unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y);

```
**Description:**  Supported


### __umulhi
```cpp 
__device__ static unsigned int __umulhi(unsigned int x, unsigned int y);

```
**Description:**  Supported


### __urhadd
```cpp 
__device__ static unsigned int __urhadd(unsigned int x, unsigned int y);

```
**Description:**  Supported


### __usad
```cpp 
__device__ static unsigned int __usad(unsigned int x, unsigned int y, unsigned int z);

```
**Description:**  Supported


### __double2float_rd
```cpp 
__device__ float __double2float_rd(double x);

```
**Description:**  Supported


### __double2float_rn
```cpp 
__device__ float __double2float_rn(double x);

```
**Description:**  Supported


### __double2float_ru
```cpp 
__device__ float __double2float_ru(double x);

```
**Description:**  Supported


### __double2float_rz
```cpp 
__device__ float __double2float_rz(double x);

```
**Description:**  Supported


### __double2hiint
```cpp 
__device__ int __double2hiint(double x);

```
**Description:**  Supported


### __double2int_rd
```cpp 
__device__ int __double2int_rd(double x);

```
**Description:**  Supported


### __double2int_rn
```cpp 
__device__ int __double2int_rn(double x);

```
**Description:**  Supported


### __double2int_ru
```cpp 
__device__ int __double2int_ru(double x);

```
**Description:**  Supported


### __double2int_rz
```cpp 
__device__ int __double2int_rz(double x);

```
**Description:**  Supported


### __double2ll_rd
```cpp 
__device__ long long int __double2ll_rd(double x);

```
**Description:**  Supported


### __double2ll_rn
```cpp 
__device__ long long int __double2ll_rn(double x);

```
**Description:**  Supported


### __double2ll_ru
```cpp 
__device__ long long int __double2ll_ru(double x);

```
**Description:**  Supported


### __double2ll_rz
```cpp 
__device__ long long int __double2ll_rz(double x);

```
**Description:**  Supported


### __double2loint
```cpp 
__device__ int __double2loint(double x);

```
**Description:**  Supported


### __double2uint_rd
```cpp 
__device__ unsigned int __double2uint_rd(double x);

```
**Description:**  Supported


### __double2uint_rn
```cpp 
__device__ unsigned int __double2uint_rn(double x);

```
**Description:**  Supported


### __double2uint_ru
```cpp 
__device__ unsigned int __double2uint_ru(double x);

```
**Description:**  Supported


### __double2uint_rz
```cpp 
__device__ unsigned int __double2uint_rz(double x);

```
**Description:**  Supported


### __double2ull_rd
```cpp 
__device__ unsigned long long int __double2ull_rd(double x);

```
**Description:**  Supported


### __double2ull_rn
```cpp 
__device__ unsigned long long int __double2ull_rn(double x);

```
**Description:**  Supported


### __double2ull_ru
```cpp 
__device__ unsigned long long int __double2ull_ru(double x);

```
**Description:**  Supported


### __double2ull_rz
```cpp 
__device__ unsigned long long int __double2ull_rz(double x);

```
**Description:**  Supported


### __double_as_longlong
```cpp 
__device__ long long int __double_as_longlong(double x);

```
**Description:**  Supported


### __float2half_rn
```cpp 
__device__ unsigned short __float2half_rn(float x);

```
**Description:**  Supported


### __half2float
```cpp 
__device__ float __half2float(unsigned short);

```
**Description:**  Supported


### __float2half_rn
```cpp 
__device__ __half __float2half_rn(float x);

```
**Description:**  Supported


### __half2float
```cpp 
__device__ float __half2float(__half);

```
**Description:**  Supported


### __float2int_rd
```cpp 
__device__ int __float2int_rd(float x);

```
**Description:**  Supported


### __float2int_rn
```cpp 
__device__ int __float2int_rn(float x);

```
**Description:**  Supported


### __float2int_ru
```cpp 
__device__ int __float2int_ru(float x);

```
**Description:**  Supported


### __float2int_rz
```cpp 
__device__ int __float2int_rz(float x);

```
**Description:**  Supported


### __float2ll_rd
```cpp 
__device__ long long int __float2ll_rd(float x);

```
**Description:**  Supported


### __float2ll_rn
```cpp 
__device__ long long int __float2ll_rn(float x);

```
**Description:**  Supported


### __float2ll_ru
```cpp 
__device__ long long int __float2ll_ru(float x);

```
**Description:**  Supported


### __float2ll_rz
```cpp 
__device__ long long int __float2ll_rz(float x);

```
**Description:**  Supported


### __float2uint_rd
```cpp 
__device__ unsigned int __float2uint_rd(float x);

```
**Description:**  Supported


### __float2uint_rn
```cpp 
__device__ unsigned int __float2uint_rn(float x);

```
**Description:**  Supported


### __float2uint_ru
```cpp 
__device__ unsigned int __float2uint_ru(float x);

```
**Description:**  Supported


### __float2uint_rz
```cpp 
__device__ unsigned int __float2uint_rz(float x);

```
**Description:**  Supported


### __float2ull_rd
```cpp 
__device__ unsigned long long int __float2ull_rd(float x);

```
**Description:**  Supported


### __float2ull_rn
```cpp 
__device__ unsigned long long int __float2ull_rn(float x);

```
**Description:**  Supported


### __float2ull_ru
```cpp 
__device__ unsigned long long int __float2ull_ru(float x);

```
**Description:**  Supported


### __float2ull_rz
```cpp 
__device__ unsigned long long int __float2ull_rz(float x);

```
**Description:**  Supported


### __float_as_int
```cpp 
__device__ int __float_as_int(float x);

```
**Description:**  Supported


### __float_as_uint
```cpp 
__device__ unsigned int __float_as_uint(float x);

```
**Description:**  Supported


### __hiloint2double
```cpp 
__device__ double __hiloint2double(int hi, int lo);

```
**Description:**  Supported


### __int2double_rn
```cpp 
__device__ double __int2double_rn(int x);

```
**Description:**  Supported


### __int2float_rd
```cpp 
__device__ float __int2float_rd(int x);

```
**Description:**  Supported


### __int2float_rn
```cpp 
__device__ float __int2float_rn(int x);

```
**Description:**  Supported


### __int2float_ru
```cpp 
__device__ float __int2float_ru(int x);

```
**Description:**  Supported


### __int2float_rz
```cpp 
__device__ float __int2float_rz(int x);

```
**Description:**  Supported


### __int_as_float
```cpp 
__device__ float __int_as_float(int x);

```
**Description:**  Supported


### __ll2double_rd
```cpp 
__device__ double __ll2double_rd(long long int x);

```
**Description:**  Supported


### __ll2double_rn
```cpp 
__device__ double __ll2double_rn(long long int x);

```
**Description:**  Supported


### __ll2double_ru
```cpp 
__device__ double __ll2double_ru(long long int x);

```
**Description:**  Supported


### __ll2double_rz
```cpp 
__device__ double __ll2double_rz(long long int x);

```
**Description:**  Supported


### __ll2float_rd
```cpp 
__device__ float __ll2float_rd(long long int x);

```
**Description:**  Supported


### __ll2float_rn
```cpp 
__device__ float __ll2float_rn(long long int x);

```
**Description:**  Supported


### __ll2float_ru
```cpp 
__device__ float __ll2float_ru(long long int x);

```
**Description:**  Supported


### __ll2float_rz
```cpp 
__device__ float __ll2float_rz(long long int x);

```
**Description:**  Supported


### __longlong_as_double
```cpp 
__device__ double __longlong_as_double(long long int x);

```
**Description:**  Supported


### __uint2double_rn
```cpp 
__device__ double __uint2double_rn(int x);

```
**Description:**  Supported


### __uint2float_rd
```cpp 
__device__ float __uint2float_rd(unsigned int x);

```
**Description:**  Supported


### __uint2float_rn
```cpp 
__device__ float __uint2float_rn(unsigned int x);

```
**Description:**  Supported


### __uint2float_ru
```cpp 
__device__ float __uint2float_ru(unsigned int x);

```
**Description:**  Supported


### __uint2float_rz
```cpp 
__device__ float __uint2float_rz(unsigned int x);

```
**Description:**  Supported


### __uint_as_float
```cpp 
__device__ float __uint_as_float(unsigned int x);

```
**Description:**  Supported


### __ull2double_rd
```cpp 
__device__ double __ull2double_rd(unsigned long long int x);

```
**Description:**  Supported


### __ull2double_rn
```cpp 
__device__ double __ull2double_rn(unsigned long long int x);

```
**Description:**  Supported


### __ull2double_ru
```cpp 
__device__ double __ull2double_ru(unsigned long long int x);

```
**Description:**  Supported


### __ull2double_rz
```cpp 
__device__ double __ull2double_rz(unsigned long long int x);

```
**Description:**  Supported


### __ull2float_rd
```cpp 
__device__ float __ull2float_rd(unsigned long long int x);

```
**Description:**  Supported


### __ull2float_rn
```cpp 
__device__ float __ull2float_rn(unsigned long long int x);

```
**Description:**  Supported


### __ull2float_ru
```cpp 
__device__ float __ull2float_ru(unsigned long long int x);

```
**Description:**  Supported


### __ull2float_rz
```cpp 
__device__ float __ull2float_rz(unsigned long long int x);

```
**Description:**  Supported


### __hadd
```cpp 
__device__ static __half __hadd(const __half a, const __half b);

```
**Description:**  Supported


### __hadd_sat
```cpp 
__device__ static __half __hadd_sat(__half a, __half b);

```
**Description:**  Supported


### __hfma
```cpp 
__device__ static __half __hfma(__half a, __half b, __half c);

```
**Description:**  Supported


### __hfma_sat
```cpp 
__device__ static __half __hfma_sat(__half a, __half b, __half c);

```
**Description:**  Supported


### __hmul
```cpp 
__device__ static __half __hmul(__half a, __half b);

```
**Description:**  Supported


### __hmul_sat
```cpp 
__device__ static __half __hmul_sat(__half a, __half b);

```
**Description:**  Supported


### __hneg
```cpp 
__device__ static __half __hneg(__half a);

```
**Description:**  Supported


### __hsub
```cpp 
__device__ static __half __hsub(__half a, __half b);

```
**Description:**  Supported


### __hsub_sat
```cpp 
__device__ static __half __hsub_sat(__half a, __half b);

```
**Description:**  Supported


### hdiv
```cpp 
__device__ static __half hdiv(__half a, __half b);

```
**Description:**  Supported


### __hadd2
```cpp 
__device__ static __half2 __hadd2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hadd2_sat
```cpp 
__device__ static __half2 __hadd2_sat(__half2 a, __half2 b);

```
**Description:**  Supported


### __hfma2
```cpp 
__device__ static __half2 __hfma2(__half2 a, __half2 b, __half2 c);

```
**Description:**  Supported


### __hfma2_sat
```cpp 
__device__ static __half2 __hfma2_sat(__half2 a, __half2 b, __half2 c);

```
**Description:**  Supported


### __hmul2
```cpp 
__device__ static __half2 __hmul2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hmul2_sat
```cpp 
__device__ static __half2 __hmul2_sat(__half2 a, __half2 b);

```
**Description:**  Supported


### __hsub2
```cpp 
__device__ static __half2 __hsub2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hneg2
```cpp 
__device__ static __half2 __hneg2(__half2 a);

```
**Description:**  Supported


### __hsub2_sat
```cpp 
__device__ static __half2 __hsub2_sat(__half2 a, __half2 b);

```
**Description:**  Supported


### h2div
```cpp 
__device__ static __half2 h2div(__half2 a, __half2 b);

```
**Description:**  Supported


### __heq
```cpp 
__device__  bool __heq(__half a, __half b);

```
**Description:**  Supported


### __hge
```cpp 
__device__  bool __hge(__half a, __half b);

```
**Description:**  Supported


### __hgt
```cpp 
__device__  bool __hgt(__half a, __half b);

```
**Description:**  Supported


### __hisinf
```cpp 
__device__  bool __hisinf(__half a);

```
**Description:**  Supported


### __hisnan
```cpp 
__device__  bool __hisnan(__half a);

```
**Description:**  Supported


### __hle
```cpp 
__device__  bool __hle(__half a, __half b);

```
**Description:**  Supported


### __hlt
```cpp 
__device__  bool __hlt(__half a, __half b);

```
**Description:**  Supported


### __hne
```cpp 
__device__  bool __hne(__half a, __half b);

```
**Description:**  Supported


### __hbeq2
```cpp 
__device__  bool __hbeq2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hbge2
```cpp 
__device__  bool __hbge2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hbgt2
```cpp 
__device__  bool __hbgt2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hble2
```cpp 
__device__  bool __hble2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hblt2
```cpp 
__device__  bool __hblt2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hbne2
```cpp 
__device__  bool __hbne2(__half2 a, __half2 b);

```
**Description:**  Supported


### __heq2
```cpp 
__device__  __half2 __heq2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hge2
```cpp 
__device__  __half2 __hge2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hgt2
```cpp 
__device__  __half2 __hgt2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hisnan2
```cpp 
__device__  __half2 __hisnan2(__half2 a);

```
**Description:**  Supported


### __hle2
```cpp 
__device__  __half2 __hle2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hlt2
```cpp 
__device__  __half2 __hlt2(__half2 a, __half2 b);

```
**Description:**  Supported


### __hne2
```cpp 
__device__  __half2 __hne2(__half2 a, __half2 b);

```
**Description:**  Supported


### hceil
```cpp 
__device__ static __half hceil(const __half h);

```
**Description:**  Supported


### hcos
```cpp 
__device__ static __half hcos(const __half h);

```
**Description:**  Supported


### hexp
```cpp 
__device__ static __half hexp(const __half h);

```
**Description:**  Supported


### hexp10
```cpp 
__device__ static __half hexp10(const __half h);

```
**Description:**  Supported


### hexp2
```cpp 
__device__ static __half hexp2(const __half h);

```
**Description:**  Supported


### hfloor
```cpp 
__device__ static __half hfloor(const __half h);

```
**Description:**  Supported


### hlog
```cpp 
__device__ static __half hlog(const __half h);

```
**Description:**  Supported


### hlog10
```cpp 
__device__ static __half hlog10(const __half h);

```
**Description:**  Supported


### hlog2
```cpp 
__device__ static __half hlog2(const __half h);

```
**Description:**  Supported


### hrcp
```cpp 
//__device__ static __half hrcp(const __half h);

```
**Description:**  **NOT Supported**


### hrint
```cpp 
__device__ static __half hrint(const __half h);

```
**Description:**  Supported


### hsin
```cpp 
__device__ static __half hsin(const __half h);

```
**Description:**  Supported


### hsqrt
```cpp 
__device__ static __half hsqrt(const __half a);

```
**Description:**  Supported


### htrunc
```cpp 
__device__ static __half htrunc(const __half a);

```
**Description:**  Supported


### h2ceil
```cpp 
__device__ static __half2 h2ceil(const __half2 h);

```
**Description:**  Supported


### h2exp
```cpp 
__device__ static __half2 h2exp(const __half2 h);

```
**Description:**  Supported


### h2exp10
```cpp 
__device__ static __half2 h2exp10(const __half2 h);

```
**Description:**  Supported


### h2exp2
```cpp 
__device__ static __half2 h2exp2(const __half2 h);

```
**Description:**  Supported


### h2floor
```cpp 
__device__ static __half2 h2floor(const __half2 h);

```
**Description:**  Supported


### h2log
```cpp 
__device__ static __half2 h2log(const __half2 h);

```
**Description:**  Supported


### h2log10
```cpp 
__device__ static __half2 h2log10(const __half2 h);

```
**Description:**  Supported


### h2log2
```cpp 
__device__ static __half2 h2log2(const __half2 h);

```
**Description:**  Supported


### h2rcp
```cpp 
__device__ static __half2 h2rcp(const __half2 h);

```
**Description:**  Supported


### h2rsqrt
```cpp 
__device__ static __half2 h2rsqrt(const __half2 h);

```
**Description:**  Supported


### h2sin
```cpp 
__device__ static __half2 h2sin(const __half2 h);

```
**Description:**  Supported


### h2sqrt
```cpp 
__device__ static __half2 h2sqrt(const __half2 h);

```
**Description:**  Supported


### __float22half2_rn
```cpp 
__device__  __half2 __float22half2_rn(const float2 a);

```
**Description:**  Supported


### __float2half
```cpp 
__device__  __half __float2half(const float a);

```
**Description:**  Supported


### __float2half2_rn
```cpp 
__device__  __half2 __float2half2_rn(const float a);

```
**Description:**  Supported


### __float2half_rd
```cpp 
__device__  __half __float2half_rd(const float a);

```
**Description:**  Supported


### __float2half_rn
```cpp 
__device__  __half __float2half_rn(const float a);

```
**Description:**  Supported


### __float2half_ru
```cpp 
__device__  __half __float2half_ru(const float a);

```
**Description:**  Supported


### __float2half_rz
```cpp 
__device__  __half __float2half_rz(const float a);

```
**Description:**  Supported


### __floats2half2_rn
```cpp 
__device__  __half2 __floats2half2_rn(const float a, const float b);

```
**Description:**  Supported


### __half22float2
```cpp 
__device__  float2 __half22float2(const __half2 a);

```
**Description:**  Supported


### __half2float
```cpp 
__device__  float __half2float(const __half a);

```
**Description:**  Supported


### half2half2
```cpp 
__device__  __half2 half2half2(const __half a);

```
**Description:**  Supported


### __half2int_rd
```cpp 
__device__  int __half2int_rd(__half h);

```
**Description:**  Supported


### __half2int_rn
```cpp 
__device__  int __half2int_rn(__half h);

```
**Description:**  Supported


### __half2int_ru
```cpp 
__device__  int __half2int_ru(__half h);

```
**Description:**  Supported


### __half2int_rz
```cpp 
__device__  int __half2int_rz(__half h);

```
**Description:**  Supported


### __half2ll_rd
```cpp 
__device__  long long int __half2ll_rd(__half h);

```
**Description:**  Supported


### __half2ll_rn
```cpp 
__device__  long long int __half2ll_rn(__half h);

```
**Description:**  Supported


### __half2ll_ru
```cpp 
__device__  long long int __half2ll_ru(__half h);

```
**Description:**  Supported


### __half2ll_rz
```cpp 
__device__  long long int __half2ll_rz(__half h);

```
**Description:**  Supported


### __half2short_rd
```cpp 
__device__  short __half2short_rd(__half h);

```
**Description:**  Supported


### __half2short_rn
```cpp 
__device__  short __half2short_rn(__half h);

```
**Description:**  Supported


### __half2short_ru
```cpp 
__device__  short __half2short_ru(__half h);

```
**Description:**  Supported


### __half2short_rz
```cpp 
__device__  short __half2short_rz(__half h);

```
**Description:**  Supported


### __half2uint_rd
```cpp 
__device__  unsigned int __half2uint_rd(__half h);

```
**Description:**  Supported


### __half2uint_rn
```cpp 
__device__  unsigned int __half2uint_rn(__half h);

```
**Description:**  Supported


### __half2uint_ru
```cpp 
__device__  unsigned int __half2uint_ru(__half h);

```
**Description:**  Supported


### __half2uint_rz
```cpp 
__device__  unsigned int __half2uint_rz(__half h);

```
**Description:**  Supported


### __half2ull_rd
```cpp 
__device__  unsigned long long int __half2ull_rd(__half h);

```
**Description:**  Supported


### __half2ull_rn
```cpp 
__device__  unsigned long long int __half2ull_rn(__half h);

```
**Description:**  Supported


### __half2ull_ru
```cpp 
__device__  unsigned long long int __half2ull_ru(__half h);

```
**Description:**  Supported


### __half2ull_rz
```cpp 
__device__  unsigned long long int __half2ull_rz(__half h);

```
**Description:**  Supported


### __half2ushort_rd
```cpp 
__device__  unsigned short int __half2ushort_rd(__half h);

```
**Description:**  Supported


### __half2ushort_rn
```cpp 
__device__  unsigned short int __half2ushort_rn(__half h);

```
**Description:**  Supported


### __half2ushort_ru
```cpp 
__device__  unsigned short int __half2ushort_ru(__half h);

```
**Description:**  Supported


### __half2ushort_rz
```cpp 
__device__  unsigned short int __half2ushort_rz(__half h);

```
**Description:**  Supported


### __half_as_short
```cpp 
__device__  short int __half_as_short(const __half h);

```
**Description:**  Supported


### __half_as_ushort
```cpp 
__device__  unsigned short int __half_as_ushort(const __half h);

```
**Description:**  Supported


### __halves2half2
```cpp 
__device__  __half2 __halves2half2(const __half a, const __half b);

```
**Description:**  Supported


### __high2float
```cpp 
__device__  float __high2float(const __half2 a);

```
**Description:**  Supported


### __high2half
```cpp 
__device__  __half __high2half(const __half2 a);

```
**Description:**  Supported


### __high2half2
```cpp 
__device__  __half2 __high2half2(const __half2 a);

```
**Description:**  Supported


### __highs2half2
```cpp 
__device__  __half2 __highs2half2(const __half2 a, const __half2 b);

```
**Description:**  Supported


### __int2half_rd
```cpp 
__device__  __half __int2half_rd(int i);

```
**Description:**  Supported


### __int2half_rn
```cpp 
__device__  __half __int2half_rn(int i);

```
**Description:**  Supported


### __int2half_ru
```cpp 
__device__  __half __int2half_ru(int i);

```
**Description:**  Supported


### __int2half_rz
```cpp 
__device__  __half __int2half_rz(int i);

```
**Description:**  Supported


### __ll2half_rd
```cpp 
__device__  __half __ll2half_rd(long long int i);

```
**Description:**  Supported


### __ll2half_rn
```cpp 
__device__  __half __ll2half_rn(long long int i);

```
**Description:**  Supported


### __ll2half_ru
```cpp 
__device__  __half __ll2half_ru(long long int i);

```
**Description:**  Supported


### __ll2half_rz
```cpp 
__device__  __half __ll2half_rz(long long int i);

```
**Description:**  Supported


### __low2float
```cpp 
__device__  float __low2float(const __half2 a);

```
**Description:**  Supported


### __low2half
```cpp 
__device__ __half __low2half(const __half2 a);

```
**Description:**  Supported


### __low2half2
```cpp 
__device__ __half2 __low2half2(const __half2 a, const __half2 b);

```
**Description:**  Supported


### __low2half2
```cpp 
__device__ __half2 __low2half2(const __half2 a);

```
**Description:**  Supported


### __lowhigh2highlow
```cpp 
__device__ __half2 __lowhigh2highlow(const __half2 a);

```
**Description:**  Supported


### __lows2half2
```cpp 
__device__ __half2 __lows2half2(const __half2 a, const __half2 b);

```
**Description:**  Supported


### __short2half_rd
```cpp 
__device__  __half __short2half_rd(short int i);

```
**Description:**  Supported


### __short2half_rn
```cpp 
__device__  __half __short2half_rn(short int i);

```
**Description:**  Supported


### __short2half_ru
```cpp 
__device__  __half __short2half_ru(short int i);

```
**Description:**  Supported


### __short2half_rz
```cpp 
__device__  __half __short2half_rz(short int i);

```
**Description:**  Supported


### __uint2half_rd
```cpp 
__device__  __half __uint2half_rd(unsigned int i);

```
**Description:**  Supported


### __uint2half_rn
```cpp 
__device__  __half __uint2half_rn(unsigned int i);

```
**Description:**  Supported


### __uint2half_ru
```cpp 
__device__  __half __uint2half_ru(unsigned int i);

```
**Description:**  Supported


### __uint2half_rz
```cpp 
__device__  __half __uint2half_rz(unsigned int i);

```
**Description:**  Supported


### __ull2half_rd
```cpp 
__device__  __half __ull2half_rd(unsigned long long int i);

```
**Description:**  Supported


### __ull2half_rn
```cpp 
__device__  __half __ull2half_rn(unsigned long long int i);

```
**Description:**  Supported


### __ull2half_ru
```cpp 
__device__  __half __ull2half_ru(unsigned long long int i);

```
**Description:**  Supported


### __ull2half_rz
```cpp 
__device__  __half __ull2half_rz(unsigned long long int i);

```
**Description:**  Supported


### __ushort2half_rd
```cpp 
__device__  __half __ushort2half_rd(unsigned short int i);

```
**Description:**  Supported


### __ushort2half_rn
```cpp 
__device__  __half __ushort2half_rn(unsigned short int i);

```
**Description:**  Supported


### __ushort2half_ru
```cpp 
__device__  __half __ushort2half_ru(unsigned short int i);

```
**Description:**  Supported


### __ushort2half_rz
```cpp 
__device__  __half __ushort2half_rz(unsigned short int i);

```
**Description:**  Supported


### __ushort_as_half
```cpp 
__device__  __half __ushort_as_half(const unsigned short int i);

```
**Description:**  Supported


