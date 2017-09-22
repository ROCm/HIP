## Table of Contents

<!-- toc -->

- [Introduction](#introduction)
- [Function-Type Qualifiers](#function-type-qualifiers)
  * [`__device__`](#__device__)
  * [`__global__`](#__global__)
  * [`__host__`](#__host__)
- [Calling `__global__` Functions](#calling-__global__-functions)
- [Kernel-Launch Example](#kernel-launch-example)
- [Variable-Type Qualifiers](#variable-type-qualifiers)
  * [`__constant__`](#__constant__)
  * [`__shared__`](#__shared__)
  * [`__managed__`](#__managed__)
  * [`__restrict__`](#__restrict__)
- [Built-In Variables](#built-in-variables)
  * [Coordinate Built-Ins](#coordinate-built-ins)
  * [warpSize](#warpsize)
- [Vector Types](#vector-types)
  * [Short Vector Types](#short-vector-types)
  * [dim3](#dim3)
- [Memory-Fence Instructions](#memory-fence-instructions)
- [Synchronization Functions](#synchronization-functions)
- [Math Functions](#math-functions)
  * [Single Precision Mathematical Functions](#single-precision-mathematical-functions)
  * [Double Precision Mathematical Functions](#double-precision-mathematical-functions)
  * [Integer Intrinsics](#integer-intrinsics)
  * [Floating-point Intrinsics](#floating-point-intrinsics)
- [Texture Functions](#texture-functions)
- [Surface Functions](#surface-functions)
- [Timer Functions](#timer-functions)
- [Atomic Functions](#atomic-functions)
  * [Caveats and Features Under-Development:](#caveats-and-features-under-development)
- [Warp Cross-Lane Functions](#warp-cross-lane-functions)
  * [Warp Vote and Ballot Functions](#warp-vote-and-ballot-functions)
  * [Warp Shuffle Functions](#warp-shuffle-functions)
- [Profiler Counter Function](#profiler-counter-function)
- [Assert](#assert)
- [Printf](#printf)
- [Device-Side Dynamic Global Memory Allocation](#device-side-dynamic-global-memory-allocation)
- [`__launch_bounds__`](#__launch_bounds__)
  * [Compiler Impact](#compiler-impact)
  * [CU and EU Definitions](#cu-and-eu-definitions)
  * [Porting from CUDA __launch_bounds](#porting-from-cuda-__launch_bounds)
  * [maxregcount](#maxregcount)
- [Register Keyword](#register-keyword)
- [Pragma Unroll](#pragma-unroll)
- [In-Line Assembly](#in-line-assembly)
- [C++ Support](#c-support)
- [Kernel Compilation](#kernel-compilation)

<!-- tocstop -->

## Introduction

HIP provides a C++ syntax that is suitable for compiling most code that commonly appears in compute kernels, including classes, namespaces, operator overloading, templates and more. Additionally, it defines other language features designed specifically to target accelerators, such as the following:
- A kernel-launch syntax that uses standard C++, resembles a function call and is portable to all HIP targets
- Short-vector headers that can serve on a host or a device
- Math functions resembling those in the "math.h" header included with standard C++ compilers
- Built-in functions for accessing specific GPU hardware capabilities

This section describes the built-in variables and functions accessible from the HIP kernel. It’s intended for readers who are familiar with Cuda kernel syntax and want to understand how HIP is different.

Features are marked with one of the following keywords:
- **Supported**---HIP supports the feature with a Cuda-equivalent function
- **Not supported**---HIP does not support the feature
- **Under development**---the feature is under development but not yet available

## Function-Type Qualifiers
### `__device__`
Supported  `__device__` functions are
  - Executed on the device
  - Called from the device only

The `__device__` keyword can combine with the host keyword (see [__host__](#host)).

### `__global__`
Supported `__global__` functions are
  - Executed on the device
  - Called ("launched") from the host

HIP `__global__` functions must have a `void` return type, and the first parameter to a HIP `__global__` function must have the type `hipLaunchParm`. See [Kernel-Launch Example](#kernel-launch-example). 

HIP lacks dynamic-parallelism support, so `__global__ ` functions cannot be called from the device.

### `__host__`
Supported `__host__` functions are
  - Executed on the host
  - Called from the host

`__host__` can combine with `__device__`, in which case the function compiles for both the host and device. These functions cannot use the HIP grid coordinate functions (for example, "hipThreadIdx_x"). A possible workaround is to pass the necessary coordinate info as an argument to the function.

`__host__` cannot combine with `__global__`.

HIP parses the `__noinline__` and `__forceinline__` keywords and converts them to the appropriate Clang attributes. The hcc compiler, however, currently in-lines all device functions, so they are effectively ignored.


## Calling `__global__` Functions

`__global__` functions are often referred to as *kernels,* and calling one is termed *launching the kernel.* These functions require the caller to specify an "execution configuration" that includes the grid and block dimensions. The execution configuration can also include other information for the launch, such as the amount of additional shared memory to allocate and the stream where the kernel should execute. HIP introduces a standard C++ calling convention to pass the execution configuration to the kernel (this convention replaces the Cuda <<< >>> syntax). In HIP,
- Kernels launch with the "hipLaunchKernel" function
- The first five parameters to hipLaunchKernel are the following:
   - **symbol kernelName**: the name of the kernel to launch.  To support template kernels which contains "," use the HIP_KERNEL_NAME macro.   The hipify tools insert this automatically.
   - **dim3 gridDim**: 3D-grid dimensions specifying the number of blocks to launch.
   - **dim3 blockDim**: 3D-block dimensions specifying the number of threads in each block.
   - **size_t dynamicShared**: amount of additional shared memory to allocate when launching the kernel (see [__shared__](#__shared__))
   - **hipStream_t**: stream where the kernel should execute. A value of 0 corresponds to the NULL stream (see [Synchronization Functions](#synchronization-functions)).
- Kernel arguments follow these first five parameters


```
// Example pseudo code introducing hipLaunchKernel:
__global__ MyKernel(hipLaunchParm lp, float *A, float *B, float *C, size_t N)
{
...
}

// Replace MyKernel<<<dim3(gridDim), dim3(gridDim), 0, 0>>> (a,b,c,n);
hipLaunchKernel(MyKernel, dim3(gridDim), dim3(groupDim), 0/*dynamicShared*/, 0/*stream), a, b, c, n);

```

The hipLaunchKernel macro always starts with the five parameters specified above, followed by the kernel arguments. The Hipify script automatically converts Cuda launch syntax to hipLaunchKernel, including conversion of optional arguments in <<< >>> to the five required hipLaunchKernel parameters. The dim3 constructor accepts zero to three arguments and will by default initialize unspecified dimensions to 1. See [dim3](#dim3). The kernel uses the coordinate built-ins (hipThread*, hipBlock*, hipGrid*) to determine coordinate index and coordinate bounds of the work item that’s currently executing. See [Coordinate Built-Ins](#coordinate-builtins).


## Kernel-Launch Example
```
// Example showing device function, __device__ __host__   
// <- compile for both device and host 
float PlusOne(float x) 
{
    return x + 1.0;
}

__global__ 
void 
MyKernel (hipLaunchParm lp, /*lp parm for execution configuration */
          const float *a, const float *b, float *c, unsigned N)
{
    unsigned gid = hipThreadIdx_x; // <- coordinate index function
    if (gid < N) {
        c[gid] = a[gid] + PlusOne(b[gid]);
    }
}
void callMyKernel()
{
    float *a, *b, *c; // initialization not shown...
    unsigned N = 1000000;
    const unsigned blockSize = 256;

    hipLaunchKernel(MyKernel, dim3(N/blockSize), dim3(blockSize), 0, 0,  a,b,c,N);
}
```

## Variable-Type Qualifiers

### `__constant__`
The `__constant__` keyword is supported. The host writes constant memory before launching the kernel; from the GPU, this memory is read-only during kernel execution. The functions for accessing constant memory (hipGetSymbolAddress(), hipGetSymbolSize(), hipMemcpyToSymbol(), hipMemcpyToSymbolAsync, hipMemcpyFromSymbol, hipMemcpyFromSymbolAsync) are under development.

### `__shared__` 
The `__shared__` keyword is supported.

`extern __shared__` allows the host to dynamically allocate shared memory and is specified as a launch parameter.  HIP uses an alternate syntax based on the HIP_DYNAMIC_SHARED macro.

### `__managed__`
Managed memory, including the `__managed__` keyword, are not supported in HIP.

### `__restrict__`
The `__restrict__` keyword tells the compiler that the associated memory pointer will not alias with any other pointer in the kernel or function.  This feature can help the compiler generate better code. In most cases, all pointer arguments must use this keyword to realize the benefit. 


## Built-In Variables

### Coordinate Built-Ins
These built-ins determine the coordinate of the active work item in the execution grid. They are defined in hip_runtime.h (rather than being implicitly defined by the compiler).   

| **HIP Syntax** | **Cuda Syntax** |
| --- | --- |
| hipThreadIdx_x | threadIdx.x |
| hipThreadIdx_y | threadIdx.y |
| hipThreadIdx_z | threadIdx.z |
|                |             |
| hipBlockIdx_x  | blockIdx.x  |
| hipBlockIdx_y  | blockIdx.y  |
| hipBlockIdx_z  | blockIdx.z  |
|                |             |
| hipBlockDim_x  | blockDim.x  |
| hipBlockDim_y  | blockDim.y  |
| hipBlockDim_z  | blockDim.z  |
|                |             |
| hipGridDim_x   | gridDim.x   |
| hipGridDim_y   | gridDim.y   |
| hipGridDim_z   | gridDim.z   |

### warpSize
The warpSize variable is of type int and contains the warp size (in threads) for the target device. Note that all current Nvidia devices return 32 for this variable, and all current AMD devices return 64. Device code should use the warpSize built-in to develop portable wave-aware code.


## Vector Types

Note that these types are defined in hip_runtime.h and are not automatically provided by the compiler. 


### Short Vector Types
Short vector types derive from the basic integer and floating-point types. They are structures defined in hip_vector_types.h. The first, second, third and fourth components of the vector are accessible through the ```x```, ```y```, ```z``` and ```w``` fields, respectively. All the short vector types support a constructor function of the form ```make_<type_name>()```. For example, ```float4 make_float4(float x, float y, float z, float w)``` creates a vector of type ```float4``` and value ```(x,y,z,w)```.

HIP supports the following short vector formats:
- Signed Integers:
    - char1, char2, char3, char4
    - short1, short2, short3, short4
    - int1, int2, int3, int4
    - long1, long2, long3, long4
    - longlong1, longlong2, longlong3, longlong4
- Unsigned Integers:
    - uchar1, uchar2, uchar3, uchar4
    - ushort1, ushort2, ushort3, ushort4
    - uint1, uint2, uint3, uint4
    - ulong1, ulong2, ulong3, ulong4
    - ulonglong1, ulonglong2, ulonglong3, ulonglong4
- Floating Points
    - float1, float2, float3, float4
    - double1, double2, double3, double4

### dim3
dim3 is a three-dimensional integer vector type commonly used to specify grid and group dimensions. Unspecified dimensions are initialized to 1. 
```
typedef struct dim3 {
  uint32_t x; 
  uint32_t y; 
  uint32_t z; 

  dim3(uint32_t _x=1, uint32_t _y=1, uint32_t _z=1) : x(_x), y(_y), z(_z) {};
};

```

## Memory-Fence Instructions
HIP supports __threadfence() and  __threadfence_block().

HIP provides workaround for threadfence_system() under HCC path.
To enable the workaround, HIP should be built with environment variable HIP_COHERENT_HOST_ALLOC enabled.
In addition,the kernels that use __threadfence_system() should be modified as follows:
- The kernel should only operate on finegrained system memory; which should be allocated with hipHostMalloc().
- Remove all memcpy for those allocated finegrained system memory regions.

## Synchronization Functions
The __syncthreads() built-in function is supported in HIP. The __syncthreads_count(int), __syncthreads_and(int) and __syncthreads_or(int) functions are under development.  

## Math Functions
hcc supports a set of math operations callable from the device.

### Single Precision Mathematical Functions
Following is the list of supported single precision mathematical functions.

| **Function** | **Supported on Host** | **Supported on Device** |
| --- | --- | --- |
| float acosf ( float  x ) <br><sub>Calculate the arc cosine of the input argument.</sub> | ✓ | ✓ |
| float acoshf ( float  x ) <br><sub>Calculate the nonnegative arc hyperbolic cosine of the input argument.</sub> | ✓ | ✓ |
| float asinf ( float  x ) <br><sub>Calculate the arc sine of the input argument.</sub> | ✓ | ✓ |
| float asinhf ( float  x ) <br><sub>Calculate the arc hyperbolic sine of the input argument.</sub> | ✓ | ✓ |
| float atan2f ( float  y, float  x ) <br><sub>Calculate the arc tangent of the ratio of first and second input arguments.</sub> | ✓ | ✓ |
| float atanf ( float  x ) <br><sub>Calculate the arc tangent of the input argument.</sub> | ✓ | ✓ |
| float atanhf ( float  x ) <br><sub>Calculate the arc hyperbolic tangent of the input argument.</sub> | ✓ | ✓ |
| float cbrtf ( float  x ) <br><sub>Calculate the cube root of the input argument.</sub> | ✓ | ✓ |
| float ceilf ( float  x ) <br><sub>Calculate ceiling of the input argument.</sub> | ✓ | ✓ |
| float copysignf ( float  x, float  y ) <br><sub>Create value with given magnitude, copying sign of second value.</sub> | ✓ | ✓ |
| float cosf ( float  x ) <br><sub>Calculate the cosine of the input argument.</sub> | ✓ | ✓ |
| float coshf ( float  x ) <br><sub>Calculate the hyperbolic cosine of the input argument.</sub> | ✓ | ✓ |
| float erfcf ( float  x ) <br><sub>Calculate the complementary error function of the input argument.</sub> | ✓ | ✓ |
| float erff ( float  x ) <br><sub>Calculate the error function of the input argument.</sub> | ✓ | ✓ |
| float exp10f ( float  x ) <br><sub>Calculate the base 10 exponential of the input argument.</sub> | ✓ | ✓ |
| float exp2f ( float  x ) <br><sub>Calculate the base 2 exponential of the input argument.</sub> | ✓ | ✓ |
| float expf ( float  x ) <br><sub>Calculate the base e exponential of the input argument.</sub> | ✓ | ✓ |
| float expm1f ( float  x ) <br><sub>Calculate the base e exponential of the input argument, minus 1.</sub> | ✓ | ✓ |
| float fabsf ( float  x ) <br><sub>Calculate the absolute value of its argument.</sub> | ✓ | ✓ |
| float fdimf ( float  x, float  y ) <br><sub>Compute the positive difference between `x` and `y`.</sub> | ✓ | ✓ |
| float floorf ( float  x ) <br><sub>Calculate the largest integer less than or equal to `x`.</sub> | ✓ | ✓ |
| float fmaf ( float  x, float  y, float  z ) <br><sub>Compute `x × y + z` as a single operation.</sub> | ✓ | ✓ |
| float fmaxf ( float  x, float  y ) <br><sub>Determine the maximum numeric value of the arguments.</sub> | ✓ | ✓ |
| float fminf ( float  x, float  y ) <br><sub>Determine the minimum numeric value of the arguments.</sub> | ✓ | ✓ |
| float fmodf ( float  x, float  y ) <br><sub>Calculate the floating-point remainder of `x / y`.</sub> | ✓ | ✓ |
| float frexpf ( float  x, int* nptr ) <br><sub>Extract mantissa and exponent of a floating-point value.</sub> | ✓ | ✗ |
| float hypotf ( float  x, float  y ) <br><sub>Calculate the square root of the sum of squares of two arguments.</sub> | ✓ | ✓ |
| int ilogbf ( float  x ) <br><sub>Compute the unbiased integer exponent of the argument.</sub> | ✓ | ✓ |
| __RETURN_TYPE<sup id="a1">[1](#f1)</sup> isfinite ( float  a ) <br><sub>Determine whether argument is finite.</sub> | ✓ | ✓ |
| __RETURN_TYPE<sup>[1](#f1)</sup> isinf ( float  a ) <br><sub>Determine whether argument is infinite.</sub> | ✓ | ✓ |
| __RETURN_TYPE<sup>[1](#f1)</sup> isnan ( float  a ) <br><sub>Determine whether argument is a NaN.</sub> | ✓ | ✓ |
| float ldexpf ( float  x, int  exp ) <br><sub>Calculate the value of x ⋅ 2<sup>exp</sup>.</sub> | ✓ | ✓ |
| float log10f ( float  x ) <br><sub>Calculate the base 10 logarithm of the input argument.</sub> | ✓ | ✓ |
| float log1pf ( float  x ) <br><sub>Calculate the value of log<sub>e</sub>( 1 + x ).</sub> | ✓ | ✓ |
| float logbf ( float  x ) <br><sub>Calculate the floating point representation of the exponent of the input argument.</sub> | ✓ | ✓ |
| float log2f ( float  x ) <br><sub>Calculate the base 2 logarithm of the input argument.</sub> | ✓ | ✓ | 
| float logf ( float  x ) <br><sub>Calculate the natural logarithm of the input argument.</sub> | ✓ | ✓ |
| float modff ( float  x, float* iptr ) <br><sub>Break down the input argument into fractional and integral parts.</sub> | ✓ | ✗ |
| float nanf ( const char* tagp ) <br><sub>Returns "Not a Number" value.</sub> | ✗ | ✓ |
| float nearbyintf ( float  x ) <br><sub>Round the input argument to the nearest integer.</sub> | ✓ | ✓ |
| float powf ( float  x, float  y ) <br><sub>Calculate the value of first argument to the power of second argument.</sub> | ✓ | ✓ |
| float remainderf ( float  x, float  y ) <br><sub>Compute single-precision floating-point remainder.</sub> | ✓ | ✓ |
| float remquof ( float  x, float  y, int* quo ) <br><sub>Compute single-precision floating-point remainder and part of quotient.</sub> | ✓ | ✗ |
| float roundf ( float  x ) <br><sub>Round to nearest integer value in floating-point.</sub> | ✓ | ✓ |
| float scalbnf ( float  x, int  n ) <br><sub>Scale floating-point input by integer power of two.</sub> | ✓ | ✓ |
| __RETURN_TYPE<sup>[1](#f1)</sup> signbit ( float  a ) <br><sub>Return the sign bit of the input.</sub> | ✓ | ✓ |
| void sincosf ( float  x, float* sptr, float* cptr ) <br><sub>Calculate the sine and cosine of the first input argument.</sub> | ✓ | ✗ |
| float sinf ( float  x ) <br><sub>Calculate the sine of the input argument.</sub> | ✓ | ✓ |
| float sinhf ( float  x ) <br><sub>Calculate the hyperbolic sine of the input argument.</sub> | ✓ | ✓ |
| float sqrtf ( float  x ) <br><sub>Calculate the square root of the input argument.</sub> | ✓ | ✓ |
| float tanf ( float  x ) <br><sub>Calculate the tangent of the input argument.</sub> | ✓ | ✓ |
| float tanhf ( float  x ) <br><sub>Calculate the hyperbolic tangent of the input argument.</sub> | ✓ | ✓ |
| float truncf ( float  x ) <br><sub>Truncate input argument to the integral part.</sub> | ✓ | ✓ |
| float tgammaf ( float  x ) <br><sub>Calculate the gamma function of the input argument.</sub> | ✓ | ✓ |
| float erfcinvf ( float  y ) <br><sub>Calculate the inverse complementary function of the input argument.</sub> | ✓ | ✓ |
| float erfcxf ( float  x ) <br><sub>Calculate the scaled complementary error function of the input argument.</sub> | ✓ | ✓ |
| float erfinvf ( float  y ) <br><sub>Calculate the inverse error function of the input argument.</sub> | ✓ | ✓ |
| float fdividef ( float x, float  y ) <br><sub>Divide two floating point values.</sub> | ✓ | ✓ |
| float frexpf ( float  x, int *nptr ) <br><sub>Extract mantissa and exponent of a floating-point value.</sub> | ✓ | ✓ |
| float j0f ( float  x ) <br><sub>Calculate the value of the Bessel function of the first kind of order 0 for the input argument.</sub> | ✓ | ✓ |
| float j1f ( float  x ) <br><sub>Calculate the value of the Bessel function of the first kind of order 1 for the input argument.</sub> | ✓ | ✓ |
| float jnf ( int n, float  x ) <br><sub>Calculate the value of the Bessel function of the first kind of order n for the input argument.</sub> | ✓ | ✓ |
| float lgammaf ( float  x ) <br><sub>Calculate the natural logarithm of the absolute value of the gamma function of the input argument.</sub> | ✓ | ✓ |
| long long int llrintf ( float  x ) <br><sub>Round input to nearest integer value.</sub> | ✓ | ✓ |
| long long int llroundf ( float  x ) <br><sub>Round to nearest integer value.</sub> | ✓ | ✓ |
| long int lrintf ( float  x ) <br><sub>Round input to nearest integer value.</sub> | ✓ | ✓ |
| long int lroundf ( float  x ) <br><sub>Round to nearest integer value.</sub> | ✓ | ✓ |
| float modff ( float  x, float *iptr ) <br><sub>Break down the input argument into fractional and integral parts.</sub> | ✓ | ✓ |
| float nextafterf ( float  x, float y ) <br><sub>Returns next representable single-precision floating-point value after argument.</sub> | ✓ | ✓ |
| float norm3df ( float  a, float b, float c ) <br><sub>Calculate the square root of the sum of squares of three coordinates of the argument.</sub> | ✓ | ✓ |
| float norm4df ( float  a, float b, float c, float d ) <br><sub>Calculate the square root of the sum of squares of four coordinates of the argument.</sub> | ✓ | ✓ |
| float normcdff ( float  y ) <br><sub>Calculate the standard normal cumulative distribution function.</sub> | ✓ | ✓ |
| float normcdfinvf ( float  y ) <br><sub>Calculate the inverse of the standard normal cumulative distribution function.</sub> | ✓ | ✓ |
| float normf ( int dim, const float *a ) <br><sub>Calculate the square root of the sum of squares of any number of coordinates.</sub> | ✓ | ✓ |
| float rcbrtf ( float x ) <br><sub>Calculate the reciprocal cube root function.</sub> | ✓ | ✓ |
| float remquof ( float x, float y, int *quo ) <br><sub>Compute single-precision floating-point remainder and part of quotient.</sub> | ✓ | ✓ |
| float rhypotf ( float x, float y ) <br><sub>Calculate one over the square root of the sum of squares of two arguments.</sub> | ✓ | ✓ |
| float rintf ( float x ) <br><sub>Round input to nearest integer value in floating-point.</sub> | ✓ | ✓ |
| float rnorm3df ( float  a, float b, float c ) <br><sub>Calculate one over the square root of the sum of squares of three coordinates of the argument.</sub> | ✓ | ✓ |
| float rnorm4df ( float  a, float b, float c, float d ) <br><sub>Calculate one over the square root of the sum of squares of four coordinates of the argument.</sub> | ✓ | ✓ |
| float rnormf ( int dim, const float *a ) <br><sub>Calculate the reciprocal of square root of the sum of squares of any number of coordinates.</sub> | ✓ | ✓ |
| float scalblnf ( float x, long int n ) <br><sub>Scale floating-point input by integer power of two.</sub> | ✓ | ✓ |
| void sincosf ( float x, float *sptr, float *cptr ) <br><sub>Calculate the sine and cosine of the first input argument.</sub> | ✓ | ✓ |
| void sincospif ( float x, float *sptr, float *cptr ) <br><sub>Calculate the sine and cosine of the first input argument multiplied by PI.</sub> | ✓ | ✓ |
| float y0f ( float  x ) <br><sub>Calculate the value of the Bessel function of the second kind of order 0 for the input argument.</sub> | ✓ | ✓ |
| float y1f ( float  x ) <br><sub>Calculate the value of the Bessel function of the second kind of order 1 for the input argument.</sub> | ✓ | ✓ |
| float ynf ( int n, float  x ) <br><sub>Calculate the value of the Bessel function of the second kind of order n for the input argument.</sub> | ✓ | ✓ |



<sub><b id="f1"><sup>[1]</sup></b> __RETURN_TYPE is dependent on compiler. It is usually 'int' for C compilers and 'bool' for C++ compilers.</sub> [↩](#a1)

### Double Precision Mathematical Functions
Following is the list of supported double precision mathematical functions.

| **Function** | **Supported on Host** | **Supported on Device** |
| --- | --- | --- |
| double acos ( double  x ) <br><sub>Calculate the arc cosine of the input argument.</sub> | ✓ | ✓ |
| double acosh ( double  x ) <br><sub>Calculate the nonnegative arc hyperbolic cosine of the input argument.</sub> | ✓ | ✓ |
| double asin ( double  x ) <br><sub>Calculate the arc sine of the input argument.</sub> | ✓ | ✓ |
| double asinh ( double  x ) <br><sub> Calculate the arc hyperbolic sine of the input argument.</sub> | ✓ | ✓ |
| double atan ( double  x ) <br><sub>Calculate the arc tangent of the input argument.</sub> | ✓ | ✓ |
| double atan2 ( double  y, double  x ) <br><sub>Calculate the arc tangent of the ratio of first and second input arguments.</sub> | ✓ | ✓ |
| double atanh ( double  x ) <br><sub>Calculate the arc hyperbolic tangent of the input argument.</sub> | ✓ | ✓ |
| double cbrt ( double  x ) <br><sub>Calculate the cube root of the input argument.</sub> | ✓ | ✓ |
| double ceil ( double  x ) <br><sub>Calculate ceiling of the input argument.</sub> | ✓ | ✓ |
| double copysign ( double  x, double  y ) <br><sub>Create value with given magnitude, copying sign of second value.</sub> | ✓ | ✓ |
| double cos ( double  x ) <br><sub>Calculate the cosine of the input argument.</sub> | ✓ | ✓ |
| double cosh ( double  x ) <br><sub>Calculate the hyperbolic cosine of the input argument.</sub> | ✓ | ✓ |
| double erf ( double  x ) <br><sub>Calculate the error function of the input argument.</sub> | ✓ | ✓ |
| double erfc ( double  x ) <br><sub>Calculate the complementary error function of the input argument.</sub> | ✓ | ✓ |
| double exp ( double  x ) <br><sub>Calculate the base e exponential of the input argument.</sub> | ✓ | ✓ |
| double exp10 ( double  x ) <br><sub>Calculate the base 10 exponential of the input argument.</sub> | ✓ | ✓ |
| double exp2 ( double  x ) <br><sub>Calculate the base 2 exponential of the input argument.</sub> | ✓ | ✓ |
| double expm1 ( double  x ) <br><sub>Calculate the base e exponential of the input argument, minus 1.</sub> | ✓ | ✓ |
| double fabs ( double  x ) <br><sub>Calculate the absolute value of the input argument.</sub> | ✓ | ✓ |
| double fdim ( double  x, double  y ) <br><sub>Compute the positive difference between `x` and `y`.</sub> | ✓ | ✓ | 
| double floor ( double  x ) <br><sub>Calculate the largest integer less than or equal to `x`.</sub> | ✓ | ✓ |
| double fma ( double  x, double  y, double  z ) <br><sub>Compute `x × y + z` as a single operation.</sub> | ✓ | ✓ |
| double fmax ( double , double ) <br><sub>Determine the maximum numeric value of the arguments.</sub> | ✓ | ✓ |
| double fmin ( double  x, double  y ) <br><sub>Determine the minimum numeric value of the arguments.</sub> | ✓ | ✓ |
| double fmod ( double  x, double  y ) <br><sub>Calculate the floating-point remainder of `x / y`.</sub> | ✓ | ✓ |
| double frexp ( double  x, int* nptr ) <br><sub>Extract mantissa and exponent of a floating-point value.</sub> | ✓ | ✗ |
| double hypot ( double  x, double  y ) <br><sub>Calculate the square root of the sum of squares of two arguments.</sub> | ✓ | ✓ |
| int ilogb ( double  x ) <br><sub>Compute the unbiased integer exponent of the argument.</sub> | ✓ | ✓ |
| __RETURN_TYPE<sup id="a2">[1](#f2)</sup> isfinite ( double  a ) <br><sub>Determine whether argument is finite.</sub> | ✓ | ✓ |
| __RETURN_TYPE<sup>[1](#f2)</sup> isinf ( double  a ) <br><sub>Determine whether argument is infinite.</sub> | ✓ | ✓ |
| __RETURN_TYPE<sup>[1](#f2)</sup> isnan ( double  a ) <br><sub>Determine whether argument is a NaN.</sub> | ✓ | ✓ |
| double ldexp ( double  x, int  exp ) <br><sub>Calculate the value of x ⋅ 2<sup>exp</sup>.</sub> | ✓ | ✓ |
| double log ( double  x ) <br><sub>Calculate the base e logarithm of the input argument.</sub> | ✓ | ✓ |
| double log10 ( double  x ) <br><sub>Calculate the base 10 logarithm of the input argument.</sub> | ✓ | ✓ |
| double log1p ( double  x ) <br><sub>Calculate the value of log<sub>e</sub>( 1 + x ).</sub> | ✓ | ✓ |
| double log2 ( double  x ) <br><sub>Calculate the base 2 logarithm of the input argument.</sub> | ✓ | ✓ |
| double logb ( double  x ) <br><sub>Calculate the floating point representation of the exponent of the input argument.</sub> | ✓ | ✓ |
| double modf ( double  x, double* iptr ) <br><sub>Break down the input argument into fractional and integral parts.</sub> | ✓ | ✗ |
| double nan ( const char* tagp ) <br><sub>Returns "Not a Number" value.</sub> | ✗ | ✓ |
| double nearbyint ( double  x ) <br><sub>Round the input argument to the nearest integer.</sub> | ✓ | ✓ |
| double pow ( double  x, double  y ) <br><sub>Calculate the value of first argument to the power of second argument.</sub> | ✓ | ✓ |
| double remainder ( double  x, double  y ) <br><sub>Compute double-precision floating-point remainder.</sub> | ✓ | ✓ |
| double remquo ( double  x, double  y, int* quo ) <br><sub>Compute double-precision floating-point remainder and part of quotient.</sub> | ✓ | ✗ |
| double round ( double  x ) <br><sub>Round to nearest integer value in floating-point.</sub> | ✓ | ✓ |
| double scalbn ( double  x, int  n ) <br><sub>Scale floating-point input by integer power of two.</sub> | ✓ | ✓ |
| __RETURN_TYPE<sup>[1](#f2)</sup> signbit ( double  a ) <br><sub>Return the sign bit of the input.</sub> | ✓ | ✓ |
| double sin ( double  x ) <br><sub>Calculate the sine of the input argument.</sub> | ✓ | ✓ |
| void sincos ( double  x, double* sptr, double* cptr ) <br><sub>Calculate the sine and cosine of the first input argument.</sub> | ✓ | ✗ |
| double sinh ( double  x ) <br><sub>Calculate the hyperbolic sine of the input argument.</sub> | ✓ | ✓ |
| double sqrt ( double  x ) <br><sub>Calculate the square root of the input argument.</sub> | ✓ | ✓ |
| double tan ( double  x ) <br><sub>Calculate the tangent of the input argument.</sub> | ✓ | ✓ |
| double tanh ( double  x ) <br><sub>Calculate the hyperbolic tangent of the input argument.</sub> | ✓ | ✓ |
| double tgamma ( double  x ) <br><sub>Calculate the gamma function of the input argument.</sub> | ✓ | ✓ |
| double trunc ( double  x ) <br><sub>Truncate input argument to the integral part.</sub> | ✓ | ✓ |
| double erfcinv ( double  y ) <br><sub>Calculate the inverse complementary function of the input argument.</sub> | ✓ | ✓ |
| double erfcx ( double  x ) <br><sub>Calculate the scaled complementary error function of the input argument.</sub> | ✓ | ✓ |
| double erfinv ( double  y ) <br><sub>Calculate the inverse error function of the input argument.</sub> | ✓ | ✓ |
| double frexp ( float  x, int *nptr ) <br><sub>Extract mantissa and exponent of a floating-point value.</sub> | ✓ | ✓ |
| double j0 ( double  x ) <br><sub>Calculate the value of the Bessel function of the first kind of order 0 for the input argument.</sub> | ✓ | ✓ |
| double j1 ( double  x ) <br><sub>Calculate the value of the Bessel function of the first kind of order 1 for the input argument.</sub> | ✓ | ✓ |
| double jn ( int n, double  x ) <br><sub>Calculate the value of the Bessel function of the first kind of order n for the input argument.</sub> | ✓ | ✓ |
| double lgamma ( double  x ) <br><sub>Calculate the natural logarithm of the absolute value of the gamma function of the input argument.</sub> | ✓ | ✓ |
| long long int llrint ( double  x ) <br><sub>Round input to nearest integer value.</sub> | ✓ | ✓ |
| long long int llround ( double  x ) <br><sub>Round to nearest integer value.</sub> | ✓ | ✓ |
| long int lrint ( double  x ) <br><sub>Round input to nearest integer value.</sub> | ✓ | ✓ |
| long int lround ( double  x ) <br><sub>Round to nearest integer value.</sub> | ✓ | ✓ |
| double modf ( double  x, double *iptr ) <br><sub>Break down the input argument into fractional and integral parts.</sub> | ✓ | ✓ |
| double nextafter ( double  x, double y ) <br><sub>Returns next representable single-precision floating-point value after argument.</sub> | ✓ | ✓ |
| double norm3d ( double  a, double b, double c ) <br><sub>Calculate the square root of the sum of squares of three coordinates of the argument.</sub> | ✓ | ✓ |
| float norm4d ( double  a, double b, double c, double d ) <br><sub>Calculate the square root of the sum of squares of four coordinates of the argument.</sub> | ✓ | ✓ |
| double normcdf ( double  y ) <br><sub>Calculate the standard normal cumulative distribution function.</sub> | ✓ | ✓ |
| double normcdfinv ( double  y ) <br><sub>Calculate the inverse of the standard normal cumulative distribution function.</sub> | ✓ | ✓ |
| double rcbrt ( double x ) <br><sub>Calculate the reciprocal cube root function.</sub> | ✓ | ✓ |
| double remquo ( double x, double y, int *quo ) <br><sub>Compute single-precision floating-point remainder and part of quotient.</sub> | ✓ | ✓ |
| double rhypot ( double x, double y ) <br><sub>Calculate one over the square root of the sum of squares of two arguments.</sub> | ✓ | ✓ |
| double rint ( double x ) <br><sub>Round input to nearest integer value in floating-point.</sub> | ✓ | ✓ |
| double rnorm3d ( double a, double b, double c ) <br><sub>Calculate one over the square root of the sum of squares of three coordinates of the argument.</sub> | ✓ | ✓ |
| double rnorm4d ( double a, double b, double c, double d ) <br><sub>Calculate one over the square root of the sum of squares of four coordinates of the argument.</sub> | ✓ | ✓ |
| double rnorm ( int dim, const double *a ) <br><sub>Calculate the reciprocal of square root of the sum of squares of any number of coordinates.</sub> | ✓ | ✓ |
| double scalbln ( double x, long int n ) <br><sub>Scale floating-point input by integer power of two.</sub> | ✓ | ✓ |
| void sincos ( double x, double *sptr, double *cptr ) <br><sub>Calculate the sine and cosine of the first input argument.</sub> | ✓ | ✓ |
| void sincospi ( double x, double *sptr, double *cptr ) <br><sub>Calculate the sine and cosine of the first input argument multiplied by PI.</sub> | ✓ | ✓ |
| double y0f ( double  x ) <br><sub>Calculate the value of the Bessel function of the second kind of order 0 for the input argument.</sub> | ✓ | ✓ |
| double y1 ( double  x ) <br><sub>Calculate the value of the Bessel function of the second kind of order 1 for the input argument.</sub> | ✓ | ✓ |
| double yn ( int n, double  x ) <br><sub>Calculate the value of the Bessel function of the second kind of order n for the input argument.</sub> | ✓ | ✓ |



<sub><b id="f2"><sup>[1]</sup></b> __RETURN_TYPE is dependent on compiler. It is usually 'int' for C compilers and 'bool' for C++ compilers.</sub> [↩](#a2)

### Integer Intrinsics
Following is the list of supported integer intrinsics. Note that intrinsics are supported on device only.

| **Function** |
| --- |
| unsigned int __brev ( unsigned int x ) <br><sub>Reverse the bit order of a 32 bit unsigned integer.</sub> |
| unsigned long long int __brevll ( unsigned long long int x ) <br><sub>Reverse the bit order of a 64 bit unsigned integer. </sub> |
| int __clz ( int  x ) <br><sub>Return the number of consecutive high-order zero bits in a 32 bit integer.</sub> |
| unsigned int __clz(unsigned int x) <br><sub>Return the number of consecutive high-order zero bits in 32 bit unsigned integer.</sub> |
| int __clzll ( long long int x ) <br><sub>Count the number of consecutive high-order zero bits in a 64 bit integer.</sub> |
| unsigned int __clzll(long long int x) <br><sub>Return the number of consecutive high-order zero bits in 64 bit signed integer.</sub> |
| unsigned int __ffs(unsigned int x) <br><sub>Find the position of least signigicant bit set to 1 in a 32 bit unsigned integer.<sup id="a3">[1](#f3)</sup></sub> |
| unsigned int __ffs(int x) <br><sub>Find the position of least signigicant bit set to 1 in a 32 bit signed integer.</sub> |
| unsigned int __ffsll(unsigned long long int x) <br><sub>Find the position of least signigicant bit set to 1 in a 64 bit unsigned integer.<sup>[1](#f3)</sup></sub> |
| unsigned int __ffsll(long long int x) <br><sub>Find the position of least signigicant bit set to 1 in a 64 bit signed integer.</sub> |
| unsigned int __popc ( unsigned int x ) <br><sub>Count the number of bits that are set to 1 in a 32 bit integer.</sub> |
| int __popcll ( unsigned long long int x )<br><sub>Count the number of bits that are set to 1 in a 64 bit integer.</sub> |
| int __mul24 ( int x, int y )<br><sub>Multiply two 24bit integers.</sub> |
| unsigned int __umul24 ( unsigned int x, unsigned int y )<br><sub>Multiply two 24bit unsigned integers.</sub> |
<sub><b id="f3"><sup>[1]</sup></b> 
The hcc implementation of __ffs() and __ffsll() contains code to add a constant +1 to produce the ffs result format.
For the cases where this overhead is not acceptable and programmer is willing to specialize for the platform, 
hcc provides hc::__lastbit_u32_u32(unsigned int input) and  hc::__lastbit_u32_u64(unsigned long long int input).
The index returned by __lastbit_ instructions starts at -1, while for ffs the index starts at 0.

### Floating-point Intrinsics
Following is the list of supported floating-point intrinsics. Note that intrinsics are supported on device only.

| **Function** |
| --- |
| float __cosf ( float  x ) <br><sub>Calculate the fast approximate cosine of the input argument.</sub> |
| float __expf ( float  x ) <br><sub>Calculate the fast approximate base e exponential of the input argument.</sub> |
| float __frsqrt_rn ( float  x ) <br><sub>Compute `1 / √x` in round-to-nearest-even mode.</sub> |
| float __fsqrt_rd ( float  x ) <br><sub>Compute `√x` in round-down mode.</sub> |
| float __fsqrt_rn ( float  x ) <br><sub>Compute `√x` in round-to-nearest-even mode.</sub> |
| float __fsqrt_ru ( float  x ) <br><sub>Compute `√x` in round-up mode.</sub> |
| float __fsqrt_rz ( float  x ) <br><sub>Compute `√x` in round-towards-zero mode.</sub> |
| float __log10f ( float  x ) <br><sub>Calculate the fast approximate base 10 logarithm of the input argument.</sub> |
| float __log2f ( float  x ) <br><sub>Calculate the fast approximate base 2 logarithm of the input argument.</sub> |
| float __logf ( float  x ) <br><sub>Calculate the fast approximate base e logarithm of the input argument.</sub> |
| float __powf ( float  x, float  y ) <br><sub>Calculate the fast approximate of x<sup>y</sup>.</sub> |
| float __sinf ( float  x ) <br><sub>Calculate the fast approximate sine of the input argument.</sub> |
| float __tanf ( float  x ) <br><sub>Calculate the fast approximate tangent of the input argument.</sub> |
| double __dsqrt_rd ( double  x ) <br><sub>Compute `√x` in round-down mode.</sub> |
| double __dsqrt_rn ( double  x ) <br><sub>Compute `√x` in round-to-nearest-even mode.</sub> |
| double __dsqrt_ru ( double  x ) <br><sub>Compute `√x` in round-up mode.</sub> |
| double __dsqrt_rz ( double  x ) <br><sub>Compute `√x` in round-towards-zero mode.</sub> |

## Texture Functions
Texture functions are not supported.

## Surface Functions
Surface functions are not supported.

## Timer Functions
HIP provides the following built-in functions for reading a high-resolution timer from the device.
```
clock_t clock()
long long int clock64()
```
Returns the value of counter that is incremented every clock cycle on device. Difference in values returned provides the cycles used.

## Atomic Functions

Atomic functions execute as read-modify-write operations residing in global or shared memory. No other device or thread can observe or modify the memory location during an atomic operation. If multiple instructions from different devices or threads target the same memory location, the 
instructions are serialized in an undefined order.  

HIP supports the following atomic operations.

| **Function** | **Supported in HIP** | **Supported in CUDA** |
| --- | --- | --- |
| int atomicAdd(int* address, int val) | ✓ | ✓ |
| unsigned int atomicAdd(unsigned int* address,unsigned int val) | ✓ | ✓ |
| unsigned long long int atomicAdd(unsigned long long int* address,unsigned long long int val) | ✓ | ✓ |
| float atomicAdd(float* address, float val) | ✓ | ✓ |
| int atomicSub(int* address, int val) | ✓ | ✓ |
| unsigned int atomicSub(unsigned int* address,unsigned int val) | ✓ | ✓ |
| int atomicExch(int* address, int val) | ✓ | ✓ |
| unsigned int atomicExch(unsigned int* address,unsigned int val) | ✓ | ✓ |
| unsigned long long int atomicExch(unsigned long long int* address,unsigned long long int val) | ✓ | ✓ |
| float atomicExch(float* address, float val) | ✓ | ✓ |
| int atomicMin(int* address, int val) | ✓ | ✓ |
| unsigned int atomicMin(unsigned int* address,unsigned int val) | ✓ | ✓ |
| unsigned long long int atomicMin(unsigned long long int* address,unsigned long long int val) | ✓ | ✓ |
| int atomicMax(int* address, int val) | ✓ | ✓ |
| unsigned int atomicMax(unsigned int* address,unsigned int val) | ✓ | ✓ |
| unsigned long long int atomicMax(unsigned long long int* address,unsigned long long int val) | ✓ | ✓ |
| unsigned int atomicInc(unsigned int* address)| ✗ | ✓  |
| unsigned int atomicDec(unsigned int* address)| ✗ | ✓ |
| int atomicCAS(int* address, int compare, int val) | ✓ | ✓ |
| unsigned int atomicCAS(unsigned int* address,unsigned int compare,unsigned int val) | ✓ | ✓ |
| unsigned long long int atomicCAS(unsigned long long int* address,unsigned long long int compare,unsigned long long int val) | ✓ | ✓ |
| int atomicAnd(int* address, int val) | ✓ | ✓ |
| unsigned int atomicAnd(unsigned int* address,unsigned int val) | ✓ | ✓ |
| unsigned long long int atomicAnd(unsigned long long int* address,unsigned long long int val) | ✓ | ✓ |
| int atomicOr(int* address, int val) | ✓ | ✓ |
| unsigned int atomicOr(unsigned int* address,unsigned int val) | ✓ | ✓ |
| unsigned long long int atomicOr(unsigned long long int* address,unsigned long long int val) | ✓ | ✓ |
| int atomicXor(int* address, int val) | ✓ | ✓ |
| unsigned int atomicXor(unsigned int* address,unsigned int val) | ✓ | ✓ |
| unsigned long long int atomicXor(unsigned long long int* address,unsigned long long int val)) | ✓ | ✓ |

### Caveats and Features Under-Development:

- HIP enables atomic operations on 32-bit integers. Additionally, it supports an atomic float add. AMD hardware, however, implements the float add using a CAS loop, so this function may not perform efficiently.

## Warp Cross-Lane Functions

Warp cross-lane functions operate across all lanes in a warp. The hardware guarantees that all warp lanes will execute in lockstep, so additional synchronization is unnecessary, and the instructions use no shared memory.

Note that Nvidia and AMD devices have different warp sizes, so portable code should use the warpSize built-ins to query the warp size. Hipified code from the Cuda path requires careful review to ensure it doesn’t assume a waveSize of 32. "Wave-aware" code that assumes a waveSize of 32 will run on a wave-64 machine, but it will utilize only half of the machine resources. In addition to the warpSize device function, host code can obtain the warpSize from the device properties:

```
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceID);
    int w = props.warpSize;  
    // implement portable algorithm based on w (rather than assume 32 or 64)
```


### Warp Vote and Ballot Functions

```
int __all(int predicate)
int __any(int predicate)
uint64_t __ballot(int predicate)
```

Threads in a warp are referred to as *lanes* and are numbered from 0 to warpSize -- 1. For these functions, each warp lane contributes 1 -- the bit value (the predicate), which is efficiently broadcast to all lanes in the warp. The 32-bit int predicate from each lane reduces to a 1-bit value: 0 (predicate = 0) or 1 (predicate != 0). `__any` and `__all` provide a summary view of the predicates that the other warp lanes contribute:

- `__any()` returns 1 if any warp lane contributes a nonzero predicate, or 0 otherwise
- `__all()` returns 1 if all other warp lanes contribute nonzero predicates, or 0 otherwise

Applications can test whether the target platform supports the any/all instruction using the `hasWarpVote` device property or the HIP_ARCH_HAS_WARP_VOTE compiler define.

`__ballot` provides a bit mask containing the 1-bit predicate value from each lane. The nth bit of the result contains the 1 bit contributed by the nth warp lane. Note that HIP's `__ballot` function supports a 64-bit return value (compared with Cuda’s 32 bits). Code ported from Cuda should support the larger warp sizes that the HIP version of this instruction supports. Applications can test whether the target platform supports the ballot instruction using the `hasWarpBallot` device property or the HIP_ARCH_HAS_WARP_BALLOT compiler define.


### Warp Shuffle Functions 

Half-float shuffles are not supported. The default width is warpSize---see [Warp Cross-Lane Functions](#warp-cross-lane-functions). Applications should not assume the warpSize is 32 or 64.

```
int   __shfl      (int var,   int srcLane, int width=warpSize);
float __shfl      (float var, int srcLane, int width=warpSize);
int   __shfl_up   (int var,   unsigned int delta, int width=warpSize);
float __shfl_up   (float var, unsigned int delta, int width=warpSize);
int   __shfl_down (int var,   unsigned int delta, int width=warpSize);
float __shfl_down (float var, unsigned int delta, int width=warpSize) ;
int   __shfl_xor  (int var,   int laneMask, int width=warpSize) 
float __shfl_xor  (float var, int laneMask, int width=warpSize);

```

## Profiler Counter Function

The Cuda `__prof_trigger()` instruction is not supported.

## Assert

The assert function is under development.
HIP does support an "abort" call which will terminate the process execution from inside the kernel.

## Printf

The printf function is under development.

## Device-Side Dynamic Global Memory Allocation

Device-side dynamic global memory allocation is under development.  HIP now includes a preliminary
implementation of malloc and free that can be called from device functions.

## `__launch_bounds__`


GPU multiprocessors have a fixed pool of resources (primarily registers and shared memory) which are shared by the actively running warps. Using more resources can increase IPC of the kernel but reduces the resources available for other warps and limits the number of warps that can be simulaneously running. Thus GPUs have a complex relationship between resource usage and performance.  

__hip_launch_bounds__ allows the application to provide usage hints that influence the resources (primarily registers) used by the generated code.
__hip_launch_bounds__ is a function attribute that must be attached to a __global__ function:

```
__global__ void `__launch_bounds__`(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EU) MyKernel(...) ...
MyKernel(hipGridLaunch lp, ...) 
...
```

__launch_bounds__ supports two parameters:
- MAX_THREADS_PER_BLOCK - The programmers guarantees that kernel will be launched with threads less than MAX_THREADS_PER_BLOCK. (On NVCC this maps to the .maxntid PTX directive). If no launch_bounds is specified, MAX_THREADS_PER_BLOCK is the maximum block size supported by the device (typically 1024 or larger). Specifying MAX_THREADS_PER_BLOCK less than the maximum effectively allows the compiler to use more resources than a default unconstrained compilation that supports all possible block sizes at launch time.
The threads-per-block is the product of (hipBlockDim_x * hipBlockDim_y * hipBlockDim_z).
- MIN_WARPS_PER_EU - directs the compiler to minimize resource usage so that the requested number of warps can be simultaneously active on a multi-processor. Since active warps compete for the same fixed pool of resources, the compiler must reduce resources required by each warp(primarily registers). MIN_WARPS_PER_EU is optional and defaults to 1 if not specified. Specifying a MIN_WARPS_PER_EU greater than the default 1 effectively constrains the compiler's resource usage.

### Compiler Impact
The compiler uses these parameters as follows:
- The compiler uses the hints only to manage register usage, and does not automatically reduce shared memory or other resources.
- Compilation fails if compiler cannot generate a kernel which meets the requirements of the specified launch bounds.
- From MAX_THREADS_PER_BLOCK, the compiler derives the maximum number of warps/block that can be used at launch time.
Values of MAX_THREADS_PER_BLOCK less than the default allows the compiler to use a larger pool of registers : each warp uses registers, and this hint constains the launch to a warps/block size which is less than maximum.
- From MIN_WARPS_PER_EU, the compiler derives a maximum number of registers that can be used by the kernel (to meet the required #simultaneous active blocks).
If MIN_WARPS_PER_EU is 1, then the kernel can use all registers supported by the multiprocessor.
- The compiler ensures that the registers used in the kernel is less than both allowed maximums, typically by spilling registers (to shared or global memory), or by using more instructions.
- The compiler may use hueristics to increase register usage, or may simply be able to avoid spilling. The MAX_THREADS_PER_BLOCK is particularly useful in this cases, since it allows the compiler to use more registers and avoid situations where the compiler constrains the register usage (potentially spilling) to meet the requirements of a large block size that is never used at launch time.


### CU and EU Definitions
A compute unit (CU) is responsible for executing the waves of a work-group. It is composed of one or more execution units (EU) which are responsible for executing waves. An EU can have enough resources to maintain the state of more than one executing wave. This allows an EU to hide latency by switching between waves in a similar way to symmetric multithreading on a CPU. In order to allow the state for multiple waves to fit on an EU, the resources used by a single wave have to be limited. Limiting such resources can allow greater latency hiding, but can result in having to spill some register state to memory. This attribute allows an advanced developer to tune the number of waves that are capable of fitting within the resources of an EU. It can be used to ensure at least a certain number will fit to help hide latency, and can also be used to ensure no more than a certain number will fit to limit cache thrashing.
 
### Porting from CUDA __launch_bounds
CUDA defines a __launch_bounds which is also designed to control occupancy:
```
__launch_bounds(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)
```

- The second parameter __launch_bounds parameters must be converted to the format used __hip_launch_bounds, which uses warps and execution-units rather than blocks and multi-processors  ( This conversion is performed automatically by the clang hipify tools.)
```
MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK) / 32
```

The key differences in the interface are:
- Warps (rather than blocks):
The developer is trying to tell the compiler to control resource utilization to guarantee some amount of active Warps/EU for latency hiding.  Specifying active warps in terms of blocks appears to hide the micro-architectural details of the warp size, but makes the interface more confusing since the developer ultimately needs to compute the number of warps to obtain the desired level of control. 
- Execution Units  (rather than multiProcessor):
The use of execution units rather than multiprocessors provides support for architectures with multiple execution units/multi-processor. For example, the AMD GCN architecture has 4 execution units per multiProcessor.  The hipDeviceProps has a field executionUnitsPerMultiprocessor.
Platform-specific coding techniques such as #ifdef can be used to specify different launch_bounds for NVCC and HCC platforms, if desired. 


### maxregcount
Unlike nvcc, hcc does not support the "--maxregcount" option.  Instead, users are encouraged to use the hip_launch_bounds directive since the parameters are more intuitive and portable than
micro-architecture details like registers, and also the directive allows per-kernel control rather than an entire file.  hip_launch_bounds works on both hcc and nvcc targets.


## Register Keyword
The register keyword is deprecated in C++, and is silently ignored by both nvcc and hcc.  To see warnings, you can pass the option `-Wdeprecated-register` to hcc.


## Pragma Unroll

Unroll with a bounds that is known at compile-time is supported.  For example:

```
#pragma unroll 16 /* hint to compiler to unroll next loop by 16 */
for (int i=0; i<16; i++) ...
```

```
#pragma unroll 1  /* tell compiler to never unroll the loop */
for (int i=0; i<16; i++) ...
```


```
#pragma unroll /* hint to compiler to completely unroll next loop. */
for (int i=0; i<16; i++) ...
```


## In-Line Assembly

GCN ISA In-line assembly, is supported. For example:

```
asm volatile ("v_mac_f32_e32 %0, %2, %3" : "=v" (out[i]) : "0"(out[i]), "v" (a), "v" (in[i]));
```

We insert the GCN isa into the kernel using `asm()` Assembler statement.
`volatile` keyword is used so that the optimizers must not change the number of volatile operations or change their order of execution relative to other volatile operations.
`v_mac_f32_e32` is the GCN instruction, for more information please refer - [AMD GCN3 ISA architecture manual](http://gpuopen.com/compute-product/amd-gcn3-isa-architecture-manual/)
Index for the respective operand in the ordered fashion is provided by `%` followed by position in the list of operands
`"v"` is the constraint code (for target-specific AMDGPU) for 32-bit VGPR register, for more info please refer - [Supported Constraint Code List for AMDGPU](https://llvm.org/docs/LangRef.html#supported-constraint-code-list)
Output Constraints are specified by an `"="` prefix as shown above ("=v"). This indicate that assemby will write to this operand, and the operand will then be made available as a return value of the asm expression. Input constraints do not have a prefix - just the constraint code. The constraint string of `"0"` says to use the assigned register for output as an input as well (it being the 0'th constraint).

## C++ Support
The following C++ features are not supported:
- Run-time-type information (RTTI)
- Virtual functions
- Try/catch

## Kernel Compilation
hipcc now supports compiling C++/HIP kernels to binary code objects. 
The user can specify the target for which the binary can be generated. HIP/HCC does not yet support fat binaries so only a single target may be specified.
The file format for binary is `.co` which means Code Object. The following command builds the code object using `hipcc`.

`hipcc --genco --target-isa=[TARGET GPU] [INPUT FILE] -o [OUTPUT FILE]`
```[TARGET GPU] = gfx803/gfx701
[INPUT FILE] = Name of the file containing kernels
[OUTPUT FILE] = Name of the generated code object file```

Note that one important fact to remember when using binary code objects is that the number of arguments to the kernel are different on HCC and NVCC path. Refer to the sample in samples/0_Intro/module_api for differences in the arguments to be passed to the kernel.

