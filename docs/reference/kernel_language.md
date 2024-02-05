# Kernel Language Syntax

HIP provides a C++ syntax that is suitable for compiling most code that commonly appears in compute kernels, including classes, namespaces, operator overloading, templates and more. Additionally, it defines other language features designed specifically to target accelerators, such as the following:
- A kernel-launch syntax that uses standard C++, resembles a function call and is portable to all HIP targets
- Short-vector headers that can serve on a host or a device
- Math functions resembling those in the "math.h" header included with standard C++ compilers
- Built-in functions for accessing specific GPU hardware capabilities

This section describes the built-in variables and functions accessible from the HIP kernel. It is intended for readers familiar with CUDA kernel syntax and wanting to understand how HIP is different from CUDA.

Features are marked with one of the following keywords:
- **Supported**---HIP supports the feature with a Cuda-equivalent function
- **Not supported**---HIP does not support the feature
- **Under development**---the feature is under development but not yet available

## Function-Type Qualifiers
### `__device__`
Supported  `__device__` functions are
  - Executed on the device
  - Called from the device only

The `__device__` keyword can combine with the host keyword (see {ref}`host_attr`).

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

`__host__` can combine with `__device__`, in which case the function compiles for both the host and device. These functions cannot use the HIP grid coordinate functions (for example, "threadIdx.x"). A possible workaround is to pass the necessary coordinate info as an argument to the function.

`__host__` cannot combine with `__global__`.

HIP parses the `__noinline__` and `__forceinline__` keywords and converts them to the appropriate Clang attributes.

## Calling `__global__` Functions

`__global__` functions are often referred to as *kernels,* and calling one is termed *launching the kernel.* These functions require the caller to specify an "execution configuration" that includes the grid and block dimensions. The execution configuration can also include other information for the launch, such as the amount of additional shared memory to allocate and the stream where the kernel should execute. HIP introduces a standard C++ calling convention to pass the execution configuration to the kernel in addition to the Cuda <<< >>> syntax. In HIP,
- Kernels launch with either <<< >>> syntax or the "hipLaunchKernelGGL" function
- The first five parameters to hipLaunchKernelGGL are the following:
   - **symbol kernelName**: the name of the kernel to launch.  To support template kernels which contains "," use the HIP_KERNEL_NAME macro.   The hipify tools insert this automatically.
   - **dim3 gridDim**: 3D-grid dimensions specifying the number of blocks to launch.
   - **dim3 blockDim**: 3D-block dimensions specifying the number of threads in each block.
   - **size_t dynamicShared**: amount of additional shared memory to allocate when launching the kernel (see [__shared__](#__shared__))
   - **hipStream_t**: stream where the kernel should execute. A value of 0 corresponds to the NULL stream (see [Synchronization Functions](#synchronization-functions)).
- Kernel arguments follow these first five parameters


```
// Example pseudo code introducing hipLaunchKernelGGL:
__global__ MyKernel(hipLaunchParm lp, float *A, float *B, float *C, size_t N)
{
...
}

MyKernel<<<dim3(gridDim), dim3(groupDim), 0, 0>>> (a,b,c,n);
// Alternatively, kernel can be launched by
// hipLaunchKernelGGL(MyKernel, dim3(gridDim), dim3(groupDim), 0/*dynamicShared*/, 0/*stream), a, b, c, n);

```

The hipLaunchKernelGGL macro always starts with the five parameters specified above, followed by the kernel arguments. HIPIFY tools optionally convert Cuda launch syntax to hipLaunchKernelGGL, including conversion of optional arguments in <<< >>> to the five required hipLaunchKernelGGL parameters. The dim3 constructor accepts zero to three arguments and will by default initialize unspecified dimensions to 1. See [dim3](#dim3). The kernel uses the coordinate built-ins (thread*, block*, grid*) to determine coordinate index and coordinate bounds of the work item that's currently executing. See [Coordinate Built-Ins](#Coordinate-Built-Ins).

Please note, HIP does not support kernel launch with total work items defined in dimension with size gridDim x blockDim >= 2^32.


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
    unsigned gid = threadIdx.x; // <- coordinate index function
    if (gid < N) {
        c[gid] = a[gid] + PlusOne(b[gid]);
    }
}
void callMyKernel()
{
    float *a, *b, *c; // initialization not shown...
    unsigned N = 1000000;
    const unsigned blockSize = 256;

    MyKernel<<<dim3(gridDim), dim3(groupDim), 0, 0>>> (a,b,c,n);
    // Alternatively, kernel can be launched by
    // hipLaunchKernelGGL(MyKernel, dim3(N/blockSize), dim3(blockSize), 0, 0,  a,b,c,N);
}
```

## Variable-Type Qualifiers

### `__constant__`
The `__constant__` keyword is supported. The host writes constant memory before launching the kernel; from the GPU, this memory is read-only during kernel execution. The functions for accessing constant memory (hipGetSymbolAddress(), hipGetSymbolSize(), hipMemcpyToSymbol(), hipMemcpyToSymbolAsync(), hipMemcpyFromSymbol(), hipMemcpyFromSymbolAsync()) are available.

### `__shared__`
The `__shared__` keyword is supported.

`extern __shared__` allows the host to dynamically allocate shared memory and is specified as a launch parameter.
Previously, it was essential to declare dynamic shared memory using the HIP_DYNAMIC_SHARED macro for accuracy, as using static shared memory in the same kernel could result in overlapping memory ranges and data-races.

Now, the HIP-Clang compiler provides support for extern shared declarations, and the HIP_DYNAMIC_SHARED option is no longer required..

### `__managed__`
Managed memory, including the `__managed__` keyword, are supported in HIP combined host/device compilation.

### `__restrict__`
The `__restrict__` keyword tells the compiler that the associated memory pointer will not alias with any other pointer in the kernel or function.  This feature can help the compiler generate better code. In most cases, all pointer arguments must use this keyword to realize the benefit.


## Built-In Variables

### Coordinate Built-Ins
Built-ins determine the coordinate of the active work item in the execution grid. They are defined in amd_hip_runtime.h (rather than being implicitly defined by the compiler).
In HIP, built-ins coordinate variable definitions are the same as in Cuda, for instance:
threadIdx.x, blockIdx.y, gridDim.y, etc.
The products gridDim.x * blockDim.x, gridDim.y * blockDim.y and gridDim.z * blockDim.z are always less than 2^32.
Coordinates builtins are implemented as structures for better performance. When used with printf, they needs to be casted to integer types explicitly.

### warpSize
The warpSize variable is of type int and contains the warp size (in threads) for the target device. Note that all current Nvidia devices return 32 for this variable, and current AMD devices return 64 for gfx9 and 32 for gfx10 and above. The warpSize variable should only be used in device functions. Device code should use the warpSize built-in to develop portable wave-aware code.


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

HIP provides workaround for threadfence_system() under the HIP-Clang path.
To enable the workaround, HIP should be built with environment variable HIP_COHERENT_HOST_ALLOC enabled.
In addition,the kernels that use __threadfence_system() should be modified as follows:
- The kernel should only operate on finegrained system memory; which should be allocated with hipHostMalloc().
- Remove all memcpy for those allocated finegrained system memory regions.

## Synchronization Functions
The __syncthreads() built-in function is supported in HIP. The __syncthreads_count(int), __syncthreads_and(int) and __syncthreads_or(int) functions are under development.  

## Math Functions
HIP-Clang supports a set of math operations callable from the device.

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
| __RETURN_TYPE[^f1] isfinite ( float  a ) <br><sub>Determine whether argument is finite.</sub> | ✓ | ✓ |
| __RETURN_TYPE[^f1]</sup> isinf ( float  a ) <br><sub>Determine whether argument is infinite.</sub> | ✓ | ✓ |
| __RETURN_TYPE[^f1]</sup> isnan ( float  a ) <br><sub>Determine whether argument is a NaN.</sub> | ✓ | ✓ |
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
| __RETURN_TYPE[^f1]</sup> signbit ( float  a ) <br><sub>Return the sign bit of the input.</sub> | ✓ | ✓ |
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


[^f1]: __RETURN_TYPE is dependent on compiler. It is usually 'int' for C compilers and 'bool' for C++ compilers.

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
| __RETURN_TYPE[^f1] isfinite ( double  a ) <br><sub>Determine whether argument is finite.</sub> | ✓ | ✓ |
| __RETURN_TYPE[^f1]</sup> isinf ( double  a ) <br><sub>Determine whether argument is infinite.</sub> | ✓ | ✓ |
| __RETURN_TYPE[^f1]</sup> isnan ( double  a ) <br><sub>Determine whether argument is a NaN.</sub> | ✓ | ✓ |
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
| __RETURN_TYPE[^f1] signbit ( double  a ) <br><sub>Return the sign bit of the input.</sub> | ✓ | ✓ |
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
| unsigned int __ffs(unsigned int x) <br><sub>Find the position of least signigicant bit set to 1 in a 32 bit unsigned integer.[^f3]</sub> |
| unsigned int __ffs(int x) <br><sub>Find the position of least signigicant bit set to 1 in a 32 bit signed integer.</sub> |
| unsigned int __ffsll(unsigned long long int x) <br><sub>Find the position of least signigicant bit set to 1 in a 64 bit unsigned integer.[^f3]</sup></sub> |
| unsigned int __ffsll(long long int x) <br><sub>Find the position of least signigicant bit set to 1 in a 64 bit signed integer.</sub> |
| unsigned int __popc ( unsigned int x ) <br><sub>Count the number of bits that are set to 1 in a 32 bit integer.</sub> |
| unsigned int __popcll ( unsigned long long int x )<br><sub>Count the number of bits that are set to 1 in a 64 bit integer.</sub> |
| int __mul24 ( int x, int y )<br><sub>Multiply two 24bit integers.</sub> |
| unsigned int __umul24 ( unsigned int x, unsigned int y )<br><sub>Multiply two 24bit unsigned integers.</sub> |
<sub>[^f3]
The HIP-Clang implementation of __ffs() and __ffsll() contains code to add a constant +1 to produce the ffs result format.
For the cases where this overhead is not acceptable and programmer is willing to specialize for the platform,
HIP-Clang provides __lastbit_u32_u32(unsigned int input) and __lastbit_u32_u64(unsigned long long int input).
The index returned by __lastbit_ instructions starts at -1, while for ffs the index starts at 0.

### Floating-point Intrinsics
Following is the list of supported floating-point intrinsics. Note that intrinsics are supported on device only.

| **Function** |
| --- |
| float __cosf ( float  x ) <br><sub>Calculate the fast approximate cosine of the input argument.</sub> |
| float __expf ( float  x ) <br><sub>Calculate the fast approximate base e exponential of the input argument.</sub> |
| float __frsqrt_rn ( float  x ) <br><sub>Compute `1 / √x` in round-to-nearest-even mode.</sub> |
| float __fsqrt_rn ( float  x ) <br><sub>Compute `√x` in round-to-nearest-even mode.</sub> |
| float __log10f ( float  x ) <br><sub>Calculate the fast approximate base 10 logarithm of the input argument.</sub> |
| float __log2f ( float  x ) <br><sub>Calculate the fast approximate base 2 logarithm of the input argument.</sub> |
| float __logf ( float  x ) <br><sub>Calculate the fast approximate base e logarithm of the input argument.</sub> |
| float __powf ( float  x, float  y ) <br><sub>Calculate the fast approximate of x<sup>y</sup>.</sub> |
| float __sinf ( float  x ) <br><sub>Calculate the fast approximate sine of the input argument.</sub> |
| float __tanf ( float  x ) <br><sub>Calculate the fast approximate tangent of the input argument.</sub> |
| double __dsqrt_rn ( double  x ) <br><sub>Compute `√x` in round-to-nearest-even mode.</sub> |

## Texture Functions
The supported Texture functions are listed in header files "texture_fetch_functions.h" and "texture_indirect_functions.h" in [HIP-AMD backend repository](https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail).

Texture functions are not supported on some devices.
Macro __HIP_NO_IMAGE_SUPPORT == 1 can be used to check whether texture functions are not supported in device code.
Attribute hipDeviceAttributeImageSupport can be queried to check whether texture functions are supported in host runtime code.

## Surface Functions
Surface functions are not supported.

## Timer Functions
HIP provides the following built-in functions for reading a high-resolution timer from the device.
```
clock_t clock()
long long int clock64()
```
Returns the value of counter that is incremented every clock cycle on device. Difference in values returned provides the cycles used.

```
long long int wall_clock64()
```
Returns wall clock count at a constant frequency on the device, which can be queried via HIP API with hipDeviceAttributeWallClockRate attribute of the device in HIP application code, for example,
```
int wallClkRate = 0; //in kilohertz
HIPCHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, deviceId));
```
Where hipDeviceAttributeWallClockRate is a device attribute.
Note that, wall clock frequency is a per-device attribute.


## Atomic Functions

Atomic functions execute as read-modify-write operations residing in global or shared memory. No other device or thread can observe or modify the memory location during an atomic operation. If multiple instructions from different devices or threads target the same memory location, the instructions are serialized in an undefined order.

HIP adds new APIs with _system as suffix to support system scope atomic operations. For example, the `atomicAnd` function is meant to be atomic and coherent within the GPU device executing the function. `atomicAnd_system` will allow developers to extend the atomic operation to system scope, from the GPU device to other CPUs and GPU devices in the system.

HIP supports the following atomic operations.

| **Function**                                                                                                         |  **Supported in HIP** |  **Supported in CUDA** |
| -------------------------------------------------------------------------------------------------------------------- | --------------------- | ---------------------- |
| int atomicAdd(int* address, int val)                                                                                 |  ✓                    |  ✓                     |
| int atomicAdd_system(int* address, int val)                                                                          |  ✓                    |  ✓                     |
| unsigned int atomicAdd(unsigned int* address,unsigned int val)                                                       |  ✓                    |  ✓                     |
| unsigned int atomicAdd_system(unsigned int* address, unsigned int val)                                               |  ✓                    |  ✓                     |
| unsigned long long atomicAdd(unsigned long long* address,unsigned long long val)                                     |  ✓                    |  ✓                     |
| unsigned long long atomicAdd_system(unsigned long long* address, unsigned long long val)                             |  ✓                    |  ✓                     |
| float atomicAdd(float* address, float val)                                                                           |  ✓                    |  ✓                     |
| float atomicAdd_system(float* address, float val)                                                                    |  ✓                    |  ✓                     |
| double atomicAdd(double* address, double val)                                                                        |  ✓                    |  ✓                     |
| double atomicAdd_system(double* address, double val)                                                                 |  ✓                    |  ✓                     |
| float unsafeAtomicAdd(float* address, float val)                                                                     |  ✓                    |  ✗                     |
| float safeAtomicAdd(float* address, float val)                                                                       |  ✓                    |  ✗                     |
| double unsafeAtomicAdd(double* address, double val)                                                                  |  ✓                    |  ✗                     |
| double safeAtomicAdd(double* address, double val)                                                                    |  ✓                    |  ✗                     |
| int atomicSub(int* address, int val)                                                                                 |  ✓                    |  ✓                     |
| int atomicSub_system(int* address, int val)                                                                          |  ✓                    |  ✓                     |
| unsigned int atomicSub(unsigned int* address,unsigned int val)                                                       |  ✓                    |  ✓                     |
| unsigned int atomicSub_system(unsigned int* address, unsigned int val)                                               |  ✓                    |  ✓                     |
| int atomicExch(int* address, int val)                                                                                |  ✓                    |  ✓                     |
| int atomicExch_system(int* address, int val)                                                                         |  ✓                    |  ✓                     |
| unsigned int atomicExch(unsigned int* address,unsigned int val)                                                      |  ✓                    |  ✓                     |
| unsigned int atomicExch_system(unsigned int* address, unsigned int val)                                              |  ✓                    |  ✓                     |
| unsigned long long atomicExch(unsigned long long int* address,unsigned long long int val)                            |  ✓                    |  ✓                     |
| unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val)                            |  ✓                    |  ✓                     |
| unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val)                            |  ✓                    |  ✓                     |
| float atomicExch(float* address, float val)                                                                          |  ✓                    |  ✓                     |
| int atomicMin(int* address, int val)                                                                                 |  ✓                    |  ✓                     |
| int atomicMin_system(int* address, int val)                                                                          |  ✓                    |  ✓                     |
| unsigned int atomicMin(unsigned int* address,unsigned int val)                                                       |  ✓                    |  ✓                     |
| unsigned int atomicMin_system(unsigned int* address, unsigned int val)                                               |  ✓                    |  ✓                     |
| unsigned long long atomicMin(unsigned long long* address,unsigned long long val)                                     |  ✓                    |  ✓                     |
| int atomicMax(int* address, int val)                                                                                 |  ✓                    |  ✓                     |
| int atomicMax_system(int* address, int val)                                                                          |  ✓                    |  ✓                     |
| unsigned int atomicMax(unsigned int* address,unsigned int val)                                                       |  ✓                    |  ✓                     |
| unsigned int atomicMax_system(unsigned int* address, unsigned int val)                                               |  ✓                    |  ✓                     |
| unsigned long long atomicMax(unsigned long long* address,unsigned long long val)                                     |  ✓                    |  ✓                     |
| unsigned int atomicInc(unsigned int* address)                                                                        |  ✗                    |  ✓                     |
| unsigned int atomicDec(unsigned int* address)                                                                        |  ✗                    |  ✓                     |
| int atomicCAS(int* address, int compare, int val)                                                                    |  ✓                    |  ✓                     |
| int atomicCAS_system(int* address, int compare, int val)                                                             |  ✓                    |  ✓                     |
| unsigned int atomicCAS(unsigned int* address,unsigned int compare,unsigned int val)                                  |  ✓                    |  ✓                     |
| unsigned int atomicCAS_system(unsigned int* address, unsigned int compare, unsigned int val)                         |  ✓                    |  ✓                     |
| unsigned long long atomicCAS(unsigned long long* address,unsigned long long compare,unsigned long long val)          |  ✓                    |  ✓                     |
| unsigned long long atomicCAS_system(unsigned long long* address, unsigned long long compare, unsigned long long val) |  ✓                    |  ✓                     |
| int atomicAnd(int* address, int val)                                                                                 |  ✓                    |  ✓                     |
| int atomicAnd_system(int* address, int val)                                                                          |  ✓                    |  ✓                     |
| unsigned int atomicAnd(unsigned int* address,unsigned int val)                                                       |  ✓                    |  ✓                     |
| unsigned int atomicAnd_system(unsigned int* address, unsigned int val)                                               |  ✓                    |  ✓                     |
| unsigned long long atomicAnd(unsigned long long* address,unsigned long long val)                                     |  ✓                    |  ✓                     |
| unsigned long long atomicAnd_system(unsigned long long* address, unsigned long long val)                             |  ✓                    |  ✓                     |
| int atomicOr(int* address, int val)                                                                                  |  ✓                    |  ✓                     |
| int atomicOr_system(int* address, int val)                                                                           |  ✓                    |  ✓                     |
| unsigned int atomicOr(unsigned int* address,unsigned int val)                                                        |  ✓                    |  ✓                     |
| unsigned int atomicOr_system(unsigned int* address, unsigned int val)                                                |  ✓                    |  ✓                     |
| unsigned int atomicOr_system(unsigned int* address, unsigned int val)                                                |  ✓                    |  ✓                     |
| unsigned long long atomicOr(unsigned long long int* address,unsigned long long val)                                  |  ✓                    |  ✓                     |
| unsigned long long atomicOr_system(unsigned long long* address, unsigned long long val)                              |  ✓                    |  ✓                     |
| int atomicXor(int* address, int val)                                                                                 |  ✓                    |  ✓                     |
| int atomicXor_system(int* address, int val)                                                                          |  ✓                    |  ✓                     |
| unsigned int atomicXor(unsigned int* address,unsigned int val)                                                       |  ✓                    |  ✓                     |
| unsigned int atomicXor_system(unsigned int* address, unsigned int val)                                               |  ✓                    |  ✓                     |
| unsigned long long atomicXor(unsigned long long* address,unsigned long long val))                                    |  ✓                    |  ✓                     |
| unsigned long long atomicXor_system(unsigned long long* address, unsigned long long val)                             |  ✓                    |  ✓                     |

### Unsafe Floating-Point Atomic RMW Operations

Some HIP devices support fast atomic read-modify-write (RMW) operations on floating-point values.
For example, `atomicAdd` on single- or double-precision floating-point values may generate a hardware RMW instruction that is faster than emulating the atomic operation using an atomic compare-and-swap (CAS) loop.

On some devices, these fast atomic RMW instructions can produce different results when compared with the same functions implemented with atomic CAS loops.
For example, some devices will produce incorrect answers if a fast atomic floating-point RMW instruction targets fine-grained memory allocations.
As another example, some devices will use different rounding or denormal modes when using fast atomic floating-point RMW instructions.

As such, the HIP-Clang compiler offers a compile-time option for users to choose whether their code will use the fast, potentially unsafe, atomic instructions.
On devices that support these fast, but unsafe, floating-point atomic RMW instructions, the compiler option `-munsafe-fp-atomics` will allow the compiler to generate them when it sees appropriate atomic RMW function calls.
By passing the `-munsafe-fp-atomics` flag to the compiler, the user is indicating that all floating-point atomic function calls are allowed to use an unsafe version if one exists.
For instance, on some devices, this flag indicates to the compiler that that no floating-point `atomicAdd` function targets fine-grained memory.

If the user instead compiles with `-mno-unsafe-fp-atomics`, the user is telling the compiler to never use a floating-point atomic RMW that may not be safe.
The compiler will default to not producing unsafe floating-point atomic RMW instructions, so the `-mno-unsafe-fp-atomics` compilation option is not strictly necessary.
Explicitly passing this flag to the compiler is good practice, however.

Whenever either of the two options described above, `-munsafe-fp-atomics` and `-mno-unsafe-fp-atomics` are passed to the compiler's command line, they are applied globally for that entire compilation.
If only a subset of the atomic RMW function calls could safely use the faster floating-point atomic RMW instructions, the developer would instead need to compile with `-mno-unsafe-fp-atomics` in order to ensure the remaining atomic RMW function calls produce correct results.
Towards this end, HIP has four extra functions to help developers more precisely control which floating-point atomic RMW functions produce unsafe atomic RMW instructions:

- `float unsafeAtomicAdd(float* address, float val)`
- `double unsafeAtomicAdd(double* address, double val)`
   - These functions will always produce fast atomic RMW instructions on devices that have them, even when `-mno-unsafe-fp-atomics` is set

- `float safeAtomicAdd(float* address, float val)`
- `double safeAtomicAdd(double* address, double val)`
   - These functions will always produce safe atomic RMW operations, even when `-munsafe-fp-atomics` is set

(warp_cross_lane_functions)=
## Warp Cross-Lane Functions

Warp cross-lane functions operate across all lanes in a warp. The hardware guarantees that all warp lanes will execute in lockstep, so additional synchronization is unnecessary, and the instructions use no shared memory.

Note that Nvidia and AMD devices have different warp sizes, so portable code should use the warpSize built-ins to query the warp size. Hipified code from the Cuda path requires careful review to ensure it doesn't assume a waveSize of 32. "Wave-aware" code that assumes a waveSize of 32 will run on a wave-64 machine, but it will utilize only half of the machine resources. WarpSize built-ins should only be used in device functions and its value depends on GPU arch. Users should not assume warpSize to be a compile-time constant. Host functions should use hipGetDeviceProperties to get the default warp size of a GPU device:

```
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceID);
    int w = props.warpSize;
    // implement portable algorithm based on w (rather than assume 32 or 64)
```

Note that assembly kernels may be built for a warp size which is different than the default warp size.

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

`__ballot` provides a bit mask containing the 1-bit predicate value from each lane. The nth bit of the result contains the 1 bit contributed by the nth warp lane. Note that HIP's `__ballot` function supports a 64-bit return value (compared with Cuda's 32 bits). Code ported from Cuda should support the larger warp sizes that the HIP version of this instruction supports. Applications can test whether the target platform supports the ballot instruction using the `hasWarpBallot` device property or the HIP_ARCH_HAS_WARP_BALLOT compiler define.


### Warp Shuffle Functions

Half-float shuffles are not supported. The default width is warpSize---see [Warp Cross-Lane Functions](#warp-cross-lane-functions). Applications should not assume the warpSize is 32 or 64.

```
int   __shfl      (int var,   int srcLane, int width=warpSize);
float __shfl      (float var, int srcLane, int width=warpSize);
int   __shfl_up   (int var,   unsigned int delta, int width=warpSize);
float __shfl_up   (float var, unsigned int delta, int width=warpSize);
int   __shfl_down (int var,   unsigned int delta, int width=warpSize);
float __shfl_down (float var, unsigned int delta, int width=warpSize);
int   __shfl_xor  (int var,   int laneMask, int width=warpSize);
float __shfl_xor  (float var, int laneMask, int width=warpSize);

```

## Cooperative Groups Functions

Cooperative groups is a mechanism for forming and communicating between groups of threads at
a granularity different than the block.  This feature was introduced in Cuda 9.

HIP supports the following kernel language cooperative groups types or functions.


| **Function** | **Supported in HIP** | **Supported in CUDA** |
| --- | --- | --- |
| `void thread_group.sync();` | ✓ | ✓ |
| `unsigned thread_group.size();` | ✓ | ✓ |
| `unsigned thread_group.thread_rank()` | ✓ | ✓ |
| `bool thread_group.is_valid();` | ✓ | ✓ |
| `grid_group this_grid()` | ✓ | ✓ |
| `void grid_group.sync()` | ✓ | ✓ |
| `unsigned grid_group.size()` | ✓ | ✓ |
| `unsigned grid_group.thread_rank()` | ✓ | ✓ |
| `bool grid_group.is_valid()` | ✓ | ✓ |
| `multi_grid_group this_multi_grid()` | ✓ | ✓ |
| `void multi_grid_group.sync()` | ✓ | ✓ |
| `unsigned multi_grid_group.size()` | ✓ | ✓ |
| `unsigned multi_grid_group.thread_rank()` | ✓ | ✓ |
| `bool multi_grid_group.is_valid()` | ✓ | ✓ |
| `unsigned multi_grid_group.num_grids()` | ✓ | ✓ |
| `unsigned multi_grid_group.grid_rank()` | ✓ | ✓ |
| `thread_block this_thread_block()` | ✓ | ✓ |
| `multi_grid_group this_multi_grid()` | ✓ | ✓ |
| `void multi_grid_group.sync()` | ✓ | ✓ |
| `void thread_block.sync()` | ✓ | ✓ |
| `unsigned thread_block.size()` | ✓ | ✓ |
| `unsigned thread_block.thread_rank()` | ✓ | ✓ |
| `bool thread_block.is_valid()` | ✓ | ✓ |
| `dim3 thread_block.group_index()` | ✓ | ✓ |
| `dim3 thread_block.thread_index()` | ✓ | ✓ |

## Warp Matrix Functions

Warp matrix functions allow a warp to cooperatively operate on small matrices
whose elements are spread over the lanes in an unspecified manner.  This feature
was introduced in Cuda 9.

HIP does not support any of the kernel language warp matrix
types or functions.

| **Function** | **Supported in HIP** | **Supported in CUDA** |
| --- | --- | --- |
| `void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned lda)` | | ✓ |
| `void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned lda, layout_t layout)` | | ✓ |
| `void store_matrix_sync(T* mptr, fragment<...> &a,  unsigned lda, layout_t layout)` | | ✓ |
| `void fill_fragment(fragment<...> &a, const T &value)` | | ✓ |
| `void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c , bool sat)` | | ✓ |

## Independent Thread Scheduling

The hardware support for independent thread scheduling introduced in certain architectures
supporting Cuda allows threads to progress independently of each other and enables
intra-warp synchronizations that were previously not allowed.

HIP does not support this type of scheduling.

## Profiler Counter Function

The Cuda `__prof_trigger()` instruction is not supported.

## Assert

The assert function is supported in HIP.
Assert function is used for debugging purpose, when the input expression equals to zero, the execution will be stopped.
```
void assert(int input)
```

There are two kinds of implementations for assert functions depending on the use sceneries,
- One is for the host version of assert, which is defined in assert.h,
- Another is the device version of assert, which is implemented in hip/hip_runtime.h.
Users need to include assert.h to use assert. For assert to work in both device and host functions, users need to include "hip/hip_runtime.h".

HIP provides the function abort() which can be used to terminate the application when terminal failures are detected.  It is implemented using the `__builtin_trap()` function.

This function produces the effects of using `asm("trap")` in the CUDA code.

Note, in HIP, the function terminates the entire application, while in CUDA, `asm("trap")`only terminates the dispatch and the application continues to run.

## Printf

Printf function is supported in HIP.
The following is a simple example to print information in the kernel.

```
#include <hip/hip_runtime.h>

__global__ void run_printf() { printf("Hello World\n"); }

int main() {
  run_printf<<<dim3(1), dim3(1), 0, 0>>>();
}
```

## Device-Side Dynamic Global Memory Allocation

Device-side dynamic global memory allocation is under development.  HIP now includes a preliminary
implementation of malloc and free that can be called from device functions.

## `__launch_bounds__`


GPU multiprocessors have a fixed pool of resources (primarily registers and shared memory) which are shared by the actively running warps. Using more resources can increase IPC of the kernel but reduces the resources available for other warps and limits the number of warps that can be simulaneously running. Thus GPUs have a complex relationship between resource usage and performance.

__launch_bounds__ allows the application to provide usage hints that influence the resources (primarily registers) used by the generated code.  It is a function attribute that must be attached to a __global__ function:

```
__global__ void `__launch_bounds__`(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)
MyKernel(hipGridLaunch lp, ...)
...
```

__launch_bounds__ supports two parameters:
- MAX_THREADS_PER_BLOCK - The programmers guarantees that kernel will be launched with threads less than MAX_THREADS_PER_BLOCK. (On NVCC this maps to the .maxntid PTX directive). If no launch_bounds is specified, MAX_THREADS_PER_BLOCK is the maximum block size supported by the device (typically 1024 or larger). Specifying MAX_THREADS_PER_BLOCK less than the maximum effectively allows the compiler to use more resources than a default unconstrained compilation that supports all possible block sizes at launch time.
The threads-per-block is the product of (blockDim.x * blockDim.y * blockDim.z).
- MIN_WARPS_PER_EXECUTION_UNIT - directs the compiler to minimize resource usage so that the requested number of warps can be simultaneously active on a multi-processor. Since active warps compete for the same fixed pool of resources, the compiler must reduce resources required by each warp(primarily registers). MIN_WARPS_PER_EXECUTION_UNIT is optional and defaults to 1 if not specified. Specifying a MIN_WARPS_PER_EXECUTION_UNIT greater than the default 1 effectively constrains the compiler's resource usage.

When launch kernel with HIP APIs, for example, hipModuleLaunchKernel(), HIP will do validation to make sure input kernel dimension size is not larger than specified launch_bounds.
In case exceeded, HIP would return launch failure, if AMD_LOG_LEVEL is set with proper value (for details, please refer to docs/markdown/hip_logging.md), detail information will be shown in the error log message, including
launch parameters of kernel dim size, launch bounds, and the name of the faulting kernel. It's helpful to figure out which is the faulting kernel, besides, the kernel dim size and launch bounds values will also assist in debugging such failures.

### Compiler Impact
The compiler uses these parameters as follows:
- The compiler uses the hints only to manage register usage, and does not automatically reduce shared memory or other resources.
- Compilation fails if compiler cannot generate a kernel which meets the requirements of the specified launch bounds.
- From MAX_THREADS_PER_BLOCK, the compiler derives the maximum number of warps/block that can be used at launch time.
Values of MAX_THREADS_PER_BLOCK less than the default allows the compiler to use a larger pool of registers : each warp uses registers, and this hint constains the launch to a warps/block size which is less than maximum.
- From MIN_WARPS_PER_EXECUTION_UNIT, the compiler derives a maximum number of registers that can be used by the kernel (to meet the required #simultaneous active blocks).
If MIN_WARPS_PER_EXECUTION_UNIT is 1, then the kernel can use all registers supported by the multiprocessor.
- The compiler ensures that the registers used in the kernel is less than both allowed maximums, typically by spilling registers (to shared or global memory), or by using more instructions.
- The compiler may use hueristics to increase register usage, or may simply be able to avoid spilling. The MAX_THREADS_PER_BLOCK is particularly useful in this cases, since it allows the compiler to use more registers and avoid situations where the compiler constrains the register usage (potentially spilling) to meet the requirements of a large block size that is never used at launch time.


### CU and EU Definitions
A compute unit (CU) is responsible for executing the waves of a work-group. It is composed of one or more execution units (EU) which are responsible for executing waves. An EU can have enough resources to maintain the state of more than one executing wave. This allows an EU to hide latency by switching between waves in a similar way to symmetric multithreading on a CPU. In order to allow the state for multiple waves to fit on an EU, the resources used by a single wave have to be limited. Limiting such resources can allow greater latency hiding, but can result in having to spill some register state to memory. This attribute allows an advanced developer to tune the number of waves that are capable of fitting within the resources of an EU. It can be used to ensure at least a certain number will fit to help hide latency, and can also be used to ensure no more than a certain number will fit to limit cache thrashing.

### Porting from CUDA __launch_bounds
CUDA defines a __launch_bounds which is also designed to control occupancy:
```
__launch_bounds(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)
```

- The second parameter __launch_bounds parameters must be converted to the format used __hip_launch_bounds, which uses warps and execution-units rather than blocks and multi-processors (this conversion is performed automatically by hipify tools).
```
MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK) / 32
```

The key differences in the interface are:
- Warps (rather than blocks):
The developer is trying to tell the compiler to control resource utilization to guarantee some amount of active Warps/EU for latency hiding.  Specifying active warps in terms of blocks appears to hide the micro-architectural details of the warp size, but makes the interface more confusing since the developer ultimately needs to compute the number of warps to obtain the desired level of control.
- Execution Units  (rather than multiProcessor):
The use of execution units rather than multiprocessors provides support for architectures with multiple execution units/multi-processor. For example, the AMD GCN architecture has 4 execution units per multiProcessor.  The hipDeviceProps has a field executionUnitsPerMultiprocessor.
Platform-specific coding techniques such as #ifdef can be used to specify different launch_bounds for NVCC and HIP-Clang platforms, if desired.


### maxregcount
Unlike nvcc, HIP-Clang does not support the "--maxregcount" option.  Instead, users are encouraged to use the hip_launch_bounds directive since the parameters are more intuitive and portable than
micro-architecture details like registers, and also the directive allows per-kernel control rather than an entire file.  hip_launch_bounds works on both HIP-Clang and nvcc targets.


## Register Keyword
The register keyword is deprecated in C++, and is silently ignored by both nvcc and HIP-Clang.  You can pass the option `-Wdeprecated-register` the compiler warning message.

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
- Try/catch
- Virtual functions
Virtual functions are not supported if objects containing virtual function tables are passed between GPU's of different offload arch's, e.g. between gfx906 and gfx1030. Otherwise virtual functions are supported.

## Kernel Compilation
hipcc now supports compiling C++/HIP kernels to binary code objects.
The file format for binary is `.co` which means Code Object. The following command builds the code object using `hipcc`.

`hipcc --genco --offload-arch=[TARGET GPU] [INPUT FILE] -o [OUTPUT FILE]`

```
[TARGET GPU] = GPU architecture
[INPUT FILE] = Name of the file containing kernels
[OUTPUT FILE] = Name of the generated code object file
```

Note: When using binary code objects is that the number of arguments to the kernel is different on HIP-Clang and NVCC path. Refer to the sample in samples/0_Intro/module_api for differences in the arguments to be passed to the kernel.

## gfx-arch-specific-kernel
Clang defined '__gfx*__' macros can be used to execute gfx arch specific codes inside the kernel. Refer to the sample 14_gpu_arch in samples/2_Cookbook.
