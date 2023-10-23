.. meta::
   :description: This chapter describes the built-in variables and functions that are accessible from the
   HIP kernel. It's intended for users who are familiar with CUDA kernel syntax and want to learn how
   HIP differs from CUDA.
   :keywords: AMD, ROCm, HIP, CUDA, kernel language syntax, HIP functions

********************************************************************************
Kernel language syntax
********************************************************************************

HIP provides a C++ syntax that is suitable for compiling most code that commonly appears in
compute kernels (classes, namespaces, operator overloading, and templates). HIP also defines other
language features that are designed to target accelerators, such as:

* A kernel-launch syntax that uses standard C++ (this resembles a function call and is portable to all
  HIP targets)
* Short-vector headers that can serve on a host or device
* Math functions that resemble those in ``math.h``, which is included with standard C++ compilers
* Built-in functions for accessing specific GPU hardware capabilities

.. note::

  This chapter describes the built-in variables and functions that are accessible from the HIP kernel. It's
  intended for users who are familiar with CUDA kernel syntax and want to learn how HIP differs from
  CUDA.

Features are labeled with one of the following keywords:

* **Supported**: HIP supports the feature with a CUDA-equivalent function
* **Not supported**: HIP does not support the feature
* **Under development**: The feature is under development and not yet available

 Function-type qualifiers
========================================================

 ``__device__``
-----------------------------------------------------------------------

Supported  ``__device__`` functions are:

  * Run on the device
  * Called from the device only

You can combine ``__device__`` with the host keyword (:ref:`host_attr`).

``__global__``
-----------------------------------------------------------------------

Supported ``__global__`` functions are:

  * Run on the device
  * Called (launched) from the host

HIP ``__global__`` functions must have a ``void`` return type. The first parameter in a HIP ``__global__``
function must have the type ``hipLaunchParm``. Refer to :ref:`kernel-launch-example` to see usage.

HIP doesn't support dynamic-parallelism, which means that you can't call ``__global__ `` functions from
the device.

.. _host_attr:

``__host__``
-----------------------------------------------------------------------

Supported ``__host__`` functions are:

  * Run on the host
  * Called from the host

You can combine ``__host__`` with ``__device__``; in this case, the function compiles for the host and the
device. Note that these functions can't use the HIP grid coordinate functions (e.g., ``threadIdx.x``). If
you need to use HIP grid coordinate functions, you can pass the necessary coordinate information as
an argument.

You can't combine ``__host__`` with ``__global__``.

HIP parses the ``__noinline__`` and ``__forceinline__`` keywords and converts them into the appropriate
Clang attributes.

Calling ``__global__`` functions
=============================================================

`__global__` functions are often referred to as *kernels*. When you call a global function, you're
*launching a kernel*. When launching a kernel, you must specify a run configuration that includes the
grid and block dimensions. The run configuration can also include other information for the launch,
such as the amount of additional shared memory to allocate and the stream where you want to run the
kernel.

HIP introduces a standard C++ calling convention (``hipLaunchKernelGGL``) to pass the run
configuration to the kernel. However, you can also use the CUDA ``<<< >>>`` syntax.

When using ``hipLaunchKernelGGL``, your first five parameters must be:

  * **symbol kernelName**: The name of the kernel you want to launch. To support template kernels
    that contain ``","``, use the ``HIP_KERNEL_NAM`` macro (HIPIFY tools insert this automatically).
  * **dim3 gridDim**: 3D-grid dimensions that specify the number of blocks to launch.
  * **dim3 blockDim**: 3D-block dimensions that specify the number of threads in each block.
  * **size_t dynamicShared**: The amount of additional shared memory that you want to allocate
    when launching the kernel (see :ref:`shared-variable-type`).
  * **hipStream_t**: The stream where you want to run the kernel. A value of ``0`` corresponds to the
    NULL stream (see :ref:`synchronization-functions`).

You can include your kernel arguments after these parameters.

.. code:: cpp

  // Example hipLaunchKernelGGL pseudocode:
  __global__ MyKernel(hipLaunchParm lp, float *A, float *B, float *C, size_t N)
  {
  ...
  }

  MyKernel<<<dim3(gridDim), dim3(groupDim), 0, 0>>> (a,b,c,n);

  // Alternatively, you can launch the kernel using:
  // hipLaunchKernelGGL(MyKernel, dim3(gridDim), dim3(groupDim), 0/*dynamicShared*/, 0/*stream), a, b, c, n);

You can use HIPIFY tools to convert CUDA launch syntax to ``hipLaunchKernelGGL``. This includes the
conversion of optional ``<<< >>>`` arguments into the five required ``hipLaunchKernelGGL``
parameters.

.. note::

  HIP doesn't support dimension sizes of *gridDim x blockDim >= 2^32* when launching a kernel.

.. kernel-launch-example:

Kernel launch example
==========================================================

.. code:: cpp
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

Variable type qualifiers
========================================================

``__constant__``
-----------------------------------------------------------------------------

The host writes constant memory before launching the kernel. This memory is read-only from the GPU
while the kernel is running. The functions for accessing constant memory are:

* ``hipGetSymbolAddress()``
* ``hipGetSymbolSize()``
* ``hipMemcpyToSymbol()``
* ``hipMemcpyToSymbolAsync()``
* ``hipMemcpyFromSymbol()``
* ``hipMemcpyFromSymbolAsync()``

.. _shared-variable-type:

``__shared__``
-----------------------------------------------------------------------------

To allow the host to dynamically allocate shared memory, you can specify ``extern __shared__`` as a
launch parameter.

.. note::
  Prior to the HIP-Clang compiler, dynamic shared memory had to be declared using the
  ``HIP_DYNAMIC_SHARED`` macro in order to ensure accuracy. This is because using static shared
  memory in the same kernel could've resulted in overlapping memory ranges and data-races. The
  HIP-Clang compiler provides support for ``extern`` shared declarations, so ``HIP_DYNAMIC_SHARED``
  is no longer required.

``__managed__``
-----------------------------------------------------------------------------

Managed memory, including the `__managed__` keyword, is supported in HIP combined host/device
compilation.

``__restrict__``
-----------------------------------------------------------------------------

``__restrict__`` tells the compiler that the associated memory pointer not to alias with any other pointer
in the kernel or function. This can help the compiler generate better code. In most use cases, every
pointer argument should use this keyword in order to achieve the benefit.

Built-in variables
====================================================

Coordinate built-ins
-----------------------------------------------------------------------------

The kernel uses coordinate built-ins (``thread*``, ``block*``, ``grid*``) to determine the coordinate index
and bounds for the active work item.

Built-ins are defined in ``amd_hip_runtime.h``, rather than being implicitly defined by the compiler.

Coordinate variable definitions for built-ins are the same for HIP and CUDA. For example: ``threadIdx.x``,
``blockIdx.y``, and ``gridDim.y``. The products ``gridDim.x * blockDim.x``, ``gridDim.y * blockDim.y``, and
``gridDim.z * blockDim.z`` are always less than ``2^32``.

Coordinate built-ins are implemented as structures for improved performance. When used with
``printf``, they must be explicitly cast to integer types.

warpSize
-----------------------------------------------------------------------------
The ``warpSize`` variable type is ``int``. It contains the warp size (in threads) for the target device.
``warpSize`` should only be used in device functions that develop portable wave-aware code.

.. note::
  NVIDIA devices return 32 for this variable; AMD devices return 64 for gfx9 and 32 for gfx10
  and above.

Vector types
====================================================

The following vector types are defined in ``hip_runtime.h``. They are not automatically provided by the
compiler.

Short vector types
--------------------------------------------------------------------------------------------

Short vector types derive from basic integer and floating-point types. These structures are defined in
``hip_vector_types.h``. The first, second, third, and fourth components of the vector are defined by the
``x``, ``y``, ``z``, and ``w`` fields, respectively. All short vector types support a constructor function of the
form ``make_<type_name>()``. For example, ``float4 make_float4(float x, float y, float z, float w)`` creates
a vector with type ``float4`` and value ``(x,y,z,w)``.

HIP supports the following short vector formats:

* Signed Integers:

  * char1, char2, char3, char4
  * short1, short2, short3, short4
  * int1, int2, int3, int4
  * long1, long2, long3, long4
  * longlong1, longlong2, longlong3, longlong4

* Unsigned Integers:

  * uchar1, uchar2, uchar3, uchar4
  * ushort1, ushort2, ushort3, ushort4
  * uint1, uint2, uint3, uint4
  * ulong1, ulong2, ulong3, ulong4
  * ulonglong1, ulonglong2, ulonglong3, ulonglong4

* Floating Points:

  * float1, float2, float3, float4
  * double1, double2, double3, double4

.. _dim3:

dim3
--------------------------------------------------------------------------------------------

dim3 is a three-dimensional integer vector type that is commonly used to specify grid and group
dimensions.

The dim3 constructor accepts between zero and three arguments. By default, it initializes unspecified
dimensions to 1.

.. code:: cpp

  typedef struct dim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    dim3(uint32_t _x=1, uint32_t _y=1, uint32_t _z=1) : x(_x), y(_y), z(_z) {};
  };


Memory fence instructions
====================================================

HIP supports ``__threadfence()`` and ``__threadfence_block()``. If you're using ``threadfence_system()`` in
the HIP-Clang path, you can use the following workaround:

#. Build HIP with the ``HIP_COHERENT_HOST_ALLOC`` environment variable enabled.
#. Modify kernels that use ``__threadfence_system()`` as follows:
  * Ensure the kernel operates only on fine-grained system memory, which should be allocated with
    ``hipHostMalloc()``.
  * Remove ``memcpy`` for all allocated fine-grained system memory regions.

.. _synchronization-functions:

Synchronization functions
====================================================
The ``__syncthreads()`` built-in function is supported in HIP. The ``__syncthreads_count(int)``,
``__syncthreads_and(int)``, and ``__syncthreads_or(int)`` functions are under development.

Math functions
====================================================
HIP-Clang supports a set of math operations that are callable from the device. These are described in
the following sections.

Single precision mathematical functions
----------------------------------------------------------------------------------------------------------------

The following table describes the supported single-precision mathematical functions.

.. list-table::
    * - **Function**
    - **Supported on host**
    - **Supported on device**

    * -  float acosf ( float  x )
        | <sub>Calculate the arc cosine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float acoshf ( float  x )
        | <sub>Calculate the nonnegative arc hyperbolic cosine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float asinf ( float  x )
        | <sub>Calculate the arc sine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float asinhf ( float  x )
        | <sub>Calculate the arc hyperbolic sine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float atan2f ( float  y, float  x )
        | <sub>Calculate the arc tangent of the ratio of first and second input arguments.</sub>
      - &#10003;
      - &#10003;

    * - float atanf ( float  x )
        | <sub>Calculate the arc tangent of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float atanhf ( float  x )
        | <sub>Calculate the arc hyperbolic tangent of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float cbrtf ( float  x )
        | <sub>Calculate the cube root of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float ceilf ( float  x )
        | <sub>Calculate ceiling of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float copysignf ( float  x, float  y )
        | <sub>Create value with given magnitude, copying sign of second value.</sub>
      - &#10003;
      - &#10003;

    * - float cosf ( float  x )
        | <sub>Calculate the cosine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float coshf ( float  x )
        | <sub>Calculate the hyperbolic cosine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float erfcf ( float  x )
        | <sub>Calculate the complementary error function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float erff ( float  x )
        | <sub>Calculate the error function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float exp10f ( float  x )
        | <sub>Calculate the base 10 exponential of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float exp2f ( float  x )
        | <sub>Calculate the base 2 exponential of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float expf ( float  x )
        | <sub>Calculate the base e exponential of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float expm1f ( float  x )
        | <sub>Calculate the base e exponential of the input argument, minus 1.</sub>
      - &#10003;
      - &#10003;

    * - float fabsf ( float  x )
        | <sub>Calculate the absolute value of its argument.</sub>
      - &#10003;
      - &#10003;

    * - float fdimf ( float  x, float  y )
        | <sub>Compute the positive difference between `x` and `y`.</sub>
      - &#10003;
      - &#10003;

    * - float floorf ( float  x )
        | <sub>Calculate the largest integer less than or equal to `x`.</sub>
      - &#10003;
      - &#10003;

    * - float fmaf ( float  x, float  y, float  z )
        | <sub>Compute `x × y + z` as a single operation.</sub>
      - &#10003;
      - &#10003;

    * - float fmaxf ( float  x, float  y )
        | <sub>Determine the maximum numeric value of the arguments.</sub>
      - &#10003;
      - &#10003;

    * - float fminf ( float  x, float  y )
        | <sub>Determine the minimum numeric value of the arguments.</sub>
      - &#10003;
      - &#10003;

    * - float fmodf ( float  x, float  y )
        | <sub>Calculate the floating-point remainder of `x / y`.</sub>
      - &#10003;
      - &#10003;

    * - float frexpf ( float  x, int* nptr )
        | <sub>Extract mantissa and exponent of a floating-point value.</sub>
      - &#10003;
      - &#10007;

    * - float hypotf ( float  x, float  y )
        | <sub>Calculate the square root of the sum of squares of two arguments.</sub>
      - &#10003;
      - &#10003;

    * - int ilogbf ( float  x )
        | <sub>Compute the unbiased integer exponent of the argument.</sub>
      - &#10003;
      - &#10003;

    * - __RETURN_TYPE[^f1] isfinite ( float  a )
        | <sub>Determine whether argument is finite.</sub>
      - &#10003;
      - &#10003;

    * - __RETURN_TYPE[^f1]</sup> isinf ( float  a )
        | <sub>Determine whether argument is infinite.</sub>
      - &#10003;
      - &#10003;

    * - __RETURN_TYPE[^f1]</sup> isnan ( float  a )
        | <sub>Determine whether argument is a NaN.</sub>
      - &#10003;
      - &#10003;

    * - float ldexpf ( float  x, int  exp )
        | <sub>Calculate the value of x ⋅ 2<sup>exp</sup>.</sub>
      - &#10003;
      - &#10003;

    * - float log10f ( float  x )
        | <sub>Calculate the base 10 logarithm of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float log1pf ( float  x )
        | <sub>Calculate the value of log<sub>e</sub>( 1 + x ).</sub>
      - &#10003;
      - &#10003;

    * - float logbf ( float  x )
        | <sub>Calculate the floating point representation of the exponent of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float log2f ( float  x )
        | <sub>Calculate the base 2 logarithm of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float logf ( float  x )
        | <sub>Calculate the natural logarithm of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float modff ( float  x, float* iptr )
        | <sub>Break down the input argument into fractional and integral parts.</sub>
      - &#10003;
      - &#10007;

    * - float nanf ( const char* tagp )
        | <sub>Returns "Not a Number" value.</sub>
      - &#10007;
      - &#10003;

    * - float nearbyintf ( float  x )
        | <sub>Round the input argument to the nearest integer.</sub>
      - &#10003;
      - &#10003;

    * - float powf ( float  x, float  y )
        | <sub>Calculate the value of first argument to the power of second argument.</sub>
      - &#10003;
      - &#10003;

    * - float remainderf ( float  x, float  y )
        | <sub>Compute single-precision floating-point remainder.</sub>
      - &#10003;
      - &#10003;

    * - float remquof ( float  x, float  y, int* quo )
        | <sub>Compute single-precision floating-point remainder and part of quotient.</sub>
      - &#10003;
      - &#10007;

    * - float roundf ( float  x )
        | <sub>Round to nearest integer value in floating-point.</sub>
      - &#10003;
      - &#10003;

    * - float scalbnf ( float  x, int  n )
        | <sub>Scale floating-point input by integer power of two.</sub>
      - &#10003;
      - &#10003;

    * - __RETURN_TYPE[^f1]</sup> signbit ( float  a )
        | <sub>Return the sign bit of the input.</sub>
      - &#10003;
      - &#10003;

    * - void sincosf ( float  x, float* sptr, float* cptr )
        | <sub>Calculate the sine and cosine of the first input argument.</sub>
      - &#10003;
      - &#10007;

    * - float sinf ( float  x )
        | <sub>Calculate the sine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float sinhf ( float  x )
        | <sub>Calculate the hyperbolic sine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float sqrtf ( float  x )
        | <sub>Calculate the square root of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float tanf ( float  x )
        | <sub>Calculate the tangent of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float tanhf ( float  x )
        | <sub>Calculate the hyperbolic tangent of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float truncf ( float  x )
        | <sub>Truncate input argument to the integral part.</sub>
      - &#10003;
      - &#10003;

    * - float tgammaf ( float  x )
        | <sub>Calculate the gamma function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float erfcinvf ( float  y )
        | <sub>Calculate the inverse complementary function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float erfcxf ( float  x )
        | <sub>Calculate the scaled complementary error function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float erfinvf ( float  y )
        | <sub>Calculate the inverse error function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float fdividef ( float x, float  y )
        | <sub>Divide two floating point values.</sub>
      - &#10003;
      - &#10003;

    * - float frexpf ( float  x, int \*nptr )
        | <sub>Extract mantissa and exponent of a floating-point value.</sub>
      - &#10003;
      - &#10003;

    * - float j0f ( float  x )
        | <sub>Calculate the value of the Bessel function of the first kind of order 0 for the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float j1f ( float  x )
        | <sub>Calculate the value of the Bessel function of the first kind of order 1 for the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float jnf ( int n, float  x )
        | <sub>Calculate the value of the Bessel function of the first kind of order n for the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float lgammaf ( float  x )
        | <sub>Calculate the natural logarithm of the absolute value of the gamma function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - long long int llrintf ( float  x )
        | <sub>Round input to nearest integer value.</sub>
      - &#10003;
      - &#10003;

    * - long long int llroundf ( float  x )
        | <sub>Round to nearest integer value.</sub>
      - &#10003;
      - &#10003;

    * - long int lrintf ( float  x )
        | <sub>Round input to nearest integer value.</sub>
      - &#10003;
      - &#10003;

    * - long int lroundf ( float  x )
        | <sub>Round to nearest integer value.</sub>
      - &#10003;
      - &#10003;

    * - float modff ( float  x, float \*iptr )
        | <sub>Break down the input argument into fractional and integral parts.</sub>
      - &#10003;
      - &#10003;

    * - float nextafterf ( float  x, float y )
        | <sub>Returns next representable single-precision floating-point value after argument.</sub>
      - &#10003;
      - &#10003;

    * - float norm3df ( float  a, float b, float c )
        | <sub>Calculate the square root of the sum of squares of three coordinates of the argument.</sub>
      - &#10003;
      - &#10003;

    * - float norm4df ( float  a, float b, float c, float d )
        | <sub>Calculate the square root of the sum of squares of four coordinates of the argument.</sub>
      - &#10003;
      - &#10003;

    * - float normcdff ( float  y )
        | <sub>Calculate the standard normal cumulative distribution function.</sub>
      - &#10003;
      - &#10003;

    * - float normcdfinvf ( float  y )
        | <sub>Calculate the inverse of the standard normal cumulative distribution function.</sub>
      - &#10003;
      - &#10003;

    * - float normf ( int dim, const float \*a )
        | <sub>Calculate the square root of the sum of squares of any number of coordinates.</sub>
      - &#10003;
      - &#10003;

    * - float rcbrtf ( float x )
        | <sub>Calculate the reciprocal cube root function.</sub>
      - &#10003;
      - &#10003;

    * - float remquof ( float x, float y, int \*quo )
        | <sub>Compute single-precision floating-point remainder and part of quotient.</sub>
      - &#10003;
      - &#10003;

    * - float rhypotf ( float x, float y )
        | <sub>Calculate one over the square root of the sum of squares of two arguments.</sub>
      - &#10003;
      - &#10003;

    * - float rintf ( float x )
        | <sub>Round input to nearest integer value in floating-point.</sub>
      - &#10003;
      - &#10003;

    * - float rnorm3df ( float  a, float b, float c )
        | <sub>Calculate one over the square root of the sum of squares of three coordinates of the argument.</sub>
      - &#10003;
      - &#10003;

    * - float rnorm4df ( float  a, float b, float c, float d )
        | <sub>Calculate one over the square root of the sum of squares of four coordinates of the argument.</sub>
      - &#10003;
      - &#10003;

    * - float rnormf ( int dim, const float \*a )
        | <sub>Calculate the reciprocal of square root of the sum of squares of any number of coordinates.</sub>
      - &#10003;
      - &#10003;

    * - float scalblnf ( float x, long int n )
        | <sub>Scale floating-point input by integer power of two.</sub>
      - &#10003;
      - &#10003;

    * - void sincosf ( float x, float *sptr, float *cptr )
        | <sub>Calculate the sine and cosine of the first input argument.</sub>
      - &#10003;
      - &#10003;

    * - void sincospif ( float x, float *sptr, float *cptr )
        | <sub>Calculate the sine and cosine of the first input argument multiplied by PI.</sub>
      - &#10003;
      - &#10003;

    * - float y0f ( float  x )
        | <sub>Calculate the value of the Bessel function of the second kind of order 0 for the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float y1f ( float  x )
        | <sub>Calculate the value of the Bessel function of the second kind of order 1 for the input argument.</sub>
      - &#10003;
      - &#10003;

    * - float ynf ( int n, float  x )
        | <sub>Calculate the value of the Bessel function of the second kind of order n for the input argument.</sub>
      - &#10003;
      - &#10003;

.. note::
  ``[^f1]: __RETURN_TYPE`` is dependent on the compiler. It is usually ``int`` for C compilers and ``bool``
  for C++ compilers.

Double precision mathematical functions
----------------------------------------------------------------------------------------------------------------

The following table describes the supported double-precision mathematical functions.

.. list-table::
    * - **Function**
    - **Supported on host**
    - **Supported on device**

    * - double acos ( double  x )
        | <sub>Calculate the arc cosine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double acosh ( double  x )
        | <sub>Calculate the nonnegative arc hyperbolic cosine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double asin ( double  x )
        | <sub>Calculate the arc sine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double asinh ( double  x )
        | <sub>Calculate the arc hyperbolic sine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double atan ( double  x )
        | <sub>Calculate the arc tangent of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double atan2 ( double  y, double  x )
        | <sub>Calculate the arc tangent of the ratio of first and second input arguments.</sub>
      - &#10003;
      - &#10003;

    * - double atanh ( double  x )
        | <sub>Calculate the arc hyperbolic tangent of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double cbrt ( double  x )
        | <sub>Calculate the cube root of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double ceil ( double  x )
        | <sub>Calculate ceiling of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double copysign ( double  x, double  y )
        | <sub>Create value with given magnitude, copying sign of second value.</sub>
      - &#10003;
      - &#10003;

    * - double cos ( double  x )
        | <sub>Calculate the cosine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double cosh ( double  x )
        | <sub>Calculate the hyperbolic cosine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double erf ( double  x )
        | <sub>Calculate the error function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double erfc ( double  x )
        | <sub>Calculate the complementary error function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double exp ( double  x )
        | <sub>Calculate the base e exponential of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double exp10 ( double  x )
        | <sub>Calculate the base 10 exponential of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double exp2 ( double  x )
        | <sub>Calculate the base 2 exponential of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double expm1 ( double  x )
        | <sub>Calculate the base e exponential of the input argument, minus 1.</sub>
      - &#10003;
      - &#10003;

    * - double fabs ( double  x )
        | <sub>Calculate the absolute value of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double fdim ( double  x, double  y )
        | <sub>Compute the positive difference between `x` and `y`.</sub>
      - &#10003;
      - &#10003;

    * - double floor ( double  x )
        | <sub>Calculate the largest integer less than or equal to `x`.</sub>
      - &#10003;
      - &#10003;

    * - double fma ( double  x, double  y, double  z )
        | <sub>Compute `x × y + z` as a single operation.</sub>
      - &#10003;
      - &#10003;

    * - double fmax ( double , double )
        | <sub>Determine the maximum numeric value of the arguments.</sub>
      - &#10003;
      - &#10003;

    * - double fmin ( double  x, double  y )
        | <sub>Determine the minimum numeric value of the arguments.</sub>
      - &#10003;
      - &#10003;

    * - double fmod ( double  x, double  y )
        | <sub>Calculate the floating-point remainder of `x / y`.</sub>
      - &#10003;
      - &#10003;

    * - double frexp ( double  x, int* nptr )
        | <sub>Extract mantissa and exponent of a floating-point value.</sub>
      - &#10003;
      - &#10007;

    * - double hypot ( double  x, double  y )
        | <sub>Calculate the square root of the sum of squares of two arguments.</sub>
      - &#10003;
      - &#10003;

    * - int ilogb ( double  x )
        | <sub>Compute the unbiased integer exponent of the argument.</sub>
      - &#10003;
      - &#10003;

    * - __RETURN_TYPE[^f1] isfinite ( double  a )
        | <sub>Determine whether argument is finite.</sub>
      - &#10003;
      - &#10003;

    * - __RETURN_TYPE[^f1]</sup> isinf ( double  a )
        | <sub>Determine whether argument is infinite.</sub>
      - &#10003;
      - &#10003;

    * - __RETURN_TYPE[^f1]</sup> isnan ( double  a )
        | <sub>Determine whether argument is a NaN.</sub>
      - &#10003;
      - &#10003;

    * - double ldexp ( double  x, int  exp )
        | <sub>Calculate the value of x ⋅ 2<sup>exp</sup>.</sub>
      - &#10003;
      - &#10003;

    * - double log ( double  x )
        | <sub>Calculate the base e logarithm of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double log10 ( double  x )
        | <sub>Calculate the base 10 logarithm of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double log1p ( double  x )
        | <sub>Calculate the value of log<sub>e</sub>( 1 + x ).</sub>
      - &#10003;
      - &#10003;

    * - double log2 ( double  x )
        | <sub>Calculate the base 2 logarithm of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double logb ( double  x )
        | <sub>Calculate the floating point representation of the exponent of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double modf ( double  x, double* iptr )
        | <sub>Break down the input argument into fractional and integral parts.</sub>
      - &#10003;
      - &#10007;

    * - double nan ( const char* tagp )
        | <sub>Returns "Not a Number" value.</sub>
      - &#10007;
      - &#10003;

    * - double nearbyint ( double  x )
        | <sub>Round the input argument to the nearest integer.</sub>
      - &#10003;
      - &#10003;

    * - double pow ( double  x, double  y )
        | <sub>Calculate the value of first argument to the power of second argument.</sub>
      - &#10003;
      - &#10003;

    * - double remainder ( double  x, double  y )
        | <sub>Compute double-precision floating-point remainder.</sub>
      - &#10003;
      - &#10003;

    * - double remquo ( double  x, double  y, int* quo )
        | <sub>Compute double-precision floating-point remainder and part of quotient.</sub>
      - &#10003;
      - &#10007;

    * - double round ( double  x )
        | <sub>Round to nearest integer value in floating-point.</sub>
      - &#10003;
      - &#10003;

    * - double scalbn ( double  x, int  n )
        | <sub>Scale floating-point input by integer power of two.</sub>
      - &#10003;
      - &#10003;

    * - __RETURN_TYPE[^f1] signbit ( double  a )
        | <sub>Return the sign bit of the input.</sub>
      - &#10003;
      - &#10003;

    * - double sin ( double  x )
        | <sub>Calculate the sine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - void sincos ( double  x, double* sptr, double* cptr )
        | <sub>Calculate the sine and cosine of the first input argument.</sub>
      - &#10003;
      - &#10007;

    * - double sinh ( double  x )
        | <sub>Calculate the hyperbolic sine of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double sqrt ( double  x )
        | <sub>Calculate the square root of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double tan ( double  x )
        | <sub>Calculate the tangent of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double tanh ( double  x )
        | <sub>Calculate the hyperbolic tangent of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double tgamma ( double  x )
        | <sub>Calculate the gamma function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double trunc ( double  x )
        | <sub>Truncate input argument to the integral part.</sub>
      - &#10003;
      - &#10003;

    * - double erfcinv ( double  y )
        | <sub>Calculate the inverse complementary function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double erfcx ( double  x )
        | <sub>Calculate the scaled complementary error function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double erfinv ( double  y )
        | <sub>Calculate the inverse error function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double frexp ( float  x, int \*nptr )
        | <sub>Extract mantissa and exponent of a floating-point value.</sub>
      - &#10003;
      - &#10003;

    * - double j0 ( double  x )
        | <sub>Calculate the value of the Bessel function of the first kind of order 0 for the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double j1 ( double  x )
        | <sub>Calculate the value of the Bessel function of the first kind of order 1 for the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double jn ( int n, double  x )
        | <sub>Calculate the value of the Bessel function of the first kind of order n for the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double lgamma ( double  x )
        | <sub>Calculate the natural logarithm of the absolute value of the gamma function of the input argument.</sub>
      - &#10003;
      - &#10003;

    * - long long int llrint ( double  x )
        | <sub>Round input to nearest integer value.</sub>
      - &#10003;
      - &#10003;

    * - long long int llround ( double  x )
        | <sub>Round to nearest integer value.</sub>
      - &#10003;
      - &#10003;

    * - long int lrint ( double  x )
        | <sub>Round input to nearest integer value.</sub>
      - &#10003;
      - &#10003;

    * - long int lround ( double  x )
        | <sub>Round to nearest integer value.</sub>
      - &#10003;
      - &#10003;

    * - double modf ( double  x, double \*iptr )
        | <sub>Break down the input argument into fractional and integral parts.</sub>
      - &#10003;
      - &#10003;

    * - double nextafter ( double  x, double y )
        | <sub>Returns next representable single-precision floating-point value after argument.</sub>
      - &#10003;
      - &#10003;

    * - double norm3d ( double  a, double b, double c )
        | <sub>Calculate the square root of the sum of squares of three coordinates of the argument.</sub>
      - &#10003;
      - &#10003;

    * - float norm4d ( double  a, double b, double c, double d )
        | <sub>Calculate the square root of the sum of squares of four coordinates of the argument.</sub>
      - &#10003;
      - &#10003;

    * - double normcdf ( double  y )
        | <sub>Calculate the standard normal cumulative distribution function.</sub>
      - &#10003;
      - &#10003;

    * - double normcdfinv ( double  y )
        | <sub>Calculate the inverse of the standard normal cumulative distribution function.</sub>
      - &#10003;
      - &#10003;

    * - double rcbrt ( double x )
        | <sub>Calculate the reciprocal cube root function.</sub>
      - &#10003;
      - &#10003;

    * - double remquo ( double x, double y, int \*quo )
        | <sub>Compute single-precision floating-point remainder and part of quotient.</sub>
      - &#10003;
      - &#10003;

    * - double rhypot ( double x, double y )
        | <sub>Calculate one over the square root of the sum of squares of two arguments.</sub>
      - &#10003;
      - &#10003;

    * - double rint ( double x )
        | <sub>Round input to nearest integer value in floating-point.</sub>
      - &#10003;
      - &#10003;

    * - double rnorm3d ( double a, double b, double c )
        | <sub>Calculate one over the square root of the sum of squares of three coordinates of the argument.</sub>
      - &#10003;
      - &#10003;

    * - double rnorm4d ( double a, double b, double c, double d )
        | <sub>Calculate one over the square root of the sum of squares of four coordinates of the argument.</sub>
      - &#10003;
      - &#10003;

    * - double rnorm ( int dim, const double \*a )
        | <sub>Calculate the reciprocal of square root of the sum of squares of any number of coordinates.</sub>
      - &#10003;
      - &#10003;

    * - double scalbln ( double x, long int n )
        | <sub>Scale floating-point input by integer power of two.</sub>
      - &#10003;
      - &#10003;

    * - void sincos ( double x, double *sptr, double *cptr )
        | <sub>Calculate the sine and cosine of the first input argument.</sub>
      - &#10003;
      - &#10003;

    * - void sincospi ( double x, double *sptr, double *cptr )
        | <sub>Calculate the sine and cosine of the first input argument multiplied by PI.</sub>
      - &#10003;
      - &#10003;

    * - double y0f ( double  x )
        | <sub>Calculate the value of the Bessel function of the second kind of order 0 for the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double y1 ( double  x )
        | <sub>Calculate the value of the Bessel function of the second kind of order 1 for the input argument.</sub>
      - &#10003;
      - &#10003;

    * - double yn ( int n, double  x )
        | <sub>Calculate the value of the Bessel function of the second kind of order n for the input argument.</sub>
      - &#10003;
      - &#10003;

Integer Intrinsics
----------------------------------------------------------------------------------------------------------------

The following integer intrinsics are supported (on device only).

.. list-table::
    * - **Function**

    * - unsigned int __brev ( unsigned int x )
        | <sub>Reverse the bit order of a 32 bit unsigned integer.</sub>

    * - unsigned long long int __brevll ( unsigned long long int x )
        | <sub>Reverse the bit order of a 64 bit unsigned integer.</sub>

    * - int __clz ( int  x )
        | <sub>Return the number of consecutive high-order zero bits in a 32 bit integer.</sub>

    * - unsigned int __clz(unsigned int x)
        | <sub>Return the number of consecutive high-order zero bits in 32 bit unsigned integer.</sub>

    * - int __clzll ( long long int x )
        | <sub>Count the number of consecutive high-order zero bits in a 64 bit integer.</sub>

    * - unsigned int __clzll(long long int x)
        | <sub>Return the number of consecutive high-order zero bits in 64 bit signed integer.</sub>

    * - unsigned int __ffs(unsigned int x)
        | <sub>Find the position of least signigicant bit set to 1 in a 32 bit unsigned integer<sup>1</sup>.</sub>

    * - unsigned int __ffs(int x)
        | <sub>Find the position of least signigicant bit set to 1 in a 32 bit signed integer.</sub>

    * - unsigned int __ffsll(unsigned long long int x)
        | <sub>Find the position of least signigicant bit set to 1 in a 64 bit unsigned integer<sup>1</sup>.</sub>

    * - unsigned int __ffsll(long long int x)
        | <sub>Find the position of least signigicant bit set to 1 in a 64 bit signed integer.</sub>

    * - unsigned int __popc ( unsigned int x )
        | <sub>Count the number of bits that are set to 1 in a 32 bit integer.</sub>

    * - unsigned int __popcll ( unsigned long long int x )
        | <sub>Count the number of bits that are set to 1 in a 64 bit integer.</sub>

    * - int __mul24 ( int x, int y )
        | <sub>Multiply two 24bit integers.</sub>

    * - unsigned int __umul24 ( unsigned int x, unsigned int y )
        | <sub>Multiply two 24bit unsigned integers.</sub>


<sup>1</sup> The HIP-Clang implementation of ``__ffs() and __ffsll()`` contains code to add a constant
+1 to produce the ffs result format. For the cases where this overhead is not acceptable and you want
to specialize for the platform, HIP-Clang provides ``__lastbit_u32_u32(unsigned int input)`` and
``__lastbit_u32_u64(unsigned long long int input)``. The index returned by ``__lastbit_`` instructions starts
at -1; the index for ffs starts at 0.

Floating-point intrinsics
----------------------------------------------------------------------------------------------------------------

The following floating-point intrinsics are supported (on device only).

.. list-table::
    * - **Function**

    * - float __cosf ( float  x )
        | <sub>Calculate the fast approximate cosine of the input argument.</sub>

    * - float __expf ( float  x )
        | <sub>Calculate the fast approximate base e exponential of the input argument.</sub>

    * - float __frsqrt_rn ( float  x )
        | <sub>Compute :math:`1 / √x` in round-to-nearest-even mode.</sub>

    * - float __fsqrt_rn ( float  x )
        | <sub>Compute :math:`√x` in round-to-nearest-even mode.</sub>

    * - float __log10f ( float  x )
        | <sub>Calculate the fast approximate base 10 logarithm of the input argument.</sub>

    * - float __log2f ( float  x )
        | <sub>Calculate the fast approximate base 2 logarithm of the input argument.</sub>

    * - float __logf ( float  x )
        | <sub>Calculate the fast approximate base e logarithm of the input argument.</sub>

    * - float __powf ( float  x, float  y )
        | <sub>Calculate the fast approximate of x<sup>y</sup>.</sub>

    * - float __sinf ( float  x )
        | <sub>Calculate the fast approximate sine of the input argument.</sub>

    * - float __tanf ( float  x )
        | <sub>Calculate the fast approximate tangent of the input argument.</sub>

    * - double __dsqrt_rn ( double  x )
        | <sub>Compute :math:`√x` in round-to-nearest-even mode.</sub>

Texture functions
===============================================

The supported texture functions are listed in
`texture_fetch_functions.h <https://github.com/ROCm-Developer-Tools/HIP/blob/main/include/hip/hcc_detail/texture_fetch_functions.h)`_
and `texture_indirect_functions.h <https://github.com/ROCm-Developer-Tools/HIP/blob/main/include/hip/hcc_detail/texture_indirect_functions.h>`_.

Texture functions are not supported on some devices. To determine if texture functions are supported
on your device, use ``Macro __HIP_NO_IMAGE_SUPPORT == 1``. You can query the attribute
``hipDeviceAttributeImageSupport`` to check if texture functions are supported in the host runtime
code.

Surface functions
===============================================

Surface functions are not supported.

Timer functions
===============================================

To read a high-resolution timer from the device, HIP provides the following built-in functions:

* Returning the incremental counter value for every clock cycle on a device:

  .. code:: cpp

    clock_t clock()
    long long int clock64()

  The difference between the values that are returned represents the cycles used.

* Returning the wall clock count at a constant frequency on the device:

  .. code:: cpp

    long long int wall_clock64()

  This can be queried using the HIP API with the ``hipDeviceAttributeWallClockRate`` attribute of the
  device in HIP application code. For example:

  .. code:: cpp

    int wallClkRate = 0; //in kilohertz
    HIPCHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, deviceId));

  Where ``hipDeviceAttributeWallClockRate`` is a device attribute. Note that wall clock frequency is a
  per-device attribute.

Atomic functions
===============================================

Atomic functions are run as read-modify-write (RMW) operations that reside in global or shared
memory. No other device or thread can observe or modify the memory location during an atomic
operation. If multiple instructions from different devices or threads target the same memory location,
the instructions are serialized in an undefined order.

To support system scope atomic operations, you can use the HIP APIs that contain the ``_system`` suffix.
For example:
* ``atomicAnd``: This function is atomic and coherent within the GPU device running the function
* ``atomicAnd_system``: This function extends the atomic operation from the GPU device to other CPUs
  and GPU devices in the system

HIP supports the following atomic operations.

.. list-table::
    * - **Function**
    - **Supported in HIP**
    - **Supported in CUDA**

    * - int atomicAdd(int* address, int val)
      - &#10003;
      - &#10003;

    * - int atomicAdd_system(int* address, int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicAdd(unsigned int* address,unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicAdd_system(unsigned int* address, unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicAdd(unsigned long long* address,unsigned long long val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicAdd_system(unsigned long long* address, unsigned long long val)
      - &#10003;
      - &#10003;

    * - float atomicAdd(float* address, float val)
      - &#10003;
      - &#10003;

    * - float atomicAdd_system(float* address, float val)
      - &#10003;
      - &#10003;

    * - double atomicAdd(double* address, double val)
      - &#10003;
      - &#10003;

    * - double atomicAdd_system(double* address, double val)
      - &#10003;
      - &#10003;

    * - float unsafeAtomicAdd(float* address, float val)
      - &#10003;
      - &#10007;

    * - float safeAtomicAdd(float* address, float val)
      - &#10003;
      - &#10007;

    * - double unsafeAtomicAdd(double* address, double val)
      - &#10003;
      - &#10007;

    * - double safeAtomicAdd(double* address, double val)
      - &#10003;
      - &#10007;

    * - int atomicSub(int* address, int val)
      - &#10003;
      - &#10003;

    * - int atomicSub_system(int* address, int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicSub(unsigned int* address,unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicSub_system(unsigned int* address, unsigned int val)
      - &#10003;
      - &#10003;

    * - int atomicExch(int* address, int val)
      - &#10003;
      - &#10003;

    * - int atomicExch_system(int* address, int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicExch(unsigned int* address,unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicExch_system(unsigned int* address, unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicExch(unsigned long long int* address,unsigned long long int val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val)
      - &#10003;
      - &#10003;

    * - float atomicExch(float* address, float val)
      - &#10003;
      - &#10003;

    * - int atomicMin(int* address, int val)
      - &#10003;
      - &#10003;

    * - int atomicMin_system(int* address, int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicMin(unsigned int* address,unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicMin_system(unsigned int* address, unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicMin(unsigned long long* address,unsigned long long val)
      - &#10003;
      - &#10003;

    * - int atomicMax(int* address, int val)
      - &#10003;
      - &#10003;

    * - int atomicMax_system(int* address, int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicMax(unsigned int* address,unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicMax_system(unsigned int* address, unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicMax(unsigned long long* address,unsigned long long val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicInc(unsigned int* address)
      - &#10007;
      - &#10003;

    * - unsigned int atomicDec(unsigned int* address)
      - &#10007;
      - &#10003;

    * - int atomicCAS(int* address, int compare, int val)
      - &#10003;
      - &#10003;

    * - int atomicCAS_system(int* address, int compare, int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicCAS(unsigned int* address,unsigned int compare,unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicCAS_system(unsigned int* address, unsigned int compare, unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicCAS(unsigned long long* address,unsigned long long compare,unsigned long long val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicCAS_system(unsigned long long* address, unsigned long long compare, unsigned long long val)
      - &#10003;
      - &#10003;

    * - int atomicAnd(int* address, int val)
      - &#10003;
      - &#10003;

    * - int atomicAnd_system(int* address, int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicAnd(unsigned int* address,unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicAnd_system(unsigned int* address, unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicAnd(unsigned long long* address,unsigned long long val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicAnd_system(unsigned long long* address, unsigned long long val)
      - &#10003;
      - &#10003;

    * - int atomicOr(int* address, int val)
      - &#10003;
      - &#10003;

    * - int atomicOr_system(int* address, int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicOr(unsigned int* address,unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicOr_system(unsigned int* address, unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicOr_system(unsigned int* address, unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicOr(unsigned long long int* address,unsigned long long val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicOr_system(unsigned long long* address, unsigned long long val)
      - &#10003;
      - &#10003;

    * - int atomicXor(int* address, int val)
      - &#10003;
      - &#10003;

    * - int atomicXor_system(int* address, int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicXor(unsigned int* address,unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned int atomicXor_system(unsigned int* address, unsigned int val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicXor(unsigned long long* address,unsigned long long val)
      - &#10003;
      - &#10003;

    * - unsigned long long atomicXor_system(unsigned long long* address, unsigned long long val)
      - &#10003;
      - &#10003;

Unsafe floating-point atomic RMW operations
----------------------------------------------------------------------------------------------------------------
Some HIP devices support fast atomic RMW operations on floating-point values. For example,
`atomicAdd` on single- or double-precision floating-point values may generate a hardware RMW
instruction that is faster than emulating the atomic operation using an atomic compare-and-swap
(CAS) loop.

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

Note that NVIDIA and AMD devices have different warp sizes, so portable code should use the warpSize built-ins to query the warp size. Hipified code from the CUDA path requires careful review to ensure it doesn’t assume a waveSize of 32. "Wave-aware" code that assumes a waveSize of 32 will run on a wave-64 machine, but it will utilize only half of the machine resources. WarpSize built-ins should only be used in device functions and its value depends on GPU arch. Users should not assume warpSize to be a compile-time constant. Host functions should use hipGetDeviceProperties to get the default warp size of a GPU device:

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

`__ballot` provides a bit mask containing the 1-bit predicate value from each lane. The nth bit of the result contains the 1 bit contributed by the nth warp lane. Note that HIP's `__ballot` function supports a 64-bit return value (compared with CUDA's 32 bits). Code ported from CUDA should support the larger warp sizes that the HIP version of this instruction supports. Applications can test whether the target platform supports the ballot instruction using the `hasWarpBallot` device property or the HIP_ARCH_HAS_WARP_BALLOT compiler define.


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
a granularity different than the block.  This feature was introduced in CUDA 9.

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
was introduced in CUDA 9.

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
supporting CUDA allows threads to progress independently of each other and enables
intra-warp synchronizations that were previously not allowed.

HIP does not support this type of scheduling.

## Profiler Counter Function

The CUDA `__prof_trigger()` instruction is not supported.

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

- The second parameter __launch_bounds parameters must be converted to the format used __hip_launch_bounds, which uses warps and execution-units rather than blocks and multi-processors (this conversion is performed automatically by HIPIFY tools).
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
