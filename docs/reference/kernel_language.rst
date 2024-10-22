.. meta::
  :description: This chapter describes the built-in variables and functions that are accessible from the
                HIP kernel. It's intended for users who are familiar with CUDA kernel syntax and want to
                learn how HIP differs from CUDA.
  :keywords: AMD, ROCm, HIP, CUDA, c++ language extensions, HIP functions

********************************************************************************
C++ Language Extensions
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

Supported ``__device__`` functions are:

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

HIP doesn't support dynamic-parallelism, which means that you can't call ``__global__`` functions from
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
*launching a kernel*. When launching a kernel, you must specify an execution configuration that includes the
grid and block dimensions. The execution configuration can also include other information for the launch,
such as the amount of additional shared memory to allocate and the stream where you want to execute the
kernel.

HIP introduces a standard C++ calling convention (``hipLaunchKernelGGL``) to pass the run
configuration to the kernel. However, you can also use the CUDA ``<<< >>>`` syntax.

When using ``hipLaunchKernelGGL``, your first five parameters must be:

  * ``symbol kernelName``: The name of the kernel you want to launch. To support template kernels
    that contain ``","``, use the ``HIP_KERNEL_NAME`` macro (HIPIFY tools insert this automatically).
  * ``dim3 gridDim``: 3D-grid dimensions that specify the number of blocks to launch.
  * ``dim3 blockDim``: 3D-block dimensions that specify the number of threads in each block.
  * ``size_t dynamicShared``: The amount of additional shared memory that you want to allocate
    when launching the kernel (see :ref:`shared-variable-type`).
  * ``hipStream_t``: The stream where you want to run the kernel. A value of ``0`` corresponds to the
    NULL stream (see :ref:`synchronization functions`).

You can include your kernel arguments after these parameters.

.. code-block:: cpp

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

  HIP doesn't support dimension sizes of :math:`gridDim * blockDim \ge 2^{32}` when launching a kernel.

.. _kernel-launch-example:

Kernel launch example
==========================================================

.. code-block:: cpp

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

.. note::

  Add ``__constant__`` to a template can lead to undefined behavior. Refer to `HIP Issue #3201 <https://github.com/ROCm/HIP/issues/3201>`_ for details.

.. _shared-variable-type:

``__shared__``
-----------------------------------------------------------------------------

To allow the host to dynamically allocate shared memory, you can specify ``extern __shared__`` as a
launch parameter.

.. note::

  Prior to the HIP-Clang compiler, dynamic shared memory had to be declared using the
  ``HIP_DYNAMIC_SHARED`` macro in order to ensure accuracy. This is because using static shared
  memory in the same kernel could've resulted in overlapping memory ranges and data-races. The
  HIP-Clang compiler provides support for ``extern __shared_`` declarations, so ``HIP_DYNAMIC_SHARED``
  is no longer required.

``__managed__``
-----------------------------------------------------------------------------

Managed memory, including the ``__managed__`` keyword, is supported in HIP combined host/device
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

``warpSize``
-----------------------------------------------------------------------------
The ``warpSize`` variable type is ``int``. It contains the warp size (in threads) for the target device.
``warpSize`` should only be used in device functions that develop portable wave-aware code.

.. note::

  NVIDIA devices return 32 for this variable; AMD devices return 64 for gfx9 and 32 for gfx10 and above.

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

  * ``char1``, ``char2``, ``char3``, ``char4``
  * ``short1``, ``short2``, ``short3``, ``short4``
  * ``int1``, ``int2``, ``int3``, ``int4``
  * ``long1``, ``long2``, ``long3``, ``long4``
  * ``longlong1``, ``longlong2``, ``longlong3``, ``longlong4``

* Unsigned Integers:

  * ``uchar1``, ``uchar2``, ``uchar3``, ``uchar4``
  * ``ushort1``, ``ushort2``, ``ushort3``, ``ushort4``
  * ``uint1``, ``uint2``, ``uint3``, ``uint4``
  * ``ulong1``, ``ulong2``, ``ulong3``, ``ulong4``
  * ``ulonglong1``, ``ulonglong2``, ``ulonglong3``, ``ulonglong4``

* Floating Points:

  * ``float1``, ``float2``, ``float3``, ``float4``
  * ``double1``, ``double2``, ``double3``, ``double4``

.. _dim3:

dim3
--------------------------------------------------------------------------------------------

``dim3`` is a three-dimensional integer vector type that is commonly used to specify grid and group
dimensions.

The dim3 constructor accepts between zero and three arguments. By default, it initializes unspecified
dimensions to 1.

.. code-block:: cpp

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

.. _synchronization functions:

Synchronization functions
====================================================
The ``__syncthreads()`` built-in function is supported in HIP. The ``__syncthreads_count(int)``,
``__syncthreads_and(int)``, and ``__syncthreads_or(int)`` functions are under development.

Math functions
====================================================

HIP-Clang supports a set of math operations that are callable from the device. HIP supports most of the device functions supported by CUDA.
These are described in the following sections.

Single precision mathematical functions
--------------------------------------------------------------------------------------------

Following is the list of supported single precision mathematical functions.

.. list-table:: Single precision mathematical functions

    * - **Function**
      - **Supported on Host**
      - **Supported on Device**

    * - | ``float abs(float x)``
        | Returns the absolute value of :math:`x`
      - ✓
      - ✓

    * - | ``float acosf(float x)``
        | Returns the arc cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``float acoshf(float x)``
        | Returns the nonnegative arc hyperbolic cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``float asinf(float x)``
        | Returns the arc sine of :math:`x`.
      - ✓
      - ✓

    * - | ``float asinhf(float x)``
        | Returns the arc hyperbolic sine of :math:`x`.
      - ✓
      - ✓

    * - | ``float atanf(float x)``
        | Returns the arc tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``float atan2f(float x, float y)``
        | Returns the arc tangent of the ratio of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``float atanhf(float x)``
        | Returns the arc hyperbolic tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``float cbrtf(float x)``
        | Returns the cube root of :math:`x`.
      - ✓
      - ✓

    * - | ``float ceilf(float x)``
        | Returns ceiling of :math:`x`.
      - ✓
      - ✓

    * - | ``float copysignf(float x, float y)``
        | Create value with given magnitude, copying sign of second value.
      - ✓
      - ✓

    * - | ``float cosf(float x)``
        | Returns the cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``float coshf(float x)``
        | Returns the hyperbolic cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``float cospif(float x)``
        | Returns the cosine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``float cyl_bessel_i0f(float x)``
        | Returns the value of the regular modified cylindrical Bessel function of order 0 for :math:`x`.
      - ✗
      - ✗

    * - | ``float cyl_bessel_i1f(float x)``
        | Returns the value of the regular modified cylindrical Bessel function of order 1 for :math:`x`.
      - ✗
      - ✗

    * - | ``float erff(float x)``
        | Returns the error function of :math:`x`.
      - ✓
      - ✓

    * - | ``float erfcf(float x)``
        | Returns the complementary error function of :math:`x`.
      - ✓
      - ✓

    * - | ``float erfcinvf(float x)``
        | Returns the inverse complementary function of :math:`x`.
      - ✓
      - ✓

    * - | ``float erfcxf(float x)``
        | Returns the scaled complementary error function of :math:`x`.
      - ✓
      - ✓

    * - | ``float erfinvf(float x)``
        | Returns the inverse error function of :math:`x`.
      - ✓
      - ✓

    * - | ``float expf(float x)``
        | Returns :math:`e^x`.
      - ✓
      - ✓

    * - | ``float exp10f(float x)``
        | Returns :math:`10^x`.
      - ✓
      - ✓

    * - | ``float exp2f( float x)``
        | Returns :math:`2^x`.
      - ✓
      - ✓

    * - | ``float expm1f(float x)``
        | Returns :math:`ln(x - 1)`
      - ✓
      - ✓

    * - | ``float fabsf(float x)``
        | Returns the absolute value of `x`
      - ✓
      - ✓

    * - | ``float fdimf(float x, float y)``
        | Returns the positive difference between :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``float fdividef(float x, float y)``
        | Divide two floating point values.
      - ✓
      - ✓

    * - | ``float floorf(float x)``
        | Returns the largest integer less than or equal to :math:`x`.
      - ✓
      - ✓

    * - | ``float fmaf(float x, float y, float z)``
        | Returns :math:`x \cdot y + z` as a single operation.
      - ✓
      - ✓

    * - | ``float fmaxf(float x, float y)``
        | Determine the maximum numeric value of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``float fminf(float x, float y)``
        | Determine the minimum numeric value of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``float fmodf(float x, float y)``
        | Returns the floating-point remainder of :math:`x / y`.
      - ✓
      - ✓

    * - | ``float modff(float x, float* iptr)``
        | Break down :math:`x` into fractional and integral parts.
      - ✓
      - ✗

    * - | ``float frexpf(float x, int* nptr)``
        | Extract mantissa and exponent of :math:`x`.
      - ✓
      - ✗

    * - | ``float hypotf(float x, float y)``
        | Returns the square root of the sum of squares of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``int ilogbf(float x)``
        | Returns the unbiased integer exponent of :math:`x`.
      - ✓
      - ✓

    * - | ``bool isfinite(float x)``
        | Determine whether :math:`x` is finite.
      - ✓
      - ✓

    * - | ``bool isinf(float x)``
        | Determine whether :math:`x` is infinite.
      - ✓
      - ✓

    * - | ``bool isnan(float x)``
        | Determine whether :math:`x` is a ``NAN``.
      - ✓
      - ✓

    * - | ``float j0f(float x)``
        | Returns the value of the Bessel function of the first kind of order 0 for :math:`x`.
      - ✓
      - ✓

    * - | ``float j1f(float x)``
        | Returns the value of the Bessel function of the first kind of order 1 for :math:`x`.
      - ✓
      - ✓

    * - | ``float jnf(int n, float x)``
        | Returns the value of the Bessel function of the first kind of order n for :math:`x`.
      - ✓
      - ✓

    * - | ``float ldexpf(float x, int exp)``
        | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
      - ✓
      - ✓

    * - | ``float lgammaf(float x)``
        | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
      - ✓
      - ✗

    * - | ``long int lrintf(float x)``
        | Round :math:`x` to nearest integer value.
      - ✓
      - ✓

    * - | ``long long int llrintf(float x)``
        | Round :math:`x` to nearest integer value.
      - ✓
      - ✓

    * - | ``long int lroundf(float x)``
        | Round to nearest integer value.
      - ✓
      - ✓

    * - | ``long long int llroundf(float x)``
        | Round to nearest integer value.
      - ✓
      - ✓

    * - | ``float log10f(float x)``
        | Returns the base 10 logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``float log1pf(float x)``
        | Returns the natural logarithm of :math:`x + 1`.
      - ✓
      - ✓

    * - | ``float log2f(float x)``
        | Returns the base 2 logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``float logf(float x)``
        | Returns the natural logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``float logbf(float x)``
        | Returns the floating point representation of the exponent of :math:`x`.
      - ✓
      - ✓

    * - | ``float nanf(const char* tagp)``
        | Returns "Not a Number" value.
      - ✗
      - ✓

    * - | ``float nearbyintf(float x)``
        | Round :math:`x` to the nearest integer.
      - ✓
      - ✓

    * - | ``float nextafterf(float x, float y)``
        | Returns next representable single-precision floating-point value after argument.
      - ✓
      - ✗

    * - | ``float norm3df(float x, float y, float z)``
        | Returns the square root of the sum of squares of :math:`x`, :math:`y` and :math:`z`.
      - ✓
      - ✓

    * - | ``float norm4df(float x, float y, float z, float w)``
        | Returns the square root of the sum of squares of :math:`x`, :math:`y`, :math:`z` and :math:`w`.
      - ✓
      - ✓

    * - | ``float normcdff(float y)``
        | Returns the standard normal cumulative distribution function.
      - ✓
      - ✓

    * - | ``float normcdfinvf(float y)``
        | Returns the inverse of the standard normal cumulative distribution function.
      - ✓
      - ✓

    * - | ``float normf(int dim, const float *a)``
        | Returns the square root of the sum of squares of any number of coordinates.
      - ✓
      - ✓

    * - | ``float powf(float x, float y)``
        | Returns :math:`x^y`.
      - ✓
      - ✓

    * - | ``float powif(float base, int iexp)``
        | Returns the value of first argument to the power of second argument.
      - ✓
      - ✓

    * - | ``float remainderf(float x, float y)``
        | Returns single-precision floating-point remainder.
      - ✓
      - ✓

    * - | ``float remquof(float x, float y, int* quo)``
        | Returns single-precision floating-point remainder and part of quotient.
      - ✓
      - ✓

    * - | ``float roundf(float x)``
        | Round to nearest integer value in floating-point.
      - ✓
      - ✓

    * - | ``float rcbrtf(float x)``
        | Returns the reciprocal cube root function.
      - ✓
      - ✓

    * - | ``float rhypotf(float x, float y)``
        | Returns one over the square root of the sum of squares of two arguments.
      - ✓
      - ✓

    * - | ``float rintf(float x)``
        | Round input to nearest integer value in floating-point.
      - ✓
      - ✓

    * - | ``float rnorm3df(float x, float y, float z)``
        | Returns one over the square root of the sum of squares of three coordinates of the argument.
      - ✓
      - ✓

    * - | ``float rnorm4df(float x, float y, float z, float w)``
        | Returns one over the square root of the sum of squares of four coordinates of the argument.
      - ✓
      - ✓

    * - | ``float rnormf(int dim, const float *a)``
        | Returns the reciprocal of square root of the sum of squares of any number of coordinates.
      - ✓
      - ✓

    * - | ``float scalblnf(float x, long int n)``
        | Scale :math:`x` by :math:`2^n`.
      - ✓
      - ✓

    * - | ``float scalbnf(float x, int n)``
        | Scale :math:`x` by :math:`2^n`.
      - ✓
      - ✓

    * - | ``bool signbit(float x)``
        | Return the sign bit of :math:`x`.
      - ✓
      - ✓

    * - | ``float sinf(float x)``
        | Returns the sine of :math:`x`.
      - ✓
      - ✓

    * - | ``float sinhf(float x)``
        | Returns the hyperbolic sine of :math:`x`.
      - ✓
      - ✓

    * - | ``float sinpif(float x)``
        | Returns the hyperbolic sine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``void sincosf(float x, float *sptr, float *cptr)``
        | Returns the sine and cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``void sincospif(float x, float *sptr, float *cptr)``
        | Returns the sine and cosine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``float sqrtf(float x)``
        | Returns the square root of :math:`x`.
      - ✓
      - ✓

    * - | ``float rsqrtf(float x)``
        | Returns the reciprocal of the square root of :math:`x`.
      - ✗
      - ✓

    * - | ``float tanf(float x)``
        | Returns the tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``float tanhf(float x)``
        | Returns the hyperbolic tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``float tgammaf(float x)``
        | Returns the gamma function of :math:`x`.
      - ✓
      - ✓

    * - | ``float truncf(float x)``
        | Truncate :math:`x` to the integral part.
      - ✓
      - ✓

    * - | ``float y0f(float x)``
        | Returns the value of the Bessel function of the second kind of order 0 for :math:`x`.
      - ✓
      - ✓

    * - | ``float y1f(float x)``
        | Returns the value of the Bessel function of the second kind of order 1 for :math:`x`.
      - ✓
      - ✓

    * - | ``float ynf(int n, float x)``
        | Returns the value of the Bessel function of the second kind of order n for :math:`x`.
      - ✓
      - ✓

Double precision mathematical functions
--------------------------------------------------------------------------------------------

Following is the list of supported double precision mathematical functions.

.. list-table:: Double precision mathematical functions

    * - **Function**
      - **Supported on Host**
      - **Supported on Device**

    * - | ``double abs(double x)``
        | Returns the absolute value of :math:`x`
      - ✓
      - ✓

    * - | ``double acos(double x)``
        | Returns the arc cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``double acosh(double x)``
        | Returns the nonnegative arc hyperbolic cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``double asin(double x)``
        | Returns the arc sine of :math:`x`.
      - ✓
      - ✓

    * - | ``double asinh(double x)``
        | Returns the arc hyperbolic sine of :math:`x`.
      - ✓
      - ✓

    * - | ``double atan(double x)``
        | Returns the arc tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``double atan2(double x, double y)``
        | Returns the arc tangent of the ratio of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``double atanh(double x)``
        | Returns the arc hyperbolic tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``double cbrt(double x)``
        | Returns the cube root of :math:`x`.
      - ✓
      - ✓

    * - | ``double ceil(double x)``
        | Returns ceiling of :math:`x`.
      - ✓
      - ✓

    * - | ``double copysign(double x, double y)``
        | Create value with given magnitude, copying sign of second value.
      - ✓
      - ✓

    * - | ``double cos(double x)``
        | Returns the cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``double cosh(double x)``
        | Returns the hyperbolic cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``double cospi(double x)``
        | Returns the cosine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``double cyl_bessel_i0(double x)``
        | Returns the value of the regular modified cylindrical Bessel function of order 0 for :math:`x`.
      - ✗
      - ✗

    * - | ``double cyl_bessel_i1(double x)``
        | Returns the value of the regular modified cylindrical Bessel function of order 1 for :math:`x`.
      - ✗
      - ✗

    * - | ``double erf(double x)``
        | Returns the error function of :math:`x`.
      - ✓
      - ✓

    * - | ``double erfc(double x)``
        | Returns the complementary error function of :math:`x`.
      - ✓
      - ✓

    * - | ``double erfcinv(double x)``
        | Returns the inverse complementary function of :math:`x`.
      - ✓
      - ✓

    * - | ``double erfcx(double x)``
        | Returns the scaled complementary error function of :math:`x`.
      - ✓
      - ✓

    * - | ``double erfinv(double x)``
        | Returns the inverse error function of :math:`x`.
      - ✓
      - ✓

    * - | ``double exp(double x)``
        | Returns :math:`e^x`.
      - ✓
      - ✓

    * - | ``double exp10(double x)``
        | Returns :math:`10^x`.
      - ✓
      - ✓

    * - | ``double exp2( double x)``
        | Returns :math:`2^x`.
      - ✓
      - ✓

    * - | ``double expm1(double x)``
        | Returns :math:`ln(x - 1)`
      - ✓
      - ✓

    * - | ``double fabs(double x)``
        | Returns the absolute value of `x`
      - ✓
      - ✓

    * - | ``double fdim(double x, double y)``
        | Returns the positive difference between :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``double floor(double x)``
        | Returns the largest integer less than or equal to :math:`x`.
      - ✓
      - ✓

    * - | ``double fma(double x, double y, double z)``
        | Returns :math:`x \cdot y + z` as a single operation.
      - ✓
      - ✓

    * - | ``double fmax(double x, double y)``
        | Determine the maximum numeric value of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``double fmin(double x, double y)``
        | Determine the minimum numeric value of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``double fmod(double x, double y)``
        | Returns the floating-point remainder of :math:`x / y`.
      - ✓
      - ✓

    * - | ``double modf(double x, double* iptr)``
        | Break down :math:`x` into fractional and integral parts.
      - ✓
      - ✗

    * - | ``double frexp(double x, int* nptr)``
        | Extract mantissa and exponent of :math:`x`.
      - ✓
      - ✗

    * - | ``double hypot(double x, double y)``
        | Returns the square root of the sum of squares of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``int ilogb(double x)``
        | Returns the unbiased integer exponent of :math:`x`.
      - ✓
      - ✓

    * - | ``bool isfinite(double x)``
        | Determine whether :math:`x` is finite.
      - ✓
      - ✓

    * - | ``bool isin(double x)``
        | Determine whether :math:`x` is infinite.
      - ✓
      - ✓

    * - | ``bool isnan(double x)``
        | Determine whether :math:`x` is a ``NAN``.
      - ✓
      - ✓

    * - | ``double j0(double x)``
        | Returns the value of the Bessel function of the first kind of order 0 for :math:`x`.
      - ✓
      - ✓

    * - | ``double j1(double x)``
        | Returns the value of the Bessel function of the first kind of order 1 for :math:`x`.
      - ✓
      - ✓

    * - | ``double jn(int n, double x)``
        | Returns the value of the Bessel function of the first kind of order n for :math:`x`.
      - ✓
      - ✓

    * - | ``double ldexp(double x, int exp)``
        | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
      - ✓
      - ✓

    * - | ``double lgamma(double x)``
        | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
      - ✓
      - ✗

    * - | ``long int lrint(double x)``
        | Round :math:`x` to nearest integer value.
      - ✓
      - ✓

    * - | ``long long int llrint(double x)``
        | Round :math:`x` to nearest integer value.
      - ✓
      - ✓

    * - | ``long int lround(double x)``
        | Round to nearest integer value.
      - ✓
      - ✓

    * - | ``long long int llround(double x)``
        | Round to nearest integer value.
      - ✓
      - ✓

    * - | ``double log10(double x)``
        | Returns the base 10 logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``double log1p(double x)``
        | Returns the natural logarithm of :math:`x + 1`.
      - ✓
      - ✓

    * - | ``double log2(double x)``
        | Returns the base 2 logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``double log(double x)``
        | Returns the natural logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``double logb(double x)``
        | Returns the floating point representation of the exponent of :math:`x`.
      - ✓
      - ✓

    * - | ``double nan(const char* tagp)``
        | Returns "Not a Number" value.
      - ✗
      - ✓

    * - | ``double nearbyint(double x)``
        | Round :math:`x` to the nearest integer.
      - ✓
      - ✓

    * - | ``double nextafter(double x, double y)``
        | Returns next representable double-precision floating-point value after argument.
      - ✓
      - ✓

    * - | ``double norm3d(double x, double y, double z)``
        | Returns the square root of the sum of squares of :math:`x`, :math:`y` and :math:`z`.
      - ✓
      - ✓

    * - | ``double norm4d(double x, double y, double z, double w)``
        | Returns the square root of the sum of squares of :math:`x`, :math:`y`, :math:`z` and :math:`w`.
      - ✓
      - ✓

    * - | ``double normcdf(double y)``
        | Returns the standard normal cumulative distribution function.
      - ✓
      - ✓

    * - | ``double normcdfinv(double y)``
        | Returns the inverse of the standard normal cumulative distribution function.
      - ✓
      - ✓

    * - | ``double norm(int dim, const double *a)``
        | Returns the square root of the sum of squares of any number of coordinates.
      - ✓
      - ✓

    * - | ``double pow(double x, double y)``
        | Returns :math:`x^y`.
      - ✓
      - ✓

    * - | ``double powi(double base, int iexp)``
        | Returns the value of first argument to the power of second argument.
      - ✓
      - ✓

    * - | ``double remainder(double x, double y)``
        | Returns double-precision floating-point remainder.
      - ✓
      - ✓

    * - | ``double remquo(double x, double y, int* quo)``
        | Returns double-precision floating-point remainder and part of quotient.
      - ✓
      - ✗

    * - | ``double round(double x)``
        | Round to nearest integer value in floating-point.
      - ✓
      - ✓

    * - | ``double rcbrt(double x)``
        | Returns the reciprocal cube root function.
      - ✓
      - ✓

    * - | ``double rhypot(double x, double y)``
        | Returns one over the square root of the sum of squares of two arguments.
      - ✓
      - ✓

    * - | ``double rint(double x)``
        | Round input to nearest integer value in floating-point.
      - ✓
      - ✓

    * - | ``double rnorm3d(double x, double y, double z)``
        | Returns one over the square root of the sum of squares of three coordinates of the argument.
      - ✓
      - ✓

    * - | ``double rnorm4d(double x, double y, double z, double w)``
        | Returns one over the square root of the sum of squares of four coordinates of the argument.
      - ✓
      - ✓

    * - | ``double rnorm(int dim, const double *a)``
        | Returns the reciprocal of square root of the sum of squares of any number of coordinates.
      - ✓
      - ✓

    * - | ``double scalbln(double x, long int n)``
        | Scale :math:`x` by :math:`2^n`.
      - ✓
      - ✓

    * - | ``double scalbn(double x, int n)``
        | Scale :math:`x` by :math:`2^n`.
      - ✓
      - ✓

    * - | ``bool signbit(double x)``
        | Return the sign bit of :math:`x`.
      - ✓
      - ✓

    * - | ``double sin(double x)``
        | Returns the sine of :math:`x`.
      - ✓
      - ✓

    * - | ``double sinh(double x)``
        | Returns the hyperbolic sine of :math:`x`.
      - ✓
      - ✓

    * - | ``double sinpi(double x)``
        | Returns the hyperbolic sine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``void sincos(double x, double *sptr, double *cptr)``
        | Returns the sine and cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``void sincospi(double x, double *sptr, double *cptr)``
        | Returns the sine and cosine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``double sqrt(double x)``
        | Returns the square root of :math:`x`.
      - ✓
      - ✓

    * - | ``double rsqrt(double x)``
        | Returns the reciprocal of the square root of :math:`x`.
      - ✗
      - ✓

    * - | ``double tan(double x)``
        | Returns the tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``double tanh(double x)``
        | Returns the hyperbolic tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``double tgamma(double x)``
        | Returns the gamma function of :math:`x`.
      - ✓
      - ✓

    * - | ``double trunc(double x)``
        | Truncate :math:`x` to the integral part.
      - ✓
      - ✓

    * - | ``double y0(double x)``
        | Returns the value of the Bessel function of the second kind of order 0 for :math:`x`.
      - ✓
      - ✓

    * - | ``double y1(double x)``
        | Returns the value of the Bessel function of the second kind of order 1 for :math:`x`.
      - ✓
      - ✓

    * - | ``double yn(int n, double x)``
        | Returns the value of the Bessel function of the second kind of order n for :math:`x`.
      - ✓
      - ✓

Integer intrinsics
--------------------------------------------------------------------------------------------

Following is the list of supported integer intrinsics. Note that intrinsics are supported on device only.

.. list-table:: Integer intrinsics mathematical functions

    * - **Function**

    * - | ``unsigned int __brev(unsigned int x)``
        | Reverse the bit order of a 32 bit unsigned integer.

    * - | ``unsigned long long int __brevll(unsigned long long int x)``
        | Reverse the bit order of a 64 bit unsigned integer.

    * - | ``unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int z)``
        | Return selected bytes from two 32-bit unsigned integers.

    * - | ``unsigned int __clz(int x)``
        | Return the number of consecutive high-order zero bits in 32 bit integer.

    * - | ``unsigned int __clzll(long long int x)``
        | Return the number of consecutive high-order zero bits in 64 bit integer.

    * - | ``unsigned int __ffs(int x)``
        | Find the position of least significant bit set to 1 in a 32 bit integer.

    * - | ``unsigned int __ffsll(long long int x)``
        | Find the position of least significant bit set to 1 in a 64 bit signed integer.

    * - | ``unsigned int __fns32(unsigned long long mask, unsigned int base, int offset)``
        | Find the position of the n-th set to 1 bit in a 32-bit integer.

    * - | ``unsigned int __fns64(unsigned long long int mask, unsigned int base, int offset)``
        | Find the position of the n-th set to 1 bit in a 64-bit integer.

    * - | ``unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)``
        | Concatenate :math:`hi` and :math:`lo`, shift left by shift & 31 bits, return the most significant 32 bits.

    * - | ``unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)``
        | Concatenate :math:`hi` and :math:`lo`, shift left by min(shift, 32) bits, return the most significant 32 bits.

    * - | ``unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)``
        | Concatenate :math:`hi` and :math:`lo`, shift right by shift & 31 bits, return the least significant 32 bits.

    * - | ``unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)``
        | Concatenate :math:`hi` and :math:`lo`, shift right by min(shift, 32) bits, return the least significant 32 bits.

    * - | ``unsigned int __hadd(int x, int y)``
        | Compute average of signed input arguments, avoiding overflow in the intermediate sum.

    * - | ``unsigned int __rhadd(int x, int y)``
        | Compute rounded average of signed input arguments, avoiding overflow in the intermediate sum.

    * - | ``unsigned int __uhadd(int x, int y)``
        | Compute average of unsigned input arguments, avoiding overflow in the intermediate sum.

    * - | ``unsigned int __urhadd (unsigned int x, unsigned int y)``
        | Compute rounded average of unsigned input arguments, avoiding overflow in the intermediate sum.

    * - | ``int __sad(int x, int y, int z)``
        | Returns :math:`|x - y| + z`, the sum of absolute difference.

    * - | ``unsigned int __usad(unsigned int x, unsigned int y, unsigned int z)``
        | Returns :math:`|x - y| + z`, the sum of absolute difference.

    * - | ``unsigned int __popc(unsigned int x)``
        | Count the number of bits that are set to 1 in a 32 bit integer.

    * - | ``unsigned int __popcll(unsigned long long int x)``
        | Count the number of bits that are set to 1 in a 64 bit integer.

    * - | ``int __mul24(int x, int y)``
        | Multiply two 24bit integers.

    * - | ``unsigned int __umul24(unsigned int x, unsigned int y)``
        | Multiply two 24bit unsigned integers.

    * - | ``int __mulhi(int x, int y)``
        | Returns the most significant 32 bits of the product of the two 32-bit integers.

    * - | ``unsigned int __umulhi(unsigned int x, unsigned int y)``
        | Returns the most significant 32 bits of the product of the two 32-bit unsigned integers.

    * - | ``long long int __mul64hi(long long int x, long long int y)``
        | Returns the most significant 64 bits of the product of the two 64-bit integers.

    * - | ``unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y)``
        | Returns the most significant 64 bits of the product of the two 64 unsigned bit integers.

The HIP-Clang implementation of ``__ffs()`` and ``__ffsll()`` contains code to add a constant +1 to produce the ``ffs`` result format.
For the cases where this overhead is not acceptable and programmer is willing to specialize for the platform,
HIP-Clang provides ``__lastbit_u32_u32(unsigned int input)`` and ``__lastbit_u32_u64(unsigned long long int input)``.
The index returned by ``__lastbit_`` instructions starts at -1, while for ``ffs`` the index starts at 0.

Floating-point Intrinsics
--------------------------------------------------------------------------------------------

Following is the list of supported floating-point intrinsics. Note that intrinsics are supported on device only.

.. note::

  Only the nearest even rounding mode supported on AMD GPUs by defaults. The ``_rz``, ``_ru`` and
  ``_rd`` suffixed intrinsic functions are existing in HIP AMD backend, if the
  ``OCML_BASIC_ROUNDED_OPERATIONS`` macro is defined.

.. list-table:: Single precision intrinsics mathematical functions

    * - **Function**

    * - | ``float __cosf(float x)``
        | Returns the fast approximate cosine of :math:`x`.

    * - | ``float __exp10f(float x)``
        | Returns the fast approximate for 10 :sup:`x`.

    * - | ``float __expf(float x)``
        | Returns the fast approximate for e :sup:`x`.

    * - | ``float __fadd_rn(float x, float y)``
        | Add two floating-point values in round-to-nearest-even mode.

    * - | ``float __fdiv_rn(float x, float y)``
        | Divide two floating point values in round-to-nearest-even mode.

    * - | ``float __fmaf_rn(float x, float y, float z)``
        | Returns ``x × y + z`` as a single operation in round-to-nearest-even mode.

    * - | ``float __fmul_rn(float x, float y)``
        | Multiply two floating-point values in round-to-nearest-even mode.

    * - | ``float __frcp_rn(float x, float y)``
        | Returns ``1 / x`` in round-to-nearest-even mode.

    * - | ``float __frsqrt_rn(float x)``
        | Returns ``1 / √x`` in round-to-nearest-even mode.

    * - | ``float __fsqrt_rn(float x)``
        | Returns ``√x`` in round-to-nearest-even mode.

    * - | ``float __fsub_rn(float x, float y)``
        | Subtract two floating-point values in round-to-nearest-even mode.

    * - | ``float __log10f(float x)``
        | Returns the fast approximate for base 10 logarithm of :math:`x`.

    * - | ``float __log2f(float x)``
        | Returns the fast approximate for base 2 logarithm of :math:`x`.

    * - | ``float __logf(float x)``
        | Returns the fast approximate for natural logarithm of :math:`x`.

    * - | ``float __powf(float x, float y)``
        | Returns the fast approximate of x :sup:`y`.

    * - | ``float __saturatef(float x)``
        | Clamp :math:`x` to [+0.0, 1.0].

    * - | ``float __sincosf(float x, float* sinptr, float* cosptr)``
        | Returns the fast approximate of sine and cosine of :math:`x`.

    * - | ``float __sinf(float x)``
        | Returns the fast approximate sine of :math:`x`.

    * - | ``float __tanf(float x)``
        | Returns the fast approximate tangent of :math:`x`.

.. list-table:: Double precision intrinsics mathematical functions

    * - **Function**

    * - | ``double __dadd_rn(double x, double y)``
        | Add two floating-point values in round-to-nearest-even mode.

    * - | ``double __ddiv_rn(double x, double y)``
        | Divide two floating-point values in round-to-nearest-even mode.

    * - | ``double __dmul_rn(double x, double y)``
        | Multiply two floating-point values in round-to-nearest-even mode.

    * - | ``double __drcp_rn(double x, double y)``
        | Returns ``1 / x`` in round-to-nearest-even mode.

    * - | ``double __dsqrt_rn(double x)``
        | Returns ``√x`` in round-to-nearest-even mode.

    * - | ``double __dsub_rn(double x, double y)``
        | Subtract two floating-point values in round-to-nearest-even mode.

    * - | ``double __fma_rn(double x, double y, double z)``
        | Returns ``x × y + z`` as a single operation in round-to-nearest-even mode.


Texture functions
===============================================

The supported texture functions are listed in ``texture_fetch_functions.h`` and
``texture_indirect_functions.h`` header files in the
`HIP-AMD backend repository <https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail>`_.

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

  .. code-block:: cpp

    clock_t clock()
    long long int clock64()

  The difference between the values that are returned represents the cycles used.

* Returning the wall clock count at a constant frequency on the device:

  .. code-block:: cpp

    long long int wall_clock64()

  This can be queried using the HIP API with the ``hipDeviceAttributeWallClockRate`` attribute of the
  device in HIP application code. For example:

  .. code-block:: cpp

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

* ``atomicAnd_system``: This function extends the atomic operation from the GPU device to other CPUs and GPU devices in the system.

HIP supports the following atomic operations.

.. list-table:: Atomic operations

    * - **Function**
      - **Supported in HIP**
      - **Supported in CUDA**

    * - ``int atomicAdd(int* address, int val)``
      - ✓
      - ✓

    * - ``int atomicAdd_system(int* address, int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicAdd(unsigned int* address,unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicAdd_system(unsigned int* address, unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicAdd(unsigned long long* address,unsigned long long val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicAdd_system(unsigned long long* address, unsigned long long val)``
      - ✓
      - ✓

    * - ``float atomicAdd(float* address, float val)``
      - ✓
      - ✓

    * - ``float atomicAdd_system(float* address, float val)``
      - ✓
      - ✓

    * - ``double atomicAdd(double* address, double val)``
      - ✓
      - ✓

    * - ``double atomicAdd_system(double* address, double val)``
      - ✓
      - ✓

    * - ``float unsafeAtomicAdd(float* address, float val)``
      - ✓
      - ✗

    * - ``float safeAtomicAdd(float* address, float val)``
      - ✓
      - ✗

    * - ``double unsafeAtomicAdd(double* address, double val)``
      - ✓
      - ✗

    * - ``double safeAtomicAdd(double* address, double val)``
      - ✓
      - ✗

    * - ``int atomicSub(int* address, int val)``
      - ✓
      - ✓

    * - ``int atomicSub_system(int* address, int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicSub(unsigned int* address,unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicSub_system(unsigned int* address, unsigned int val)``
      - ✓
      - ✓

    * - ``int atomicExch(int* address, int val)``
      - ✓
      - ✓

    * - ``int atomicExch_system(int* address, int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicExch(unsigned int* address,unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicExch_system(unsigned int* address, unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicExch(unsigned long long int* address,unsigned long long int val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val)``
      - ✓
      - ✓

    * - ``float atomicExch(float* address, float val)``
      - ✓
      - ✓

    * - ``int atomicMin(int* address, int val)``
      - ✓
      - ✓

    * - ``int atomicMin_system(int* address, int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicMin(unsigned int* address,unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicMin_system(unsigned int* address, unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicMin(unsigned long long* address,unsigned long long val)``
      - ✓
      - ✓

    * - ``int atomicMax(int* address, int val)``
      - ✓
      - ✓

    * - ``int atomicMax_system(int* address, int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicMax(unsigned int* address,unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicMax_system(unsigned int* address, unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicMax(unsigned long long* address,unsigned long long val)``
      - ✓
      - ✓

    * - ``unsigned int atomicInc(unsigned int* address)``
      - ✗
      - ✓

    * - ``unsigned int atomicDec(unsigned int* address)``
      - ✗
      - ✓

    * - ``int atomicCAS(int* address, int compare, int val)``
      - ✓
      - ✓

    * - ``int atomicCAS_system(int* address, int compare, int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicCAS(unsigned int* address,unsigned int compare,unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicCAS_system(unsigned int* address, unsigned int compare, unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicCAS(unsigned long long* address,unsigned long long compare,unsigned long long val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicCAS_system(unsigned long long* address, unsigned long long compare, unsigned long long val)``
      - ✓
      - ✓

    * - ``int atomicAnd(int* address, int val)``
      - ✓
      - ✓

    * - ``int atomicAnd_system(int* address, int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicAnd(unsigned int* address,unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicAnd_system(unsigned int* address, unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicAnd(unsigned long long* address,unsigned long long val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicAnd_system(unsigned long long* address, unsigned long long val)``
      - ✓
      - ✓

    * - ``int atomicOr(int* address, int val)``
      - ✓
      - ✓

    * - ``int atomicOr_system(int* address, int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicOr(unsigned int* address,unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicOr_system(unsigned int* address, unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicOr_system(unsigned int* address, unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicOr(unsigned long long int* address,unsigned long long val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicOr_system(unsigned long long* address, unsigned long long val)``
      - ✓
      - ✓

    * - ``int atomicXor(int* address, int val)``
      - ✓
      - ✓

    * - ``int atomicXor_system(int* address, int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicXor(unsigned int* address,unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned int atomicXor_system(unsigned int* address, unsigned int val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicXor(unsigned long long* address,unsigned long long val)``
      - ✓
      - ✓

    * - ``unsigned long long atomicXor_system(unsigned long long* address, unsigned long long val)``
      - ✓
      - ✓

Unsafe floating-point atomic RMW operations
----------------------------------------------------------------------------------------------------------------
Some HIP devices support fast atomic RMW operations on floating-point values. For example,
``atomicAdd`` on single- or double-precision floating-point values may generate a hardware RMW
instruction that is faster than emulating the atomic operation using an atomic compare-and-swap
(CAS) loop.

On some devices, fast atomic RMW instructions can produce results that differ from the same
functions implemented with atomic CAS loops. For example, some devices will use different rounding
or denormal modes, and some devices produce incorrect answers if fast floating-point atomic RMW
instructions target fine-grained memory allocations.

The HIP-Clang compiler offers a compile-time option, so you can choose fast--but potentially
unsafe--atomic instructions for your code. On devices that support these instructions, you can include
the ``-munsafe-fp-atomics`` option. This flag indicates to the compiler that all floating-point atomic
function calls are allowed to use an unsafe version, if one exists. For example, on some devices, this
flag indicates to the compiler that no floating-point ``atomicAdd`` function can target fine-grained
memory.

If you want to avoid using unsafe use a floating-point atomic RMW operations, you can use the
``-mno-unsafe-fp-atomics`` option. Note that the compiler default is to not produce unsafe
floating-point atomic RMW instructions, so the ``-mno-unsafe-fp-atomics`` option is not necessarily
required. However, passing this option to the compiler is good practice.

When you pass ``-munsafe-fp-atomics`` or ``-mno-unsafe-fp-atomics`` to the compiler's command line,
the option is applied globally for the entire compilation. Note that if some of the atomic RMW function
calls cannot safely use the faster floating-point atomic RMW instructions, you must use
``-mno-unsafe-fp-atomics`` in order to ensure that your atomic RMW function calls produce correct
results.

HIP has four extra functions that you can use to more precisely control which floating-point atomic
RMW functions produce unsafe atomic RMW instructions:

* ``float unsafeAtomicAdd(float* address, float val)``
* ``double unsafeAtomicAdd(double* address, double val)`` (Always produces fast atomic RMW
  instructions on devices that have them, even when ``-mno-unsafe-fp-atomics`` is used)
* `float safeAtomicAdd(float* address, float val)`
* ``double safeAtomicAdd(double* address, double val)`` (Always produces safe atomic RMW
  operations, even when ``-munsafe-fp-atomics`` is used)

.. _warp-cross-lane:

Warp cross-lane functions
========================================================

Threads in a warp are referred to as ``lanes`` and are numbered from ``0`` to ``warpSize - 1``.
Warp cross-lane functions operate across all lanes in a warp. The hardware guarantees that all warp
lanes will execute in lockstep, so additional synchronization is unnecessary, and the instructions
use no shared memory.

Note that NVIDIA and AMD devices have different warp sizes. You can use ``warpSize`` built-ins in you
portable code to query the warp size.

.. tip::
  Be sure to review HIP code generated from the CUDA path to ensure that it doesn't assume a
  ``waveSize`` of 32. "Wave-aware" code that assumes a ``waveSize`` of 32 can run on a wave-64
  machine, but it only utilizes half of the machine's resources.

To get the default warp size of a GPU device, use ``hipGetDeviceProperties`` in you host functions.

.. code-block:: cpp

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceID);
  int w = props.warpSize;
    // implement portable algorithm based on w (rather than assume 32 or 64)

Only use ``warpSize`` built-ins in device functions, and don't assume ``warpSize`` to be a compile-time
constant.

Note that assembly kernels may be built for a warp size that is different from the default.
All mask values either returned or accepted by these builtins are 64-bit
unsigned integer values, even when compiled for a wave-32 device, where all the
higher bits are unused. CUDA code ported to HIP requires changes to ensure that
the correct type is used.

Note that the ``__sync`` variants are made available in ROCm 6.2, but disabled by
default to help with the transition to 64-bit masks. They can be enabled by
setting the preprocessor macro ``HIP_ENABLE_WARP_SYNC_BUILTINS``. These builtins
will be enabled unconditionally in ROCm 6.3. Wherever possible, the
implementation includes a static assert to check that the program source uses
the correct type for the mask.

Warp vote and ballot functions
-------------------------------------------------------------------------------------------------------------

.. code-block:: cpp

  int __all(int predicate)
  int __any(int predicate)
  unsigned long long __ballot(int predicate)
  unsigned long long __activemask()

  int __all_sync(unsigned long long mask, int predicate)
  int __any_sync(unsigned long long mask, int predicate)
  unsigned long long __ballot_sync(unsigned long long mask, int predicate)

You can use ``__any`` and ``__all`` to get a summary view of the predicates evaluated by the
participating lanes.

* ``__any()``: Returns 1 if the predicate is non-zero for any participating lane, otherwise it returns 0.

* ``__all()``: Returns 1 if the predicate is non-zero for all participating lanes, otherwise it returns 0.

To determine if the target platform supports the any/all instruction, you can use the ``hasWarpVote``
device property or the ``HIP_ARCH_HAS_WARP_VOTE`` compiler definition.

``__ballot`` returns a bit mask containing the 1-bit predicate value from each
lane. The nth bit of the result contains the 1 bit contributed by the nth warp
lane.

``__activemask()`` returns a bit mask of currently active warp lanes. The nth bit
of the result is 1 if the nth warp lane is active.

Note that the ``__ballot`` and ``__activemask`` builtins in HIP have a 64-bit return
value (unlike the 32-bit value returned by the CUDA builtins). Code ported from
CUDA should be adapted to support the larger warp sizes that the HIP version
requires.

Applications can test whether the target platform supports the ``__ballot`` or
``__activemask`` instructions using the ``hasWarpBallot`` device property in host
code or the ``HIP_ARCH_HAS_WARP_BALLOT`` macro defined by the compiler for device
code.

The ``_sync`` variants require a 64-bit unsigned integer mask argument that
specifies the lanes in the warp that will participate in cross-lane
communication with the calling lane. Each participating thread must have its own
bit set in its mask argument, and all active threads specified in any mask
argument must execute the same call with the same mask, otherwise the result is
undefined.

Warp match functions
-------------------------------------------------------------------------------------------------------------

.. code-block:: cpp

  unsigned long long __match_any(T value)
  unsigned long long __match_all(T value, int *pred)

  unsigned long long __match_any_sync(unsigned long long mask, T value)
  unsigned long long __match_all_sync(unsigned long long mask, T value, int *pred)

``T`` can be a 32-bit integer type, 64-bit integer type or a single precision or
double precision floating point type.

``__match_any`` returns a bit mask containing a 1-bit for every participating lane
if and only if that lane has the same value in ``value`` as the current lane, and
a 0-bit for all other lanes.

``__match_all`` returns a bit mask containing a 1-bit for every participating lane
if and only if they all have the same value in ``value`` as the current lane, and
a 0-bit for all other lanes. The predicate ``pred`` is set to true if and only if
all participating threads have the same value in ``value``.

The ``_sync`` variants require a 64-bit unsigned integer mask argument that
specifies the lanes in the warp that will participate in cross-lane
communication with the calling lane. Each participating thread must have its own
bit set in its mask argument, and all active threads specified in any mask
argument must execute the same call with the same mask, otherwise the result is
undefined.

Warp shuffle functions
-------------------------------------------------------------------------------------------------------------

The default width is ``warpSize`` (see :ref:`warp-cross-lane`). Half-float shuffles are not supported.

.. code-block:: cpp

  T __shfl      (T var, int srcLane, int width=warpSize);
  T __shfl_up   (T var, unsigned int delta, int width=warpSize);
  T __shfl_down (T var, unsigned int delta, int width=warpSize);
  T __shfl_xor  (T var, int laneMask, int width=warpSize);

  T __shfl_sync      (unsigned long long mask, T var, int srcLane, int width=warpSize);
  T __shfl_up_sync   (unsigned long long mask, T var, unsigned int delta, int width=warpSize);
  T __shfl_down_sync (unsigned long long mask, T var, unsigned int delta, int width=warpSize);
  T __shfl_xor_sync  (unsigned long long mask, T var, int laneMask, int width=warpSize);

``T`` can be a 32-bit integer type, 64-bit integer type or a single precision or
double precision floating point type.

The ``_sync`` variants require a 64-bit unsigned integer mask argument that
specifies the lanes in the warp that will participate in cross-lane
communication with the calling lane. Each participating thread must have its own
bit set in its mask argument, and all active threads specified in any mask
argument must execute the same call with the same mask, otherwise the result is
undefined.

Cooperative groups functions
==============================================================

You can use cooperative groups to synchronize groups of threads. Cooperative groups also provide a
way of communicating between groups of threads at a granularity that is different from the block.

HIP supports the following kernel language cooperative groups types and functions:

.. list-table:: Cooperative groups functions

    * - **Function**
      - **Supported in HIP**
      - **Supported in CUDA**

    * - ``void thread_group.sync();``
      - ✓
      - ✓

    * - ``unsigned thread_group.size();``
      - ✓
      - ✓

    * - ``unsigned thread_group.thread_rank()``
      - ✓
      - ✓

    * - ``bool thread_group.is_valid();``
      - ✓
      - ✓

    * - ``grid_group this_grid()``
      - ✓
      - ✓

    * - ``void grid_group.sync()``
      - ✓
      - ✓

    * - ``unsigned grid_group.size()``
      - ✓
      - ✓

    * - ``unsigned grid_group.thread_rank()``
      - ✓
      - ✓

    * - ``bool grid_group.is_valid()``
      - ✓
      - ✓

    * - ``multi_grid_group this_multi_grid()``
      - ✓
      - ✓

    * - ``void multi_grid_group.sync()``
      - ✓
      - ✓

    * - ``unsigned multi_grid_group.size()``
      - ✓
      - ✓

    * - ``unsigned multi_grid_group.thread_rank()``
      - ✓
      - ✓

    * - ``bool multi_grid_group.is_valid()``
      - ✓
      - ✓

    * - ``unsigned multi_grid_group.num_grids()``
      - ✓
      - ✓

    * - ``unsigned multi_grid_group.grid_rank()``
      - ✓
      - ✓

    * - ``thread_block this_thread_block()``
      - ✓
      - ✓

    * - ``multi_grid_group this_multi_grid()``
      - ✓
      - ✓

    * - ``void multi_grid_group.sync()``
      - ✓
      - ✓

    * - ``void thread_block.sync()``
      - ✓
      - ✓

    * - ``unsigned thread_block.size()``
      - ✓
      - ✓

    * - ``unsigned thread_block.thread_rank()``
      - ✓
      - ✓

    * - ``bool thread_block.is_valid()``
      - ✓
      - ✓

    * - ``dim3 thread_block.group_index()``
      - ✓
      - ✓

    * - ``dim3 thread_block.thread_index()``
      - ✓
      - ✓

Warp matrix functions
============================================================

Warp matrix functions allow a warp to cooperatively operate on small matrices that have elements
spread over lanes in an unspecified manner.

HIP does not support kernel language warp matrix types or functions.

.. list-table:: Warp matrix functions

    * - **Function**
      - **Supported in HIP**
      - **Supported in CUDA**

    * - ``void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned lda)``
      - ✗
      - ✓

    * - ``void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned lda, layout_t layout)``
      - ✗
      - ✓

    * - ``void store_matrix_sync(T* mptr, fragment<...> &a,  unsigned lda, layout_t layout)``
      - ✗
      - ✓

    * - ``void fill_fragment(fragment<...> &a, const T &value)``
      - ✗
      - ✓

    * - ``void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c , bool sat)``
      - ✗
      - ✓

Independent thread scheduling
============================================================

Certain architectures that support CUDA allow threads to progress independently of each other. This
independent thread scheduling makes intra-warp synchronization possible.

HIP does not support this type of scheduling.

Profiler Counter Function
============================================================

The CUDA ``__prof_trigger()`` instruction is not supported.

Assert
============================================================

The assert function is supported in HIP.
Assert function is used for debugging purpose, when the input expression equals to zero, the execution will be stopped.

.. code-block:: cpp

  void assert(int input)

There are two kinds of implementations for assert functions depending on the use sceneries,
- One is for the host version of assert, which is defined in ``assert.h``,
- Another is the device version of assert, which is implemented in ``hip/hip_runtime.h``.
Users need to include ``assert.h`` to use ``assert``. For assert to work in both device and host functions, users need to include ``"hip/hip_runtime.h"``.

HIP provides the function ``abort()`` which can be used to terminate the application when terminal failures are detected. It is implemented using the ``__builtin_trap()`` function.

This function produces a similar effect of using ``asm("trap")`` in the CUDA code.

.. note::

  In HIP, the function terminates the entire application, while in CUDA, ``asm("trap")`` only terminates the dispatch and the application continues to run.


``printf``
============================================================

``printf`` function is supported in HIP.
The following is a simple example to print information in the kernel.

.. code-block:: cpp

  #include <hip/hip_runtime.h>

  __global__ void run_printf() { printf("Hello World\n"); }

  int main() {
    run_printf<<<dim3(1), dim3(1), 0, 0>>>();
  }


Device-Side Dynamic Global Memory Allocation
============================================================

Device-side dynamic global memory allocation is under development. HIP now includes a preliminary
implementation of malloc and free that can be called from device functions.

``__launch_bounds__``
============================================================

GPU multiprocessors have a fixed pool of resources (primarily registers and shared memory) which are shared by the actively running warps. Using more resources can increase IPC of the kernel but reduces the resources available for other warps and limits the number of warps that can be simultaneously running. Thus GPUs have a complex relationship between resource usage and performance.

``__launch_bounds__`` allows the application to provide usage hints that influence the resources (primarily registers) used by the generated code. It is a function attribute that must be attached to a __global__ function:

.. code-block:: cpp

  __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)
  MyKernel(hipGridLaunch lp, ...)
  ...

``__launch_bounds__`` supports two parameters:
- MAX_THREADS_PER_BLOCK - The programmers guarantees that kernel will be launched with threads less than MAX_THREADS_PER_BLOCK. (On NVCC this maps to the ``.maxntid`` PTX directive). If no launch_bounds is specified, MAX_THREADS_PER_BLOCK is the maximum block size supported by the device (typically 1024 or larger). Specifying MAX_THREADS_PER_BLOCK less than the maximum effectively allows the compiler to use more resources than a default unconstrained compilation that supports all possible block sizes at launch time.
The threads-per-block is the product of (``blockDim.x * blockDim.y * blockDim.z``).
- MIN_WARPS_PER_EXECUTION_UNIT - directs the compiler to minimize resource usage so that the requested number of warps can be simultaneously active on a multi-processor. Since active warps compete for the same fixed pool of resources, the compiler must reduce resources required by each warp(primarily registers). MIN_WARPS_PER_EXECUTION_UNIT is optional and defaults to 1 if not specified. Specifying a MIN_WARPS_PER_EXECUTION_UNIT greater than the default 1 effectively constrains the compiler's resource usage.

When launch kernel with HIP APIs, for example, ``hipModuleLaunchKernel()``, HIP will do validation to make sure input kernel dimension size is not larger than specified launch_bounds.
In case exceeded, HIP would return launch failure, if AMD_LOG_LEVEL is set with proper value (for details, please refer to ``docs/markdown/hip_logging.md``), detail information will be shown in the error log message, including
launch parameters of kernel dim size, launch bounds, and the name of the faulting kernel. It's helpful to figure out which is the faulting kernel, besides, the kernel dim size and launch bounds values will also assist in debugging such failures.

Compiler Impact
--------------------------------------------------------------------------------------------

The compiler uses these parameters as follows:
- The compiler uses the hints only to manage register usage, and does not automatically reduce shared memory or other resources.
- Compilation fails if compiler cannot generate a kernel which meets the requirements of the specified launch bounds.
- From MAX_THREADS_PER_BLOCK, the compiler derives the maximum number of warps/block that can be used at launch time.
Values of MAX_THREADS_PER_BLOCK less than the default allows the compiler to use a larger pool of registers : each warp uses registers, and this hint constrains the launch to a warps/block size which is less than maximum.
- From MIN_WARPS_PER_EXECUTION_UNIT, the compiler derives a maximum number of registers that can be used by the kernel (to meet the required #simultaneous active blocks).
If MIN_WARPS_PER_EXECUTION_UNIT is 1, then the kernel can use all registers supported by the multiprocessor.
- The compiler ensures that the registers used in the kernel is less than both allowed maximums, typically by spilling registers (to shared or global memory), or by using more instructions.
- The compiler may use heuristics to increase register usage, or may simply be able to avoid spilling. The MAX_THREADS_PER_BLOCK is particularly useful in this cases, since it allows the compiler to use more registers and avoid situations where the compiler constrains the register usage (potentially spilling) to meet the requirements of a large block size that is never used at launch time.

CU and EU Definitions
--------------------------------------------------------------------------------------------

A compute unit (CU) is responsible for executing the waves of a work-group. It is composed of one or more execution units (EU) which are responsible for executing waves. An EU can have enough resources to maintain the state of more than one executing wave. This allows an EU to hide latency by switching between waves in a similar way to symmetric multithreading on a CPU. In order to allow the state for multiple waves to fit on an EU, the resources used by a single wave have to be limited. Limiting such resources can allow greater latency hiding, but can result in having to spill some register state to memory. This attribute allows an advanced developer to tune the number of waves that are capable of fitting within the resources of an EU. It can be used to ensure at least a certain number will fit to help hide latency, and can also be used to ensure no more than a certain number will fit to limit cache thrashing.

Porting from CUDA ``__launch_bounds``
--------------------------------------------------------------------------------------------

CUDA defines a ``__launch_bounds`` which is also designed to control occupancy:

.. code-block:: cpp

  __launch_bounds(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)

- The second parameter ``__launch_bounds`` parameters must be converted to the format used __hip_launch_bounds, which uses warps and execution-units rather than blocks and multi-processors (this conversion is performed automatically by HIPIFY tools).

.. code-block:: cpp

  MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK) / 32

The key differences in the interface are:
- Warps (rather than blocks):
The developer is trying to tell the compiler to control resource utilization to guarantee some amount of active Warps/EU for latency hiding. Specifying active warps in terms of blocks appears to hide the micro-architectural details of the warp size, but makes the interface more confusing since the developer ultimately needs to compute the number of warps to obtain the desired level of control.
- Execution Units (rather than multiprocessor):
The use of execution units rather than multiprocessors provides support for architectures with multiple execution units/multi-processor. For example, the AMD GCN architecture has 4 execution units per multiprocessor. The ``hipDeviceProps`` has a field ``executionUnitsPerMultiprocessor``.
Platform-specific coding techniques such as ``#ifdef`` can be used to specify different launch_bounds for NVCC and HIP-Clang platforms, if desired.

``maxregcount``
--------------------------------------------------------------------------------------------

Unlike NVCC, HIP-Clang does not support the ``--maxregcount`` option. Instead, users are encouraged to use the hip_launch_bounds directive since the parameters are more intuitive and portable than
micro-architecture details like registers, and also the directive allows per-kernel control rather than an entire file. hip_launch_bounds works on both HIP-Clang and NVCC targets.

Asynchronous Functions
============================================================

Memory stream
--------------------------------------------------------------------------------------------

.. doxygengroup:: Stream
   :content-only:

.. doxygengroup:: StreamO
   :content-only:

Peer to peer
--------------------------------------------------------------------------------------------

.. doxygengroup:: PeerToPeer
   :content-only:

Memory management
--------------------------------------------------------------------------------------------

.. doxygengroup:: Memory
   :content-only:

External Resource Interoperability
--------------------------------------------------------------------------------------------

.. doxygengroup:: External
   :content-only:

Register Keyword
============================================================

The register keyword is deprecated in C++, and is silently ignored by both NVCC and HIP-Clang. You can pass the option ``-Wdeprecated-register`` the compiler warning message.

Pragma Unroll
============================================================

Unroll with a bounds that is known at compile-time is supported. For example:

.. code-block:: cpp

  #pragma unroll 16 /* hint to compiler to unroll next loop by 16 */
  for (int i=0; i<16; i++) ...

.. code-block:: cpp

  #pragma unroll 1  /* tell compiler to never unroll the loop */
  for (int i=0; i<16; i++) ...

.. code-block:: cpp

  #pragma unroll /* hint to compiler to completely unroll next loop. */
  for (int i=0; i<16; i++) ...

In-Line Assembly
============================================================

GCN ISA In-line assembly is supported.

There are some usage limitations in ROCm compiler for inline asm support, please refer to `Inline ASM statements <https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocmcc.html#inline-asm-statements>`_ for details.

Users can get related background resources on `how to use inline assembly <https://gcc.gnu.org/onlinedocs/gcc/extensions-to-the-c-language-family/how-to-use-inline-assembly-language-in-c-code.html>`_ for any usage of inline assembly features.

A short example program including an inline assembly statement can be found at `inline asm tutorial <https://github.com/ROCm/hip-tests/tree/amd-staging/samples/2_Cookbook/10_inline_asm>`_.

For further usage of special AMD GPU hardware features that are available through assembly, please refer to the ISA manual for `AMDGPU usage <https://llvm.org/docs/AMDGPUUsage.html#additional-documentation>`_, in which AMD GCN is listed from gfx906 to RDNA 3.5.

C++ Support
============================================================

The following C++ features are not supported:

* Run-time-type information (RTTI)
* Try/catch

Partially supported features:

* Virtual functions

Virtual functions are not supported if objects containing virtual function tables are passed between GPU's of different offload arch's, e.g. between gfx906 and gfx1030. Otherwise virtual functions are supported.

Kernel Compilation
============================================================

hipcc now supports compiling C++/HIP kernels to binary code objects.
The file format for binary is ``.co`` which means Code Object. The following command builds the code object using ``hipcc``.

.. code-block:: bash

  hipcc --genco --offload-arch=[TARGET GPU] [INPUT FILE] -o [OUTPUT FILE]

  [TARGET GPU] = GPU architecture
  [INPUT FILE] = Name of the file containing kernels
  [OUTPUT FILE] = Name of the generated code object file

.. note::

  When using binary code objects is that the number of arguments to the kernel is different on HIP-Clang and NVCC path. Refer to the `HIP module_api sample <https://github.com/ROCm/hip-tests/tree/develop/samples/0_Intro/module_api>`_ for differences in the arguments to be passed to the kernel.

gfx-arch-specific-kernel
============================================================

Clang defined '__gfx*__' macros can be used to execute gfx arch specific codes inside the kernel. Refer to the sample in `HIP 14_gpu_arch sample <https://github.com/ROCm/hip-tests/tree/develop/samples/2_Cookbook/14_gpu_arch>`_.
