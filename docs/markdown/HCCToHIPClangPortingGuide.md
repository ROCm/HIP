# HIP/HCC To HIP-Clang Porting Guide

For most HIP programs the migration from HCC to HIP-Clang should be transparent
as hipcc and HIP cmake files hide the differences by translating the compilation
options.

Issues may occur due to HIP-Clang has stricter syntax and semantic checks for
HIP programs and due to difference in some default options. This document
lists the common issues when migrating from HCC to HIP-Clang and their solutions.

## No matching function call. Call to `__host__` function from `__device__` function.

In HCC device functions can call non-device functions.

In HIP-Clang a function by default is a host function, unless it is a constexpr
function. Device functions can only call device or host device functions.	

To fix this, add `__device__` to device functions. Or if the function is in a header
file that you cannot change, add this before and after your include to mark
all functions in the header file as both host and device without changing
the header file.

```C++
#pragma clang force_cuda_host_device begin

#include ...

#pragma clang force_cuda_host_device end
```

## Parameters cannot have `__fp16` type.

In HCC functions can have `__fp16` type by default.

In HIP-Clang Functions cannot have `__fp16` type by default.	

Fix: Replace `__fp16` with `_Float16` or `__half` (recommended).

Or add option `-fallow-half-arguments-and-returns` for clang (not recommended).

## Dynamic initialization is not supported for `__device__`, `__constant__`, and `__shared__` variables.

For example

```C++
struct A {
    //constexpr A():a(1) {}
    A() { a = 1;}
    int a;
};
__constant__ A a;
```

HCC does not diagnose this. HIP-Clang diagnose this.

Some of the data members (or data members of its data members, etc.)
contains non-trivial default ctor.

Since the global variable needs to be initialized before any kernel launching,
this requires the ctors of these data members to be called on device side
before any kernel launching. This is not supported by either HCC, CUDA or HIP.

HCC does not diagnose this issue. The global variable will ends up as not
initialized and run time error will likely occur.

HIP-Clang and nvcc diagnose such situations. This avoids the error at run time.

As a simple example to demonstrate how nvcc diagnose this:

https://godbolt.org/z/J32NfM 

To fix this issue, all the user-defined default ctors of the data members
(and data member of data members, etc) of that global variables need to
be re-written to be constexpr ctor, by following this example

https://godbolt.org/z/6YPwQM 

## Floating point result changes

The default fp contract option is different in HIP-Clang and HCC.

HCC by default assumes `-ffp-contract=off`.

HIP-Clang by default assumes `-ffp-contract=fast`, which is the default
option for cuda-clang and nvcc.

`-ffp-contract=fast` allows compiler to use fma to replace
add/multiplication operations. fma does not round the intermediate
results, therefore it can get more accurate result and is faster.

If a portion of a program relies on rounding of intermediate result,
it needs to force fp contract off for that portion of the code.
To achieve this the user can either pass the `-ffp-contract=off`
option to the compiler, enclose that part of code with
`#pragma STDC FP_CONTRACT OFF` and `#pragma STDC FP_CONTRACT ON`.

## No matching function for a kernel when expanding a call of hipLaunchKernelGGL due to function argument mismatch for the kernel.

In HCC the type checking for kernel arguments passed to hipLaunchKernelGGL
are relaxed.

In HIP-Clang the kernel arguments passed to hipLaunchKernelGGL are checked
against the defintion of the kernel by following the normal C++ type checking
rules.

Fix kernel argument type mismatch.

## No viable conversion from returned value of type 'double' to function return type 'Acc<double>'

For a simple test case:

```C++
#include "hip/hip_runtime.h"
#define ATTR __host__ __device__
template<typename S>
struct Acc {
 S x;
 ATTR operator S() const { return x;}
};
template<typename T>
ATTR Acc<T> foo(T x) {
 return x; // this line should fail
 //return {x}; // Fix for this issue
}
void foo2() {
 foo(1.0);
}
```

This issues especially happens when HIP vector type is used with template
functions where a scalar type is returned as a ScalarAccessor type.

HCC inserts conversions for `__host__ __device__` template functions even
though user defined constructors for such conversion do not exist. HCC
will generate constructors for such conversions and insert it where it
is needed. HCC only does this for `__host__ __device__` template functions
and only if the class/struct has other `__host__ __device__` member functions.
`
HIP-Clang does not generate such constructors if they are not defined by user.	
User needs to insert conversions explicitly.

For HIP vector type, since defining conversion constructor in ScalarAccessor
type can cause ambiguity. It is not desired to define conversion constructor
in ScalarAccessor type. Instead, users are expected to insert conversions
where they are needed, by using list initialization. For example, in the
test case given, return {x} instead of x.

