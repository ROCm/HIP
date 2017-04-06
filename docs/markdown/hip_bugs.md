# HIP Bugs 

<!-- toc -->

- [Errors related to undefined reference to `__hcLaunchKernel__***__grid_launch_parm**](#errors-related-to-undefined-reference-to-hclaunchkernel__grid_launch_parm)
- [Application hangs after a hipLaunchKernel call](#what-if-i-see-application-hangs-after-a-hiplaunchkernel-call)
- [What is the current limitation of HIP Generic Grid Launch method?](#what-is-the-current-limitation-of-hip-generic-grid-launch-method)
- [HIP is more restrictive in enforcing restrictions](#hip-is-more-restrictive-in-enforcing-restrictions)

<!-- tocstop -->

### Errors related to undefined reference to `__hcLaunchKernel__***__grid_launch_parm**

Some common code practices may lead to hipcc generating a error with the form :
undefined reference to `__hcLaunchKernel__ZN15vecAddNamespace6vecAddIidEEv16grid_launch_parmPT0_S3_S3_T_

To workaround, try:
- Avoid calling hipLaunchKernel from a function with the __host__ attribute
```
__host__ MyFunc(…) {
hipLaunchKernel(myKernel, …)
```
- Avoid use of static with kernel definition:
static __global__ MyKernel 
- Avoid defining kernels in anonymous namespace
namespace {
__global__ MyKernel …
- Avoid calling member functions 


### What is the current limitation of HIP Generic Grid Launch method?
1. __global__ functions cannot be marked as static or put in an unnamed namespace i.e. they cannot be given internal linkage (this would clash with __attribute__((weak)));
2. using the macro based dispatch mechanism i.e. hipLaunchKernel* only works for functions that take no more than 20 arguments (this limit can be increased up to 126, and is temporary until we can enable C++14 mode and use variadic generic lambdas); no such limitation applies do dispatching directly through grid_launch.


### Errors related to `no matching constructor`

The symptom is the compiler would complain about errors like `no matching constructor` for classes/structs passed as arguments into a GPU kernel. Often, this is caused by a design limitation in HCC where array-typed member variables inside a class/struct can’t be correctly passed into GPU kernels. To mitigate this issue, a custom serializer/deserializer pair is provided.

For example, `Foo` in the code snippets below contains an array-typed member variable `table`, which would fail the compiler if used as a kernel argument.

```
struct Foo {
  // table is an array, which makes foo
  int table[3];
};
```

An workaround is to provide a custom serializer on CPU side, and append the contents of the array as kernel arguments:

```

struct Foo {
  int table[3];

  // user-provided CPU serializer
  // must append the contents of the array member as kernel arguments
#ifdef __HCC__
  __attribute__((annotate(“serialize”)))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 3; ++i)
      s.Append(sizeof(int), &table[i]);
  }
#endif
};
```

Then, provide a custom deserializer on GPU side, to help reconstruct the array within GPU kernels. Notice that the deserializer can not be a function template, and should have scalar-typed parameters of the number equals to the length of the array-typed member variable. For example:

```
struct Foo {
  int table[3];

  // user-provided GPU deserializer
  // table has 3 int elements, so deserializer must have 3 int parameters.
#ifdef __HCC__
  __attribute__((annotate(“user_deserialize”)))
  Foo(int x0, int x1, int x2) [[cpu]][[hc]] {
    table[0] = x0;
    table[1] = x1;
    table[2] = x2;
  }
#endif

#ifdef __HCC__
  __attribute__((annotate(“serialize”)))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    s.Append(sizeof(int), &table[0]);
    s.Append(sizeof(int), &table[1]);
    s.Append(sizeof(int), &table[2]);
  }
#endif
};
```


Rather than create serializer functions, another workaround is to pass the member fields from the structure as simple data types.


### HIP is more restrictive in enforcing restrictions
The language specification for HIP and CUDA forbid calling a
`__device__` function in a `__host__` context. In practice, you may observe
differences in the strictness of this restriction, with HIP exhibiting a tighter
adherence to the specification and thus less tolerant of infringing code. The
solution is to ensure that all functions which are called in a
`__device__` context are correctly annotated to reflect it. An interesting case
where these differences emerge is shown below.  This relies on a the common 
[C++ Member Detector idiom][1], as it would be implemented pre C++11):  

```c++
#include <cassert>
#include <type_traits>

struct aye { bool a[1]; };
struct nay { bool a[2]; };

// Dual restriction is necessary in HIP if the detector is to work for
// __device__ contexts as well as __host__ ones. NVCC is less strict.
template<typename T>
__host__ __device__
const T& cref_t();

template<typename T>
struct Has_call_operator {
    // Dual restriction is necessary in HIP if the detector is to work for
    // __device__ contexts as well as __host__ ones. NVCC is less strict.
    template<typename C>
    __host__ __device__
    static
    aye test(
        C const *,
        typename std::enable_if<
            (sizeof(cref_t<C>().operator()()) > 0)>::type* = nullptr);
    static
    nay test(...);

    enum { value = sizeof(test(static_cast<T*>(0))) == sizeof(aye) };
};

template<typename T, typename U, bool callable = has_call_operator<U>::value>
struct Wrapper {
    template<typename V>
    V f() const { return T{1}; }
};


template<typename T, typename U>
struct Wrapper<T, U, true> {
    template<typename V>
    V f() const { return T{10}; }
};

// This specialisation will yield a compile-time error, if selected.
template<typename T, typename U>
struct Wrapper<T, U, false> {};

template<typename T>
struct Functor;

template<> struct Functor<float> {
    __device__
    float operator()() const { return 42.0f; }
};

__device__
void this_will_not_compile_if_detector_is_not_marked_device()
{
    float f = Wrapper<float, Functor<float>>().f<float>();
}

__host__
void this_will_not_compile_if_detector_is_marked_device_only()
{
    float f = Wrapper<float, Functor<float>>().f<float>();
}
```
[1]: https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Member_Detector
