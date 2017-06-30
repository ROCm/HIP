# HIP Bugs

<!-- toc -->

- [Errors related to undefined reference to `__hcLaunchKernel__***__grid_launch_parm**`](#errors-related-to-undefined-reference-to-__hclaunchkernel____grid_launch_parm)
- [Can't find kernels inside dynamic linked library](#cant-find-kernels-inside-dynamic-linked-library)
- [What is the current limitation of HIP Generic Grid Launch method?](#what-is-the-current-limitation-of-hip-generic-grid-launch-method)
- [Errors related to `no matching constructor`](#errors-related-to-no-matching-constructor)
- [HIP is more restrictive in enforcing restrictions](#hip-is-more-restrictive-in-enforcing-restrictions)

<!-- tocstop -->

### Errors related to undefined reference to `__hcLaunchKernel__***__grid_launch_parm**`

Some common code practices may lead to hipcc generating a error with the form :
```
undefined reference to `__hcLaunchKernel__ZN15vecAddNamespace6vecAddIidEEv16grid_launch_parmPT0_S3_S3_T_
```
Or:
```
error: weak declaration cannot have internal linkage
```

Suggested workarounds:
- Avoid use of static with kernel definition:
```c++
static __global__ MyKernel 
```

- Avoid defining kernels in anonymous namespace :
```c++
namespace {
  __global__ MyKernel
}
```

### Can't find kernels inside dynamic linked library

HCC requires use of the "-Bdynamic" flag when creating a dynamic library which contains kernels.  The dynamic flag causes the symbols to be created with a signature which allows HCC to discover and load the kernels in the dynamic library.   This flag is often not set by default and must be added to the link step of the library.  If not done, HCC will be unable to find the kernels defined in the library, and will emit a message such as:

```
HSADevice::CreateKernel(): Unable to create kernel"
```

To correct, add the following flag to hcc or hipcc:
```
$ hipcc -Wl,-Bsymbolic ...
```

Ensure there is no space in the "Wl,-Bsymbolic" option.


### What is the current limitation of HIP Generic Grid Launch method?
1. __global__ functions cannot be marked as static or put in an unnamed namespace i.e. they cannot be given internal linkage (this would clash with __attribute__((weak)));
2. using the macro based dispatch mechanism i.e. hipLaunchKernel* only works for functions that take no more than 20 arguments (this limit can be increased up to 126, and is temporary until we can enable C++14 mode and use variadic generic lambdas); no such limitation applies do dispatching directly through grid_launch.


### Errors related to `no matching constructor`

The symptom is the compiler would complain about errors like `no matching constructor` for classes/structs passed as arguments into a GPU kernel. Often, this is caused by a design limitation in HCC where array-typed member variables inside a class/struct can’t be correctly passed into GPU kernels. To mitigate this issue, a custom serializer/deserializer pair is provided.

For example, `Foo` in the code snippets below contains an array-typed member variable `table`, which would fail the compiler if used as a kernel argument.

```
struct Foo {
  float  _data;
  // table is an array, which makes foo
  int table[3];
};
```

A workaround is to provide a custom serializer on host side which appends the contents of the array as kernel arguments, and a custome deserializaer on the device path to reconstruct the array inside the GPU kernels.
The deserializer can not be a function template, and should have scalar-typed parameters of the number equals to the length of the array-typed member variable. For example:

```

struct Foo {
  float  _data;
  int   _table[3];


#ifdef __HCC__
  // user-provided CPU serializer
  // Append the contents of the array member as kernel arguments
  __attribute__((annotate(“serialize”)))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    s.Append(sizeof(float), &_data);
    for (int i = 0; i < 3; ++i)
      s.Append(sizeof(int), &_table[i]);
  }


  // user-provided GPU deserializer
  // table has 3 int elements, so deserializer must have 3 int parameters.
  __attribute__((annotate(“user_deserialize”)))
  Foo(float d, int x0, int x1, int x2) [[cpu]][[hc]] {
    _data = d;
    _table[0] = x0;
    _table[1] = x1;
    _table[2] = x2;
  }

#endif
};
```


Rather than create serializer functions, another workaround is to pass the member fields from the structure as simple data types.
Note a class or struct can contain only one "user_deserialize" constructor.
For types which contain arrays which are based on template parameter, you can use partial template instantiation to implement one constructor per specialization.
However, an easier approach may be to create one user_deserializer which processes the maximum supported dimension.
This will take more memory in the structure and also require additional kernel arguments, but this may have little performance impact and the conversion is easier than partial template specialization.  An example:

```
#define MAX_Dim 4
template<typename T, int Dim> struct MyArray {

   T* dataPtr_;
   //int size_[Dim];    // Original code with template-sized Dims
   int size_[MAX_dim];  // Workaround code - allocate an array big enough for all dims so one serializer works.


...

#ifdef __HCC__
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    s.Append(sizeof(float), &_dataPtr);
    for (int i=0; i<MAX_Dim; i++) {
      s.Append(sizeof(size_[0]), &size_[i]);
    }
  }


  __attribute__((annotate("user_deserialize")))
  MyArray(T* data, int size0, int size1, int size2, int size3) [[cpu]][[hc]] {

    data_      = data;
    size_[0]   = size0;
    size_[1]   = size1;
    size_[2]   = size2;
    size_[3]   = size3;
  }
#endif
```


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
