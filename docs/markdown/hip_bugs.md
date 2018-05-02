# HIP Bugs

<!-- toc -->
- [HIP is more restrictive in enforcing restrictions](#hip-is-more-restrictive-in-enforcing-restrictions)

<!-- tocstop -->



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
