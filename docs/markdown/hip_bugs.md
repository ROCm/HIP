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
- Avoid calling hcLaunchKernel from a function with the __host__ attribute
__host__ MyFunc(…) {
hipLaunchKernel(myKernel, …)
- Avoid use of static with kernel definition:
static __global__ MyKernel 
- Avoid defining kernels in anonymous namespace
namespace {
__global__ MyKernel …
- Avoid calling member functions 

If hipLaunchKernel takes parameters that request explicitly memcpy, then it will cause application hang. 
Reason is that the hipLaunchKernel macro locks the stream.  
If kernel paramters are actually function calls which invoke other hip apis (i.e. memcpy) to the same stream, then deadlock occurs.

To workaround, try:
Move the function calls so they occur outside the hipLaunchKernel macro, store results in temps, then use the tems inside the kernel.

```
// Example pseudo code causing system hang:
// "bottom[0]->gpu_data()" calls hipMemcpy() implicitly and using the same stream, cause deadlock condition.
hipLaunchKernel(HIP_KERNEL_NAME(LRNComputeDiff),dim3(CAFFE_GET_BLOCKS(n_threads)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0, n_threads,
       bottom[0]->gpu_data());
       
// Move "gpu_data()" ouside of hipLaunchKernel to avoid hang. 
auto bot_gpu_data = bottom[0]->gpu_data();
hipLaunchKernel( LRNComputeDiff, dim3(CAFFE_GET_BLOCKS(n_threads)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0, n_threads, 
       bot_gpu_data);

```

### What is the current limitation of HIP Generic Grid Launch method?
1. __global__ functions cannot be marked as static or put in an unnamed namespace i.e. they cannot be given internal linkage (this would clash with __attribute__((weak)));
2. using the macro based dispatch mechanism i.e. hipLaunchKernel* only works for functions that take no more than 20 arguments (this limit can be increased up to 126, and is temporary until we can enable C++14 mode and use variadic generic lambdas); no such limitation applies do dispatching directly through grid_launch.


### HIP is more restrictive in enforcing restrictions
By the language specification, both for HIP and CUDA it is forbidden to call a
`__device__` function in a `__host__` context. In practice, you may observe
differences in the strictness of this restriction, with HIP exhibiting a tighter
adherence to the specification i.e. being less tolerant of infringing code. The
solution is to always ensure that all functions which are to be called in a
`__device__` context are correctly annotated to reflect it. An interesting case
where these differences emerge is shown below (this has been lifted from
production code, and relies on a the common [C++ Member Detector idiom][1], as it
would be implemented pre C++11):
```c++
#include <cassert>
#include <type_traits>

struct meta_yes { char a[1]; };
struct meta_no  { char a[2]; };

// Dual restriction is necessary in HIP if the detector is to work for
// __device__ contexts as well as __host__ ones. NVCC is less strict.
template<typename T>
__host__ __device__
const T& return_ref();

template<typename T>
struct has_nullary_operator {
    // Dual restriction is necessary in HIP if the detector is to work for
    // __device__ contexts as well as __host__ ones. NVCC is less strict.
    template<typename C>
    __host__ __device__
    static
    meta_yes testFunctor(
        C const *,
        typename std::enable_if<
            (sizeof(return_ref<C>().operator()()) > 0)>::type* = nullptr);
    static
    meta_no testFunctor(...);

    enum {
        value = sizeof(testFunctor(static_cast<T*>(0))) == sizeof(meta_yes) };
};

template<
    typename Scalar,
    typename NullaryOp,
    bool has_nullary = has_nullary_operator<NullaryOp>::value>
struct nullary_wrapper {
    template<typename T>
    T packetOp() const { return T{1}; }
};


template<typename Scalar, typename NullaryOp>
struct nullary_wrapper<Scalar, NullaryOp, true> {
    template<typename T>
    T packetOp() const { return T{10}; }
};

// This specialisation will fail to compile.
template<typename Scalar, typename NullaryOp>
struct nullary_wrapper<Scalar, NullaryOp, false> {};

template<typename T>
struct UniformRandomGenerator;

template<> struct UniformRandomGenerator<float> {
    float operator()() const [[hc]] { return 42.0; }
};

__device__
void this_will_not_compile_if_detector_is_not_marked_device()
{
    float f =
        nullary_wrapper<
            float, UniformRandomGenerator<float>>().packetOp<float>();
}

__host__
void this_will_not_compile_if_detector_is_marked_device_only()
{
    float f =
        nullary_wrapper<
            float, UniformRandomGenerator<float>>().packetOp<float>();
}
```
[1]: https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Member_Detector