# HIP Bugs 

<!-- toc -->

- [Errors related to undefined reference to `__hcLaunchKernel__***__grid_launch_parm**](#errors-related-to-undefined-reference-to-hclaunchkernel__grid_launch_parm)
- [Application hangs after a hipLaunchKernel call](#what-if-i-see-application-hangs-after-a-hiplaunchkernel-call)

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


### What if I see application hangs after a hipLaunchKernel call?
If hipLaunchKernel takes parameters that request explicitly memcpy, then it will cause application hang. 
Reason is that the hipLaunchKernel macro locks the stream.  
If kernel paramters are actually function calls which invoke other hip apis (i.e. memcpy) to the same stream, then deadlock occurs.

To workaround, try:
Move the function calls so they occur outside the hipLaunchKernel macro, store results in temps, then use the tems inside the kernel.