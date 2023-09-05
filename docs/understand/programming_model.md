# Programming Model

The HIP programming model makes it as easy as reasonably possible to map
data-parallel C/C++ algorithms and map them to massively parallel, wide SIMD
architectures, such as GPUs. As a consequence, one needs a basic understanding
of the underlying device architecture to make efficient use of HIP and GPGPU
(General Purpose Graphics Processing Unit) programming in general.

## RDNA & CDNA architecture summary

## Single Instruction Multiple Threads

The SIMT programming model behind the HIP device-side execution is a
middle-ground between SMT (Simultaneous Multi-Threading) programming known from
multi-core CPUs, and SIMD (Single Instruction, Multiple Data) programming
mostly known from exploiting relevant instruction sets on CPUs (eg.
SSE/AVX/Neon).

A HIP device compiler maps our SIMT code written in HIP C++ to an inherently
SIMD architecture (like GPUs) not by exploiting data parallelism within a
single instance of a kernel and spreading identical instructions over the SIMD
engines at hand, but by scalarizing the entire kernel and issuing the scalar
instructions of multiple kernel instances to each of the SIMD engine lanes.

Consider the following kernel

```cu
__global__ void k(float4* arr)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bdim = blockDim.x;

  arr[tid] =
    (tid + bid - bdim) *
    arr[tid] +
    arr[tid];
}
```
