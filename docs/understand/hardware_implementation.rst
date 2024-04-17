.. meta::
  :description: This chapter describes the typical hardware implementation of GPUs supported by HIP.
  :keywords: AMD, ROCm, HIP, Hardware, Compute Unit, ALU, Cache, Registers, LDS

*******************************************************************************
Hardware Implementation
*******************************************************************************

This chapter describes the typical hardware implementation of GPUs supported by
HIP, and how the :ref:`inherent_thread_model` maps to the hardware.

Compute Units
===============================================================================

The basic building block of a GPU is a so called Compute Unit (CU), also known
as Streaming Multiprocessor (SM) on NVIDIA GPUs. The blocks making up a grid
are scheduled for execution on the CUs. Each block is assigned to an individual
CU, and a CU can accommodate several blocks. Depending on their resource usage
up to thousands of threads can reside on a CU.

Compute units contain an array of processing elements, also referred to as
vector ALU (VALU), that execute the actual instructions of the threads,
together with the necessary registers and caches.

The threads of a block are executed in groupings called warps. The amount of
threads making up a warp is architecture dependent.

On AMD GPUs the warp size commonly is 64 threads, with the exception of RDNA
based architectures, which can run in wave32 or wave64 mode, utilizing a warp
size of 32 or 64 respectively. The warp size of each supported AMD GPU is
listed in :doc:`rocm:reference/gpu-arch-specs`. NVIDIA GPUs have a warp size of
32.

In contrast to CPUs, GPUs generally don't employ complex cache structures or
control logic, like branch prediction or out-of-order execution, but instead
rely on massive hardware multithreading to hide latency.

Context switching between warps residing on a compute unit incurs no overhead,
as the context for the warps is stored on the compute unit and doesn't need to
be fetched from memory. If there aren't enough free registers to accommodate
all warps of a block, the block can't be scheduled to that compute unit and it
has to wait until other blocks finish execution.

The amount of warps that can reside on a compute unit concurrently, also known
as occupancy, is determined by the warps resource usage, like registers and
shared memory.

.. figure:: ../data/understand/hardware_implementation/compute_unit.svg
    :alt: Diagram depicting the general structure of a compute unit of an AMD
          GPU.
    
    An AMD Graphics Core Next (GCN) CU. The CDNA and RDNA CUs are based on
    variations of the GCN CU.

On AMD GCN GPUs the basic structure of a compute unit is
* four Single Instruction Multiple Data units (SIMDs)
* a vector cache
* a local data share
* and a scalar unit

SIMD
-------------------------------------------------------------------------------

A SIMD consists of a VALU, that executes the instruction of a warp, together
with a register file, that provides the registers warps.

The size of the warp is inherently related to the width of the vector ALU of
the SIMD. The instructions of a warp are effectively executed in lock-step.

A SIMD always executes the same instruction for the whole VALU. If the control
flow of a warp diverges, the performance is decreased, as the results for the
threads that don't participate in that branch have to be masked out, and the
instructions of the other branch have to be executed in the same way. The best
performance can therefore be achieved when thread divergence is kept to a warp
level, i.e. when all threads in a warp take the same execution path.

Vector Cache
-------------------------------------------------------------------------------

The usage of cache on a GPU differs from that on a CPU, as there is less cache
available per thread. Its main purpose is to coalesce memory accesses of the
warps in order to reduce the amount of accesses to device memory, and make that
memory available for other warps that currently reside on the compute unit, that
also need to load those values.

Local Data Share
-------------------------------------------------------------------------------

The local data share is memory that is accessible to all threads within a block.
Its latency and bandwidth is comparable to that of the vector cache. It can be
used to share memory between the threads in a block, or as a software managed
cache.

Scalar Unit
-------------------------------------------------------------------------------

The scalar unit performs instructions that are uniform within a warp. It
thereby improves efficiency and reduces the pressure on the vector ALUs and the
vector register file.
