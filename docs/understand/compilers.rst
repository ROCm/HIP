.. meta::
  :description: Compilation workflow of the HIP compilers.
  :keywords: AMD, ROCm, HIP, CUDA, HIP runtime API

.. _hip_compilers:

********************************************************************************
HIP compilers
********************************************************************************

ROCm provides tools for compiling HIP applications for use on
AMD GPUs. The compilers set up the default libraries, and include paths for the
HIP and ROCm libraries, along with required environment variables. For more
information, see the :doc:`ROCm compiler reference
<llvm-project:reference/rocmcc>`.

Compilation workflow
================================================================================

HIP provides a flexible compilation workflow that supports both offline
compilation and runtime or just-in-time (JIT) compilation. Each approach has
advantages depending on the use case, target architecture, and performance
needs.

The offline compilation is ideal for production environments, where the
performance is critical, and the target GPU architecture is known in advance.

The runtime compilation is useful in development environments or when
distributing software that must run on a wide range of hardware without the
knowledge of the GPU in advance. It provides flexibility at the cost of some
The runtime compilation is useful in development environments or when
distributing software that must run on a wide range of hardware without prior
knowledge of the GPU. It provides flexibility at the cost of some
performance overhead.

Offline compilation
--------------------------------------------------------------------------------

Offline compilation is performed in two steps: host and  device code
compilation. 

- Host-code compilation: On the host side, ``amdclang++`` or ``hipcc`` can
  compile the host code in one step without other C++ compilers. 

- Device-code compilation: The compiled device code is embedded into the
  host object file. Depending on the platform, the device code can be compiled
  into assembly or binary. 

For an example on how to compile HIP from the command line, see :ref:`SAXPY
tutorial <compiling_on_the_command_line>` .

Runtime compilation
--------------------------------------------------------------------------------

HIP allows you to compile kernels at runtime using the ``hiprtc*`` API. Kernels
are stored as a text string, which is passed to HIPRTC alongside options to
guide the compilation.

For more information, see :doc:`HIP runtime compiler <../how-to/hip_rtc>`.

.. _gfx_ip:

Target GPU architectures (GFX IP)
=================================

Instructions in the AMDGPU instruction set architecture (ISA) are compatible
only with certain physical GPU architectures. The versioning system that
abstracts hardware details from compilers and software is called the GFX IP
version. Each AMD GPU family, such as RDNA or CDNA, defines its own GFX IP
identifiers. These identifiers specify which instruction formats, memory
models, and compute features are supported by that generation of hardware.

A GFX version is expressed as a short string, for example:

* ``gfx90a`` — CDNA2 (MI250 series)
* ``gfx942`` — CDNA3 (MI300 series)
* ``gfx1100`` — RDNA3 (RX 7900 series)

Most GFX IP versions are composed of three numerical fields, which act roughly
like a major-minor-subminor versioning system:

* The major version corresponds to the architectural family (for example,
  CDNA3 versus RDNA3).
* The minor version encodes core feature updates within that family (for
  example, new :ref:`MFMA <mfma_units>` or :ref:`LDS <lds>` capabilities).
* The subminor version may specify packaging or stepping differences (for
  example, MI300A versus MI300X).

Each new generation introduces additional hardware features, for example,
mixed-precision MFMA instructions, asynchronous data movement engines, and
updated cache hierarchies, while maintaining broad forward compatibility for
code compiled at the intermediate representation (IR) level. When compiling GPU
programs with ``amdclang++``, the target GFX architecture determines the ISA
and hardware features available to the kernel.

For example, compiling for the MI300X would use:

.. code-block:: shell

  amdclang++ --offload-arch=gfx942 kernel.cpp -o kernel.out

The compiler uses this flag to emit optimized machine code for that GPU's
:ref:`compute units <compute_unit>`, :ref:`matrix cores <mfma_units>`, and
memory subsystem. The GFX IP acts as a virtual hardware target, decoupling
high-level programming models (HIP, OpenMP, OpenCL) from the underlying physical
GPU design.

While AMD does not explicitly name its compatibility model as "onion-layered,"
the GFX IP system follows a similar principle: newer architectures generally
extend, rather than break, existing instruction sets.

Thus, code compiled for an older architecture (for example, ``gfx90a``) can
often be re-optimized or recompiled for a newer one (``gfx942``) with minimal
modification. However, compatibility is not guaranteed across major GFX
families, since those may introduce fundamentally new pipelines or execution
semantics.

.. _amdgpu_assembly:

AMDGPU assembly
===============

AMDGPU assembly, sometimes called GFX ISA (Instruction Set Architecture), is
the low-level assembly format for programs running directly on AMD GPUs. It is
the lowest-level human-readable representation of GPU instructions, typically
generated by the ROCm compiler toolchain and converted into binary microcode
for execution on the device.

Each AMD GPU architecture family, such as RDNA or CDNA, defines its own GFX ISA
version. For example:

* ``gfx90a`` corresponds to CDNA2 (MI250 series)
* ``gfx942`` corresponds to CDNA3 (MI300 series)
* ``gfx1100`` corresponds to RDNA3 consumer GPUs

Each version defines its unique instruction encodings, supported data types,
and pipeline behavior. Compiled GPU kernels must target a specific :ref:`GFX
architecture <gfx_ip>` to ensure instruction compatibility and optimal
scheduling.

AMDGPU assembly is rich and expressive, exposing every level of GPU behavior:
scalar operations, vector math, memory access, synchronization, and
matrix-multiply instructions. A few examples of instructions from the
``gfx942`` architecture include:

* ``v_add_f32 v0, v1, v2`` — add 32-bit floats in vector registers
* ``s_mov_b32 s4, 0x3f800000`` — move immediate value (1.0f) into a scalar
  register
* ``v_mfma_f32_16x16x4f16 v[0:15], v[16:31], v[32:47], v[0:15]`` — perform a
  16×16×4 matrix fused multiply-add
* ``s_barrier`` — synchronize all warps in the work-group

In this syntax:

* ``v_`` instructions operate on vector registers (VGPRs) per thread.
* ``s_`` instructions operate on scalar registers (SGPRs) shared by the warp.
* Specialized matrix instructions (``v_mfma_*``) invoke the :ref:`Matrix Core
  (MFMA) hardware units <mfma_units>`.

Although AMDGPU assembly can be written by hand, this is uncommon. Developers typically inspect compiler‑generated assembly when optimizing kernels, diagnosing register pressure, or tuning memory‑access patterns

The ROCm disassembler (``llvm-objdump``) and ROCm profiler (``rocprofv3``)
allow inspection of generated GFX ISA alongside high-level HIP or OpenMP source
code. Because the instruction semantics and binary encodings are publicly
documented by AMD, the GFX ISA is a fully open, compiler-targetable standard.

This openness allows tool developers, performance engineers, and researchers to
reason about GPU behavior at the instruction level, bridging the gap
between hardware and high-level kernel code.

.. _amdgpu_ir:

AMDGPU intermediate representation
===================================

AMDGPU intermediate representation (AMDGPU IR) is an intermediate
representation for code that runs on AMD GPUs and other parallel processors. It
is one of the key outputs of the ROCm compiler toolchain, produced by
``amdclang++`` before being translated into architecture-specific GPU assembly
(:ref:`GFX ISA <amdgpu_assembly>`).

AMD documentation refers to this layer as both a virtual instruction set and
a target-specific dialect of LLVM IR. From a programmer's perspective, AMDGPU
IR defines a virtual machine model for parallel thread execution: compilers and
optimizers emit this IR with the expectation that it will execute with consistent semantics
across multiple generations of AMD hardware, including future architectures not
yet released.

This makes AMDGPU IR a "narrow waist" between software and hardware,
abstracting the details of the physical :ref:`compute units <compute_unit>`,
:ref:`warp <wavefront>` schedulers, and memory hierarchies, while still
providing explicit control over threads, registers, and memory spaces.

Unlike traditional CPU ISAs, AMDGPU IR is not executed directly. Instead, it is
further compiled (either just‑in‑time or ahead‑of‑time) into GFX ISA, the
device-specific binary that runs on a given GPU architecture (for example,
``gfx942`` for CDNA3 MI300X or ``gfx1200`` for RDNA3). This multi-stage
compilation allows forward compatibility: kernels compiled for one generation
can often be re-optimized and executed efficiently on newer hardware through
updated ROCm runtimes and drivers. Some exemplary fragments of AMDGPU IR:

.. code-block:: llvm

  ; declare usage of 64 vector registers
  !amdgpu.num_vgpr = !{i32 64}

  ; perform fused multiply-add
  %f5 = call float @llvm.fma.f32(float %f3, float %f4, float 1.5)

  ; read thread and workgroup indices
  %tid  = call i32 @llvm.amdgcn.workitem.id.x()
  %wgid = call i32 @llvm.amdgcn.workgroup.id.x()

These instructions represent operations in the virtual machine model:

* Registers are virtual VGPRs/SGPRs dynamically mapped to physical hardware
  registers by the compiler.
* Arithmetic and memory intrinsics (``llvm.fma``, ``llvm.amdgcn.buffer.load``)
  map one-to-one to GPU instructions.
* Built-in functions like ``llvm.amdgcn.workitem.id.x()`` access special
  per-thread state, such as the current thread or
  :ref:`block <inherent_thread_hierarchy_block>` index.

The AMDGPU IR machine model reflects the hardware reality of AMD GPUs: a single
instruction stream drives a :ref:`wavefront <wavefront>` of 64 threads that
execute in lockstep on the SIMD pipelines, each maintaining its own register
state while sharing program flow and :ref:`local memory <lds>`.
Warps cooperate through the :ref:`Local Data Share (LDS) <lds>` and synchronize
via barriers, the same abstractions exposed in the high-level ROCm programming
model.

Since AMDGPU IR is integrated with the open-source LLVM compiler
infrastructure, its syntax and semantics are well-documented and publicly
accessible. Developers can inspect, modify, or emit AMDGPU IR directly for
performance analysis, research, or custom toolchains.

.. _binary_utilities:

ROCm binary utilities
=====================

The ROCm binary utilities are a collection of command-line tools for examining
and manipulating GPU binaries produced by ``amdclang++`` or other ROCm build
tools. These utilities allow developers to inspect, disassemble, and analyze
AMDGPU code objects, the compiled GPU kernels embedded in host executables or
distributed as standalone ``.hsaco`` files.

The ``llvm-objdump`` utility provides multiple capabilities for analyzing GPU
binaries. With the ``--offloading`` flag, it can list and extract information
from the contents of ROCm binaries, including code object metadata, kernel
symbols, target architectures (for example, ``gfx90a``, ``gfx1100``), and
linkage details. It supports both standalone ``.hsaco`` files and "fat
binaries" embedded within host executables. With the ``--triple=amdgcn`` flag,
it can disassemble GPU kernels into human-readable AMDGPU ISA, allowing
inspection of instruction sequences, register allocation, and control flow.
These capabilities are essential for performance debugging, code verification,
and low-level kernel analysis, for example, when tuning :ref:`MFMA
<mfma_units>` instructions or checking compiler optimizations.

Together, these utilities provide developers with insight into how HIP C++ code
is compiled, optimized, and mapped to GPU hardware. They complement profiling
tools like ``rocprofv3`` by exposing the static structure of compiled GPU
binaries.

Static libraries
================================================================================

``amdclang++`` supports generating two types of static libraries.

- The first type of static library only exports and launches host functions
  within the same library and not the device functions. This library type
  offers the ability to link with another compiler such as ``gcc``.
  Additionally, this library type contains host objects with device code
  embedded as fat binaries. This library type is generated using the flag
  ``--emit-static-lib``:

  .. code-block:: shell

    amdclang++ hipOptLibrary.cpp --emit-static-lib -fPIC -o libHipOptLibrary.a
    gcc test.cpp -L. -lhipOptLibrary -L/path/to/hip/lib -lamdhip64 -o test.out

- The second type of static library exports device functions to be linked by
  other code objects by using ``amdclang++`` as the linker. This library type
  contains relocatable device objects and is generated using
  ``ar``:

  .. code-block:: shell

    amdclang++ hipDevice.cpp -c -fgpu-rdc -o hipDevice.o
    ar rcsD libHipDevice.a hipDevice.o
    amdclang++ libHipDevice.a test.cpp -fgpu-rdc -o test.out

Examples of this can be found in `rocm-examples
<https://github.com/ROCm/rocm-examples>`_ under `static host libraries
<https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/static_host_library>`_
or `static device libraries
<https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/static_device_library>`_.
