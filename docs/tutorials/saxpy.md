# Tutorial: SAXPY - Hello, HIP

This tutorial will show you some of the basic concepts of the single-source HIP
programming model, the most essential tooling around it and briefly rehash some
commonalities of heterogenous APIs in general. Mild familiarity with the C/C++
compilation model and the language is assumed throughout this article.

## Prerequisites

In order to follow this tutorial you will need properly installed drivers and a
HIP compiler toolchain to compile your code. Because HIP provided by ROCm
supports compiling and running on Linux and Windows with AMD and NVIDIA GPUs
alike, the combination of install instructions are more then worth covering as
part of this tutorial. Please refer to {doc}`/how_to_guides/install` on how to
install HIP development packages.

## Heterogenous Programming

Heterogenous programming and offloading APIs are often mentioned together.
Heterogenous programming deals with devices of varying capabilities at once
while the term offloading focuses on the "remote" and asnychronous aspect of
the computation. HIP encompasses both: it exposes GPGPU (General Purpose GPU)
programming much like ordinary host-side CPU programming and let's us move data
to and from device as need be.

```{important}
Most of the HIP-specific code will deal with "where", "what" and "how".
```

One thing to keep in mind while programming in HIP (and other heterogenous APIs
for that matter), is that target devices are built for a specific purpose. They
are designed with different tradeoffs than traditional CPUs and therefore have
very different performance characteristics. Ever so subtle changes in code may
effect execution time adversely. Rest assured one can always find the root
cause and design a fix.

## Your First Piece of HIP Code

First, let's take the "Hello, World!" of GPGPU: SAXPY. The name comes from
math, a vector equation at that: {math}`a\cdot x+y=z` where
{math}`a\in\mathbb{R}` is a scalar and {math}`x,y,z\in\mathbb{V}` are vector
quantities of some large dimensionality. (Our vector space is defined over the
set of reals.) From a practical perspective we can compute this using a single
`for` loop over 3 arrays.

```c++
for (int i = 0 ; i < N ; ++i)
    z[i] = a * x[i] + y[i];
```

In linear algebra libraries, such as BLAS (Basic Linear Algebra Subsystem) this
operation is defined as AXPY "A times X Plus Y". The "S" comes from
single-precision, meaning that every element of our array are `float`s. (We
choose IEEE 754: binary32 arithmetic as the representation of our algebra.)

To get quickly off the ground, we'll take off-the-shelf piece of code, the set
of [HIP samples from GitHub](https://github.com/amd/rocm-examples/). Assuming
Git is on your Path, open a command-line and navigate to where you'll want to
work, then issue:

```shell
git clone https://github.com/amd/rocm-examples.git
```

Inside the repo, you should find `HIP-Basic\saxpy\main.hip` which is a
sufficiently simple implementation of SAXPY. It was already mentioned
that HIP code will mostly deal with where and when data has to be and
how devices will transform it. The very first HIP calls deal with
allocating device-side memory and dispatching data from host-side
memory in a C Runtime-like fashion.

```hip
// Allocate and copy vectors to device memory.
float* d_x{};
float* d_y{};
HIP_CHECK(hipMalloc(&d_x, size_bytes));
HIP_CHECK(hipMalloc(&d_y, size_bytes));
HIP_CHECK(hipMemcpy(d_x, x.data(), size_bytes, hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(d_y, y.data(), size_bytes, hipMemcpyHostToDevice));
```

`HIP_CHECK` is a custom macro borrowed from the examples utilities which
checks the error code returned by API functions for errors and reports them to
the console. It's not quintessential to the API.

If you're wondering: how does it know which device to allocate on / dispatch
to... wonder no more. Commands are issued to the HIP runtime on a per-thread
basis and every thread has a device set as the target of commands. The default
device is `0`, which is equivalent to calling `hipSetDevice(0)`.

Once our data has been dispatched, we can launch our calculation on the device.

```cu
__global__ void saxpy_kernel(const float a, const float* d_x, float* d_y, const unsigned int size)
{
    ...
}

int main()
{
    ...

    // Launch the kernel on the default stream.
    saxpy_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(a, d_x, d_y, size);
}
```

First let's discuss the signature of the offloaded function:

- `__global__` instructs the compiler to generate code for this function as an
  entrypoint to a device program, such that it can be launched from the host.
- The function does not return anything, because there is no trivial way to
  construct a return channel of a parallel invocation. Device-side entrypoints
  may not return a value, their results should be communicated using out
  params.
- Device-side functions are typically called compute kernels, or just kernels
  for short. This is to distinguish them from non-graphics-related graphics
  shaders, or just shaders for short.
- Arguments are taken by value and all arguments shall be
  [TriviallyCopyable](https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable),
  meaning they should be `memcpy`-friendly. _(Imagine if they had custom copy
  constructors. Where would that logic execute? On the host? On the device?)_
  Pointer arguments are pointers to device memory, one typically backed by
  VRAM.
- We said that we'll be computing {math}`a\cdot x+y=z`, however we only pass
  two pointers to the function. We'll be canonically reusing one of the inputs
  as outputs.

There's quite a lot to unpack already. How is this function launched from the
host? Using a language extension, the so-called triple chevron syntax. Inside
the angle brackets we can provide the following:

- The number of blocks to launch (our grid size)
- The number of threads in a block (our block size)
- The amount of shared memory to allocate by the host
- The device stream to enqueue the operation on

Following the triple chevron is ordinary function argument passing. Now let's
take a look how the kernel is implemented.

```cu
__global__ void saxpy_kernel(const float a, const float* d_x, float* d_y, const unsigned int size)
{
    // Compute the current thread's index in the grid.
    const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // The grid can be larger than the number of items in the vectors. Avoid out-of-bounds addressing.
    if(global_idx < size)
    {
        d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
    }
}
```

- The unique linear index identifying the thread is computed from the block id
  the thread is a member of, the block's size and the id of the thread within
  the block.
- A check is made to avoid overindexing the input.
- The useful part of the computation is carried out.

Retrieval of the result from the device is done much like its dispatch:

```cu
HIP_CHECK(hipMemcpy(y.data(), d_y, size_bytes, hipMemcpyDeviceToHost));
```

## Compiling on the Command-Line

(setting_up_the_command-line)=

### Setting Up the Command-Line

While strictly speaking there's no such thing as "setting up the command-line
for compilation" on Linux, just to make invocations more terse let's do it on
both Linux and Windows.

::::{tab-set}
:::{tab-item} Linux & AMD
:sync: linux-amd

While distro maintainers may package ROCm such that they install to
system-default locations, AMD's installation don't and need to be added to the
Path by the user.

```bash
export PATH=/opt/rocm/bin:${PATH}
```

You should be able to call the compiler on the command-line now:

```bash
amdclang++ --version
```

```{tip}
Docker images distributed by AMD, such as
[`rocm-terminal`](https://hub.docker.com/r/rocm/rocm-terminal/) already have
`/opt/rocm/bin` on the Path for convenience. (This subtly affects CMake package
detection logic of ROCm libraries.)
```

:::
:::{tab-item} Linux & NVIDIA
:sync: linux-nvidia

Both distro maintainers and NVIDIA package CUDA as such that `nvcc` and related
tools are on the command-line by default. You should be able to call the
compiler on the command-line simply:

```bash
nvcc --version
```

:::
:::{tab-item} Windows & AMD
:sync: windows-amd

Windows compilers and command-line tooling have traditionally
relied on extra environmental variables and Path entries to function correctly.
Visual Studio refers to command-lines with these setup as "Developer
Command Prompt" or "Developer PowerShell" for `cmd.exe` and PowerShell
respectively.

The HIP SDK on Windows doesn't ship a complete toolchain, you will also need:

- the Windows SDK, most crucially providing the import libs to crucial system
  libraries all executables must link to and some auxiliary compiler tooling.
- a Standard Template Library, aka. STL, which HIP too relies on.

The prior may be installed separately, though it's most conveniently obtained
through the Visual Studio installer, while the latter is part of the Microsoft
Visual C++ compiler, aka. MSVC, also installed via Visual Studio.

If you don't already have some SKU of Visual Studio 2022 installed, for a
minimal command-line experience, install the
[Build Tools for Visual Studio 2022](https://aka.ms/vs/17/release/vs_BuildTools.exe)
with the Desktop Developemnt Workload and under Individual Components select:

- some version of the Windows SDK
- "MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)"
- "C++ CMake tools for Windows" (optional)

```{tip}
The "C++ CMake tools for Windows" individual component is a convenience which
puts both `cmake.exe` and `ninja.exe` onto the `PATH` inside developer
command-prompts. You can install these manually, but then you need to manage
them manually.
```

Visual Studio installations as of VS 2017 are detectable as COM object
instances via WMI. To setup a command-line from any shell for the latest
Visual Studio's default (latest) Visual C++ toolset issue:

```pwsh
$InstallationPath = Get-CimInstance MSFT_VSInstance | Sort-Object -Property Version -Descending | Select-Object -First 1 -ExpandProperty InstallLocation
Import-Module $InstallationPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll
Enter-VsDevShell -InstallPath $InstallationPath -SkipAutomaticLocation -Arch amd64 -HostArch amd64 -DevCmdArguments '-no_logo'
$env:PATH = "${env:HIP_PATH}bin;${env:PATH}"
```

You should be able to call the compiler on the command-line now:

```pwsh
clang++ --version
```

:::
:::{tab-item} Windows & NVIDIA
:sync: windows-nvidia

Windows compilers and command-line tooling have traditionally
relied on extra environmental variables and Path entries to function correctly.
Visual Studio refers to command-lines with these setup as "Developer
Command Prompt" or "Developer PowerShell" for `cmd.exe` and PowerShell
respectively.

The HIP and CUDA SDKs on Windows doesn't ship complete toolchains, you will
also need:

- the Windows SDK, most crucially providing the import libs to crucial system
  libraries all executables must link to and some auxiliary compiler tooling.
- a Standard Template Library, aka. STL, which HIP too relies on.

The prior may be installed separately, though it's most conveniently obtained
through the Visual Studio installer, while the latter is part of the Microsoft
Visual C++ compiler, aka. MSVC, also installed via Visual Studio.

If you don't already have some SKU of Visual Studio 2022 installed, for a
minimal command-line experience, install the
[Build Tools for Visual Studio 2022](https://aka.ms/vs/17/release/vs_BuildTools.exe)
with the Desktop Developemnt Workload and under Individual Components select:

- some version of the Windows SDK
- "MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)"
- "C++ CMake tools for Windows" (optional)

```{tip}
The "C++ CMake tools for Windows" individual component is a convenience which
puts both `cmake.exe` and `ninja.exe` onto the `PATH` inside developer
command-prompts. You can install these manually, but then you need to manage
them manually.
```

Visual Studio installations as of VS 2017 are detectable as COM object
instances via WMI. To setup a command-line from any shell for the latest
Visual Studio's default (latest) Visual C++ toolset issue:

```pwsh
$InstallationPath = Get-CimInstance MSFT_VSInstance | Sort-Object -Property Version -Descending | Select-Object -First 1 -ExpandProperty InstallLocation
Import-Module $InstallationPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll
Enter-VsDevShell -InstallPath $InstallationPath -SkipAutomaticLocation -Arch amd64 -HostArch amd64 -DevCmdArguments '-no_logo'
```

You should be able to call the compiler on the command-line now:

```pwsh
nvcc --version
```

:::
::::

### Invoking the Compiler Manually

To compile and link a single-file application, one may use the following
command:

::::{tab-set}
:::{tab-item} Linux & AMD
:sync: linux-amd

```bash
amdclang++ ./HIP-Basic/saxpy/main.hip -o saxpy -I ./Common -lamdhip64 -L /opt/rocm/lib -O2
```

:::
:::{tab-item} Linux & NVIDIA
:sync: linux-nvidia

```bash
nvcc ./HIP-Basic/saxpy/main.hip -o saxpy -I ./Common -I /opt/rocm/include -O2 -x cu
```

:::
:::{tab-item} Windows & AMD
:sync: windows-amd

```pwsh
clang++ .\HIP-Basic\saxpy\main.hip -o saxpy.exe -I .\Common -lamdhip64 -L ${env:HIP_PATH}lib -O2
```

:::
:::{tab-item} Windows & NVIDIA
:sync: windows-nvidia

```pwsh
nvcc .\HIP-Basic\saxpy\main.hip -o saxpy.exe -I ${env:HIP_PATH}include -I .\Common -O2 -x cu
```

:::
::::

Depending on your computer, the resulting binary may or may not run. If not, it
will typically complain about about "Invalid device function". That error
(corresponding to the `hipErrorInvalidDeviceFunction` entry of `hipError_t`)
means that the runtime could not find a device program binary of the
appropriate flavor embedded into the executable.

So far we've only talked about how our data makes it from the host to the
device and back. We've also seen our device code as source, but the HIP runtime
was arguing about not finding the right binary to dispatch for execution. How
can one find out what device binary flavors are embedded into the executable?

::::{tab-set}
:::{tab-item} Linux & AMD
:sync: linux-amd

The set of `roc-*` utilities shipping with ROCm help significantly to inspect
binary artifacts on disk. If you wish to use these utilities, add the ROCmCC
installation folder to your PATH (the utilities expect them to be on the PATH).

Lisitng of the embedded program binaries can be done using `roc-obj-ls`

```bash
roc-obj-ls ./saxpy
```

It may return something like:

```shell
1       host-x86_64-unknown-linux         file://./saxpy#offset=12288&size=0
1       hipv4-amdgcn-amd-amdhsa--gfx803   file://./saxpy#offset=12288&size=9760
```

We can see that the compiler embedded a version 4 code object (more on code
object versions
[here](https://www.llvm.org/docs/AMDGPUUsage.html#code-object-metadata)) and
used the LLVM target triple `amdgcn-amd-amdhsa--gfx803` (more on target triples
[here](https://www.llvm.org/docs/AMDGPUUsage.html#target-triples)). We can
extract that program object in a disassembled fashion for human consumption via
`roc-obj`

```bash
roc-obj -t gfx803 -d ./saxpy
```

Which will create two files on disk and we'll be interested in the one with the
`.s` extension. Opening up said file or dumping it to the console using `cat`
one will find the disassembled binary of our saxpy compute kernel, something
similar to:

```none
Disassembly of section .text:

<_Z12saxpy_kernelfPKfPfj>:
    s_load_dword s0, s[4:5], 0x2c        // 000000001000: C0020002 0000002C
    s_load_dword s1, s[4:5], 0x18        // 000000001008: C0020042 00000018
    s_waitcnt lgkmcnt(0)                 // 000000001010: BF8C007F
    s_and_b32 s0, s0, 0xffff             // 000000001014: 8600FF00 0000FFFF
    s_mul_i32 s6, s6, s0                 // 00000000101C: 92060006
    v_add_u32_e32 v0, vcc, s6, v0        // 000000001020: 32000006
    v_cmp_gt_u32_e32 vcc, s1, v0         // 000000001024: 7D980001
    s_and_saveexec_b64 s[0:1], vcc       // 000000001028: BE80206A
    s_cbranch_execz 22                   // 00000000102C: BF880016 <_Z12saxpy_kernelfPKfPfj+0x88>
    s_load_dwordx4 s[0:3], s[4:5], 0x8   // 000000001030: C00A0002 00000008
    v_mov_b32_e32 v1, 0                  // 000000001038: 7E020280
    v_lshlrev_b64 v[0:1], 2, v[0:1]      // 00000000103C: D28F0000 00020082
    s_waitcnt lgkmcnt(0)                 // 000000001044: BF8C007F
    v_mov_b32_e32 v3, s1                 // 000000001048: 7E060201
    v_add_u32_e32 v2, vcc, s0, v0        // 00000000104C: 32040000
    v_addc_u32_e32 v3, vcc, v3, v1, vcc  // 000000001050: 38060303
    flat_load_dword v2, v[2:3]           // 000000001054: DC500000 02000002
    v_mov_b32_e32 v3, s3                 // 00000000105C: 7E060203
    v_add_u32_e32 v0, vcc, s2, v0        // 000000001060: 32000002
    v_addc_u32_e32 v1, vcc, v3, v1, vcc  // 000000001064: 38020303
    flat_load_dword v3, v[0:1]           // 000000001068: DC500000 03000000
    s_load_dword s0, s[4:5], 0x0         // 000000001070: C0020002 00000000
    s_waitcnt vmcnt(0) lgkmcnt(0)        // 000000001078: BF8C0070
    v_mac_f32_e32 v3, s0, v2             // 00000000107C: 2C060400
    flat_store_dword v[0:1], v3          // 000000001080: DC700000 00000300
    s_endpgm                             // 000000001088: BF810000
```

Alternatively we can call the compiler with `--save-temps` to dump all device
binary to disk in separate files.

```bash
amdclang++ ./HIP-Basic/saxpy/main.hip -o saxpy -I ./Common -lamdhip64 -L /opt/rocm/lib -O2 --save-temps
```

Now we can list all the temporaries created while compiling `main.hip` via

```bash
ls main-hip-amdgcn-amd-amdhsa-*
main-hip-amdgcn-amd-amdhsa-gfx803.bc
main-hip-amdgcn-amd-amdhsa-gfx803.cui
main-hip-amdgcn-amd-amdhsa-gfx803.o
main-hip-amdgcn-amd-amdhsa-gfx803.out
main-hip-amdgcn-amd-amdhsa-gfx803.out.resolution.txt
main-hip-amdgcn-amd-amdhsa-gfx803.s
```

Files with the `.s` extension hold the disassembled contents of the binary and
the filename directly informs us of the graphics IPs used by the compiler. The
contents of this file is very similar to what `roc-obj` printed to the console.

:::
:::{tab-item} Linux & NVIDIA
:sync: linux-nvidia

Unlike HIP on AMD, when compiling using the NVIDIA support of HIP the resulting
binary will be a valid CUDA executable as far as the binary goes. Therefor
it'll incorporate PTX ISA (Parallel Thread eXecution Instruction Set
Architecture) instead of AMDGPU binary. As s result, tooling shipping with the
CUDA SDK can be used to inspect which device ISA got compiled into a specific
executable. The tool most useful to us currently is `cuobjdump`.

```bash
cuobjdump --list-ptx ./saxpy 
```

Which will print something like:

```none
PTX file    1: saxpy.1.sm_52.ptx
```

From this we can see that the saxpy kernel is stored as `sm_52`, which shows
that a compute capability 5.2 ISA got embedded into the executable, so devices
which sport compute capability 5.2 or newer will be able to run this code.

:::
:::{tab-item} Windows & AMD
:sync: windows-amd

The HIP SDK for Windows don't yet sport the `roc-*` set of utilities to work
with binary artifacts. To find out what binary formats are embedded into an
executable, one may use `dumpbin` tool from the Windows SDK to obtain the
raw data of the `.hip_fat` section of an executable. (This binary payload is
what gets parsed by the `roc-*` set of utilities on Linux.) Skipping over the
reported header, the rendered raw data as ASCII has ~3 lines per entries.
Depending on how many binaries are embedded, you may need to alter the number
of rendered lines. An invocation such as:

```pwsh
dumpbin.exe /nologo /section:.hip_fat /rawdata:8 .\saxpy.exe | select -Skip 20 -First 12
```

The output may look like:

```none
000000014004C000: 5F474E414C435F5F 5F44414F4C46464F   __CLANG_OFFLOAD_
000000014004C010: 5F5F454C444E5542 0000000000000002   BUNDLE__........
000000014004C020: 0000000000001000 0000000000000000   ................
000000014004C030: 0000000000000019 3638782D74736F68   ........host-x86
000000014004C040: 6E6B6E752D34365F 756E696C2D6E776F   _64-unknown-linu
000000014004C050: 0000000000100078 00000000000D9800   x...............
000000014004C060: 0000000000001F00 612D347670696800   .........hipv4-a
000000014004C070: 6D612D6E6367646D 617368646D612D64   mdgcn-amd-amdhsa
000000014004C080: 3630397866672D2D 0000000000000000   --gfx906........
000000014004C090: 0000000000000000 0000000000000000   ................
000000014004C0A0: 0000000000000000 0000000000000000   ................
000000014004C0B0: 0000000000000000 0000000000000000   ................
```

We can see that the compiler embedded a version 4 code object (more on code
object versions
[here](https://www.llvm.org/docs/AMDGPUUsage.html#code-object-metadata)) and
used the LLVM target triple `amdgcn-amd-amdhsa--gfx906` (more on target triples
[here](https://www.llvm.org/docs/AMDGPUUsage.html#target-triples)). Don't be
alarmed about linux showing up as a binary format, AMDGPU binaries uploaded to
the GPU for execution are proper linux ELF binaries in their format.

Alternatively we can call the compiler with `--save-temps` to dump all device
binary to disk in separate files.

```pwsh
clang++ .\HIP-Basic\saxpy\main.hip -o saxpy.exe -I .\Common -lamdhip64 -L ${env:HIP_PATH}lib -O2 --save-temps
```

Now we can list all the temporaries created while compiling `main.hip` via

```pwsh
Get-ChildItem -Filter main-hip-* | select -Property Name

Name
----
main-hip-amdgcn-amd-amdhsa-gfx906.bc
main-hip-amdgcn-amd-amdhsa-gfx906.hipi
main-hip-amdgcn-amd-amdhsa-gfx906.o
main-hip-amdgcn-amd-amdhsa-gfx906.out
main-hip-amdgcn-amd-amdhsa-gfx906.out.resolution.txt
main-hip-amdgcn-amd-amdhsa-gfx906.s
```

Files with the `.s` extension hold the disassembled contents of the binary and
the filename directly informs us of the graphics IPs used by the compiler.

```pwsh
Get-ChildItem main-hip-*.s | Get-Content
        .text
        .amdgcn_target "amdgcn-amd-amdhsa--gfx906"
        .protected      _Z12saxpy_kernelfPKfPfj ; -- Begin function _Z12saxpy_kernelfPKfPfj
        .globl  _Z12saxpy_kernelfPKfPfj
        .p2align        8
        .type   _Z12saxpy_kernelfPKfPfj,@function
_Z12saxpy_kernelfPKfPfj:                ; @_Z12saxpy_kernelfPKfPfj
; %bb.0:
        s_load_dword s0, s[4:5], 0x4
        s_load_dword s1, s[6:7], 0x18
        s_waitcnt lgkmcnt(0)
        s_and_b32 s0, s0, 0xffff
        s_mul_i32 s8, s8, s0
        v_add_u32_e32 v0, s8, v0
        v_cmp_gt_u32_e32 vcc, s1, v0
        s_and_saveexec_b64 s[0:1], vcc
        s_cbranch_execz .LBB0_2
; %bb.1:
        s_load_dwordx4 s[0:3], s[6:7], 0x8
        v_mov_b32_e32 v1, 0
        v_lshlrev_b64 v[0:1], 2, v[0:1]
        s_waitcnt lgkmcnt(0)
        v_mov_b32_e32 v3, s1
        v_add_co_u32_e32 v2, vcc, s0, v0
        v_addc_co_u32_e32 v3, vcc, v3, v1, vcc
        global_load_dword v2, v[2:3], off
        v_mov_b32_e32 v3, s3
        v_add_co_u32_e32 v0, vcc, s2, v0
        v_addc_co_u32_e32 v1, vcc, v3, v1, vcc
        global_load_dword v3, v[0:1], off
        s_load_dword s0, s[6:7], 0x0
        s_waitcnt vmcnt(0) lgkmcnt(0)
        v_fmac_f32_e32 v3, s0, v2
        global_store_dword v[0:1], v3, off
.LBB0_2:
        s_endpgm
        ...
```

:::
:::{tab-item} Windows & NVIDIA
:sync: windows-nvidia

Unlike HIP on AMD, when compiling using the NVIDIA support of HIP the resulting
binary will be a valid CUDA executable as far as the binary goes. Therefor
it'll incorporate PTX ISA (Parallel Thread eXecution Instruction Set
Architecture) instead of AMDGPU binary. As s result, tooling shipping with the
CUDA SDK can be used to inspect which device ISA got compiled into a specific
executable. The tool most useful to us currently is `cuobjdump`.

```bash
cuobjdump.exe --list-ptx .\saxpy.exe
```

Which will print something like:

```none
PTX file    1: saxpy.1.sm_52.ptx
```

From this we can see that the saxpy kernel is stored as `sm_52`, which shows
that a compute capability 5.2 ISA got embedded into the executable, so devices
which sport compute capability 5.2 or newer will be able to run this code.

:::
::::

Now that we've found what binary got embedded into the executable, we only need
to find which format our available devices use.

::::{tab-set}
:::{tab-item} Linux & AMD
:sync: linux-amd

On Linux a utility called `rocminfo` can help us list all the properties of the
devices available on the system, including which version of graphics IP
(`gfxXYZ`) they employ. We'll filter the output to have only these lines:

```bash
/opt/rocm/bin/rocminfo | grep gfx
  Name:                    gfx906
      Name:                    amdgcn-amd-amdhsa--gfx906:sramecc+:xnack-
```

_(For the time being let's not discuss what the colon-dlimited list of device
features are after the graphics IP. Until further notice we'll treat them as
part of the binary version.)_

:::
:::{tab-item} Linux & NVIDIA
:sync: linux-nvidia

On Linux HIP with the NVIDIA back-end a CUDA SDK sample called `deviceQuery`
can help us list all the properties of the devices available on the system,
including which version of compute capability a device sports.
(`<major>.<minor>` compute capability is passed to `nvcc` on the
command-line as `sm_<major><minor>`, for eg. `8.6` is `sm_86`.)

Because it's not shipped as a binary, we may as well compile the matching
example from ROCm.

```bash
nvcc ./HIP-Basic/device_query/main.cpp -o device_query -I ./Common -I /opt/rocm/include -O2
```

We'll filter the output to have only the lines of interest, for eg.:

```bash
./device_query | grep "major.minor"
major.minor:              8.6
major.minor:              7.0
```

```{tip}
Next to the `nvcc` executable is another tool called `__nvcc_device_query`
which simply prints the SM Architecture numbers to standard out as a comma
separated list of numbers. The naming of this utility suggests it's not a user
facing executable but is used by `nvcc` to determine what devices are in the
system at hand.
```

:::
:::{tab-item} Windows & AMD
:sync: windows-amd

On Windows a utility called `hipInfo.exe` can help us list all the properties
of the devices available on the system, including which version of graphics IP
(`gfxXYZ`) they employ. We'll filter the output to have only these lines:

```pwsh
& ${env:HIP_PATH}bin\hipInfo.exe | Select-String gfx

gcnArchName:                      gfx1032
gcnArchName:                      gfx1035
```

:::
:::{tab-item} Winodws & NVIDIA
:sync: windows-nvidia

On Windows HIP with the NVIDIA back-end a CUDA SDK sample called `deviceQuery`
can help us list all the properties of the devices available on the system,
including which version of compute capability a device sports.
(`<major>.<minor>` compute capability is passed to `nvcc` on the
command-line as `sm_<major><minor>`, for eg. `8.6` is `sm_86`.)

Because it's not shipped as a binary, we may as well compile the matching
example from ROCm.

```pwsh
nvcc .\HIP-Basic\device_query\main.cpp -o device_query.exe -I .\Common -I ${env:HIP_PATH}include -O2
```

We'll filter the output to have only the lines of interest, for eg.:

```pwsh
.\device_query.exe | Select-String "major.minor"

major.minor:              8.6
major.minor:              7.0
```

```{tip}
Next to the `nvcc` executable is another tool called `__nvcc_device_query.exe`
which simply prints the SM Architecture numbers to standard out as a comma
separated list of numbers. The naming of this utility suggests it's not a user
facing executable but is used by `nvcc` to determine what devices are in the
system at hand.
```

:::
::::

Now that we know which versions of graphics IP our devices use, we can
recompile our program with said parameters.

::::{tab-set}
:::{tab-item} Linux & AMD
:sync: linux-amd

```bash
amdclang++ ./HIP-Basic/saxpy/main.hip -o saxpy -I ./Common -lamdhip64 -L /opt/rocm/lib -O2 --offload-arch=gfx906:sramecc+:xnack-
```

Now our sample will surely run.

```none
./saxpy
Calculating y[i] = a * x[i] + y[i] over 1000000 elements.
First 10 elements of the results: [ 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 ]
```

:::
:::{tab-item} Linux & NVIDIA
:sync: linux-nvidia

```bash
nvcc ./HIP-Basic/saxpy/main.hip -o saxpy -I ./Common -I /opt/rocm/include -O2 -x cu -arch=sm_70,sm_86
```

```{tip}
If you want to portably target the development machine which is compiling, you
may specify `-arch=native` instead.
```

Now our sample will surely run.

```none
./saxpy
Calculating y[i] = a * x[i] + y[i] over 1000000 elements.
First 10 elements of the results: [ 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 ]
```

:::
:::{tab-item} Windows & AMD
:sync: windows-amd

```pwsh
clang++ .\HIP-Basic\saxpy\main.hip -o saxpy.exe -I .\Common -lamdhip64 -L ${env:HIP_PATH}lib -O2 --offload-arch=gfx1032 --offload-arch=gfx1035
```

Now our sample will surely run.

```none
.\saxpy.exe
Calculating y[i] = a * x[i] + y[i] over 1000000 elements.
First 10 elements of the results: [ 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 ]
```

:::
:::{tab-item} Windows & NVIDIA
:sync: windows-nvidia

```pwsh
nvcc .\HIP-Basic\saxpy\main.hip -o saxpy.exe -I ${env:HIP_PATH}include -I .\Common -O2 -x cu -arch=sm_70,sm_86
```

```{tip}
If you want to portably target the development machine which is compiling, you
may specify `-arch=native` instead.
```

Now our sample will surely run.

```none
.\saxpy.exe
Calculating y[i] = a * x[i] + y[i] over 1000000 elements.
First 10 elements of the results: [ 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 ]
```

:::
::::
