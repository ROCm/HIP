# HIPIFY
### Tools to translate CUDA source code into portable HIP C++ automatically
## Table of Contents

<!-- toc -->

- [hipify-clang](#clang)
     * [Dependencies](#dependencies)
     * [Usage](#hipify-clang-usage)
     * [Building](#building)
     * [Testing](#testing)
     * [Linux](#linux)
     * [Windows](#windows)
- [hipify-perl](#perl)
     * [Usage](#hipify-perl-usage)
     * [Building](#hipify-perl-building)
- [Supported CUDA APIs](#cuda-apis)
- [Disclaimer](#disclaimer)

<!-- tocstop -->

## <a name="clang"></a> hipify-clang

`hipify-clang` is a clang-based tool for translation CUDA sources into HIP sources.
It translates CUDA source into an abstract syntax tree, which is being traversed by transformation matchers.
After applying all the matchers, the output HIP source is produced.

**Advantages:**

1. It is a translator; thus, any even very complicated constructs will be parsed successfully, or an error will be reported.
2. It supports clang options like [`-I`](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-i-dir), [`-D`](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-d-macro), [`--cuda-path`](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-cuda-path), etc.
3. Seamless support of new CUDA versions as it is clang's responsibility.
4. Ease in support.

**Disadvantages:**

1. The main advantage is also the main disadvantage: the input CUDA code should be correct; incorrect code wouldn't be translated to HIP.
2. CUDA should be installed and provided in case of multiple installations by `--cuda-path` option.
3. All the includes and defines should be provided to transform code successfully.

### <a name="dependencies"></a> hipify-clang: dependencies

`hipify-clang` requires:

1. [**LLVM+CLANG**](http://releases.llvm.org) of at least version [3.8.0](http://releases.llvm.org/download.html#3.8.0); the latest stable and recommended release: [**9.0.1**](http://releases.llvm.org/download.html#9.0.1), the latest release candidate: [10.0.0-rc2](https://github.com/llvm/llvm-project/releases/tag/llvmorg-10.0.0-rc2).

2. [**CUDA**](https://developer.nvidia.com/cuda-downloads) of at least version [7.0](https://developer.nvidia.com/cuda-toolkit-70), the latest supported version is [**10.1 Update 2**](https://developer.nvidia.com/cuda-10.1-download-archive-base).
To use the latest CUDA version [10.2](https://developer.nvidia.com/cuda-downloads) please use the latest `LLVM` release candidate: [10.0.0-rc2](https://github.com/llvm/llvm-project/releases/tag/llvmorg-10.0.0-rc2).

| **LLVM release version**                                   | **CUDA latest supported version**                                        | **Windows** | **Linux** |
|:----------------------------------------------------------:|:------------------------------------------------------------------------:|:-----------:|:---------:|
| [3.8.0](http://releases.llvm.org/download.html#3.8.0)      | [7.5](https://developer.nvidia.com/cuda-75-downloads-archive)            | +           | +         |
| [3.8.1](http://releases.llvm.org/download.html#3.8.1)      | [7.5](https://developer.nvidia.com/cuda-75-downloads-archive)            | +           | +         |
| [3.9.0](http://releases.llvm.org/download.html#3.9.0)      | [7.5](https://developer.nvidia.com/cuda-75-downloads-archive)            | +           | +         |
| [3.9.1](http://releases.llvm.org/download.html#3.9.1)      | [7.5](https://developer.nvidia.com/cuda-75-downloads-archive)            | +           | +         |
| [4.0.0](http://releases.llvm.org/download.html#4.0.0)      | [8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)         | +           | +         |
| [4.0.1](http://releases.llvm.org/download.html#4.0.1)      | [8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)         | +           | +         |
| [5.0.0](http://releases.llvm.org/download.html#5.0.0)      | [8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)         | +           | +         |
| [5.0.1](http://releases.llvm.org/download.html#5.0.1)      | [8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)         | +           | +         |
| [5.0.2](http://releases.llvm.org/download.html#5.0.2)      | [8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)         | +           | +         |
| [6.0.0](http://releases.llvm.org/download.html#6.0.0)      | [9.0](https://developer.nvidia.com/cuda-90-download-archive)             | +           | +         |
| [6.0.1](http://releases.llvm.org/download.html#6.0.1)      | [9.0](https://developer.nvidia.com/cuda-90-download-archive)             | +           | +         |
| [7.0.0](http://releases.llvm.org/download.html#7.0.0)      | [9.2](https://developer.nvidia.com/cuda-92-download-archive)             | - <br/> not working due to <br/> the clang's bug [38811](https://bugs.llvm.org/show_bug.cgi?id=38811) <br/>+<br/>[patch](patches/patch_for_clang_7.0.0_bug_38811.zip)*</br> | - <br/> not working due to <br/> the clang's bug [36384](https://bugs.llvm.org/show_bug.cgi?id=36384) |
| [7.0.1](http://releases.llvm.org/download.html#7.0.1)      | [9.2](https://developer.nvidia.com/cuda-92-download-archive)             | - <br/> not working due to <br/> the clang's bug [38811](https://bugs.llvm.org/show_bug.cgi?id=38811) <br/>+<br/>[patch](patches/patch_for_clang_7.0.1_bug_38811.zip)*</br> | - <br/> not working due to <br/> the clang's bug [36384](https://bugs.llvm.org/show_bug.cgi?id=36384) |
| [7.1.0](http://releases.llvm.org/download.html#7.1.0)      | [9.2](https://developer.nvidia.com/cuda-92-download-archive)             | - <br/> not working due to <br/> the clang's bug [38811](https://bugs.llvm.org/show_bug.cgi?id=38811) <br/>+<br/>[patch](patches/patch_for_clang_7.1.0_bug_38811.zip)*</br> | - <br/> not working due to <br/> the clang's bug [36384](https://bugs.llvm.org/show_bug.cgi?id=36384) |
| [8.0.0](http://releases.llvm.org/download.html#8.0.0)      | [10.0](https://developer.nvidia.com/cuda-10.0-download-archive)          | - <br/> not working due to <br/> the clang's bug [38811](https://bugs.llvm.org/show_bug.cgi?id=38811) <br/>+<br/>[patch](patches/patch_for_clang_8.0.0_bug_38811.zip)*</br> | + |
| [8.0.1](http://releases.llvm.org/download.html#8.0.1)      | [10.0](https://developer.nvidia.com/cuda-10.0-download-archive)          | - <br/> not working due to <br/> the clang's bug [38811](https://bugs.llvm.org/show_bug.cgi?id=38811) <br/>+<br/>[patch](patches/patch_for_clang_8.0.1_bug_38811.zip)*</br> | + |
| [9.0.0](http://releases.llvm.org/download.html#9.0.0)      | [10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)     | +           | +         |
| [**9.0.1**](http://releases.llvm.org/download.html#9.0.1)  | [**10.1**](https://developer.nvidia.com/cuda-10.1-download-archive-base) | + <br/> **LATEST STABLE RELEASE** | + <br/> **LATEST STABLE RELEASE** |
| [10.0.0-rc2](https://github.com/llvm/llvm-project/releases/tag/llvmorg-10.0.0-rc2) | [10.2](https://developer.nvidia.com/cuda-downloads)             | +           | +         |

`*` Download the patch and unpack it into your `LLVM` distributive directory; a few header files will be overwritten; rebuilding of `LLVM` is not needed.

In most cases, you can get a suitable version of `LLVM+CLANG` with your package manager.

Failing that or having multiple versions of `LLVM`, you can [download a release archive](http://releases.llvm.org/), build or install it, and set
[CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/v3.5/variable/CMAKE_PREFIX_PATH.html) so `cmake` can find it; for instance: `-DCMAKE_PREFIX_PATH=f:\LLVM\9.0.1\dist`

### <a name="hipify-clang-usage"></a> hipify-clang: usage

To process a file, `hipify-clang` needs access to the same headers that would be required to compile it with clang.

For example:

```shell
./hipify-clang square.cu --cuda-path=/usr/local/cuda-10.1 -I /usr/local/cuda-10.1/samples/common/inc
```

`hipify-clang` arguments are given first, followed by a separator `'--'`, and then the arguments you'd pass to `clang` if you
were compiling the input file. For example:

```bash
./hipify-clang cpp17.cu --cuda-path=/usr/local/cuda-10.1 -- -std=c++17
```

The [Clang manual for compiling CUDA](https://llvm.org/docs/CompileCudaWithLLVM.html#compiling-cuda-code) may be useful.

For a list of `hipify-clang` options, run `hipify-clang --help`.

### <a name="building"></a> hipify-clang: building

Assuming this repository is at `./HIP`:

```bash
cd hipify-clang
mkdir build dist
cd build

cmake \
 -DCMAKE_INSTALL_PREFIX=../dist \
 -DCMAKE_BUILD_TYPE=Release \
 ..

make -j install
```
On Windows, the following option should be specified for `cmake` at first place: `-G "Visual Studio 16 2019 Win64"`; the generated `hipify-clang.sln` should be built by `Visual Studio 16 2019` instead of `make.`
Please, see [hipify-clang: Windows](#windows) for the supported tools for building.

Debug build type `-DCMAKE_BUILD_TYPE=Debug` is also supported and tested; `LLVM+CLANG` should be built in `Debug` mode as well.
64-bit build mode (`-Thost=x64` on Windows) is also supported; `LLVM+CLANG` should be built in 64-bit mode as well.

The binary can then be found at `./dist/bin/hipify-clang`.

### <a name="testing"></a> hipify-clang: testing

`hipify-clang` has unit tests using `LLVM` [`lit`](https://llvm.org/docs/CommandGuide/lit.html)/[`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html).

`LLVM+CLANG` should be built from sources, pre-built binaries are not exhaustive for testing.

**LLVM 9.0.1 or older:**

1. download [`LLVM`](http://releases.llvm.org/9.0.1/llvm-9.0.1.src.tar.xz)+[`CLANG`](http://releases.llvm.org/9.0.1/cfe-9.0.1.src.tar.xz) sources; 
2. build [`LLVM+CLANG`](http://releases.llvm.org/9.0.0/docs/CMake.html):

 **Linux**:
   ```bash
        cmake \
         -DCMAKE_INSTALL_PREFIX=../dist \
         -DLLVM_SOURCE_DIR=../llvm \
         -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
         -DCMAKE_BUILD_TYPE=Release \
         ../llvm
        make -j install
   ```
 **Windows**:
   ```shell
        cmake \
         -G "Visual Studio 16 2019" \
         -A x64 \
         -DCMAKE_INSTALL_PREFIX=../dist \
         -DLLVM_SOURCE_DIR=../llvm \
         -DLLVM_TARGETS_TO_BUILD="NVPTX" \
         -DCMAKE_BUILD_TYPE=Release \
         -Thost=x64 \
         ../llvm
   ```
Run `Visual Studio 16 2019`, open the generated `LLVM.sln`, build all, build project `INSTALL`.

**LLVM 10.0.0 or newer:**

1. download [`LLVM project`](https://github.com/llvm/llvm-project/archive/llvmorg-10.0.0-rc2.tar.gz) sources;
2. build [`LLVM project`](http://llvm.org/docs/CMake.html):

 **Linux**:
   ```bash
        cmake \
         -DCMAKE_INSTALL_PREFIX=../dist \
         -DLLVM_SOURCE_DIR=../llvm-project \
         -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
         -DLLVM_ENABLE_PROJECTS="clang" \
         -DCMAKE_BUILD_TYPE=Release \
         ../llvm-project/llvm
        make -j install
   ```
 **Windows**:
   ```shell
        cmake \
         -G "Visual Studio 16 2019" \
         -A x64 \
         -DCMAKE_INSTALL_PREFIX=../dist \
         -DLLVM_SOURCE_DIR=../llvm-project \
         -DLLVM_TARGETS_TO_BUILD="NVPTX" \
         -DLLVM_ENABLE_PROJECTS="clang" \
         -DCMAKE_BUILD_TYPE=Release \
         -Thost=x64 \
         ../llvm-project/llvm
   ```
Run `Visual Studio 16 2019`, open the generated `LLVM.sln`, build all, build project `INSTALL`.

3. Ensure [`CUDA`](https://developer.nvidia.com/cuda-toolkit-archive) of minimum version 7.0 is installed.

    * Having multiple CUDA installations to choose a particular version the `DCUDA_TOOLKIT_ROOT_DIR` option should be specified:

        - ***Linux***: `-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1`

        - ***Windows***: `-DCUDA_TOOLKIT_ROOT_DIR="c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1"`

          `-DCUDA_SDK_ROOT_DIR="c:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1"`

4. Ensure [`cuDNN`](https://developer.nvidia.com/rdp/cudnn-archive) of the version corresponding to CUDA's version is installed.

    * Path to cuDNN should be specified by the `CUDA_DNN_ROOT_DIR` option:

        - ***Linux***: `-DCUDA_DNN_ROOT_DIR=/srv/CUDNN/cudnn-10.1-v7.6.5.32`

        - ***Windows***: `-DCUDA_DNN_ROOT_DIR=f:/CUDNN/cudnn-10.1-windows10-x64-v7.6.5.32`

5. Ensure [`CUB`](https://github.com/NVlabs/cub) of the version corresponding to CUDA's version is installed.

    * Path to CUB should be specified by the `CUDA_CUB_ROOT_DIR` option:

        - ***Linux***: `-DCUDA_CUB_ROOT_DIR=/srv/git/CUB`

        - ***Windows***: `-DCUDA_CUB_ROOT_DIR=f:/GIT/cub`

5. Ensure [`python`](https://www.python.org/downloads) of minimum required version 2.7 is installed.

6. Ensure `lit` and `FileCheck` are installed - these are distributed with `LLVM`.

    * Install `lit` into `python`:

        - ***Linux***: `python /srv/git/LLVM/9.0.1/llvm/utils/lit/setup.py install`

        - ***Windows***: `python f:/LLVM/9.0.1/llvm/utils/lit/setup.py install`

    * Starting with LLVM 6.0.1 path to `llvm-lit` python script should be specified by the `LLVM_EXTERNAL_LIT` option:

        - ***Linux***: `-DLLVM_EXTERNAL_LIT=/srv/git/LLVM/9.0.1/build/bin/llvm-lit`

        - ***Windows***: `-DLLVM_EXTERNAL_LIT=f:/LLVM/9.0.1/build/Release/bin/llvm-lit.py`

    * `FileCheck`:

        - ***Linux***: copy from `/srv/git/LLVM/9.0.1/build/bin/` to `CMAKE_INSTALL_PREFIX/dist/bin`

        - ***Windows***: copy from `f:/LLVM/9.0.1/build/Release/bin` to `CMAKE_INSTALL_PREFIX/dist/bin`

        - Or specify the path to `FileCheck` in `CMAKE_INSTALL_PREFIX` option

7. Set `HIPIFY_CLANG_TESTS` option turned on: `-DHIPIFY_CLANG_TESTS=1`.

8. Build and run tests:

### <a name="Linux"></a > hipify-clang: Linux

On Linux the following configurations are tested:

Ubuntu 14: LLVM 5.0.0 - 6.0.1, CUDA 7.0 - 9.0, cudnn-5.0.5 - cudnn-7.6.5.32

Ubuntu 16-18: LLVM 8.0.0 - 10.0.0-rc2, CUDA 8.0 - 10.2, cudnn-5.1.10 - cudnn-7.6.5.32

Minimum build system requirements for the above configurations:

Python 2.7, cmake 3.5.1, GNU C/C++ 5.4.0.

Here is an example of building `hipify-clang` with testing support on `Ubuntu 16.04`:

```bash
cmake
 -DHIPIFY_CLANG_TESTS=1 \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=../dist \
 -DCMAKE_PREFIX_PATH=/srv/git/LLVM/9.0.1/dist \
 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1 \
 -DCUDA_DNN_ROOT_DIR=/srv/CUDNN/cudnn-10.1-v7.6.5.32 \
 -DCUDA_CUB_ROOT_DIR=/srv/git/CUB \
 -DLLVM_EXTERNAL_LIT=/srv/git/LLVM/9.0.1/build/bin/llvm-lit \
 ..
```
*A corresponding successful output:*
```shell
-- The C compiler identification is GNU 7.4.0
-- The CXX compiler identification is GNU 7.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found LLVM 9.0.1:
--    - CMake module path: /srv/git/LLVM/9.0.1/dist/lib/cmake/llvm
--    - Include path     : /srv/git/LLVM/9.0.1/dist/include
--    - Binary path      : /srv/git/LLVM/9.0.1/dist/bin
-- Linker detection: GNU ld
-- Found PythonInterp: /usr/bin/python2.7 (found suitable version "2.7.12", minimum required is "2.7")
-- Found lit: /usr/local/bin/lit
-- Found FileCheck: /srv/git/LLVM/9.0.1/dist/bin/FileCheck
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Found CUDA: /usr/local/cuda-10.1 (found version "10.1")
-- Configuring done
-- Generating done
-- Build files have been written to: /srv/git/HIP/hipify-clang/build
```
```shell
make test-hipify
```
*A corresponding successful output:*
```shell
Running HIPify regression tests
========================================
CUDA 10.1 - will be used for testing
LLVM 9.0.1 - will be used for testing
x86_64 - Platform architecture
Linux 5.2.0 - Platform OS
64 - hipify-clang binary bitness
64 - python 2.7.12 binary bitness
========================================
-- Testing: 67 tests, 12 threads --
PASS: hipify :: unit_tests/casts/reinterpret_cast.cu (1 of 67)
PASS: hipify :: unit_tests/device/math_functions.cu (2 of 67)
PASS: hipify :: unit_tests/device/atomics.cu (3 of 67)
PASS: hipify :: unit_tests/device/device_symbols.cu (4 of 67)
PASS: hipify :: unit_tests/headers/headers_test_01.cu (5 of 67)
PASS: hipify :: unit_tests/headers/headers_test_02.cu (6 of 67)
PASS: hipify :: unit_tests/headers/headers_test_03.cu (7 of 67)
PASS: hipify :: unit_tests/headers/headers_test_05.cu (8 of 67)
PASS: hipify :: unit_tests/headers/headers_test_04.cu (9 of 67)
PASS: hipify :: unit_tests/headers/headers_test_06.cu (10 of 67)
PASS: hipify :: unit_tests/headers/headers_test_07.cu (11 of 67)
PASS: hipify :: unit_tests/headers/headers_test_10.cu (12 of 67)
PASS: hipify :: unit_tests/headers/headers_test_11.cu (13 of 67)
PASS: hipify :: unit_tests/headers/headers_test_08.cu (14 of 67)
PASS: hipify :: unit_tests/kernel_launch/kernel_launch_01.cu (15 of 67)
PASS: hipify :: unit_tests/headers/headers_test_09.cu (16 of 67)
PASS: hipify :: unit_tests/libraries/CAFFE2/caffe2_02.cu (17 of 67)
PASS: hipify :: unit_tests/libraries/CAFFE2/caffe2_01.cu (18 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/cublas_0_based_indexing.cu (19 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/cublas_1_based_indexing.cu (20 of 67)
PASS: hipify :: unit_tests/libraries/CUB/cub_03.cu (21 of 67)
PASS: hipify :: unit_tests/libraries/CUB/cub_01.cu (22 of 67)
PASS: hipify :: unit_tests/libraries/CUB/cub_02.cu (23 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/rocBLAS/cublas_0_based_indexing_rocblas.cu (24 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/cublas_sgemm_matrix_multiplication.cu (25 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/rocBLAS/cublas_1_based_indexing_rocblas.cu (26 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/rocBLAS/cublas_sgemm_matrix_multiplication_rocblas.cu (27 of 67)
PASS: hipify :: unit_tests/libraries/cuComplex/cuComplex_Julia.cu (28 of 67)
PASS: hipify :: unit_tests/libraries/cuFFT/simple_cufft.cu (29 of 67)
PASS: hipify :: unit_tests/libraries/cuDNN/cudnn_softmax.cu (30 of 67)
PASS: hipify :: unit_tests/libraries/cuDNN/cudnn_convolution_forward.cu (31 of 67)
PASS: hipify :: unit_tests/libraries/cuRAND/poisson_api_example.cu (32 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_01.cu (33 of 67)
PASS: hipify :: unit_tests/libraries/cuRAND/benchmark_curand_generate.cpp (34 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_02.cu (35 of 67)
PASS: hipify :: unit_tests/libraries/cuRAND/benchmark_curand_kernel.cpp (36 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_03.cu (37 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_04.cu (38 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_05.cu (39 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_07.cu (40 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_06.cu (41 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_08.cu (42 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_09.cu (43 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_11.cu (44 of 67)
PASS: hipify :: unit_tests/namespace/ns_kernel_launch.cu (45 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_10.cu (46 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_12.cu (47 of 67)
PASS: hipify :: unit_tests/pp/pp_if_else_conditionals.cu (48 of 67)
PASS: hipify :: unit_tests/pp/pp_if_else_conditionals_01.cu (49 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/11_texture_driver/tex2dKernel.cpp (50 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/0_MatrixTranspose/MatrixTranspose.cpp (51 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/11_texture_driver/texture2dDrv.cpp (52 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/13_occupancy/occupancy.cpp (53 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/1_hipEvent/hipEvent.cpp (54 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/2_Profiler/Profiler.cpp (55 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/7_streams/stream.cpp (56 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/8_peer2peer/peer2peer.cpp (57 of 67)
PASS: hipify :: unit_tests/samples/MallocManaged.cpp (58 of 67)
PASS: hipify :: unit_tests/samples/allocators.cu (59 of 67)
PASS: hipify :: unit_tests/samples/coalescing.cu (60 of 67)
PASS: hipify :: unit_tests/samples/dynamic_shared_memory.cu (61 of 67)
PASS: hipify :: unit_tests/samples/axpy.cu (62 of 67)
PASS: hipify :: unit_tests/samples/intro.cu (63 of 67)
PASS: hipify :: unit_tests/samples/cudaRegister.cu (64 of 67)
PASS: hipify :: unit_tests/samples/square.cu (65 of 67)
PASS: hipify :: unit_tests/samples/static_shared_memory.cu (66 of 67)
PASS: hipify :: unit_tests/samples/vec_add.cu (67 of 67)
Testing Time: 3.07s
  Expected Passes    : 67
[100%] Built target test-hipify
```
### <a name="windows"></a > hipify-clang: Windows

*Tested configurations:*

|    **LLVM**    | **CUDA** |      **cuDNN**      |     **Visual Studio**     | **cmake** | **Python** |
|:--------------:|---------:|--------------------:|--------------------------:|----------:|-----------:|
| 5.0.0 - 5.0.2  | 8.0      | 5.1.10 - 7.1.4.18   | 2017.15.5.2               | 3.5.1     | 3.6.4      |
| 6.0.0 - 6.0.1  | 9.0      | 7.0.5.15 - 7.6.5.32 | 2017.15.5.5               | 3.6.0     | 3.7.2      |
| 7.0.0 - 7.1.0  | 9.2      | 7.6.5.32            | 2017.15.9.11              | 3.13.3    | 3.7.3      |
| 8.0.0 - 8.0.1  | 10.0     | 7.6.5.32            | 2017.15.9.15              | 3.14.2    | 3.7.4      |
| 9.0.0 - 9.0.1  | 10.1     | 7.6.5.32            | 2017.15.9.20, 2019.16.4.5 | 3.16.4    | 3.8.0      |
| 10.0.0-rc1,rc2 | 10.2     | 7.6.5.32            | 2017.15.9.20, 2019.16.4.5 | 3.16.4    | 3.8.1      |
| 11.0.0git      | 10.2     | 7.6.5.32            | 2017.15.9.20, 2019.16.4.5 | 3.16.4    | 3.8.1      |

*Building with testing support on `Windows 10` by `Visual Studio 16 2019`:*

```shell
cmake
 -G "Visual Studio 16 2019" \
 -A x64 \
 -DHIPIFY_CLANG_TESTS=1 \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=../dist \
 -DCMAKE_PREFIX_PATH=f:/LLVM/9.0.1/dist \
 -DCUDA_TOOLKIT_ROOT_DIR="c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1" \
 -DCUDA_SDK_ROOT_DIR="c:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1" \
 -DCUDA_DNN_ROOT_DIR=f:/CUDNN/cudnn-10.1-windows10-x64-v7.6.5.32 \
 -DCUDA_CUB_ROOT_DIR=f:/GIT/cub \
 -DLLVM_EXTERNAL_LIT=f:/LLVM/9.0.1/build/Release/bin/llvm-lit.py \
 -Thost=x64
 ..
```
*A corresponding successful output:*
```shell
-- Found LLVM 9.0.1:
--    - CMake module path: F:/LLVM/9.0.1/dist/lib/cmake/llvm
--    - Include path     : F:/LLVM/9.0.1/dist/include
--    - Binary path      : F:/LLVM/9.0.1/dist/bin
-- Found PythonInterp: C:/Program Files/Python38/python.exe (found suitable version "3.8.1", minimum required is "3.6")
-- Found lit: C:/Program Files/Python38/Scripts/lit.exe
-- Found FileCheck: F:/LLVM/9.0.1/dist/bin/FileCheck.exe
-- Found CUDA: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1 (found version "10.1")
-- Configuring done
-- Generating done
-- Build files have been written to: f:/HIP/hipify-clang/build
```

Run `Visual Studio 16 2019`, open the generated `hipify-clang.sln`, build project `test-hipify`.

## <a name="perl"></a> hipify-perl

`hipify-perl` is autogenerated perl-based script which heavily uses regular expressions.

**Advantages:**

1. Ease in use.

2. It doesn't check the input source CUDA code for correctness.

3. It doesn't have dependencies on 3rd party tools, including CUDA.

**Disadvantages:**

1. Current disability (and difficulty in implementing) of transforming the following constructs:

    * macros expansion;

    * namespaces:

        - redefines of CUDA entities in user namespaces;

        - using directive;

    * templates (some cases);

    * device/host function calls distinguishing;

    * header files correct injection;

    * complicated argument lists parsing.

2. Difficulties in supporting.

### <a name="hipify-perl-usage"></a> hipify-perl: usage

```shell
perl hipify-perl square.cu > square.cu.hip
```

### <a name="hipify-perl-building"></a> hipify-perl: building

To generate `hipify-perl`, run `hipify-clang --perl`. Output directory for the generated `hipify-perl` file might be specified by `--o-hipify-perl-dir` option.

## <a name="cuda-apis"></a> Supported CUDA APIs

- [Runtime API](../docs/markdown/CUDA_Runtime_API_functions_supported_by_HIP.md)
- [Driver API](../docs/markdown/CUDA_Driver_API_functions_supported_by_HIP.md)
- [cuComplex API](../docs/markdown/cuComplex_API_supported_by_HIP.md)
- [cuBLAS](../docs/markdown/CUBLAS_API_supported_by_HIP.md)
- [cuRAND](../docs/markdown/CURAND_API_supported_by_HIP.md)
- [cuDNN](../docs/markdown/CUDNN_API_supported_by_HIP.md)
- [cuFFT](../docs/markdown/CUFFT_API_supported_by_HIP.md)
- [cuSPARSE](../docs/markdown/CUSPARSE_API_supported_by_HIP.md)

## <a name="disclaimer"></a> Disclaimer

The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD's products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Copyright (c) 2014-2020 Advanced Micro Devices, Inc. All rights reserved.
