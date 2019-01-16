# hipify-clang

`hipify-clang` is a clang-based tool to automatically translate CUDA source code into portable HIP C++.

## Table of Contents

<!-- toc -->

- [Supported CUDA APIs](#cuda-apis)
- [Dependencies](#dependencies)
- [Build and install](#build-and-install)
     * [Building](#building)
     * [Testing](#testing)
     * [Linux](#linux)
     * [Windows](#windows)
- [Running and using hipify-clang](#running-and-using-hipify-clang)
- [Disclaimer](#disclaimer)

<!-- tocstop -->

## <a name="cuda-apis"></a> Supported CUDA APIs

- [Runtime API](../docs/markdown/CUDA_Runtime_API_functions_supported_by_HIP.md)
- [Driver API](../docs/markdown/CUDA_Driver_API_functions_supported_by_HIP.md)
- [cuComplex API](../docs/markdown/cuComplex_API_supported_by_HIP.md)
- [cuBLAS](../docs/markdown/CUBLAS_API_supported_by_HIP.md)
- [cuRAND](../docs/markdown/CURAND_API_supported_by_HIP.md)
- [cuDNN](../docs/markdown/CUDNN_API_supported_by_HIP.md)
- [cuFFT](../docs/markdown/CUFFT_API_supported_by_HIP.md)
- [cuSPARSE](../docs/markdown/CUSPARSE_API_supported_by_HIP.md)

## <a name="dependencies"></a> Dependencies

`hipify-clang` requires:
1. LLVM+CLANG of at least version 3.8.0, latest stable and recommended release: 6.0.1 (linux and windows).

2. CUDA at least version 7.0, latest supported version is 9.0.

| **LLVM release version** | **CUDA latest supported version** | **Comments** |
|:------------------------:|:---------------------------------:|:------------:|
| 3.8.0                    | 7.5                               |
| 3.8.1                    | 7.5                               |
| 3.9.0                    | 7.5                               |
| 3.9.1                    | 7.5                               |
| 4.0.0                    | 8.0                               |
| 4.0.1                    | 8.0                               |
| 5.0.0                    | 8.0                               |
| 5.0.1                    | 8.0                               |
| 5.0.2                    | 8.0                               |
| 6.0.0                    | 9.0                               |
| **6.0.1**                | **9.0**                           | **LATEST STABLE RELEASE** |
| 7.0.0                    | 9.2                               | windows is not supported, on linux there is a clang bug: https://bugs.llvm.org/show_bug.cgi?id=36384  |
|                          | 10.0                              | not yet supported |

In most cases, you can get a suitable version of LLVM+CLANG with your package manager.

Failing that or having multiple versions of LLVM, you can [download a release archive](http://releases.llvm.org/), build or install it, and set
[CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/v3.12/variable/CMAKE_PREFIX_PATH.html) so `cmake` can find it; for instance: `-DCMAKE_PREFIX_PATH=f:\LLVM\6.0.1\dist`

## <a name="build-and-install"></a> Build and install

### <a name="building"></a> Build

Assuming this repository is at `./HIP`:

```shell
cd hipify-clang
mkdir build dist
cd build

cmake \
 -DCMAKE_INSTALL_PREFIX=../dist \
 -DCMAKE_BUILD_TYPE=Release \
 ..

make -j install
```
On Windows, the following option should be specified for `cmake` at first place: `-G "Visual Studio 15 2017 Win64"`; the generated `hipify-clang.sln` should be built by `Visual Studio 15 2017` instead of `make.`

Debug build type `-DCMAKE_BUILD_TYPE=Debug` is also supported and tested; `LLVM+CLANG` should be built in `Debug` mode as well.
64 bit build mode `-Thost=x64` is supported as well; `LLVM+CLANG` should be built in 64bit mode as well.

The binary can then be found at `./dist/bin/hipify-clang`.

### <a name="testing"></a> Testing

`hipify-clang` has unit tests using LLVM [`lit`](https://llvm.org/docs/CommandGuide/lit.html)/[`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html).

**LLVM+CLANG should be built from sources, Pre-Built Binaries are not exhaustive for testing.**

To run it:
1. Download [`LLVM`](http://releases.llvm.org/6.0.1/llvm-6.0.1.src.tar.xz)+[`CLANG`](http://releases.llvm.org/6.0.1/cfe-6.0.1.src.tar.xz) sources.
2. Build [`LLVM+CLANG`](http://llvm.org/docs/CMake.html):
   ```shell
   cd llvm
   mkdir build dist
   cd build
   ```
     - **Linux**:

   ```shell
        cmake \
         -DCMAKE_INSTALL_PREFIX=../dist \
         -DLLVM_SOURCE_DIR=../llvm \
         -DCMAKE_BUILD_TYPE=Release \
         ../llvm
        make -j install
   ```
     - **Windows**:

```shell
        cmake \
         -G "Visual Studio 15 2017 Win64" \
         -DCMAKE_INSTALL_PREFIX=../dist \
         -DLLVM_SOURCE_DIR=../llvm \
         -DCMAKE_BUILD_TYPE=Release \
         -Thost=x64 \
         ../llvm
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Run `Visual Studio 15 2017`, open the generated `LLVM.sln`, build all, build project `INSTALL`.


3. Ensure [`CUDA`](https://developer.nvidia.com/cuda-toolkit-archive) of minimum version 7.5 is installed.

    * Having multiple CUDA installations to choose a particular version the `DCUDA_TOOLKIT_ROOT_DIR` option should be specified:

        - Linux: `-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0`

        - Windows: `-DCUDA_TOOLKIT_ROOT_DIR="c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0"`

          `-DCUDA_SDK_ROOT_DIR="c:/ProgramData/NVIDIA Corporation/CUDA Samples/v9.0"`

4. Ensure [`cuDNN`](https://developer.nvidia.com/rdp/cudnn-archive) of version corresponding to CUDA's version is installed.

    * Path to cuDNN should be specified by the `CUDA_DNN_ROOT_DIR` option:

        - Linux: `-DCUDA_DNN_ROOT_DIR=/srv/CUDNN/cudnn-8.0-v7.1`

        - Windows: `-DCUDA_DNN_ROOT_DIR=f:/CUDNN/cudnn-9.0-windows10-x64-v7.1`

5. Ensure [`python`](https://www.python.org/downloads) of minimum required version 2.7 is installed.

6. Ensure `lit` and `FileCheck` are installed - these are distributed with LLVM.

    * Install `lit` into `python`:

        - Linux: `python /srv/git/LLVM/6.0.1/llvm/utils/lit/setup.py install`

        - Windows: `python f:/LLVM/6.0.1/llvm/utils/lit/setup.py install`

    * Starting with LLVM 6.0.1 path to `llvm-lit` python script should be specified by the `LLVM_EXTERNAL_LIT` option:

        - Linux: `-DLLVM_EXTERNAL_LIT=/srv/git/LLVM/6.0.1/build/bin/llvm-lit`

        - Windows: `-DLLVM_EXTERNAL_LIT=f:/LLVM/6.0.1/build/Release/bin/llvm-lit.py`

7. Set `HIPIFY_CLANG_TESTS` option turned on: `-DHIPIFY_CLANG_TESTS=1`.

8. Run `cmake`:
     * [Linux](#linux)
     * [Windows](#windows)

9. Run tests:

     - Linux: `make test-hipify`.

     - Windows: run `Visual Studio 15 2017`, open the generated `hipify-clang.sln`, build project `test-hipify`.

### <a name="linux"></a >Linux

On Linux (Ubuntu 14-18) the following configurations are tested:

LLVM 5.0.0 - 6.0.1, CUDA 8.0, cudnn-8.0

Build system for the above configurations:

Python 2.7 (min), cmake 3.12.3 (min), GNU C/C++ 5.4.0 (min).

Here is an example of building `hipify-clang` with testing support on `Ubuntu 16.04`:

```shell
cmake
 -DHIPIFY_CLANG_TESTS=1 \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=../dist \
 -DCMAKE_PREFIX_PATH=/srv/git/LLVM/6.0.1/dist \
 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
 -DCUDA_DNN_ROOT_DIR=/srv/CUDNN/cudnn-8.0-v7.1 \
 -DLLVM_EXTERNAL_LIT=/srv/git/LLVM/6.0.1/build/bin/llvm-lit \
 ..
```
*A corresponding successful output:*
```shell
-- The C compiler identification is GNU 5.4.0
-- The CXX compiler identification is GNU 5.4.0
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
-- Found LLVM 6.0.1:
--    - CMake module path: /srv/git/LLVM/6.0.1/dist/lib/cmake/llvm
--    - Include path     : /srv/git/LLVM/6.0.1/dist/include
--    - Binary path      : /srv/git/LLVM/6.0.1/dist/bin
-- Linker detection: GNU ld
-- Found PythonInterp: /usr/bin/python2.7 (found suitable version "2.7.12", minimum required is "2.7")
-- Found lit: /usr/local/bin/lit
-- Found FileCheck: /srv/git/LLVM/6.0.1/dist/bin/FileCheck
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Found CUDA: /usr/local/cuda-8.0 (found version "8.0")
-- Configuring done
-- Generating done
-- Build files have been written to: /srv/git/HIP/hipify-clang/build
```
```shell
make test-hipify
```
*A corresponding successful output:*
```shell
[100%] Running HIPify regression tests
-- Testing: 28 tests, 12 threads --
PASS: hipify :: allocators.cu (1 of 28)
PASS: hipify :: coalescing.cu (2 of 28)
PASS: hipify :: cuDNN/cudnn_softmax.cu (3 of 28)
PASS: hipify :: cuFFT/simple_cufft.cu (4 of 28)
PASS: hipify :: cuComplex/cuComplex_Julia.cu (5 of 28)
PASS: hipify :: cuBLAS/cublas_sgemm_matrix_multiplication.cu (6 of 28)
PASS: hipify :: cuBLAS/cublas_1_based_indexing.cu (7 of 28)
PASS: hipify :: cuBLAS/cublas_0_based_indexing.cu (8 of 28)
PASS: hipify :: axpy.cu (9 of 28)
PASS: hipify :: dynamic_shared_memory.cu (10 of 28)
PASS: hipify :: headers_test_01.cu (11 of 28)
PASS: hipify :: headers_test_02.cu (12 of 28)
PASS: hipify :: headers_test_03.cu (13 of 28)
PASS: hipify :: headers_test_05.cu (14 of 28)
PASS: hipify :: cuDNN/cudnn_convolution_forward.cu (15 of 28)
PASS: hipify :: cuRAND/poisson_api_example.cu (16 of 28)
PASS: hipify :: cudaRegister.cu (17 of 28)
PASS: hipify :: headers_test_06.cu (18 of 28)
PASS: hipify :: headers_test_04.cu (19 of 28)
PASS: hipify :: intro.cu (20 of 28)
PASS: hipify :: headers_test_07.cu (21 of 28)
PASS: hipify :: square.cu (22 of 28)
PASS: hipify :: static_shared_memory.cu (23 of 28)
PASS: hipify :: vec_add.cu (24 of 28)
PASS: hipify :: headers_test_08.cu (25 of 28)
PASS: hipify :: cuRAND/benchmark_curand_generate.cpp (26 of 28)
PASS: hipify :: cuRAND/benchmark_curand_kernel.cpp (27 of 28)
PASS: hipify :: headers_test_09.cu (28 of 28)
Testing Time: 1.71s
  Expected Passes    : 28
[100%] Built target test-hipify
```

### <a name="windows"></a >Windows

On Windows the following configurations are tested:

LLVM 6.0.0 - 6.0.1, CUDA 9.0, cudnn-9.0

LLVM 5.0.0 - 5.0.2, CUDA 8.0, cudnn-8.0

Build system for the above configurations:

Python 3.6 (min), cmake 3.12.3 (min), Visual Studio 15.5 2017 (min).

Here is an example of building `hipify-clang` with testing support on `Windows 10` by `Visual Studio 15 2017`:

```shell
cmake
 -G "Visual Studio 15 2017 Win64" \
 -DHIPIFY_CLANG_TESTS=1 \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=../dist \
 -DCMAKE_PREFIX_PATH=f:/LLVM/6.0.1/dist \
 -DCUDA_TOOLKIT_ROOT_DIR="c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0" \
 -DCUDA_SDK_ROOT_DIR="c:/ProgramData/NVIDIA Corporation/CUDA Samples/v9.0" \
 -DCUDA_DNN_ROOT_DIR=f:/CUDNN/cudnn-9.0-windows10-x64-v7.1 \
 -DLLVM_EXTERNAL_LIT=f:/LLVM/6.0.1/build/Release/bin/llvm-lit.py \
 -Thost=x64
 ..
```
*A corresponding successful output:*
```shell
-- Found LLVM 6.0.1:
--    - CMake module path: F:/LLVM/6.0.1/dist/lib/cmake/llvm
--    - Include path     : F:/LLVM/6.0.1/dist/include
--    - Binary path      : F:/LLVM/6.0.1/dist/bin
-- Found PythonInterp: C:/Program Files/Python36/python.exe (found suitable version "3.6.4", minimum required is "3.6")
-- Found lit: C:/Program Files/Python36/Scripts/lit.exe
-- Found FileCheck: F:/LLVM/6.0.1/dist/bin/FileCheck.exe
-- Found CUDA: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0 (found version "9.0")
-- Configuring done
-- Generating done
-- Build files have been written to: f:/HIP/hipify-clang/build
```

## <a name="running-and-using-hipify-clang"></a> Running and using hipify-clang

To process a file, `hipify-clang` needs access to the same headers that would be needed to compile it with clang.

For example:

```shell
./hipify-clang \
  square.cu \
  --cuda-path=/usr/local/cuda-8.0 \
  -I /usr/local/cuda-8.0/samples/common/inc
```

`hipify-clang` arguments are given first, followed by a separator, and then the arguments you'd pass to `clang` if you
were compiling the input file. The [Clang manual for compiling CUDA](https://llvm.org/docs/CompileCudaWithLLVM.html#compiling-cuda-code)
may be useful.

For a list of `hipify-clang` options, run `hipify-clang --help`.

## <a name="disclaimer"></a> Disclaimer

The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD's products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Copyright (c) 2014-2019 Advanced Micro Devices, Inc. All rights reserved.

