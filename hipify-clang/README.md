# hipify-clang

`hipify-clang` is a clang-based tool to automatically translate CUDA source code into portable HIP C++.

## Table of Contents

<!-- toc -->

- [Using hipify-clang](#using-hipify-clang)
  * [Build and install](#build-and-install)
  * [Running and using hipify-clang](#running-and-using-hipify-clang)
- [Disclaimer](#disclaimer)

<!-- tocstop -->

## Build and install

### Dependencies

`hipify-clang` requires clang+llvm of at least version 3.8.

In most cases, you can get a suitable version of clang+llvm with your package manager.

Failing that, you can [download a release archive](http://releases.llvm.org/), extract it somewhere, and set
[CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/v3.0/variable/CMAKE_PREFIX_PATH.html) so `cmake` can find it.

### Build

Assuming this repository is at `./HIP`:

```shell
mkdir build inst

cd build
cmake \
 -DCMAKE_INSTALL_PREFIX=../inst \
 -DCMAKE_BUILD_TYPE=Release \
 -DBUILD_HIPIFY_CLANG=ON \
 ../HIP

make -j install
```

The binary can then be found at `./inst/bin/hipify-clang`.

### Test

`hipify-clang` has unit tests using LLVM [`lit`](https://llvm.org/docs/CommandGuide/lit.html)/[`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html).

To run it:
1. Ensure `lit` and `FileCheck` are installed - these are distributed with LLVM.
2. Ensure `socat` is installed - your distro almost certainly has a package for this.
3. Build with the `HIPIFY_CLANG_TESTS` option turned on.
4. `make test-hipify`

## Running and using hipify-clang

To process a file, `hipify-clang` needs access to the same headers that would be needed to compile it with clang.

For example:

```shell
hipify-clang square.cu -- \
  -x cuda \
  --cuda-path=/opt/cuda \
  --cuda-gpu-arch=sm_30 \
  -isystem /opt/cuda/samples/common/inc
```

`hipify-clang` arguments are given first, followed by a separator, and then the arguments you'd pass to `clang` if you
were compiling the input file. The [Clang manual for compiling CUDA](https://llvm.org/docs/CompileCudaWithLLVM.html#compiling-cuda-code)
may be useful.

For a list of `hipify-clang` options, run `hipify-clang --help`.

## Disclaimer

The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD's products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Copyright (c) 2014-2017 Advanced Micro Devices, Inc. All rights reserved.

