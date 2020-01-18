## Table of Contents

<!-- toc -->

- [Installing pre-built packages](#installing-pre-built-packages)
  * [Prerequisites](#prerequisites)
  * [HIP-hcc](#hip-hcc)
  * [HIP-clang](#hip-clang)
  * [HIP-nvcc](#hip-nvcc)
  * [Verify your installation](#verify-your-installation)
- [Building HIP from source](#building-hip-from-source)
  * [HCC Options](#hcc-options)
    + [Using HIP with the AMD Native-GCN compiler.](#using-hip-with-the-amd-native-gcn-compiler)

<!-- tocstop -->

# Installing pre-built packages

HIP can be easily installed using pre-built binary packages using the package manager for your platform.

## Prerequisites
HIP code can be developed either on AMD ROCm platform using hcc or clang compiler, or a CUDA platform with nvcc installed:

## HIP-hcc

* Add the ROCm package server to your system as per the OS-specific guide available [here](https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories).
* Install the "hip-hcc" package. This will install HCC and the HIP porting layer.
```
apt-get install hip-hcc
```

* Default paths and environment variables:

   * By default HIP looks for hcc in /opt/rocm/hcc (can be overridden by setting HCC_HOME environment variable)
   * By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by setting HSA_PATH environment variable) 
   * By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
   * Optionally, consider adding /opt/rocm/bin to your PATH to make it easier to use the tools.

## HIP-clang

* Using clang to compile HIP program for AMD GPU is under development. Users need to build LLVM, clang, lld, ROCm device library, and HIP from source.

* Install the [rocm](http://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/) packages.  ROCm will install some of the necessary components, including the kernel driver, HSA runtime, etc.

* Build HIP-Clang

```
git clone https://github.com/llvm/llvm-project.git
cd llvm-project/llvm/tools
ln -s clang ../../clang
ln -s lld ../../lld
cd ../..
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" ../llvm
make -j
sudo make install
```

* Build Rocm device library

```
export PATH=/opt/rocm/llvm/bin:$PATH
git clone -b master https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git
cd ROCm-Device-Libs
mkdir -p build && cd build
CC=clang CXX=clang++ cmake -DLLVM_DIR=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_WERROR=1 -DLLVM_ENABLE_ASSERTIONS=1 ..
make -j
sudo make install
```

* Build HIP

```
git clone -b master https://github.com/ROCm-Developer-Tools/HIP.git
cd HIP
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/hip -DHIP_COMPILER=clang -DCMAKE_BUILD_TYPE=Release ..
make -j
sudo make install
```

* Default paths and environment variables:

   * By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by setting HSA_PATH environment variable) 
   * By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
   * By default HIP looks for clang in /opt/rocm/llvm/bin (can be overridden by setting HIP_CLANG_PATH environment variable)
   * By default HIP looks for device library in /opt/rocm/lib (can be overriden by setting DEVICE_LIB_PATH environment variable).
   * Optionally, consider adding /opt/rocm/bin to your PATH to make it easier to use the tools.
   * Optionally, set HIPCC_VERBOSE=7 to output the command line for compilation to make sure clang is used instead of hcc.

## HIP-nvcc
* Add the ROCm package server to your system as per the OS-specific guide available [here](https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories).
* Install the "hip-nvcc" package.  This will install CUDA SDK and the HIP porting layer.
```
apt-get install hip-nvcc
```

* Default paths and environment variables:
   * By default HIP looks for CUDA SDK in /usr/local/cuda (can be overriden by setting CUDA_PATH env variable)
   * By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
   * Optionally, consider adding /opt/rocm/bin to your path to make it easier to use the tools.


## Verify your installation
Run hipconfig (instructions below assume default installation path) :
```shell
/opt/rocm/bin/hipconfig --full
```

Compile and run the [square sample](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square). 


# Building HIP from source
HIP source code is available and the project can be built from source on the HCC platform. 

1. Follow the above steps to install and validate the binary packages.
2. Download HIP source code (from the [GitHub repot](https://github.com/ROCm-Developer-Tools/HIP).)
3. Install HIP build-time dependencies using ```sudo apt-get install libelf-dev```.
4. Build and install HIP (This is the simple version assuming default paths ; see below for additional options.)

By default, HIP uses HCC to compile programs. To use HIP-Clang, add -DHIP_COMPILER=clang to cmake command line.

```
cd HIP
mkdir build
cd build
cmake .. 
make
make install
```

* Default paths:
  * By default cmake looks for hcc in /opt/rocm/hcc (can be overridden by setting ```-DHCC_HOME=/path/to/hcc``` in the cmake step).*
  * By default cmake looks for HSA in /opt/rocm/hsa (can be overridden by setting ```-DHSA_PATH=/path/to/hsa``` in the cmake step).*
  * By default cmake installs HIP to /opt/rocm/hip (can be overridden by setting ```-DCMAKE_INSTALL_PREFIX=/where/to/install/hip``` in the cmake step).*

Here's a richer command-line that overrides the default paths:

```shell
cd HIP
mkdir build
cd build
cmake -DHSA_PATH=/path/to/hsa -DHCC_HOME=/path/to/hcc -DCMAKE_INSTALL_PREFIX=/where/to/install/hip -DCMAKE_BUILD_TYPE=Release ..
make
make install
```

* After installation, make sure HIP_PATH is pointed to `/where/to/install/hip`. 

