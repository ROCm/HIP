## Table of Contents

<!-- toc -->

- [Installing pre-built packages](#installing-pre-built-packages)
  * [Prerequisites](#prerequisites)
  * [AMD-hcc](#amd-hcc)
  * [AMD-clang](#amd-clang)
  * [NVIDIA-nvcc](#nvidia-nvcc)
  * [Verify your installation](#verify-your-installation)
- [Building HIP from source](#building-hip-from-source)
  * [HCC Options](#hcc-options)
    + [Using HIP with the AMD Native-GCN compiler.](#using-hip-with-the-amd-native-gcn-compiler)

<!-- tocstop -->

# Installing pre-built packages

HIP can be easily installed using pre-built binary packages using the package manager for your platform.

## Prerequisites
HIP code can be developed either on AMD ROCm platform using hcc or clang compiler, or a CUDA platform with nvcc installed:

## AMD-hcc

* Add the ROCm package server to your system as per the OS-specific guide available [here](https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories).
* Install the "hip_hcc" package. This will install HCC and the HIP porting layer.
```
apt-get install hip_hcc
```

* Default paths and environment variables:

   * By default HIP looks for hcc in /opt/rocm/hcc (can be overridden by setting HCC_HOME environment variable)
   * By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by setting HSA_PATH environment variable) 
   * By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
   * Optionally, consider adding /opt/rocm/bin to your PATH to make it easier to use the tools.

## AMD-clang

* Using clang to compile HIP program for AMD GPU is under development. Users need to build LLVM, clang, lld, ROCm device library, and HIP from source.

* Install the [rocm](http://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/) packages.  ROCm will install some of the necessary components, including the kernel driver, HSA runtime, etc.

* Build LLVM/clang/lld by using the following repository and branch and following the general LLVM/clang build procedure:

   * LLVM: https://github.com/RadeonOpenCompute/llvm.git amd-common branch
   * clang: https://github.com/RadeonOpenCompute/clang checkout amd-hip-upstream branch, then merge with amd-common branch to match LLVM/lld
   * lld: https://github.com/RadeonOpenCompute/lld amd-common branch
   
* Build Rocm device library

   * Checkout https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git amd-hip branch and build it with clang built from the last step.
   
* Build HIP

   * Checkout https://github.com/ROCm-Developer-Tools/HIP.git hip-clang branch and build it with HCC installed with ROCm packages.
   
* Environment variables to let hipcc to use clang to compile HIP program

   By default hipcc uses hcc to compile HIP program for AMD GPU. To let hipcc to use clang to compile HIP program, the following environment variables must be set:

   * HIP_CLANG_PATH - Path to clang   
   * DEVICE_LIB_PATH - Path to the device library
   
* Default paths and environment variables:

   * By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by setting HSA_PATH environment variable) 
   * By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
   * Optionally, consider adding /opt/rocm/bin to your PATH to make it easier to use the tools.
   * Optionally, set HIPCC_VERBOSE=7 to output the command line for compilation to make sure clang is used instead of hcc.

## NVIDIA-nvcc
* Add the ROCm package server to your system as per the OS-specific guide available [here](https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories).
* Install the "hip_nvcc" package.  This will install CUDA SDK and the HIP porting layer.
```
apt-get install hip_nvcc
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

