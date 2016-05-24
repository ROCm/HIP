<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Installation** 

- [Installing pre-built packages:](#installing-pre-built-packages)
  - [Prerequisites](#prerequisites)
  - [AMD (hcc)](#amd-hcc)
  - [NVIDIA (nvcc)](#nvidia-nvcc)
  - [Verify your installation](#verify-your-installation)
- [Building HIP from source](#building-hip-from-source)
  - [HCC Options](#hcc-options)
    - [Using HIP with the AMD Native-GCN compiler.](#using-hip-with-the-amd-native-gcn-compiler)
    - [Compiling CodeXL markers for HIP Functions](#compiling-codexl-markers-for-hip-functions)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Installing pre-built packages

HIP can be easily installed using pre-built binary packages using the package manager for your platform.

## Prerequisites
HIP code can be developed either on AMD ROCm platform using hcc compiler, or a CUDA platform with nvcc installed:

## AMD-hcc

* Install the [rocm](http://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/) packages.  Rocm will install all of the necessary components, including the kernel driver, runtime software, HCC compiler, and HIP.

* Default paths and environment variables:

   * By default HIP looks for hcc in /opt/rocm/hcc (can be overridden by setting HCC_HOME environment variable)
   * By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by setting HSA_PATH environment variable) 
   * By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
   * Optionally, consider adding /opt/rocm/bin to your path to make it easier to use the tools.


## NVIDIA-nvcc
* Configure the additional package server as described [here](http://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/).  
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

Compile and run the [square sample](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/tree/master/samples/0_Intro/square). 


# Building HIP from source
HIP source code is available and the project can be built from source on the HCC platform. 

1. Follow the above steps to install and validate the binary packages.
2. Download HIP source code (from the [GitHub repot](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP).)
3. Build and install HIP (This is the simple version assuming default paths ; see below for additional options.)
```
cd HIP-privatestaging
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
cd HIP-privatestaging
mkdir build
cd build
cmake -DHSA_PATH=/path/to/hsa -DHCC_HOME=/path/to/hcc -DCMAKE_INSTALL_PREFIX=/where/to/install/hip -DCMAKE_BUILD_TYPE=Release ..
make
make install
```

* After installation, make sure HIP_PATH is pointed to `/where/to/install/hip`. 

## HCC Options

### Using HIP with the AMD Native-GCN compiler.
AMD recently released a direct-to-GCN-ISA target.  This compiler generates GCN ISA directly from LLVM, without going through an intermediate compiler 
IR such as HSAIL or PTX.
The native GCN target is included with upstream LLVM, and has also been integrated with HCC compiler and can be used to compiler HIP programs for AMD.
Binary packages for the direct-to-isa package are included with the [rocm](http://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/) package. 
Alternatively, this sections describes how to build it from source: 

1. Install the rocm packages as described above.
2. Follow the instructions [here](https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA/wiki)
   * In the make step for HCC, we recommend setting -DCMAKE_INSTALL_PREFIX.  
   * Set HCC_HOME environment variable before compiling HIP program to point to the native compiler:
```shell
export HCC_HOME=/path/to/native/hcc
```


### Compiling CodeXL markers for HIP Functions
HIP can generate markers at function begin/end which are displayed on the CodeXL timeline view.  To do this, you need to install CodeXL, tell HIP
where the CodeXL install directory lives, and enable HIP to generate the markers:

1. Install CodeXL
See [CodeXL Download](http://developer.amd.com/tools-and-sdks/opencl-zone/codexl/?webSyncID=9d9c2cb9-3d73-5e65-268a-c7b06428e5e0&sessionGUID=29beacd0-d654-ddc6-a3e2-b9e6c0b0cc77) for the installation file.
Also this [blog](http://gpuopen.com/getting-up-to-speed-with-the-codexl-gpu-profiler-and-radeon-open-compute/) provides more information and tips for using CodeXL.  In addition to installing the CodeXL profiling 
and visualization tools, CodeXL also comes with an SDK that allow applications to add markers to the timeline viewer.  We'll be linking HIP against this library.

2. Set CODEXL_PATH
```shell
# set to your code-xl installation location:
export CODEXL_PATH=/opt/AMD/CodeXL
```

3. Enable in source code.
In src/hip_hcc.cpp, enable the define 
```c
#define COMPILE_TRACE_MARKER 1
```


Then recompile the target application, run with profiler enabled to generate ATP file or trace log.
```shell
# Use profiler to generate timeline view:
$CODEXL_PATH/CodeXLGpuProfiler -A  -o  ./myHipApp  
...
Session output path: /home/me/HIP-privatestaging/tests/b1/mytrace.atp
```

You can also print the HIP function strings to stderr using HIP_TRACE_API environment variable.  This can be useful for tracing application flow.  Also can be combined with the more detailed debug information provided
by the HIP_DB switch.  For example:
```shell
# Trace to stderr showing begin/end of each function (with arguments) + intermediate debug trace during the execution of each function.
HIP_TRACE_API=1 HIP_DB=0x2 ./myHipApp  
```

Note this trace mode uses colors.  "less -r" can handle raw control characters and will display the debug output in proper colors.
