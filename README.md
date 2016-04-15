## What is this repository for? ###

HIP allows developers to convert CUDA code to portable C++.  The same source code can be compiled to run on NVIDIA or AMD GPUs. 
Key features include:

* HIP is very thin and has little or no performance impact over coding directly in CUDA or hcc "HC" mode.
* HIP allows coding in a single-source C++ programming language including features such as templates, C++11 lambdas, classes, namespaces, and more.
* HIP allows developers to use the "best" development environment and tools on each target platform.
* The "hipify" tool automatically converts source from CUDA to HIP.
* Developers can specialize for the platform (CUDA or hcc) to tune for performance or handle tricky cases 

New projects can be developed directly in the portable HIP C++ language and can run on either NVIDIA or AMD platforms.  Additionally, HIP provides porting tools which make it easy to port existing CUDA codes to the HIP layer, with no loss of performance as compared to the original CUDA application.  HIP is not intended to be a drop-in replacement for CUDA, and developers should expect to do some manual coding and performance tuning work to complete the port.
## Installation
```
cd HIP-privatestaging
mkdir build
cd build
cmake -DHSA_PATH=/path/to/hsa -DHCC_HOME=/path/to/hcc -DCMAKE_INSTALL_PREFIX=/where/to/install/hip -DCMAKE_BUILD_TYPE=Release ..
make
make install
```
Make sure HIP_PATH is pointed to `/where/to/install/hip` and PATH includes `$HIP_PATH/bin`. This requirement is optional, but required to run any HIP test infrastructure.

## More Info:
- [HIP FAQ](docs/markdown/hip_faq.md)
- [HIP Kernel Language](docs/markdown/hip_kernel_language.md)
- [HIP Runtime API (Doxygen)](http://gpuopen-professionalcompute-tools.github.io/HIP)
- [HIP Porting Guide](docs/markdown/hip_porting_guide.md)
- [HIP Terminology](docs/markdown/hip_terms.md) (including Rosetta Stone of GPU computing terms across CUDA/HIP/HC/AMP/OpenL)
- [Developer/CONTRIBUTING Info](CONTRIBUTING.md)
- [Release Notes](RELEASE.md)


## How do I get set up?

### Prerequisites - Choose Your Platform
HIP code can be developed either on AMD ROCm platform using hcc compiler, or a CUDA platform with nvcc installed:

#### AMD (hcc):

* Install [hcc](https://bitbucket.org/multicoreware/hcc/wiki/Home) including supporting HSA kernel and runtime driver stack 
* By default HIP looks for hcc in /opt/rocm/hcc (can be overridden by setting HCC_HOME environment variable)
* By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by setting HSA_PATH environment variable) 
* Ensure that ROCR runtime is installed and added to LD_LIBRARY_PATH
* Install HIP (from this GitHub repot).  By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).

* Optionally, consider adding /opt/rocm/bin to your path to make it easier to use the tools.
   
#### NVIDIA (nvcc)
* Install CUDA SDK from manufacturer website
* By default HIP looks for CUDA SDK in /usr/local/cuda (can be overriden by setting CUDA_PATH env variable)

```

#### Verify your installation
Run hipconfig (instructions below assume default installation path) :
```
>  /opt/rocm/bin/hipconfig --full
```

Compile and run the [square sample](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/tree/master/samples/0_Intro/square). 



### HCC Options

#### Compiling CodeXL markers for HIP Functions
HIP can generate markers at function begin/end which are displayed on the CodeXL timeline view.  To do this, you need to install CodeXL, tell HIP
where the CodeXL install directory lives, and enable HIP to generate the markers:

1. Install CodeXL
See [CodeXL Download](http://developer.amd.com/tools-and-sdks/opencl-zone/codexl/?webSyncID=9d9c2cb9-3d73-5e65-268a-c7b06428e5e0&sessionGUID=29beacd0-d654-ddc6-a3e2-b9e6c0b0cc77) for the installation file.
Also this [blog](http://gpuopen.com/getting-up-to-speed-with-the-codexl-gpu-profiler-and-radeon-open-compute/) provides more information and tips for using CodeXL.  In addition to installing the CodeXL profiling 
and visualization tools, CodeXL also comes with an SDK that allow applications to add markers to the timeline viewer.  We'll be linking HIP against this library.

2. Set CODEXL_PATH
```
# set to your code-xl installation location:
export CODEXL_PATH=/opt/AMD/CodeXL
```

3. Enable in source code.
In src/hip_hcc.cpp, enable the define 
```
#define COMPILE_TRACE_MARKER 1
```


Then recompile the target application, run with profiler enabled to generate ATP file or trace log.
```
# Use profiler to generate timeline view:
$  $CODEXL_PATH/CodeXLGpuProfiler -A  -o  ./myHipApp  
...
Session output path: /home/me/HIP-privatestaging/tests/b1/mytrace.atp
```

You can also print the HIP function strings to stderr using HIP_TRACE_API environment variable.  This can be useful for tracing application flow.  Also can be combined with the more detailed debug information provided
by the HIP_DB switch.  For example:
```
# Trace to stderr showing begin/end of each function (with arguments) + intermediate debug trace during the execution of each function.
$  HIP_TRACE_API=1  HIP_DB=0x2  ./myHipApp  
```

Note this trace mode uses colors.  "less -r" can handle raw control characters and will display the debug output in proper colors.


#### Using HIP with the AMD Native-GCN compiler.
AMD recently released a direct-to-GCN-ISA target.  This compiler generates GCN ISA directly from LLVM, without going through an intermediate compiler 
IR such as HSAIL or PTX.
The native GCN target is included with upstream LLVM, and has also been integrated with HCC compiler and can be used to compiler HIP programs for AMD.
Here's how to use it with HIP:

- Follow the instructions here to compile the HCC and native LLVM compiler:
> https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA/wiki
> (In the make step for HCC, we recommend setting -DCMAKE_INSTALL_PREFIX=/opt/hcc-native)

Set HCC_HOME environment variable before compiling HIP program to point to the native compiler:
> export HCC_HOME=/opt/hcc-native
```

## Examples and Getting Started:

* A sample and [blog](http://gpuopen.com/hip-to-be-squared-an-introductory-hip-tutorial) that uses hipify to convert a simple app from CUDA to HIP:

 
```shell
> cd samples/01_Intro/square
# follow README / blog steps to hipify the application.
```

* A sample and [blog](http://gpuopen.com/platform-aware-coding-inside-hip/) demonstrating platform specialization:
```shell
> cd samples/01_Intro/bit_extract
> make
```

* Guide to [Porting a New Cuda Project](docs/markdown/hip_porting_guide.md#porting-a-new-cuda-project" aria-hidden="true"><span aria-hidden="true)

 
## More Examples
The GitHub repot [HIP-Examples](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP-Examples.git) contains a hipified vesion of the popular Rodinia benchmark suite.
The README with the procedures and tips the team used during this porting effort is here: [Rodinia Porting Guide](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP-Examples/blob/master/rodinia_3.0/hip/README.hip_porting)

## Tour of the HIP Directories
* **include**: 
    * **hip_runtime_api.h** : Defines HIP runtime APIs and can be compiled with many standard Linux compilers (hcc, GCC, ICC, CLANG, etc), in either C or C++ mode.
    * **hip_runtime.h** : Includes everything in hip_runtime_api.h PLUS hipLaunchKernel and syntax for writing device kernels and device functions.  hip_runtime.h can only be compiled with hcc.
    * **hcc_detail/**** , **nvcc_detail/**** : Implementation details for specific platforms.  HIP applications should not include these files directly.
    * **hcc.h** : Includes interop APIs for HIP and HCC
    
* **bin**: Tools and scripts to help with hip porting
    * **hipify** : Tool to convert CUDA code to portable CPP.  Converts CUDA APIs and kernel builtins.  
    * **hipcc** : Compiler driver that can be used to replace nvcc in existing CUDA code.  hipcc ill call nvcc or hcc depending on platform, and include appropriate platform-specific headers and libraries.
    * **hipconfig** : Print HIP configuration (HIP_PATH, HIP_PLATFORM, CXX config flags, etc)
    * **hipexamine.sh** : Script to scan directory, find all code, and report statistics on how much can be ported with HIP (and identify likely features not yet supported)

* **doc**: Documentation - markdown and doxygen info

## Reporting an issue
Use the [GitHub issue tracker] (https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/issues).
If reporting a bug, include the output of "hipconfig --full" and samples/1_hipInfo/hipInfo (if possible).

