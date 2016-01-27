## What is this repository for? ###

HIP allows developers to convert CUDA code to portable C++.  The same source code can be compiled to run on NVIDIA or AMD GPUs. 
Key features include:

* HIP is very thin and has little or no performance impact over coding directly in CUDA or HCC "HC" mode.
* HIP allows coding in a single-source C++ programming language including features such as templates, C++11 lambdas, classes, namespaces, and more.
* HIP allows developers to use the "best" development environment and tools on each target platform.
* The "hipify" script automatically converts source from CUDA to HIP.
* Developers can specialize for the platform (CUDA or HCC) to tune for performance or handle tricky cases 

HIP is intended to support new developments in a portable C++ language that can run on either NVIDIA or AMD platforms.  Additionally, HIP provides porting tools which make it easy to port existing CUDA codes to the portable HIP layer, with no loss of performance as compared to the original CUDA application.  HIP is not intended to be a drop-in replacement for CUDA, and developers should expect to do some manual coding and performance tuning work to complete the port.

## More Info:
- [HIP FAQ](docs/markdown/hip_faq.md)
- [HIP Kernel Language](docs/markdown/hip_kernel_language.md)
- [HIP Runtime API (Doxygen)](docs/RuntimeAPI/html/group__API.html)
- [HIP Porting Guide](docs/markdown/hip_porting_guide.md)
- [HIP Terminology](docs/markdown/hip_terms.md) (including Rosetta Stone of GPU computing terms across CUDA/HIP/HC/AMP/OpenL)


## How do I get set up?

### Prerequisites - Choose Your Platform
HIP code can be developed either on AMD HSA or Boltzmann platform using HCC compiler, or a CUDA platform with NVCC installed:

#### AMD (HCC):

* Install HCC including supporting HSA kernel and runtime driver stack (https://bitbucket.org/multicoreware/hcc/wiki/Home).  
* By default HIP looks for HCC in /opt/hcc (can be overridden by setting HCC_HOME environment variable)
* By default HIP looks for HSA in /opt/hsa (can be overridden by setting HSA_PATH environment variable) 
   
#### NVIDIA (NVCC)
* Install CUDA SDK from manufacturer website.
* By default HIP looks for CUDA SDK in /usr/local/cuda (can be overriden by setting CUDA_PATH env variable)

### Add HIP/bin to your path.
For example, if this repot is clones to ~/HIP, and you are running bash:
```
> export PATH=$PATH:~/HIP/bin
```
Verify we can find hipconfig (one of the hip tools in bin dir):
```
>  hipconfig -pn
/home/me/HIP
```

## Examples and Getting Started:

* A sample that uses hipify to convert from CUDA to HIP:
 
```shell
> cd samples/01_Intro/square
# follow README / blog steps to hipify the application.
```

* A sample written directly in HIP, showing platform specialization:
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
    * **hip_runtime_api.h** : Defines HIP runtime APIs and can be compiled with many standard Linux compilers (HCC, GCC, ICC, CLANG, etc), in either C or C++ mode.
    * **hip_runtime.h** : Includes everything in hip_runtime_api.h PLUS hipLaunchKernel and syntax for writing device kernels and device functions.  hip_runtime.h can only be compiled with HCC.
    * **hcc_detail/**** , **nvcc_detail/**** : Implementation details for specific platforms.  HIP applications should not include these files directly.
    
* **bin**: Tools and scripts to help with hip porting
    * **hipify** : Tool to convert CUDA code to portable CPP.  Converts CUDA APIs and kernel builtins.  
    * **hipcc** : Compiler driver that can be used to replace NVCC in existing CUDA code.  hipcc ill call NVCC or HCC depending on platform, and include appropriate platform-specific headers and libraries.
    * **hipconfig** : Print HIP configuration (HIP_PATH, HIP_PLATFORM, CXX config flags, etc)
    * **hipexamine.sh** : Script to scan directory, find all code, and report statistics on how much can be ported with HIP (and identify likely features not yet supported)

* **doc**: Documentation - markdown and doxygen info
