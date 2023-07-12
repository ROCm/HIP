# Installing HIP

HIP can be installed either on AMD ROCm platform with HIP-Clang compiler, or a CUDA platform with nvcc installed.

Note: The version definition for the HIP runtime is different from CUDA. Users can use hipRuntimeGerVersion function, on the AMD platform it returns the HIP runtime version, while on the NVIDIA platform, it returns the CUDA runtime version. There is no mapping or correlation between the HIP version and CUDA version.

## Prerequisites
On AMD platform, see Prerequisite Actions in ROCm_Installation_Guide (https://docs.amd.com/) for the release.
On NVIDIA platform, check system requirements in NVIDIA CUDA Installation Guide at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/.


## AMD Platform

HIP is part of ROCM packages, it will be automatically installed following the ROCm Installation Guide on AMD public documentation site (https://docs.amd.com/) for the coresponding ROCm release.

By default, HIP is installed into /opt/rocm/hip.


## NVIDIA Platform


* Install Nvidia Driver
```
sudo apt-get install ubuntu-drivers-common && sudo ubuntu-drivers autoinstall
sudo reboot
```
Or download the latest cuda-toolkit at https://developer.nvidia.com/cuda-downloads
The driver will be installed automatically.

* Add the ROCm package server to your system as per the OS-specific guide in ROCm Installation Guide (https://docs.amd.com/) for the release.
* Install the "hip-runtime-nvidia" and "hip-dev" package. This will install CUDA SDK and the HIP porting layer.
```
apt-get install hip-runtime-nvidia hip-dev
```
* Default paths:
   * By default HIP looks for CUDA SDK in /usr/local/cuda.
   * By default HIP is installed into /opt/rocm/hip.
   * Optionally, consider adding /opt/rocm/bin to your path to make it easier to use the tools.


# Verify your installation

Run hipconfig (instructions below assume default installation path):
```
/opt/rocm/bin/hipconfig --full
```

# How to build HIP from source

Developers can build HIP from source on either AMD or NVIDIA platforms, see
detailed instructions at  [building HIP from source](../developer_guide/build.md).


