# Building HIP from Source

## Prerequisites

HIP code can be developed either on AMD ROCm platform using HIP-Clang compiler, or a CUDA platform with nvcc installed.
Before build and run HIP, make sure drivers and pre-build packages are installed properly on the platform.

### AMD platform
Install ROCm packages or pre-built binary packages using the package manager. Refer to the ROCm Installation Guide at https://rocm.docs.amd.com for more information on installing ROCm.

```shell
sudo apt install mesa-common-dev
sudo apt install clang
sudo apt install comgr
sudo apt-get -y install rocm-dkms
sudo apt-get install -y libelf-dev
```

### NVIDIA platform

Install Nvidia driver and pre-build packages (see HIP Installation Guide at https://docs.amd.com/ for the release)

### Branch of repository

Before get HIP source code, set the expected branch of repository at the variable `ROCM_BRANCH`.
For example, for ROCm6.0 release branch, set
```shell
export ROCM_BRANCH=rocm-6.0.x
```

ROCm5.7 release branch, set
```shell
export ROCM_BRANCH=rocm-5.7.x
```
Similiar format for future branches.

`ROCM_PATH` is path where ROCM is installed. BY default `ROCM_PATH` is at `/opt/rocm`.


## Build HIP on AMD platform


### Get HIP source code

```shell
git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/clr.git
git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/hip.git
git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/HIPCC.git
```

### Set the environment variables

```shell
export CLR_DIR="$(readlink -f clr)"
export HIP_DIR="$(readlink -f hip)"
export HIPCC_DIR="$(readlink -f hipcc)"
```

Note, starting from ROCM 5.6 release, clr is a new repository including the previous ROCclr, HIPAMD and OpenCl repositories.
ROCclr is defined on AMD platform that HIP uses Radeon Open Compute Common Language Runtime (ROCclr), which is a virtual device interface that HIP runtimes interact with different backends.
HIPAMD provides implementation specifically for AMD platform.
OpenCL provides headers that ROCclr runtime currently depends on.

### Build the HIPCC runtime

```shell
cd "$HIPCC_DIR"
mkdir -p build; cd build
cmake ..
make -j4
```

### Build HIP

```shell
cd "$CLR_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=amd -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install -DHIPCC_BIN_DIR=$HIPCC_DIR/build -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF ..

make -j$(nproc)
sudo make install
```

Note, if `CMAKE_INSTALL_PREFIX` is not specified, hip runtime will be installed to `<ROCM_PATH>/hip`.
By default, release version of HIP is built.


### Default paths and environment variables

   * By default HIP looks for HSA in `<ROCM_PATH>/hsa` (can be overridden by setting `HSA_PATH` environment variable).
   * By default HIP is installed into `<ROCM_PATH>/hip` (can be overridden by setting HIP_PATH environment variable).
   * By default HIP looks for clang in `<ROCM_PATH>/llvm/bin` (can be overridden by setting `HIP_CLANG_PATH` environment variable)
   * By default HIP looks for device library in `<ROCM_PATH>/lib` (can be overridden by setting `DEVICE_LIB_PATH` environment variable).
   * Optionally, consider adding `<ROCM_PATH>/bin` to your `PATH` to make it easier to use the tools.
   * Optionally, set `HIPCC_VERBOSE=7` to output the command line for compilation.

After make install command, make sure `HIP_PATH` is pointed to `$PWD/install/hip`.

### Generating profiling header after adding/changing a HIP API

When you add or change a HIP API, you might need to generate a new `hip_prof_str.h` header. This header is used by rocm tools to track HIP APIs like rocprofiler/roctracer etc.
To generate the header after your change, use the tool `hip_prof_gen.py` present in `hipamd/src`.

Usage:

`hip_prof_gen.py [-v] <input HIP API .h file> <patched srcs path> <previous output> [<output>]`

Flags:

  * -v - verbose messages
  * -r - process source directory recursively
  * -t - API types matching check
  * --priv - private API check
  * -e - on error exit mode
  * -p - HIP_INIT_API macro patching mode

Example Usage:
```shell
hip_prof_gen.py -v -p -t --priv <hip>/include/hip/hip_runtime_api.h \
<hipamd>/src <hipamd>/include/hip/amd_detail/hip_prof_str.h         \
<hipamd>/include/hip/amd_detail/hip_prof_str.h.new
```

### Build HIP tests

HIP catch tests, with new architectured Catch2, are official seperated from HIP project, exist in HIP tests repository, can be built via the following instructions.

#### Get HIP tests source code

```shell
git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/hip-tests.git
```
#### Build HIP tests from source

```shell
export HIPTESTS_DIR="$(readlink -f hip-tests)"
cd "$HIPTESTS_DIR"
mkdir -p build; cd build
export HIP_PATH=$CLR_DIR/build/install (or any path where HIP is installed, for example, /opt/rocm)
cmake ../catch/ -DHIP_PLATFORM=amd
make -j$(nproc) build_tests
ctest # run tests
```
HIP catch tests are built under the folder $HIPTESTS_DIR/build.

To run any single catch test, the following is an example,

```shell
cd $HIPTESTS_DIR/build/catch_tests/unit/texture
./TextureTest
```

#### Build HIP Catch2 standalone test

HIP Catch2 supports build a standalone test, for example,

```shell
cd "$HIPTESTS_DIR"
hipcc $HIPTESTS_DIR/catch/unit/memory/hipPointerGetAttributes.cc -I ./catch/include ./catch/hipTestMain/standalone_main.cc -I ./catch/external/Catch2 -o hipPointerGetAttributes
./hipPointerGetAttributes
...

All tests passed
```

## Build HIP on NVIDIA platform


### Get HIP source code

```shell
git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/hip.git
git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/clr.git
git clone -b "$ROCM_BRANCH" https://github.com/ROCm-Developer-Tools/HIPCC.git
```

### Set the environment variables

```shell
export HIP_DIR="$(readlink -f hip)"
export CLR_DIR="$(readlink -f hipamd)"
export HIPCC_DIR="$(readlink -f hipcc)"
```

### Build the HIPCC runtime

```shell
cd "$HIPCC_DIR"
mkdir -p build; cd build
cmake ..
make -j4
```

### Build HIP

```shell
cd "$CLR_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=nvidia -DCMAKE_INSTALL_PREFIX=$PWD/install -DHIPCC_BIN_DIR=$HIPCC_DIR/build -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF ..
make -j$(nproc)
sudo make install
```

### Build HIP tests
Build HIP tests commands on NVIDIA platform are basically the same as AMD, except set `-DHIP_PLATFORM=nvidia`.

## Run HIP

Compile and run the [square sample](https://github.com/ROCm-Developer-Tools/hip-tests/tree/rocm-5.5.x/samples/0_Intro/square).

