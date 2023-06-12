# Building HIP from Source

## Prerequisites

HIP code can be developed either on AMD ROCm platform using HIP-Clang compiler, or a CUDA platform with nvcc installed.
Before build and run HIP, make sure drivers and pre-build packages are installed properly on the platform.

### AMD platform
Install ROCm packages (see ROCm Installation Guide on AMD public documentation site (https://docs.amd.com/)) or install pre-built binary packages using the package manager,

```shell
sudo apt install mesa-common-dev
sudo apt install clang
sudo apt install comgr
sudo apt-get -y install rocm-dkms
```

### NVIDIA platform

Install Nvidia driver and pre-build packages (see HIP Installation Guide at https://docs.amd.com/ for the release)

### Branch of repository

Before get HIP source code, set the expected branch of repository at the variable `HIP_BRANCH`.
For example, for ROCm5.0 release branch, set
```shell
export HIP_BRANCH=rocm-5.0.x
```

ROCm5.1 release branch, set
```shell
export HIP_BRANCH=rocm-5.1.x
```
Similiar format for future branches.

`ROCM_PATH` is path where ROCM is installed. BY default `ROCM_PATH` is at `/opt/rocm`.


## Build HIP on AMD platform


### Get HIP source code

```shell
git clone -b $HIP_BRANCH https://github.com/ROCm-Developer-Tools/hipamd.git
git clone -b $HIP_BRANCH https://github.com/ROCm-Developer-Tools/hip.git
git clone -b $HIP_BRANCH https://github.com/ROCm-Developer-Tools/ROCclr.git
```

### Set the environment variables

```shell
export HIPAMD_DIR="$(readlink -f hipamd)"
export HIP_DIR="$(readlink -f hip)"
```

ROCclr is defined on AMD platform that HIP use Radeon Open Compute Common Language Runtime (ROCclr), which is a virtual device interface that HIP runtimes interact with different backends.
See https://github.com/ROCm-Developer-Tools/ROCclr

HIPAMD repository provides implementation specifically for AMD platform.
See https://github.com/ROCm-Developer-Tools/hipamd

### Build HIP

```shell
cd "$HIPAMD_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DCMAKE_PREFIX_PATH="<ROCM_PATH>/" -DCMAKE_INSTALL_PREFIX=$PWD/install ..
make -j$(nproc)
sudo make install
```
::::{note}
If you don't specify `CMAKE_INSTALL_PREFIX`, hip runtime will be installed to `<ROCM_PATH>/hip`.
By default, release version of AMDHIP is built.

::::
#### Default paths and environment variables

   * By default HIP looks for HSA in `<ROCM_PATH>/hsa` (can be overridden by setting `HSA_PATH` environment variable).
   * By default HIP is installed into `<ROCM_PATH>/hip` (can be overridden by setting HIP_PATH environment variable).
   * By default HIP looks for clang in `<ROCM_PATH>/llvm/bin` (can be overridden by setting `HIP_CLANG_PATH` environment variable)
   * By default HIP looks for device library in `<ROCM_PATH>/lib` (can be overridden by setting `DEVICE_LIB_PATH` environment variable).
   * Optionally, consider adding `<ROCM_PATH>/bin` to your `PATH` to make it easier to use the tools.
   * Optionally, set `HIPCC_VERBOSE=7` to output the command line for compilation.

After make install command, make sure `HIP_PATH` is pointed to `$PWD/install/hip`.

#### Generating profiling header after adding/changing a HIP API

When you add or change a HIP API, you might need to generate a new `hip_prof_str.h` header. This header is used by rocm tools to track HIP APIs like rocprofiler/roctracer etc.
To generate the header after your change, use the tool `hip_prof_gen.py` present in `hipamd/src`.

Usage:

`hip_prof_gen.py [-v] <input HIP API .h file> <patched srcs path> <previous output> [<output>]`

```
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
sudo make install

### Build HIP tests

#### Build HIP directed tests
Developers can build HIP directed tests right after build HIP commands,

```shell
make -j$(nproc) build_tests
```
By default, all HIP directed tests will be built and generated under the folder `$HIPAMD_DIR/build/`directed_tests.
Take HIP directed device APIs tests, as an example, all available test applications will have executable files generated under,
`$HIPAMD_DIR/build/directed_tests/runtimeApi/device`.

Run all HIP directed_tests, use the command,

```
ctest
```shell
Or
```
make test
```shell

Build and run a single directed test, use the follow command as an example,

```shell
make directed_tests.texture.hipTexObjPitch
cd $HIPAMD_DIR/build/directed_tests/texcture
./hipTexObjPitch
```

#### Build HIP catch tests

After build and install HIP commands, catch tests can be built via the following instructions,

```shell
cd "$HIP_DIR"
mkdir -p build; cd build
export HIP_PATH=$HIPAMD_DIR/build
cmake  ../tests/catch/ -DHIP_PLATFORM=amd
make -j$(nproc) build_tests
ctest # run tests
```shell

HIP catch tests are built under the folder $HIP_DIR/build.

To run a single catch test, the following is an example,

```
cd $HIP_DIR/build/unit/texture
./TextureTest
```

### Build HIP Catch2 standalone test

HIP Catch2 supports build a standalone test, for example,

```shell
export PATH=$HIP_DIR/bin:$PATH
export HIP_PATH=$HIPAMD_DIR/build/install

hipcc $HIP_DIR/tests/catch/unit/memory/hipPointerGetAttributes.cc -I ./tests/catch/include ./tests/catch/hipTestMain/standalone_main.cc -I ./tests/catch/external/Catch2 -g -o hipPointerGetAttributes
./hipPointerGetAttributes
...

All tests passed
```shell

Please note, the integrated HIP directed tests, which are currently built by default, will be deprecated in future release. HIP catch tests, especially new architectured Catch2, will be official HIP tests in the repository and can be built alone as with the instructions shown above. Besides, Catch2 is more beneficial which brings the flexibility to build any standalone test/application.

## Build HIP on NVIDIA platform


### Get HIP source code

```
git clone -b $HIP_BRANCH https://github.com/ROCm-Developer-Tools/hip.git
git clone -b $HIP_BRANCH https://github.com/ROCm-Developer-Tools/hipamd.git
```

### Set the environment variables

```shell
export HIP_DIR="$(readlink -f hip)"
export HIPAMD_DIR="$(readlink -f hipamd)"
```

### Build HIP

```shell
cd "$HIPAMD_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=nvidia -DCMAKE_INSTALL_PREFIX=$PWD/install ..
make -j$(nproc)
sudo make install
```shell

### Build HIP tests
Build HIP tests commands on NVIDIA platform are basically the same as AMD, except set `-DHIP_PLATFORM=nvidia`.

## Run HIP

Compile and run the [square sample](https://github.com/ROCm-Developer-Tools/HIP/tree/rocm-5.0.x/samples/0_Intro/square).
