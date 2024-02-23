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
For example, for ROCm 6.1 release branch, set
```shell
export ROCM_BRANCH=rocm-6.1.x
```

ROCm5.7 release branch, set
```shell
export ROCM_BRANCH=rocm-5.7.x
```
Similiar format for future branches.

`ROCM_PATH` is path where ROCM is installed. BY default `ROCM_PATH` is at `/opt/rocm`.


## Build HIP


### Get HIP source code

A new repository 'hipother' is added in the ROCm 6.1 release, which is branched out from HIP.
The 'hipother'provides files required to support the HIP back-end implementation on some non-AMD platforms, like NVIDIA.

```shell
git clone -b "$ROCM_BRANCH" https://github.com/ROCm/clr.git
git clone -b "$ROCM_BRANCH" https://github.com/ROCm/hip.git
git clone -b "$ROCM_BRANCH" https://github.com/ROCm/hipother.git
git clone -b "$ROCM_BRANCH" https://github.com/ROCm/HIPCC.git hipcc
```

### Set the environment variables

```shell
export CLR_DIR="$(readlink -f clr)"
export HIP_DIR="$(readlink -f hip)"
export HIP_OTHER="$(readlink -f hipother)"
export HIPCC_DIR="$(readlink -f hipcc)"
```

Note, starting from ROCM 5.6 release, clr is a new repository including the previous ROCclr, HIPAMD and OpenCl repositories.
ROCclr is defined on AMD platform that HIP uses Radeon Open Compute Common Language Runtime (ROCclr), which is a virtual device interface that HIP runtimes interact with different backends.
HIPAMD provides implementation specifically for the AMD platform.
OpenCL provides headers that ROCclr runtime currently depends on.
hipother provides headers and implementation specifically for the non-AMD platform, like NVIDIA.

### Build the HIPCC runtime

```shell
cd "$HIPCC_DIR"
mkdir -p build; cd build
cmake ..
make -j4
```

### Build HIP on the AMD platform

#### Build HIP runtime
Commands to build HIP runtime on the AMD platform are as following. The option 'HIP_PLATFORM=amd' should be defined.

```shell
cd "$CLR_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=amd -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install -DHIPCC_BIN_DIR=$HIPCC_DIR/build -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF ..
make -j$(nproc)
sudo make install
```

Note, if `CMAKE_INSTALL_PREFIX` is not specified, hip runtime will be installed to `<ROCM_PATH>/hip`.
By default, release version of HIP is built.


#### Default paths and environment variables

   * By default HIP looks for HSA in `<ROCM_PATH>/hsa` (can be overridden by setting `HSA_PATH` environment variable).
   * By default HIP is installed into `<ROCM_PATH>/hip`.
   * By default HIP looks for clang in `<ROCM_PATH>/llvm/bin` (can be overridden by setting `HIP_CLANG_PATH` environment variable)
   * By default HIP looks for device library in `<ROCM_PATH>/lib` (can be overridden by setting `DEVICE_LIB_PATH` environment variable).
   * Optionally, consider adding `<ROCM_PATH>/bin` to your `PATH` to make it easier to use the tools.
   * Optionally, set `HIPCC_VERBOSE=7` to output the command line for compilation.

#### Generate profiling header after adding/changing a HIP API

When you add or change a HIP API, you must generate a new `hip_prof_str.h` header. ROCm tools like ROCProfiler and ROCTracer use this header to track HIP APIs.
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

#### Build HIP tests

HIP catch tests, with the newly architectured Catch2, are officially separated from the HIP project. The HIP catch tests are moved to the HIP tests repository and can be built using the  instructions in the following sections.

##### Get HIP tests source code

```shell
git clone -b "$ROCM_BRANCH" https://github.com/ROCm/hip-tests.git
```
##### Build HIP tests from source

```shell
export HIPTESTS_DIR="$(readlink -f hip-tests)"
cd "$HIPTESTS_DIR"
mkdir -p build; cd build
cmake ../catch -DHIP_PLATFORM=amd -DHIP_PATH=$CLR_DIR/build/install # or any path where HIP is installed, for example, /opt/rocm.
make -j$(nproc) build_tests # build tests
cd build/catch_tests && ctest # to run all tests.
```
HIP catch tests are built under the folder $HIPTESTS_DIR/build.
Note that when using ctest, you can use different ctest options, for example, to run all tests with the keyword hipMemset,
```
ctest -R hipMemset
```
Use the below option will print test failures for failed tests,
```
ctest --output-on-failure
```
For more information, refer to https://cmake.org/cmake/help/v3.5/manual/ctest.1.html.

To run any single catch test, the following is an example,

```shell
cd $HIPTESTS_DIR/build/catch_tests/unit/texture
./TextureTest
```

##### Build HIP Catch2 standalone tests

HIP Catch2 supports building standalone tests, for example,

```shell
cd "$HIPTESTS_DIR"
hipcc $HIPTESTS_DIR/catch/unit/memory/hipPointerGetAttributes.cc -I ./catch/include ./catch/hipTestMain/standalone_main.cc -I ./catch/external/Catch2 -o hipPointerGetAttributes
./hipPointerGetAttributes
...

All tests passed
```

### Build HIP on the NVIDIA platform


#### Build HIP runtime
Commands to build HIP on the NVIDIA platform are as following. The options 'HIPNV_DIR=$HIP_OTHER/hipnv' and 'HIP_PLATFORM=nvidia' should be defined.

```shell
cd "$CLR_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install -DHIPCC_BIN_DIR=$HIPCC_DIR/build -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIPNV_DIR=$HIP_OTHER/hipnv -DHIP_PLATFORM=nvidia ..
make -j$(nproc)
sudo make install
```

#### Build HIP tests
Build HIP tests commands on NVIDIA platform are basically the same as AMD, except set `-DHIP_PLATFORM=nvidia`.

## Run HIP

Compile and run the [square sample](https://github.com/ROCm/hip-tests/tree/rocm-6.0.x/samples/0_Intro/square).

