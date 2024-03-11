# Developing a HIP Back-end

This section provides guidelines and recommendations for development of a new HIP backend. Many of these recommendations are based on development and usage of hipamd (AMD specific backend) implementation.


## Functional Requirement

Each HIP backend is required to implement host APIs(interfaces) listed in [hip_runtime_api.h](https://github.com/ROCm-Developer-Tools/HIP/blob/develop/include/hip/hip_runtime_api.h) in HIP ( HIP-Common) project. Complete list of supported HIP APIs is available here [https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP_API_Guide.html](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP_API_Guide.html). HIP supports a hybrid model that provides equivalent functionality to CUDA APIs. "Hybrid", means that there is no clear separation of runtime and driver APIs. Only for certain cases where hybrid model is not able to support equivalent runtime and driver APIs, separate APIs are included. Backend can return hipErrorNotSupported for HIP APIs not yet implemented.HIP also includes extension APIs.These APIs provide functionalities, which are not yet supported with existing CUDA equivalent APIs. Currently these APIs are AMD backend specific. These APIs are listed in [hip_ext.h](https://github.com/ROCm-Developer-Tools/HIP/blob/develop/include/hip/hip_ext.h). Specific backend can choose not to implement those APIs (return hipErrorNotSupported). Every extension API must be accompanied by a device property attribute.

Device functions are categorized in multiple top-level headers in common HIP project. Each top-level device functions header includes the respective back-end's implementation.

## Testing Requirement

Each backend is expected to build and run HIP dtests (directed tests). Currently there are two sets of tests being used in HIP.

### HIT Tests in HIP-Common

These tests are available in HIP-Common at [https://github.com/ROCm-Developer-Tools/HIP/tree/develop/tests/src](https://github.com/ROCm-Developer-Tools/HIP/tree/develop/tests/src)
-   Each test's source file contains a HIT_START and HIT_END block as below-
```sh
/* HIT_START
* BUILD_CMD: <build step> EXCLUDE_HIP_PLATFORM <HIP platform to exclude>
* TEST: <test step> EXCLUDE_HIP_PLATFORM <HIP platform to exclude>
* HIT_END
*/
```
-   Support and usage of HIT based tests is expected to be discontinued by ROCm 5.0

### Catch2 Tests in HIP-Common

These tests are available in HIP-Common project at [https://github.com/ROCm-Developer-Tools/HIP/tree/develop/tests/catch](https://github.com/ROCm-Developer-Tools/HIP/tree/develop/tests/catch)
-   Important features of Catch2 tests are
-   Skip tests without recompiling tests, by addition of a JSON file
-   Better CI integration via xunit compatible output

## Installation Requirement

Each backend can decide its own packaging and installation ways. Currently for hipamd and hipnvidia backends, AMD supports  the following packages,
-   hip-runtime-amd 
	- This package includes all required libraries and cmake files to executes already built HIP applications with hipamd backend
	-	Includes
		- libhipamd64
		- libhiprtc-builtins
		- hip-config.cmake
		- hip-targets.cmake
- hip-runtime-nvidia
	- As the NVIDIA backend is supported through a header only implementation, following files are being packaged
		- hip-config.cmake
		-   hip-targets.cmake
-   hip-devel
	-   This package includes all required headers,  binaries,
	- Headers so for hipamd and hipnvidia backends are included
-   hip-doc
	-  This package includes HIP documentation
-   hip-samples
	-   Package to install HIP-samples

## Implementation details and Recommendations

### HIP (HIP-Common) Overview

HIP-Common contains:
-   HIP interfaces (hip_runtime.h, hip_runtime_api.h, and descendants)
-   Binaries (hipcc, hipconfig)
-   HIP Unit Tests
-   HIT based tests (Expected to be discontinued by ROCm 5.0)
-   Catch2 Tests

### hipamd Overview
hipamd contains:
-   AMD specific backend implementation of HIP
-   AMD specific header include files (amd_detail) and binaries
Directory Structure
-   Platform specific common include files with platform specific (eg  amd) prefix
-   Platform specific includes are under amd_detail

## Building the Back-end

The HIP (HIP-common) project must be used to access shared common files and unit tests only.
HIP (HIP-common) project cannot be built independently. While building the HIP backend, HIP (HIP-common) directory should be passed as a CMake variable.

### Build Steps
-   Clone HIP (HIP-common) project
-   Clone back-end project
-   Create build folder inside back-end project
-   Run Cmake command by passing HIP (HIP-common) source directory as a CMake variable. Currently Nvidia backend is part of hipamd project. Please check [https://github.com/ROCm-Developer-Tools/hipamd/blob/develop/CMakeLists.txt](https://github.com/ROCm-Developer-Tools/hipamd/blob/develop/CMakeLists.txt).  Instructions to build HIP for AMD backend is available at
[https://github.com/ROCm-Developer-Tools/HIP/blob/develop/INSTALL.md#build-hip](https://github.com/ROCm-Developer-Tools/HIP/blob/develop/INSTALL.md#build-hip)

### Build Example
-   hipamd project's CMakeList uses HIP_COMMON_DIR cmake variable
-   https://github.com/ROCm-Developer-Tools/hipamd/blob/develop/CMakeLists.txt#L37
    
### Building and Testing dtests

-   HIP (HIP-common) project contains the HIP directed tests (dtests). HIP dtests provide basic sanity and conformance. HIP dtests are constantly being evolved to enhance test coverage.
    
-   Currently, HIP dtests are using HIT test framework, which is being replaced by Catch2 test framework. Both HIT and Catch2 are frameworks for implementing test cases. HIP dtests are always executed via ctest.
    
-   HIP dtests can be built after building the HIP backend OR against an already installed HIP backend. In the hipamd project, HIP dtests are being built after building the hipamd backend.
    
-   Although the HIP dtests sources are part of HIP (HIP-common) project, building and testing of HIP dtests is usually done as part of hipamd build process. However, Catch2 tests can be independently built against a pre-installed HIP backend by defining HIP_PATH environment variable.

## Installation

After building the backend, HIP should be installed in a custom or a default location. As AMD specific backend implementation is part of AMD's ROCm offering, by default, built HIP gets installed in /opt/rocm-XXXX/hip (Defined by the environment variable ROCM_PATH).

The installed folder should contain all common files as well as platform specific dependencies.



