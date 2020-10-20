# CUFFT API supported by HIP

## **1. CUFFT Data types**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`CUFFT_ALLOC_FAILED`|  |  |  |`HIPFFT_ALLOC_FAILED`|
|`CUFFT_C2C`|  |  |  |`HIPFFT_C2C`|
|`CUFFT_C2R`|  |  |  |`HIPFFT_C2R`|
|`CUFFT_COMPATIBILITY_DEFAULT`|  |  |  ||
|`CUFFT_COMPATIBILITY_FFTW_PADDING`|  |  |  ||
|`CUFFT_D2Z`|  |  |  |`HIPFFT_D2Z`|
|`CUFFT_EXEC_FAILED`|  |  |  |`HIPFFT_EXEC_FAILED`|
|`CUFFT_FORWARD`|  |  |  |`HIPFFT_FORWARD`|
|`CUFFT_INCOMPLETE_PARAMETER_LIST`|  |  |  |`HIPFFT_INCOMPLETE_PARAMETER_LIST`|
|`CUFFT_INTERNAL_ERROR`|  |  |  |`HIPFFT_INTERNAL_ERROR`|
|`CUFFT_INVALID_DEVICE`|  |  |  |`HIPFFT_INVALID_DEVICE`|
|`CUFFT_INVALID_PLAN`|  |  |  |`HIPFFT_INVALID_PLAN`|
|`CUFFT_INVALID_SIZE`|  |  |  |`HIPFFT_INVALID_SIZE`|
|`CUFFT_INVALID_TYPE`|  |  |  |`HIPFFT_INVALID_TYPE`|
|`CUFFT_INVALID_VALUE`|  |  |  |`HIPFFT_INVALID_VALUE`|
|`CUFFT_INVERSE`|  |  |  |`HIPFFT_BACKWARD`|
|`CUFFT_LICENSE_ERROR`|  |  |  ||
|`CUFFT_NOT_IMPLEMENTED`|  |  |  |`HIPFFT_NOT_IMPLEMENTED`|
|`CUFFT_NOT_SUPPORTED`| 8.0 |  |  |`HIPFFT_NOT_SUPPORTED`|
|`CUFFT_NO_WORKSPACE`|  |  |  |`HIPFFT_NO_WORKSPACE`|
|`CUFFT_PARSE_ERROR`|  |  |  |`HIPFFT_PARSE_ERROR`|
|`CUFFT_R2C`|  |  |  |`HIPFFT_R2C`|
|`CUFFT_SETUP_FAILED`|  |  |  |`HIPFFT_SETUP_FAILED`|
|`CUFFT_SUCCESS`|  |  |  |`HIPFFT_SUCCESS`|
|`CUFFT_UNALIGNED_DATA`|  |  |  |`HIPFFT_UNALIGNED_DATA`|
|`CUFFT_VERSION`| 10.2 |  |  ||
|`CUFFT_VER_BUILD`| 10.2 |  |  ||
|`CUFFT_VER_MAJOR`| 10.2 |  |  ||
|`CUFFT_VER_MINOR`| 10.2 |  |  ||
|`CUFFT_VER_PATCH`| 10.2 |  |  ||
|`CUFFT_Z2D`|  |  |  |`HIPFFT_Z2D`|
|`CUFFT_Z2Z`|  |  |  |`HIPFFT_Z2Z`|
|`MAX_CUFFT_ERROR`|  |  |  ||
|`cufftCompatibility`|  |  |  ||
|`cufftCompatibility_t`|  |  |  ||
|`cufftComplex`|  |  |  |`hipfftComplex`|
|`cufftDoubleComplex`|  |  |  |`hipfftDoubleComplex`|
|`cufftDoubleReal`|  |  |  |`hipfftDoubleReal`|
|`cufftHandle`|  |  |  |`hipfftHandle`|
|`cufftReal`|  |  |  |`hipfftReal`|
|`cufftResult`|  |  |  |`hipfftResult`|
|`cufftResult_t`|  |  |  |`hipfftResult_t`|
|`cufftType`|  |  |  |`hipfftType`|
|`cufftType_t`|  |  |  |`hipfftType_t`|

## **2. CUFFT API functions**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cufftCreate`|  |  |  |`hipfftCreate`|
|`cufftDestroy`|  |  |  |`hipfftDestroy`|
|`cufftEstimate1d`|  |  |  |`hipfftEstimate1d`|
|`cufftEstimate2d`|  |  |  |`hipfftEstimate2d`|
|`cufftEstimate3d`|  |  |  |`hipfftEstimate3d`|
|`cufftEstimateMany`|  |  |  |`hipfftEstimateMany`|
|`cufftExecC2C`|  |  |  |`hipfftExecC2C`|
|`cufftExecC2R`|  |  |  |`hipfftExecC2R`|
|`cufftExecD2Z`|  |  |  |`hipfftExecD2Z`|
|`cufftExecR2C`|  |  |  |`hipfftExecR2C`|
|`cufftExecZ2D`|  |  |  |`hipfftExecZ2D`|
|`cufftExecZ2Z`|  |  |  |`hipfftExecZ2Z`|
|`cufftGetProperty`| 8.0 |  |  |`hipfftGetProperty`|
|`cufftGetSize`|  |  |  |`hipfftGetSize`|
|`cufftGetSize1d`|  |  |  |`hipfftGetSize1d`|
|`cufftGetSize2d`|  |  |  |`hipfftGetSize2d`|
|`cufftGetSize3d`|  |  |  |`hipfftGetSize3d`|
|`cufftGetSizeMany`|  |  |  |`hipfftGetSizeMany`|
|`cufftGetSizeMany64`| 7.5 |  |  |`hipfftGetSizeMany64`|
|`cufftGetVersion`|  |  |  |`hipfftGetVersion`|
|`cufftMakePlan1d`|  |  |  |`hipfftMakePlan1d`|
|`cufftMakePlan2d`|  |  |  |`hipfftMakePlan2d`|
|`cufftMakePlan3d`|  |  |  |`hipfftMakePlan3d`|
|`cufftMakePlanMany`|  |  |  |`hipfftMakePlanMany`|
|`cufftMakePlanMany64`| 7.5 |  |  |`hipfftMakePlanMany64`|
|`cufftPlan1d`|  |  |  |`hipfftPlan1d`|
|`cufftPlan2d`|  |  |  |`hipfftPlan2d`|
|`cufftPlan3d`|  |  |  |`hipfftPlan3d`|
|`cufftPlanMany`|  |  |  |`hipfftPlanMany`|
|`cufftSetAutoAllocation`|  |  |  |`hipfftSetAutoAllocation`|
|`cufftSetStream`|  |  |  |`hipfftSetStream`|
|`cufftSetWorkArea`|  |  |  |`hipfftSetWorkArea`|


\* A - Added, D - Deprecated, R - Removed