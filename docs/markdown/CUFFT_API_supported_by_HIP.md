# CUFFT API supported by HIP

## **1. CUFFT Data types**

| **type**     |   **CUDA**                                                    |**CUDA version\***|   **HIP**                                                  |**HIP value** (if differs) |
|-------------:|---------------------------------------------------------------|:----------------:|------------------------------------------------------------|---------------------------|
| enum         |***`cufftResult_t`***                                          |                  |***`hipfftResult_t`***                                      |
| enum         |***`cufftResult`***                                            |                  |***`hipfftResult`***                                        |
|          0x0 |*`CUFFT_SUCCESS`*                                              |                  |*`HIPFFT_SUCCESS`*                                          | 0                         |
|          0x1 |*`CUFFT_INVALID_PLAN`*                                         |                  |*`HIPFFT_INVALID_PLAN`*                                     | 1                         |
|          0x2 |*`CUFFT_ALLOC_FAILED`*                                         |                  |*`HIPFFT_ALLOC_FAILED`*                                     | 2                         |
|          0x3 |*`CUFFT_INVALID_TYPE`*                                         |                  |*`HIPFFT_INVALID_TYPE`*                                     | 3                         |
|          0x4 |*`CUFFT_INVALID_VALUE`*                                        |                  |*`HIPFFT_INVALID_VALUE`*                                    | 4                         |
|          0x5 |*`CUFFT_INTERNAL_ERROR`*                                       |                  |*`HIPFFT_INTERNAL_ERROR`*                                   | 5                         |
|          0x6 |*`CUFFT_EXEC_FAILED`*                                          |                  |*`HIPFFT_EXEC_FAILED`*                                      | 6                         |
|          0x7 |*`CUFFT_SETUP_FAILED`*                                         |                  |*`HIPFFT_SETUP_FAILED`*                                     | 7                         |
|          0x8 |*`CUFFT_INVALID_SIZE`*                                         |                  |*`HIPFFT_INVALID_SIZE`*                                     | 8                         |
|          0x9 |*`CUFFT_UNALIGNED_DATA`*                                       |                  |*`HIPFFT_UNALIGNED_DATA`*                                   | 9                         |
|          0xA |*`CUFFT_INCOMPLETE_PARAMETER_LIST`*                            |                  |*`HIPFFT_INCOMPLETE_PARAMETER_LIST`*                        | 10                        |
|          0xB |*`CUFFT_INVALID_DEVICE`*                                       |                  |*`HIPFFT_INVALID_DEVICE`*                                   | 11                        |
|          0xC |*`CUFFT_PARSE_ERROR`*                                          |                  |*`HIPFFT_PARSE_ERROR`*                                      | 12                        |
|          0xD |*`CUFFT_NO_WORKSPACE`*                                         |                  |*`HIPFFT_NO_WORKSPACE`*                                     | 13                        |
|          0xE |*`CUFFT_NOT_IMPLEMENTED`*                                      |                  |*`HIPFFT_NOT_IMPLEMENTED`*                                  | 14                        |
|          0xF |*`CUFFT_LICENSE_ERROR`*                                        |                  |                                                            |
|         0x10 |*`CUFFT_NOT_SUPPORTED`*                                        | 8.0              |*`HIPFFT_NOT_SUPPORTED`*                                    | 16                        |
| float        |***`cufftReal`***                                              |                  |***`hipfftReal`***                                          |
| double       |***`cufftDoubleReal`***                                        |                  |***`hipfftDoubleReal`***                                    |
| float2       |***`cufftComplex`***                                           |                  |***`hipfftComplex`***                                       |
| double2      |***`cufftDoubleComplex`***                                     |                  |***`hipfftDoubleComplex`***                                 |
| define       |`CUFFT_FORWARD`                                                |                  |`HIPFFT_FORWARD`                                            |
| define       |`CUFFT_INVERSE`                                                |                  |`HIPFFT_BACKWARD`                                           |
| enum         |***`cufftType_t`***                                            |                  |***`hipfftType_t`***                                        |
| enum         |***`cufftType`***                                              |                  |***`hipfftType`***                                          |
|         0x2a |*`CUFFT_R2C`*                                                  |                  |*`HIPFFT_R2C`*                                              |
|         0x2c |*`CUFFT_C2R`*                                                  |                  |*`HIPFFT_C2R`*                                              |
|         0x29 |*`CUFFT_C2C`*                                                  |                  |*`HIPFFT_C2C`*                                              |
|         0x6a |*`CUFFT_D2Z`*                                                  |                  |*`HIPFFT_D2Z`*                                              |
|         0x6c |*`CUFFT_Z2D`*                                                  |                  |*`HIPFFT_Z2D`*                                              |
|         0x69 |*`CUFFT_Z2Z`*                                                  |                  |*`HIPFFT_Z2Z`*                                              |
| enum         |***`cufftCompatibility_t`***                                   |                  |                                                            |
| enum         |***`cufftCompatibility`***                                     |                  |                                                            |
|         0x01 |*`CUFFT_COMPATIBILITY_FFTW_PADDING`*                           |                  |                                                            |
| define       |`CUFFT_COMPATIBILITY_DEFAULT`                                  |                  |                                                            |
| int          |***`cufftHandle`***                                            |                  |***`hipfftHandle`***                                        |

## **2. CUFFT API functions**

|   **CUDA**                                                |   **HIP**                                       |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------|:----------------:|
|`cufftPlan1d`                                              |`hipfftPlan1d`                                   |
|`cufftPlan2d`                                              |`hipfftPlan2d`                                   |
|`cufftPlan3d`                                              |`hipfftPlan3d`                                   |
|`cufftPlanMany`                                            |`hipfftPlanMany`                                 |
|`cufftMakePlan1d`                                          |`hipfftMakePlan1d`                               |
|`cufftMakePlan2d`                                          |`hipfftMakePlan2d`                               |
|`cufftMakePlan3d`                                          |`hipfftMakePlan3d`                               |
|`cufftMakePlanMany`                                        |`hipfftMakePlanMany`                             |
|`cufftMakePlanMany64`                                      |`hipfftMakePlanMany64`                           | 7.5              |
|`cufftGetSizeMany64`                                       |`hipfftGetSizeMany64`                            | 7.5              |
|`cufftEstimate1d`                                          |`hipfftEstimate1d`                               |
|`cufftEstimate2d`                                          |`hipfftEstimate2d`                               |
|`cufftEstimate3d`                                          |`hipfftEstimate3d`                               |
|`cufftEstimateMany`                                        |`hipfftEstimateMany`                             |
|`cufftCreate`                                              |`hipfftCreate`                                   |
|`cufftGetSize1d`                                           |`hipfftGetSize1d`                                |
|`cufftGetSize2d`                                           |`hipfftGetSize2d`                                |
|`cufftGetSize3d`                                           |`hipfftGetSize3d`                                |
|`cufftGetSizeMany`                                         |`hipfftGetSizeMany`                              |
|`cufftGetSize`                                             |`hipfftGetSize`                                  |
|`cufftSetWorkArea`                                         |`hipfftSetWorkArea`                              |
|`cufftSetAutoAllocation`                                   |`hipfftSetAutoAllocation`                        |
|`cufftExecC2C`                                             |`hipfftExecC2C`                                  |
|`cufftExecR2C`                                             |`hipfftExecR2C`                                  |
|`cufftExecC2R`                                             |`hipfftExecC2R`                                  |
|`cufftExecZ2Z`                                             |`hipfftExecZ2Z`                                  |
|`cufftExecD2Z`                                             |`hipfftExecD2Z`                                  |
|`cufftExecZ2D`                                             |`hipfftExecZ2D`                                  |
|`cufftSetStream`                                           |`hipfftSetStream`                                |
|`cufftDestroy`                                             |`hipfftDestroy`                                  |
|`cufftGetVersion`                                          |`hipfftGetVersion`                               |
|`cufftGetProperty`                                         |                                                 | 8.0              |
