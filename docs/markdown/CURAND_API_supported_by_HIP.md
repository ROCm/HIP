# CURAND API supported by HIP

## **1. CURAND Data types**

| **type**     |   **CUDA**                                                    |   **HIP**                                                  | **HIP value** (if differs) |
|-------------:|---------------------------------------------------------------|------------------------------------------------------------|----------------------------|
| enum         |***`curandStatus`***                                           |***`hiprandStatus`***                                       |
| enum         |***`curandStatus_t`***                                         |***`hiprandStatus_t`***                                     |
|            0 |*`CURAND_STATUS_SUCCESS`*                                      |*`HIPRAND_STATUS_SUCCESS`*                                  |
|          100 |*`CURAND_STATUS_VERSION_MISMATCH`*                             |*`HIPRAND_STATUS_VERSION_MISMATCH`*                         |
|          101 |*`CURAND_STATUS_NOT_INITIALIZED`*                              |*`HIPRAND_STATUS_NOT_INITIALIZED`*                          |
|          102 |*`CURAND_STATUS_ALLOCATION_FAILED`*                            |*`HIPRAND_STATUS_ALLOCATION_FAILED`*                        |
|          103 |*`CURAND_STATUS_TYPE_ERROR`*                                   |*`HIPRAND_STATUS_TYPE_ERROR`*                               |
|          104 |*`CURAND_STATUS_OUT_OF_RANGE`*                                 |*`HIPRAND_STATUS_OUT_OF_RANGE`*                             |
|          105 |*`CURAND_STATUS_LENGTH_NOT_MULTIPLE`*                          |*`HIPRAND_STATUS_LENGTH_NOT_MULTIPLE`*                      |
|          106 |*`CURAND_STATUS_DOUBLE_PRECISION_REQUIRED`*                    |*`HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED`*                |
|          201 |*`CURAND_STATUS_LAUNCH_FAILURE`*                               |*`HIPRAND_STATUS_LAUNCH_FAILURE`*                           |
|          202 |*`CURAND_STATUS_PREEXISTING_FAILURE`*                          |*`HIPRAND_STATUS_PREEXISTING_FAILURE`*                      |
|          203 |*`CURAND_STATUS_INITIALIZATION_FAILED`*                        |*`HIPRAND_STATUS_INITIALIZATION_FAILED`*                    |
|          204 |*`CURAND_STATUS_ARCH_MISMATCH`*                                |*`HIPRAND_STATUS_ARCH_MISMATCH`*                            |
|          999 |*`CURAND_STATUS_INTERNAL_ERROR`*                               |*`HIPRAND_STATUS_INTERNAL_ERROR`*                           |
| enum         |***`curandRngType`***                                          |***`hiprandRngType`***                                      |
| enum         |***`curandRngType_t`***                                        |***`hiprandRngType_t`***                                    |
|            0 |*`CURAND_RNG_TEST`*                                            |*`HIPRAND_RNG_TEST`*                                        |
|          100 |*`CURAND_RNG_PSEUDO_DEFAULT`*                                  |*`HIPRAND_RNG_PSEUDO_DEFAULT`*                              | 400                        |
|          101 |*`CURAND_RNG_PSEUDO_XORWOW`*                                   |*`HIPRAND_RNG_PSEUDO_XORWOW`*                               | 401                        |
|          121 |*`CURAND_RNG_PSEUDO_MRG32K3A`*                                 |*`HIPRAND_RNG_PSEUDO_MRG32K3A`*                             | 402                        |
|          141 |*`CURAND_RNG_PSEUDO_MTGP32`*                                   |*`HIPRAND_RNG_PSEUDO_MTGP32`*                               | 403                        |
|          142 |*`CURAND_RNG_PSEUDO_MT19937`*                                  |*`HIPRAND_RNG_PSEUDO_MT19937`*                              | 404                        |
|          161 |*`CURAND_RNG_PSEUDO_PHILOX4_32_10`*                            |*`HIPRAND_RNG_PSEUDO_PHILOX4_32_10`*                        | 405                        |
|          200 |*`CURAND_RNG_QUASI_DEFAULT`*                                   |*`HIPRAND_RNG_QUASI_DEFAULT`*                               | 500                        |
|          201 |*`CURAND_RNG_QUASI_SOBOL32`*                                   |*`HIPRAND_RNG_QUASI_SOBOL32`*                               | 501                        |
|          202 |*`CURAND_RNG_QUASI_SCRAMBLED_SOBOL32`*                         |*`HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32`*                     | 502                        |
|          203 |*`CURAND_RNG_QUASI_SOBOL64`*                                   |*`HIPRAND_RNG_QUASI_SOBOL64`*                               | 503                        |
|          204 |*`CURAND_RNG_QUASI_SCRAMBLED_SOBOL64`*                         |*`HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64`*                     | 504                        |
| enum         |***`curandOrdering`***                                         |                                                            |
| enum         |***`curandOrdering_t`***                                       |                                                            |
|          100 |*`CURAND_ORDERING_PSEUDO_BEST`*                                |                                                            |
|          101 |*`CURAND_ORDERING_PSEUDO_DEFAULT`*                             |                                                            |
|          102 |*`CURAND_ORDERING_PSEUDO_SEEDED`*                              |                                                            |
|          201 |*`CURAND_ORDERING_QUASI_DEFAULT`*                              |                                                            |
| enum         |***`curandDirectionVectorSet`***                               |                                                            |
| enum         |***`curandDirectionVectorSet_t`***                             |                                                            |
|          101 |*`CURAND_DIRECTION_VECTORS_32_JOEKUO6`*                        |                                                            |
|          102 |*`CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6`*              |                                                            |
|          103 |*`CURAND_DIRECTION_VECTORS_64_JOEKUO6`*                        |                                                            |
|          104 |*`CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6`*              |                                                            |
| uint         | `curandDirectionVectors32_t`                                  | `hiprandDirectionVectors32_t`                              |
| uint         | `curandDirectionVectors64_t`                                  |                                                            |
| struct       | `curandGenerator_st`                                          | `hiprandGenerator_st`                                      |
| struct*      | `curandGenerator_t`                                           | `hiprandGenerator_t`                                       |
| enum         |***`curandOrdering`***                                         |                                                            |
| enum         |***`curandOrdering_t`***                                       |                                                            |
|          100 |*`CURAND_ORDERING_PSEUDO_BEST`*                                |                                                            |
|          101 |*`CURAND_ORDERING_PSEUDO_DEFAULT`*                             |                                                            |
|          102 |*`CURAND_ORDERING_PSEUDO_SEEDED`*                              |                                                            |
|          201 |*`CURAND_ORDERING_QUASI_DEFAULT`*                              |                                                            |
| enum         |***`curandDirectionVectorSet`***                               |                                                            |
| enum         |***`curandDirectionVectorSet_t`***                             |                                                            |
|          101 |*`CURAND_DIRECTION_VECTORS_32_JOEKUO6`*                        |                                                            |
|          102 |*`CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6`*              |                                                            |
|          103 |*`CURAND_DIRECTION_VECTORS_64_JOEKUO6`*                        |                                                            |
|          104 |*`CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6`*              |                                                            |
| uint         | `curandDirectionVectors32_t`                                  | `hiprandDirectionVectors32_t`                              |
| uint         | `curandDirectionVectors64_t`                                  |                                                            |
| double       | `curandDistribution_st`                                       |                                                            |
| double       | `curandHistogramM2V_st`                                       |                                                            |
| double*      | `curandDistribution_t`                                        |                                                            |
| double*      | `curandHistogramM2V_t`                                        |                                                            |
| struct       | `curandDistributionShift_st`                                  |                                                            |
| struct*      | `curandDistributionShift_t`                                   |                                                            |
| struct       | `curandDistributionM2Shift_st`                                |                                                            |
| struct*      | `curandDistributionM2Shift_t`                                 |                                                            |
| struct       | `curandHistogramM2_st`                                        |                                                            |
| struct*      | `curandHistogramM2_t`                                         |                                                            |
| uint         | `curandHistogramM2K_st`                                       |                                                            |
| uint*        | `curandHistogramM2K_t`                                        |                                                            |
| struct       | `curandDiscreteDistribution_st`                               | `hiprandDiscreteDistribution_st`                           |
| struct*      | `curandDiscreteDistribution_t`                                | `hiprandDiscreteDistribution_t`                            |
| enum         |***`curandMethod`***                                           |                                                            |
| enum         |***`curandMethod_t`***                                         |                                                            |
|            0 |*`CURAND_CHOOSE_BEST`*                                         |                                                            |
|            1 |*`CURAND_ITR`*                                                 |                                                            |
|            2 |*`CURAND_KNUTH`*                                               |                                                            |
|            3 |*`CURAND_HITR`*                                                |                                                            |
|            4 |*`CURAND_M1`*                                                  |                                                            |
|            5 |*`CURAND_M2`*                                                  |                                                            |
|            6 |*`CURAND_BINARY_SEARCH`*                                       |                                                            |
|            7 |*`CURAND_DISCRETE_GAUSS`*                                      |                                                            |
|            8 |*`CURAND_REJECTION`*                                           |                                                            |
|            9 |*`CURAND_DEVICE_API`*                                          |                                                            |
|           10 |*`CURAND_FAST_REJECTION`*                                      |                                                            |
|           11 |*`CURAND_3RD`*                                                 |                                                            |
|           12 |*`CURAND_DEFINITION`*                                          |                                                            |
|           13 |*`CURAND_POISSON`*                                             |                                                            |
| struct       | `curandStateMtgp32_t`                                         | `hiprandStateMtgp32_t`                                     |
| struct       | `curandStateScrambledSobol64_t`                               |                                                            |
| struct       | `curandStateSobol64_t`                                        |                                                            |
| struct       | `curandStateScrambledSobol32_t`                               |                                                            |
| struct       | `curandStateSobol32_t`                                        | `hiprandStateSobol32_t`                                    |
| struct       | `curandStateMRG32k3a_t`                                       | `hiprandStateMRG32k3a_t`                                   |
| struct       | `curandStatePhilox4_32_10_t`                                  | `hiprandStatePhilox4_32_10_t`                              |
| struct       | `curandStateXORWOW_t`                                         | `hiprandStateXORWOW_t`                                     |
| struct       | `curandState_t`                                               | `hiprandState_t`                                           |
| struct       | `curandState`                                                 | `hiprandState_t`                                           |

## **2. Host API Functions**

|   **CUDA**                                                |   **HIP**                                  |
|-----------------------------------------------------------|--------------------------------------------|
| `curandCreateGenerator`                                   | `hiprandCreateGenerator`                   |
| `curandCreateGeneratorHost`                               | `hiprandCreateGeneratorHost`               |
| `curandCreatePoissonDistribution`                         | `hiprandCreatePoissonDistribution`         |
| `curandDestroyDistribution`                               | `hiprandDestroyDistribution`               |
| `curandDestroyGenerator`                                  | `hiprandDestroyGenerator`                  |
| `curandGenerate`                                          | `hiprandGenerate`                          |
| `curandGenerateLogNormal`                                 | `hiprandGenerateLogNormal`                 |
| `curandGenerateLogNormalDouble`                           | `hiprandGenerateLogNormalDouble`           |
| `curandGenerateLongLong`                                  |                                            |
| `curandGenerateNormal`                                    | `hiprandGenerateNormal`                    |
| `curandGenerateNormalDouble`                              | `hiprandGenerateNormalDouble`              |
| `curandGeneratePoisson`                                   | `hiprandGeneratePoisson`                   |
| `curandGenerateSeeds`                                     | `hiprandGenerateSeeds`                     |
| `curandGenerateUniform`                                   | `hiprandGenerateUniform`                   |
| `curandGenerateUniformDouble`                             | `hiprandGenerateUniformDouble`             |
| `curandGetDirectionVectors32`                             |                                            |
| `curandGetDirectionVectors64`                             |                                            |
| `curandGetProperty`                                       |                                            |
| `curandGetScrambleConstants32`                            |                                            |
| `curandGetScrambleConstants64`                            |                                            |
| `curandGetVersion`                                        | `hiprandGetVersion`                        |
| `curandSetGeneratorOffset`                                | `hiprandSetGeneratorOffset`                |
| `curandSetGeneratorOrdering`                              |                                            |
| `curandSetPseudoRandomGeneratorSeed`                      | `hiprandSetPseudoRandomGeneratorSeed`      |
| `curandSetQuasiRandomGeneratorDimensions`                 | `hiprandSetQuasiRandomGeneratorDimensions` |
| `curandSetStream`                                         | `hiprandSetStream`                         |

## **3. Device API Functions**

|   **CUDA**                                                |   **HIP**                                  |
|-----------------------------------------------------------|--------------------------------------------|
| `curand`                                                  | `hiprand`                                  |
| `curand_init`                                             | `hiprand_init`                             |
| `curand_log_normal`                                       | `hiprand_log_normal`                       |
| `curand_log_normal_double`                                | `hiprand_log_normal_double`                |
| `curand_log_normal2`                                      | `hiprand_log_normal2`                      |
| `curand_log_normal2_double`                               | `hiprand_log_normal2_double`               |
| `curand_log_normal4`                                      | `hiprand_log_normal4`                      |
| `curand_log_normal4_double`                               | `hiprand_log_normal4_double`               |
| `curand_mtgp32_single`                                    |                                            |
| `curand_mtgp32_single_specific`                           |                                            |
| `curand_mtgp32_specific`                                  |                                            |
| `curand_normal`                                           | `hiprand_normal`                           |
| `curand_normal_double`                                    | `hiprand_normal_double`                    |
| `curand_normal2`                                          | `hiprand_normal2`                          |
| `curand_normal2_double`                                   | `hiprand_normal2_double`                   |
| `curand_normal4`                                          | `hiprand_normal4`                          |
| `curand_normal4_double`                                   | `hiprand_normal4_double`                   |
| `curand_uniform`                                          | `hiprand_uniform`                          |
| `curand_uniform_double`                                   | `hiprand_uniform_double`                   |
| `curand_uniform2_double`                                  | `hiprand_uniform2_double`                  |
| `curand_uniform4`                                         | `hiprand_uniform4`                         |
| `curand_uniform4_double`                                  | `hiprand_uniform4_double`                  |
| `curand_discrete`                                         | `hiprand_discrete`                         |
| `curand_discrete4`                                        | `hiprand_discrete4`                        |
| `curand_poisson`                                          | `hiprand_poisson`                          |
| `curand_poisson4`                                         | `hiprand_poisson4`                         |
| `curand_Philox4x32_10`                                    |                                            |
| `skipahead`                                               | `skipahead`                                |
| `skipahead_sequence`                                      | `skipahead_sequence`                       |
| `skipahead_subsequence`                                   | `skipahead_subsequence`                    |
