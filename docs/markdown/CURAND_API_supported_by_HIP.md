# CURAND API supported by HIP

## **1. CURAND Data types**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`CURAND_3RD`|  |  |  |  |  |  |  | 
|`CURAND_BINARY_SEARCH`|  |  |  |  |  |  |  | 
|`CURAND_CHOOSE_BEST`|  |  |  |  |  |  |  | 
|`CURAND_DEFINITION`|  |  |  |  |  |  |  | 
|`CURAND_DEVICE_API`|  |  |  |  |  |  |  | 
|`CURAND_DIRECTION_VECTORS_32_JOEKUO6`|  |  |  |  |  |  |  | 
|`CURAND_DIRECTION_VECTORS_64_JOEKUO6`|  |  |  |  |  |  |  | 
|`CURAND_DISCRETE_GAUSS`|  |  |  |  |  |  |  | 
|`CURAND_FAST_REJECTION`|  |  |  |  |  |  |  | 
|`CURAND_HITR`|  |  |  |  |  |  |  | 
|`CURAND_ITR`|  |  |  |  |  |  |  | 
|`CURAND_KNUTH`|  |  |  |  |  |  |  | 
|`CURAND_M1`|  |  |  |  |  |  |  | 
|`CURAND_M2`|  |  |  |  |  |  |  | 
|`CURAND_ORDERING_PSEUDO_BEST`|  |  |  |  |  |  |  | 
|`CURAND_ORDERING_PSEUDO_DEFAULT`|  |  |  |  |  |  |  | 
|`CURAND_ORDERING_PSEUDO_LEGACY`| 11.0 |  |  |  |  |  |  | 
|`CURAND_ORDERING_PSEUDO_SEEDED`|  |  |  |  |  |  |  | 
|`CURAND_ORDERING_QUASI_DEFAULT`|  |  |  |  |  |  |  | 
|`CURAND_POISSON`|  |  |  |  |  |  |  | 
|`CURAND_REJECTION`|  |  |  |  |  |  |  | 
|`CURAND_RNG_PSEUDO_DEFAULT`|  |  |  | `HIPRAND_RNG_PSEUDO_DEFAULT` | 1.5.0 |  |  | 
|`CURAND_RNG_PSEUDO_MRG32K3A`|  |  |  | `HIPRAND_RNG_PSEUDO_MRG32K3A` | 1.5.0 |  |  | 
|`CURAND_RNG_PSEUDO_MT19937`|  |  |  | `HIPRAND_RNG_PSEUDO_MT19937` | 1.5.0 |  |  | 
|`CURAND_RNG_PSEUDO_MTGP32`|  |  |  | `HIPRAND_RNG_PSEUDO_MTGP32` | 1.5.0 |  |  | 
|`CURAND_RNG_PSEUDO_PHILOX4_32_10`|  |  |  | `HIPRAND_RNG_PSEUDO_PHILOX4_32_10` | 1.5.0 |  |  | 
|`CURAND_RNG_PSEUDO_XORWOW`|  |  |  | `HIPRAND_RNG_PSEUDO_XORWOW` | 1.5.0 |  |  | 
|`CURAND_RNG_QUASI_DEFAULT`|  |  |  | `HIPRAND_RNG_QUASI_DEFAULT` | 1.5.0 |  |  | 
|`CURAND_RNG_QUASI_SCRAMBLED_SOBOL32`|  |  |  | `HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32` | 1.5.0 |  |  | 
|`CURAND_RNG_QUASI_SCRAMBLED_SOBOL64`|  |  |  | `HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64` | 1.5.0 |  |  | 
|`CURAND_RNG_QUASI_SOBOL32`|  |  |  | `HIPRAND_RNG_QUASI_SOBOL32` | 1.5.0 |  |  | 
|`CURAND_RNG_QUASI_SOBOL64`|  |  |  | `HIPRAND_RNG_QUASI_SOBOL64` | 1.5.0 |  |  | 
|`CURAND_RNG_TEST`|  |  |  | `HIPRAND_RNG_TEST` | 1.5.0 |  |  | 
|`CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6`|  |  |  |  |  |  |  | 
|`CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6`|  |  |  |  |  |  |  | 
|`CURAND_STATUS_ALLOCATION_FAILED`|  |  |  | `HIPRAND_STATUS_ALLOCATION_FAILED` | 1.5.0 |  |  | 
|`CURAND_STATUS_ARCH_MISMATCH`|  |  |  | `HIPRAND_STATUS_ARCH_MISMATCH` | 1.5.0 |  |  | 
|`CURAND_STATUS_DOUBLE_PRECISION_REQUIRED`|  |  |  | `HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED` | 1.5.0 |  |  | 
|`CURAND_STATUS_INITIALIZATION_FAILED`|  |  |  | `HIPRAND_STATUS_INITIALIZATION_FAILED` | 1.5.0 |  |  | 
|`CURAND_STATUS_INTERNAL_ERROR`|  |  |  | `HIPRAND_STATUS_INTERNAL_ERROR` | 1.5.0 |  |  | 
|`CURAND_STATUS_LAUNCH_FAILURE`|  |  |  | `HIPRAND_STATUS_LAUNCH_FAILURE` | 1.5.0 |  |  | 
|`CURAND_STATUS_LENGTH_NOT_MULTIPLE`|  |  |  | `HIPRAND_STATUS_LENGTH_NOT_MULTIPLE` | 1.5.0 |  |  | 
|`CURAND_STATUS_NOT_INITIALIZED`|  |  |  | `HIPRAND_STATUS_NOT_INITIALIZED` | 1.5.0 |  |  | 
|`CURAND_STATUS_OUT_OF_RANGE`|  |  |  | `HIPRAND_STATUS_OUT_OF_RANGE` | 1.5.0 |  |  | 
|`CURAND_STATUS_PREEXISTING_FAILURE`|  |  |  | `HIPRAND_STATUS_PREEXISTING_FAILURE` | 1.5.0 |  |  | 
|`CURAND_STATUS_SUCCESS`|  |  |  | `HIPRAND_STATUS_SUCCESS` | 1.5.0 |  |  | 
|`CURAND_STATUS_TYPE_ERROR`|  |  |  | `HIPRAND_STATUS_TYPE_ERROR` | 1.5.0 |  |  | 
|`CURAND_STATUS_VERSION_MISMATCH`|  |  |  | `HIPRAND_STATUS_VERSION_MISMATCH` | 1.5.0 |  |  | 
|`CURAND_VERSION`| 10.2 |  |  | `HIPRAND_VERSION` | 1.5.0 |  |  | 
|`CURAND_VER_BUILD`| 10.2 |  |  |  |  |  |  | 
|`CURAND_VER_MAJOR`| 10.2 |  |  |  |  |  |  | 
|`CURAND_VER_MINOR`| 10.2 |  |  |  |  |  |  | 
|`CURAND_VER_PATCH`| 10.2 |  |  |  |  |  |  | 
|`curandDirectionVectorSet`|  |  |  |  |  |  |  | 
|`curandDirectionVectorSet_t`|  |  |  |  |  |  |  | 
|`curandDirectionVectors32_t`|  |  |  | `hiprandDirectionVectors32_t` | 1.5.0 |  |  | 
|`curandDirectionVectors64_t`|  |  |  |  |  |  |  | 
|`curandDiscreteDistribution_st`|  |  |  | `hiprandDiscreteDistribution_st` | 1.5.0 |  |  | 
|`curandDiscreteDistribution_t`|  |  |  | `hiprandDiscreteDistribution_t` | 1.5.0 |  |  | 
|`curandDistributionM2Shift_st`|  |  |  |  |  |  |  | 
|`curandDistributionM2Shift_t`|  |  |  |  |  |  |  | 
|`curandDistributionShift_st`|  |  |  |  |  |  |  | 
|`curandDistributionShift_t`|  |  |  |  |  |  |  | 
|`curandDistribution_st`|  |  |  |  |  |  |  | 
|`curandDistribution_t`|  |  |  |  |  |  |  | 
|`curandGenerator_st`|  |  |  | `hiprandGenerator_st` | 1.5.0 |  |  | 
|`curandGenerator_t`|  |  |  | `hiprandGenerator_t` | 1.5.0 |  |  | 
|`curandHistogramM2K_st`|  |  |  |  |  |  |  | 
|`curandHistogramM2K_t`|  |  |  |  |  |  |  | 
|`curandHistogramM2V_st`|  |  |  |  |  |  |  | 
|`curandHistogramM2V_t`|  |  |  |  |  |  |  | 
|`curandHistogramM2_st`|  |  |  |  |  |  |  | 
|`curandHistogramM2_t`|  |  |  |  |  |  |  | 
|`curandMethod`|  |  |  |  |  |  |  | 
|`curandMethod_t`|  |  |  |  |  |  |  | 
|`curandOrdering`|  |  |  |  |  |  |  | 
|`curandOrdering_t`|  |  |  |  |  |  |  | 
|`curandRngType`|  |  |  | `hiprandRngType_t` | 1.5.0 |  |  | 
|`curandRngType_t`|  |  |  | `hiprandRngType_t` | 1.5.0 |  |  | 
|`curandState`|  |  |  | `hiprandState` | 1.8.0 |  |  | 
|`curandStateMRG32k3a`|  |  |  | `hiprandStateMRG32k3a` | 1.8.0 |  |  | 
|`curandStateMRG32k3a_t`|  |  |  | `hiprandStateMRG32k3a_t` | 1.5.0 |  |  | 
|`curandStateMtgp32`|  |  |  | `hiprandStateMtgp32` | 1.8.0 |  |  | 
|`curandStateMtgp32_t`|  |  |  | `hiprandStateMtgp32_t` | 1.5.0 |  |  | 
|`curandStatePhilox4_32_10`|  |  |  | `hiprandStatePhilox4_32_10` | 1.8.0 |  |  | 
|`curandStatePhilox4_32_10_t`|  |  |  | `hiprandStatePhilox4_32_10_t` | 1.8.0 |  |  | 
|`curandStateScrambledSobol32`|  |  |  |  |  |  |  | 
|`curandStateScrambledSobol32_t`|  |  |  |  |  |  |  | 
|`curandStateScrambledSobol64`|  |  |  |  |  |  |  | 
|`curandStateScrambledSobol64_t`|  |  |  |  |  |  |  | 
|`curandStateSobol32`|  |  |  | `hiprandStateSobol32` | 1.8.0 |  |  | 
|`curandStateSobol32_t`|  |  |  | `hiprandStateSobol32_t` | 1.5.0 |  |  | 
|`curandStateSobol64`|  |  |  |  |  |  |  | 
|`curandStateSobol64_t`|  |  |  |  |  |  |  | 
|`curandStateXORWOW`|  |  |  | `hiprandStateXORWOW` | 1.8.0 |  |  | 
|`curandStateXORWOW_t`|  |  |  | `hiprandStateXORWOW_t` | 1.5.0 |  |  | 
|`curandState_t`|  |  |  | `hiprandState_t` | 1.5.0 |  |  | 
|`curandStatus`|  |  |  | `hiprandStatus_t` | 1.5.0 |  |  | 
|`curandStatus_t`|  |  |  | `hiprandStatus_t` | 1.5.0 |  |  | 

## **2. Host API Functions**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`curandCreateGenerator`|  |  |  | `hiprandCreateGenerator` | 1.5.0 |  |  | 
|`curandCreateGeneratorHost`|  |  |  | `hiprandCreateGeneratorHost` | 1.5.0 |  |  | 
|`curandCreatePoissonDistribution`|  |  |  | `hiprandCreatePoissonDistribution` | 1.5.0 |  |  | 
|`curandDestroyDistribution`|  |  |  | `hiprandDestroyDistribution` | 1.5.0 |  |  | 
|`curandDestroyGenerator`|  |  |  | `hiprandDestroyGenerator` | 1.5.0 |  |  | 
|`curandGenerate`|  |  |  | `hiprandGenerate` | 1.5.0 |  |  | 
|`curandGenerateLogNormal`|  |  |  | `hiprandGenerateLogNormal` | 1.5.0 |  |  | 
|`curandGenerateLogNormalDouble`|  |  |  | `hiprandGenerateLogNormalDouble` | 1.5.0 |  |  | 
|`curandGenerateLongLong`|  |  |  |  |  |  |  | 
|`curandGenerateNormal`|  |  |  | `hiprandGenerateNormal` | 1.5.0 |  |  | 
|`curandGenerateNormalDouble`|  |  |  | `hiprandGenerateNormalDouble` | 1.5.0 |  |  | 
|`curandGeneratePoisson`|  |  |  | `hiprandGeneratePoisson` | 1.5.0 |  |  | 
|`curandGenerateSeeds`|  |  |  | `hiprandGenerateSeeds` | 1.5.0 |  |  | 
|`curandGenerateUniform`|  |  |  | `hiprandGenerateUniform` | 1.5.0 |  |  | 
|`curandGenerateUniformDouble`|  |  |  | `hiprandGenerateUniformDouble` | 1.5.0 |  |  | 
|`curandGetDirectionVectors32`|  |  |  |  |  |  |  | 
|`curandGetDirectionVectors64`|  |  |  |  |  |  |  | 
|`curandGetProperty`| 8.0 |  |  |  |  |  |  | 
|`curandGetScrambleConstants32`|  |  |  |  |  |  |  | 
|`curandGetScrambleConstants64`|  |  |  |  |  |  |  | 
|`curandGetVersion`|  |  |  | `hiprandGetVersion` | 1.5.0 |  |  | 
|`curandMakeMTGP32Constants`|  |  |  | `hiprandMakeMTGP32Constants` | 1.5.0 |  |  | 
|`curandMakeMTGP32KernelState`|  |  |  | `hiprandMakeMTGP32KernelState` | 1.5.0 |  |  | 
|`curandSetGeneratorOffset`|  |  |  | `hiprandSetGeneratorOffset` | 1.5.0 |  |  | 
|`curandSetGeneratorOrdering`|  |  |  |  |  |  |  | 
|`curandSetPseudoRandomGeneratorSeed`|  |  |  | `hiprandSetPseudoRandomGeneratorSeed` | 1.5.0 |  |  | 
|`curandSetQuasiRandomGeneratorDimensions`|  |  |  | `hiprandSetQuasiRandomGeneratorDimensions` | 1.5.0 |  |  | 
|`curandSetStream`|  |  |  | `hiprandSetStream` | 1.5.0 |  |  | 

## **3. Device API Functions**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`curand`|  |  |  | `hiprand` | 1.5.0 |  |  | 
|`curand_Philox4x32_10`|  |  |  |  |  |  |  | 
|`curand_discrete`|  |  |  | `hiprand_discrete` | 1.5.0 |  |  | 
|`curand_discrete4`|  |  |  | `hiprand_discrete4` | 1.5.0 |  |  | 
|`curand_init`|  |  |  | `hiprand_init` | 1.5.0 |  |  | 
|`curand_log_normal`|  |  |  | `hiprand_log_normal` | 1.5.0 |  |  | 
|`curand_log_normal2`|  |  |  | `hiprand_log_normal2` | 1.5.0 |  |  | 
|`curand_log_normal2_double`|  |  |  | `hiprand_log_normal2_double` | 1.5.0 |  |  | 
|`curand_log_normal4`|  |  |  | `hiprand_log_normal4` | 1.5.0 |  |  | 
|`curand_log_normal4_double`|  |  |  | `hiprand_log_normal4_double` | 1.5.0 |  |  | 
|`curand_log_normal_double`|  |  |  | `hiprand_log_normal_double` | 1.5.0 |  |  | 
|`curand_mtgp32_single`|  |  |  |  |  |  |  | 
|`curand_mtgp32_single_specific`|  |  |  |  |  |  |  | 
|`curand_mtgp32_specific`|  |  |  |  |  |  |  | 
|`curand_normal`|  |  |  | `hiprand_normal` | 1.5.0 |  |  | 
|`curand_normal2`|  |  |  | `hiprand_normal2` | 1.5.0 |  |  | 
|`curand_normal2_double`|  |  |  | `hiprand_normal2_double` | 1.5.0 |  |  | 
|`curand_normal4`|  |  |  | `hiprand_normal4` | 1.5.0 |  |  | 
|`curand_normal4_double`|  |  |  | `hiprand_normal4_double` | 1.5.0 |  |  | 
|`curand_normal_double`|  |  |  | `hiprand_normal_double` | 1.5.0 |  |  | 
|`curand_poisson`|  |  |  | `hiprand_poisson` | 1.5.0 |  |  | 
|`curand_poisson4`|  |  |  | `hiprand_poisson4` | 1.5.0 |  |  | 
|`curand_uniform`|  |  |  | `hiprand_uniform` | 1.5.0 |  |  | 
|`curand_uniform2_double`|  |  |  | `hiprand_uniform2_double` | 1.5.0 |  |  | 
|`curand_uniform4`|  |  |  | `hiprand_uniform4` | 1.5.0 |  |  | 
|`curand_uniform4_double`|  |  |  | `hiprand_uniform4_double` | 1.5.0 |  |  | 
|`curand_uniform_double`|  |  |  | `hiprand_uniform_double` | 1.5.0 |  |  | 


\*A - Added; D - Deprecated; R - Removed