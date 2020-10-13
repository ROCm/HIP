# CURAND API supported by HIP

## **1. CURAND Data types**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`CURAND_3RD`|  |  |  ||
|`CURAND_BINARY_SEARCH`|  |  |  ||
|`CURAND_CHOOSE_BEST`|  |  |  ||
|`CURAND_DEFINITION`|  |  |  ||
|`CURAND_DEVICE_API`|  |  |  ||
|`CURAND_DIRECTION_VECTORS_32_JOEKUO6`|  |  |  ||
|`CURAND_DIRECTION_VECTORS_64_JOEKUO6`|  |  |  ||
|`CURAND_DISCRETE_GAUSS`|  |  |  ||
|`CURAND_FAST_REJECTION`|  |  |  ||
|`CURAND_HITR`|  |  |  ||
|`CURAND_ITR`|  |  |  ||
|`CURAND_KNUTH`|  |  |  ||
|`CURAND_M1`|  |  |  ||
|`CURAND_M2`|  |  |  ||
|`CURAND_ORDERING_PSEUDO_BEST`|  |  |  ||
|`CURAND_ORDERING_PSEUDO_DEFAULT`|  |  |  ||
|`CURAND_ORDERING_PSEUDO_LEGACY`| 11.0 |  |  ||
|`CURAND_ORDERING_PSEUDO_SEEDED`|  |  |  ||
|`CURAND_ORDERING_QUASI_DEFAULT`|  |  |  ||
|`CURAND_POISSON`|  |  |  ||
|`CURAND_REJECTION`|  |  |  ||
|`CURAND_RNG_PSEUDO_DEFAULT`|  |  |  |`HIPRAND_RNG_PSEUDO_DEFAULT`|
|`CURAND_RNG_PSEUDO_MRG32K3A`|  |  |  |`HIPRAND_RNG_PSEUDO_MRG32K3A`|
|`CURAND_RNG_PSEUDO_MT19937`|  |  |  |`HIPRAND_RNG_PSEUDO_MT19937`|
|`CURAND_RNG_PSEUDO_MTGP32`|  |  |  |`HIPRAND_RNG_PSEUDO_MTGP32`|
|`CURAND_RNG_PSEUDO_PHILOX4_32_10`|  |  |  |`HIPRAND_RNG_PSEUDO_PHILOX4_32_10`|
|`CURAND_RNG_PSEUDO_XORWOW`|  |  |  |`HIPRAND_RNG_PSEUDO_XORWOW`|
|`CURAND_RNG_QUASI_DEFAULT`|  |  |  |`HIPRAND_RNG_QUASI_DEFAULT`|
|`CURAND_RNG_QUASI_SCRAMBLED_SOBOL32`|  |  |  |`HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32`|
|`CURAND_RNG_QUASI_SCRAMBLED_SOBOL64`|  |  |  |`HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64`|
|`CURAND_RNG_QUASI_SOBOL32`|  |  |  |`HIPRAND_RNG_QUASI_SOBOL32`|
|`CURAND_RNG_QUASI_SOBOL64`|  |  |  |`HIPRAND_RNG_QUASI_SOBOL64`|
|`CURAND_RNG_TEST`|  |  |  |`HIPRAND_RNG_TEST`|
|`CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6`|  |  |  ||
|`CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6`|  |  |  ||
|`CURAND_STATUS_ALLOCATION_FAILED`|  |  |  |`HIPRAND_STATUS_ALLOCATION_FAILED`|
|`CURAND_STATUS_ARCH_MISMATCH`|  |  |  |`HIPRAND_STATUS_ARCH_MISMATCH`|
|`CURAND_STATUS_DOUBLE_PRECISION_REQUIRED`|  |  |  |`HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED`|
|`CURAND_STATUS_INITIALIZATION_FAILED`|  |  |  |`HIPRAND_STATUS_INITIALIZATION_FAILED`|
|`CURAND_STATUS_INTERNAL_ERROR`|  |  |  |`HIPRAND_STATUS_INTERNAL_ERROR`|
|`CURAND_STATUS_LAUNCH_FAILURE`|  |  |  |`HIPRAND_STATUS_LAUNCH_FAILURE`|
|`CURAND_STATUS_LENGTH_NOT_MULTIPLE`|  |  |  |`HIPRAND_STATUS_LENGTH_NOT_MULTIPLE`|
|`CURAND_STATUS_NOT_INITIALIZED`|  |  |  |`HIPRAND_STATUS_NOT_INITIALIZED`|
|`CURAND_STATUS_OUT_OF_RANGE`|  |  |  |`HIPRAND_STATUS_OUT_OF_RANGE`|
|`CURAND_STATUS_PREEXISTING_FAILURE`|  |  |  |`HIPRAND_STATUS_PREEXISTING_FAILURE`|
|`CURAND_STATUS_SUCCESS`|  |  |  |`HIPRAND_STATUS_SUCCESS`|
|`CURAND_STATUS_TYPE_ERROR`|  |  |  |`HIPRAND_STATUS_TYPE_ERROR`|
|`CURAND_STATUS_VERSION_MISMATCH`|  |  |  |`HIPRAND_STATUS_VERSION_MISMATCH`|
|`CURAND_VERSION`| 10.2 |  |  ||
|`CURAND_VER_BUILD`| 10.2 |  |  ||
|`CURAND_VER_MAJOR`| 10.2 |  |  ||
|`CURAND_VER_MINOR`| 10.2 |  |  ||
|`CURAND_VER_PATCH`| 10.2 |  |  ||
|`curandDirectionVectorSet`|  |  |  ||
|`curandDirectionVectorSet_t`|  |  |  ||
|`curandDirectionVectors32_t`|  |  |  |`hiprandDirectionVectors32_t`|
|`curandDirectionVectors64_t`|  |  |  ||
|`curandDiscreteDistribution_st`|  |  |  |`hiprandDiscreteDistribution_st`|
|`curandDiscreteDistribution_t`|  |  |  |`hiprandDiscreteDistribution_t`|
|`curandDistributionM2Shift_st`|  |  |  ||
|`curandDistributionM2Shift_t`|  |  |  ||
|`curandDistributionShift_st`|  |  |  ||
|`curandDistributionShift_t`|  |  |  ||
|`curandDistribution_st`|  |  |  ||
|`curandDistribution_t`|  |  |  ||
|`curandGenerator_st`|  |  |  |`hiprandGenerator_st`|
|`curandGenerator_t`|  |  |  |`hiprandGenerator_t`|
|`curandHistogramM2K_st`|  |  |  ||
|`curandHistogramM2K_t`|  |  |  ||
|`curandHistogramM2V_st`|  |  |  ||
|`curandHistogramM2V_t`|  |  |  ||
|`curandHistogramM2_st`|  |  |  ||
|`curandHistogramM2_t`|  |  |  ||
|`curandMethod`|  |  |  ||
|`curandMethod_t`|  |  |  ||
|`curandOrdering`|  |  |  ||
|`curandOrdering_t`|  |  |  ||
|`curandRngType`|  |  |  |`hiprandRngType_t`|
|`curandRngType_t`|  |  |  |`hiprandRngType_t`|
|`curandState`|  |  |  |`hiprandState`|
|`curandStateMRG32k3a`|  |  |  |`hiprandStateMRG32k3a`|
|`curandStateMRG32k3a_t`|  |  |  |`hiprandStateMRG32k3a_t`|
|`curandStateMtgp32`|  |  |  |`hiprandStateMtgp32`|
|`curandStateMtgp32_t`|  |  |  |`hiprandStateMtgp32_t`|
|`curandStatePhilox4_32_10`|  |  |  |`hiprandStatePhilox4_32_10`|
|`curandStatePhilox4_32_10_t`|  |  |  |`hiprandStatePhilox4_32_10_t`|
|`curandStateScrambledSobol32`|  |  |  ||
|`curandStateScrambledSobol32_t`|  |  |  ||
|`curandStateScrambledSobol64`|  |  |  ||
|`curandStateScrambledSobol64_t`|  |  |  ||
|`curandStateSobol32`|  |  |  |`hiprandStateSobol32`|
|`curandStateSobol32_t`|  |  |  |`hiprandStateSobol32_t`|
|`curandStateSobol64`|  |  |  ||
|`curandStateSobol64_t`|  |  |  ||
|`curandStateXORWOW`|  |  |  |`hiprandStateXORWOW`|
|`curandStateXORWOW_t`|  |  |  |`hiprandStateXORWOW_t`|
|`curandState_t`|  |  |  |`hiprandState_t`|
|`curandStatus`|  |  |  |`hiprandStatus_t`|
|`curandStatus_t`|  |  |  |`hiprandStatus_t`|

## **2. Host API Functions**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`curandCreateGenerator`|  |  |  |`hiprandCreateGenerator`|
|`curandCreateGeneratorHost`|  |  |  |`hiprandCreateGeneratorHost`|
|`curandCreatePoissonDistribution`|  |  |  |`hiprandCreatePoissonDistribution`|
|`curandDestroyDistribution`|  |  |  |`hiprandDestroyDistribution`|
|`curandDestroyGenerator`|  |  |  |`hiprandDestroyGenerator`|
|`curandGenerate`|  |  |  |`hiprandGenerate`|
|`curandGenerateLogNormal`|  |  |  |`hiprandGenerateLogNormal`|
|`curandGenerateLogNormalDouble`|  |  |  |`hiprandGenerateLogNormalDouble`|
|`curandGenerateLongLong`|  |  |  ||
|`curandGenerateNormal`|  |  |  |`hiprandGenerateNormal`|
|`curandGenerateNormalDouble`|  |  |  |`hiprandGenerateNormalDouble`|
|`curandGeneratePoisson`|  |  |  |`hiprandGeneratePoisson`|
|`curandGenerateSeeds`|  |  |  |`hiprandGenerateSeeds`|
|`curandGenerateUniform`|  |  |  |`hiprandGenerateUniform`|
|`curandGenerateUniformDouble`|  |  |  |`hiprandGenerateUniformDouble`|
|`curandGetDirectionVectors32`|  |  |  ||
|`curandGetDirectionVectors64`|  |  |  ||
|`curandGetProperty`| 8.0 |  |  ||
|`curandGetScrambleConstants32`|  |  |  ||
|`curandGetScrambleConstants64`|  |  |  ||
|`curandGetVersion`|  |  |  |`hiprandGetVersion`|
|`curandMakeMTGP32Constants`|  |  |  |`hiprandMakeMTGP32Constants`|
|`curandMakeMTGP32KernelState`|  |  |  |`hiprandMakeMTGP32KernelState`|
|`curandSetGeneratorOffset`|  |  |  |`hiprandSetGeneratorOffset`|
|`curandSetGeneratorOrdering`|  |  |  ||
|`curandSetPseudoRandomGeneratorSeed`|  |  |  |`hiprandSetPseudoRandomGeneratorSeed`|
|`curandSetQuasiRandomGeneratorDimensions`|  |  |  |`hiprandSetQuasiRandomGeneratorDimensions`|
|`curandSetStream`|  |  |  |`hiprandSetStream`|

## **3. Device API Functions**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`curand`|  |  |  |`hiprand`|
|`curand_Philox4x32_10`|  |  |  ||
|`curand_discrete`|  |  |  |`hiprand_discrete`|
|`curand_discrete4`|  |  |  |`hiprand_discrete4`|
|`curand_init`|  |  |  |`hiprand_init`|
|`curand_log_normal`|  |  |  |`hiprand_log_normal`|
|`curand_log_normal2`|  |  |  |`hiprand_log_normal2`|
|`curand_log_normal2_double`|  |  |  |`hiprand_log_normal2_double`|
|`curand_log_normal4`|  |  |  |`hiprand_log_normal4`|
|`curand_log_normal4_double`|  |  |  |`hiprand_log_normal4_double`|
|`curand_log_normal_double`|  |  |  |`hiprand_log_normal_double`|
|`curand_mtgp32_single`|  |  |  ||
|`curand_mtgp32_single_specific`|  |  |  ||
|`curand_mtgp32_specific`|  |  |  ||
|`curand_normal`|  |  |  |`hiprand_normal`|
|`curand_normal2`|  |  |  |`hiprand_normal2`|
|`curand_normal2_double`|  |  |  |`hiprand_normal2_double`|
|`curand_normal4`|  |  |  |`hiprand_normal4`|
|`curand_normal4_double`|  |  |  |`hiprand_normal4_double`|
|`curand_normal_double`|  |  |  |`hiprand_normal_double`|
|`curand_poisson`|  |  |  |`hiprand_poisson`|
|`curand_poisson4`|  |  |  |`hiprand_poisson4`|
|`curand_uniform`|  |  |  |`hiprand_uniform`|
|`curand_uniform2_double`|  |  |  |`hiprand_uniform2_double`|
|`curand_uniform4`|  |  |  |`hiprand_uniform4`|
|`curand_uniform4_double`|  |  |  |`hiprand_uniform4_double`|
|`curand_uniform_double`|  |  |  |`hiprand_uniform_double`|


\* A - Added, D - Deprecated, R - Removed