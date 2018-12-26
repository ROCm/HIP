/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "CUDA2HIP.h"

// Map of all functions
const std::map<llvm::StringRef, hipCounter> CUDA_RAND_TYPE_NAME_MAP{
  {"curandStatus",                  {"hiprandStatus_t",                "", CONV_TYPE, API_RAND}},
  {"curandStatus_t",                {"hiprandStatus_t",                "", CONV_TYPE, API_RAND}},
  {"curandRngType",                 {"hiprandRngType_t",               "", CONV_TYPE, API_RAND}},
  {"curandRngType_t",               {"hiprandRngType_t",               "", CONV_TYPE, API_RAND}},
  {"curandGenerator_st",            {"hiprandGenerator_st",            "", CONV_TYPE, API_RAND}},
  {"curandGenerator_t",             {"hiprandGenerator_t",             "", CONV_TYPE, API_RAND}},
  {"curandDirectionVectorSet",      {"hiprandDirectionVectorSet_t",    "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandDirectionVectorSet_t",    {"hiprandDirectionVectorSet_t",    "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandOrdering",                {"hiprandOrdering_t",              "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandOrdering_t",              {"hiprandOrdering_t",              "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandDistribution_st",         {"hiprandDistribution_st",         "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandHistogramM2V_st",         {"hiprandDistribution_st",         "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandDistribution_t",          {"hiprandDistribution_t",          "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandHistogramM2V_t",          {"hiprandDistribution_t",          "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandDistributionShift_st",    {"hiprandDistributionShift_st",    "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandDistributionShift_t",     {"hiprandDistributionShift_t",     "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandDistributionM2Shift_st",  {"hiprandDistributionM2Shift_st",  "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandDistributionM2Shift_t",   {"hiprandDistributionM2Shift_t",   "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandHistogramM2_st",          {"hiprandHistogramM2_st",          "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandHistogramM2_t",           {"hiprandHistogramM2_t",           "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandHistogramM2K_st",         {"hiprandHistogramM2K_st",         "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandHistogramM2K_t",          {"hiprandHistogramM2K_t",          "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandDiscreteDistribution_st", {"hiprandDiscreteDistribution_st", "", CONV_TYPE, API_RAND}},
  {"curandDiscreteDistribution_t",  {"hiprandDiscreteDistribution_t",  "", CONV_TYPE, API_RAND}},
  {"curandMethod",                  {"hiprandMethod_t",                "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandMethod_t",                {"hiprandMethod_t",                "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandDirectionVectors32_t",    {"hiprandDirectionVectors32_t",    "", CONV_TYPE, API_RAND}},
  {"curandDirectionVectors64_t",    {"hiprandDirectionVectors64_t",    "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  // cuRAND types for Device functions
  {"curandStateMtgp32_t",           {"hiprandStateMtgp32_t",           "", CONV_TYPE, API_RAND}},
  {"curandStateScrambledSobol64_t", {"hiprandStateScrambledSobol64_t", "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandStateSobol64_t",          {"hiprandStateSobol64_t",          "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandStateScrambledSobol32_t", {"hiprandStateScrambledSobol32_t", "", CONV_TYPE, API_RAND, HIP_UNSUPPORTED}},
  {"curandStateSobol32_t",          {"hiprandStateSobol32_t",          "", CONV_TYPE, API_RAND}},
  {"curandStateMRG32k3a_t",         {"hiprandStateMRG32k3a_t",         "", CONV_TYPE, API_RAND}},
  {"curandStatePhilox4_32_10_t",    {"hiprandStatePhilox4_32_10_t",    "", CONV_TYPE, API_RAND}},
  {"curandStateXORWOW_t",           {"hiprandStateXORWOW_t",           "", CONV_TYPE, API_RAND}},
  {"curandState_t",                 {"hiprandState_t",                 "", CONV_TYPE, API_RAND}},
  {"curandState",                   {"hiprandState_t",                 "", CONV_TYPE, API_RAND}},


  // RAND function call status types (enum curandStatus)
  {"CURAND_STATUS_SUCCESS",                         {"HIPRAND_STATUS_SUCCESS",                         "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_VERSION_MISMATCH",                {"HIPRAND_STATUS_VERSION_MISMATCH",                "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_NOT_INITIALIZED",                 {"HIPRAND_STATUS_NOT_INITIALIZED",                 "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_ALLOCATION_FAILED",               {"HIPRAND_STATUS_ALLOCATION_FAILED",               "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_TYPE_ERROR",                      {"HIPRAND_STATUS_TYPE_ERROR",                      "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_OUT_OF_RANGE",                    {"HIPRAND_STATUS_OUT_OF_RANGE",                    "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_LENGTH_NOT_MULTIPLE",             {"HIPRAND_STATUS_LENGTH_NOT_MULTIPLE",             "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_DOUBLE_PRECISION_REQUIRED",       {"HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED",       "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_LAUNCH_FAILURE",                  {"HIPRAND_STATUS_LAUNCH_FAILURE",                  "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_PREEXISTING_FAILURE",             {"HIPRAND_STATUS_PREEXISTING_FAILURE",             "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_INITIALIZATION_FAILED",           {"HIPRAND_STATUS_INITIALIZATION_FAILED",           "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_ARCH_MISMATCH",                   {"HIPRAND_STATUS_ARCH_MISMATCH",                   "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_STATUS_INTERNAL_ERROR",                  {"HIPRAND_STATUS_INTERNAL_ERROR",                  "", CONV_NUMERIC_LITERAL, API_RAND}},

  // RAND generator types (enum curandRngType)
  {"CURAND_RNG_TEST",                               {"HIPRAND_RNG_TEST",                               "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_RNG_PSEUDO_DEFAULT",                     {"HIPRAND_RNG_PSEUDO_DEFAULT",                     "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_RNG_PSEUDO_XORWOW",                      {"HIPRAND_RNG_PSEUDO_XORWOW",                      "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_RNG_PSEUDO_MRG32K3A",                    {"HIPRAND_RNG_PSEUDO_MRG32K3A",                    "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_RNG_PSEUDO_MTGP32",                      {"HIPRAND_RNG_PSEUDO_MTGP32",                      "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_RNG_PSEUDO_MT19937",                     {"HIPRAND_RNG_PSEUDO_MT19937",                     "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_RNG_PSEUDO_PHILOX4_32_10",               {"HIPRAND_RNG_PSEUDO_PHILOX4_32_10",               "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_RNG_QUASI_DEFAULT",                      {"HIPRAND_RNG_QUASI_DEFAULT",                      "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_RNG_QUASI_SOBOL32",                      {"HIPRAND_RNG_QUASI_SOBOL32",                      "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_RNG_QUASI_SCRAMBLED_SOBOL32",            {"HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32",            "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_RNG_QUASI_SOBOL64",                      {"HIPRAND_RNG_QUASI_SOBOL64",                      "", CONV_NUMERIC_LITERAL, API_RAND}},
  {"CURAND_RNG_QUASI_SCRAMBLED_SOBOL64",            {"HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64",            "", CONV_NUMERIC_LITERAL, API_RAND}},

  // RAND ordering of results in memory (enum curandOrdering)
  {"CURAND_ORDERING_PSEUDO_BEST",                   {"HIPRAND_ORDERING_PSEUDO_BEST",                   "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_ORDERING_PSEUDO_DEFAULT",                {"HIPRAND_ORDERING_PSEUDO_DEFAULT",                "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_ORDERING_PSEUDO_SEEDED",                 {"HIPRAND_ORDERING_PSEUDO_SEEDED",                 "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_ORDERING_QUASI_DEFAULT",                 {"HIPRAND_ORDERING_QUASI_DEFAULT",                 "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},

  // RAND choice of direction vector set (enum curandDirectionVectorSet)
  {"CURAND_DIRECTION_VECTORS_32_JOEKUO6",           {"HIPRAND_DIRECTION_VECTORS_32_JOEKUO6",           "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6", {"HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6", "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_DIRECTION_VECTORS_64_JOEKUO6",           {"HIPRAND_DIRECTION_VECTORS_64_JOEKUO6",           "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6", {"HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6", "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},

  // RAND method (enum curandMethod)
  {"CURAND_CHOOSE_BEST",                            {"HIPRAND_CHOOSE_BEST",                            "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_ITR",                                    {"HIPRAND_ITR",                                    "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_KNUTH",                                  {"HIPRAND_KNUTH",                                  "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_HITR",                                   {"HIPRAND_HITR",                                   "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_M1",                                     {"HIPRAND_M1",                                     "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_M2",                                     {"HIPRAND_M2",                                     "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_BINARY_SEARCH",                          {"HIPRAND_BINARY_SEARCH",                          "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_DISCRETE_GAUSS",                         {"HIPRAND_DISCRETE_GAUSS",                         "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_REJECTION",                              {"HIPRAND_REJECTION",                              "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_DEVICE_API",                             {"HIPRAND_DEVICE_API",                             "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_FAST_REJECTION",                         {"HIPRAND_FAST_REJECTION",                         "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_3RD",                                    {"HIPRAND_3RD",                                    "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_DEFINITION",                             {"HIPRAND_DEFINITION",                             "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
  {"CURAND_POISSON",                                {"HIPRAND_POISSON",                                "", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED}},
};
