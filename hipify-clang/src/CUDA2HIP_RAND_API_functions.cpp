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
const std::map<llvm::StringRef, hipCounter> CUDA_RAND_FUNCTION_MAP{
  // RAND Host functions
  {"curandCreateGenerator",                         {"hiprandCreateGenerator",                         "", CONV_LIB_FUNC, API_RAND}},
  {"curandCreateGeneratorHost",                     {"hiprandCreateGeneratorHost",                     "", CONV_LIB_FUNC, API_RAND}},
  {"curandCreatePoissonDistribution",               {"hiprandCreatePoissonDistribution",               "", CONV_LIB_FUNC, API_RAND}},
  {"curandDestroyDistribution",                     {"hiprandDestroyDistribution",                     "", CONV_LIB_FUNC, API_RAND}},
  {"curandDestroyGenerator",                        {"hiprandDestroyGenerator",                        "", CONV_LIB_FUNC, API_RAND}},
  {"curandGenerate",                                {"hiprandGenerate",                                "", CONV_LIB_FUNC, API_RAND}},
  {"curandGenerateLogNormal",                       {"hiprandGenerateLogNormal",                       "", CONV_LIB_FUNC, API_RAND}},
  {"curandGenerateLogNormalDouble",                 {"hiprandGenerateLogNormalDouble",                 "", CONV_LIB_FUNC, API_RAND}},
  {"curandGenerateLongLong",                        {"hiprandGenerateLongLong",                        "", CONV_LIB_FUNC, API_RAND, HIP_UNSUPPORTED}},
  {"curandGenerateNormal",                          {"hiprandGenerateNormal",                          "", CONV_LIB_FUNC, API_RAND}},
  {"curandGenerateNormalDouble",                    {"hiprandGenerateNormalDouble",                    "", CONV_LIB_FUNC, API_RAND}},
  {"curandGeneratePoisson",                         {"hiprandGeneratePoisson",                         "", CONV_LIB_FUNC, API_RAND}},
  {"curandGenerateSeeds",                           {"hiprandGenerateSeeds",                           "", CONV_LIB_FUNC, API_RAND}},
  {"curandGenerateUniform",                         {"hiprandGenerateUniform",                         "", CONV_LIB_FUNC, API_RAND}},
  {"curandGenerateUniformDouble",                   {"hiprandGenerateUniformDouble",                   "", CONV_LIB_FUNC, API_RAND}},
  {"curandGetDirectionVectors32",                   {"hiprandGetDirectionVectors32",                   "", CONV_LIB_FUNC, API_RAND, HIP_UNSUPPORTED}},
  {"curandGetDirectionVectors64",                   {"hiprandGetDirectionVectors64",                   "", CONV_LIB_FUNC, API_RAND, HIP_UNSUPPORTED}},
  {"curandGetProperty",                             {"hiprandGetProperty",                             "", CONV_LIB_FUNC, API_RAND, HIP_UNSUPPORTED}},
  {"curandGetScrambleConstants32",                  {"hiprandGetScrambleConstants32",                  "", CONV_LIB_FUNC, API_RAND, HIP_UNSUPPORTED}},
  {"curandGetScrambleConstants64",                  {"hiprandGetScrambleConstants64",                  "", CONV_LIB_FUNC, API_RAND, HIP_UNSUPPORTED}},
  {"curandGetVersion",                              {"hiprandGetVersion",                              "", CONV_LIB_FUNC, API_RAND}},
  {"curandSetGeneratorOffset",                      {"hiprandSetGeneratorOffset",                      "", CONV_LIB_FUNC, API_RAND}},
  {"curandSetGeneratorOrdering",                    {"hiprandSetGeneratorOrdering",                    "", CONV_LIB_FUNC, API_RAND, HIP_UNSUPPORTED}},
  {"curandSetPseudoRandomGeneratorSeed",            {"hiprandSetPseudoRandomGeneratorSeed",            "", CONV_LIB_FUNC, API_RAND}},
  {"curandSetQuasiRandomGeneratorDimensions",       {"hiprandSetQuasiRandomGeneratorDimensions",       "", CONV_LIB_FUNC, API_RAND}},
  {"curandSetStream",                               {"hiprandSetStream",                               "", CONV_LIB_FUNC, API_RAND}},

  // RAND Device functions
  {"curand",                                        {"hiprand",                                        "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_init",                                   {"hiprand_init",                                   "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_log_normal",                             {"hiprand_log_normal",                             "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_log_normal_double",                      {"hiprand_log_normal_double",                      "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_log_normal2",                            {"hiprand_log_normal2",                            "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_log_normal2_double",                     {"hiprand_log_normal2_double",                     "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_log_normal4",                            {"hiprand_log_normal4",                            "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_log_normal4_double",                     {"hiprand_log_normal4_double",                     "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_mtgp32_single",                          {"hiprand_mtgp32_single",                          "", CONV_LIB_DEVICE_FUNC, API_RAND, HIP_UNSUPPORTED}},
  {"curand_mtgp32_single_specific",                 {"hiprand_mtgp32_single_specific",                 "", CONV_LIB_DEVICE_FUNC, API_RAND, HIP_UNSUPPORTED}},
  {"curand_mtgp32_specific",                        {"hiprand_mtgp32_specific",                        "", CONV_LIB_DEVICE_FUNC, API_RAND, HIP_UNSUPPORTED}},
  {"curand_normal",                                 {"hiprand_normal",                                 "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_normal_double",                          {"hiprand_normal_double",                          "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_normal2",                                {"hiprand_normal2",                                "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_normal2_double",                         {"hiprand_normal2_double",                         "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_normal4",                                {"hiprand_normal4",                                "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_normal4_double",                         {"hiprand_normal4_double",                         "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_uniform",                                {"hiprand_uniform",                                "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_uniform_double",                         {"hiprand_uniform_double",                         "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_uniform2_double",                        {"hiprand_uniform2_double",                        "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_uniform4",                               {"hiprand_uniform4",                               "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_uniform4_double",                        {"hiprand_uniform4_double",                        "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_discrete",                               {"hiprand_discrete",                               "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_discrete4",                              {"hiprand_discrete4",                              "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_poisson",                                {"hiprand_poisson",                                "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_poisson4",                               {"hiprand_poisson4",                               "", CONV_LIB_DEVICE_FUNC, API_RAND}},
  {"curand_Philox4x32_10",                          {"hiprand_Philox4x32_10",                          "", CONV_LIB_DEVICE_FUNC, API_RAND, HIP_UNSUPPORTED}},
  // unchanged function names: skipahead, skipahead_sequence, skipahead_subsequence
};
