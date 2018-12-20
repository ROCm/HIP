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

// Maps the names of CUDA Complex API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_COMPLEX_FUNCTION_MAP{
  {"cuCrealf",               {"hipCrealf",               "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCimagf",               {"hipCimagf",               "", CONV_COMPLEX, API_COMPLEX}},
  {"make_cuFloatComplex",    {"make_hipFloatComplex",    "", CONV_COMPLEX, API_COMPLEX}},
  {"cuConjf",                {"hipConjf",                "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCaddf",                {"hipCaddf",                "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCsubf",                {"hipCsubf",                "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCmulf",                {"hipCmulf",                "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCdivf",                {"hipCdivf",                "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCabsf",                {"hipCabsf",                "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCreal",                {"hipCreal",                "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCimag",                {"hipCimag",                "", CONV_COMPLEX, API_COMPLEX}},
  {"make_cuDoubleComplex",   {"make_hipDoubleComplex",   "", CONV_COMPLEX, API_COMPLEX}},
  {"cuConj",                 {"hipConj",                 "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCadd",                 {"hipCadd",                 "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCsub",                 {"hipCsub",                 "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCmul",                 {"hipCmul",                 "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCdiv",                 {"hipCdiv",                 "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCabs",                 {"hipCabs",                 "", CONV_COMPLEX, API_COMPLEX}},
  {"make_cuComplex",         {"make_hipComplex",         "", CONV_COMPLEX, API_COMPLEX}},
  {"cuComplexFloatToDouble", {"hipComplexFloatToDouble", "", CONV_COMPLEX, API_COMPLEX}},
  {"cuComplexDoubleToFloat", {"hipComplexDoubleToFloat", "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCfmaf",                {"hipCfmaf",                "", CONV_COMPLEX, API_COMPLEX}},
  {"cuCfma",                 {"hipCfma",                 "", CONV_COMPLEX, API_COMPLEX}},
};
