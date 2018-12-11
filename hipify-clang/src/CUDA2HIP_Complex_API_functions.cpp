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
