#include "CUDA2HIP.h"

// Maps the names of CUDA Complex API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_COMPLEX_TYPE_NAME_MAP{
  {"cuFloatComplex",  {"hipFloatComplex",  "", CONV_TYPE, API_COMPLEX}},
  {"cuDoubleComplex", {"hipDoubleComplex", "", CONV_TYPE, API_COMPLEX}},
  {"cuComplex",       {"hipComplex",       "", CONV_TYPE, API_COMPLEX}},
};
