#pragma once

#include "llvm/ADT/StringRef.h"
#include <set>
#include <map>
#include "Statistics.h"

#define HIP_UNSUPPORTED true

// Maps CUDA header names to HIP header names
extern const std::map<llvm::StringRef, hipCounter> CUDA_INCLUDE_MAP;
// Maps the names of CUDA DRIVER API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_DRIVER_TYPE_NAME_MAP;
// Maps the names of CUDA DRIVER API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_DRIVER_FUNCTION_MAP;
// Maps the names of CUDA RUNTIME API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_TYPE_NAME_MAP;
// Maps the names of CUDA Complex API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_COMPLEX_TYPE_NAME_MAP;
// Maps the names of CUDA Complex API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_COMPLEX_FUNCTION_MAP;
// Maps the names of CUDA RUNTIME API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_FUNCTION_MAP;
// Maps the names of CUDA BLAS API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_BLAS_TYPE_NAME_MAP;
// Maps the names of CUDA BLAS API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_BLAS_FUNCTION_MAP;
// Maps the names of CUDA RAND API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_RAND_TYPE_NAME_MAP;
// Maps the names of CUDA RAND API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_RAND_FUNCTION_MAP;
// Maps the names of CUDA DNN API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_DNN_TYPE_NAME_MAP;
// Maps the names of CUDA DNN API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_DNN_FUNCTION_MAP;
// Maps the names of CUDA FFT API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_FFT_TYPE_NAME_MAP;
// Maps the names of CUDA FFT API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_FFT_FUNCTION_MAP;

/**
  * The union of all the above maps, except includes.
  *
  * This should be used rarely, but is still needed to convert macro definitions (which can
  * contain any combination of the above things). AST walkers can usually get away with just
  * looking in the lookup table for the type of element they are processing, however, saving
  * a great deal of time.
  */
const std::map<llvm::StringRef, hipCounter>& CUDA_RENAMES_MAP();
