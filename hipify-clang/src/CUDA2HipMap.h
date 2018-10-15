#pragma once

#include "llvm/ADT/StringRef.h"
#include <set>
#include <map>
#include "Statistics.h"

#define HIP_UNSUPPORTED true

// Maps cuda header names to hip header names.
extern const std::map<llvm::StringRef, hipCounter> CUDA_INCLUDE_MAP;

// Maps the names of CUDA types to the corresponding hip types.
extern const std::map<llvm::StringRef, hipCounter> CUDA_TYPE_NAME_MAP;

// Map all other CUDA identifiers (function/macro names, enum values) to hip versions.
extern const std::map<llvm::StringRef, hipCounter> CUDA_IDENTIFIER_MAP;

/**
  * The union of all the above maps, except includes.
  *
  * This should be used rarely, but is still needed to convert macro definitions (which can
  * contain any combination of the above things). AST walkers can usually get away with just
  * looking in the lookup table for the type of element they are processing, however, saving
  * a great deal of time.
  */
const std::map<llvm::StringRef, hipCounter>& CUDA_RENAMES_MAP();
