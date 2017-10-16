#pragma once

#include "llvm/ADT/StringRef.h"
#include <set>
#include <map>

#include "Types.h"

// TODO: This shouldn't really be here. More restructuring needed...
struct hipCounter {
    llvm::StringRef hipName;
    ConvTypes countType;
    ApiTypes countApiType;
    int unsupported;
};

#define HIP_UNSUPPORTED -1

// Static lookup tables for mapping the CUDA API to the HIP API.
extern const std::set<llvm::StringRef> CUDA_EXCLUDES;
extern const std::map<llvm::StringRef, hipCounter> CUDA_TO_HIP_RENAMES;
