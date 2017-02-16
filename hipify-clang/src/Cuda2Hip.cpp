/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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
/**
 * @file Cuda2Hip.cpp
 *
 * This file is compiled and linked into clang based hipify tool.
 */
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"

#include <cstdio>
#include <fstream>
#include <set>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

#define DEBUG_TYPE "cuda2hip"
#define HIP_UNSUPPORTED -1

enum ConvTypes {
  CONV_DRIVER = 0,
  CONV_DEV,
  CONV_MEM,
  CONV_KERN,
  CONV_COORD_FUNC,
  CONV_MATH_FUNC,
  CONV_SPECIAL_FUNC,
  CONV_STREAM,
  CONV_EVENT,
  CONV_OCCUPANCY,
  CONV_CONTEXT,
  CONV_MODULE,
  CONV_CACHE,
  CONV_ERR,
  CONV_DEF,
  CONV_TEX,
  CONV_OTHER,
  CONV_INCLUDE,
  CONV_INCLUDE_CUDA_MAIN_H,
  CONV_TYPE,
  CONV_LITERAL,
  CONV_NUMERIC_LITERAL,
  CONV_LAST
};

const char *counterNames[CONV_LAST] = {
    "driver",       "dev",      "mem",   "kern",      "coord_func", "math_func",
    "special_func", "stream",   "event", "occupancy", "ctx",        "module",
    "cache",        "err",      "def",   "tex",       "other",      "include",
    "include_cuda_main_header", "type",  "literal",   "numeric_literal"};

enum ApiTypes {
  API_DRIVER = 0,
  API_RUNTIME,
  API_BLAS,
  API_LAST
};

const char *apiNames[API_LAST] = {
    "CUDA Driver API", "CUDA RT API", "CUBLAS API"};

// Set up the command line options
static cl::OptionCategory ToolTemplateCategory("CUDA to HIP source translator options");

static cl::opt<std::string> OutputFilename("o",
  cl::desc("Output filename"),
  cl::value_desc("filename"),
  cl::cat(ToolTemplateCategory));

static cl::opt<bool> Inplace("inplace",
  cl::desc("Modify input file inplace, replacing input with hipified output, save backup in .prehip file"),
  cl::value_desc("inplace"),
  cl::cat(ToolTemplateCategory));

static cl::opt<bool> NoBackup("no-backup",
  cl::desc("Don't create a backup file for the hipified source"),
  cl::value_desc("no-backup"),
  cl::cat(ToolTemplateCategory));

static cl::opt<bool> NoOutput("no-output",
  cl::desc("Don't write any translated output to stdout"),
  cl::value_desc("no-output"),
  cl::cat(ToolTemplateCategory));

static cl::opt<bool> PrintStats("print-stats",
  cl::desc("Print translation statistics"),
  cl::value_desc("print-stats"),
  cl::cat(ToolTemplateCategory));

static cl::opt<std::string> OutputStatsFilename("o-stats",
  cl::desc("Output filename for statistics"),
  cl::value_desc("filename"),
  cl::cat(ToolTemplateCategory));

static cl::opt<bool> Examine("examine",
  cl::desc("Combines -no-output and -print-stats options"),
  cl::value_desc("examine"),
  cl::cat(ToolTemplateCategory));

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

namespace {

uint64_t countRepsTotal[CONV_LAST] = { 0 };
uint64_t countApiRepsTotal[API_LAST] = { 0 };
uint64_t countRepsTotalUnsupported[CONV_LAST] = { 0 };
uint64_t countApiRepsTotalUnsupported[API_LAST] = { 0 };
std::map<std::string, uint64_t> cuda2hipConvertedTotal;
std::map<std::string, uint64_t> cuda2hipUnconvertedTotal;

struct hipCounter {
  StringRef hipName;
  ConvTypes countType;
  ApiTypes countApiType;
  int unsupported;
};

struct cuda2hipMap {
  std::map<StringRef, hipCounter> cuda2hipRename;
  std::set<StringRef> cudaExcludes;

  cuda2hipMap() {

    // Replacement Excludes
    cudaExcludes = {"CHECK_CUDA_ERROR", "CUDA_SAFE_CALL"};

    // Defines
    cuda2hipRename["__CUDACC__"] = {"__HIPCC__", CONV_DEF, API_RUNTIME};

    // CUDA includes
    cuda2hipRename["cuda.h"]             = {"hip/hip_runtime.h", CONV_INCLUDE_CUDA_MAIN_H, API_DRIVER};
    cuda2hipRename["cuda_runtime.h"] =     {"hip/hip_runtime.h", CONV_INCLUDE_CUDA_MAIN_H, API_RUNTIME};
    cuda2hipRename["cuda_runtime_api.h"] = {"hip/hip_runtime_api.h", CONV_INCLUDE, API_RUNTIME};

    // HIP includes
    // TODO: uncomment this when hip/cudacommon.h will be renamed to hip/hipcommon.h
    //cuda2hipRename["cudacommon.h"] = {"hipcommon.h", CONV_INCLUDE, API_RUNTIME};

    // CUBLAS includes
    cuda2hipRename["cublas.h"]           = {"hipblas.h", CONV_INCLUDE, API_BLAS};
    cuda2hipRename["cublas_v2.h"]        = {"hipblas.h", CONV_INCLUDE, API_BLAS};

    // Error codes and return types
    cuda2hipRename["CUresult"]                                  = {"hipError_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["cudaError_t"]                               = {"hipError_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaError"]                                 = {"hipError", CONV_TYPE, API_RUNTIME};

    // CUDA Driver API error code only
    cuda2hipRename["CUDA_ERROR_INVALID_CONTEXT"]                = {"hipErrorInvalidContext", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_CONTEXT_ALREADY_CURRENT"]        = {"hipErrorContextAlreadyCurrent", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_MAP_FAILED"]                     = {"hipErrorMapFailed", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_UNMAP_FAILED"]                   = {"hipErrorUnmapFailed", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_ARRAY_IS_MAPPED"]                = {"hipErrorArrayIsMapped", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_ALREADY_MAPPED"]                 = {"hipErrorAlreadyMapped", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_ALREADY_ACQUIRED"]               = {"hipErrorAlreadyAcquired", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_NOT_MAPPED"]                     = {"hipErrorNotMapped", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_NOT_MAPPED_AS_ARRAY"]            = {"hipErrorNotMappedAsArray", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_NOT_MAPPED_AS_POINTER"]          = {"hipErrorNotMappedAsPointer", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_CONTEXT_ALREADY_IN_USE"]         = {"hipErrorContextAlreadyInUse", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_INVALID_SOURCE"]                 = {"hipErrorInvalidSource", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_FILE_NOT_FOUND"]                 = {"hipErrorFileNotFound", CONV_ERR, API_DRIVER};
    cuda2hipRename["CUDA_ERROR_NOT_FOUND"]                      = {"hipErrorNotFound", CONV_ERR, API_DRIVER};

    // CUDA RT API error code only
    cuda2hipRename["cudaErrorInvalidDeviceFunction"]            = {"hipErrorInvalidDeviceFunction", CONV_ERR, API_RUNTIME};
    cuda2hipRename["cudaErrorInvalidConfiguration"]             = {"hipErrorInvalidConfiguration", CONV_ERR, API_RUNTIME};
    cuda2hipRename["cudaErrorPriorLaunchFailure"]               = {"hipErrorPriorLaunchFailure", CONV_ERR, API_RUNTIME};
    cuda2hipRename["cudaErrorInvalidMemcpyDirection"]           = {"hipErrorInvalidMemcpyDirection", CONV_ERR, API_RUNTIME};
    cuda2hipRename["cudaErrorInvalidDevicePointer"]             = {"hipErrorInvalidDevicePointer", CONV_ERR, API_RUNTIME};
    cuda2hipRename["cudaErrorMissingConfiguration"]             = {"hipErrorMissingConfiguration", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_SUCCESS"]                              = {"hipSuccess", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaSuccess"]                               = {"hipSuccess", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_UNKNOWN"]                        = {"hipErrorUnknown", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorUnknown"]                          = {"hipErrorUnknown", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_NOT_INITIALIZED"]                = {"hipErrorNotInitialized", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorInitializationError"]              = {"hipErrorNotInitialized", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_DEINITIALIZED"]                  = {"hipErrorDeinitialized", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorCudartUnloading"]                  = {"hipErrorDeinitialized", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_OUT_OF_MEMORY"]                  = {"hipErrorMemoryAllocation", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorMemoryAllocation"]                 = {"hipErrorMemoryAllocation", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_INVALID_HANDLE"]                 = {"hipErrorInvalidResourceHandle", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorInvalidResourceHandle"]            = {"hipErrorInvalidResourceHandle", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_INVALID_VALUE"]                  = {"hipErrorInvalidValue", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorInvalidValue"]                     = {"hipErrorInvalidValue", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_INVALID_DEVICE"]                 = {"hipErrorInvalidDevice", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorInvalidDevice"]                    = {"hipErrorInvalidDevice", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_NOT_INITIALIZED"]                = {"hipErrorInitializationError", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorInitializationError"]              = {"hipErrorInitializationError", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_NO_DEVICE"]                      = {"hipErrorNoDevice", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorNoDevice"]                         = {"hipErrorNoDevice", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_NOT_READY"]                      = {"hipErrorNotReady", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorNotReady"]                         = {"hipErrorNotReady", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_PEER_ACCESS_NOT_ENABLED"]        = {"hipErrorPeerAccessNotEnabled", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorPeerAccessNotEnabled"]             = {"hipErrorPeerAccessNotEnabled", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED"]    = {"hipErrorPeerAccessAlreadyEnabled", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorPeerAccessAlreadyEnabled"]         = {"hipErrorPeerAccessAlreadyEnabled", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_PEER_ACCESS_UNSUPPORTED"]        = {"hipErrorPeerAccessUnsupported", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorPeerAccessUnsupported"]            = {"hipErrorPeerAccessUnsupported", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_INVALID_PTX"]                    = {"hipErrorInvalidKernelFile", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorInvalidPtx"]                       = {"hipErrorInvalidKernelFile", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_INVALID_GRAPHICS_CONTEXT"]       = {"hipErrorInvalidGraphicsContext", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorInvalidGraphicsContext"]           = {"hipErrorInvalidGraphicsContext", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND"] = {"hipErrorSharedObjectSymbolNotFound", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorSharedObjectSymbolNotFound"]       = {"hipErrorSharedObjectSymbolNotFound", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_SHARED_OBJECT_INIT_FAILED"]      = {"hipErrorSharedObjectInitFailed", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorSharedObjectInitFailed"]           = {"hipErrorSharedObjectInitFailed", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_OPERATING_SYSTEM"]               = {"hipErrorOperatingSystem", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorOperatingSystem"]                  = {"hipErrorOperatingSystem", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_ILLEGAL_ADDRESS"]                = {"hipErrorIllegalAddress", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorIllegalAddress"]                   = {"hipErrorIllegalAddress", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_LAUNCH_FAILED"]                  = {"hipErrorLaunchFailure", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorLaunchFailure"]                    = {"hipErrorLaunchFailure", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_LAUNCH_TIMEOUT"]                 = {"hipErrorLaunchTimeOut", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorLaunchTimeout"]                    = {"hipErrorLaunchTimeOut", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"]        = {"hipErrorLaunchOutOfResources", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorLaunchOutOfResources"]             = {"hipErrorLaunchOutOfResources", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_ECC_UNCORRECTABLE"]              = {"hipErrorECCNotCorrectable", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorECCUncorrectable"]                 = {"hipErrorECCNotCorrectable", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED"] = {"hipErrorHostMemoryAlreadyRegistered", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorHostMemoryAlreadyRegistered"]      = {"hipErrorHostMemoryAlreadyRegistered", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED"]     = {"hipErrorHostMemoryNotRegistered", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorHostMemoryNotRegistered"]          = {"hipErrorHostMemoryNotRegistered", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_NO_BINARY_FOR_GPU"]              = {"hipErrorNoBinaryForGpu", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorNoKernelImageForDevice"]           = {"hipErrorNoBinaryForGpu", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_UNSUPPORTED_LIMIT"]              = {"hipErrorUnsupportedLimit", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorUnsupportedLimit"]                 = {"hipErrorUnsupportedLimit", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_INVALID_IMAGE"]                  = {"hipErrorInvalidImage", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorInvalidKernelImage"]               = {"hipErrorInvalidImage", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_PROFILER_DISABLED"]              = {"hipErrorProfilerDisabled", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorProfilerDisabled"]                 = {"hipErrorProfilerDisabled", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_PROFILER_NOT_INITIALIZED"]       = {"hipErrorProfilerNotInitialized", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorProfilerNotInitialized"]           = {"hipErrorProfilerNotInitialized", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_PROFILER_ALREADY_STARTED"]       = {"hipErrorProfilerAlreadyStarted", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorProfilerAlreadyStarted"]           = {"hipErrorProfilerAlreadyStarted", CONV_ERR, API_RUNTIME};

    cuda2hipRename["CUDA_ERROR_PROFILER_ALREADY_STOPPED"]       = {"hipErrorProfilerAlreadyStopped", CONV_ERR, API_DRIVER};
    cuda2hipRename["cudaErrorProfilerAlreadyStopped"]           = {"hipErrorProfilerAlreadyStopped", CONV_ERR, API_RUNTIME};

    ///////////////////////////// CUDA DRIVER API /////////////////////////////
    // Defines
    cuda2hipRename["CU_LAUNCH_PARAM_BUFFER_POINTER"]                              = {"HIP_LAUNCH_PARAM_BUFFER_POINTER", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_LAUNCH_PARAM_BUFFER_SIZE"]                                 = {"HIP_LAUNCH_PARAM_BUFFER_SIZE", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_LAUNCH_PARAM_END"]                                         = {"HIP_LAUNCH_PARAM_END", CONV_DEV, API_DRIVER};

    // Types
    // NOTE: CUdevice might be changed to typedef int in the future.
    cuda2hipRename["CUdevice"]                                  = {"hipDevice_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUdevice_attribute_enum"]                   = {"hipDeviceAttribute_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUdevice_attribute"]                        = {"hipDeviceAttribute_t", CONV_TYPE, API_DRIVER};

    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK"]                   = {"hipDeviceAttributeMaxThreadsPerBlock", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X"]                         = {"hipDeviceAttributeMaxBlockDimX", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y"]                         = {"hipDeviceAttributeMaxBlockDimY", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z"]                         = {"hipDeviceAttributeMaxBlockDimZ", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X"]                          = {"hipDeviceAttributeMaxGridDimX", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y"]                          = {"hipDeviceAttributeMaxGridDimY", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z"]                          = {"hipDeviceAttributeMaxGridDimZ", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK"]             = {"hipDeviceAttributeMaxSharedMemoryPerBlock", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK"]                 = {"hipDeviceAttributeMaxSharedMemoryPerBlock", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY"]                   = {"hipDeviceAttributeTotalConstantMemory", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_WARP_SIZE"]                               = {"hipDeviceAttributeWarpSize", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK"]                 = {"hipDeviceAttributeMaxRegistersPerBlock", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK"]                     = {"hipDeviceAttributeMaxRegistersPerBlock", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_CLOCK_RATE"]                              = {"hipDeviceAttributeClockRate", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE"]                       = {"hipDeviceAttributeMemoryClockRate", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH"]                 = {"hipDeviceAttributeMemoryBusWidth", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_COMPUTE_MODE"]                            = {"hipDeviceAttributeComputeMode", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE"]                           = {"hipDeviceAttributeL2CacheSize", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR"]          = {"hipDeviceAttributeMaxThreadsPerMultiProcessor", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR"]                = {"hipDeviceAttributeComputeCapabilityMajor", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR"]                = {"hipDeviceAttributeComputeCapabilityMinor", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS"]                      = {"hipDeviceAttributeConcurrentKernels", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_PCI_BUS_ID"]                              = {"hipDeviceAttributePciBusId", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID"]                           = {"hipDeviceAttributePciDeviceId", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR"]    = {"hipDeviceAttributeMaxSharedMemoryPerMultiprocessor", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD"]                         = {"hipDeviceAttributeIsMultiGpuBoard", CONV_DEV, API_DRIVER};
    // unsupported yet by HIP
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_PITCH"]                               = {"hipDeviceAttributeMaxPitch", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT"]                       = {"hipDeviceAttributeTextureAlignment", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT"]                      = {"hipDeviceAttributeAsyncEngineCount", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    // Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_GPU_OVERLAP"]                             = {"hipDeviceAttributeAsyncEngineCount", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT"]                    = {"hipDeviceAttributeMultiprocessorCount", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT"]                     = {"hipDeviceAttributeKernelExecTimeout", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_INTEGRATED"]                              = {"hipDeviceAttributeIntegrated", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY"]                     = {"hipDeviceAttributeCanMapHostMemory", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH"]                 = {"hipDeviceAttributeMaxTexture1DWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH"]                 = {"hipDeviceAttributeMaxTexture2DWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT"]                = {"hipDeviceAttributeMaxTexture2DHeight", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH"]                 = {"hipDeviceAttributeMaxTexture3DWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT"]                = {"hipDeviceAttributeMaxTexture3DHeight", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH"]                 = {"hipDeviceAttributeMaxTexture3DDepth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH"]         = {"hipDeviceAttributeMaxTexture2DLayeredWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT"]        = {"hipDeviceAttributeMaxTexture2DLayeredHeight", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS"]        = {"hipDeviceAttributeMaxTexture2DLayeredLayers", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH"]           = {"hipDeviceAttributeMaxTexture2DLayeredWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT"]          = {"hipDeviceAttributeMaxTexture2DLayeredHeight", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES"]       = {"hipDeviceAttributeMaxTexture2DLayeredLayers", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT"]                       = {"hipDeviceAttributeSurfaceAlignment", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_ECC_ENABLED"]                             = {"hipDeviceAttributeEccEnabled", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_TCC_DRIVER"]                              = {"hipDeviceAttributeTccDriver", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING"]                      = {"hipDeviceAttributeUnifiedAddressing", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH"]         = {"hipDeviceAttributeMaxTexture1DLayeredWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS"]        = {"hipDeviceAttributeMaxTexture1DLayeredLayers", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH"]          = {"hipDeviceAttributeMaxTexture2DGatherWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT"]         = {"hipDeviceAttributeMaxTexture2DGatherHeight", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE"]       = {"hipDeviceAttributeMaxTexture3DWidthAlternate", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE"]      = {"hipDeviceAttributeMaxTexture3DHeightAlternate", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE"]       = {"hipDeviceAttributeMaxTexture3DDepthAlternate", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID"]                           = {"hipDeviceAttributePciDomainId", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT"]                 = {"hipDeviceAttributeTexturePitchAlignment", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH"]            = {"hipDeviceAttributeMaxTextureCubemapWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH"]    = {"hipDeviceAttributeMaxTextureCubemapLayeredWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS"]   = {"hipDeviceAttributeMaxTextureCubemapLayeredLayers", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH"]                 = {"hipDeviceAttributeMaxSurface1DWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH"]                 = {"hipDeviceAttributeMaxSurface2DWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT"]                = {"hipDeviceAttributeMaxSurface2DHeight", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH"]                 = {"hipDeviceAttributeMaxSurface3DWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT"]                = {"hipDeviceAttributeMaxSurface3DHeight", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH"]                 = {"hipDeviceAttributeMaxSurface3DDepth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH"]         = {"hipDeviceAttributeMaxSurface1DLayeredWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS"]        = {"hipDeviceAttributeMaxSurface1DLayeredLayers", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH"]         = {"hipDeviceAttributeMaxSurface2DLayeredWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT"]        = {"hipDeviceAttributeMaxSurface2DLayeredHeight", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS"]        = {"hipDeviceAttributeMaxSurface2DLayeredLayers", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH"]            = {"hipDeviceAttributeMaxSurfaceCubemapWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH"]    = {"hipDeviceAttributeMaxSurfaceCubemapLayeredWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS"]   = {"hipDeviceAttributeMaxSurfaceCubemapLayeredLayers", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH"]          = {"hipDeviceAttributeMaxTexture1DLinearWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH"]          = {"hipDeviceAttributeMaxTexture2DLinearWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT"]         = {"hipDeviceAttributeMaxTexture2DLinearHeight", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH"]          = {"hipDeviceAttributeMaxTexture2DLinearPitch", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH"]       = {"hipDeviceAttributeMaxTexture2DMipmappedWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT"]      = {"hipDeviceAttributeMaxTexture2DMipmappedHeight", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH"]       = {"hipDeviceAttributeMaxTexture1DMipmappedWidth", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED"]             = {"hipDeviceAttributeStreamPrioritiesSupported", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED"]               = {"hipDeviceAttributeGlobalL1CacheSupported", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED"]                = {"hipDeviceAttributeLocalL1CacheSupported", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR"]        = {"hipDeviceAttributeMaxRegistersPerMultiprocessor", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY"]                          = {"hipDeviceAttributeManagedMemory", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID"]                = {"hipDeviceAttributeMultiGpuBoardGroupId", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX"]                                     = {"hipDeviceAttributeMax", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    // deprecated, do not use
    // cuda2hipRename["CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER"]                     = {"hipDeviceAttributeCanTex2DGather", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    // unsupported yet by HIP [CUDA 8.0.44]
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED"]            = {"hipDeviceAttributeHostNativeAtomicSupported", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO"]   = {"hipDeviceAttributeSingleToDoublePrecisionPerfRatio", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS"]                  = {"hipDeviceAttributePageableMemoryAccess", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS"]               = {"hipDeviceAttributeConcurrentManagedAccess", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED"]            = {"hipDeviceAttributeComputePreemptionSupported", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM"] = {"hipDeviceAttributeCanUseHostPointerForRegisteredMem", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};

    cuda2hipRename["CUdevprop_st"]                              = {"hipDeviceProp_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUdevprop"]                                 = {"hipDeviceProp_t", CONV_TYPE, API_DRIVER};

    // TODO: Analogues enum is needed in HIP. Couldn't map enum to struct hipPointerAttribute_t.
    // TODO: Do for Pointer Attributes the same as for Device Attributes.
    // cuda2hipRename["CUpointer_attribute_enum"]               = {"hipPointerAttribute_t", CONV_TYPE, API_DRIVER};
    // cuda2hipRename["CUpointer_attribute"]                    = {"hipPointerAttribute_t", CONV_TYPE, API_DRIVER};

    cuda2hipRename["CUfunction"]                                = {"hipFunction_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUfunc_st"]                                 = {"hipFunction_t *", CONV_TYPE, API_DRIVER};

    // unsupported yet by HIP
    cuda2hipRename["CUfunction_attribute_enum"]                 = {"hipFuncAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CUfunction_attribute"]                      = {"hipFuncAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};

    cuda2hipRename["CUfunc_cache_enum"]                         = {"hipFuncCache", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUfunc_cache"]                              = {"hipFuncCache", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CU_FUNC_CACHE_PREFER_NONE"]                 = {"hipFuncCachePreferNone", CONV_CACHE, API_DRIVER};
    cuda2hipRename["CU_FUNC_CACHE_PREFER_SHARED"]               = {"hipFuncCachePreferShared", CONV_CACHE, API_DRIVER};
    cuda2hipRename["CU_FUNC_CACHE_PREFER_L1"]                   = {"hipFuncCachePreferL1", CONV_CACHE, API_DRIVER};
    cuda2hipRename["CU_FUNC_CACHE_PREFER_EQUAL"]                = {"hipFuncCachePreferEqual", CONV_CACHE, API_DRIVER};

    cuda2hipRename["CUsharedconfig_enum"]                       = {"hipSharedMemConfig", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUsharedconfig"]                            = {"hipSharedMemConfig", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE"]    = {"hipSharedMemBankSizeDefault", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE"]  = {"hipSharedMemBankSizeFourByte", CONV_DEV, API_DRIVER};
    cuda2hipRename["CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE"] = {"hipSharedMemBankSizeEightByte", CONV_DEV, API_DRIVER};

    cuda2hipRename["CUcontext"]                                 = {"hipCtx_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUctx_st"]                                  = {"hipCtx_t *", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUmodule"]                                  = {"hipModule_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUmod_st"]                                  = {"hipModule_t *", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUstream"]                                  = {"hipStream_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUstream_st"]                               = {"hipStream_t *", CONV_TYPE, API_DRIVER};
    // Stream Flags
    cuda2hipRename["CU_STREAM_DEFAULT"]                         = {"hipStreamDefault", CONV_STREAM, API_DRIVER};
    cuda2hipRename["CU_STREAM_NON_BLOCKING"]                    = {"hipStreamNonBlocking", CONV_STREAM, API_DRIVER};

    // Init
    cuda2hipRename["cuInit"]                                    = {"hipInit", CONV_DRIVER, API_DRIVER};

    // Driver
    cuda2hipRename["cuDriverGetVersion"]                        = {"hipDriverGetVersion", CONV_DRIVER, API_DRIVER};

    // Context
    cuda2hipRename["cuCtxCreate_v2"]                            = {"hipCtxCreate", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxDestroy_v2"]                           = {"hipCtxDestroy", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxPopCurrent_v2"]                        = {"hipCtxPopCurrent", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxPushCurrent_v2"]                       = {"hipCtxPushCurrent", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxSetCurrent"]                           = {"hipCtxSetCurrent", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetCurrent"]                           = {"hipCtxGetCurrent", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetDevice"]                            = {"hipCtxGetDevice", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetApiVersion"]                        = {"hipCtxGetApiVersion", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetCacheConfig"]                       = {"hipCtxGetCacheConfig", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxSetCacheConfig"]                       = {"hipCtxSetCacheConfig", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxSetSharedMemConfig"]                   = {"hipCtxSetSharedMemConfig", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetSharedMemConfig"]                   = {"hipCtxGetSharedMemConfig", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxSynchronize"]                          = {"hipCtxSynchronize", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetFlags"]                             = {"hipCtxGetFlags", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxEnablePeerAccess"]                     = {"hipCtxEnablePeerAccess", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxDisablePeerAccess"]                    = {"hipCtxDisablePeerAccess", CONV_CONTEXT, API_DRIVER};
    // unsupported yet by HIP
    cuda2hipRename["cuCtxSetLimit"]                             = {"hipCtxSetLimit", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuCtxGetLimit"]                             = {"hipCtxGetLimit", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED};

    // Device
    cuda2hipRename["cuDeviceGet"]                               = {"hipGetDevice", CONV_DEV, API_DRIVER};
    cuda2hipRename["cuDeviceGetName"]                           = {"hipDeviceGetName", CONV_DEV, API_DRIVER};
    cuda2hipRename["cuDeviceGetCount"]                          = {"hipGetDeviceCount", CONV_DEV, API_DRIVER};
    cuda2hipRename["cuDeviceGetAttribute"]                      = {"hipDeviceGetAttribute", CONV_DEV, API_DRIVER};
    cuda2hipRename["cuDeviceGetProperties"]                     = {"hipGetDeviceProperties", CONV_DEV, API_DRIVER};
    cuda2hipRename["cuDeviceGetPCIBusId"]                       = {"hipDeviceGetPCIBusId", CONV_DEV, API_DRIVER};
    // unsupported yet by HIP
    cuda2hipRename["cuDeviceGetByPCIBusId"]                     = {"hipDeviceGetByPCIBusId", CONV_DEV, API_DRIVER, HIP_UNSUPPORTED};

    cuda2hipRename["cuDeviceTotalMem_v2"]                       = {"hipDeviceTotalMem", CONV_DEV, API_DRIVER};
    cuda2hipRename["cuDeviceComputeCapability"]                 = {"hipDeviceComputeCapability", CONV_DEV, API_DRIVER};
    cuda2hipRename["cuDeviceCanAccessPeer"]                     = {"hipDeviceCanAccessPeer", CONV_DEV, API_DRIVER};

    // Events
    cuda2hipRename["CUevent"]                                   = {"hipEvent_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUevent_st"]                                = {"hipEvent_t *", CONV_TYPE, API_DRIVER};
    // Event Flags
    cuda2hipRename["CU_EVENT_DEFAULT"]                          = {"hipEventDefault", CONV_EVENT, API_DRIVER};
    cuda2hipRename["CU_EVENT_BLOCKING_SYNC"]                    = {"hipEventBlockingSync", CONV_EVENT, API_DRIVER};
    cuda2hipRename["CU_EVENT_DISABLE_TIMING"]                   = {"hipEventDisableTiming", CONV_EVENT, API_DRIVER};
    cuda2hipRename["CU_EVENT_INTERPROCESS"]                     = {"hipEventInterprocess", CONV_EVENT, API_DRIVER};

    cuda2hipRename["cuEventCreate"]                             = {"hipEventCreate", CONV_EVENT, API_DRIVER};
    cuda2hipRename["cuEventDestroy_v2"]                         = {"hipEventDestroy", CONV_EVENT, API_DRIVER};
    cuda2hipRename["cuEventElapsedTime"]                        = {"hipEventElapsedTime", CONV_EVENT, API_DRIVER};
    cuda2hipRename["cuEventQuery"]                              = {"hipEventQuery", CONV_EVENT, API_DRIVER};
    cuda2hipRename["cuEventRecord"]                             = {"hipEventRecord", CONV_EVENT, API_DRIVER};
    cuda2hipRename["cuEventSynchronize"]                        = {"hipEventSynchronize", CONV_EVENT, API_DRIVER};

    // Module
    cuda2hipRename["cuModuleGetFunction"]                       = {"hipModuleGetFunction", CONV_MODULE, API_DRIVER};
    cuda2hipRename["cuModuleGetGlobal_v2"]                      = {"hipModuleGetGlobal", CONV_MODULE, API_DRIVER};
    cuda2hipRename["cuModuleLoad"]                              = {"hipModuleLoad", CONV_MODULE, API_DRIVER};
    cuda2hipRename["cuModuleLoadData"]                          = {"hipModuleLoadData", CONV_MODULE, API_DRIVER};
    // unsupported yet by HIP
    cuda2hipRename["cuModuleLoadDataEx"]                        = {"hipModuleLoadDataEx", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuModuleLoadFatBinary"]                     = {"hipModuleLoadFatBinary", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};

    cuda2hipRename["cuModuleUnload"]                            = {"hipModuleUnload", CONV_MODULE, API_DRIVER};
    cuda2hipRename["cuLaunchKernel"]                            = {"hipModuleLaunchKernel", CONV_MODULE, API_DRIVER};

    // Streams
    // unsupported yet by HIP
    cuda2hipRename["cuStreamAddCallback"]                       = {"hipStreamAddCallback", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};

    cuda2hipRename["cuStreamCreate"]                            = {"hipStreamCreate", CONV_STREAM, API_DRIVER};
    cuda2hipRename["cuStreamDestroy_v2"]                        = {"hipStreamDestroy", CONV_STREAM, API_DRIVER};
    cuda2hipRename["cuStreamQuery"]                             = {"hipStreamQuery", CONV_STREAM, API_DRIVER};
    cuda2hipRename["cuStreamSynchronize"]                       = {"hipStreamSynchronize", CONV_STREAM, API_DRIVER};
    cuda2hipRename["cuStreamWaitEvent"]                         = {"hipStreamWaitEvent", CONV_STREAM, API_DRIVER};

    // Memory management
    cuda2hipRename["cuMemAlloc_v2"]                             = {"hipMalloc", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemFree_v2"]                              = {"hipFree", CONV_MEM, API_DRIVER};

    cuda2hipRename["cuMemHostAlloc"]                            = {"hipHostMalloc", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemFreeHost"]                             = {"hipHostFree", CONV_MEM, API_DRIVER};

    cuda2hipRename["cuMemcpyDtoD_v2"]                           = {"hipMemcpyDtoD", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemcpyDtoDAsync_v2"]                      = {"hipMemcpyDtoDAsync", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemcpyDtoH_v2"]                           = {"hipMemcpyDtoH", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemcpyDtoHAsync_v2"]                      = {"hipMemcpyDtoHAsync", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemcpyHtoD_v2"]                           = {"hipMemcpyHtoD", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemcpyHtoDAsync_v2"]                      = {"hipMemcpyHtoDAsync", CONV_MEM, API_DRIVER};

    // unsupported yet by HIP
    cuda2hipRename["cuMemsetD8_v2"]                             = {"hipMemsetD8", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD8Async"]                           = {"hipMemsetD8Async", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD2D8_v2"]                           = {"hipMemsetD2D8", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD2D8Async"]                         = {"hipMemsetD2D8Async", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD16_v2"]                            = {"hipMemsetD16", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD16Async"]                          = {"hipMemsetD16Async", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD2D16_v2"]                          = {"hipMemsetD2D16", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};
     cuda2hipRename["cuMemsetD2D16Async"]                       = {"hipMemsetD2D16Async", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};

    cuda2hipRename["cuMemsetD32_v2"]                            = {"hipMemset", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemsetD32Async"]                          = {"hipMemsetAsync", CONV_MEM, API_DRIVER};
    // unsupported yet by HIP
    cuda2hipRename["cuMemsetD2D32_v2"]                          = {"hipMemsetD2D32", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD2D32Async"]                        = {"hipMemsetD2D32Async", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};

    cuda2hipRename["cuMemGetInfo_v2"]                           = {"hipMemGetInfo", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemHostRegister_v2"]                      = {"hipHostRegister", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemHostUnregister"]                       = {"hipHostUnregister", CONV_MEM, API_DRIVER};

    // Profiler
    // unsupported yet by HIP
    cuda2hipRename["cuProfilerInitialize"]                      = {"hipProfilerInitialize", CONV_OTHER, API_DRIVER, HIP_UNSUPPORTED};

    cuda2hipRename["cuProfilerStart"]                           = {"hipProfilerStart", CONV_OTHER, API_DRIVER};
    cuda2hipRename["cuProfilerStop"]                            = {"hipProfilerStop", CONV_OTHER, API_DRIVER};

    /////////////////////////////// CUDA RT API ///////////////////////////////
    // Data types
    // unsupported yet by HIP [CUDA 8.0.44]
    cuda2hipRename["cudaDataType_t"]              = {"hipDataType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDataType"]                = {"hipDataType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_R_16F"]                  = {"hipR16F", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_C_16F"]                  = {"hipC16F", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_R_32F"]                  = {"hipR32F", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_C_32F"]                  = {"hipC32F", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_R_64F"]                  = {"hipR64F", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_C_64F"]                  = {"hipC64F", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_R_8I"]                   = {"hipR8I", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_C_8I"]                   = {"hipC8I", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_R_8U"]                   = {"hipR8U", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_C_8U"]                   = {"hipC8U", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_R_32I"]                  = {"hipR32I", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_C_32I"]                  = {"hipC32I", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_R_32U"]                  = {"hipR32U", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_C_32U"]                  = {"hipC32U", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};

    // Library property types
    // IMPORTANT: no cuda prefix
    // TO_DO: new matcher is needed
    // unsupported yet by HIP [CUDA 8.0.44]
    cuda2hipRename["libraryPropertyType_t"]       = {"hipLibraryPropertyType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["libraryPropertyType"]         = {"hipLibraryPropertyType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["MAJOR_VERSION"]               = {"hipLibraryMajorVersion", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["MINOR_VERSION"]               = {"hipLibraryMinorVersion", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["PATCH_LEVEL"]                 = {"hipLibraryPatchVersion", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};

    // Error API
    cuda2hipRename["cudaGetLastError"]            = {"hipGetLastError", CONV_ERR, API_RUNTIME};
    cuda2hipRename["cudaPeekAtLastError"]         = {"hipPeekAtLastError", CONV_ERR, API_RUNTIME};
    cuda2hipRename["cudaGetErrorName"]            = {"hipGetErrorName", CONV_ERR, API_RUNTIME};
    cuda2hipRename["cudaGetErrorString"]          = {"hipGetErrorString", CONV_ERR, API_RUNTIME};

    // Memcpy
    cuda2hipRename["cudaMemcpy"]                  = {"hipMemcpy", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyToArray"]           = {"hipMemcpyToArray", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyToSymbol"]          = {"hipMemcpyToSymbol", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyToSymbolAsync"]     = {"hipMemcpyToSymbolAsync", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemset"]                  = {"hipMemset", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemsetAsync"]             = {"hipMemsetAsync", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyAsync"]             = {"hipMemcpyAsync", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemGetInfo"]              = {"hipMemGetInfo", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpy2D"]                = {"hipMemcpy2D", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpy2DToArray"]         = {"hipMemcpy2DToArray", CONV_MEM, API_RUNTIME};


    // Memcpy kind
    cuda2hipRename["cudaMemcpyKind"]              = {"hipMemcpyKind", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyHostToHost"]        = {"hipMemcpyHostToHost", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyHostToDevice"]      = {"hipMemcpyHostToDevice", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyDeviceToHost"]      = {"hipMemcpyDeviceToHost", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyDeviceToDevice"]    = {"hipMemcpyDeviceToDevice", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyDefault"]           = {"hipMemcpyDefault", CONV_MEM, API_RUNTIME};

    // Memory management
    cuda2hipRename["cudaMalloc"]                  = {"hipMalloc", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMallocHost"]              = {"hipHostMalloc", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMallocArray"]             = {"hipMallocArray", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaFree"]                    = {"hipFree", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaFreeHost"]                = {"hipHostFree", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaFreeArray"]               = {"hipFreeArray", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaHostRegister"]            = {"hipHostRegister", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaHostUnregister"]          = {"hipHostUnregister", CONV_MEM, API_RUNTIME};
    // hipHostAlloc deprecated - use hipHostMalloc instead
    cuda2hipRename["cudaHostAlloc"]               = {"hipHostMalloc", CONV_MEM, API_RUNTIME};

    // Memory types
    cuda2hipRename["cudaMemoryType"]              = {"hipMemoryType", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemoryTypeHost"]          = {"hipMemoryTypeHost", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemoryTypeDevice"]        = {"hipMemoryTypeDevice", CONV_MEM, API_RUNTIME};

    // Host Malloc Flags
    cuda2hipRename["cudaHostAllocDefault"]        = {"hipHostMallocDefault", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaHostAllocPortable"]       = {"hipHostMallocPortable", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaHostAllocMapped"]         = {"hipHostMallocMapped", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaHostAllocWriteCombined"]  = {"hipHostMallocWriteCombined", CONV_MEM, API_RUNTIME};

    // Host Register Flags
    cuda2hipRename["cudaHostGetFlags"]            = {"hipHostGetFlags", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaHostRegisterDefault"]     = {"hipHostRegisterDefault", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaHostRegisterPortable"]    = {"hipHostRegisterPortable", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaHostRegisterMapped"]      = {"hipHostRegisterMapped", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaHostRegisterIoMemory"]    = {"hipHostRegisterIoMemory", CONV_MEM, API_RUNTIME};

    // Coordinate Indexing and Dimensions
    cuda2hipRename["threadIdx.x"] = {"hipThreadIdx_x", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["threadIdx.y"] = {"hipThreadIdx_y", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["threadIdx.z"] = {"hipThreadIdx_z", CONV_COORD_FUNC, API_RUNTIME};

    cuda2hipRename["blockIdx.x"]  = {"hipBlockIdx_x", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["blockIdx.y"]  = {"hipBlockIdx_y", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["blockIdx.z"]  = {"hipBlockIdx_z", CONV_COORD_FUNC, API_RUNTIME};

    cuda2hipRename["blockDim.x"]  = {"hipBlockDim_x", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["blockDim.y"]  = {"hipBlockDim_y", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["blockDim.z"]  = {"hipBlockDim_z", CONV_COORD_FUNC, API_RUNTIME};

    cuda2hipRename["gridDim.x"]   = {"hipGridDim_x", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["gridDim.y"]   = {"hipGridDim_y", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["gridDim.z"]   = {"hipGridDim_z", CONV_COORD_FUNC, API_RUNTIME};

    cuda2hipRename["blockIdx.x"]  = {"hipBlockIdx_x", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["blockIdx.y"]  = {"hipBlockIdx_y", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["blockIdx.z"]  = {"hipBlockIdx_z", CONV_COORD_FUNC, API_RUNTIME};

    cuda2hipRename["blockDim.x"]  = {"hipBlockDim_x", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["blockDim.y"]  = {"hipBlockDim_y", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["blockDim.z"]  = {"hipBlockDim_z", CONV_COORD_FUNC, API_RUNTIME};

    cuda2hipRename["gridDim.x"]   = {"hipGridDim_x", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["gridDim.y"]   = {"hipGridDim_y", CONV_COORD_FUNC, API_RUNTIME};
    cuda2hipRename["gridDim.z"]   = {"hipGridDim_z", CONV_COORD_FUNC, API_RUNTIME};

    cuda2hipRename["warpSize"]    = {"hipWarpSize", CONV_SPECIAL_FUNC, API_RUNTIME};

    // Events
    cuda2hipRename["cudaEvent_t"]              = {"hipEvent_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaEventCreate"]          = {"hipEventCreate", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventCreateWithFlags"] = {"hipEventCreateWithFlags", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventDestroy"]         = {"hipEventDestroy", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventRecord"]          = {"hipEventRecord", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventElapsedTime"]     = {"hipEventElapsedTime", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventSynchronize"]     = {"hipEventSynchronize", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventQuery"]           = {"hipEventQuery", CONV_EVENT, API_RUNTIME};
    // Event Flags
    cuda2hipRename["cudaEventDefault"]         = {"hipEventDefault", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventBlockingSync"]    = {"hipEventBlockingSync", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventDisableTiming"]   = {"hipEventDisableTiming", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventInterprocess"]    = {"hipEventInterprocess", CONV_EVENT, API_RUNTIME};

    // Streams
    cuda2hipRename["cudaStream_t"]              = {"hipStream_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaStreamCreate"]          = {"hipStreamCreate", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamCreateWithFlags"] = {"hipStreamCreateWithFlags", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamDestroy"]         = {"hipStreamDestroy", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamWaitEvent"]       = {"hipStreamWaitEvent", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamSynchronize"]     = {"hipStreamSynchronize", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamGetFlags"]        = {"hipStreamGetFlags", CONV_STREAM, API_RUNTIME};
    // Stream Flags
    cuda2hipRename["cudaStreamDefault"]         = {"hipStreamDefault", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamNonBlocking"]     = {"hipStreamNonBlocking", CONV_STREAM, API_RUNTIME};

    // Other synchronization
    cuda2hipRename["cudaDeviceSynchronize"]     = {"hipDeviceSynchronize", CONV_DEV, API_RUNTIME};
    // translate deprecated cudaThreadSynchronize
    cuda2hipRename["cudaThreadSynchronize"]     = {"hipDeviceSynchronize", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDeviceReset"]           = {"hipDeviceReset", CONV_DEV, API_RUNTIME};
    // translate deprecated cudaThreadExit
    cuda2hipRename["cudaThreadExit"]            = {"hipDeviceReset", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaSetDevice"]             = {"hipSetDevice", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaGetDevice"]             = {"hipGetDevice", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaGetDeviceCount"]        = {"hipGetDeviceCount", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaChooseDevice"]          = {"hipChooseDevice", CONV_DEV, API_RUNTIME};

    // Attributes
    cuda2hipRename["cudaDeviceAttr"]            = {"hipDeviceAttribute_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaDeviceGetAttribute"]    = {"hipDeviceGetAttribute", CONV_DEV, API_RUNTIME};

    cuda2hipRename["cudaDevAttrMaxThreadsPerBlock"]                = {"hipDeviceAttributeMaxThreadsPerBlock", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMaxBlockDimX"]                      = {"hipDeviceAttributeMaxBlockDimX", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMaxBlockDimY"]                      = {"hipDeviceAttributeMaxBlockDimY", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMaxBlockDimZ"]                      = {"hipDeviceAttributeMaxBlockDimZ", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMaxGridDimX"]                       = {"hipDeviceAttributeMaxGridDimX", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMaxGridDimY"]                       = {"hipDeviceAttributeMaxGridDimY", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMaxGridDimZ"]                       = {"hipDeviceAttributeMaxGridDimZ", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMaxSharedMemoryPerBlock"]           = {"hipDeviceAttributeMaxSharedMemoryPerBlock", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrTotalConstantMemory"]               = {"hipDeviceAttributeTotalConstantMemory", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrWarpSize"]                          = {"hipDeviceAttributeWarpSize", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMaxRegistersPerBlock"]              = {"hipDeviceAttributeMaxRegistersPerBlock", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrClockRate"]                         = {"hipDeviceAttributeClockRate", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMemoryClockRate"]                   = {"hipDeviceAttributeMemoryClockRate", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrGlobalMemoryBusWidth"]              = {"hipDeviceAttributeMemoryBusWidth", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMultiProcessorCount"]               = {"hipDeviceAttributeMultiprocessorCount", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrComputeMode"]                       = {"hipDeviceAttributeComputeMode", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrL2CacheSize"]                       = {"hipDeviceAttributeL2CacheSize", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMaxThreadsPerMultiProcessor"]       = {"hipDeviceAttributeMaxThreadsPerMultiProcessor", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrComputeCapabilityMajor"]            = {"hipDeviceAttributeComputeCapabilityMajor", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrComputeCapabilityMinor"]            = {"hipDeviceAttributeComputeCapabilityMinor", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrConcurrentKernels"]                 = {"hipDeviceAttributeConcurrentKernels", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrPciBusId"]                          = {"hipDeviceAttributePciBusId", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrPciDeviceId"]                       = {"hipDeviceAttributePciDeviceId", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrMaxSharedMemoryPerMultiprocessor"]  = {"hipDeviceAttributeMaxSharedMemoryPerMultiprocessor", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDevAttrIsMultiGpuBoard"]                   = {"hipDeviceAttributeIsMultiGpuBoard", CONV_DEV, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaDevAttrMaxPitch"]                          = {"hipDeviceAttributeMaxPitch", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrTextureAlignment"]                  = {"hipDeviceAttributeTextureAlignment", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    // Is not deprecated as CUDA Driver's API analogue CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
    cuda2hipRename["cudaDevAttrGpuOverlap"]                        = {"hipDeviceAttributeGpuOverlap", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrKernelExecTimeout"]                 = {"hipDeviceAttributeKernelExecTimeout", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrIntegrated"]                        = {"hipDeviceAttributeIntegrated", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrCanMapHostMemory"]                  = {"hipDeviceAttributeCanMapHostMemory", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture1DWidth"]                 = {"hipDeviceAttributeMaxTexture1DWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DWidth"]                 = {"hipDeviceAttributeMaxTexture2DWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DHeight"]                = {"hipDeviceAttributeMaxTexture2DHeight", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture3DWidth"]                 = {"hipDeviceAttributeMaxTexture3DWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture3DHeight"]                = {"hipDeviceAttributeMaxTexture3DHeight", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture3DDepth"]                 = {"hipDeviceAttributeMaxTexture3DDepth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DLayeredWidth"]          = {"hipDeviceAttributeMaxTexture2DLayeredWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DLayeredHeight"]         = {"hipDeviceAttributeMaxTexture2DLayeredHeight", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DLayeredLayers"]         = {"hipDeviceAttributeMaxTexture2DLayeredLayers", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrSurfaceAlignment"]                  = {"hipDeviceAttributeSurfaceAlignment", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrEccEnabled"]                        = {"hipDeviceAttributeEccEnabled", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrTccDriver"]                         = {"hipDeviceAttributeTccDriver", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrUnifiedAddressing"]                 = {"hipDeviceAttributeUnifiedAddressing", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture1DLayeredWidth"]          = {"hipDeviceAttributeMaxTexture1DLayeredWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture1DLayeredLayers"]         = {"hipDeviceAttributeMaxTexture1DLayeredLayers", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DGatherWidth"]           = {"hipDeviceAttributeMaxTexture2DGatherWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DGatherHeight"]          = {"hipDeviceAttributeMaxTexture2DGatherHeight", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture3DWidthAlt"]              = {"hipDeviceAttributeMaxTexture3DWidthAlternate", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture3DHeightAlt"]             = {"hipDeviceAttributeMaxTexture3DHeightAlternate", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture3DDepthAlt"]              = {"hipDeviceAttributeMaxTexture3DDepthAlternate", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrPciDomainId"]                       = {"hipDeviceAttributePciDomainId", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrTexturePitchAlignment"]             = {"hipDeviceAttributeTexturePitchAlignment", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTextureCubemapWidth"]            = {"hipDeviceAttributeMaxTextureCubemapWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTextureCubemapLayeredWidth"]     = {"hipDeviceAttributeMaxTextureCubemapLayeredWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTextureCubemapLayeredLayers"]    = {"hipDeviceAttributeMaxTextureCubemapLayeredLayers", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurface1DWidth"]                 = {"hipDeviceAttributeMaxSurface1DWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurface2DWidth"]                 = {"hipDeviceAttributeMaxSurface2DWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurface2DHeight"]                = {"hipDeviceAttributeMaxSurface2DHeight", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurface3DWidth"]                 = {"hipDeviceAttributeMaxSurface3DWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurface3DHeight"]                = {"hipDeviceAttributeMaxSurface3DHeight", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurface3DDepth"]                 = {"hipDeviceAttributeMaxSurface3DDepth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurface1DLayeredWidth"]          = {"hipDeviceAttributeMaxSurface1DLayeredWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurface1DLayeredLayers"]         = {"hipDeviceAttributeMaxSurface1DLayeredLayers", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurface2DLayeredWidth"]          = {"hipDeviceAttributeMaxSurface2DLayeredWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurface2DLayeredHeight"]         = {"hipDeviceAttributeMaxSurface2DLayeredHeight", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurface2DLayeredLayers"]         = {"hipDeviceAttributeMaxSurface2DLayeredLayers", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurfaceCubemapWidth"]            = {"hipDeviceAttributeMaxSurfaceCubemapWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurfaceCubemapLayeredWidth"]     = {"hipDeviceAttributeMaxSurfaceCubemapLayeredWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxSurfaceCubemapLayeredLayers"]    = {"hipDeviceAttributeMaxSurfaceCubemapLayeredLayers", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture1DLinearWidth"]           = {"hipDeviceAttributeMaxTexture1DLinearWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DLinearWidth"]           = {"hipDeviceAttributeMaxTexture2DLinearWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DLinearHeight"]          = {"hipDeviceAttributeMaxTexture2DLinearHeight", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DLinearPitch"]           = {"hipDeviceAttributeMaxTexture2DLinearPitch", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DMipmappedWidth"]        = {"hipDeviceAttributeMaxTexture2DMipmappedWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture2DMipmappedHeight"]       = {"hipDeviceAttributeMaxTexture2DMipmappedHeight", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxTexture1DMipmappedWidth"]        = {"hipDeviceAttributeMaxTexture1DMipmappedWidth", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrStreamPrioritiesSupported"]         = {"hipDeviceAttributeStreamPrioritiesSupported", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrGlobalL1CacheSupported"]            = {"hipDeviceAttributeGlobalL1CacheSupported", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrLocalL1CacheSupported"]             = {"hipDeviceAttributeLocalL1CacheSupported", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMaxRegistersPerMultiprocessor"]     = {"hipDeviceAttributeMaxRegistersPerMultiprocessor", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrManagedMemory"]                     = {"hipDeviceAttributeManagedMemory", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrMultiGpuBoardGroupID"]              = {"hipDeviceAttributeMultiGpuBoardGroupID", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    // unsupported yet by HIP [CUDA 8.0.44]
    cuda2hipRename["cudaDevAttrHostNativeAtomicSupported"]         = {"hipDeviceAttributeHostNativeAtomicSupported", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrSingleToDoublePrecisionPerfRatio"]  = {"hipDeviceAttributeSingleToDoublePrecisionPerfRatio", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrPageableMemoryAccess"]              = {"hipDeviceAttributePageableMemoryAccess", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrConcurrentManagedAccess"]           = {"hipDeviceAttributeConcurrentManagedAccess", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrComputePreemptionSupported"]        = {"hipDeviceAttributeComputePreemptionSupported", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDevAttrCanUseHostPointerForRegisteredMem"] = {"hipDeviceAttributeCanUseHostPointerForRegisteredMem", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};

    // Pointer Attributes
    cuda2hipRename["cudaPointerAttributes"]      = {"hipPointerAttribute_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaPointerGetAttributes"]   = {"hipPointerGetAttributes", CONV_MEM, API_RUNTIME};

    cuda2hipRename["cudaHostGetDevicePointer"]   = {"hipHostGetDevicePointer", CONV_MEM, API_RUNTIME};

    // Device
    cuda2hipRename["cudaDeviceProp"]            = {"hipDeviceProp_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaGetDeviceProperties"]   = {"hipGetDeviceProperties", CONV_DEV, API_RUNTIME};

    // Device Flags
    cuda2hipRename["cudaSetDeviceFlags"]               = {"hipSetDeviceFlags", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDeviceScheduleAuto"]           = {"hipDeviceScheduleAuto", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDeviceScheduleSpin"]           = {"hipDeviceScheduleSpin", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDeviceScheduleYield"]          = {"hipDeviceScheduleYield", CONV_DEV, API_RUNTIME};
    // deprecated as of CUDA 4.0 and replaced with cudaDeviceScheduleBlockingSync
    cuda2hipRename["cudaDeviceBlockingSync"]           = {"hipDeviceBlockingSync", CONV_DEV, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaDeviceScheduleBlockingSync"]   = {"hipDeviceScheduleBlockingSync", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDeviceScheduleMask"]           = {"hipDeviceScheduleMask", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};

    cuda2hipRename["cudaDeviceMapHost"]                = {"hipDeviceMapHost", CONV_DEV, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaDeviceLmemResizeToMax"]        = {"hipDeviceLmemResizeToMax", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDeviceMask"]                   = {"hipDeviceMask", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};

    // Cache config
    cuda2hipRename["cudaDeviceSetCacheConfig"]  = {"hipDeviceSetCacheConfig", CONV_CACHE, API_RUNTIME};
    // translate deprecated
    cuda2hipRename["cudaThreadSetCacheConfig"]  = {"hipDeviceSetCacheConfig", CONV_CACHE, API_RUNTIME};
    cuda2hipRename["cudaDeviceGetCacheConfig"]  = {"hipDeviceGetCacheConfig", CONV_CACHE, API_RUNTIME};
    // translate deprecated
    cuda2hipRename["cudaThreadGetCacheConfig"]  = {"hipDeviceGetCacheConfig", CONV_CACHE, API_RUNTIME};
    cuda2hipRename["cudaFuncCache"]             = {"hipFuncCache", CONV_CACHE, API_RUNTIME};
    cuda2hipRename["cudaFuncCachePreferNone"]   = {"hipFuncCachePreferNone", CONV_CACHE, API_RUNTIME};
    cuda2hipRename["cudaFuncCachePreferShared"] = {"hipFuncCachePreferShared", CONV_CACHE, API_RUNTIME};
    cuda2hipRename["cudaFuncCachePreferL1"]     = {"hipFuncCachePreferL1", CONV_CACHE, API_RUNTIME};
    cuda2hipRename["cudaFuncCachePreferEqual"]  = {"hipFuncCachePreferEqual", CONV_CACHE, API_RUNTIME};
    cuda2hipRename["cudaFuncSetCacheConfig"]    = {"hipFuncSetCacheConfig", CONV_CACHE, API_RUNTIME};

    // Driver/Runtime
    cuda2hipRename["cudaDriverGetVersion"]      = {"hipDriverGetVersion", CONV_DRIVER, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaRuntimeGetVersion"]     = {"hipRuntimeGetVersion", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};

    // Occupancy
    // unsupported yet by HIP
    cuda2hipRename["cudaOccupancyMaxPotentialBlockSize"]                      = {"hipOccupancyMaxPotentialBlockSize", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cudaOccupancyMaxPotentialBlockSizeWithFlags"]             = {"hipOccupancyMaxPotentialBlockSizeWithFlags", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cudaOccupancyMaxActiveBlocksPerMultiprocessor"]           = {"hipOccupancyMaxActiveBlocksPerMultiprocessor", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"]  = {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cudaOccupancyMaxPotentialBlockSizeVariableSMem"]          = {"hipOccupancyMaxPotentialBlockSizeVariableSMem", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags"] = {"hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED};

    // Peer2Peer
    cuda2hipRename["cudaDeviceCanAccessPeer"]        = {"hipDeviceCanAccessPeer", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDeviceDisablePeerAccess"]    = {"hipDeviceDisablePeerAccess", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDeviceEnablePeerAccess"]     = {"hipDeviceEnablePeerAccess", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaMemcpyPeerAsync"]            = {"hipMemcpyPeerAsync", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyPeer"]                 = {"hipMemcpyPeer", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaIpcMemLazyEnablePeerAccess"] = {"hipIpcMemLazyEnablePeerAccess", CONV_ERR, API_RUNTIME};

    // Shared memory
    cuda2hipRename["cudaDeviceSetSharedMemConfig"]   = {"hipDeviceSetSharedMemConfig", CONV_DEV, API_RUNTIME};
    // translate deprecated
    cuda2hipRename["cudaThreadSetSharedMemConfig"]   = {"hipDeviceSetSharedMemConfig", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaDeviceGetSharedMemConfig"]   = {"hipDeviceGetSharedMemConfig", CONV_DEV, API_RUNTIME};
    // translate deprecated
    cuda2hipRename["cudaThreadGetSharedMemConfig"]   = {"hipDeviceGetSharedMemConfig", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaSharedMemConfig"]            = {"hipSharedMemConfig", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaSharedMemBankSizeDefault"]   = {"hipSharedMemBankSizeDefault", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaSharedMemBankSizeFourByte"]  = {"hipSharedMemBankSizeFourByte", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaSharedMemBankSizeEightByte"] = {"hipSharedMemBankSizeEightByte", CONV_DEV, API_RUNTIME};

    // Limits
    cuda2hipRename["cudaLimit"]                             = {"hipLimit_t", CONV_DEV, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaLimitStackSize"]                    = {"hipLimitStackSize", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaLimitPrintfFifoSize"]               = {"hipLimitPrintfFifoSize", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};

    cuda2hipRename["cudaLimitMallocHeapSize"]               = {"hipLimitMallocHeapSize", CONV_DEV, API_RUNTIME};

    // unsupported yet by HIP
    cuda2hipRename["cudaLimitDevRuntimeSyncDepth"]          = {"hipLimitPrintfFifoSize", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaLimitDevRuntimePendingLaunchCount"] = {"hipLimitMallocHeapSize", CONV_DEV, API_RUNTIME, HIP_UNSUPPORTED};

    cuda2hipRename["cudaDeviceGetLimit"]                    = {"hipDeviceGetLimit", CONV_DEV, API_RUNTIME};

    // Profiler
    // unsupported yet by HIP
    cuda2hipRename["cudaProfilerInitialize"]                = {"hipProfilerInitialize", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED};

    cuda2hipRename["cudaProfilerStart"]                     = {"hipProfilerStart", CONV_OTHER, API_RUNTIME};
    cuda2hipRename["cudaProfilerStop"]                      = {"hipProfilerStop", CONV_OTHER, API_RUNTIME};
    cuda2hipRename["cudaFilterModePoint"]                   = {"hipFilterModePoint", CONV_TEX, API_RUNTIME};

    cuda2hipRename["cudaReadModeElementType"]               = {"hipReadModeElementType", CONV_TEX, API_RUNTIME};

    // Textures
    cuda2hipRename["cudaBindTexture"]                       = {"hipBindTexture", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaUnbindTexture"]                     = {"hipUnbindTexture", CONV_TEX, API_RUNTIME};
    // Channel
    cuda2hipRename["cudaChannelFormatKind"]                 = {"hipChannelFormatKind", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaChannelFormatKindSigned"]           = {"hipChannelFormatKindSigned", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaChannelFormatKindUnsigned"]         = {"hipChannelFormatKindUnsigned", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaChannelFormatKindFloat"]            = {"hipChannelFormatKindFloat", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaChannelFormatKindNone"]             = {"hipChannelFormatKindNone", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaChannelFormatDesc"]                 = {"hipChannelFormatDesc", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaCreateChannelDesc"]                 = {"hipCreateChannelDesc", CONV_TEX, API_RUNTIME};

    // Inter-Process Communications (IPC)
    // IPC types
    cuda2hipRename["cudaIpcEventHandle_t"]                  = {"hipIpcEventHandle_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaIpcEventHandle_st"]                 = {"hipIpcEventHandle_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaIpcMemHandle_t"]                    = {"hipIpcMemHandle_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaIpcMemHandle_st"]                   = {"hipIpcMemHandle_t", CONV_TYPE, API_RUNTIME};

    // IPC functions
    cuda2hipRename["cudaIpcCloseMemHandle"]                 = {"hipIpcCloseMemHandle", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaIpcGetEventHandle"]                 = {"hipIpcGetEventHandle", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaIpcGetMemHandle"]                   = {"hipIpcGetMemHandle", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaIpcOpenEventHandle"]                = {"hipIpcOpenEventHandle", CONV_DEV, API_RUNTIME};
    cuda2hipRename["cudaIpcOpenMemHandle"]                  = {"hipIpcOpenMemHandle", CONV_DEV, API_RUNTIME};

    //---------------------------------------BLAS-------------------------------------//
    // Blas types
    cuda2hipRename["cublasHandle_t"]                 = {"hipblasHandle_t", CONV_TYPE, API_BLAS};
    // TODO: dereferencing: typedef struct cublasContext *cublasHandle_t;
    cuda2hipRename["cublasContext"]                  = {"hipblasHandle_t", CONV_TYPE, API_BLAS};
    // Blas management functions
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasInit"]                     = {"hipblasInit", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasShutdown"]                 = {"hipblasShutdown", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasGetVersion"]               = {"hipblasGetVersion", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasGetError"]                 = {"hipblasGetError", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasAlloc"]                    = {"hipblasAlloc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasFree"]                     = {"hipblasFree", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasSetKernelStream"]          = {"hipblasSetKernelStream", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasGetAtomicsMode"]           = {"hipblasGetAtomicsMode", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasSetAtomicsMode"]           = {"hipblasSetAtomicsMode", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    // Blas operations
    cuda2hipRename["cublasOperation_t"]              = {"hipblasOperation_t", CONV_TYPE, API_BLAS};
    cuda2hipRename["CUBLAS_OP_N"]                    = {"HIPBLAS_OP_N", CONV_NUMERIC_LITERAL, API_BLAS};
    cuda2hipRename["CUBLAS_OP_T"]                    = {"HIPBLAS_OP_T", CONV_NUMERIC_LITERAL, API_BLAS};
    cuda2hipRename["CUBLAS_OP_C"]                    = {"HIPBLAS_OP_C", CONV_NUMERIC_LITERAL, API_BLAS};
    // Blas statuses
    cuda2hipRename["cublasStatus_t"]                 = {"hipblasStatus_t", CONV_TYPE, API_BLAS};
    cuda2hipRename["CUBLAS_STATUS_SUCCESS"]          = {"HIPBLAS_STATUS_SUCCESS", CONV_NUMERIC_LITERAL, API_BLAS};
    cuda2hipRename["CUBLAS_STATUS_NOT_INITIALIZED"]  = {"HIPBLAS_STATUS_NOT_INITIALIZED", CONV_NUMERIC_LITERAL, API_BLAS};
    cuda2hipRename["CUBLAS_STATUS_ALLOC_FAILED"]     = {"HIPBLAS_STATUS_ALLOC_FAILED", CONV_NUMERIC_LITERAL, API_BLAS};
    cuda2hipRename["CUBLAS_STATUS_INVALID_VALUE"]    = {"HIPBLAS_STATUS_INVALID_VALUE", CONV_NUMERIC_LITERAL, API_BLAS};
    cuda2hipRename["CUBLAS_STATUS_MAPPING_ERROR"]    = {"HIPBLAS_STATUS_MAPPING_ERROR", CONV_NUMERIC_LITERAL, API_BLAS};
    cuda2hipRename["CUBLAS_STATUS_EXECUTION_FAILED"] = {"HIPBLAS_STATUS_EXECUTION_FAILED", CONV_NUMERIC_LITERAL, API_BLAS};
    cuda2hipRename["CUBLAS_STATUS_INTERNAL_ERROR"]   = {"HIPBLAS_STATUS_INTERNAL_ERROR", CONV_NUMERIC_LITERAL, API_BLAS};
    cuda2hipRename["CUBLAS_STATUS_NOT_SUPPORTED"]    = {"HIPBLAS_STATUS_INTERNAL_ERROR", CONV_NUMERIC_LITERAL, API_BLAS};
    // Blas Fill Modes
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasFillMode_t"]               = {"hipblasFillMode_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_FILL_MODE_LOWER"]         = {"HIPBLAS_FILL_MODE_LOWER", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_FILL_MODE_UPPER"]         = {"HIPBLAS_FILL_MODE_UPPER", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    // Blas Diag Types
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDiagType_t"]               = {"hipblasDiagType_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_DIAG_NON_UNIT"]           = {"HIPBLAS_DIAG_NON_UNIT", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_DIAG_UNIT"]               = {"HIPBLAS_DIAG_UNIT", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    // Blas Side Modes
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSideMode_t"]               = {"hipblasSideMode_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_SIDE_LEFT"]               = {"HIPBLAS_SIDE_LEFT", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_SIDE_RIGHT"]              = {"HIPBLAS_SIDE_RIGHT", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    // Blas Pointer Modes
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasPointerMode_t"]            = {"hipblasPointerMode_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_POINTER_MODE_HOST"]       = {"HIPBLAS_POINTER_MODE_HOST", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_POINTER_MODE_DEVICE"]     = {"HIPBLAS_POINTER_MODE_DEVICE", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    // Blas Atomics Modes
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasAtomicsMode_t"]            = {"hipblasAtomicsMode_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_ATOMICS_NOT_ALLOWED"]     = {"HIPBLAS_ATOMICS_NOT_ALLOWED", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_ATOMICS_ALLOWED"]         = {"HIPBLAS_ATOMICS_ALLOWED", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    // Blas Data Type
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDataType_t"]               = {"hipblasDataType_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_DATA_FLOAT"]              = {"HIPBLAS_DATA_FLOAT", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_DATA_DOUBLE"]             = {"HIPBLAS_DATA_DOUBLE", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_DATA_HALF"]               = {"HIPBLAS_DATA_HALF", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["CUBLAS_DATA_INT8"]               = {"HIPBLAS_DATA_INT8", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED};

    // Blas1 (v1) Routines
    cuda2hipRename["cublasCreate"]                   = {"hipblasCreate", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasDestroy"]                  = {"hipblasDestroy", CONV_MATH_FUNC, API_BLAS};

    cuda2hipRename["cublasSetVector"]                = {"hipblasSetVector", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasGetVector"]                = {"hipblasGetVector", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasSetMatrix"]                = {"hipblasSetMatrix", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasGetMatrix"]                = {"hipblasGetMatrix", CONV_MATH_FUNC, API_BLAS};

    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasGetMatrixAsync"]           = {"hipblasGetMatrixAsync", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasSetMatrixAsync"]           = {"hipblasSetMatrixAsync", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // NRM2
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSnrm2"]                    = {"hipblasSnrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDnrm2"]                    = {"hipblasDnrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasScnrm2"]                   = {"hipblasScnrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDznrm2"]                   = {"hipblasDznrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // DOT
    cuda2hipRename["cublasSdot"]                     = {"hipblasSdot", CONV_MATH_FUNC, API_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasSdotBatched"]              = {"hipblasSdotBatched",CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasDdot"]                     = {"hipblasDdot", CONV_MATH_FUNC, API_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasDdotBatched"]              = {"hipblasDdotBatched", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasCdotu"]                    = {"hipblasCdotu", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCdotc"]                    = {"hipblasCdotc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZdotu"]                    = {"hipblasZdotu", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZdotc"]                    = {"hipblasZdotc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SCAL
    cuda2hipRename["cublasSscal"]                    = {"hipblasSscal", CONV_MATH_FUNC, API_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasSscalBatched"]             = {"hipblasSscalBatched", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasDscal"]                    = {"hipblasDscal", CONV_MATH_FUNC, API_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasDscalBatched"]             = {"hipblasDscalBatched", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasCscal"]                    = {"hipblasCscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsscal"]                   = {"hipblasCsscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZscal"]                    = {"hipblasZscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZdscal"]                   = {"hipblasZdscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // AXPY
    cuda2hipRename["cublasSaxpy"]                    = {"hipblasSaxpy", CONV_MATH_FUNC, API_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasSaxpyBatched"]             = {"hipblasSaxpyBatched", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDaxpy"]                    = {"hipblasDaxpy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCaxpy"]                    = {"hipblasCaxpy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZaxpy"]                    = {"hipblasZaxpy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // COPY
    cuda2hipRename["cublasScopy"]                    = {"hipblasScopy", CONV_MATH_FUNC, API_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasScopyBatched"]             = {"hipblasScopyBatched", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasDcopy"]                    = {"hipblasDcopy", CONV_MATH_FUNC, API_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasDcopyBatched"]             = {"hipblasDcopyBatched", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasCcopy"]                    = {"hipblasCcopy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZcopy"]                    = {"hipblasZcopy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SWAP
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSswap"]                    = {"hipblasSswap", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDswap"]                    = {"hipblasDswap", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCswap"]                    = {"hipblasCswap", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZswap"]                    = {"hipblasZswap", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // AMAX
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasIsamax"]                   = {"hipblasIsamax", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIdamax"]                   = {"hipblasIdamax", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIcamax"]                   = {"hipblasIcamax", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIzamax"]                   = {"hipblasIzamax", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // AMIN
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasIsamin"]                   = {"hipblasIsamin", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIdamin"]                   = {"hipblasIdamin", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIcamin"]                   = {"hipblasIcamin", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIzamin"]                   = {"hipblasIzamin", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // ASUM
    cuda2hipRename["cublasSasum"]                    = {"hipblasSasum", CONV_MATH_FUNC, API_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasSasumBatched"]             = {"hipblasSasumBatched", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasDasum"]                    = {"hipblasDasum", CONV_MATH_FUNC, API_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasDasumBatched"]             = {"hipblasDasumBatched", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasScasum"]                   = {"hipblasScasum", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDzasum"]                   = {"hipblasDzasum", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // ROT
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSrot"]                     = {"hipblasSrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDrot"]                     = {"hipblasDrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCrot"]                     = {"hipblasCrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsrot"]                    = {"hipblasCsrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZrot"]                     = {"hipblasZrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZdrot"]                    = {"hipblasZdrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // ROTG
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSrotg"]                    = {"hipblasSrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDrotg"]                    = {"hipblasDrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCrotg"]                    = {"hipblasCrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZrotg"]                    = {"hipblasZrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // ROTM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSrotm"]                    = {"hipblasSrotm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDrotm"]                    = {"hipblasDrotm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // ROTMG
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSrotmg"]                   = {"hipblasSrotmg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDrotmg"]                   = {"hipblasDrotmg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // GEMV
    cuda2hipRename["cublasSgemv"]                    = {"hipblasSgemv", CONV_MATH_FUNC, API_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasSgemvBatched"]             = {"hipblasSgemvBatched", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDgemv"]                    = {"hipblasDgemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgemv"]                    = {"hipblasCgemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgemv"]                    = {"hipblasZgemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // GBMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSgbmv"]                    = {"hipblasSgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDgbmv"]                    = {"hipblasDgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgbmv"]                    = {"hipblasCgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgbmv"]                    = {"hipblasZgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TRMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStrmv"]                    = {"hipblasStrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtrmv"]                    = {"hipblasDtrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtrmv"]                    = {"hipblasCtrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtrmv"]                    = {"hipblasZtrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TBMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStbmv"]                    = {"hipblasStbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtbmv"]                    = {"hipblasDtbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtbmv"]                    = {"hipblasCtbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtbmv"]                    = {"hipblasZtbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TPMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStpmv"]                    = {"hipblasStpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtpmv"]                    = {"hipblasDtpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtpmv"]                    = {"hipblasCtpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtpmv"]                    = {"hipblasZtpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TRSV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStrsv"]                    = {"hipblasStrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtrsv"]                    = {"hipblasDtrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtrsv"]                    = {"hipblasCtrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtrsv"]                    = {"hipblasZtrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TPSV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStpsv"]                    = {"hipblasStpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtpsv"]                    = {"hipblasDtpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtpsv"]                    = {"hipblasCtpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtpsv"]                    = {"hipblasZtpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TBSV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStbsv"]                    = {"hipblasStbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtbsv"]                    = {"hipblasDtbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtbsv"]                    = {"hipblasCtbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtbsv"]                    = {"hipblasZtbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYMV/HEMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsymv"]                    = {"hipblasSsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsymv"]                    = {"hipblasDsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsymv"]                    = {"hipblasCsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZsymv"]                    = {"hipblasZsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasChemv"]                    = {"hipblasChemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhemv"]                    = {"hipblasZhemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SBMV/HBMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsbmv"]                    = {"hipblasSsbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsbmv"]                    = {"hpiblasDsbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasChbmv"]                    = {"hipblasChbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhbmv"]                    = {"hipblasZhbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SPMV/HPMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSspmv"]                    = {"hipblasSspmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDspmv"]                    = {"hipblasDspmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasChpmv"]                    = {"hipblasChpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhpmv"]                    = {"hipblasZhpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // GER
    cuda2hipRename["cublasSger"]                     = {"hipblasSger", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDger"]                     = {"hipblasDger", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgeru"]                    = {"hipblasCgeru", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgerc"]                    = {"hipblasCgerc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgeru"]                    = {"hipblasZgeru", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgerc"]                    = {"hipblasZgerc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYR/HER
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsyr"]                     = {"hipblasSsyr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsyr"]                     = {"hipblasDsyr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCher"]                     = {"hipblasCher", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZher"]                     = {"hipblasZher", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SPR/HPR
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSspr"]                     = {"hipblasSspr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDspr"]                     = {"hipblasDspr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasChpr"]                     = {"hipblasChpr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhpr"]                     = {"hipblasZhpr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYR2/HER2
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsyr2"]                    = {"hipblasSsyr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsyr2"]                    = {"hipblasDsyr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCher2"]                    = {"hipblasCher2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZher2"]                    = {"hipblasZher2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SPR2/HPR2
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSspr2"]                    = {"hipblasSspr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDspr2"]                    = {"hipblasDspr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasChpr2"]                    = {"hipblasChpr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhpr2"]                    = {"hipblasZhpr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // Blas3 (v1) Routines
    // GEMM
    cuda2hipRename["cublasSgemm"]                    = {"hipblasSgemm", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDgemm"]                    = {"hipblasDgemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    cuda2hipRename["cublasCgemm"]                    = {"hipblasCgemm", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasZgemm"]                    = {"hipblasZgemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // BATCH GEMM
    cuda2hipRename["cublasSgemmBatched"]             = {"hipblasSgemmBatched", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDgemmBatched"]             = {"hipblasDgemmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    cuda2hipRename["cublasCgemmBatched"]             = {"hipblasCgemmBatched", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasZgemmBatched"]             = {"hipblasZgemmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYRK
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsyrk"]                    = {"hipblasSsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsyrk"]                    = {"hipblasDsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsyrk"]                    = {"hipblasCsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZsyrk"]                    = {"hipblasZsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // HERK
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasCherk"]                    = {"hipblasCherk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZherk"]                    = {"hipblasZherk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYR2K
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsyr2k"]                   = {"hipblasSsyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsyr2k"]                   = {"hipblasDsyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsyr2k"]                   = {"hipblasCsyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZsyr2k"]                   = {"hipblasZsyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYRKX - eXtended SYRK
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsyrkx"]                   = {"hipblasSsyrkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsyrkx"]                   = {"hipblasDsyrkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsyrkx"]                   = {"hipblasCsyrkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZsyrkx"]                   = {"hipblasZsyrkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};


    // HER2K
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasCher2k"]                   = {"hipblasCher2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZher2k"]                   = {"hipblasZher2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // HERKX - eXtended HERK
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasCherkx"]                   = {"hipblasCherkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZherkx"]                   = {"hipblasZherkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYMM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsymm"]                    = {"hipblasSsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsymm"]                    = {"hipblasDsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsymm"]                    = {"hipblasCsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZsymm"]                    = {"hipblasZsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // HEMM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasChemm"]                    = {"hipblasChemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhemm"]                    = {"hipblasZhemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TRSM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStrsm"]                    = {"hipblasStrsm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtrsm"]                    = {"hipblasDtrsm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtrsm"]                    = {"hipblasCtrsm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtrsm"]                    = {"hipblasZtrsm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TRSM - Batched Triangular Solver
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStrsmBatched"]             = {"hipblasStrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtrsmBatched"]             = {"hipblasDtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtrsmBatched"]             = {"hipblasCtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtrsmBatched"]             = {"hipblasZtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TRMM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStrmm"]                    = {"hipblasStrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtrmm"]                    = {"hipblasDtrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtrmm"]                    = {"hipblasCtrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtrmm"]                    = {"hipblasZtrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // ------------------------ CUBLAS BLAS - like extension (cublas_api.h)
    // GEAM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSgeam"]                    = {"hipblasSgeam", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDgeam"]                    = {"hipblasDgeam", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgeam"]                    = {"hipblasCgeam", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgeam"]                    = {"hipblasZgeam", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // GETRF - Batched LU
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSgetrfBatched"]            = {"hipblasSgetrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDgetrfBatched"]            = {"hipblasDgetrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgetrfBatched"]            = {"hipblasCgetrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgetrfBatched"]            = {"hipblasZgetrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // Batched inversion based on LU factorization from getrf
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSgetriBatched"]            = {"hipblasSgetriBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDgetriBatched"]            = {"hipblasDgetriBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgetriBatched"]            = {"hipblasCgetriBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgetriBatched"]            = {"hipblasZgetriBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // Batched solver based on LU factorization from getrf
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSgetrsBatched"]            = {"hipblasSgetrsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDgetrsBatched"]            = {"hipblasDgetrsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgetrsBatched"]            = {"hipblasCgetrsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgetrsBatched"]            = {"hipblasZgetrsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TRSM - Batched Triangular Solver
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStrsmBatched"]             = {"hipblasStrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtrsmBatched"]             = {"hipblasDtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtrsmBatched"]             = {"hipblasCtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtrsmBatched"]             = {"hipblasZtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // MATINV - Batched
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSmatinvBatched"]           = {"hipblasSmatinvBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDmatinvBatched"]           = {"hipblasDmatinvBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCmatinvBatched"]           = {"hipblasCmatinvBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZmatinvBatched"]           = {"hipblasZmatinvBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // Batch QR Factorization
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSgeqrfBatched"]            = {"hipblasSgeqrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDgeqrfBatched"]            = {"hipblasDgeqrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgeqrfBatched"]            = {"hipblasCgeqrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgeqrfBatched"]            = {"hipblasZgeqrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // Least Square Min only m >= n and Non-transpose supported
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSgelsBatched"]             = {"hipblasSgelsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDgelsBatched"]             = {"hipblasDgelsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgelsBatched"]             = {"hipblasCgelsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgelsBatched"]             = {"hipblasZgelsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // DGMM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSdgmm"]                    = {"hipblasSdgmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDdgmm"]                    = {"hipblasDdgmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCdgmm"]                    = {"hipblasCdgmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZdgmm"]                    = {"hipblasZdgmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TPTTR - Triangular Pack format to Triangular format
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStpttr"]                   = {"hipblasStpttr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtpttr"]                   = {"hipblasDtpttr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtpttr"]                   = {"hipblasCtpttr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtpttr"]                   = {"hipblasZtpttr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TRTTP - Triangular format to Triangular Pack format
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStrttp"]                   = {"hipblasStrttp", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtrttp"]                   = {"hipblasDtrttp", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtrttp"]                   = {"hipblasCtrttp", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtrttp"]                   = {"hipblasZtrttp", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // Blas2 (v2) Routines
    cuda2hipRename["cublasCreate_v2"]                = {"hipblasCreate", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasDestroy_v2"]               = {"hipblasDestroy", CONV_MATH_FUNC, API_BLAS};

    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasGetVersion_v2"]            = {"hipblasGetVersion", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasSetStream_v2"]             = {"hipblasSetStream", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasGetStream_v2"]             = {"hipblasGetStream", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasGetPointerMode_v2"]        = {"hipblasGetPointerMode", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasSetPointerMode_v2"]        = {"hipblasSetPointerMode", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // GEMV
    cuda2hipRename["cublasSgemv_v2"]                 = {"hipblasSgemv", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDgemv_v2"]                 = {"hipblasDgemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgemv_v2"]                 = {"hipblasCgemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgemv_v2"]                 = {"hipblasZgemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // GBMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSgbmv_v2"]                 = {"hipblasSgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDgbmv_v2"]                 = {"hipblasDgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgbmv_v2"]                 = {"hipblasCgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgbmv_v2"]                 = {"hipblasZgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TRMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStrmv_v2"]                 = {"hipblasStrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtrmv_v2"]                 = {"hipblasDtrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtrmv_v2"]                 = {"hipblasCtrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtrmv_v2"]                 = {"hipblasZtrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TBMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStbmv_v2"]                 = {"hipblasStbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtbmv_v2"]                 = {"hipblasDtbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtbmv_v2"]                 = {"hipblasCtbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtbmv_v2"]                 = {"hipblasZtbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TPMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStpmv_v2"]                 = {"hipblasStpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtpmv_v2"]                 = {"hipblasDtpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtpmv_v2"]                 = {"hipblasCtpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtpmv_v2"]                 = {"hipblasZtpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TRSV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStrsv_v2"]                 = {"hipblasStrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtrsv_v2"]                 = {"hipblasDtrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtrsv_v2"]                 = {"hipblasCtrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtrsv_v2"]                 = {"hipblasZtrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TPSV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStpsv_v2"]                 = {"hipblasStpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtpsv_v2"]                 = {"hipblasDtpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtpsv_v2"]                 = {"hipblasCtpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtpsv_v2"]                 = {"hipblasZtpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TBSV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStbsv_v2"]                 = {"hipblasStbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtbsv_v2"]                 = {"hipblasDtbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtbsv_v2"]                 = {"hipblasCtbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtbsv_v2"]                 = {"hipblasZtbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYMV/HEMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsymv_v2"]                 = {"hipblasSsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsymv_v2"]                 = {"hipblasDsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsymv_v2"]                 = {"hipblasCsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZsymv_v2"]                 = {"hipblasZsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasChemv_v2"]                 = {"hipblasChemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhemv_v2"]                 = {"hipblasZhemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SBMV/HBMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsbmv_v2"]                 = {"hipblasSsbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsbmv_v2"]                 = {"hpiblasDsbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasChbmv_v2"]                 = {"hipblasChbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhbmv_v2"]                 = {"hipblasZhbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SPMV/HPMV
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSspmv_v2"]                 = {"hipblasSspmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDspmv_v2"]                 = {"hipblasDspmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasChpmv_v2"]                 = {"hipblasChpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhpmv_v2"]                 = {"hipblasZhpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // GER
    cuda2hipRename["cublasSger_v2"]                  = {"hipblasSger", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDger_v2"]                  = {"hipblasDger", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgeru_v2"]                 = {"hipblasCgeru", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCgerc_v2"]                 = {"hipblasCgerc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgeru_v2"]                 = {"hipblasZgeru", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZgerc_v2"]                 = {"hipblasZgerc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYR/HER
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsyr_v2"]                  = {"hipblasSsyr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsyr_v2"]                  = {"hipblasDsyr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsyr_v2"]                  = {"hipblasCsyr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZsyr_v2"]                  = {"hipblasZsyr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCher_v2"]                  = {"hipblasCher", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZher_v2"]                  = {"hipblasZher", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SPR/HPR
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSspr_v2"]                  = {"hipblasSspr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDspr_v2"]                  = {"hipblasDspr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasChpr_v2"]                  = {"hipblasChpr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhpr_v2"]                  = {"hipblasZhpr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYR2/HER2
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsyr2_v2"]                 = {"hipblasSsyr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsyr2_v2"]                 = {"hipblasDsyr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsyr2_v2"]                 = {"hipblasCsyr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZsyr2_v2"]                 = {"hipblasZsyr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCher2_v2"]                 = {"hipblasCher2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZher2_v2"]                 = {"hipblasZher2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SPR2/HPR2
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSspr2_v2"]                 = {"hipblasSspr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDspr2_v2"]                 = {"hipblasDspr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasChpr2_v2"]                 = {"hipblasChpr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhpr2_v2"]                 = {"hipblasZhpr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // Blas3 (v2) Routines
    // GEMM
    cuda2hipRename["cublasSgemm_v2"]                 = {"hipblasSgemm", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDgemm_v2"]                 = {"hipblasDgemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    cuda2hipRename["cublasCgemm_v2"]                 = {"hipblasCgemm", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasZgemm_v2"]                 = {"hipblasZgemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    //IO in FP16 / FP32, computation in float
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSgemmEx"]                  = {"hipblasSgemmEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYRK
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsyrk_v2"]                 = {"hipblasSsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsyrk_v2"]                 = {"hipblasDsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsyrk_v2"]                 = {"hipblasCsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZsyrk_v2"]                 = {"hipblasZsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // HERK
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasCherk_v2"]                 = {"hipblasCherk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZherk_v2"]                 = {"hipblasZherk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYR2K
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsyr2k_v2"]                = {"hipblasSsyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsyr2k_v2"]                = {"hipblasDsyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsyr2k_v2"]                = {"hipblasCsyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZsyr2k_v2"]                = {"hipblasZsyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // HER2K
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasCher2k_v2"]                = {"hipblasCher2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZher2k_v2"]                = {"hipblasZher2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SYMM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSsymm_v2"]                 = {"hipblasSsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDsymm_v2"]                 = {"hipblasDsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsymm_v2"]                 = {"hipblasCsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZsymm_v2"]                 = {"hipblasZsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // HEMM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasChemm_v2"]                 = {"hipblasChemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZhemm_v2"]                 = {"hipblasZhemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TRSM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStrsm_v2"]                 = {"hipblasStrsm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtrsm_v2"]                 = {"hipblasDtrsm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtrsm_v2"]                 = {"hipblasCtrsm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtrsm_v2"]                 = {"hipblasZtrsm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // TRMM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasStrmm_v2"]                 = {"hipblasStrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDtrmm_v2"]                 = {"hipblasDtrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCtrmm_v2"]                 = {"hipblasCtrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZtrmm_v2"]                 = {"hipblasZtrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // NRM2
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSnrm2_v2"]                 = {"hipblasSnrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDnrm2_v2"]                 = {"hipblasDnrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasScnrm2_v2"]                = {"hipblasScnrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDznrm2_v2"]                = {"hipblasDznrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // DOT
    cuda2hipRename["cublasSdot_v2"]                  = {"hipblasSdot", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasDdot_v2"]                  = {"hipblasDdot", CONV_MATH_FUNC, API_BLAS};

    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasCdotu_v2"]                 = {"hipblasCdotu", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCdotc_v2"]                 = {"hipblasCdotc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZdotu_v2"]                 = {"hipblasZdotu", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZdotc_v2"]                 = {"hipblasZdotc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SCAL
    cuda2hipRename["cublasSscal_v2"]                 = {"hipblasSscal", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasDscal_v2"]                 = {"hipblasDscal", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasCscal_v2"]                 = {"hipblasCscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsscal_v2"]                = {"hipblasCsscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZscal_v2"]                 = {"hipblasZscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZdscal_v2"]                = {"hipblasZdscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // AXPY
    cuda2hipRename["cublasSaxpy_v2"]                 = {"hipblasSaxpy", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDaxpy_v2"]                 = {"hipblasDaxpy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCaxpy_v2"]                 = {"hipblasCaxpy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZaxpy_v2"]                 = {"hipblasZaxpy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // COPY
    cuda2hipRename["cublasScopy_v2"]                 = {"hipblasScopy", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasDcopy_v2"]                 = {"hipblasDcopy", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasCcopy_v2"]                 = {"hipblasCcopy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZcopy_v2"]                 = {"hipblasZcopy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // SWAP
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSswap_v2"]                 = {"hipblasSswap", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDswap_v2"]                 = {"hipblasDswap", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCswap_v2"]                 = {"hipblasCswap", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZswap_v2"]                 = {"hipblasZswap", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // AMAX
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasIsamax_v2"]                = {"hipblasIsamax", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIdamax_v2"]                = {"hipblasIdamax", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIcamax_v2"]                = {"hipblasIcamax", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIzamax_v2"]                = {"hipblasIzamax", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // AMIN
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasIsamin_v2"]                = {"hipblasIsamin", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIdamin_v2"]                = {"hipblasIdamin", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIcamin_v2"]                = {"hipblasIcamin", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasIzamin_v2"]                = {"hipblasIzamin", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // ASUM
    cuda2hipRename["cublasSasum_v2"]                 = {"hipblasSasum", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasDasum_v2"]                 = {"hipblasDasum", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasScasum_v2"]                = {"hipblasScasum", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDzasum_v2"]                = {"hipblasDzasum", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // ROT
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSrot_v2"]                  = {"hipblasSrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDrot_v2"]                  = {"hipblasDrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCrot_v2"]                  = {"hipblasCrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCsrot_v2"]                 = {"hipblasCsrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZrot_v2"]                  = {"hipblasZrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZdrot_v2"]                 = {"hipblasZdrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // ROTG
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSrotg_v2"]                 = {"hipblasSrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDrotg_v2"]                 = {"hipblasDrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasCrotg_v2"]                 = {"hipblasCrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasZrotg_v2"]                 = {"hipblasZrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // ROTM
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSrotm_v2"]                 = {"hipblasSrotm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDrotm_v2"]                 = {"hipblasDrotm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // ROTMG
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasSrotmg_v2"]                = {"hipblasSrotmg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasDrotmg_v2"]                = {"hipblasDrotmg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
  }
};

StringRef unquoteStr(StringRef s) {
  if (s.size() > 1 && s.front() == '"' && s.back() == '"')
    return s.substr(1, s.size() - 2);
  return s;
}

class Cuda2Hip {
public:
  Cuda2Hip(Replacements *R, const std::string &srcFileName) :
    Replace(R), mainFileName(srcFileName) {}
  uint64_t countReps[CONV_LAST] = { 0 };
  uint64_t countApiReps[API_LAST] = { 0 };
  uint64_t countRepsUnsupported[CONV_LAST] = { 0 };
  uint64_t countApiRepsUnsupported[API_LAST] = { 0 };
  std::map<std::string, uint64_t> cuda2hipConverted;
  std::map<std::string, uint64_t> cuda2hipUnconverted;
  std::set<unsigned> LOCs;

  enum msgTypes {
    HIPIFY_ERROR = 0,
    HIPIFY_WARNING
  };

  std::string getMsgType(msgTypes type) {
    switch (type) {
      case HIPIFY_ERROR: return "error";
      default:
      case HIPIFY_WARNING: return "warning";
    }
  }

protected:
  struct cuda2hipMap N;
  Replacements *Replace;
  std::string mainFileName;

  virtual void insertReplacement(const Replacement &rep, const FullSourceLoc &fullSL) {
    Replace->insert(rep);
    if (PrintStats) {
      LOCs.insert(fullSL.getExpansionLineNumber());
    }
  }
  void insertHipHeaders(Cuda2Hip *owner, const SourceManager &SM) {
    if (owner->countReps[CONV_INCLUDE_CUDA_MAIN_H] == 0 && countReps[CONV_INCLUDE_CUDA_MAIN_H] == 0 && Replace->size() > 0) {
      std::string repName = "#include <hip/hip_runtime.h>";
      hipCounter counter = { repName, CONV_INCLUDE_CUDA_MAIN_H, API_RUNTIME };
      updateCounters(counter, repName);
      SourceLocation sl = SM.getLocForStartOfFile(SM.getMainFileID());
      FullSourceLoc fullSL(sl, SM);
      Replacement Rep(SM, sl, 0, repName + "\n");
      insertReplacement(Rep, fullSL);
    }
  }

  void printHipifyMessage(const SourceManager &SM, const SourceLocation &sl, const std::string &message, msgTypes msgType = HIPIFY_WARNING) {
    FullSourceLoc fullSL(sl, SM);
    llvm::errs() << "[HIPIFY] " << getMsgType(msgType) << ": " << mainFileName << ":" << fullSL.getExpansionLineNumber() << ":" << fullSL.getExpansionColumnNumber() << ": " << message << "\n";
  }

  void updateCountersExt(const hipCounter &counter, const std::string &cudaName) {
    std::map<std::string, uint64_t> *map = &cuda2hipConverted;
    std::map<std::string, uint64_t> *mapTotal = &cuda2hipConvertedTotal;
    if (counter.unsupported) {
      map = &cuda2hipUnconverted;
      mapTotal = &cuda2hipUnconvertedTotal;
    }
    auto found = map->find(cudaName);
    if (found == map->end()) {
      map->insert(std::pair<std::string, uint64_t>(cudaName, 1));
    } else {
      found->second++;
    }
    auto foundT = mapTotal->find(cudaName);
    if (foundT == mapTotal->end()) {
      mapTotal->insert(std::pair<std::string, uint64_t>(cudaName, 1));
    } else {
      foundT->second++;
    }
  }

  virtual void updateCounters(const hipCounter &counter, const std::string &cudaName) {
    if (!PrintStats) {
      return;
    }
    updateCountersExt(counter, cudaName);
    if (counter.unsupported) {
      countRepsUnsupported[counter.countType]++;
      countRepsTotalUnsupported[counter.countType]++;
      countApiRepsUnsupported[counter.countApiType]++;
      countApiRepsTotalUnsupported[counter.countApiType]++;
    } else {
      countReps[counter.countType]++;
      countRepsTotal[counter.countType]++;
      countApiReps[counter.countApiType]++;
      countApiRepsTotal[counter.countApiType]++;
    }
  }

  void processString(StringRef s, SourceManager &SM, SourceLocation start) {
    size_t begin = 0;
    while ((begin = s.find("cu", begin)) != StringRef::npos) {
      const size_t end = s.find_first_of(" ", begin + 4);
      StringRef name = s.slice(begin, end);
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        StringRef repName = found->second.hipName;
        hipCounter counter = {"", CONV_LITERAL, API_RUNTIME, found->second.unsupported};
        updateCounters(counter, name.str());
        if (!counter.unsupported) {
          SourceLocation sl = start.getLocWithOffset(begin + 1);
          Replacement Rep(SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, SM);
          insertReplacement(Rep, fullSL);
        }
      } else {
        // std::string msg = "the following reference is not handled: '" + name.str() + "' [string literal].";
        // printHipifyMessage(SM, start, msg);
      }
      if (end == StringRef::npos) {
        break;
      }
      begin = end + 1;
    }
  }
};

class Cuda2HipCallback;

class HipifyPPCallbacks : public PPCallbacks, public SourceFileCallbacks, public Cuda2Hip {
public:
  HipifyPPCallbacks(Replacements *R, const std::string &mainFileName)
    : Cuda2Hip(R, mainFileName), SeenEnd(false), _sm(nullptr), _pp(nullptr) {}

  virtual bool handleBeginSource(CompilerInstance &CI, StringRef Filename) override {
    Preprocessor &PP = CI.getPreprocessor();
    SourceManager &SM = CI.getSourceManager();
    setSourceManager(&SM);
    PP.addPPCallbacks(std::unique_ptr<HipifyPPCallbacks>(this));
    PP.Retain();
    setPreprocessor(&PP);
    return true;
  }

  virtual void handleEndSource() override;

  virtual void InclusionDirective(SourceLocation hash_loc,
                                  const Token &include_token,
                                  StringRef file_name, bool is_angled,
                                  CharSourceRange filename_range,
                                  const FileEntry *file, StringRef search_path,
                                  StringRef relative_path,
                                  const clang::Module *imported) override {
    if (_sm->isWrittenInMainFile(hash_loc)) {
      if (is_angled) {
        const auto found = N.cuda2hipRename.find(file_name);
        if (found != N.cuda2hipRename.end()) {
          updateCounters(found->second, file_name.str());
          if (!found->second.unsupported) {
            StringRef repName = found->second.hipName;
            DEBUG(dbgs() << "Include file found: " << file_name << "\n"
                         << "SourceLocation: "
                         << filename_range.getBegin().printToString(*_sm) << "\n"
                         << "Will be replaced with " << repName << "\n");
            SourceLocation sl = filename_range.getBegin();
            SourceLocation sle = filename_range.getEnd();
            const char *B = _sm->getCharacterData(sl);
            const char *E = _sm->getCharacterData(sle);
            SmallString<128> tmpData;
            Replacement Rep(*_sm, sl, E - B, Twine("<" + repName + ">").toStringRef(tmpData));
            FullSourceLoc fullSL(sl, *_sm);
            insertReplacement(Rep, fullSL);
          }
        } else {
//          llvm::outs() << "[HIPIFY] warning: the following reference is not handled: '" << file_name << "' [inclusion directive].\n";
        }
      }
    }
  }

  virtual void MacroDefined(const Token &MacroNameTok,
                            const MacroDirective *MD) override {
    if (_sm->isWrittenInMainFile(MD->getLocation()) &&
        MD->getKind() == MacroDirective::MD_Define) {
      for (auto T : MD->getMacroInfo()->tokens()) {
        if (T.isAnyIdentifier()) {
          StringRef name = T.getIdentifierInfo()->getName();
          const auto found = N.cuda2hipRename.find(name);
          if (found != N.cuda2hipRename.end()) {
            updateCounters(found->second, name.str());
            if (!found->second.unsupported) {
              StringRef repName = found->second.hipName;
              SourceLocation sl = T.getLocation();
              DEBUG(dbgs() << "Identifier " << name << " found in definition of macro "
                           << MacroNameTok.getIdentifierInfo()->getName() << "\n"
                           << "will be replaced with: " << repName << "\n"
                           << "SourceLocation: " << sl.printToString(*_sm) << "\n");
              Replacement Rep(*_sm, sl, name.size(), repName);
              FullSourceLoc fullSL(sl, *_sm);
              insertReplacement(Rep, fullSL);
            }
          } else {
            // llvm::outs() << "[HIPIFY] warning: the following reference is not handled: '" << name << "' [macro].\n";
          }
        }
      }
    }
  }

  virtual void MacroExpands(const Token &MacroNameTok,
                            const MacroDefinition &MD, SourceRange Range,
                            const MacroArgs *Args) override {
    if (_sm->isWrittenInMainFile(MacroNameTok.getLocation())) {
      for (unsigned int i = 0; Args && i < MD.getMacroInfo()->getNumArgs(); i++) {
        std::vector<Token> toks;
        // Code below is a kind of stolen from 'MacroArgs::getPreExpArgument'
        // to workaround the 'const' MacroArgs passed into this hook.
        const Token *start = Args->getUnexpArgument(i);
        size_t len = Args->getArgLength(start) + 1;
#if (LLVM_VERSION_MAJOR >= 3) && (LLVM_VERSION_MINOR >= 9)
        _pp->EnterTokenStream(ArrayRef<Token>(start, len), false);
#else
        _pp->EnterTokenStream(start, len, false, false);
#endif
        do {
          toks.push_back(Token());
          Token &tk = toks.back();
          _pp->Lex(tk);
        } while (toks.back().isNot(tok::eof));
        _pp->RemoveTopOfLexerStack();
        // end of stolen code
        for (auto tok : toks) {
          if (tok.isAnyIdentifier()) {
            StringRef name = tok.getIdentifierInfo()->getName();
            const auto found = N.cuda2hipRename.find(name);
            if (found != N.cuda2hipRename.end()) {
              updateCounters(found->second, name.str());
              if (!found->second.unsupported) {
                StringRef repName = found->second.hipName;
                DEBUG(dbgs() << "Identifier " << name
                             << " found as an actual argument in expansion of macro "
                             << macroName << "\n"
                             << "will be replaced with: " << repName << "\n");
                size_t length = name.size();
                SourceLocation sl = tok.getLocation();
                if (_sm->isMacroBodyExpansion(sl)) {
                  LangOptions DefaultLangOptions;
                  SourceLocation sl_macro = _sm->getExpansionLoc(sl);
                  SourceLocation sl_end = Lexer::getLocForEndOfToken(sl_macro, 0, *_sm, DefaultLangOptions);
                  length = _sm->getCharacterData(sl_end) - _sm->getCharacterData(sl_macro);
                  name = StringRef(_sm->getCharacterData(sl_macro), length);
                  sl = sl_macro;
                }
                Replacement Rep(*_sm, sl, length, repName);
                FullSourceLoc fullSL(sl, *_sm);
                insertReplacement(Rep, fullSL);
              }
            } else {
              // llvm::outs() << "[HIPIFY] warning: the following reference is not handled: '" << name << "' [macro expansion].\n";
            }
          } else if (tok.isLiteral()) {
            SourceLocation sl = tok.getLocation();
            if (_sm->isMacroBodyExpansion(sl)) {
              LangOptions DefaultLangOptions;
              SourceLocation sl_macro = _sm->getExpansionLoc(sl);
              SourceLocation sl_end = Lexer::getLocForEndOfToken(sl_macro, 0, *_sm, DefaultLangOptions);
              size_t length = _sm->getCharacterData(sl_end) - _sm->getCharacterData(sl_macro);
              StringRef name = StringRef(_sm->getCharacterData(sl_macro), length);
              const auto found = N.cuda2hipRename.find(name);
              if (found != N.cuda2hipRename.end()) {
                updateCounters(found->second, name.str());
                if (!found->second.unsupported) {
                  StringRef repName = found->second.hipName;
                  sl = sl_macro;
                  Replacement Rep(*_sm, sl, length, repName);
                  FullSourceLoc fullSL(sl, *_sm);
                  insertReplacement(Rep, fullSL);
                }
              } else {
                // llvm::outs() << "[HIPIFY] warning: the following reference is not handled: '" << name << "' [literal macro expansion].\n";
              }
            } else {
              if (tok.is(tok::string_literal)) {
                StringRef s(tok.getLiteralData(), tok.getLength());
                processString(unquoteStr(s), *_sm, tok.getLocation());
              }
            }
          }
        }
      }
    }
  }

  void EndOfMainFile() override {}

  bool SeenEnd;
  void setSourceManager(SourceManager *sm) { _sm = sm; }
  void setPreprocessor(Preprocessor *pp) { _pp = pp; }
  void setMatch(Cuda2HipCallback *match) { Match = match; }

private:
  SourceManager *_sm;
  Preprocessor *_pp;
  Cuda2HipCallback *Match;
};

class Cuda2HipCallback : public MatchFinder::MatchCallback, public Cuda2Hip {
private:
  void convertKernelDecl(const FunctionDecl *kernelDecl, const MatchFinder::MatchResult &Result) {
    SourceManager *SM = Result.SourceManager;
    LangOptions DefaultLangOptions;
    SmallString<40> XStr;
    raw_svector_ostream OS(XStr);
    StringRef initialParamList;
    OS << "hipLaunchParm lp";
    size_t repLength = OS.str().size();
    SourceLocation sl = kernelDecl->getNameInfo().getEndLoc();
    SourceLocation kernelArgListStart = Lexer::findLocationAfterToken(sl, tok::l_paren, *SM, DefaultLangOptions, true);
    DEBUG(dbgs() << kernelArgListStart.printToString(*SM));
    if (kernelDecl->getNumParams() > 0) {
      const ParmVarDecl *pvdFirst = kernelDecl->getParamDecl(0);
      const ParmVarDecl *pvdLast =  kernelDecl->getParamDecl(kernelDecl->getNumParams() - 1);
      SourceLocation kernelArgListStart(pvdFirst->getLocStart());
      SourceLocation kernelArgListEnd(pvdLast->getLocEnd());
      SourceLocation stop = Lexer::getLocForEndOfToken(kernelArgListEnd, 0, *SM, DefaultLangOptions);
      repLength += SM->getCharacterData(stop) - SM->getCharacterData(kernelArgListStart);
      initialParamList = StringRef(SM->getCharacterData(kernelArgListStart), repLength);
      OS << ", " << initialParamList;
    }
    DEBUG(dbgs() << "initial paramlist: " << initialParamList << "\n" << "new paramlist: " << OS.str() << "\n");
    Replacement Rep0(*(Result.SourceManager), kernelArgListStart, repLength, OS.str());
    FullSourceLoc fullSL(sl, *(Result.SourceManager));
    insertReplacement(Rep0, fullSL);
  }

  bool cudaCall(const MatchFinder::MatchResult &Result) {
    if (const CallExpr *call = Result.Nodes.getNodeAs<CallExpr>("cudaCall")) {
      const FunctionDecl *funcDcl = call->getDirectCallee();
      StringRef name = funcDcl->getDeclName().getAsString();
      SourceManager *SM = Result.SourceManager;
      SourceLocation sl = call->getLocStart();
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        if (!found->second.unsupported) {
          StringRef repName = found->second.hipName;
          size_t length = name.size();
          bool bReplace = true;
          if (SM->isMacroArgExpansion(sl)) {
            sl = SM->getImmediateSpellingLoc(sl);
          } else if (SM->isMacroBodyExpansion(sl)) {
            LangOptions DefaultLangOptions;
            SourceLocation sl_macro = SM->getExpansionLoc(sl);
            SourceLocation sl_end = Lexer::getLocForEndOfToken(sl_macro, 0, *SM, DefaultLangOptions);
            length = SM->getCharacterData(sl_end) - SM->getCharacterData(sl_macro);
            StringRef macroName = StringRef(SM->getCharacterData(sl_macro), length);
            if (N.cudaExcludes.end() != N.cudaExcludes.find(macroName)) {
              bReplace = false;
            } else {
              sl = sl_macro;
            }
          }
          if (bReplace) {
            updateCounters(found->second, name.str());
            Replacement Rep(*SM, sl, length, repName);
            FullSourceLoc fullSL(sl, *SM);
            insertReplacement(Rep, fullSL);
          }
        } else {
          updateCounters(found->second, name.str());
        }
      } else {
        std::string msg = "the following reference is not handled: '" + name.str() + "' [function call].";
        printHipifyMessage(*SM, sl, msg);
      }
      return true;
    }
    return false;
  }

  bool cudaLaunchKernel(const MatchFinder::MatchResult &Result) {
    StringRef refName = "cudaLaunchKernel";
    if (const CUDAKernelCallExpr *launchKernel = Result.Nodes.getNodeAs<CUDAKernelCallExpr>(refName)) {
      SmallString<40> XStr;
      raw_svector_ostream OS(XStr);
      StringRef calleeName;
      const FunctionDecl *kernelDecl = launchKernel->getDirectCallee();
      if (kernelDecl) {
        calleeName = kernelDecl->getName();
        convertKernelDecl(kernelDecl, Result);
      } else {
        const Expr *e = launchKernel->getCallee();
        if (const UnresolvedLookupExpr *ule =
          dyn_cast<UnresolvedLookupExpr>(e)) {
          calleeName = ule->getName().getAsIdentifierInfo()->getName();
          owner->addMatcher(functionTemplateDecl(hasName(calleeName))
            .bind("unresolvedTemplateName"), this);
        }
      }
      XStr.clear();
      if (calleeName.find(',') != StringRef::npos) {
        SmallString<128> tmpData;
        calleeName = Twine("HIP_KERNEL_NAME(" + calleeName + ")").toStringRef(tmpData);
      }
      OS << "hipLaunchKernel(" << calleeName << ",";
      const CallExpr *config = launchKernel->getConfig();
      DEBUG(dbgs() << "Kernel config arguments:" << "\n");
      SourceManager *SM = Result.SourceManager;
      LangOptions DefaultLangOptions;
      for (unsigned argno = 0; argno < config->getNumArgs(); argno++) {
        const Expr *arg = config->getArg(argno);
        if (!isa<CXXDefaultArgExpr>(arg)) {
          const ParmVarDecl *pvd = config->getDirectCallee()->getParamDecl(argno);
          SourceLocation sl(arg->getLocStart());
          SourceLocation el(arg->getLocEnd());
          SourceLocation stop = Lexer::getLocForEndOfToken(el, 0, *SM, DefaultLangOptions);
          StringRef outs(SM->getCharacterData(sl), SM->getCharacterData(stop) - SM->getCharacterData(sl));
          DEBUG(dbgs() << "args[ " << argno << "]" << outs << " <" << pvd->getType().getAsString() << ">\n");
          if (pvd->getType().getAsString().compare("dim3") == 0) {
            OS << " dim3(" << outs << "),";
          } else {
            OS << " " << outs << ",";
          }
        } else {
          OS << " 0,";
        }
      }
      for (unsigned argno = 0; argno < launchKernel->getNumArgs(); argno++) {
        const Expr *arg = launchKernel->getArg(argno);
        SourceLocation sl(arg->getLocStart());
        SourceLocation el(arg->getLocEnd());
        SourceLocation stop = Lexer::getLocForEndOfToken(el, 0, *SM, DefaultLangOptions);
        std::string outs(SM->getCharacterData(sl), SM->getCharacterData(stop) - SM->getCharacterData(sl));
        DEBUG(dbgs() << outs << "\n");
        OS << " " << outs << ",";
      }
      XStr.pop_back();
      OS << ")";
      size_t length = SM->getCharacterData(Lexer::getLocForEndOfToken(
                        launchKernel->getLocEnd(), 0, *SM, DefaultLangOptions)) -
                        SM->getCharacterData(launchKernel->getLocStart());
      Replacement Rep(*SM, launchKernel->getLocStart(), length, OS.str());
      FullSourceLoc fullSL(launchKernel->getLocStart(), *SM);
      insertReplacement(Rep, fullSL);
      hipCounter counter = {"hipLaunchKernel", CONV_KERN, API_RUNTIME};
      updateCounters(counter, refName.str());
      return true;
    }
    return false;
  }

  bool cudaBuiltin(const MatchFinder::MatchResult &Result) {
    if (const MemberExpr *threadIdx = Result.Nodes.getNodeAs<MemberExpr>("cudaBuiltin")) {
      if (const OpaqueValueExpr *refBase =
        dyn_cast<OpaqueValueExpr>(threadIdx->getBase())) {
        if (const DeclRefExpr *declRef =
          dyn_cast<DeclRefExpr>(refBase->getSourceExpr())) {
          SourceLocation sl = threadIdx->getLocStart();
          SourceManager *SM = Result.SourceManager;
          StringRef name = declRef->getDecl()->getName();
          StringRef memberName = threadIdx->getMemberDecl()->getName();
          size_t pos = memberName.find_first_not_of("__fetch_builtin_");
          memberName = memberName.slice(pos, memberName.size());
          SmallString<128> tmpData;
          name = Twine(name + "." + memberName).toStringRef(tmpData);
          const auto found = N.cuda2hipRename.find(name);
          if (found != N.cuda2hipRename.end()) {
            updateCounters(found->second, name.str());
            if (!found->second.unsupported) {
              StringRef repName = found->second.hipName;
              Replacement Rep(*SM, sl, name.size(), repName);
              FullSourceLoc fullSL(sl, *SM);
              insertReplacement(Rep, fullSL);
            }
          } else {
            std::string msg = "the following reference is not handled: '" + name.str() + "' [builtin].";
            printHipifyMessage(*SM, sl, msg);
          }
        }
      }
      return true;
    }
    return false;
  }

  bool cudaEnumConstantRef(const MatchFinder::MatchResult &Result) {
    if (const DeclRefExpr *enumConstantRef = Result.Nodes.getNodeAs<DeclRefExpr>("cudaEnumConstantRef")) {
      StringRef name = enumConstantRef->getDecl()->getNameAsString();
      SourceLocation sl = enumConstantRef->getLocStart();
      SourceManager *SM = Result.SourceManager;
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        updateCounters(found->second, name.str());
        if (!found->second.unsupported) {
          StringRef repName = found->second.hipName;
          Replacement Rep(*SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, *SM);
          insertReplacement(Rep, fullSL);
        }
      } else {
        std::string msg = "the following reference is not handled: '" + name.str() + "' [enum constant ref].";
        printHipifyMessage(*SM, sl, msg);
      }
      return true;
    }
    return false;
  }

  bool cudaEnumConstantDecl(const MatchFinder::MatchResult &Result) {
    if (const VarDecl *enumConstantDecl = Result.Nodes.getNodeAs<VarDecl>("cudaEnumConstantDecl")) {
      StringRef name =
        enumConstantDecl->getType()->getAsTagDecl()->getNameAsString();
      // anonymous typedef enum
      if (name.empty()) {
        QualType QT = enumConstantDecl->getType().getUnqualifiedType();
        name = QT.getAsString();
      }
      SourceLocation sl = enumConstantDecl->getLocStart();
      SourceManager *SM = Result.SourceManager;
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        updateCounters(found->second, name.str());
        if (!found->second.unsupported) {
          StringRef repName = found->second.hipName;
          Replacement Rep(*SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, *SM);
          insertReplacement(Rep, fullSL);
        }
      } else {
        std::string msg = "the following reference is not handled: '" + name.str() + "' [enum constant decl].";
        printHipifyMessage(*SM, sl, msg);
      }
      return true;
    }
    return false;
  }

  bool cudaTypedefVar(const MatchFinder::MatchResult &Result) {
    if (const VarDecl *typedefVar = Result.Nodes.getNodeAs<VarDecl>("cudaTypedefVar")) {
      QualType QT = typedefVar->getType();
      if (QT->isArrayType()) {
        QT = QT.getTypePtr()->getAsArrayTypeUnsafe()->getElementType();
      }
      QT = QT.getUnqualifiedType();
      StringRef name = QT.getAsString();
      SourceLocation sl = typedefVar->getLocStart();
      SourceManager *SM = Result.SourceManager;
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        updateCounters(found->second, name.str());
        if (!found->second.unsupported) {
          StringRef repName = found->second.hipName;
          Replacement Rep(*SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, *SM);
          insertReplacement(Rep, fullSL);
        }
      } else {
        std::string msg = "the following reference is not handled: '" + name.str() + "' [typedef var].";
        printHipifyMessage(*SM, sl, msg);
      }
      return true;
    }
    return false;
  }

  bool cudaTypedefVarPtr(const MatchFinder::MatchResult &Result) {
    if (const VarDecl *typedefVarPtr = Result.Nodes.getNodeAs<VarDecl>("cudaTypedefVarPtr")) {
      const Type *t = typedefVarPtr->getType().getTypePtrOrNull();
      if (t) {
        SourceManager *SM = Result.SourceManager;
        TypeLoc TL = typedefVarPtr->getTypeSourceInfo()->getTypeLoc();
        SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
        QualType QT = t->getPointeeType();
        QT = QT.getUnqualifiedType();
        StringRef name = QT.getAsString();
        const auto found = N.cuda2hipRename.find(name);
        if (found != N.cuda2hipRename.end()) {
          updateCounters(found->second, name.str());
          if (!found->second.unsupported) {
            StringRef repName = found->second.hipName;
            Replacement Rep(*SM, sl, name.size(), repName);
            FullSourceLoc fullSL(sl, *SM);
            insertReplacement(Rep, fullSL);
          }
        }
        else {
          std::string msg = "the following reference is not handled: '" + name.str() + "' [typedef var ptr].";
          printHipifyMessage(*SM, sl, msg);
        }
      }
      return true;
    }
    return false;
  }

  bool cudaStructVar(const MatchFinder::MatchResult &Result) {
    if (const VarDecl *structVar = Result.Nodes.getNodeAs<VarDecl>("cudaStructVar")) {
      QualType QT = structVar->getType();
      // ToDo: find case-studies with types other than Struct.
      if (QT->isStructureType()) {
        StringRef name = QT.getTypePtr()->getAsStructureType()->getDecl()->getNameAsString();
        TypeLoc TL = structVar->getTypeSourceInfo()->getTypeLoc();
        SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
        SourceManager *SM = Result.SourceManager;
        const auto found = N.cuda2hipRename.find(name);
        if (found != N.cuda2hipRename.end()) {
          updateCounters(found->second, name.str());
          if (!found->second.unsupported) {
            StringRef repName = found->second.hipName;
            Replacement Rep(*SM, sl, name.size(), repName);
            FullSourceLoc fullSL(sl, *SM);
            insertReplacement(Rep, fullSL);
          }
        }
        else {
          std::string msg = "the following reference is not handled: '" + name.str() + "' [struct var].";
          printHipifyMessage(*SM, sl, msg);
        }
      }
      return true;
    }
    return false;
  }

  bool cudaStructVarPtr(const MatchFinder::MatchResult &Result) {
    if (const VarDecl *structVarPtr = Result.Nodes.getNodeAs<VarDecl>("cudaStructVarPtr")) {
      const Type *t = structVarPtr->getType().getTypePtrOrNull();
      if (t) {
        TypeLoc TL = structVarPtr->getTypeSourceInfo()->getTypeLoc();
        SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
        SourceManager *SM = Result.SourceManager;
        StringRef name = t->getPointeeCXXRecordDecl()->getName();
        const auto found = N.cuda2hipRename.find(name);
        if (found != N.cuda2hipRename.end()) {
          updateCounters(found->second, name.str());
          if (!found->second.unsupported) {
            StringRef repName = found->second.hipName;
            Replacement Rep(*SM, sl, name.size(), repName);
            FullSourceLoc fullSL(sl, *SM);
            insertReplacement(Rep, fullSL);
          }
        } else {
          std::string msg = "the following reference is not handled: '" + name.str() + "' [struct var ptr].";
          printHipifyMessage(*SM, sl, msg);
        }
      }
      return true;
    }
    return false;
  }

  bool cudaStructSizeOf(const MatchFinder::MatchResult &Result) {
    if (const UnaryExprOrTypeTraitExpr *expr = Result.Nodes.getNodeAs<UnaryExprOrTypeTraitExpr>("cudaStructSizeOf")) {
      TypeSourceInfo *typeInfo = expr->getArgumentTypeInfo();
      TypeLoc TL = typeInfo->getTypeLoc();
      SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
      SourceManager *SM = Result.SourceManager;
      QualType QT = typeInfo->getType().getUnqualifiedType();
      const Type *type = QT.getTypePtr();
      CXXRecordDecl *rec = type->getAsCXXRecordDecl();
      if (!rec) {
        return false;
      }
      StringRef name = rec->getName();
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        updateCounters(found->second, name.str());
        if (!found->second.unsupported) {
          StringRef repName = found->second.hipName;
          Replacement Rep(*SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, *SM);
          insertReplacement(Rep, fullSL);
        }
      } else {
        std::string msg = "the following reference is not handled: '" + name.str() + "' [struct sizeof].";
        printHipifyMessage(*SM, sl, msg);
      }
      return true;
    }
    return false;
  }

  bool cudaSharedIncompleteArrayVar(const MatchFinder::MatchResult &Result) {
    StringRef refName = "cudaSharedIncompleteArrayVar";
    if (const VarDecl *sharedVar = Result.Nodes.getNodeAs<VarDecl>(refName)) {
      // Example: extern __shared__ uint sRadix1[];
      if (sharedVar->hasExternalFormalLinkage()) {
        QualType QT = sharedVar->getType();
        StringRef typeName;
        if (QT->isIncompleteArrayType()) {
          const ArrayType *AT = QT.getTypePtr()->getAsArrayTypeUnsafe();
          QT = AT->getElementType();
          if (QT.getTypePtr()->isBuiltinType()) {
            QT = QT.getCanonicalType();
            const BuiltinType *BT = dyn_cast<BuiltinType>(QT);
            if (BT) {
              LangOptions LO;
              LO.CUDA = true;
              PrintingPolicy policy(LO);
              typeName = BT->getName(policy);
            }
          } else {
            typeName = QT.getAsString();
          }
        }
        if (!typeName.empty()) {
          SourceLocation slStart = sharedVar->getLocStart();
          SourceLocation slEnd = sharedVar->getLocEnd();
          SourceManager *SM = Result.SourceManager;
          size_t repLength = SM->getCharacterData(slEnd) - SM->getCharacterData(slStart) + 1;
          SmallString<128> tmpData;
          StringRef varName = sharedVar->getNameAsString();
          StringRef repName = Twine("HIP_DYNAMIC_SHARED(" + typeName + ", " + varName + ")").toStringRef(tmpData);
          Replacement Rep(*SM, slStart, repLength, repName);
          FullSourceLoc fullSL(slStart, *SM);
          insertReplacement(Rep, fullSL);
          hipCounter counter = { "HIP_DYNAMIC_SHARED", CONV_MEM, API_RUNTIME };
          updateCounters(counter, refName.str());
        }
      }
      return true;
    }
    return false;
  }

  bool cudaParamDecl(const MatchFinder::MatchResult &Result) {
    if (const ParmVarDecl *paramDecl = Result.Nodes.getNodeAs<ParmVarDecl>("cudaParamDecl")) {
      QualType QT = paramDecl->getOriginalType().getUnqualifiedType();
      StringRef name = QT.getAsString();
      const Type *t = QT.getTypePtr();
      if (t->isStructureOrClassType()) {
        name = t->getAsCXXRecordDecl()->getName();
      }
      TypeLoc TL = paramDecl->getTypeSourceInfo()->getTypeLoc();
      SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
      SourceManager *SM = Result.SourceManager;
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        updateCounters(found->second, name.str());
        if (!found->second.unsupported) {
          StringRef repName = found->second.hipName;
          Replacement Rep(*SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, *SM);
          insertReplacement(Rep, fullSL);
        }
      } else {
        std::string msg = "the following reference is not handled: '" + name.str() + "' [param decl].";
        printHipifyMessage(*SM, sl, msg);
      }
      return true;
    }
    return false;
  }

  bool cudaParamDeclPtr(const MatchFinder::MatchResult &Result) {
    if (const ParmVarDecl *paramDeclPtr = Result.Nodes.getNodeAs<ParmVarDecl>("cudaParamDeclPtr")) {
      const Type *pt = paramDeclPtr->getType().getTypePtrOrNull();
      if (pt) {
        TypeLoc TL = paramDeclPtr->getTypeSourceInfo()->getTypeLoc();
        SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
        SourceManager *SM = Result.SourceManager;
        QualType QT = pt->getPointeeType();
        const Type *t = QT.getTypePtr();
        StringRef name = t->isStructureOrClassType()
          ? t->getAsCXXRecordDecl()->getName()
          : StringRef(QT.getAsString());
        const auto found = N.cuda2hipRename.find(name);
        if (found != N.cuda2hipRename.end()) {
          updateCounters(found->second, name.str());
          if (!found->second.unsupported) {
            StringRef repName = found->second.hipName;
            Replacement Rep(*SM, sl, name.size(), repName);
            FullSourceLoc fullSL(sl, *SM);
            insertReplacement(Rep, fullSL);
          }
        } else {
          std::string msg = "the following reference is not handled: '" + name.str() + "' [param decl ptr].";
          printHipifyMessage(*SM, sl, msg);
        }
      }
      return true;
    }
    return false;
  }

  bool unresolvedTemplateName(const MatchFinder::MatchResult &Result) {
    if (const FunctionTemplateDecl *templateDecl = Result.Nodes.getNodeAs<FunctionTemplateDecl>("unresolvedTemplateName")) {
      FunctionDecl *kernelDecl = templateDecl->getTemplatedDecl();
      convertKernelDecl(kernelDecl, Result);
      return true;
    }
    return false;
  }

  bool stringLiteral(const MatchFinder::MatchResult &Result) {
    if (const StringLiteral *sLiteral = Result.Nodes.getNodeAs<StringLiteral>("stringLiteral")) {
      if (sLiteral->getCharByteWidth() == 1) {
        StringRef s = sLiteral->getString();
        SourceManager *SM = Result.SourceManager;
        processString(s, *SM, sLiteral->getLocStart());
      }
      return true;
    }
    return false;
  }

public:
  Cuda2HipCallback(Replacements *Replace, ast_matchers::MatchFinder *parent, HipifyPPCallbacks *PPCallbacks, const std::string &mainFileName)
    : Cuda2Hip(Replace, mainFileName), owner(parent), PP(PPCallbacks) {
    PP->setMatch(this);
  }

  void run(const MatchFinder::MatchResult &Result) override {
    do {
      if (cudaCall(Result)) break;
      if (cudaBuiltin(Result)) break;
      if (cudaEnumConstantRef(Result)) break;
      if (cudaEnumConstantDecl(Result)) break;
      if (cudaTypedefVar(Result)) break;
      if (cudaTypedefVarPtr(Result)) break;
      if (cudaStructVar(Result)) break;
      if (cudaStructVarPtr(Result)) break;
      if (cudaStructSizeOf(Result)) break;
      if (cudaParamDecl(Result)) break;
      if (cudaParamDeclPtr(Result)) break;
      if (cudaLaunchKernel(Result)) break;
      if (cudaSharedIncompleteArrayVar(Result)) break;
      if (stringLiteral(Result)) break;
      if (unresolvedTemplateName(Result)) break;
      break;
    } while (false);
    insertHipHeaders(PP, *Result.SourceManager);
  }

private:
  ast_matchers::MatchFinder *owner;
  HipifyPPCallbacks *PP;
};

void HipifyPPCallbacks::handleEndSource() {
  insertHipHeaders(Match, *_sm);
}

} // end anonymous namespace

void addAllMatchers(ast_matchers::MatchFinder &Finder, Cuda2HipCallback *Callback) {
  Finder.addMatcher(callExpr(isExpansionInMainFile(),
                             callee(functionDecl(matchesName("cu.*"))))
                             .bind("cudaCall"),
                             Callback);
  Finder.addMatcher(cudaKernelCallExpr(isExpansionInMainFile()).bind("cudaLaunchKernel"), Callback);
  Finder.addMatcher(memberExpr(isExpansionInMainFile(),
                               hasObjectExpression(hasType(cxxRecordDecl(
                               matchesName("__cuda_builtin_")))))
                               .bind("cudaBuiltin"),
                               Callback);
  Finder.addMatcher(declRefExpr(isExpansionInMainFile(),
                                to(enumConstantDecl(
                                matchesName("cu.*|CU.*"))))
                                .bind("cudaEnumConstantRef"),
                                Callback);
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(enumDecl()))
                            .bind("cudaEnumConstantDecl"),
                            Callback);
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(typedefDecl(matchesName("cu.*|CU.*"))))
                            .bind("cudaTypedefVar"),
                            Callback);
  // Array of elements of typedef type. Example:
  // cudaStream_t streams[2];
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(arrayType(hasElementType(typedefType(
                            hasDeclaration(typedefDecl(matchesName("cu.*|CU.*"))))))))
                            .bind("cudaTypedefVar"),
                            Callback);
  // Pointer to typedef type. Examples:
  // 1.
  // cudaEvent_t *event = NULL;
  // typedef __device_builtin__ struct CUevent_st *cudaEvent_t;
  // 2.
  // CUevent *event = NULL;
  // typedef struct CUevent_st *CUevent;
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(pointsTo(typedefDecl(
                            matchesName("cu.*|CU.*")))))
                            .bind("cudaTypedefVarPtr"),
                            Callback);
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(cxxRecordDecl(matchesName("cu.*|CU.*"))))
                            .bind("cudaStructVar"),
                            Callback);
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(pointsTo(cxxRecordDecl(
                            matchesName("cu.*|CU.*")))))
                            .bind("cudaStructVarPtr"),
                            Callback);
  Finder.addMatcher(parmVarDecl(isExpansionInMainFile(),
                                hasType(namedDecl(matchesName("cu.*|CU.*"))))
                                .bind("cudaParamDecl"),
                                Callback);
  Finder.addMatcher(parmVarDecl(isExpansionInMainFile(),
                                hasType(pointsTo(namedDecl(
                                matchesName("cu.*|CU.*")))))
                                .bind("cudaParamDeclPtr"),
                                Callback);
  Finder.addMatcher(expr(isExpansionInMainFile(),
                         sizeOfExpr(hasArgumentOfType(
                         recordType(hasDeclaration(cxxRecordDecl(matchesName("cu.*|CU.*")))))))
                        .bind("cudaStructSizeOf"),
                         Callback);
  Finder.addMatcher(stringLiteral(isExpansionInMainFile()).bind("stringLiteral"),
                                  Callback);
  Finder.addMatcher(varDecl(isExpansionInMainFile(), allOf(
                            hasAttr(attr::CUDAShared),
                            hasType(incompleteArrayType())))
                           .bind("cudaSharedIncompleteArrayVar"),
                            Callback);
}

int64_t printStats(const std::string &csvFile, const std::string &srcFile,
                   HipifyPPCallbacks &PPCallbacks, Cuda2HipCallback &Callback,
                   uint64_t replacedBytes, uint64_t totalBytes, unsigned totalLines,
                   const std::chrono::steady_clock::time_point &start) {
  std::ofstream csv(csvFile, std::ios::app);
  int64_t sum = 0, sum_interm = 0;
  std::string str;
  const std::string hipify_info = "[HIPIFY] info: ", separator = ";";
  for (int i = 0; i < CONV_LAST; i++) {
    sum += Callback.countReps[i] + PPCallbacks.countReps[i];
  }
  int64_t sum_unsupported = 0;
  for (int i = 0; i < CONV_LAST; i++) {
    sum_unsupported += Callback.countRepsUnsupported[i] + PPCallbacks.countRepsUnsupported[i];
  }
  if (sum > 0 || sum_unsupported > 0) {
    str = "file \'" + srcFile + "\' statistics:\n";
    llvm::outs() << "\n" << hipify_info << str;
    csv << "\n" << str;
    str = "CONVERTED refs count";
    llvm::outs() << "  " << str << ": " << sum << "\n";
    csv << "\n" << str << separator << sum << "\n";
    str = "UNCONVERTED refs count";
    llvm::outs() << "  " << str << ": " << sum_unsupported << "\n";
    csv << str << separator << sum_unsupported << "\n";
    str = "CONVERSION %";
    long conv = 100 - std::lround(double(sum_unsupported*100)/double(sum + sum_unsupported));
    llvm::outs() << "  " << str << ": " << conv << "%\n";
    csv << str << separator << conv << "%\n";
    str = "REPLACED bytes";
    llvm::outs() << "  " << str << ": " << replacedBytes << "\n";
    csv << str << separator << replacedBytes << "\n";
    str = "TOTAL bytes";
    llvm::outs() << "  " << str << ": " << totalBytes << "\n";
    csv << str << separator << totalBytes << "\n";
    str = "CHANGED lines of code";
    unsigned changedLines = Callback.LOCs.size() + PPCallbacks.LOCs.size();
    llvm::outs() << "  " << str << ": " << changedLines << "\n";
    csv << str << separator << changedLines << "\n";
    str = "TOTAL lines of code";
    llvm::outs() << "  " << str << ": " << totalLines << "\n";
    csv << str << separator << totalLines << "\n";
    if (totalBytes > 0) {
      str = "CODE CHANGED (in bytes) %";
      conv = std::lround(double(replacedBytes * 100) / double(totalBytes));
      llvm::outs() << "  " << str << ": " << conv << "%\n";
      csv << str << separator << conv << "%\n";
    }
    if (totalLines > 0) {
      str = "CODE CHANGED (in lines) %";
      conv = std::lround(double(changedLines * 100) / double(totalLines));
      llvm::outs() << "  " << str << ": " << conv << "%\n";
      csv << str << separator << conv << "%\n";
    }
    typedef std::chrono::duration<double, std::milli> duration;
    duration elapsed = std::chrono::steady_clock::now() - start;
    str = "TIME ELAPSED s";
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << elapsed.count() / 1000;
    llvm::outs() << "  " << str << ": " << stream.str() << "\n";
    csv << str << separator << stream.str() << "\n";
  }
  if (sum > 0) {
    llvm::outs() << hipify_info << "CONVERTED refs by type:\n";
    csv << "\nCUDA ref type" << separator << "Count\n";
    for (int i = 0; i < CONV_LAST; i++) {
      sum_interm = Callback.countReps[i] + PPCallbacks.countReps[i];
      if (0 == sum_interm) {
        continue;
      }
      llvm::outs() << "  " << counterNames[i] << ": " << sum_interm << "\n";
      csv << counterNames[i] << separator << sum_interm << "\n";
    }
    llvm::outs() << hipify_info << "CONVERTED refs by API:\n";
    csv << "\nCUDA API" << separator << "Count\n";
    for (int i = 0; i < API_LAST; i++) {
      llvm::outs() << "  " << apiNames[i] << ": " << Callback.countApiReps[i] + PPCallbacks.countApiReps[i] << "\n";
      csv << apiNames[i] << separator << Callback.countApiReps[i] + PPCallbacks.countApiReps[i] << "\n";
    }
    for (const auto & it : PPCallbacks.cuda2hipConverted) {
      const auto found = Callback.cuda2hipConverted.find(it.first);
      if (found == Callback.cuda2hipConverted.end()) {
        Callback.cuda2hipConverted.insert(std::pair<std::string, uint64_t>(it.first, 1));
      } else {
        found->second += it.second;
      }
    }
    llvm::outs() << hipify_info << "CONVERTED refs by names:\n";
    csv << "\nCUDA ref name" << separator << "Count\n";
    for (const auto & it : Callback.cuda2hipConverted) {
      llvm::outs() << "  " << it.first << ": " << it.second << "\n";
      csv << it.first << separator << it.second << "\n";
    }
  }
  if (sum_unsupported > 0) {
    str = "UNCONVERTED refs by type:";
    llvm::outs() << hipify_info << str << "\n";
    csv << "\nUNCONVERTED CUDA ref type" << separator << "Count\n";
    for (int i = 0; i < CONV_LAST; i++) {
      sum_interm = Callback.countRepsUnsupported[i] + PPCallbacks.countRepsUnsupported[i];
      if (0 == sum_interm) {
        continue;
      }
      llvm::outs() << "  " << counterNames[i] << ": " << sum_interm << "\n";
      csv << counterNames[i] << separator << sum_interm << "\n";
    }
    llvm::outs() << hipify_info << "UNCONVERTED refs by API:\n";
    csv << "\nUNCONVERTED CUDA API" << separator << "Count\n";
    for (int i = 0; i < API_LAST; i++) {
      llvm::outs() << "  " << apiNames[i] << ": " << Callback.countApiRepsUnsupported[i] + PPCallbacks.countApiRepsUnsupported[i] << "\n";
      csv << apiNames[i] << separator << Callback.countApiRepsUnsupported[i] + PPCallbacks.countApiRepsUnsupported[i] << "\n";
    }
    for (const auto & it : PPCallbacks.cuda2hipUnconverted) {
      const auto found = Callback.cuda2hipUnconverted.find(it.first);
      if (found == Callback.cuda2hipUnconverted.end()) {
        Callback.cuda2hipUnconverted.insert(std::pair<std::string, uint64_t>(it.first, 1));
      } else {
        found->second += it.second;
      }
    }
    llvm::outs() << hipify_info << "UNCONVERTED refs by names:\n";
    csv << "\nUNCONVERTED CUDA ref name" << separator << "Count\n";
    for (const auto & it : Callback.cuda2hipUnconverted) {
      llvm::outs() << "  " << it.first << ": " << it.second << "\n";
      csv << it.first << separator << it.second << "\n";
    }
  }
  csv.close();
  return sum;
}

void printAllStats(const std::string &csvFile, int64_t totalFiles, int64_t convertedFiles,
                   uint64_t replacedBytes, uint64_t totalBytes, unsigned changedLines, unsigned totalLines,
                   const std::chrono::steady_clock::time_point &start) {
  std::ofstream csv(csvFile, std::ios::app);
  int64_t sum = 0, sum_interm = 0;
  std::string str;
  const std::string hipify_info = "[HIPIFY] info: ", separator = ";";
  for (int i = 0; i < CONV_LAST; i++) {
    sum += countRepsTotal[i];
  }
  int64_t sum_unsupported = 0;
  for (int i = 0; i < CONV_LAST; i++) {
    sum_unsupported += countRepsTotalUnsupported[i];
  }
  if (sum > 0 || sum_unsupported > 0) {
    str = "TOTAL statistics:\n";
    llvm::outs() << "\n" << hipify_info << str;
    csv << "\n" << str;
    str = "CONVERTED files";
    llvm::outs() << "  " << str << ": " << convertedFiles << "\n";
    csv << "\n" << str << separator << convertedFiles << "\n";
    str = "PROCESSED files";
    llvm::outs() << "  " << str << ": " << totalFiles << "\n";
    csv << str << separator << totalFiles << "\n";
    str = "CONVERTED refs count";
    llvm::outs() << "  " << str << ": " << sum << "\n";
    csv << str << separator << sum << "\n";
    str = "UNCONVERTED refs count";
    llvm::outs() << "  " << str << ": " << sum_unsupported << "\n";
    csv << str << separator << sum_unsupported << "\n";
    str = "CONVERSION %";
    long conv = 100 - std::lround(double(sum_unsupported * 100) / double(sum + sum_unsupported));
    llvm::outs() << "  " << str << ": " << conv << "%\n";
    csv << str << separator << conv << "%\n";
    str = "REPLACED bytes";
    llvm::outs() << "  " << str << ": " << replacedBytes << "\n";
    csv << str << separator << replacedBytes << "\n";
    str = "TOTAL bytes";
    llvm::outs() << "  " << str << ": " << totalBytes << "\n";
    csv << str << separator << totalBytes << "\n";
    str = "CHANGED lines of code";
    llvm::outs() << "  " << str << ": " << changedLines << "\n";
    csv << str << separator << changedLines << "\n";
    str = "TOTAL lines of code";
    llvm::outs() << "  " << str << ": " << totalLines << "\n";
    csv << str << separator << totalLines << "\n";
    if (totalBytes > 0) {
      str = "CODE CHANGED (in bytes) %";
      conv = std::lround(double(replacedBytes * 100) / double(totalBytes));
      llvm::outs() << "  " << str << ": " << conv << "%\n";
      csv << str << separator << conv << "%\n";
    }
    if (totalLines > 0) {
      str = "CODE CHANGED (in lines) %";
      conv = std::lround(double(changedLines * 100) / double(totalLines));
      llvm::outs() << "  " << str << ": " << conv << "%\n";
      csv << str << separator << conv << "%\n";
    }
    typedef std::chrono::duration<double, std::milli> duration;
    duration elapsed = std::chrono::steady_clock::now() - start;
    str = "TIME ELAPSED s";
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << elapsed.count() / 1000;
    llvm::outs() << "  " << str << ": " << stream.str() << "\n";
    csv << str << separator << stream.str() << "\n";
  }
  if (sum > 0) {
    llvm::outs() << hipify_info << "CONVERTED refs by type:\n";
    csv << "\nCUDA ref type" << separator << "Count\n";
    for (int i = 0; i < CONV_LAST; i++) {
      sum_interm = countRepsTotal[i];
      if (0 == sum_interm) {
        continue;
      }
      llvm::outs() << "  " << counterNames[i] << ": " << sum_interm << "\n";
      csv << counterNames[i] << separator << sum_interm << "\n";
    }
    llvm::outs() << hipify_info << "CONVERTED refs by API:\n";
    csv << "\nCUDA API" << separator << "Count\n";
    for (int i = 0; i < API_LAST; i++) {
      llvm::outs() << "  " << apiNames[i] << ": " << countApiRepsTotal[i] << "\n";
      csv << apiNames[i] << separator << countApiRepsTotal[i] << "\n";
    }
    llvm::outs() << hipify_info << "CONVERTED refs by names:\n";
    csv << "\nCUDA ref name" << separator << "Count\n";
    for (const auto & it : cuda2hipConvertedTotal) {
      llvm::outs() << "  " << it.first << ": " << it.second << "\n";
      csv << it.first << separator << it.second << "\n";
    }
  }
  if (sum_unsupported > 0) {
    str = "UNCONVERTED refs by type:";
    llvm::outs() << hipify_info << str << "\n";
    csv << "\nUNCONVERTED CUDA ref type" << separator << "Count\n";
    for (int i = 0; i < CONV_LAST; i++) {
      sum_interm = countRepsTotalUnsupported[i];
      if (0 == sum_interm) {
        continue;
      }
      llvm::outs() << "  " << counterNames[i] << ": " << sum_interm << "\n";
      csv << counterNames[i] << separator << sum_interm << "\n";
    }
    llvm::outs() << hipify_info << "UNCONVERTED refs by API:\n";
    csv << "\nUNCONVERTED CUDA API" << separator << "Count\n";
    for (int i = 0; i < API_LAST; i++) {
      llvm::outs() << "  " << apiNames[i] << ": " << countApiRepsTotalUnsupported[i] << "\n";
      csv << apiNames[i] << separator << countApiRepsTotalUnsupported[i] << "\n";
    }
    llvm::outs() << hipify_info << "UNCONVERTED refs by names:\n";
    csv << "\nUNCONVERTED CUDA ref name" << separator << "Count\n";
    for (const auto & it : cuda2hipUnconvertedTotal) {
      llvm::outs() << "  " << it.first << ": " << it.second << "\n";
      csv << it.first << separator << it.second << "\n";
    }
  }
  csv.close();
}

int main(int argc, const char **argv) {
  auto start = std::chrono::steady_clock::now();
  auto begin = start;
  llvm::sys::PrintStackTraceOnErrorSignal();
  CommonOptionsParser OptionsParser(argc, argv, ToolTemplateCategory, llvm::cl::OneOrMore);
  std::vector<std::string> fileSources = OptionsParser.getSourcePathList();
  std::string dst = OutputFilename;
  if (!dst.empty() && fileSources.size() > 1) {
    llvm::errs() << "[HIPIFY] conflict: -o and multiple source files are specified.\n";
    return 1;
  }
  if (NoOutput) {
    if (Inplace) {
      llvm::errs() << "[HIPIFY] conflict: both -no-output and -inplace options are specified.\n";
      return 1;
    }
    if (!dst.empty()) {
      llvm::errs() << "[HIPIFY] conflict: both -no-output and -o options are specified.\n";
      return 1;
    }
  }
  if (Examine) {
    NoOutput = PrintStats = true;
  }
  int Result = 0;
  std::string csv;
  if (!OutputStatsFilename.empty()) {
    csv = OutputStatsFilename;
  } else {
    csv = "hipify_stats.csv";
  }
  size_t filesTranslated = fileSources.size();
  uint64_t repBytesTotal = 0;
  uint64_t bytesTotal = 0;
  unsigned changedLinesTotal = 0;
  unsigned linesTotal = 0;
  if (PrintStats && filesTranslated > 1) {
    std::remove(csv.c_str());
  }
  for (const auto & src : fileSources) {
    if (dst.empty()) {
      dst = src;
      if (!Inplace) {
        size_t pos = dst.rfind(".");
        if (pos != std::string::npos && pos + 1 < dst.size()) {
          dst = dst.substr(0, pos) + ".hip." + dst.substr(pos + 1, dst.size() - pos - 1);
        } else {
          dst += ".hip.cu";
        }
      }
    } else {
      if (Inplace) {
        llvm::errs() << "[HIPIFY] conflict: both -o and -inplace options are specified.\n";
        return 1;
      }
      dst += ".hip";
    }
    // backup source file since tooling may change "inplace"
    if (!NoBackup || !Inplace) {
      std::ifstream source(src, std::ios::binary);
      std::ofstream dest(Inplace ? dst + ".prehip" : dst, std::ios::binary);
      dest << source.rdbuf();
      source.close();
      dest.close();
    }
    RefactoringTool Tool(OptionsParser.getCompilations(), dst);
    ast_matchers::MatchFinder Finder;
    HipifyPPCallbacks PPCallbacks(&Tool.getReplacements(), src);
    Cuda2HipCallback Callback(&Tool.getReplacements(), &Finder, &PPCallbacks, src);

    addAllMatchers(Finder, &Callback);

    auto action = newFrontendActionFactory(&Finder, &PPCallbacks);
    std::vector<const char*> compilationStages;
    compilationStages.push_back("--cuda-host-only");

    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(compilationStages[0], ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-std=c++11"));
#if defined(HIPIFY_CLANG_RES)
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-resource-dir=" HIPIFY_CLANG_RES));
#endif
    Tool.appendArgumentsAdjuster(getClangSyntaxOnlyAdjuster());
    Result += Tool.run(action.get());
    Tool.clearArgumentsAdjusters();

    LangOptions DefaultLangOptions;
    IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
    TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
    DiagnosticsEngine Diagnostics(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts, &DiagnosticPrinter, false);

    uint64_t repBytes = 0;
    uint64_t bytes = 0;
    unsigned lines = 0;
    SourceManager SM(Diagnostics, Tool.getFiles());
    if (PrintStats) {
      DEBUG(dbgs() << "Replacements collected by the tool:\n");
      for (const auto &r : Tool.getReplacements()) {
        DEBUG(dbgs() << r.toString() << "\n");
        repBytes += r.getLength();
      }
      std::ifstream src_file(dst, std::ios::binary | std::ios::ate);
      src_file.clear();
      src_file.seekg(0);
      lines = std::count(std::istreambuf_iterator<char>(src_file), std::istreambuf_iterator<char>(), '\n');
      bytes = src_file.tellg();
    }
    Rewriter Rewrite(SM, DefaultLangOptions);
    if (!Tool.applyAllReplacements(Rewrite)) {
      DEBUG(dbgs() << "Skipped some replacements.\n");
    }
    if (!NoOutput) {
      Result += Rewrite.overwriteChangedFiles();
    }
    if (!Inplace && !NoOutput) {
      size_t pos = dst.rfind(".");
      if (pos != std::string::npos) {
        rename(dst.c_str(), dst.substr(0, pos).c_str());
      }
    }
    if (NoOutput) {
      remove(dst.c_str());
    }
    if (PrintStats) {
      if (fileSources.size() == 1) {
        if (OutputStatsFilename.empty()) {
          csv = dst + ".csv";
        }
        std::remove(csv.c_str());
      }
      if (0 == printStats(csv, src, PPCallbacks, Callback, repBytes, bytes, lines, start)) {
        filesTranslated--;
      }
      start = std::chrono::steady_clock::now();
      repBytesTotal += repBytes;
      bytesTotal += bytes;
      changedLinesTotal += PPCallbacks.LOCs.size() + Callback.LOCs.size();
      linesTotal += lines;
    }
    dst.clear();
  }
  if (PrintStats && fileSources.size() > 1) {
    printAllStats(csv, fileSources.size(), filesTranslated, repBytesTotal, bytesTotal, changedLinesTotal, linesTotal, begin);
  }
  return Result;
}
