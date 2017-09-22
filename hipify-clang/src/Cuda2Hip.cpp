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
  CONV_VERSION = 0,
  CONV_INIT,
  CONV_DEVICE,
  CONV_MEM,
  CONV_KERN,
  CONV_COORD_FUNC,
  CONV_MATH_FUNC,
  CONV_SPECIAL_FUNC,
  CONV_STREAM,
  CONV_EVENT,
  CONV_OCCUPANCY,
  CONV_CONTEXT,
  CONV_PEER,
  CONV_MODULE,
  CONV_CACHE,
  CONV_EXEC,
  CONV_ERROR,
  CONV_DEF,
  CONV_TEX,
  CONV_GL,
  CONV_GRAPHICS,
  CONV_SURFACE,
  CONV_JIT,
  CONV_D3D9,
  CONV_D3D10,
  CONV_D3D11,
  CONV_VDPAU,
  CONV_EGL,
  CONV_THREAD,
  CONV_OTHER,
  CONV_INCLUDE,
  CONV_INCLUDE_CUDA_MAIN_H,
  CONV_TYPE,
  CONV_LITERAL,
  CONV_NUMERIC_LITERAL,
  CONV_LAST
};

const char *counterNames[CONV_LAST] = {
    "version",      "init",     "device",  "mem",       "kern",        "coord_func", "math_func",
    "special_func", "stream",   "event",   "occupancy", "ctx",         "peer",       "module",
    "cache",        "exec",     "err",     "def",       "tex",         "gl",         "graphics",
    "surface",      "jit",      "d3d9",    "d3d10",     "d3d11",       "vdpau",      "egl",
    "thread",       "other",    "include", "include_cuda_main_header", "type",       "literal",
    "numeric_literal"};

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
//  cuda2hipRename["cudaError_enum"]                            = {"hipError_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["cudaError_t"]                               = {"hipError_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaError"]                                 = {"hipError_t", CONV_TYPE, API_RUNTIME};

    // CUDA Driver API error codes only
    cuda2hipRename["CUDA_ERROR_INVALID_CONTEXT"]                = {"hipErrorInvalidContext", CONV_TYPE, API_DRIVER};                                 // 201
    cuda2hipRename["CUDA_ERROR_CONTEXT_ALREADY_CURRENT"]        = {"hipErrorContextAlreadyCurrent", CONV_TYPE, API_DRIVER};                          // 202
    cuda2hipRename["CUDA_ERROR_ARRAY_IS_MAPPED"]                = {"hipErrorArrayIsMapped", CONV_TYPE, API_DRIVER};                                  // 207
    cuda2hipRename["CUDA_ERROR_ALREADY_MAPPED"]                 = {"hipErrorAlreadyMapped", CONV_TYPE, API_DRIVER};                                  // 208
    cuda2hipRename["CUDA_ERROR_ALREADY_ACQUIRED"]               = {"hipErrorAlreadyAcquired", CONV_TYPE, API_DRIVER};                                // 210
    cuda2hipRename["CUDA_ERROR_NOT_MAPPED"]                     = {"hipErrorNotMapped", CONV_TYPE, API_DRIVER};                                      // 211
    cuda2hipRename["CUDA_ERROR_NOT_MAPPED_AS_ARRAY"]            = {"hipErrorNotMappedAsArray", CONV_TYPE, API_DRIVER};                               // 212
    cuda2hipRename["CUDA_ERROR_NOT_MAPPED_AS_POINTER"]          = {"hipErrorNotMappedAsPointer", CONV_TYPE, API_DRIVER};                             // 213
    cuda2hipRename["CUDA_ERROR_CONTEXT_ALREADY_IN_USE"]         = {"hipErrorContextAlreadyInUse", CONV_TYPE, API_DRIVER};                            // 216
    cuda2hipRename["CUDA_ERROR_INVALID_SOURCE"]                 = {"hipErrorInvalidSource", CONV_TYPE, API_DRIVER};                                  // 300
    cuda2hipRename["CUDA_ERROR_FILE_NOT_FOUND"]                 = {"hipErrorFileNotFound", CONV_TYPE, API_DRIVER};                                   // 301
    cuda2hipRename["CUDA_ERROR_NOT_FOUND"]                      = {"hipErrorNotFound", CONV_TYPE, API_DRIVER};                                       // 500
    cuda2hipRename["CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"]  = {"hipErrorLaunchIncompatibleTexturing", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 703
    cuda2hipRename["CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE"]         = {"hipErrorPrimaryContextActive", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // 708
    cuda2hipRename["CUDA_ERROR_CONTEXT_IS_DESTROYED"]           = {"hipErrorContextIsDestroyed", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};            // 709
    cuda2hipRename["CUDA_ERROR_NOT_PERMITTED"]                  = {"hipErrorNotPermitted", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                  // 800
    cuda2hipRename["CUDA_ERROR_NOT_SUPPORTED"]                  = {"hipErrorNotSupported", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                  // 801

    // CUDA RT API error code only
    cuda2hipRename["cudaErrorMissingConfiguration"]             = {"hipErrorMissingConfiguration", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 1
    cuda2hipRename["cudaErrorPriorLaunchFailure"]               = {"hipErrorPriorLaunchFailure", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 5
    cuda2hipRename["cudaErrorInvalidDeviceFunction"]            = {"hipErrorInvalidDeviceFunction", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};        // 8
    cuda2hipRename["cudaErrorInvalidConfiguration"]             = {"hipErrorInvalidConfiguration", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 9
    cuda2hipRename["cudaErrorInvalidPitchValue"]                = {"hipErrorInvalidPitchValue", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};            // 12
    cuda2hipRename["cudaErrorInvalidSymbol"]                    = {"hipErrorInvalidSymbol", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                // 13
    cuda2hipRename["cudaErrorInvalidHostPointer"]               = {"hipErrorInvalidHostPointer", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 16
    cuda2hipRename["cudaErrorInvalidDevicePointer"]             = {"hipErrorInvalidDevicePointer", CONV_TYPE, API_RUNTIME};                          // 17
    cuda2hipRename["cudaErrorInvalidTexture"]                   = {"hipErrorInvalidTexture", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};               // 18
    cuda2hipRename["cudaErrorInvalidTextureBinding"]            = {"hipErrorInvalidTextureBinding", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};        // 19
    cuda2hipRename["cudaErrorInvalidChannelDescriptor"]         = {"hipErrorInvalidChannelDescriptor", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};     // 20
    cuda2hipRename["cudaErrorInvalidMemcpyDirection"]           = {"hipErrorInvalidMemcpyDirection", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 21
    cuda2hipRename["cudaErrorAddressOfConstant"]                = {"hipErrorAddressOfConstant", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};            // 22
    cuda2hipRename["cudaErrorTextureFetchFailed"]               = {"hipErrorTextureFetchFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 23
    cuda2hipRename["cudaErrorTextureNotBound"]                  = {"hipErrorTextureNotBound", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};              // 24
    cuda2hipRename["cudaErrorSynchronizationError"]             = {"hipErrorSynchronizationError", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 25
    cuda2hipRename["cudaErrorInvalidFilterSetting"]             = {"hipErrorInvalidFilterSetting", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 26
    cuda2hipRename["cudaErrorInvalidNormSetting"]               = {"hipErrorInvalidNormSetting", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 27
    cuda2hipRename["cudaErrorMixedDeviceExecution"]             = {"hipErrorMixedDeviceExecution", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 28
    // Deprecated as of CUDA 4.1
    cuda2hipRename["cudaErrorNotYetImplemented"]                = {"hipErrorNotYetImplemented", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};            // 31
    // Deprecated as of CUDA 3.1
    cuda2hipRename["cudaErrorMemoryValueTooLarge"]              = {"hipErrorMemoryValueTooLarge", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 32
    cuda2hipRename["cudaErrorInsufficientDriver"]               = {"hipErrorInsufficientDriver", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 35
    cuda2hipRename["cudaErrorSetOnActiveProcess"]               = {"hipErrorSetOnActiveProcess", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 36
    cuda2hipRename["cudaErrorInvalidSurface"]                   = {"hipErrorInvalidSurface", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};               // 37
    cuda2hipRename["cudaErrorDuplicateVariableName"]            = {"hipErrorDuplicateVariableName", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};        // 43
    cuda2hipRename["cudaErrorDuplicateTextureName"]             = {"hipErrorDuplicateTextureName", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 44
    cuda2hipRename["cudaErrorDuplicateSurfaceName"]             = {"hipErrorDuplicateSurfaceName", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 45
    cuda2hipRename["cudaErrorDevicesUnavailable"]               = {"hipErrorDevicesUnavailable", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 46
    cuda2hipRename["cudaErrorIncompatibleDriverContext"]        = {"hipErrorIncompatibleDriverContext", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};    // 49
    cuda2hipRename["cudaErrorDeviceAlreadyInUse"]               = {"hipErrorDeviceAlreadyInUse", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 54
    cuda2hipRename["cudaErrorLaunchMaxDepthExceeded"]           = {"hipErrorLaunchMaxDepthExceeded", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 65
    cuda2hipRename["cudaErrorLaunchFileScopedTex"]              = {"hipErrorLaunchFileScopedTex", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 66
    cuda2hipRename["cudaErrorLaunchFileScopedSurf"]             = {"hipErrorLaunchFileScopedSurf", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 67
    cuda2hipRename["cudaErrorSyncDepthExceeded"]                = {"hipErrorSyncDepthExceeded", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};            // 68
    cuda2hipRename["cudaErrorLaunchPendingCountExceeded"]       = {"hipErrorLaunchPendingCountExceeded", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};   // 69
    cuda2hipRename["cudaErrorNotPermitted"]                     = {"hipErrorNotPermitted", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 70
    cuda2hipRename["cudaErrorNotSupported"]                     = {"hipErrorNotSupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 71
    cuda2hipRename["cudaErrorStartupFailure"]                   = {"hipErrorStartupFailure", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};               // 0x7f
    // Deprecated as of CUDA 4.1
    cuda2hipRename["cudaErrorApiFailureBase"]                   = {"hipErrorApiFailureBase", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};               // 10000

    cuda2hipRename["CUDA_SUCCESS"]                              = {"hipSuccess", CONV_TYPE, API_DRIVER};                                             // 0
    cuda2hipRename["cudaSuccess"]                               = {"hipSuccess", CONV_TYPE, API_RUNTIME};                                            // 0

    cuda2hipRename["CUDA_ERROR_INVALID_VALUE"]                  = {"hipErrorInvalidValue", CONV_TYPE, API_DRIVER};                                   // 1
    cuda2hipRename["cudaErrorInvalidValue"]                     = {"hipErrorInvalidValue", CONV_TYPE, API_RUNTIME};                                  // 11

    cuda2hipRename["CUDA_ERROR_OUT_OF_MEMORY"]                  = {"hipErrorMemoryAllocation", CONV_TYPE, API_DRIVER};                               // 2
    cuda2hipRename["cudaErrorMemoryAllocation"]                 = {"hipErrorMemoryAllocation", CONV_TYPE, API_RUNTIME};                              // 2

    cuda2hipRename["CUDA_ERROR_NOT_INITIALIZED"]                = {"hipErrorNotInitialized", CONV_TYPE, API_DRIVER};                                 // 3
    cuda2hipRename["cudaErrorInitializationError"]              = {"hipErrorInitializationError", CONV_TYPE, API_RUNTIME};                           // 3

    cuda2hipRename["CUDA_ERROR_DEINITIALIZED"]                  = {"hipErrorDeinitialized", CONV_TYPE, API_DRIVER};                                  // 4
    // TODO: double check, that these errors match
    cuda2hipRename["cudaErrorCudartUnloading"]                  = {"hipErrorDeinitialized", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                // 29

    cuda2hipRename["CUDA_ERROR_PROFILER_DISABLED"]              = {"hipErrorProfilerDisabled", CONV_TYPE, API_DRIVER};                               // 5
    cuda2hipRename["cudaErrorProfilerDisabled"]                 = {"hipErrorProfilerDisabled", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};             // 55

    cuda2hipRename["CUDA_ERROR_PROFILER_NOT_INITIALIZED"]       = {"hipErrorProfilerNotInitialized", CONV_TYPE, API_DRIVER};                         // 6
    // Deprecated as of CUDA 5.0
    cuda2hipRename["cudaErrorProfilerNotInitialized"]           = {"hipErrorProfilerNotInitialized", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 56

    cuda2hipRename["CUDA_ERROR_PROFILER_ALREADY_STARTED"]       = {"hipErrorProfilerAlreadyStarted", CONV_TYPE, API_DRIVER};                         // 7
    // Deprecated as of CUDA 5.0
    cuda2hipRename["cudaErrorProfilerAlreadyStarted"]           = {"hipErrorProfilerAlreadyStarted", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 57

    cuda2hipRename["CUDA_ERROR_PROFILER_ALREADY_STOPPED"]       = {"hipErrorProfilerAlreadyStopped", CONV_TYPE, API_DRIVER};                         // 8
    // Deprecated as of CUDA 5.0
    cuda2hipRename["cudaErrorProfilerAlreadyStopped"]           = {"hipErrorProfilerAlreadyStopped", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 58

    cuda2hipRename["CUDA_ERROR_NO_DEVICE"]                      = {"hipErrorNoDevice", CONV_TYPE, API_DRIVER};                                       // 100
    cuda2hipRename["cudaErrorNoDevice"]                         = {"hipErrorNoDevice", CONV_TYPE, API_RUNTIME};                                      // 38

    cuda2hipRename["CUDA_ERROR_INVALID_DEVICE"]                 = {"hipErrorInvalidDevice", CONV_TYPE, API_DRIVER};                                  // 101
    cuda2hipRename["cudaErrorInvalidDevice"]                    = {"hipErrorInvalidDevice", CONV_TYPE, API_RUNTIME};                                 // 10

    cuda2hipRename["CUDA_ERROR_INVALID_IMAGE"]                  = {"hipErrorInvalidImage", CONV_TYPE, API_DRIVER};                                   // 200
    cuda2hipRename["cudaErrorInvalidKernelImage"]               = {"hipErrorInvalidImage", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 47

    cuda2hipRename["CUDA_ERROR_MAP_FAILED"]                     = {"hipErrorMapFailed", CONV_TYPE, API_DRIVER};                                      // 205
    // TODO: double check, that these errors match
    cuda2hipRename["cudaErrorMapBufferObjectFailed"]            = {"hipErrorMapFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                    // 14

    cuda2hipRename["CUDA_ERROR_UNMAP_FAILED"]                   = {"hipErrorUnmapFailed", CONV_TYPE, API_DRIVER};                                    // 206
    // TODO: double check, that these errors match
    cuda2hipRename["cudaErrorUnmapBufferObjectFailed"]          = {"hipErrorUnmapFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                  // 15

    cuda2hipRename["CUDA_ERROR_NO_BINARY_FOR_GPU"]              = {"hipErrorNoBinaryForGpu", CONV_TYPE, API_DRIVER};                                 // 209
    cuda2hipRename["cudaErrorNoKernelImageForDevice"]           = {"hipErrorNoBinaryForGpu", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};               // 48

    cuda2hipRename["CUDA_ERROR_ECC_UNCORRECTABLE"]              = {"hipErrorECCNotCorrectable", CONV_TYPE, API_DRIVER};                              // 214
    cuda2hipRename["cudaErrorECCUncorrectable"]                 = {"hipErrorECCNotCorrectable", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};            // 39

    cuda2hipRename["CUDA_ERROR_UNSUPPORTED_LIMIT"]              = {"hipErrorUnsupportedLimit", CONV_TYPE, API_DRIVER};                               // 215
    cuda2hipRename["cudaErrorUnsupportedLimit"]                 = {"hipErrorUnsupportedLimit", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};             // 42

    cuda2hipRename["CUDA_ERROR_PEER_ACCESS_UNSUPPORTED"]        = {"hipErrorPeerAccessUnsupported", CONV_TYPE, API_DRIVER};                          // 217
    cuda2hipRename["cudaErrorPeerAccessUnsupported"]            = {"hipErrorPeerAccessUnsupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};        // 64

    cuda2hipRename["CUDA_ERROR_INVALID_PTX"]                    = {"hipErrorInvalidKernelFile", CONV_TYPE, API_DRIVER};                              // 218
    cuda2hipRename["cudaErrorInvalidPtx"]                       = {"hipErrorInvalidKernelFile", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};            // 78

    cuda2hipRename["CUDA_ERROR_INVALID_GRAPHICS_CONTEXT"]       = {"hipErrorInvalidGraphicsContext", CONV_TYPE, API_DRIVER};                         // 219
    cuda2hipRename["cudaErrorInvalidGraphicsContext"]           = {"hipErrorInvalidGraphicsContext", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 79

    cuda2hipRename["CUDA_ERROR_NVLINK_UNCORRECTABLE"]           = {"hipErrorNvlinkUncorrectable", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 220 [CUDA 8.0.44]
    cuda2hipRename["cudaErrorNvlinkUncorrectable"]              = {"hipErrorNvlinkUncorrectable", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 80  [CUDA 8.0.44]

    cuda2hipRename["CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND"] = {"hipErrorSharedObjectSymbolNotFound", CONV_TYPE, API_DRIVER};                     // 302
    cuda2hipRename["cudaErrorSharedObjectSymbolNotFound"]       = {"hipErrorSharedObjectSymbolNotFound", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};   // 40

    cuda2hipRename["CUDA_ERROR_SHARED_OBJECT_INIT_FAILED"]      = {"hipErrorSharedObjectInitFailed", CONV_TYPE, API_DRIVER};                         // 303
    cuda2hipRename["cudaErrorSharedObjectInitFailed"]           = {"hipErrorSharedObjectInitFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 41

    cuda2hipRename["CUDA_ERROR_OPERATING_SYSTEM"]               = {"hipErrorOperatingSystem", CONV_TYPE, API_DRIVER};                                // 304
    cuda2hipRename["cudaErrorOperatingSystem"]                  = {"hipErrorOperatingSystem", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};              // 63

    cuda2hipRename["CUDA_ERROR_INVALID_HANDLE"]                 = {"hipErrorInvalidResourceHandle", CONV_TYPE, API_DRIVER};                          // 400
    cuda2hipRename["cudaErrorInvalidResourceHandle"]            = {"hipErrorInvalidResourceHandle", CONV_TYPE, API_RUNTIME};                         // 33

    cuda2hipRename["CUDA_ERROR_NOT_READY"]                      = {"hipErrorNotReady", CONV_TYPE, API_DRIVER};                                       // 600
    cuda2hipRename["cudaErrorNotReady"]                         = {"hipErrorNotReady", CONV_TYPE, API_RUNTIME};                                      // 34

    cuda2hipRename["CUDA_ERROR_ILLEGAL_ADDRESS"]                = {"hipErrorIllegalAddress", CONV_TYPE, API_DRIVER};                                 // 700
    cuda2hipRename["cudaErrorIllegalAddress"]                   = {"hipErrorIllegalAddress", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};               // 77

    cuda2hipRename["CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"]        = {"hipErrorLaunchOutOfResources", CONV_TYPE, API_DRIVER};                           // 701
    cuda2hipRename["cudaErrorLaunchOutOfResources"]             = {"hipErrorLaunchOutOfResources", CONV_TYPE, API_RUNTIME};                          // 7

    cuda2hipRename["CUDA_ERROR_LAUNCH_TIMEOUT"]                 = {"hipErrorLaunchTimeOut", CONV_TYPE, API_DRIVER};                                  // 702
    cuda2hipRename["cudaErrorLaunchTimeout"]                    = {"hipErrorLaunchTimeOut", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                // 6

    cuda2hipRename["CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED"]    = {"hipErrorPeerAccessAlreadyEnabled", CONV_TYPE, API_DRIVER};                       // 704
    cuda2hipRename["cudaErrorPeerAccessAlreadyEnabled"]         = {"hipErrorPeerAccessAlreadyEnabled", CONV_TYPE, API_RUNTIME};                      // 50

    cuda2hipRename["CUDA_ERROR_PEER_ACCESS_NOT_ENABLED"]        = {"hipErrorPeerAccessNotEnabled", CONV_TYPE, API_DRIVER};                           // 705
    cuda2hipRename["cudaErrorPeerAccessNotEnabled"]             = {"hipErrorPeerAccessNotEnabled", CONV_TYPE, API_RUNTIME};                          // 51

    cuda2hipRename["CUDA_ERROR_ASSERT"]                         = {"hipErrorAssert", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                        // 710
    cuda2hipRename["cudaErrorAssert"]                           = {"hipErrorAssert", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                       // 59

    cuda2hipRename["CUDA_ERROR_TOO_MANY_PEERS"]                 = {"hipErrorTooManyPeers", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                  // 711
    cuda2hipRename["cudaErrorTooManyPeers"]                     = {"hipErrorTooManyPeers", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 60

    cuda2hipRename["CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED"] = {"hipErrorHostMemoryAlreadyRegistered", CONV_TYPE, API_DRIVER};                    // 712
    cuda2hipRename["cudaErrorHostMemoryAlreadyRegistered"]      = {"hipErrorHostMemoryAlreadyRegistered", CONV_TYPE, API_RUNTIME};                   // 61

    cuda2hipRename["CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED"]     = {"hipErrorHostMemoryNotRegistered", CONV_TYPE, API_DRIVER};                        // 713
    cuda2hipRename["cudaErrorHostMemoryNotRegistered"]          = {"hipErrorHostMemoryNotRegistered", CONV_TYPE, API_RUNTIME};                       // 62

    cuda2hipRename["CUDA_ERROR_HARDWARE_STACK_ERROR"]           = {"hipErrorHardwareStackError", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};            // 714
    cuda2hipRename["cudaErrorHardwareStackError"]               = {"hipErrorHardwareStackError", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 72

    cuda2hipRename["CUDA_ERROR_ILLEGAL_INSTRUCTION"]            = {"hipErrorIllegalInstruction", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};            // 715
    cuda2hipRename["cudaErrorIllegalInstruction"]               = {"hipErrorIllegalInstruction", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 73

    cuda2hipRename["CUDA_ERROR_MISALIGNED_ADDRESS"]             = {"hipErrorMisalignedAddress", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};             // 716
    cuda2hipRename["cudaErrorMisalignedAddress"]                = {"hipErrorMisalignedAddress", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};            // 74

    cuda2hipRename["CUDA_ERROR_INVALID_ADDRESS_SPACE"]          = {"hipErrorInvalidAddressSpace", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 717
    cuda2hipRename["cudaErrorInvalidAddressSpace"]              = {"hipErrorInvalidAddressSpace", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 75

    cuda2hipRename["CUDA_ERROR_INVALID_PC"]                     = {"hipErrorInvalidPc", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                     // 718
    cuda2hipRename["cudaErrorInvalidPc"]                        = {"hipErrorInvalidPc", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                    // 76

    cuda2hipRename["CUDA_ERROR_LAUNCH_FAILED"]                  = {"hipErrorLaunchFailure", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 719
    cuda2hipRename["cudaErrorLaunchFailure"]                    = {"hipErrorLaunchFailure", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                // 4

    cuda2hipRename["CUDA_ERROR_UNKNOWN"]                        = {"hipErrorUnknown", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                       // 999
    cuda2hipRename["cudaErrorUnknown"]                          = {"hipErrorUnknown", CONV_TYPE, API_RUNTIME};                                       // 30

    ///////////////////////////// CUDA DRIVER API /////////////////////////////
    // structs
    cuda2hipRename["CUDA_ARRAY3D_DESCRIPTOR"]                                     = {"HIP_ARRAY3D_DESCRIPTOR", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_ARRAY_DESCRIPTOR"]                                       = {"HIP_ARRAY_DESCRIPTOR", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_MEMCPY2D"]                                               = {"HIP_MEMCPY2D", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_MEMCPY3D"]                                               = {"HIP_MEMCPY3D", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_MEMCPY3D_PEER"]                                          = {"HIP_MEMCPY3D_PEER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_POINTER_ATTRIBUTE_P2P_TOKENS"]                           = {"HIP_POINTER_ATTRIBUTE_P2P_TOKENS", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_RESOURCE_DESC"]                                          = {"HIP_RESOURCE_DESC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CUDA_RESOURCE_VIEW_DESC"]                                     = {"HIP_RESOURCE_VIEW_DESC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};

    cuda2hipRename["CUipcEventHandle"]                                            = {"hipIpcEventHandle", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CUipcMemHandle"]                                              = {"hipIpcMemHandle", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};

    cuda2hipRename["CUaddress_mode"]                                              = {"hipAddress_mode", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TR_ADDRESS_MODE_WRAP"]                                     = {"HIP_TR_ADDRESS_MODE_WRAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};      // 0
    cuda2hipRename["CU_TR_ADDRESS_MODE_CLAMP"]                                    = {"HIP_TR_ADDRESS_MODE_CLAMP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};     // 1
    cuda2hipRename["CU_TR_ADDRESS_MODE_MIRROR"]                                   = {"HIP_TR_ADDRESS_MODE_MIRROR", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};    // 2
    cuda2hipRename["CU_TR_ADDRESS_MODE_BORDER"]                                   = {"HIP_TR_ADDRESS_MODE_BORDER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};    // 3

    cuda2hipRename["CUarray_cubemap_face"]                                        = {"hipArray_cubemap_face", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_CUBEMAP_FACE_POSITIVE_X"]                                  = {"HIP_CUBEMAP_FACE_POSITIVE_X", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 0x00
    cuda2hipRename["CU_CUBEMAP_FACE_NEGATIVE_X"]                                  = {"HIP_CUBEMAP_FACE_NEGATIVE_X", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 0x01
    cuda2hipRename["CU_CUBEMAP_FACE_POSITIVE_Y"]                                  = {"HIP_CUBEMAP_FACE_POSITIVE_Y", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 0x02
    cuda2hipRename["CU_CUBEMAP_FACE_NEGATIVE_Y"]                                  = {"HIP_CUBEMAP_FACE_NEGATIVE_Y", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 0x03
    cuda2hipRename["CU_CUBEMAP_FACE_POSITIVE_Z"]                                  = {"HIP_CUBEMAP_FACE_POSITIVE_Z", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 0x04
    cuda2hipRename["CU_CUBEMAP_FACE_NEGATIVE_Z"]                                  = {"HIP_CUBEMAP_FACE_NEGATIVE_Z", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 0x05

    cuda2hipRename["CUarray_format"]                                              = {"hipArray_format", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_AD_FORMAT_UNSIGNED_INT8"]                                  = {"HIP_AD_FORMAT_UNSIGNED_INT8", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 0x01
    cuda2hipRename["CU_AD_FORMAT_UNSIGNED_INT16"]                                 = {"HIP_AD_FORMAT_UNSIGNED_INT16", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};  // 0x02
    cuda2hipRename["CU_AD_FORMAT_UNSIGNED_INT32"]                                 = {"HIP_AD_FORMAT_UNSIGNED_INT32", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};  // 0x03
    cuda2hipRename["CU_AD_FORMAT_SIGNED_INT8"]                                    = {"HIP_AD_FORMAT_SIGNED_INT8", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};     // 0x08
    cuda2hipRename["CU_AD_FORMAT_SIGNED_INT16"]                                   = {"HIP_AD_FORMAT_SIGNED_INT16", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};    // 0x09
    cuda2hipRename["CU_AD_FORMAT_SIGNED_INT32"]                                   = {"HIP_AD_FORMAT_SIGNED_INT32", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};    // 0x0a
    cuda2hipRename["CU_AD_FORMAT_HALF"]                                           = {"HIP_AD_FORMAT_HALF", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};            // 0x10
    cuda2hipRename["CU_AD_FORMAT_FLOAT"]                                          = {"HIP_AD_FORMAT_FLOAT", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 0x20
    // Compute mode
    cuda2hipRename["CUcomputemode"]                                               = {"hipComputemode", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                      // API_RUNTIME ANALOGUE (cudaComputeMode)
    cuda2hipRename["CU_COMPUTEMODE_DEFAULT"]                                      = {"hipComputeModeDefault", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // 0 // API_RUNTIME ANALOGUE (cudaComputeModeDefault = 0)
    cuda2hipRename["CU_COMPUTEMODE_EXCLUSIVE"]                                    = {"hipComputeModeExclusive", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};        // 1 // API_RUNTIME ANALOGUE (cudaComputeModeExclusive = 1)
    cuda2hipRename["CU_COMPUTEMODE_PROHIBITED"]                                   = {"hipComputeModeProhibited", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 2 // API_RUNTIME ANALOGUE (cudaComputeModeProhibited = 2)
    cuda2hipRename["CU_COMPUTEMODE_EXCLUSIVE_PROCESS"]                            = {"hipComputeModeExclusiveProcess", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}; // 3 // API_RUNTIME ANALOGUE (cudaComputeModeExclusiveProcess = 3)

    // unsupported yet by HIP [CUDA 8.0.44]
    // Memory advise values
    cuda2hipRename["CUmem_advise"]                                                = {"hipMemAdvise", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                  // API_RUNTIME ANALOGUE (cudaComputeMode)
    // cuda2hipRename["CUmem_advise_enum"]                                        = {"hipMemAdvise", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_MEM_ADVISE_SET_READ_MOSTLY"]                               = {"hipMemAdviseSetReadMostly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                // 1 // API_RUNTIME ANALOGUE (cudaMemAdviseSetReadMostly = 1)
    cuda2hipRename["CU_MEM_ADVISE_UNSET_READ_MOSTLY"]                             = {"hipMemAdviseUnsetReadMostly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};              // 2 // API_RUNTIME ANALOGUE (cudaMemAdviseUnsetReadMostly = 2)
    cuda2hipRename["CU_MEM_ADVISE_SET_PREFERRED_LOCATION"]                        = {"hipMemAdviseSetPreferredLocation", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 3 // API_RUNTIME ANALOGUE (cudaMemAdviseSetPreferredLocation = 3)
    cuda2hipRename["CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION"]                      = {"hipMemAdviseUnsetPreferredLocation", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 4 // API_RUNTIME ANALOGUE (cudaMemAdviseUnsetPreferredLocation = 4)
    cuda2hipRename["CU_MEM_ADVISE_SET_ACCESSED_BY"]                               = {"hipMemAdviseSetAccessedBy", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                // 5 // API_RUNTIME ANALOGUE (cudaMemAdviseSetAccessedBy = 5)
    cuda2hipRename["CU_MEM_ADVISE_UNSET_ACCESSED_BY"]                             = {"hipMemAdviseUnsetAccessedBy", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};              // 6 // API_RUNTIME ANALOGUE (cudaMemAdviseUnsetAccessedBy = 6)
    // CUmem_range_attribute
    cuda2hipRename["CUmem_range_attribute"]                                       = {"hipMemRangeAttribute", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                          // API_RUNTIME ANALOGUE (cudaMemRangeAttribute)
    // cuda2hipRename["CUmem_range_attribute_enum"]                               = {"hipMemRangeAttribute", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY"]                          = {"hipMemRangeAttributeReadMostly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 1 // API_RUNTIME ANALOGUE (cudaMemRangeAttributeReadMostly = 1)
    cuda2hipRename["CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION"]                   = {"hipMemRangeAttributePreferredLocation", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};    // 2 // API_RUNTIME ANALOGUE (cudaMemRangeAttributePreferredLocation = 2)
    cuda2hipRename["CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY"]                          = {"hipMemRangeAttributeAccessedBy", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 3 // API_RUNTIME ANALOGUE (cudaMemRangeAttributeAccessedBy = 3)
    cuda2hipRename["CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION"]               = {"hipMemRangeAttributeLastPrefetchLocation", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}; // 4 // API_RUNTIME ANALOGUE (cudaMemRangeAttributeLastPrefetchLocation = 4)

    // Context flags
    cuda2hipRename["CUctx_flags"]                                                  = {"hipCctx_flags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_CTX_SCHED_AUTO"]                                            = {"HIP_CTX_SCHED_AUTO", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 0x00
    cuda2hipRename["CU_CTX_SCHED_SPIN"]                                            = {"HIP_CTX_SCHED_SPIN", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 0x01
    cuda2hipRename["CU_CTX_SCHED_YIELD"]                                           = {"HIP_CTX_SCHED_YIELD", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // 0x02
    cuda2hipRename["CU_CTX_SCHED_BLOCKING_SYNC"]                                   = {"HIP_CTX_SCHED_BLOCKING_SYNC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};  // 0x04
    cuda2hipRename["CU_CTX_BLOCKING_SYNC"]                                         = {"HIP_CTX_BLOCKING_SYNC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};        // 0x04
    cuda2hipRename["CU_CTX_SCHED_MASK"]                                            = {"HIP_CTX_SCHED_MASK", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 0x07
    cuda2hipRename["CU_CTX_MAP_HOST"]                                              = {"HIP_CTX_MAP_HOST", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};             // 0x08
    cuda2hipRename["CU_CTX_LMEM_RESIZE_TO_MAX"]                                    = {"HIP_CTX_LMEM_RESIZE_TO_MAX", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 0x10
    cuda2hipRename["CU_CTX_FLAGS_MASK"]                                            = {"HIP_CTX_FLAGS_MASK", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 0x1f

    // Defines
    cuda2hipRename["CU_LAUNCH_PARAM_BUFFER_POINTER"]                              = {"HIP_LAUNCH_PARAM_BUFFER_POINTER", CONV_TYPE, API_DRIVER};                 // ((void*)0x01)
    cuda2hipRename["CU_LAUNCH_PARAM_BUFFER_SIZE"]                                 = {"HIP_LAUNCH_PARAM_BUFFER_SIZE", CONV_TYPE, API_DRIVER};                    // ((void*)0x02)
    cuda2hipRename["CU_LAUNCH_PARAM_END"]                                         = {"HIP_LAUNCH_PARAM_END", CONV_TYPE, API_DRIVER};                            // ((void*)0x00)
    cuda2hipRename["CU_IPC_HANDLE_SIZE"]                                          = {"HIP_LAUNCH_PARAM_END", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 64
    cuda2hipRename["CU_MEMHOSTALLOC_DEVICEMAP"]                                   = {"HIP_MEMHOSTALLOC_DEVICEMAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};     // 0x02
    cuda2hipRename["CU_MEMHOSTALLOC_PORTABLE"]                                    = {"HIP_MEMHOSTALLOC_PORTABLE", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};      // 0x01
    cuda2hipRename["CU_MEMHOSTALLOC_WRITECOMBINED"]                               = {"HIP_MEMHOSTALLOC_WRITECOMBINED", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}; // 0x04
    cuda2hipRename["CU_MEMHOSTREGISTER_DEVICEMAP"]                                = {"HIP_MEMHOSTREGISTER_DEVICEMAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};  // 0x02
    cuda2hipRename["CU_MEMHOSTREGISTER_IOMEMORY"]                                 = {"HIP_MEMHOSTREGISTER_IOMEMORY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 0x04
    cuda2hipRename["CU_MEMHOSTREGISTER_PORTABLE"]                                 = {"HIP_MEMHOSTREGISTER_PORTABLE", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 0x01
    cuda2hipRename["CU_PARAM_TR_DEFAULT"]                                         = {"HIP_PARAM_TR_DEFAULT", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // -1
    cuda2hipRename["CU_STREAM_LEGACY"]                                            = {"HIP_STREAM_LEGACY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};              // ((CUstream)0x1)
    cuda2hipRename["CU_STREAM_PER_THREAD"]                                        = {"HIP_STREAM_PER_THREAD", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // ((CUstream)0x2)
    cuda2hipRename["CU_TRSA_OVERRIDE_FORMAT"]                                     = {"HIP_TRSA_OVERRIDE_FORMAT", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 0x01
    cuda2hipRename["CU_TRSF_NORMALIZED_COORDINATES"]                              = {"HIP_TRSF_NORMALIZED_COORDINATES", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};// 0x02
    cuda2hipRename["CU_TRSF_READ_AS_INTEGER"]                                     = {"HIP_TRSF_READ_AS_INTEGER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 0x01
    cuda2hipRename["CU_TRSF_SRGB"]                                                = {"HIP_TRSF_SRGB", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                  // 0x10
    // Deprecated, use CUDA_ARRAY3D_LAYERED
    cuda2hipRename["CUDA_ARRAY3D_2DARRAY"]                                        = {"HIP_ARRAY3D_LAYERED", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};            // 0x01
    cuda2hipRename["CUDA_ARRAY3D_CUBEMAP"]                                        = {"HIP_ARRAY3D_CUBEMAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};            // 0x04
    cuda2hipRename["CUDA_ARRAY3D_DEPTH_TEXTURE"]                                  = {"HIP_ARRAY3D_DEPTH_TEXTURE", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};      // 0x10
    cuda2hipRename["CUDA_ARRAY3D_LAYERED"]                                        = {"HIP_ARRAY3D_LAYERED", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};            // 0x01
    cuda2hipRename["CUDA_ARRAY3D_SURFACE_LDST"]                                   = {"HIP_ARRAY3D_SURFACE_LDST", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 0x02
    cuda2hipRename["CUDA_ARRAY3D_TEXTURE_GATHER"]                                 = {"HIP_ARRAY3D_TEXTURE_GATHER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};     // 0x08
    cuda2hipRename["CUDA_VERSION"]                                                = {"HIP_VERSION", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                    // 7050

    // Types
    // NOTE: CUdevice might be changed to typedef int in the future.
    cuda2hipRename["CUdevice"]                                                    = {"hipDevice_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUdevice_attribute_enum"]                                     = {"hipDeviceAttribute_t", CONV_TYPE, API_DRIVER};                           // API_Runtime ANALOGUE (cudaDeviceAttr)
    cuda2hipRename["CUdevice_attribute"]                                          = {"hipDeviceAttribute_t", CONV_TYPE, API_DRIVER};                           // API_Runtime ANALOGUE (cudaDeviceAttr)
    cuda2hipRename["CUdeviceptr"]                                                 = {"hipDeviceptr_t", CONV_TYPE, API_DRIVER};
    // CUDA: "The types::CUarray and struct ::cudaArray * represent the same data type and may be used interchangeably by casting the two types between each other."
    //    typedef struct cudaArray  *cudaArray_t;
    //    typedef struct CUarray_st *CUarray;
    cuda2hipRename["CUarray_st"]                                                  = {"hipArray", CONV_TYPE, API_DRIVER};                                       // API_Runtime ANALOGUE (cudaArray)
    cuda2hipRename["CUarray"]                                                     = {"hipArray *", CONV_TYPE, API_DRIVER};                                     // API_Runtime ANALOGUE (cudaArray_t)

    // unsupported yet by HIP
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK"]                   = {"hipDeviceAttributeMaxThreadsPerBlock", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                //  1 // API_Runtime ANALOGUE (cudaDevAttrMaxThreadsPerBlock = 1)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X"]                         = {"hipDeviceAttributeMaxBlockDimX", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                      //  2 // API_Runtime ANALOGUE (cudaDevAttrMaxBlockDimX = 2)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y"]                         = {"hipDeviceAttributeMaxBlockDimY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                      //  3 // API_Runtime ANALOGUE (cudaDevAttrMaxBlockDimY = 3)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z"]                         = {"hipDeviceAttributeMaxBlockDimZ", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                      //  4 // API_Runtime ANALOGUE (cudaDevAttrMaxBlockDimZ = 4)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X"]                          = {"hipDeviceAttributeMaxGridDimX", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                       //  5 // API_Runtime ANALOGUE (cudaDevAttrMaxGridDimX =5)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y"]                          = {"hipDeviceAttributeMaxGridDimY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                       //  6 // API_Runtime ANALOGUE (cudaDevAttrMaxGridDimY = 6)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z"]                          = {"hipDeviceAttributeMaxGridDimZ", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                       //  7 // API_Runtime ANALOGUE (cudaDevAttrMaxGridDimZ - 7)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK"]             = {"hipDeviceAttributeMaxSharedMemoryPerBlock", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           //  8 // API_Runtime ANALOGUE (cudaDevAttrMaxSharedMemoryPerBlock = 8)
    // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK"]                 = {"hipDeviceAttributeMaxSharedMemoryPerBlock", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           //  8
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY"]                   = {"hipDeviceAttributeTotalConstantMemory", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};               //  9 // API_Runtime ANALOGUE (cudaDevAttrTotalConstantMemory = 9)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_WARP_SIZE"]                               = {"hipDeviceAttributeWarpSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                          // 10 // API_Runtime ANALOGUE (cudaDevAttrWarpSize = 10)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_PITCH"]                               = {"hipDeviceAttributeMaxPitch", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                          // 11 // API_Runtime ANALOGUE (cudaDevAttrMaxPitch = 11)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK"]                 = {"hipDeviceAttributeMaxRegistersPerBlock", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};              // 12 // API_Runtime ANALOGUE (cudaDevAttrMaxRegistersPerBlock = 12)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK"]                     = {"hipDeviceAttributeMaxRegistersPerBlock", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};              // 12
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_CLOCK_RATE"]                              = {"hipDeviceAttributeClockRate", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                         // 13 // API_Runtime ANALOGUE (cudaDevAttrMaxRegistersPerBlock = 13)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT"]                       = {"hipDeviceAttributeTextureAlignment", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                  // 14 // API_Runtime ANALOGUE (cudaDevAttrTextureAlignment = 14)
    // Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_GPU_OVERLAP"]                             = {"hipDeviceAttributeAsyncEngineCount", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                  // 15 // API_Runtime ANALOGUE (cudaDevAttrGpuOverlap = 15)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT"]                    = {"hipDeviceAttributeMultiprocessorCount", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};               // 16 // API_Runtime ANALOGUE (cudaDevAttrMultiProcessorCount = 16)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT"]                     = {"hipDeviceAttributeKernelExecTimeout", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 17 // API_Runtime ANALOGUE (cudaDevAttrKernelExecTimeout = 17)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_INTEGRATED"]                              = {"hipDeviceAttributeIntegrated", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                        // 18 // API_Runtime ANALOGUE (cudaDevAttrIntegrated = 18)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY"]                     = {"hipDeviceAttributeCanMapHostMemory", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                  // 19 // API_Runtime ANALOGUE (cudaDevAttrCanMapHostMemory = 19)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_COMPUTE_MODE"]                            = {"hipDeviceAttributeComputeMode", CONV_TYPE, API_DRIVER};                                        // 20 // API_Runtime ANALOGUE (cudaDevAttrComputeMode = 20)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH"]                 = {"hipDeviceAttributeMaxTexture1DWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 21 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture1DWidth = 21)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH"]                 = {"hipDeviceAttributeMaxTexture2DWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 22 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DWidth = 22)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT"]                = {"hipDeviceAttributeMaxTexture2DHeight", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                // 23 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DHeight = 23)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH"]                 = {"hipDeviceAttributeMaxTexture3DWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 24 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DWidth = 24)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT"]                = {"hipDeviceAttributeMaxTexture3DHeight", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                // 25 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DHeight = 25)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH"]                 = {"hipDeviceAttributeMaxTexture3DDepth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 26 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DDepth = 26)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH"]         = {"hipDeviceAttributeMaxTexture2DLayeredWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // 27 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredWidth = 27)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT"]        = {"hipDeviceAttributeMaxTexture2DLayeredHeight", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 28 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredHeight = 28)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS"]        = {"hipDeviceAttributeMaxTexture2DLayeredLayers", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 29 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredLayers = 29)
    // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH"]           = {"hipDeviceAttributeMaxTexture2DLayeredWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // 27 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredWidth = 27)
    // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT"]          = {"hipDeviceAttributeMaxTexture2DLayeredHeight", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 28 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredHeight = 28)
    // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES"]       = {"hipDeviceAttributeMaxTexture2DLayeredLayers", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 29 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredLayers = 29)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT"]                       = {"hipDeviceAttributeSurfaceAlignment", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                  // 30 // API_Runtime ANALOGUE (cudaDevAttrSurfaceAlignment = 30)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS"]                      = {"hipDeviceAttributeConcurrentKernels", CONV_TYPE, API_DRIVER};                                  // 31 // API_Runtime ANALOGUE (cudaDevAttrConcurrentKernels = 31)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_ECC_ENABLED"]                             = {"hipDeviceAttributeEccEnabled", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                        // 32 // API_Runtime ANALOGUE (cudaDevAttrEccEnabled = 32)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_PCI_BUS_ID"]                              = {"hipDeviceAttributePciBusId", CONV_TYPE, API_DRIVER};                                           // 33 // API_Runtime ANALOGUE (cudaDevAttrPciBusId = 33)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID"]                           = {"hipDeviceAttributePciDeviceId", CONV_TYPE, API_DRIVER};                                        // 34 // API_Runtime ANALOGUE (cudaDevAttrPciDeviceId = 34)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_TCC_DRIVER"]                              = {"hipDeviceAttributeTccDriver", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                         // 35 // API_Runtime ANALOGUE (cudaDevAttrTccDriver = 35)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE"]                       = {"hipDeviceAttributeMemoryClockRate", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                   // 36 // API_Runtime ANALOGUE (cudaDevAttrMemoryClockRate = 36)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH"]                 = {"hipDeviceAttributeMemoryBusWidth", CONV_TYPE, API_DRIVER};                                     // 37 // API_Runtime ANALOGUE (cudaDevAttrGlobalMemoryBusWidth = 37)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE"]                           = {"hipDeviceAttributeL2CacheSize", CONV_TYPE, API_DRIVER};                                        // 38 // API_Runtime ANALOGUE (cudaDevAttrL2CacheSize = 38)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR"]          = {"hipDeviceAttributeMaxThreadsPerMultiProcessor", CONV_TYPE, API_DRIVER};                        // 39 // API_Runtime ANALOGUE (cudaDevAttrMaxThreadsPerMultiProcessor = 39)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT"]                      = {"hipDeviceAttributeAsyncEngineCount", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                  // 40 // API_Runtime ANALOGUE (cudaDevAttrAsyncEngineCount = 40)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING"]                      = {"hipDeviceAttributeUnifiedAddressing", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 41 // API_Runtime ANALOGUE (cudaDevAttrUnifiedAddressing = 41)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH"]         = {"hipDeviceAttributeMaxTexture1DLayeredWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // 42 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture1DLayeredWidth = 42)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS"]        = {"hipDeviceAttributeMaxTexture1DLayeredLayers", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 43 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture1DLayeredLayers = 43)
    // deprecated, do not use
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER"]                        = {"hipDeviceAttributeCanTex2DGather", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                    // 44 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH"]          = {"hipDeviceAttributeMaxTexture2DGatherWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 45 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DGatherWidth = 45)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT"]         = {"hipDeviceAttributeMaxTexture2DGatherHeight", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // 46 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DGatherHeight = 46)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE"]       = {"hipDeviceAttributeMaxTexture3DWidthAlternate", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};        // 47 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DWidthAlt = 47)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE"]      = {"hipDeviceAttributeMaxTexture3DHeightAlternate", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 48 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DHeightAlt = 48)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE"]       = {"hipDeviceAttributeMaxTexture3DDepthAlternate", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};        // 49 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DDepthAlt = 49)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID"]                           = {"hipDeviceAttributePciDomainId", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                       // 50 // API_Runtime ANALOGUE (cudaDevAttrPciDomainId = 50)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT"]                 = {"hipDeviceAttributeTexturePitchAlignment", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};             // 51 // API_Runtime ANALOGUE (cudaDevAttrTexturePitchAlignment = 51)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH"]            = {"hipDeviceAttributeMaxTextureCubemapWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};            // 52 // API_Runtime ANALOGUE (cudaDevAttrMaxTextureCubemapWidth = 52)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH"]    = {"hipDeviceAttributeMaxTextureCubemapLayeredWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};     // 53 // API_Runtime ANALOGUE (cudaDevAttrMaxTextureCubemapLayeredWidth = 53)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS"]   = {"hipDeviceAttributeMaxTextureCubemapLayeredLayers", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};    // 54 // API_Runtime ANALOGUE (cudaDevAttrMaxTextureCubemapLayeredLayers = 54)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH"]                 = {"hipDeviceAttributeMaxSurface1DWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 55 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface1DWidth = 55)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH"]                 = {"hipDeviceAttributeMaxSurface2DWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 56 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface2DWidth = 56)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT"]                = {"hipDeviceAttributeMaxSurface2DHeight", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                // 57 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface2DHeight = 57)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH"]                 = {"hipDeviceAttributeMaxSurface3DWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 58 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface3DWidth = 58)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT"]                = {"hipDeviceAttributeMaxSurface3DHeight", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                // 59 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface3DHeight = 59)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH"]                 = {"hipDeviceAttributeMaxSurface3DDepth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 60 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface3DDepth = 60)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH"]         = {"hipDeviceAttributeMaxSurface1DLayeredWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // 61 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface1DLayeredWidth = 61)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS"]        = {"hipDeviceAttributeMaxSurface1DLayeredLayers", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 62 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface1DLayeredLayers = 62)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH"]         = {"hipDeviceAttributeMaxSurface2DLayeredWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // 63 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface2DLayeredWidth = 63)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT"]        = {"hipDeviceAttributeMaxSurface2DLayeredHeight", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 64 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface2DLayeredHeight = 64)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS"]        = {"hipDeviceAttributeMaxSurface2DLayeredLayers", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 65 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface2DLayeredLayers = 65)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH"]            = {"hipDeviceAttributeMaxSurfaceCubemapWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};            // 66 // API_Runtime ANALOGUE (cudaDevAttrMaxSurfaceCubemapWidth = 66)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH"]    = {"hipDeviceAttributeMaxSurfaceCubemapLayeredWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};     // 67 // API_Runtime ANALOGUE (cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS"]   = {"hipDeviceAttributeMaxSurfaceCubemapLayeredLayers", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};    // 68 // API_Runtime ANALOGUE (cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH"]          = {"hipDeviceAttributeMaxTexture1DLinearWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 69 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture1DLinearWidth = 69)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH"]          = {"hipDeviceAttributeMaxTexture2DLinearWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 70 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLinearWidth = 70)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT"]         = {"hipDeviceAttributeMaxTexture2DLinearHeight", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // 71 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLinearHeight = 71)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH"]          = {"hipDeviceAttributeMaxTexture2DLinearPitch", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 72 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLinearPitch = 72)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH"]       = {"hipDeviceAttributeMaxTexture2DMipmappedWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};        // 73 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DMipmappedWidth = 73)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT"]      = {"hipDeviceAttributeMaxTexture2DMipmappedHeight", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 74 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DMipmappedHeight = 74)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR"]                = {"hipDeviceAttributeComputeCapabilityMajor", CONV_TYPE, API_DRIVER};                             // 75 // API_Runtime ANALOGUE (cudaDevAttrComputeCapabilityMajor = 75)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR"]                = {"hipDeviceAttributeComputeCapabilityMinor", CONV_TYPE, API_DRIVER};                             // 76 // API_Runtime ANALOGUE (cudaDevAttrComputeCapabilityMinor = 76)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH"]       = {"hipDeviceAttributeMaxTexture1DMipmappedWidth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};        // 77 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture1DMipmappedWidth = 77)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED"]             = {"hipDeviceAttributeStreamPrioritiesSupported", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 78 // API_Runtime ANALOGUE (cudaDevAttrStreamPrioritiesSupported = 78)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED"]               = {"hipDeviceAttributeGlobalL1CacheSupported", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};            // 79 // API_Runtime ANALOGUE (cudaDevAttrGlobalL1CacheSupported = 79)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED"]                = {"hipDeviceAttributeLocalL1CacheSupported", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};             // 80 // API_Runtime ANALOGUE (cudaDevAttrLocalL1CacheSupported = 80)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR"]    = {"hipDeviceAttributeMaxSharedMemoryPerMultiprocessor", CONV_TYPE, API_DRIVER};                   // 81 // API_Runtime ANALOGUE (cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR"]        = {"hipDeviceAttributeMaxRegistersPerMultiprocessor", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};     // 82 // API_Runtime ANALOGUE (cudaDevAttrMaxRegistersPerMultiprocessor = 82)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY"]                          = {"hipDeviceAttributeManagedMemory", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                     // 83 // API_Runtime ANALOGUE (cudaDevAttrManagedMemory = 83)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD"]                         = {"hipDeviceAttributeIsMultiGpuBoard", CONV_TYPE, API_DRIVER};                                    // 84 // API_Runtime ANALOGUE (cudaDevAttrIsMultiGpuBoard = 84)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID"]                = {"hipDeviceAttributeMultiGpuBoardGroupId", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};              // 85 // API_Runtime ANALOGUE (cudaDevAttrMultiGpuBoardGroupID = 85)
    // unsupported yet by HIP [CUDA 8.0.44]
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED"]            = {"hipDeviceAttributeHostNativeAtomicSupported", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 86 // API_Runtime ANALOGUE (cudaDevAttrHostNativeAtomicSupported = 86)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO"]   = {"hipDeviceAttributeSingleToDoublePrecisionPerfRatio", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};  // 87 // API_Runtime ANALOGUE (cudaDevAttrSingleToDoublePrecisionPerfRatio = 87)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS"]                  = {"hipDeviceAttributePageableMemoryAccess", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};              // 88 // API_Runtime ANALOGUE (cudaDevAttrPageableMemoryAccess = 88)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS"]               = {"hipDeviceAttributeConcurrentManagedAccess", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 89 // API_Runtime ANALOGUE (cudaDevAttrConcurrentManagedAccess = 89)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED"]            = {"hipDeviceAttributeComputePreemptionSupported", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};        // 90 // API_Runtime ANALOGUE (cudaDevAttrComputePreemptionSupported = 90)
    cuda2hipRename["CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM"] = {"hipDeviceAttributeCanUseHostPointerForRegisteredMem", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}; // 91 // API_Runtime ANALOGUE (cudaDevAttrCanUseHostPointerForRegisteredMem = 91)

    cuda2hipRename["CU_DEVICE_ATTRIBUTE_MAX"]                                     = {"hipDeviceAttributeMax", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                               // 92 // API_Runtime ANALOGUE (no)

    cuda2hipRename["CUdevprop_st"]                                                = {"hipDeviceProp_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUdevprop"]                                                   = {"hipDeviceProp_t", CONV_TYPE, API_DRIVER};

    // TODO: Analogues enum is needed in HIP. Couldn't map enum to struct hipPointerAttribute_t.
    // TODO: Do for Pointer Attributes the same as for Device Attributes.
    // cuda2hipRename["CUpointer_attribute_enum"]                                 = {"hipPointerAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                 // API_Runtime ANALOGUE (no)
    // cuda2hipRename["CUpointer_attribute"]                                      = {"hipPointerAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_POINTER_ATTRIBUTE_CONTEXT"]                                = {"hipPointerAttributeContext", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                        // 1 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_POINTER_ATTRIBUTE_MEMORY_TYPE"]                            = {"hipPointerAttributeMemoryType", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                     // 2 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_POINTER_ATTRIBUTE_DEVICE_POINTER"]                         = {"hipPointerAttributeDevicePointer", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                  // 3 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_POINTER_ATTRIBUTE_HOST_POINTER"]                           = {"hipPointerAttributeHostPointer", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                    // 4 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_POINTER_ATTRIBUTE_P2P_TOKENS"]                             = {"hipPointerAttributeP2pTokens", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                      // 5 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_POINTER_ATTRIBUTE_SYNC_MEMOPS"]                            = {"hipPointerAttributeSyncMemops", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                     // 6 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_POINTER_ATTRIBUTE_BUFFER_ID"]                              = {"hipPointerAttributeBufferId", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                       // 7 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_POINTER_ATTRIBUTE_IS_MANAGED"]                             = {"hipPointerAttributeIsManaged", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                      // 8 // API_Runtime ANALOGUE (no)

    // pointer to CUfunc_st
    cuda2hipRename["CUfunction"]                                                  = {"hipFunction_t", CONV_TYPE, API_DRIVER};
    // TODO: move "typedef struct ihipModuleSymbol_t *hipFunction_t;" from hcc_details to HIP
    //             typedef struct CUfunc_st          *CUfunction;
    // cuda2hipRename["CUfunc_st"]                                                = {"ihipModuleSymbol_t", CONV_TYPE, API_DRIVER};

    // typedef struct CUgraphicsResource_st *CUgraphicsResource;
    cuda2hipRename["CUgraphicsResource"]                                          = {"hipGraphicsResource_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    // typedef struct CUmipmappedArray_st *CUmipmappedArray;
    cuda2hipRename["CUmipmappedArray"]                                            = {"hipMipmappedArray_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};

    // unsupported yet by HIP
    cuda2hipRename["CUfunction_attribute"]                         = {"hipFuncAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CUfunction_attribute_enum"]                    = {"hipFuncAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"]      = {"hipFuncAttributeMaxThreadsPerBlocks", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"]          = {"hipFuncAttributeSharedSizeBytes", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES"]           = {"hipFuncAttributeConstSizeBytes", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"]           = {"hipFuncAttributeLocalSizeBytes", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_FUNC_ATTRIBUTE_NUM_REGS"]                   = {"hipFuncAttributeNumRegs", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_FUNC_ATTRIBUTE_PTX_VERSION"]                = {"hipFuncAttributePtxVersion", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_FUNC_ATTRIBUTE_BINARY_VERSION"]             = {"hipFuncAttributeBinaryVersion", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_FUNC_ATTRIBUTE_CACHE_MODE_CA"]              = {"hipFuncAttributeCacheModeCA", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_FUNC_ATTRIBUTE_MAX"]                        = {"hipFuncAttributeMax", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};

    // enum CUgraphicsMapResourceFlags/CUgraphicsMapResourceFlags_enum
    cuda2hipRename["CUgraphicsMapResourceFlags"]                   = {"hipGraphicsMapFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                              // API_Runtime ANALOGUE (cudaGraphicsMapFlags)
    cuda2hipRename["CUgraphicsMapResourceFlags_enum"]              = {"hipGraphicsMapFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                              // API_Runtime ANALOGUE (cudaGraphicsMapFlags)
    cuda2hipRename["CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE"]          = {"hipGraphicsMapFlagsNone", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                  // 0x00 // API_Runtime ANALOGUE (cudaGraphicsMapFlagsNone = 0)
    cuda2hipRename["CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY"]     = {"hipGraphicsMapFlagsReadOnly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};              // 0x01 // API_Runtime ANALOGUE (cudaGraphicsMapFlagsReadOnly = 1)
    cuda2hipRename["CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD"] = {"hipGraphicsMapFlagsWriteDiscard", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};          // 0x02 // API_Runtime ANALOGUE (cudaGraphicsMapFlagsWriteDiscard = 2)

    // enum CUgraphicsRegisterFlags/CUgraphicsRegisterFlags_enum
    cuda2hipRename["CUgraphicsRegisterFlags"]                      = {"hipGraphicsRegisterFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                         // API_Runtime ANALOGUE (cudaGraphicsRegisterFlags)
    cuda2hipRename["CUgraphicsRegisterFlags_enum"]                 = {"hipGraphicsRegisterFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                         // API_Runtime ANALOGUE (cudaGraphicsRegisterFlags)
    cuda2hipRename["CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE"]          = {"hipGraphicsRegisterFlagsNone", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};             // 0x00 // API_Runtime ANALOGUE (cudaGraphicsRegisterFlagsNone = 0)
    cuda2hipRename["CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY"]     = {"hipGraphicsRegisterFlagsReadOnly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};         // 0x01 // API_Runtime ANALOGUE (cudaGraphicsRegisterFlagsReadOnly = 1)
    cuda2hipRename["CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD"]     = {"hipGraphicsRegisterFlagsWriteDiscard", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};     // 0x02 // API_Runtime ANALOGUE (cudaGraphicsRegisterFlagsWriteDiscard = 2)
    cuda2hipRename["CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST"]      = {"hipGraphicsRegisterFlagsSurfaceLoadStore", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}; // 0x04 // API_Runtime ANALOGUE (cudaGraphicsRegisterFlagsSurfaceLoadStore = 4)
    cuda2hipRename["CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER"]    = {"hipGraphicsRegisterFlagsTextureGather", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};    // 0x08 // API_Runtime ANALOGUE (cudaGraphicsRegisterFlagsTextureGather = 8)

    // enum CUoccupancy_flags/CUoccupancy_flags_enum
    cuda2hipRename["CUoccupancy_flags"]                            = {"hipOccupancyFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                // API_Runtime ANALOGUE (no)
    cuda2hipRename["CUoccupancy_flags_enum"]                       = {"hipOccupancyFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_OCCUPANCY_DEFAULT"]                         = {"hipOccupancyDefault", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                      // 0x00 // API_Runtime ANALOGUE (cudaOccupancyDefault = 0x0)
    cuda2hipRename["CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE"]        = {"hipOccupancyDisableCachingOverride", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 0x01 // API_Runtime ANALOGUE (cudaOccupancyDisableCachingOverride = 0x1)

    cuda2hipRename["CUfunc_cache_enum"]                            = {"hipFuncCache", CONV_TYPE, API_DRIVER};                                                      // API_Runtime ANALOGUE (cudaFuncCache)
    cuda2hipRename["CUfunc_cache"]                                 = {"hipFuncCache", CONV_TYPE, API_DRIVER};                                                      // API_Runtime ANALOGUE (cudaFuncCache)
    cuda2hipRename["CU_FUNC_CACHE_PREFER_NONE"]                    = {"hipFuncCachePreferNone", CONV_CACHE, API_DRIVER};                                   // 0x00 // API_Runtime ANALOGUE (cudaFilterModePoint = 0)
    cuda2hipRename["CU_FUNC_CACHE_PREFER_SHARED"]                  = {"hipFuncCachePreferShared", CONV_CACHE, API_DRIVER};                                 // 0x01 // API_Runtime ANALOGUE (cudaFuncCachePreferShared = 1)
    cuda2hipRename["CU_FUNC_CACHE_PREFER_L1"]                      = {"hipFuncCachePreferL1", CONV_CACHE, API_DRIVER};                                     // 0x02 // API_Runtime ANALOGUE (cudaFuncCachePreferL1 = 2)
    cuda2hipRename["CU_FUNC_CACHE_PREFER_EQUAL"]                   = {"hipFuncCachePreferEqual", CONV_CACHE, API_DRIVER};                                  // 0x03 // API_Runtime ANALOGUE (cudaFuncCachePreferEqual = 3)

    // enum CUipcMem_flags/CUipcMem_flags_enum
    cuda2hipRename["CUipcMem_flags"]                               = {"hipIpcMemFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                   // API_Runtime ANALOGUE (no)
    cuda2hipRename["CUipcMem_flags_enum"]                          = {"hipIpcMemFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                   // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS"]           = {"hipIpcMemLazyEnablePeerAccess", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};             // 0x1 // API_Runtime ANALOGUE (cudaIpcMemLazyEnablePeerAccess = 0x01)

    // enum CUipcMem_flags/CUipcMem_flags_enum
    cuda2hipRename["CUipcMem_flags"]                               = {"hipIpcMemFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                   // API_Runtime ANALOGUE (no)

    // JIT
    // enum CUjit_cacheMode/CUjit_cacheMode_enum
    cuda2hipRename["CUjit_cacheMode"]                              = {"hipJitCacheMode", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};                                  // API_Runtime ANALOGUE (no)
    cuda2hipRename["CUjit_cacheMode_enum"]                         = {"hipJitCacheMode", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_JIT_CACHE_OPTION_NONE"]                     = {"hipJitCacheModeOptionNone", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_JIT_CACHE_OPTION_CG"]                       = {"hipJitCacheModeOptionCG", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_JIT_CACHE_OPTION_CA"]                       = {"hipJitCacheModeOptionCA", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    // enum CUjit_fallback/CUjit_fallback_enum
    cuda2hipRename["CUjit_fallback"]                               = {"hipJitFallback", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};                                  // API_Runtime ANALOGUE (no)
    cuda2hipRename["CUjit_fallback_enum"]                          = {"hipJitFallback", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_PREFER_PTX"]                                = {"hipJitFallbackPreferPtx", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_PREFER_BINARY"]                             = {"hipJitFallbackPreferBinary", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    // enum CUjit_option/CUjit_option_enum
    cuda2hipRename["CUjit_option"]                                 = {"hipJitOption", CONV_JIT, API_DRIVER};  // API_Runtime ANALOGUE (no)
    cuda2hipRename["CUjit_option_enum"]                            = {"hipJitOption", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_MAX_REGISTERS"]                         = {"hipJitOptionMaxRegisters", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_THREADS_PER_BLOCK"]                     = {"hipJitOptionThreadsPerBlock", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_WALL_TIME"]                             = {"hipJitOptionWallTime", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_INFO_LOG_BUFFER"]                       = {"hipJitOptionInfoLogBuffer", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES"]            = {"hipJitOptionInfoLogBufferSizeBytes", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_ERROR_LOG_BUFFER"]                      = {"hipJitOptionErrorLogBuffer", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES"]           = {"hipJitOptionErrorLogBufferSizeBytes", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_OPTIMIZATION_LEVEL"]                    = {"hipJitOptionOptimizationLevel", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_TARGET_FROM_CUCONTEXT"]                 = {"hipJitOptionTargetFromContext", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_TARGET"]                                = {"hipJitOptionTarget", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_FALLBACK_STRATEGY"]                     = {"hipJitOptionFallbackStrategy", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_GENERATE_DEBUG_INFO"]                   = {"hipJitOptionGenerateDebugInfo", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_LOG_VERBOSE"]                           = {"hipJitOptionLogVerbose", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_GENERATE_LINE_INFO"]                    = {"hipJitOptionGenerateLineInfo", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_CACHE_MODE"]                            = {"hipJitOptionCacheMode", CONV_JIT, API_DRIVER};
    // unsupported yet by HIP [CUDA 8.0.44]
    cuda2hipRename["CU_JIT_NEW_SM3X_OPT"]                          = {"hipJitOptionSm3xOpt", CONV_JIT, API_DRIVER};
    cuda2hipRename["CU_JIT_FAST_COMPILE"]                          = {"hipJitOptionFastCompile", CONV_JIT, API_DRIVER};

    cuda2hipRename["CU_JIT_NUM_OPTIONS"]                           = {"hipJitOptionNumOptions", CONV_JIT, API_DRIVER};
    // enum CUjit_target/CUjit_target_enum
    cuda2hipRename["CUjit_target"]                                 = {"hipJitTarget", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};  // API_Runtime ANALOGUE (no)
    cuda2hipRename["CUjit_target_enum"]                            = {"hipJitTarget", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_10"]                         = {"hipJitTargetCompute10", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_11"]                         = {"hipJitTargetCompute11", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_12"]                         = {"hipJitTargetCompute12", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_13"]                         = {"hipJitTargetCompute13", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_20"]                         = {"hipJitTargetCompute20", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_21"]                         = {"hipJitTargetCompute21", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_30"]                         = {"hipJitTargetCompute30", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_32"]                         = {"hipJitTargetCompute32", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_35"]                         = {"hipJitTargetCompute35", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_37"]                         = {"hipJitTargetCompute37", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_50"]                         = {"hipJitTargetCompute50", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_52"]                         = {"hipJitTargetCompute52", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    // unsupported yet by HIP [CUDA 8.0.44]
    cuda2hipRename["CU_TARGET_COMPUTE_53"]                         = {"hipJitTargetCompute53", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_60"]                         = {"hipJitTargetCompute60", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_61"]                         = {"hipJitTargetCompute61", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_TARGET_COMPUTE_62"]                         = {"hipJitTargetCompute62", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    // enum CUjitInputType/CUjitInputType_enum
    cuda2hipRename["CUjitInputType"]                               = {"hipJitInputType", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};  // API_Runtime ANALOGUE (no)
    cuda2hipRename["CUjitInputType_enum"]                          = {"hipJitInputType", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_JIT_INPUT_CUBIN"]                           = {"hipJitInputTypeBin", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_JIT_INPUT_PTX"]                             = {"hipJitInputTypePtx", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_JIT_INPUT_FATBINARY"]                       = {"hipJitInputTypeFatBinary", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_JIT_INPUT_OBJECT"]                          = {"hipJitInputTypeObject", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_JIT_INPUT_LIBRARY"]                         = {"hipJitInputTypeLibrary", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_JIT_NUM_INPUT_TYPES"]                       = {"hipJitInputTypeNumInputTypes", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED};

    // Limits
    cuda2hipRename["CUlimit"]                                      = {"hipLimit_t", CONV_TYPE, API_DRIVER};                                                          // API_Runtime ANALOGUE (cudaLimit)
    cuda2hipRename["CUlimit_enum"]                                 = {"hipLimit_t", CONV_TYPE, API_DRIVER};                                                          // API_Runtime ANALOGUE (cudaLimit)
    cuda2hipRename["CU_LIMIT_STACK_SIZE"]                          = {"hipLimitStackSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                          // 0x00 // API_Runtime ANALOGUE (cudaLimitStackSize = 0x00)
    cuda2hipRename["CU_LIMIT_PRINTF_FIFO_SIZE"]                    = {"hipLimitPrintfFifoSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                     // 0x01 // API_Runtime ANALOGUE (cudaLimitPrintfFifoSize = 0x01)
    cuda2hipRename["CU_LIMIT_MALLOC_HEAP_SIZE"]                    = {"hipLimitMallocHeapSize", CONV_TYPE, API_DRIVER};                                      // 0x02 // API_Runtime ANALOGUE (cudaLimitMallocHeapSize = 0x02)
    cuda2hipRename["CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH"]              = {"hipLimitDevRuntimeSyncDepth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                // 0x03 // API_Runtime ANALOGUE (cudaLimitDevRuntimeSyncDepth = 0x03)
    cuda2hipRename["CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT"]    = {"hipLimitDevRuntimePendingLaunchCount", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 0x04 // API_Runtime ANALOGUE (cudaLimitDevRuntimePendingLaunchCount = 0x04)
    cuda2hipRename["CU_LIMIT_STACK_SIZE"]                          = {"hipLimitStackSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                  // API_Runtime ANALOGUE (no)

    // enum CUmemAttach_flags/CUmemAttach_flags_enum
    cuda2hipRename["CUmemAttach_flags"]                            = {"hipMemAttachFlags_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                // API_Runtime ANALOGUE (no)
    cuda2hipRename["CUmemAttach_flags_enum"]                       = {"hipMemAttachFlags_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_MEM_ATTACH_GLOBAL"]                         = {"hipMemAttachGlobal", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                          // 0x1 // API_Runtime ANALOGUE (#define cudaMemAttachGlobal 0x01)
    cuda2hipRename["CU_MEM_ATTACH_HOST"]                           = {"hipMemAttachHost", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                            // 0x2 // API_Runtime ANALOGUE (#define cudaMemAttachHost 0x02)
    cuda2hipRename["CU_MEM_ATTACH_SINGLE"]                         = {"hipMemAttachSingle", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                          // 0x4 // API_Runtime ANALOGUE (#define cudaMemAttachSingle 0x04)

    // enum CUmemorytype/CUmemorytype_enum
    cuda2hipRename["CUmemorytype"]                                 = {"hipMemType_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                       // API_Runtime ANALOGUE (no - cudaMemoryType is not an analogue)
    cuda2hipRename["CUmemorytype_enum"]                            = {"hipMemType_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                                       // API_Runtime ANALOGUE (no - cudaMemoryType is not an analogue)
    cuda2hipRename["CU_MEMORYTYPE_HOST"]                           = {"hipMemTypeHost", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                             // 0x01 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_MEMORYTYPE_DEVICE"]                         = {"hipMemTypeDevice", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                           // 0x02 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_MEMORYTYPE_ARRAY"]                          = {"hipMemTypeArray", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                            // 0x03 // API_Runtime ANALOGUE (no)
    cuda2hipRename["CU_MEMORYTYPE_UNIFIED"]                        = {"hipMemTypeUnified", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                          // 0x04 // API_Runtime ANALOGUE (no)

    // enum CUresourcetype
    cuda2hipRename["CUresourcetype"]                               = {"hipResourceType", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                                     // API_Runtime ANALOGUE (cudaResourceType)
    cuda2hipRename["CUresourcetype_enum"]                          = {"hipResourceType", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                                     // API_Runtime ANALOGUE (cudaResourceType)
    cuda2hipRename["CU_RESOURCE_TYPE_ARRAY"]                       = {"hipResourceTypeArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                        // 0x00 // API_Runtime ANALOGUE (cudaResourceTypeArray = 0x00)
    cuda2hipRename["CU_RESOURCE_TYPE_MIPMAPPED_ARRAY"]             = {"hipResourceTypeMipmappedArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};               // 0x01 // API_Runtime ANALOGUE (cudaResourceTypeMipmappedArray = 0x01)
    cuda2hipRename["CU_RESOURCE_TYPE_LINEAR"]                      = {"hipResourceTypeLinear", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                       // 0x02 // API_Runtime ANALOGUE (cudaResourceTypeLinear = 0x02)
    cuda2hipRename["CU_RESOURCE_TYPE_PITCH2D"]                     = {"hipResourceTypePitch2D", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                      // 0x03 // API_Runtime ANALOGUE (cudaResourceTypePitch2D = 0x03)

    // enum CUresourceViewFormat/CUresourceViewFormat_enum
    cuda2hipRename["CUresourceViewFormat"]                         = {"hipResourceViewFormat", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                               // API_Runtime ANALOGUE (cudaResourceViewFormat)
    cuda2hipRename["CUresourceViewFormat_enum"]                    = {"hipResourceViewFormat", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                               // API_Runtime ANALOGUE (cudaResourceViewFormat)
    cuda2hipRename["CU_RES_VIEW_FORMAT_NONE"]                      = {"hipResViewFormatNone", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                        // 0x00 // API_Runtime ANALOGUE (cudaResViewFormatNone = 0x00)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UINT_1X8"]                  = {"hipResViewFormatUnsignedChar1", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};               // 0x01 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedChar1 = 0x01)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UINT_2X8"]                  = {"hipResViewFormatUnsignedChar2", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};               // 0x02 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedChar2 = 0x02)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UINT_4X8"]                  = {"hipResViewFormatUnsignedChar4", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};               // 0x03 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedChar4 = 0x03)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SINT_1X8"]                  = {"hipResViewFormatSignedChar1", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                 // 0x04 // API_Runtime ANALOGUE (cudaResViewFormatSignedChar1 = 0x04)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SINT_2X8"]                  = {"hipResViewFormatSignedChar2", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                 // 0x05 // API_Runtime ANALOGUE (cudaResViewFormatSignedChar2 = 0x05)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SINT_4X8"]                  = {"hipResViewFormatSignedChar4", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                 // 0x06 // API_Runtime ANALOGUE (cudaResViewFormatSignedChar4 = 0x06)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UINT_1X16"]                 = {"hipResViewFormatUnsignedShort1", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};              // 0x07 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedShort1 = 0x07)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UINT_2X16"]                 = {"hipResViewFormatUnsignedShort2", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};              // 0x08 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedShort2 = 0x08)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UINT_4X16"]                 = {"hipResViewFormatUnsignedShort4", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};              // 0x09 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedShort4 = 0x09)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SINT_1X16"]                 = {"hipResViewFormatSignedShort1", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                // 0x0a // API_Runtime ANALOGUE (cudaResViewFormatSignedShort1 = 0x0a)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SINT_2X16"]                 = {"hipResViewFormatSignedShort2", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                // 0x0b // API_Runtime ANALOGUE (cudaResViewFormatSignedShort2 = 0x0b)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SINT_4X16"]                 = {"hipResViewFormatSignedShort4", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                // 0x0c // API_Runtime ANALOGUE (cudaResViewFormatSignedShort4 = 0x0c)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UINT_1X32"]                 = {"hipResViewFormatUnsignedInt1", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                // 0x0d // API_Runtime ANALOGUE (cudaResViewFormatUnsignedInt1 = 0x0d)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UINT_2X32"]                 = {"hipResViewFormatUnsignedInt2", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                // 0x0e // API_Runtime ANALOGUE (cudaResViewFormatUnsignedInt2 = 0x0e)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UINT_4X32"]                 = {"hipResViewFormatUnsignedInt4", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                // 0x0f // API_Runtime ANALOGUE (cudaResViewFormatUnsignedInt4 = 0x0f)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SINT_1X32"]                 = {"hipResViewFormatSignedInt1", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                  // 0x10 // API_Runtime ANALOGUE (cudaResViewFormatSignedInt1 = 0x10)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SINT_2X32"]                 = {"hipResViewFormatSignedInt2", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                  // 0x11 // API_Runtime ANALOGUE (cudaResViewFormatSignedInt2 = 0x11)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SINT_4X32"]                 = {"hipResViewFormatSignedInt4", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                  // 0x12 // API_Runtime ANALOGUE (cudaResViewFormatSignedInt4 = 0x12)
    cuda2hipRename["CU_RES_VIEW_FORMAT_FLOAT_1X16"]                = {"hipResViewFormatHalf1", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                       // 0x13 // API_Runtime ANALOGUE (cudaResViewFormatHalf1 = 0x13)
    cuda2hipRename["CU_RES_VIEW_FORMAT_FLOAT_2X16"]                = {"hipResViewFormatHalf2", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                       // 0x14 // API_Runtime ANALOGUE (cudaResViewFormatHalf2 = 0x14)
    cuda2hipRename["CU_RES_VIEW_FORMAT_FLOAT_4X16"]                = {"hipResViewFormatHalf4", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                       // 0x15 // API_Runtime ANALOGUE (cudaResViewFormatHalf4 = 0x15)
    cuda2hipRename["CU_RES_VIEW_FORMAT_FLOAT_1X32"]                = {"hipResViewFormatFloat1", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                      // 0x16 // API_Runtime ANALOGUE (cudaResViewFormatFloat1 = 0x16)
    cuda2hipRename["CU_RES_VIEW_FORMAT_FLOAT_2X32"]                = {"hipResViewFormatFloat2", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                      // 0x17 // API_Runtime ANALOGUE (cudaResViewFormatFloat2 = 0x17)
    cuda2hipRename["CU_RES_VIEW_FORMAT_FLOAT_4X32"]                = {"hipResViewFormatFloat4", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};                      // 0x18 // API_Runtime ANALOGUE (cudaResViewFormatFloat4 = 0x18)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UNSIGNED_BC1"]              = {"hipResViewFormatUnsignedBlockCompressed1", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};    // 0x19 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed1 = 0x19)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UNSIGNED_BC2"]              = {"hipResViewFormatUnsignedBlockCompressed2", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};    // 0x1a // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed2 = 0x1a)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UNSIGNED_BC3"]              = {"hipResViewFormatUnsignedBlockCompressed3", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};    // 0x1b // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed3 = 0x1b)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UNSIGNED_BC4"]              = {"hipResViewFormatUnsignedBlockCompressed4", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};    // 0x1c // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed4 = 0x1c)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SIGNED_BC4"]                = {"hipResViewFormatSignedBlockCompressed4", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};      // 0x1d // API_Runtime ANALOGUE (cudaResViewFormatSignedBlockCompressed4 = 0x1d)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UNSIGNED_BC5"]              = {"hipResViewFormatUnsignedBlockCompressed5", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};    // 0x1e // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed5 = 0x1e)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SIGNED_BC5"]                = {"hipResViewFormatSignedBlockCompressed5", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};      // 0x1f // API_Runtime ANALOGUE (cudaResViewFormatSignedBlockCompressed5 = 0x1f)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UNSIGNED_BC6H"]             = {"hipResViewFormatUnsignedBlockCompressed6H", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};   // 0x20 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed6H = 0x20)
    cuda2hipRename["CU_RES_VIEW_FORMAT_SIGNED_BC6H"]               = {"hipResViewFormatSignedBlockCompressed6H", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};     // 0x21 // API_Runtime ANALOGUE (cudaResViewFormatSignedBlockCompressed6H = 0x21)
    cuda2hipRename["CU_RES_VIEW_FORMAT_UNSIGNED_BC7"]              = {"hipResViewFormatUnsignedBlockCompressed7", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};    // 0x22 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed7 = 0x22)

    cuda2hipRename["CUsharedconfig"]                               = {"hipSharedMemConfig", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUsharedconfig_enum"]                          = {"hipSharedMemConfig", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE"]       = {"hipSharedMemBankSizeDefault", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE"]     = {"hipSharedMemBankSizeFourByte", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE"]    = {"hipSharedMemBankSizeEightByte", CONV_TYPE, API_DRIVER};

    cuda2hipRename["CUcontext"]                                    = {"hipCtx_t", CONV_TYPE, API_DRIVER};
    // TODO: move "typedef struct ihipCtx_t *hipCtx_t;" from hcc_details to HIP
    //             typedef struct CUctx_st  *CUcontext;
    // cuda2hipRename["CUctx_st"]                                  = {"ihipCtx_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUmodule"]                                     = {"hipModule_t", CONV_TYPE, API_DRIVER};
    // TODO: move "typedef struct ihipModule_t *hipModule_t;" from hcc_details to HIP
    //             typedef struct CUmod_st     *CUmodule;
    // cuda2hipRename["CUmod_st"]                                  = {"ihipModule_t", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CUstream"]                                     = {"hipStream_t", CONV_TYPE, API_DRIVER};
    // TODO: move "typedef struct ihipStream_t *hipStream_t;" from hcc_details to HIP
    //             typedef struct CUstream_st *CUstream;
    // cuda2hipRename["CUstream_st"]                               = {"ihipStream_t", CONV_TYPE, API_DRIVER};

    // typedef void (*hipStreamCallback_t)      (hipStream_t stream, hipError_t status, void* userData);
    // typedef void (CUDA_CB *CUstreamCallback) (CUstream hStream, CUresult status, void* userData)
    cuda2hipRename["CUstreamCallback"]                             = {"hipStreamCallback_t", CONV_TYPE, API_DRIVER};

    cuda2hipRename["CUsurfObject"]                                 = {"hipSurfaceObject", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    // typedef struct CUsurfref_st *CUsurfref;
    cuda2hipRename["CUsurfref"]                                    = {"hipSurfaceReference_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    // cuda2hipRename["CUsurfref_st"]                              = {"ihipSurfaceReference_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CUtexObject"]                                  = {"hipTextureObject", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    // typedef struct CUtexref_st *CUtexref;
    cuda2hipRename["CUtexref"]                                     = {"hipTextureReference_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    // cuda2hipRename["CUtexref_st"]                               = {"ihipTextureReference_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};

    // Stream Flags enum
    cuda2hipRename["CUstream_flags"]                               = {"hipStreamFlags", CONV_TYPE, API_DRIVER};
    // cuda2hipRename["CUstream_flags_enum"]                       = {"hipStreamFlags", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CU_STREAM_DEFAULT"]                            = {"hipStreamDefault", CONV_TYPE, API_DRIVER};
    cuda2hipRename["CU_STREAM_NON_BLOCKING"]                       = {"hipStreamNonBlocking", CONV_TYPE, API_DRIVER};

    // unsupported yet by HIP [CUDA 8.0.44]
    // Flags for ::cuStreamWaitValue32
    cuda2hipRename["CUstreamWaitValue_flags"]                      = {"hipStreamWaitValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    // cuda2hipRename["CUstreamWaitValue_flags_enum"]              = {"hipStreamWaitValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_STREAM_WAIT_VALUE_GEQ"]                     = {"hipStreamWaitValueGeq", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                // 0x0
    cuda2hipRename["CU_STREAM_WAIT_VALUE_EQ"]                      = {"hipStreamWaitValueEq", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                 // 0x1
    cuda2hipRename["CU_STREAM_WAIT_VALUE_AND"]                     = {"hipStreamWaitValueAnd", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                // 0x2
    cuda2hipRename["CU_STREAM_WAIT_VALUE_FLUSH"]                   = {"hipStreamWaitValueFlush", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};              // 1<<30
    // Flags for ::cuStreamWriteValue32
    cuda2hipRename["CUstreamWriteValue_flags"]                     = {"hipStreamWriteValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    // cuda2hipRename["CUstreamWriteValue_flags"]                  = {"hipStreamWriteValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_STREAM_WRITE_VALUE_DEFAULT"]                = {"hipStreamWriteValueDefault", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};           // 0x0
    cuda2hipRename["CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER"]      = {"hipStreamWriteValueNoMemoryBarrier", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};   // 0x1
    // Flags for ::cuStreamBatchMemOp
    cuda2hipRename["CUstreamBatchMemOpType"]                       = {"hipStreamBatchMemOpType", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    // cuda2hipRename["CUstreamBatchMemOpType_enum"]               = {"hipStreamBatchMemOpType", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_STREAM_MEM_OP_WAIT_VALUE_32"]               = {"hipStreamBatchMemOpWaitValue32", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 1
    cuda2hipRename["CU_STREAM_MEM_OP_WRITE_VALUE_32"]              = {"hipStreamBatchMemOpWriteValue32", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};      // 2
    cuda2hipRename["CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES"]         = {"hipStreamBatchMemOpFlushRemoteWrites", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}; // 3

    // Error Handling
    cuda2hipRename["cuGetErrorName"]                               = {"hipGetErrorName___", CONV_ERROR, API_DRIVER, HIP_UNSUPPORTED};   // cudaGetErrorName (hipGetErrorName) has different signature
    cuda2hipRename["cuGetErrorString"]                             = {"hipGetErrorString___", CONV_ERROR, API_DRIVER, HIP_UNSUPPORTED}; // cudaGetErrorString (hipGetErrorString) has different signature

    // Init
    cuda2hipRename["cuInit"]                                       = {"hipInit", CONV_INIT, API_DRIVER};

    // Driver
    cuda2hipRename["cuDriverGetVersion"]                           = {"hipDriverGetVersion", CONV_VERSION, API_DRIVER};

    // Context Management
    cuda2hipRename["cuCtxCreate_v2"]                            = {"hipCtxCreate", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxDestroy_v2"]                           = {"hipCtxDestroy", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetApiVersion"]                        = {"hipCtxGetApiVersion", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetCacheConfig"]                       = {"hipCtxGetCacheConfig", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetCurrent"]                           = {"hipCtxGetCurrent", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetDevice"]                            = {"hipCtxGetDevice", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetFlags"]                             = {"hipCtxGetFlags", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetLimit"]                             = {"hipCtxGetLimit", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuCtxGetSharedMemConfig"]                   = {"hipCtxGetSharedMemConfig", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxGetStreamPriorityRange"]               = {"hipCtxGetStreamPriorityRange", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuCtxPopCurrent_v2"]                        = {"hipCtxPopCurrent", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxPushCurrent_v2"]                       = {"hipCtxPushCurrent", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxSetCacheConfig"]                       = {"hipCtxSetCacheConfig", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxSetCurrent"]                           = {"hipCtxSetCurrent", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxSetLimit"]                             = {"hipCtxSetLimit", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuCtxSetSharedMemConfig"]                   = {"hipCtxSetSharedMemConfig", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuCtxSynchronize"]                          = {"hipCtxSynchronize", CONV_CONTEXT, API_DRIVER};
    // Context Management [DEPRECATED]
    cuda2hipRename["cuCtxAttach"]                               = {"hipCtxAttach", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuCtxDetach"]                               = {"hipCtxDetach", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED};

    // Peer Context Memory Access
    cuda2hipRename["cuCtxEnablePeerAccess"]                     = {"hipCtxEnablePeerAccess", CONV_PEER, API_DRIVER};
    cuda2hipRename["cuCtxDisablePeerAccess"]                    = {"hipCtxDisablePeerAccess", CONV_PEER, API_DRIVER};
    cuda2hipRename["cuDeviceCanAccessPeer"]                     = {"hipDeviceCanAccessPeer", CONV_PEER, API_DRIVER};
    cuda2hipRename["cuDeviceGetP2PAttribute"]                   = {"hipDeviceGetP2PAttribute", CONV_PEER, API_DRIVER, HIP_UNSUPPORTED};  // API_Runtime ANALOGUE (cudaDeviceGetP2PAttribute)

    // Primary Context Management
    cuda2hipRename["cuDevicePrimaryCtxGetState"]                = {"hipDevicePrimaryCtxGetState", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuDevicePrimaryCtxRelease"]                 = {"hipDevicePrimaryCtxRelease", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuDevicePrimaryCtxReset"]                   = {"hipDevicePrimaryCtxReset", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuDevicePrimaryCtxRetain"]                  = {"hipDevicePrimaryCtxRetain", CONV_CONTEXT, API_DRIVER};
    cuda2hipRename["cuDevicePrimaryCtxSetFlags"]                = {"hipDevicePrimaryCtxSetFlags", CONV_CONTEXT, API_DRIVER};

    // Device Management
    cuda2hipRename["cuDeviceGet"]                               = {"hipGetDevice", CONV_DEVICE, API_DRIVER};
    cuda2hipRename["cuDeviceGetName"]                           = {"hipDeviceGetName", CONV_DEVICE, API_DRIVER};
    cuda2hipRename["cuDeviceGetCount"]                          = {"hipGetDeviceCount", CONV_DEVICE, API_DRIVER};
    cuda2hipRename["cuDeviceGetAttribute"]                      = {"hipDeviceGetAttribute", CONV_DEVICE, API_DRIVER};
    cuda2hipRename["cuDeviceGetPCIBusId"]                       = {"hipDeviceGetPCIBusId", CONV_DEVICE, API_DRIVER};
    cuda2hipRename["cuDeviceGetByPCIBusId"]                     = {"hipDeviceGetByPCIBusId", CONV_DEVICE, API_DRIVER};
    cuda2hipRename["cuDeviceTotalMem_v2"]                       = {"hipDeviceTotalMem", CONV_DEVICE, API_DRIVER};

    // Device Management [DEPRECATED]
    cuda2hipRename["cuDeviceComputeCapability"]                 = {"hipDeviceComputeCapability", CONV_DEVICE, API_DRIVER};
    cuda2hipRename["cuDeviceGetProperties"]                     = {"hipGetDeviceProperties", CONV_DEVICE, API_DRIVER};

    // Module Management
    cuda2hipRename["cuLinkAddData"]                             = {"hipLinkAddData", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuLinkAddFile"]                             = {"hipLinkAddFile", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuLinkComplete"]                            = {"hipLinkComplete", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuLinkCreate"]                              = {"hipLinkCreate", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuLinkDestroy"]                             = {"hipLinkDestroy", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuModuleGetFunction"]                       = {"hipModuleGetFunction", CONV_MODULE, API_DRIVER};
    cuda2hipRename["cuModuleGetGlobal_v2"]                      = {"hipModuleGetGlobal", CONV_MODULE, API_DRIVER};
    cuda2hipRename["cuModuleGetSurfRef"]                        = {"hipModuleGetSurfRef", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuModuleGetTexRef"]                         = {"hipModuleGetTexRef", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuModuleLoad"]                              = {"hipModuleLoad", CONV_MODULE, API_DRIVER};
    cuda2hipRename["cuModuleLoadData"]                          = {"hipModuleLoadData", CONV_MODULE, API_DRIVER};
    cuda2hipRename["cuModuleLoadDataEx"]                        = {"hipModuleLoadDataEx", CONV_MODULE, API_DRIVER};
    cuda2hipRename["cuModuleLoadFatBinary"]                     = {"hipModuleLoadFatBinary", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuModuleUnload"]                            = {"hipModuleUnload", CONV_MODULE, API_DRIVER};

    // unsupported yet by HIP [CUDA 8.0.44]
    // P2P Attributes
    cuda2hipRename["CUdevice_P2PAttribute"]                            = {"hipDeviceP2PAttribute", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};                              // API_Runtime ANALOGUE (cudaDeviceP2PAttr)
    // cuda2hipRename["CUdevice_P2PAttribute_enum"]                    = {"hipDeviceP2PAttribute", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK"]         = {"hipDeviceP2PAttributePerformanceRank", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 0x01 // API_Runtime ANALOGUE (cudaDevP2PAttrPerformanceRank = 0x01)
    cuda2hipRename["CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED"]         = {"hipDeviceP2PAttributeAccessSupported", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED};       // 0x02 // API_Runtime ANALOGUE (cudaDevP2PAttrAccessSupported = 0x02)
    cuda2hipRename["CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED"]  = {"hipDeviceP2PAttributeNativeAtomicSupported", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}; // 0x03 // API_Runtime ANALOGUE (cudaDevP2PAttrNativeAtomicSupported = 0x03)

    // Events
    // pointer to CUevent_st
    cuda2hipRename["CUevent"]                                   = {"hipEvent_t", CONV_TYPE, API_DRIVER};
    // ToDo:
    // cuda2hipRename["CUevent_st"]                             = {"XXXX", CONV_TYPE, API_DRIVER};
    // Event Flags
    cuda2hipRename["CUevent_flags"]                             = {"hipEventFlags", CONV_EVENT, API_DRIVER, HIP_UNSUPPORTED};
    // ToDo:
    // cuda2hipRename["CUevent_flags_enum"]                     = {"hipEventFlags", CONV_EVENT, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_EVENT_DEFAULT"]                          = {"hipEventDefault", CONV_EVENT, API_DRIVER};
    cuda2hipRename["CU_EVENT_BLOCKING_SYNC"]                    = {"hipEventBlockingSync", CONV_EVENT, API_DRIVER};
    cuda2hipRename["CU_EVENT_DISABLE_TIMING"]                   = {"hipEventDisableTiming", CONV_EVENT, API_DRIVER};
    cuda2hipRename["CU_EVENT_INTERPROCESS"]                     = {"hipEventInterprocess", CONV_EVENT, API_DRIVER};
    // Event functions
    cuda2hipRename["cuEventCreate"]                             = {"hipEventCreate", CONV_EVENT, API_DRIVER};
    cuda2hipRename["cuEventDestroy_v2"]                         = {"hipEventDestroy", CONV_EVENT, API_DRIVER};
    cuda2hipRename["cuEventElapsedTime"]                        = {"hipEventElapsedTime", CONV_EVENT, API_DRIVER};
    cuda2hipRename["cuEventQuery"]                              = {"hipEventQuery", CONV_EVENT, API_DRIVER};
    cuda2hipRename["cuEventRecord"]                             = {"hipEventRecord", CONV_EVENT, API_DRIVER};
    cuda2hipRename["cuEventSynchronize"]                        = {"hipEventSynchronize", CONV_EVENT, API_DRIVER};

    // Execution Control
    cuda2hipRename["cuFuncGetAttribute"]                        = {"hipFuncGetAttribute", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuFuncSetCacheConfig"]                      = {"hipFuncSetCacheConfig", CONV_MODULE, API_DRIVER};
    cuda2hipRename["cuFuncSetSharedMemConfig"]                  = {"hipFuncSetSharedMemConfig", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuLaunchKernel"]                            = {"hipModuleLaunchKernel", CONV_MODULE, API_DRIVER};

    // Execution Control [DEPRECATED]
    cuda2hipRename["cuFuncSetBlockShape"]                       = {"hipFuncSetBlockShape", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuFuncSetSharedSize"]                       = {"hipFuncSetSharedSize", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuLaunch"]                                  = {"hipLaunch", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};              // API_Runtime ANALOGUE (cudaLaunch)
    cuda2hipRename["cuLaunchGrid"]                              = {"hipLaunchGrid", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuLaunchGridAsync"]                         = {"hipLaunchGridAsync", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuParamSetf"]                               = {"hipParamSetf", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuParamSeti"]                               = {"hipParamSeti", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuParamSetSize"]                            = {"hipParamSetSize", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuParamSetSize"]                            = {"hipParamSetSize", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuParamSetv"]                               = {"hipParamSetv", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED};

    // Occupancy
    cuda2hipRename["cuOccupancyMaxActiveBlocksPerMultiprocessor"]          = {"hipOccupancyMaxActiveBlocksPerMultiprocessor", CONV_OCCUPANCY, API_DRIVER};                           // API_Runtime ANALOGUE (cudaOccupancyMaxActiveBlocksPerMultiprocessor)
    cuda2hipRename["cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"] = {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED}; // API_Runtime ANALOGUE (cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
    cuda2hipRename["cuOccupancyMaxPotentialBlockSize"]                     = {"hipOccupancyMaxPotentialBlockSize", CONV_OCCUPANCY, API_DRIVER};                                      // API_Runtime ANALOGUE (cudaOccupancyMaxPotentialBlockSize)
    cuda2hipRename["cuOccupancyMaxPotentialBlockSizeWithFlags"]            = {"hipOccupancyMaxPotentialBlockSizeWithFlags", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED};            // API_Runtime ANALOGUE (cudaOccupancyMaxPotentialBlockSizeWithFlags)

    // Streams
    cuda2hipRename["cuStreamAddCallback"]                       = {"hipStreamAddCallback", CONV_STREAM, API_DRIVER};
    cuda2hipRename["cuStreamAttachMemAsync"]                    = {"hipStreamAttachMemAsync", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuStreamCreate"]                            = {"hipStreamCreate__", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};      // Not equal to cudaStreamCreate due to different signatures
    cuda2hipRename["cuStreamCreateWithPriority"]                = {"hipStreamCreateWithPriority", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuStreamDestroy_v2"]                        = {"hipStreamDestroy", CONV_STREAM, API_DRIVER};
    cuda2hipRename["cuStreamGetFlags"]                          = {"hipStreamGetFlags", CONV_STREAM, API_DRIVER};
    cuda2hipRename["cuStreamGetPriority"]                       = {"hipStreamGetPriority", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuStreamQuery"]                             = {"hipStreamQuery", CONV_STREAM, API_DRIVER};
    cuda2hipRename["cuStreamSynchronize"]                       = {"hipStreamSynchronize", CONV_STREAM, API_DRIVER};
    cuda2hipRename["cuStreamWaitEvent"]                         = {"hipStreamWaitEvent", CONV_STREAM, API_DRIVER};
    cuda2hipRename["cuStreamWaitValue32"]                       = {"hipStreamWaitValue32", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};   // [CUDA 8.0.44] // no API_Runtime ANALOGUE
    cuda2hipRename["cuStreamWriteValue32"]                      = {"hipStreamWriteValue32", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};  // [CUDA 8.0.44] // no API_Runtime ANALOGUE
    cuda2hipRename["cuStreamBatchMemOp"]                        = {"hipStreamBatchMemOp", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED};    // [CUDA 8.0.44] // no API_Runtime ANALOGUE

    // Memory management
    cuda2hipRename["cuArray3DCreate"]                           = {"hipArray3DCreate", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuArray3DGetDescriptor"]                    = {"hipArray3DGetDescriptor", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuArrayCreate"]                             = {"hipArrayCreate", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuArrayDestroy"]                            = {"hipArrayDestroy", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuArrayGetDescriptor"]                      = {"hipArrayGetDescriptor", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuIpcCloseMemHandle"]                       = {"hipIpcCloseMemHandle", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuIpcGetEventHandle"]                       = {"hipIpcGetEventHandle", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuIpcGetMemHandle"]                         = {"hipIpcGetMemHandle", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuIpcOpenEventHandle"]                      = {"hipIpcOpenEventHandle", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuIpcOpenMemHandle"]                        = {"hipIpcOpenMemHandle", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemAlloc_v2"]                             = {"hipMalloc", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemAllocHost"]                            = {"hipMemAllocHost", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemAllocManaged"]                         = {"hipMemAllocManaged", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemAllocPitch"]                           = {"hipMemAllocPitch__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};        // Not equal to cudaMemAllocPitch due to different signatures
    cuda2hipRename["cuMemcpy"]                                  = {"hipMemcpy__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};               // Not equal to cudaMemcpy due to different signatures
    cuda2hipRename["cuMemcpy2D"]                                = {"hipMemcpy2D__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};             // Not equal to cudaMemcpy2D due to different signatures
    cuda2hipRename["cuMemcpy2DAsync"]                           = {"hipMemcpy2DAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};        // Not equal to cudaMemcpy2DAsync due to different signatures
    cuda2hipRename["cuMemcpy2DUnaligned"]                       = {"hipMemcpy2DUnaligned", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemcpy3D"]                                = {"hipMemcpy3D__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};             // Not equal to cudaMemcpy3D due to different signatures
    cuda2hipRename["cuMemcpy3DAsync"]                           = {"hipMemcpy3DAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};        // Not equal to cudaMemcpy3DAsync due to different signatures
    cuda2hipRename["cuMemcpy3DPeer"]                            = {"hipMemcpy3DPeer__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};         // Not equal to cudaMemcpy3DPeer due to different signatures
    cuda2hipRename["cuMemcpy3DPeerAsync"]                       = {"hipMemcpy3DPeerAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};    // Not equal to cudaMemcpy3DPeerAsync due to different signatures
    cuda2hipRename["cuMemcpyAsync"]                             = {"hipMemcpyAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};          // Not equal to cudaMemcpyAsync due to different signatures
    cuda2hipRename["cuMemcpyAtoA"]                              = {"hipMemcpyAtoA", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemcpyAtoD"]                              = {"hipMemcpyAtoD", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemcpyAtoH"]                              = {"hipMemcpyAtoH", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemcpyAtoHAsync"]                         = {"hipMemcpyAtoHAsync", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemcpyDtoA"]                              = {"hipMemcpyDtoA", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemcpyDtoD_v2"]                           = {"hipMemcpyDtoD", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemcpyDtoDAsync_v2"]                      = {"hipMemcpyDtoDAsync", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemcpyDtoH_v2"]                           = {"hipMemcpyDtoH", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemcpyDtoHAsync_v2"]                      = {"hipMemcpyDtoHAsync", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemcpyHtoA"]                              = {"hipMemcpyHtoA", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemcpyHtoAAsync"]                         = {"hipMemcpyHtoAAsync", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemcpyHtoD_v2"]                           = {"hipMemcpyHtoD", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemcpyHtoDAsync_v2"]                      = {"hipMemcpyHtoDAsync", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemcpyPeerAsync"]                         = {"hipMemcpyPeerAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};      // Not equal to cudaMemcpyPeerAsync due to different signatures
    cuda2hipRename["cuMemcpyPeer"]                              = {"hipMemcpyPeer__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};           // Not equal to cudaMemcpyPeer due to different signatures
    cuda2hipRename["cuMemFree_v2"]                              = {"hipFree", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemFreeHost"]                             = {"hipHostFree", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemGetAddressRange"]                      = {"hipMemGetAddressRange", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemGetInfo_v2"]                           = {"hipMemGetInfo", CONV_MEM, API_DRIVER};
    cuda2hipRename["cuMemHostAlloc"]                            = {"hipHostMalloc", CONV_MEM, API_DRIVER};                              // API_Runtime ANALOGUE (cudaHostAlloc)
    cuda2hipRename["cuMemHostGetDevicePointer"]                 = {"hipMemHostGetDevicePointer", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemHostGetFlags"]                         = {"hipMemHostGetFlags", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemHostRegister_v2"]                      = {"hipHostRegister", CONV_MEM, API_DRIVER};                            // API_Runtime ANALOGUE (cudaHostAlloc)
    cuda2hipRename["cuMemHostUnregister"]                       = {"hipHostUnregister", CONV_MEM, API_DRIVER};                          // API_Runtime ANALOGUE (cudaHostUnregister)
    cuda2hipRename["cuMemsetD16_v2"]                            = {"hipMemsetD16", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD16Async"]                          = {"hipMemsetD16Async", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD2D16_v2"]                          = {"hipMemsetD2D16", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD2D16Async"]                        = {"hipMemsetD2D16Async", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD2D32_v2"]                          = {"hipMemsetD2D32", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD2D32Async"]                        = {"hipMemsetD2D32Async", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD2D8_v2"]                           = {"hipMemsetD2D8", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD2D8Async"]                         = {"hipMemsetD2D8Async", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD32_v2"]                            = {"hipMemset", CONV_MEM, API_DRIVER};                                  // API_Runtime ANALOGUE (cudaMemset)
    cuda2hipRename["cuMemsetD32Async"]                          = {"hipMemsetAsync", CONV_MEM, API_DRIVER};                             // API_Runtime ANALOGUE (cudaMemsetAsync)
    cuda2hipRename["cuMemsetD8_v2"]                             = {"hipMemsetD8", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMemsetD8Async"]                           = {"hipMemsetD8Async", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMipmappedArrayCreate"]                    = {"hipMipmappedArrayCreate", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMipmappedArrayDestroy"]                   = {"hipMipmappedArrayDestroy", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuMipmappedArrayGetLevel"]                  = {"hipMipmappedArrayGetLevel", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};

    // Unified Addressing
    cuda2hipRename["cuMemPrefetchAsync"]                        = {"hipMemPrefetchAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};    // [CUDA 8.0.44] // no API_Runtime ANALOGUE (cudaMemPrefetchAsync has different signature)
    cuda2hipRename["cuMemAdvise"]                               = {"hipMemAdvise", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};             // [CUDA 8.0.44] // API_Runtime ANALOGUE (cudaMemAdvise)
    cuda2hipRename["cuMemRangeGetAttribute"]                    = {"hipMemRangeGetAttribute", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};  // [CUDA 8.0.44] // API_Runtime ANALOGUE (cudaMemRangeGetAttribute)
    cuda2hipRename["cuMemRangeGetAttributes"]                   = {"hipMemRangeGetAttributes", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}; // [CUDA 8.0.44] // API_Runtime ANALOGUE (cudaMemRangeGetAttributes)
    cuda2hipRename["cuPointerGetAttribute"]                     = {"hipPointerGetAttribute", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuPointerGetAttributes"]                    = {"hipPointerGetAttributes", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuPointerSetAttribute"]                     = {"hipPointerSetAttribute", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED};

    // Texture Reference Mngmnt
    // Texture reference filtering modes
    cuda2hipRename["CUfilter_mode"]                             = {"hipTextureFilterMode", CONV_TEX, API_DRIVER};                         // API_Runtime ANALOGUE (cudaTextureFilterMode)
    // ToDo:
    // cuda2hipRename["CUfilter_mode"]                          = {"CUfilter_mode_enum", CONV_TEX, API_DRIVER};                           // API_Runtime ANALOGUE (cudaTextureFilterMode)
    cuda2hipRename["CU_TR_FILTER_MODE_POINT"]                   = {"hipFilterModePoint", CONV_TEX, API_DRIVER};                      // 0 // API_Runtime ANALOGUE (cudaFilterModePoint = 0)
    cuda2hipRename["CU_TR_FILTER_MODE_LINEAR"]                  = {"hipFilterModeLinear", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};    // 1 // API_Runtime ANALOGUE (cudaFilterModeLinear = 1)

    cuda2hipRename["cuTexRefGetAddress"]                        = {"hipTexRefGetAddress", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefGetAddressMode"]                    = {"hipTexRefGetAddressMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefGetArray"]                          = {"hipTexRefGetArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefGetBorderColor"]                    = {"hipTexRefGetBorderColor", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}; // [CUDA 8.0.44] // no API_Runtime ANALOGUE
    cuda2hipRename["cuTexRefGetFilterMode"]                     = {"hipTexRefGetFilterMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefGetFlags"]                          = {"hipTexRefGetFlags", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefGetFormat"]                         = {"hipTexRefGetFormat", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefGetMaxAnisotropy"]                  = {"hipTexRefGetMaxAnisotropy", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefGetMipmapFilterMode"]               = {"hipTexRefGetMipmapFilterMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefGetMipmapLevelBias"]                = {"hipTexRefGetMipmapLevelBias", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefGetMipmapLevelClamp"]               = {"hipTexRefGetMipmapLevelClamp", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefGetMipmappedArray"]                 = {"hipTexRefGetMipmappedArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetAddress"]                        = {"hipTexRefSetAddress", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetAddress2D"]                      = {"hipTexRefSetAddress2D", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetAddressMode"]                    = {"hipTexRefSetAddressMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetArray"]                          = {"hipTexRefSetArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetBorderColor"]                    = {"hipTexRefSetBorderColor", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}; // [CUDA 8.0.44] // no API_Runtime ANALOGUE
    cuda2hipRename["cuTexRefSetFilterMode"]                     = {"hipTexRefSetFilterMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetFlags"]                          = {"hipTexRefSetFlags", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetFormat"]                         = {"hipTexRefSetFormat", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetMaxAnisotropy"]                  = {"hipTexRefSetMaxAnisotropy", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetMipmapFilterMode"]               = {"hipTexRefSetMipmapFilterMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetMipmapLevelBias"]                = {"hipTexRefSetMipmapLevelBias", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetMipmapLevelClamp"]               = {"hipTexRefSetMipmapLevelClamp", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefSetMipmappedArray"]                 = {"hipTexRefSetMipmappedArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};

    // Texture Reference Mngmnt [DEPRECATED]
    cuda2hipRename["cuTexRefCreate"]                            = {"hipTexRefCreate", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexRefDestroy"]                           = {"hipTexRefDestroy", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};

    // Surface Reference Mngmnt
    cuda2hipRename["cuSurfRefGetArray"]                         = {"hipSurfRefGetArray", CONV_SURFACE, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuSurfRefSetArray"]                         = {"hipSurfRefSetArray", CONV_SURFACE, API_DRIVER, HIP_UNSUPPORTED};

    // Texture Object Mngmnt
    cuda2hipRename["cuTexObjectCreate"]                         = {"hipTexObjectCreate", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexObjectDestroy"]                        = {"hipTexObjectDestroy", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexObjectGetResourceDesc"]                = {"hipTexObjectGetResourceDesc", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexObjectGetResourceViewDesc"]            = {"hipTexObjectGetResourceViewDesc", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuTexObjectGetTextureDesc"]                 = {"hipTexObjectGetTextureDesc", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};

    // Surface Object Mngmnt
    cuda2hipRename["cuSurfObjectCreate"]                        = {"hipSurfObjectCreate", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuSurfObjectDestroy"]                       = {"hipSurfObjectDestroy", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["cuSurfObjectGetResourceDesc"]               = {"hipSurfObjectGetResourceDesc", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED};

    // Graphics Interoperability
    cuda2hipRename["cuGraphicsMapResources"]                    = {"hipGraphicsMapResources", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED};                     // API_Runtime ANALOGUE (cudaGraphicsMapResources)
    cuda2hipRename["cuGraphicsResourceGetMappedMipmappedArray"] = {"hipGraphicsResourceGetMappedMipmappedArray", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED};  // API_Runtime ANALOGUE (cudaGraphicsResourceGetMappedMipmappedArray)
    cuda2hipRename["cuGraphicsResourceGetMappedPointer"]        = {"hipGraphicsResourceGetMappedPointer", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED};         // API_Runtime ANALOGUE (cudaGraphicsResourceGetMappedPointer)
    cuda2hipRename["cuGraphicsResourceSetMapFlags"]             = {"hipGraphicsResourceSetMapFlags", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED};              // API_Runtime ANALOGUE (cudaGraphicsResourceSetMapFlags)
    cuda2hipRename["cuGraphicsSubResourceGetMappedArray"]       = {"hipGraphicsSubResourceGetMappedArray", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED};        // API_Runtime ANALOGUE (cudaGraphicsSubResourceGetMappedArray)
    cuda2hipRename["cuGraphicsUnmapResources"]                  = {"hipGraphicsUnmapResources", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED};                    // API_Runtime ANALOGUE (cudaGraphicsUnmapResources)
    cuda2hipRename["cuGraphicsUnregisterResource"]              = {"hipGraphicsUnregisterResource", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED};                // API_Runtime ANALOGUE (cudaGraphicsUnregisterResource)

    // Profiler
    cuda2hipRename["cuProfilerInitialize"]                      = {"hipProfilerInitialize", CONV_OTHER, API_DRIVER, HIP_UNSUPPORTED};                           // API_Runtime ANALOGUE (cudaProfilerInitialize)
    cuda2hipRename["cuProfilerStart"]                           = {"hipProfilerStart", CONV_OTHER, API_DRIVER};                                                 // API_Runtime ANALOGUE (cudaProfilerStart)
    cuda2hipRename["cuProfilerStop"]                            = {"hipProfilerStop", CONV_OTHER, API_DRIVER};                                                  // API_Runtime ANALOGUE (cudaProfilerStop)

    // OpenGL Interoperability
    // enum CUGLDeviceList/CUGLDeviceList_enum
    cuda2hipRename["CUGLDeviceList"]                            = {"hipGLDeviceList", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                                     // API_Runtime ANALOGUE (cudaGLDeviceList)
    // cuda2hipRename["CUGLDeviceList_enum"]                    = {"hipGLDeviceList", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_GL_DEVICE_LIST_ALL"]                     = {"HIP_GL_DEVICE_LIST_ALL", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                      // 0x01 // API_Runtime ANALOGUE (cudaGLDeviceListAll)
    cuda2hipRename["CU_GL_DEVICE_LIST_CURRENT_FRAME"]           = {"HIP_GL_DEVICE_LIST_CURRENT_FRAME", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};            // 0x02 // API_Runtime ANALOGUE (cudaGLDeviceListCurrentFrame)
    cuda2hipRename["CU_GL_DEVICE_LIST_NEXT_FRAME"]              = {"HIP_GL_DEVICE_LIST_NEXT_FRAME", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};               // 0x03 // API_Runtime ANALOGUE (cudaGLDeviceListNextFrame)

    cuda2hipRename["cuGLGetDevices"]                            = {"hipGLGetDevices", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                                     // API_Runtime ANALOGUE (cudaGLGetDevices)
    cuda2hipRename["cuGraphicsGLRegisterBuffer"]                = {"hipGraphicsGLRegisterBuffer", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                         // API_Runtime ANALOGUE (cudaGraphicsGLRegisterBuffer)
    cuda2hipRename["cuGraphicsGLRegisterImage"]                 = {"hipGraphicsGLRegisterImage", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                          // API_Runtime ANALOGUE (cudaGraphicsGLRegisterImage)
    cuda2hipRename["cuWGLGetDevice"]                            = {"hipWGLGetDevice", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                                     // API_Runtime ANALOGUE (cudaWGLGetDevice)

    // OpenGL Interoperability [DEPRECATED]
    // enum CUGLmap_flags/CUGLmap_flags_enum
    cuda2hipRename["CUGLmap_flags"]                             = {"hipGLMapFlags", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                                       // API_Runtime ANALOGUE (cudaGLMapFlags)
    // cuda2hipRename["CUGLmap_flags_enum"]                     = {"hipGLMapFlags", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_GL_MAP_RESOURCE_FLAGS_NONE"]             = {"HIP_GL_MAP_RESOURCE_FLAGS_NONE", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};              // 0x00 // API_Runtime ANALOGUE (cudaGLMapFlagsNone)
    cuda2hipRename["CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY"]        = {"HIP_GL_MAP_RESOURCE_FLAGS_READ_ONLY", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};         // 0x01 // API_Runtime ANALOGUE (cudaGLMapFlagsReadOnly)
    cuda2hipRename["CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD"]    = {"HIP_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};     // 0x02 // API_Runtime ANALOGUE (cudaGLMapFlagsWriteDiscard)

    cuda2hipRename["cuGLCtxCreate"]                             = {"hipGLCtxCreate", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                                      // no API_Runtime ANALOGUE
    cuda2hipRename["cuGLInit"]                                  = {"hipGLInit", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                                           // no API_Runtime ANALOGUE
    cuda2hipRename["cuGLMapBufferObject"]                       = {"hipGLMapBufferObject", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                                // Not equal to cudaGLMapBufferObject due to different signatures
    cuda2hipRename["cuGLMapBufferObjectAsync"]                  = {"hipGLMapBufferObjectAsync", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                           // Not equal to cudaGLMapBufferObjectAsync due to different signatures
    cuda2hipRename["cuGLRegisterBufferObject"]                  = {"hipGLRegisterBufferObject", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                           // API_Runtime ANALOGUE (cudaGLRegisterBufferObject)
    cuda2hipRename["cuGLSetBufferObjectMapFlags"]               = {"hipGLSetBufferObjectMapFlags", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                        // API_Runtime ANALOGUE (cudaGLSetBufferObjectMapFlags)
    cuda2hipRename["cuGLUnmapBufferObject"]                     = {"hipGLUnmapBufferObject", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                              // API_Runtime ANALOGUE (cudaGLUnmapBufferObject)
    cuda2hipRename["cuGLUnmapBufferObjectAsync"]                = {"hipGLUnmapBufferObjectAsync", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                         // API_Runtime ANALOGUE (cudaGLUnmapBufferObjectAsync)
    cuda2hipRename["cuGLUnregisterBufferObject"]                = {"hipGLUnregisterBufferObject", CONV_GL, API_DRIVER, HIP_UNSUPPORTED};                         // API_Runtime ANALOGUE (cudaGLUnregisterBufferObject)

    // Direct3D 9 Interoperability
    // enum CUd3d9DeviceList/CUd3d9DeviceList_enum
    cuda2hipRename["CUd3d9DeviceList"]                          = {"hipD3D9DeviceList", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                                 // API_Runtime ANALOGUE (cudaD3D9DeviceList)
    // cuda2hipRename["CUd3d9DeviceList_enum"]                  = {"hipD3D9DeviceList", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_D3D9_DEVICE_LIST_ALL"]                   = {"HIP_D3D9_DEVICE_LIST_ALL", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                  // 0x01 // API_Runtime ANALOGUE (cudaD3D9DeviceListAll)
    cuda2hipRename["CU_D3D9_DEVICE_LIST_CURRENT_FRAME"]         = {"HIP_D3D9_DEVICE_LIST_CURRENT_FRAME", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};        // 0x02 // API_Runtime ANALOGUE (cudaD3D9DeviceListCurrentFrame)
    cuda2hipRename["CU_D3D9_DEVICE_LIST_NEXT_FRAME"]            = {"HIP_D3D9_DEVICE_LIST_NEXT_FRAME", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};           // 0x03 // API_Runtime ANALOGUE (cudaD3D9DeviceListNextFrame)

    cuda2hipRename["cuD3D9CtxCreate"]                           = {"hipD3D9CtxCreate", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                                  // no API_Runtime ANALOGUE
    cuda2hipRename["cuD3D9CtxCreateOnDevice"]                   = {"hipD3D9CtxCreateOnDevice", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                          // no API_Runtime ANALOGUE
    cuda2hipRename["cuD3D9GetDevice"]                           = {"hipD3D9GetDevice", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                                  // API_Runtime ANALOGUE (cudaD3D9GetDevice)
    cuda2hipRename["cuD3D9GetDevices"]                          = {"hipD3D9GetDevices", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                                 // API_Runtime ANALOGUE (cudaD3D9GetDevices)
    cuda2hipRename["cuD3D9GetDirect3DDevice"]                   = {"hipD3D9GetDirect3DDevice", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                          // API_Runtime ANALOGUE (cudaD3D9GetDirect3DDevice)
    cuda2hipRename["cuGraphicsD3D9RegisterResource"]            = {"hipGraphicsD3D9RegisterResource", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                   // API_Runtime ANALOGUE (cudaGraphicsD3D9RegisterResource)

    // Direct3D 9 Interoperability [DEPRECATED]
    // enum CUd3d9map_flags/CUd3d9map_flags_enum
    cuda2hipRename["CUd3d9map_flags"]                           = {"hipD3D9MapFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                                   // API_Runtime ANALOGUE (cudaD3D9MapFlags)
    // cuda2hipRename["CUd3d9map_flags_enum"]                   = {"hipD3D9MapFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_D3D9_MAPRESOURCE_FLAGS_NONE"]            = {"HIP_D3D9_MAPRESOURCE_FLAGS_NONE", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};           // 0x00 // API_Runtime ANALOGUE (cudaD3D9MapFlagsNone)
    cuda2hipRename["CU_D3D9_MAPRESOURCE_FLAGS_READONLY"]        = {"HIP_D3D9_MAPRESOURCE_FLAGS_READONLY", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};       // 0x01 // API_Runtime ANALOGUE (cudaD3D9MapFlagsReadOnly)
    cuda2hipRename["CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD"]    = {"HIP_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};   // 0x02 // API_Runtime ANALOGUE (cudaD3D9MapFlagsWriteDiscard)

    // enum CUd3d9register_flags/CUd3d9register_flags_enum
    cuda2hipRename["CUd3d9register_flags"]                      = {"hipD3D9RegisterFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                              // API_Runtime ANALOGUE (cudaD3D9RegisterFlags)
    // cuda2hipRename["CUd3d9register_flags_enum"]              = {"hipD3D9RegisterFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_D3D9_REGISTER_FLAGS_NONE"]               = {"HIP_D3D9_REGISTER_FLAGS_NONE", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};              // 0x00 // API_Runtime ANALOGUE (cudaD3D9RegisterFlagsNone)
    cuda2hipRename["CU_D3D9_REGISTER_FLAGS_ARRAY"]              = {"HIP_D3D9_REGISTER_FLAGS_ARRAY", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};             // 0x01 // API_Runtime ANALOGUE (cudaD3D9RegisterFlagsArray)

    cuda2hipRename["cuD3D9MapResources"]                        = {"hipD3D9MapResources", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                               // API_Runtime ANALOGUE (cudaD3D9MapResources)
    cuda2hipRename["cuD3D9RegisterResource"]                    = {"hipD3D9RegisterResource", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                           // API_Runtime ANALOGUE (cudaD3D9RegisterResource)
    cuda2hipRename["cuD3D9ResourceGetMappedArray"]              = {"hipD3D9ResourceGetMappedArray", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                     // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedArray)
    cuda2hipRename["cuD3D9ResourceGetMappedPitch"]              = {"hipD3D9ResourceGetMappedPitch", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                     // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedPitch)
    cuda2hipRename["cuD3D9ResourceGetMappedPointer"]            = {"hipD3D9ResourceGetMappedPointer", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                   // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedPointer)
    cuda2hipRename["cuD3D9ResourceGetMappedSize"]               = {"hipD3D9ResourceGetMappedSize", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                      // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedSize)
    cuda2hipRename["cuD3D9ResourceGetSurfaceDimensions"]        = {"hipD3D9ResourceGetSurfaceDimensions", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};               // API_Runtime ANALOGUE (cudaD3D9ResourceGetSurfaceDimensions)
    cuda2hipRename["cuD3D9ResourceSetMapFlags"]                 = {"hipD3D9ResourceSetMapFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                        // API_Runtime ANALOGUE (cudaD3D9ResourceSetMapFlags)
    cuda2hipRename["cuD3D9UnmapResources"]                      = {"hipD3D9UnmapResources", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                             // API_Runtime ANALOGUE (cudaD3D9UnmapResources)
    cuda2hipRename["cuD3D9UnregisterResource"]                  = {"hipD3D9UnregisterResource", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED};                         // API_Runtime ANALOGUE (cudaD3D9UnregisterResource)

    // Direct3D 10 Interoperability
    // enum CUd3d10DeviceList/CUd3d10DeviceList_enum
    cuda2hipRename["CUd3d10DeviceList"]                         = {"hipd3d10DeviceList", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                               // API_Runtime ANALOGUE (cudaD3D10DeviceList)
    // cuda2hipRename["CUd3d10DeviceList_enum"]                 = {"hipD3D10DeviceList", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_D3D10_DEVICE_LIST_ALL"]                  = {"HIP_D3D10_DEVICE_LIST_ALL", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                // 0x01 // API_Runtime ANALOGUE (cudaD3D10DeviceListAll)
    cuda2hipRename["CU_D3D10_DEVICE_LIST_CURRENT_FRAME"]        = {"HIP_D3D10_DEVICE_LIST_CURRENT_FRAME", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};      // 0x02 // API_Runtime ANALOGUE (cudaD3D10DeviceListCurrentFrame)
    cuda2hipRename["CU_D3D10_DEVICE_LIST_NEXT_FRAME"]           = {"HIP_D3D10_DEVICE_LIST_NEXT_FRAME", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};         // 0x03 // API_Runtime ANALOGUE (cudaD3D10DeviceListNextFrame)

    cuda2hipRename["cuD3D10GetDevice"]                          = {"hipD3D10GetDevice", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                                // API_Runtime ANALOGUE (cudaD3D10GetDevice)
    cuda2hipRename["cuD3D10GetDevices"]                         = {"hipD3D10GetDevices", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                               // API_Runtime ANALOGUE (cudaD3D10GetDevices)
    cuda2hipRename["cuGraphicsD3D10RegisterResource"]           = {"hipGraphicsD3D10RegisterResource", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                 // API_Runtime ANALOGUE (cudaGraphicsD3D10RegisterResource)

    // Direct3D 10 Interoperability [DEPRECATED]
    // enum CUd3d10map_flags/CUd3d10map_flags_enum
    cuda2hipRename["CUd3d10map_flags"]                          = {"hipD3D10MapFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                                 // API_Runtime ANALOGUE (cudaD3D10MapFlags)
    // cuda2hipRename["CUd3d10map_flags_enum"]                  = {"hipD3D10MapFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_D3D10_MAPRESOURCE_FLAGS_NONE"]           = {"HIP_D3D10_MAPRESOURCE_FLAGS_NONE", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};         // 0x00 // API_Runtime ANALOGUE (cudaD3D10MapFlagsNone)
    cuda2hipRename["CU_D3D10_MAPRESOURCE_FLAGS_READONLY"]       = {"HIP_D3D10_MAPRESOURCE_FLAGS_READONLY", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};     // 0x01 // API_Runtime ANALOGUE (cudaD3D10MapFlagsReadOnly)
    cuda2hipRename["CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD"]   = {"HIP_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}; // 0x02 // API_Runtime ANALOGUE (cudaD3D10MapFlagsWriteDiscard)

    // enum CUd3d10register_flags/CUd3d10register_flags_enum
    cuda2hipRename["CUd3d10register_flags"]                     = {"hipD3D10RegisterFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                            // API_Runtime ANALOGUE (cudaD3D10RegisterFlags)
    // cuda2hipRename["CUd3d10register_flags_enum"]             = {"hipD3D10RegisterFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_D3D10_REGISTER_FLAGS_NONE"]              = {"HIP_D3D10_REGISTER_FLAGS_NONE", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};            // 0x00 // API_Runtime ANALOGUE (cudaD3D10RegisterFlagsNone)
    cuda2hipRename["CU_D3D10_REGISTER_FLAGS_ARRAY"]             = {"HIP_D3D10_REGISTER_FLAGS_ARRAY", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};           // 0x01 // API_Runtime ANALOGUE (cudaD3D10RegisterFlagsArray)

    cuda2hipRename["cuD3D10CtxCreate"]                          = {"hipD3D10CtxCreate", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                                // no API_Runtime ANALOGUE
    cuda2hipRename["cuD3D10CtxCreateOnDevice"]                  = {"hipD3D10CtxCreateOnDevice", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                        // no API_Runtime ANALOGUE
    cuda2hipRename["cuD3D10GetDirect3DDevice"]                  = {"hipD3D10GetDirect3DDevice", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                        // API_Runtime ANALOGUE (cudaD3D10GetDirect3DDevice)
    cuda2hipRename["cuD3D10MapResources"]                       = {"hipD3D10MapResources", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                             // API_Runtime ANALOGUE (cudaD3D10MapResources)
    cuda2hipRename["cuD3D10RegisterResource"]                   = {"hipD3D10RegisterResource", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                         // API_Runtime ANALOGUE (cudaD3D10RegisterResource)
    cuda2hipRename["cuD3D10ResourceGetMappedArray"]             = {"hipD3D10ResourceGetMappedArray", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                   // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedArray)
    cuda2hipRename["cuD3D10ResourceGetMappedPitch"]             = {"hipD3D10ResourceGetMappedPitch", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                   // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedPitch)
    cuda2hipRename["cuD3D10ResourceGetMappedPointer"]           = {"hipD3D10ResourceGetMappedPointer", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                 // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedPointer)
    cuda2hipRename["cuD3D10ResourceGetMappedSize"]              = {"hipD3D10ResourceGetMappedSize", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                    // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedSize)
    cuda2hipRename["cuD3D10ResourceGetSurfaceDimensions"]       = {"hipD3D10ResourceGetSurfaceDimensions", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};             // API_Runtime ANALOGUE (cudaD3D10ResourceGetSurfaceDimensions)
    cuda2hipRename["cuD310ResourceSetMapFlags"]                 = {"hipD3D10ResourceSetMapFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                      // API_Runtime ANALOGUE (cudaD3D10ResourceSetMapFlags)
    cuda2hipRename["cuD3D10UnmapResources"]                     = {"hipD3D10UnmapResources", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                           // API_Runtime ANALOGUE (cudaD3D10UnmapResources)
    cuda2hipRename["cuD3D10UnregisterResource"]                 = {"hipD3D10UnregisterResource", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED};                       // API_Runtime ANALOGUE (cudaD3D10UnregisterResource)

    // Direct3D 11 Interoperability
    // enum CUd3d11DeviceList/CUd3d11DeviceList_enum
    cuda2hipRename["CUd3d11DeviceList"]                         = {"hipd3d11DeviceList", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED};                               // API_Runtime ANALOGUE (cudaD3D11DeviceList)
    // cuda2hipRename["CUd3d11DeviceList_enum"]                 = {"hipD3D11DeviceList", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED};
    cuda2hipRename["CU_D3D11_DEVICE_LIST_ALL"]                  = {"HIP_D3D11_DEVICE_LIST_ALL", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED};                // 0x01 // API_Runtime ANALOGUE (cudaD3D11DeviceListAll)
    cuda2hipRename["CU_D3D11_DEVICE_LIST_CURRENT_FRAME"]        = {"HIP_D3D11_DEVICE_LIST_CURRENT_FRAME", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED};      // 0x02 // API_Runtime ANALOGUE (cudaD3D11DeviceListCurrentFrame)
    cuda2hipRename["CU_D3D11_DEVICE_LIST_NEXT_FRAME"]           = {"HIP_D3D11_DEVICE_LIST_NEXT_FRAME", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED};         // 0x03 // API_Runtime ANALOGUE (cudaD3D11DeviceListNextFrame)

    cuda2hipRename["cuD3D11GetDevice"]                          = {"hipD3D11GetDevice", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED};                                // API_Runtime ANALOGUE (cudaD3D11GetDevice)
    cuda2hipRename["cuD3D11GetDevices"]                         = {"hipD3D11GetDevices", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED};                               // API_Runtime ANALOGUE (cudaD3D11GetDevices)
    cuda2hipRename["cuGraphicsD3D11RegisterResource"]           = {"hipGraphicsD3D11RegisterResource", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED};                 // API_Runtime ANALOGUE (cudaGraphicsD3D11RegisterResource)

    // Direct3D 11 Interoperability [DEPRECATED]
    cuda2hipRename["cuD3D11CtxCreate"]                          = {"hipD3D11CtxCreate", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED};                                // no API_Runtime ANALOGUE
    cuda2hipRename["cuD3D11CtxCreateOnDevice"]                  = {"hipD3D11CtxCreateOnDevice", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED};                        // no API_Runtime ANALOGUE
    cuda2hipRename["cuD3D11GetDirect3DDevice"]                  = {"hipD3D11GetDirect3DDevice", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED};                        // API_Runtime ANALOGUE (cudaD3D11GetDirect3DDevice)

    // VDPAU Interoperability
    cuda2hipRename["cuGraphicsVDPAURegisterOutputSurface"]      = {"hipGraphicsVDPAURegisterOutputSurface", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED};            // API_Runtime ANALOGUE (cudaGraphicsVDPAURegisterOutputSurface)
    cuda2hipRename["cuGraphicsVDPAURegisterVideoSurface"]       = {"hipGraphicsVDPAURegisterVideoSurface", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED};             // API_Runtime ANALOGUE (cudaGraphicsVDPAURegisterVideoSurface)
    cuda2hipRename["cuVDPAUGetDevice"]                          = {"hipVDPAUGetDevice", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED};                                // API_Runtime ANALOGUE (cudaVDPAUGetDevice)
    cuda2hipRename["cuVDPAUCtxCreate"]                          = {"hipVDPAUCtxCreate", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED};                                // no API_Runtime ANALOGUE

    // EGL Interoperability
    cuda2hipRename["CUeglStreamConnection_st"]                  = {"hipEglStreamConnection", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};                             // API_Runtime ANALOGUE (cudaEglStreamConnection)
    cuda2hipRename["CUeglStreamConnection"]                     = {"hipEglStreamConnection", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};                             // API_Runtime ANALOGUE (cudaEglStreamConnection)

    cuda2hipRename["cuEGLStreamConsumerAcquireFrame"]           = {"hipEGLStreamConsumerAcquireFrame", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};                   // API_Runtime ANALOGUE (cudaEGLStreamConsumerAcquireFrame)
    cuda2hipRename["cuEGLStreamConsumerConnect"]                = {"hipEGLStreamConsumerConnect", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};                        // API_Runtime ANALOGUE (cudaEGLStreamConsumerConnect)
    cuda2hipRename["cuEGLStreamConsumerConnectWithFlags"]       = {"hipEGLStreamConsumerConnectWithFlags", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};               // API_Runtime ANALOGUE (cudaEGLStreamConsumerConnectWithFlags)
    cuda2hipRename["cuEGLStreamConsumerDisconnect"]             = {"hipEGLStreamConsumerDisconnect", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};                     // no API_Runtime ANALOGUE
    cuda2hipRename["cuEGLStreamConsumerReleaseFrame"]           = {"hipEGLStreamConsumerReleaseFrame", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};                   // API_Runtime ANALOGUE (cudaEGLStreamConsumerReleaseFrame)
    cuda2hipRename["cuEGLStreamProducerConnect"]                = {"hipEGLStreamProducerConnect", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};                        // API_Runtime ANALOGUE (cudaEGLStreamProducerConnect)
    cuda2hipRename["cuEGLStreamProducerDisconnect"]             = {"hipEGLStreamProducerDisconnect", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};                     // API_Runtime ANALOGUE (cudaEGLStreamProducerDisconnect)
    cuda2hipRename["cuEGLStreamProducerPresentFrame"]           = {"hipEGLStreamProducerPresentFrame", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};                   // API_Runtime ANALOGUE (cudaEGLStreamProducerPresentFrame)
    cuda2hipRename["cuEGLStreamProducerReturnFrame"]            = {"hipEGLStreamProducerReturnFrame", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};                    // API_Runtime ANALOGUE (cudaEGLStreamProducerReturnFrame)
    cuda2hipRename["cuGraphicsEGLRegisterImage"]                = {"hipGraphicsEGLRegisterImage", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};                        // API_Runtime ANALOGUE (cudaGraphicsEGLRegisterImage)
    cuda2hipRename["cuGraphicsResourceGetMappedEglFrame"]       = {"hipGraphicsResourceGetMappedEglFrame", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED};               // API_Runtime ANALOGUE (cudaGraphicsResourceGetMappedEglFrame)

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

    // defines
    cuda2hipRename["cudaMemAttachGlobal"]                        = {"hipMemAttachGlobal", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                       // 0x01 // API_Driver ANALOGUE (CU_MEM_ATTACH_GLOBAL = 0x1)
    cuda2hipRename["cudaMemAttachHost"]                          = {"hipMemAttachHost", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                         // 0x02 // API_Driver ANALOGUE (CU_MEM_ATTACH_HOST = 0x2)
    cuda2hipRename["cudaMemAttachSingle"]                        = {"hipMemAttachSingle", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                       // 0x04 // API_Driver ANALOGUE (CU_MEM_ATTACH_SINGLE = 0x4)

    cuda2hipRename["cudaOccupancyDefault"]                       = {"hipOccupancyDefault", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                      // 0x00 // API_Driver ANALOGUE (CU_OCCUPANCY_DEFAULT = 0x0)
    cuda2hipRename["cudaOccupancyDisableCachingOverride"]        = {"hipOccupancyDisableCachingOverride", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 0x01 // API_Driver ANALOGUE (CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 0x1)

    cuda2hipRename["cudaStreamCallback_t"]                       = {"hipStreamCallback_t", CONV_TYPE, API_RUNTIME};

    // Error API
    cuda2hipRename["cudaGetLastError"]            = {"hipGetLastError", CONV_ERROR, API_RUNTIME};
    cuda2hipRename["cudaPeekAtLastError"]         = {"hipPeekAtLastError", CONV_ERROR, API_RUNTIME};
    cuda2hipRename["cudaGetErrorName"]            = {"hipGetErrorName", CONV_ERROR, API_RUNTIME};
    cuda2hipRename["cudaGetErrorString"]          = {"hipGetErrorString", CONV_ERROR, API_RUNTIME};

    // Arrays
    cuda2hipRename["cudaArray"]                   = {"hipArray", CONV_MEM, API_RUNTIME};
    // typedef struct cudaArray *cudaArray_t;
    cuda2hipRename["cudaArray_t"]                 = {"hipArray_t", CONV_MEM, API_RUNTIME};
    // typedef const struct cudaArray *cudaArray_const_t;
    cuda2hipRename["cudaArray_const_t"]           = {"hipArray_const_t", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMipmappedArray_t"]        = {"hipMipmappedArray_t", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMipmappedArray_const_t"]  = {"hipMipmappedArray_const_t", CONV_MEM, API_RUNTIME};

    // memcpy
    // memcpy structs
    cuda2hipRename["cudaMemcpy3DParms"]           = {"hipMemcpy3DParms", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpy3DPeerParms"]       = {"hipMemcpy3DPeerParms", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};

      // memcpy functions
    cuda2hipRename["cudaMemcpy"]                  = {"hipMemcpy", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyToArray"]           = {"hipMemcpyToArray", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyToSymbol"]          = {"hipMemcpyToSymbol", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyToSymbolAsync"]     = {"hipMemcpyToSymbolAsync", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyAsync"]             = {"hipMemcpyAsync", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpy2D"]                = {"hipMemcpy2D", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpy2DAsync"]           = {"hipMemcpy2DAsync", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpy2DToArray"]         = {"hipMemcpy2DToArray", CONV_MEM, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaMemcpy2DArrayToArray"]    = {"hipMemcpy2DArrayToArray", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemcpy2DFromArray"]       = {"hipMemcpy2DFromArray", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemcpy2DFromArrayAsync"]  = {"hipMemcpy2DFromArrayAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemcpy2DToArrayAsync"]    = {"hipMemcpy2DToArrayAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemcpy3D"]                = {"hipMemcpy3D", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpy3DAsync"]           = {"hipMemcpy3DAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemcpy3DPeer"]            = {"hipMemcpy3DPeer", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemcpy3DPeerAsync"]       = {"hipMemcpy3DPeerAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemcpyArrayToArray"]      = {"hipMemcpyArrayToArray", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemcpyFromArrayAsync"]    = {"hipMemcpyFromArrayAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemcpyFromSymbol"]        = {"hipMemcpyFromSymbol", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyFromSymbolAsync"]   = {"hipMemcpyFromSymbolAsync", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemAdvise"]               = {"hipMemAdvise", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}; // [CUDA 8.0.44]
    cuda2hipRename["cudaMemRangeGetAttribute"]    = {"hipMemRangeGetAttribute", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}; // [CUDA 8.0.44]
    cuda2hipRename["cudaMemRangeGetAttributes"]   = {"hipMemRangeGetAttributes", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}; // [CUDA 8.0.44]

    // unsupported yet by HIP [CUDA 8.0.44]
    // Memory advise values
    cuda2hipRename["cudaMemoryAdvise"]                                            = {"hipMemAdvise", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                                  // API_Driver ANALOGUE (CUmem_advise)
    cuda2hipRename["cudaMemAdviseSetReadMostly"]                                  = {"hipMemAdviseSetReadMostly", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                // 1 // API_Driver ANALOGUE (CU_MEM_ADVISE_SET_READ_MOSTLY = 1)
    cuda2hipRename["cudaMemAdviseUnsetReadMostly"]                                = {"hipMemAdviseUnsetReadMostly", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};              // 2 // API_Driver ANALOGUE (CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2)
    cuda2hipRename["cudaMemAdviseSetPreferredLocation"]                           = {"hipMemAdviseSetPreferredLocation", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 3 // API_Driver ANALOGUE (CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3)
    cuda2hipRename["cudaMemAdviseUnsetPreferredLocation"]                         = {"hipMemAdviseUnsetPreferredLocation", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 4 // API_Driver ANALOGUE (CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4)
    cuda2hipRename["cudaMemAdviseSetAccessedBy"]                                  = {"hipMemAdviseSetAccessedBy", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                // 5 // API_Driver ANALOGUE (CU_MEM_ADVISE_SET_ACCESSED_BY = 5)
    cuda2hipRename["cudaMemAdviseUnsetAccessedBy"]                                = {"hipMemAdviseUnsetAccessedBy", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};              // 6 // API_Driver ANALOGUE (CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6)
    // CUmem_range_attribute
    cuda2hipRename["cudaMemRangeAttribute"]                                       = {"hipMemRangeAttribute", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                          // API_Driver ANALOGUE (CUmem_range_attribute)
    cuda2hipRename["cudaMemRangeAttributeReadMostly"]                             = {"hipMemRangeAttributeReadMostly", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 1 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1)
    cuda2hipRename["cudaMemRangeAttributePreferredLocation"]                      = {"hipMemRangeAttributePreferredLocation", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};    // 2 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2)
    cuda2hipRename["cudaMemRangeAttributeAccessedBy"]                             = {"hipMemRangeAttributeAccessedBy", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 3 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3)
    cuda2hipRename["cudaMemRangeAttributeLastPrefetchLocation"]                   = {"hipMemRangeAttributeLastPrefetchLocation", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}; // 4 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4)

    // memcpy kind
    cuda2hipRename["cudaMemcpyKind"]              = {"hipMemcpyKind", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyHostToHost"]        = {"hipMemcpyHostToHost", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyHostToDevice"]      = {"hipMemcpyHostToDevice", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyDeviceToHost"]      = {"hipMemcpyDeviceToHost", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyDeviceToDevice"]    = {"hipMemcpyDeviceToDevice", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyDefault"]           = {"hipMemcpyDefault", CONV_MEM, API_RUNTIME};

    // memset
    cuda2hipRename["cudaMemset"]                  = {"hipMemset", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemsetAsync"]             = {"hipMemsetAsync", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemset2D"]                = {"hipMemset2D", CONV_MEM, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaMemset2DAsync"]           = {"hipMemset2DAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemset3D"]                = {"hipMemset3D", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemset3DAsync"]           = {"hipMemset3DAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};

    // Memory management
    cuda2hipRename["cudaMemGetInfo"]              = {"hipMemGetInfo", CONV_MEM, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaArrayGetInfo"]            = {"hipArrayGetInfo", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaFreeMipmappedArray"]      = {"hipFreeMipmappedArray", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGetMipmappedArrayLevel"]  = {"hipGetMipmappedArrayLevel", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGetSymbolAddress"]        = {"hipGetSymbolAddress", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGetSymbolSize"]           = {"hipGetSymbolSize", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMemPrefetchAsync"]        = {"hipMemPrefetchAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}; // [CUDA 8.0.44] // API_Driver ANALOGUE (cuMemPrefetchAsync)

    // malloc
    cuda2hipRename["cudaMalloc"]                  = {"hipMalloc", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMallocHost"]              = {"hipHostMalloc", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMallocArray"]             = {"hipMallocArray", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMalloc3D"]                = {"hipMalloc3D", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMalloc3DArray"]           = {"hipMalloc3DArray", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMallocManaged"]           = {"hipMallocManaged", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMallocMipmappedArray"]    = {"hipMallocMipmappedArray", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaMallocPitch"]             = {"hipMallocPitch", CONV_MEM, API_RUNTIME};

    cuda2hipRename["cudaFree"]                    = {"hipFree", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaFreeHost"]                = {"hipHostFree", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaFreeArray"]               = {"hipFreeArray", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaHostRegister"]            = {"hipHostRegister", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaHostUnregister"]          = {"hipHostUnregister", CONV_MEM, API_RUNTIME};
    // hipHostAlloc deprecated - use hipHostMalloc instead
    cuda2hipRename["cudaHostAlloc"]               = {"hipHostMalloc", CONV_MEM, API_RUNTIME};

    // Memory types
    cuda2hipRename["cudaMemoryType"]              = {"hipMemoryType", CONV_MEM, API_RUNTIME};                         // API_Driver ANALOGUE (no -  CUmemorytype is not an analogue)
    cuda2hipRename["cudaMemoryTypeHost"]          = {"hipMemoryTypeHost", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemoryTypeDevice"]        = {"hipMemoryTypeDevice", CONV_MEM, API_RUNTIME};

    // make memory functions
    cuda2hipRename["make_cudaExtent"]             = {"make_hipExtent", CONV_MEM, API_RUNTIME};
    cuda2hipRename["make_cudaPitchedPtr"]         = {"make_hipPitchedPtr", CONV_MEM, API_RUNTIME};
    cuda2hipRename["make_cudaPos"]                = {"make_hipPos", CONV_MEM, API_RUNTIME};

    cuda2hipRename["cudaExtent"]                  = {"hipExtent", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaPitchedPtr"]              = {"hipPitchedPtr", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaPos"]                     = {"hipPos", CONV_MEM, API_RUNTIME};

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
    cuda2hipRename["cudaEvent_t"]                   = {"hipEvent_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaEventCreate"]               = {"hipEventCreate", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventCreateWithFlags"]      = {"hipEventCreateWithFlags", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventDestroy"]              = {"hipEventDestroy", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventRecord"]               = {"hipEventRecord", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventElapsedTime"]          = {"hipEventElapsedTime", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventSynchronize"]          = {"hipEventSynchronize", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventQuery"]                = {"hipEventQuery", CONV_EVENT, API_RUNTIME};
    // Event Flags
    cuda2hipRename["cudaEventDefault"]              = {"hipEventDefault", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventBlockingSync"]         = {"hipEventBlockingSync", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventDisableTiming"]        = {"hipEventDisableTiming", CONV_EVENT, API_RUNTIME};
    cuda2hipRename["cudaEventInterprocess"]         = {"hipEventInterprocess", CONV_EVENT, API_RUNTIME};
    // Streams
    cuda2hipRename["cudaStream_t"]                  = {"hipStream_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaStreamCreate"]              = {"hipStreamCreate", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamCreateWithFlags"]     = {"hipStreamCreateWithFlags", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamCreateWithPriority"]  = {"hipStreamCreateWithPriority", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaStreamDestroy"]             = {"hipStreamDestroy", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamWaitEvent"]           = {"hipStreamWaitEvent", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamSynchronize"]         = {"hipStreamSynchronize", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamGetFlags"]            = {"hipStreamGetFlags", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamQuery"]               = {"hipStreamQuery", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamAddCallback"]         = {"hipStreamAddCallback", CONV_STREAM, API_RUNTIME};
    cuda2hipRename["cudaStreamAttachMemAsync"]      = {"hipStreamAttachMemAsync", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaStreamGetPriority"]         = {"hipStreamGetPriority", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED};

    // Stream Flags
    cuda2hipRename["cudaStreamDefault"]             = {"hipStreamDefault", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaStreamNonBlocking"]         = {"hipStreamNonBlocking", CONV_TYPE, API_RUNTIME};

    // Other synchronization
    cuda2hipRename["cudaDeviceSynchronize"]         = {"hipDeviceSynchronize", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaDeviceReset"]               = {"hipDeviceReset", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaSetDevice"]                 = {"hipSetDevice", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaGetDevice"]                 = {"hipGetDevice", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaGetDeviceCount"]            = {"hipGetDeviceCount", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaChooseDevice"]              = {"hipChooseDevice", CONV_DEVICE, API_RUNTIME};

    // Thread Management
    cuda2hipRename["cudaThreadExit"]                = {"hipDeviceReset", CONV_THREAD, API_RUNTIME};
    cuda2hipRename["cudaThreadGetCacheConfig"]      = {"hipDeviceGetCacheConfig", CONV_THREAD, API_RUNTIME};
    cuda2hipRename["cudaThreadGetLimit"]            = {"hipThreadGetLimit", CONV_THREAD, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaThreadSetCacheConfig"]      = {"hipDeviceSetCacheConfig", CONV_THREAD, API_RUNTIME};
    cuda2hipRename["cudaThreadSetLimit"]            = {"hipThreadSetLimit", CONV_THREAD, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaThreadSynchronize"]         = {"hipDeviceSynchronize", CONV_THREAD, API_RUNTIME};

    // Attributes
    cuda2hipRename["cudaDeviceGetAttribute"]                       = {"hipDeviceGetAttribute", CONV_DEVICE, API_RUNTIME};

    cuda2hipRename["cudaDeviceAttr"]                               = {"hipDeviceAttribute_t", CONV_TYPE, API_RUNTIME};                                                      // API_DRIVER ANALOGUE (CUdevice_attribute)
    cuda2hipRename["cudaDevAttrMaxThreadsPerBlock"]                = {"hipDeviceAttributeMaxThreadsPerBlock", CONV_TYPE, API_RUNTIME};                                 //  1 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1)
    cuda2hipRename["cudaDevAttrMaxBlockDimX"]                      = {"hipDeviceAttributeMaxBlockDimX", CONV_TYPE, API_RUNTIME};                                       //  2 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2)
    cuda2hipRename["cudaDevAttrMaxBlockDimY"]                      = {"hipDeviceAttributeMaxBlockDimY", CONV_TYPE, API_RUNTIME};                                       //  3 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3)
    cuda2hipRename["cudaDevAttrMaxBlockDimZ"]                      = {"hipDeviceAttributeMaxBlockDimZ", CONV_TYPE, API_RUNTIME};                                       //  4 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4)
    cuda2hipRename["cudaDevAttrMaxGridDimX"]                       = {"hipDeviceAttributeMaxGridDimX", CONV_TYPE, API_RUNTIME};                                        //  5 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5)
    cuda2hipRename["cudaDevAttrMaxGridDimY"]                       = {"hipDeviceAttributeMaxGridDimY", CONV_TYPE, API_RUNTIME};                                        //  6 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 6)
    cuda2hipRename["cudaDevAttrMaxGridDimZ"]                       = {"hipDeviceAttributeMaxGridDimZ", CONV_TYPE, API_RUNTIME};                                        //  7 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 7)
    cuda2hipRename["cudaDevAttrMaxSharedMemoryPerBlock"]           = {"hipDeviceAttributeMaxSharedMemoryPerBlock", CONV_TYPE, API_RUNTIME};                            //  8 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8)
    cuda2hipRename["cudaDevAttrTotalConstantMemory"]               = {"hipDeviceAttributeTotalConstantMemory", CONV_TYPE, API_RUNTIME};                                //  9 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY =9)
    cuda2hipRename["cudaDevAttrWarpSize"]                          = {"hipDeviceAttributeWarpSize", CONV_TYPE, API_RUNTIME};                                           // 10 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10)
    cuda2hipRename["cudaDevAttrMaxPitch"]                          = {"hipDeviceAttributeMaxPitch", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                          // 11 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11)
    cuda2hipRename["cudaDevAttrMaxRegistersPerBlock"]              = {"hipDeviceAttributeMaxRegistersPerBlock", CONV_TYPE, API_RUNTIME};                               // 12 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12)
    cuda2hipRename["cudaDevAttrClockRate"]                         = {"hipDeviceAttributeClockRate", CONV_TYPE, API_RUNTIME};                                          // 13 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13)
    cuda2hipRename["cudaDevAttrTextureAlignment"]                  = {"hipDeviceAttributeTextureAlignment", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                  // 14 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14)
    // Is not deprecated as CUDA Driver's API analogue CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
    cuda2hipRename["cudaDevAttrGpuOverlap"]                        = {"hipDeviceAttributeGpuOverlap", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                        // 15 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15)
    cuda2hipRename["cudaDevAttrMultiProcessorCount"]               = {"hipDeviceAttributeMultiprocessorCount", CONV_TYPE, API_RUNTIME};                                // 16 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16)
    cuda2hipRename["cudaDevAttrKernelExecTimeout"]                 = {"hipDeviceAttributeKernelExecTimeout", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 17 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17)
    cuda2hipRename["cudaDevAttrIntegrated"]                        = {"hipDeviceAttributeIntegrated", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                        // 18 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_INTEGRATED = 18)
    cuda2hipRename["cudaDevAttrCanMapHostMemory"]                  = {"hipDeviceAttributeCanMapHostMemory", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                  // 19 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19)
    cuda2hipRename["cudaDevAttrComputeMode"]                       = {"hipDeviceAttributeComputeMode", CONV_TYPE, API_RUNTIME};                                        // 20 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20)
    cuda2hipRename["cudaDevAttrMaxTexture1DWidth"]                 = {"hipDeviceAttributeMaxTexture1DWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 21 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21)
    cuda2hipRename["cudaDevAttrMaxTexture2DWidth"]                 = {"hipDeviceAttributeMaxTexture2DWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 22 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22)
    cuda2hipRename["cudaDevAttrMaxTexture2DHeight"]                = {"hipDeviceAttributeMaxTexture2DHeight", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                // 23 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23)
    cuda2hipRename["cudaDevAttrMaxTexture3DWidth"]                 = {"hipDeviceAttributeMaxTexture3DWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 24 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24)
    cuda2hipRename["cudaDevAttrMaxTexture3DHeight"]                = {"hipDeviceAttributeMaxTexture3DHeight", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                // 25 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25)
    cuda2hipRename["cudaDevAttrMaxTexture3DDepth"]                 = {"hipDeviceAttributeMaxTexture3DDepth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 26 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26)
    cuda2hipRename["cudaDevAttrMaxTexture2DLayeredWidth"]          = {"hipDeviceAttributeMaxTexture2DLayeredWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 27 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27)
    cuda2hipRename["cudaDevAttrMaxTexture2DLayeredHeight"]         = {"hipDeviceAttributeMaxTexture2DLayeredHeight", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 28 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28)
    cuda2hipRename["cudaDevAttrMaxTexture2DLayeredLayers"]         = {"hipDeviceAttributeMaxTexture2DLayeredLayers", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 29 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29)
    cuda2hipRename["cudaDevAttrSurfaceAlignment"]                  = {"hipDeviceAttributeSurfaceAlignment", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                  // 30 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30)
    cuda2hipRename["cudaDevAttrConcurrentKernels"]                 = {"hipDeviceAttributeConcurrentKernels", CONV_TYPE, API_RUNTIME};                                  // 31 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31)
    cuda2hipRename["cudaDevAttrEccEnabled"]                        = {"hipDeviceAttributeEccEnabled", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                        // 32 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32)
    cuda2hipRename["cudaDevAttrPciBusId"]                          = {"hipDeviceAttributePciBusId", CONV_TYPE, API_RUNTIME};                                           // 33 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33)
    cuda2hipRename["cudaDevAttrPciDeviceId"]                       = {"hipDeviceAttributePciDeviceId", CONV_TYPE, API_RUNTIME};                                        // 34 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34)
    cuda2hipRename["cudaDevAttrTccDriver"]                         = {"hipDeviceAttributeTccDriver", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                         // 35 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35)
    cuda2hipRename["cudaDevAttrMemoryClockRate"]                   = {"hipDeviceAttributeMemoryClockRate", CONV_TYPE, API_RUNTIME};                                    // 36 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36)
    cuda2hipRename["cudaDevAttrGlobalMemoryBusWidth"]              = {"hipDeviceAttributeMemoryBusWidth", CONV_TYPE, API_RUNTIME};                                     // 37 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37)
    cuda2hipRename["cudaDevAttrL2CacheSize"]                       = {"hipDeviceAttributeL2CacheSize", CONV_TYPE, API_RUNTIME};                                        // 38 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38)
    cuda2hipRename["cudaDevAttrMaxThreadsPerMultiProcessor"]       = {"hipDeviceAttributeMaxThreadsPerMultiProcessor", CONV_TYPE, API_RUNTIME};                        // 39 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39)
    cuda2hipRename["cudaDevAttrAsyncEngineCount"]                  = {"hipDeviceAttributeAsyncEngineCount", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                  // 40 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40)
    cuda2hipRename["cudaDevAttrUnifiedAddressing"]                 = {"hipDeviceAttributeUnifiedAddressing", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 41 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41)
    cuda2hipRename["cudaDevAttrMaxTexture1DLayeredWidth"]          = {"hipDeviceAttributeMaxTexture1DLayeredWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 42 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42)
    cuda2hipRename["cudaDevAttrMaxTexture1DLayeredLayers"]         = {"hipDeviceAttributeMaxTexture1DLayeredLayers", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 43 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43)
    // 44 - no
    cuda2hipRename["cudaDevAttrMaxTexture2DGatherWidth"]           = {"hipDeviceAttributeMaxTexture2DGatherWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 45 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45)
    cuda2hipRename["cudaDevAttrMaxTexture2DGatherHeight"]          = {"hipDeviceAttributeMaxTexture2DGatherHeight", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 46 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46)
    cuda2hipRename["cudaDevAttrMaxTexture3DWidthAlt"]              = {"hipDeviceAttributeMaxTexture3DWidthAlternate", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};        // 47 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47)
    cuda2hipRename["cudaDevAttrMaxTexture3DHeightAlt"]             = {"hipDeviceAttributeMaxTexture3DHeightAlternate", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 48 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48)
    cuda2hipRename["cudaDevAttrMaxTexture3DDepthAlt"]              = {"hipDeviceAttributeMaxTexture3DDepthAlternate", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};        // 49 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49)
    cuda2hipRename["cudaDevAttrPciDomainId"]                       = {"hipDeviceAttributePciDomainId", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                       // 50 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50)
    cuda2hipRename["cudaDevAttrTexturePitchAlignment"]             = {"hipDeviceAttributeTexturePitchAlignment", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};             // 51 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51)
    cuda2hipRename["cudaDevAttrMaxTextureCubemapWidth"]            = {"hipDeviceAttributeMaxTextureCubemapWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};            // 52 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52)
    cuda2hipRename["cudaDevAttrMaxTextureCubemapLayeredWidth"]     = {"hipDeviceAttributeMaxTextureCubemapLayeredWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};     // 53 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53)
    cuda2hipRename["cudaDevAttrMaxTextureCubemapLayeredLayers"]    = {"hipDeviceAttributeMaxTextureCubemapLayeredLayers", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};    // 54 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54)
    cuda2hipRename["cudaDevAttrMaxSurface1DWidth"]                 = {"hipDeviceAttributeMaxSurface1DWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 55 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55)
    cuda2hipRename["cudaDevAttrMaxSurface2DWidth"]                 = {"hipDeviceAttributeMaxSurface2DWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 56 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56)
    cuda2hipRename["cudaDevAttrMaxSurface2DHeight"]                = {"hipDeviceAttributeMaxSurface2DHeight", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                // 57 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57)
    cuda2hipRename["cudaDevAttrMaxSurface3DWidth"]                 = {"hipDeviceAttributeMaxSurface3DWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 58 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58)
    cuda2hipRename["cudaDevAttrMaxSurface3DHeight"]                = {"hipDeviceAttributeMaxSurface3DHeight", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                // 59 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59)
    cuda2hipRename["cudaDevAttrMaxSurface3DDepth"]                 = {"hipDeviceAttributeMaxSurface3DDepth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                 // 60 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60)
    cuda2hipRename["cudaDevAttrMaxSurface1DLayeredWidth"]          = {"hipDeviceAttributeMaxSurface1DLayeredWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 61 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61)
    cuda2hipRename["cudaDevAttrMaxSurface1DLayeredLayers"]         = {"hipDeviceAttributeMaxSurface1DLayeredLayers", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 62 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62)
    cuda2hipRename["cudaDevAttrMaxSurface2DLayeredWidth"]          = {"hipDeviceAttributeMaxSurface2DLayeredWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 63 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63)
    cuda2hipRename["cudaDevAttrMaxSurface2DLayeredHeight"]         = {"hipDeviceAttributeMaxSurface2DLayeredHeight", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 64 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64)
    cuda2hipRename["cudaDevAttrMaxSurface2DLayeredLayers"]         = {"hipDeviceAttributeMaxSurface2DLayeredLayers", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 65 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65)
    cuda2hipRename["cudaDevAttrMaxSurfaceCubemapWidth"]            = {"hipDeviceAttributeMaxSurfaceCubemapWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};            // 66 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66)
    cuda2hipRename["cudaDevAttrMaxSurfaceCubemapLayeredWidth"]     = {"hipDeviceAttributeMaxSurfaceCubemapLayeredWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};     // 67 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67)
    cuda2hipRename["cudaDevAttrMaxSurfaceCubemapLayeredLayers"]    = {"hipDeviceAttributeMaxSurfaceCubemapLayeredLayers", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};    // 68 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68)
    cuda2hipRename["cudaDevAttrMaxTexture1DLinearWidth"]           = {"hipDeviceAttributeMaxTexture1DLinearWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 69 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69)
    cuda2hipRename["cudaDevAttrMaxTexture2DLinearWidth"]           = {"hipDeviceAttributeMaxTexture2DLinearWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 70 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70)
    cuda2hipRename["cudaDevAttrMaxTexture2DLinearHeight"]          = {"hipDeviceAttributeMaxTexture2DLinearHeight", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 71 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71)
    cuda2hipRename["cudaDevAttrMaxTexture2DLinearPitch"]           = {"hipDeviceAttributeMaxTexture2DLinearPitch", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 72 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72)
    cuda2hipRename["cudaDevAttrMaxTexture2DMipmappedWidth"]        = {"hipDeviceAttributeMaxTexture2DMipmappedWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};        // 73 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73)
    cuda2hipRename["cudaDevAttrMaxTexture2DMipmappedHeight"]       = {"hipDeviceAttributeMaxTexture2DMipmappedHeight", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 74 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74)
    cuda2hipRename["cudaDevAttrComputeCapabilityMajor"]            = {"hipDeviceAttributeComputeCapabilityMajor", CONV_TYPE, API_RUNTIME};                             // 75 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75)
    cuda2hipRename["cudaDevAttrComputeCapabilityMinor"]            = {"hipDeviceAttributeComputeCapabilityMinor", CONV_TYPE, API_RUNTIME};                             // 76 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76)
    cuda2hipRename["cudaDevAttrMaxTexture1DMipmappedWidth"]        = {"hipDeviceAttributeMaxTexture1DMipmappedWidth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};        // 77 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77)
    cuda2hipRename["cudaDevAttrStreamPrioritiesSupported"]         = {"hipDeviceAttributeStreamPrioritiesSupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 78 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78)
    cuda2hipRename["cudaDevAttrGlobalL1CacheSupported"]            = {"hipDeviceAttributeGlobalL1CacheSupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};            // 79 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79)
    cuda2hipRename["cudaDevAttrLocalL1CacheSupported"]             = {"hipDeviceAttributeLocalL1CacheSupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};             // 80 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80)
    cuda2hipRename["cudaDevAttrMaxSharedMemoryPerMultiprocessor"]  = {"hipDeviceAttributeMaxSharedMemoryPerMultiprocessor", CONV_TYPE, API_RUNTIME};                   // 81 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81)
    cuda2hipRename["cudaDevAttrMaxRegistersPerMultiprocessor"]     = {"hipDeviceAttributeMaxRegistersPerMultiprocessor", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};     // 82 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82)
    cuda2hipRename["cudaDevAttrManagedMemory"]                     = {"hipDeviceAttributeManagedMemory", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                     // 83 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83)
    cuda2hipRename["cudaDevAttrIsMultiGpuBoard"]                   = {"hipDeviceAttributeIsMultiGpuBoard", CONV_TYPE, API_RUNTIME};                                    // 84 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84)
    cuda2hipRename["cudaDevAttrMultiGpuBoardGroupID"]              = {"hipDeviceAttributeMultiGpuBoardGroupID", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};              // 85 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85)

    // unsupported yet by HIP [CUDA 8.0.44]
    cuda2hipRename["cudaDevAttrHostNativeAtomicSupported"]         = {"hipDeviceAttributeHostNativeAtomicSupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};         // 86 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86)
    cuda2hipRename["cudaDevAttrSingleToDoublePrecisionPerfRatio"]  = {"hipDeviceAttributeSingleToDoublePrecisionPerfRatio", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};  // 87 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87)
    cuda2hipRename["cudaDevAttrPageableMemoryAccess"]              = {"hipDeviceAttributePageableMemoryAccess", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};              // 88 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88)
    cuda2hipRename["cudaDevAttrConcurrentManagedAccess"]           = {"hipDeviceAttributeConcurrentManagedAccess", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};           // 89 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89)
    cuda2hipRename["cudaDevAttrComputePreemptionSupported"]        = {"hipDeviceAttributeComputePreemptionSupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};        // 90 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90)
    cuda2hipRename["cudaDevAttrCanUseHostPointerForRegisteredMem"] = {"hipDeviceAttributeCanUseHostPointerForRegisteredMem", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}; // 91 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91)

    // Pointer Attributes
    // struct cudaPointerAttributes
    cuda2hipRename["cudaPointerAttributes"]                        = {"hipPointerAttribute_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaPointerGetAttributes"]                     = {"hipPointerGetAttributes", CONV_MEM, API_RUNTIME};

    cuda2hipRename["cudaHostGetDevicePointer"]                     = {"hipHostGetDevicePointer", CONV_MEM, API_RUNTIME};

    // Device
    cuda2hipRename["cudaDeviceProp"]                   = {"hipDeviceProp_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaGetDeviceProperties"]          = {"hipGetDeviceProperties", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaDeviceGetPCIBusId"]            = {"hipDeviceGetPCIBusId", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaDeviceGetByPCIBusId"]          = {"hipDeviceGetByPCIBusId", CONV_DEVICE, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaDeviceGetStreamPriorityRange"] = {"hipDeviceGetStreamPriorityRange", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaSetValidDevices"]              = {"hipSetValidDevices", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED};

    // unsupported yet by HIP [CUDA 8.0.44]
    // P2P Attributes
    cuda2hipRename["cudaDeviceP2PAttr"]                    = {"hipDeviceP2PAttribute", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                              // API_DRIVER ANALOGUE (CUdevice_P2PAttribute)
    cuda2hipRename["cudaDevP2PAttrPerformanceRank"]        = {"hipDeviceP2PAttributePerformanceRank", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 0x01 // API_DRIVER ANALOGUE (CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x01)
    cuda2hipRename["cudaDevP2PAttrAccessSupported"]        = {"hipDeviceP2PAttributeAccessSupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 0x02 // API_DRIVER ANALOGUE (CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x02)
    cuda2hipRename["cudaDevP2PAttrNativeAtomicSupported"]  = {"hipDeviceP2PAttributeNativeAtomicSupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}; // 0x03 // API_DRIVER ANALOGUE (CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03)
    // [CUDA 8.0.44]
    cuda2hipRename["cudaDeviceGetP2PAttribute"]            = {"hipDeviceGetP2PAttribute", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED};                              // API_DRIVER ANALOGUE (cuDeviceGetP2PAttribute)

    // Compute mode
    cuda2hipRename["cudaComputeMode"]                  = {"hipComputeMode", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                      // API_DRIVER ANALOGUE (CUcomputemode)
    cuda2hipRename["cudaComputeModeDefault"]           = {"hipComputeModeDefault", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 0 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_DEFAULT = 0)
    cuda2hipRename["cudaComputeModeExclusive"]         = {"hipComputeModeExclusive", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};        // 1 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_EXCLUSIVE = 1)
    cuda2hipRename["cudaComputeModeProhibited"]        = {"hipComputeModeProhibited", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};       // 2 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_PROHIBITED = 2)
    cuda2hipRename["cudaComputeModeExclusiveProcess"]  = {"hipComputeModeExclusiveProcess", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}; // 3 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3)

    // Device Flags
    cuda2hipRename["cudaGetDeviceFlags"]               = {"hipGetDeviceFlags", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaSetDeviceFlags"]               = {"hipSetDeviceFlags", CONV_DEVICE, API_RUNTIME};

    cuda2hipRename["cudaDeviceScheduleAuto"]           = {"hipDeviceScheduleAuto", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaDeviceScheduleSpin"]           = {"hipDeviceScheduleSpin", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaDeviceScheduleYield"]          = {"hipDeviceScheduleYield", CONV_TYPE, API_RUNTIME};
    // deprecated as of CUDA 4.0 and replaced with cudaDeviceScheduleBlockingSync
    cuda2hipRename["cudaDeviceBlockingSync"]           = {"hipDeviceScheduleBlockingSync", CONV_TYPE, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaDeviceScheduleBlockingSync"]   = {"hipDeviceScheduleBlockingSync", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaDeviceScheduleMask"]           = {"hipDeviceScheduleMask", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};

    cuda2hipRename["cudaDeviceMapHost"]                = {"hipDeviceMapHost", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaDeviceLmemResizeToMax"]        = {"hipDeviceLmemResizeToMax", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDeviceMask"]                   = {"hipDeviceMask", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};

    // Cache config
    cuda2hipRename["cudaDeviceSetCacheConfig"]         = {"hipDeviceSetCacheConfig", CONV_CACHE, API_RUNTIME};
    cuda2hipRename["cudaDeviceGetCacheConfig"]         = {"hipDeviceGetCacheConfig", CONV_CACHE, API_RUNTIME};
    cuda2hipRename["cudaFuncSetCacheConfig"]           = {"hipFuncSetCacheConfig", CONV_CACHE, API_RUNTIME};

    // Execution control
    // CUDA function cache configurations
    cuda2hipRename["cudaFuncCache"]                    = {"hipFuncCache_t", CONV_CACHE, API_RUNTIME};                                // API_Driver ANALOGUE (CUfunc_cache)
    cuda2hipRename["cudaFuncCachePreferNone"]          = {"hipFuncCachePreferNone", CONV_CACHE, API_RUNTIME};                   // 0 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_NONE = 0x00)
    cuda2hipRename["cudaFuncCachePreferShared"]        = {"hipFuncCachePreferShared", CONV_CACHE, API_RUNTIME};                 // 1 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_SHARED = 0x01)
    cuda2hipRename["cudaFuncCachePreferL1"]            = {"hipFuncCachePreferL1", CONV_CACHE, API_RUNTIME};                     // 2 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_L1 = 0x02)
    cuda2hipRename["cudaFuncCachePreferEqual"]         = {"hipFuncCachePreferEqual", CONV_CACHE, API_RUNTIME};                  // 3 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_EQUAL = 0x03)

    // Execution control functions
    // unsupported yet by HIP
    cuda2hipRename["cudaFuncAttributes"]               = {"hipFuncAttributes", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaFuncGetAttributes"]            = {"hipFuncGetAttributes", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaFuncSetSharedMemConfig"]       = {"hipFuncSetSharedMemConfig", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGetParameterBuffer"]           = {"hipGetParameterBuffer", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaSetDoubleForDevice"]           = {"hipSetDoubleForDevice", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaSetDoubleForHost"]             = {"hipSetDoubleForHost", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED};

    // Execution Control [deprecated since 7.0]
    // unsupported yet by HIP
    cuda2hipRename["cudaConfigureCall"]                = {"hipConfigureCall", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaLaunch"]                       = {"hipLaunch", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaSetupArgument"]                = {"hipSetupArgument", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED};

    // Version Management
    cuda2hipRename["cudaDriverGetVersion"]      = {"hipDriverGetVersion", CONV_VERSION, API_RUNTIME};
    cuda2hipRename["cudaRuntimeGetVersion"]     = {"hipRuntimeGetVersion", CONV_VERSION, API_RUNTIME, HIP_UNSUPPORTED};

    // Occupancy
    cuda2hipRename["cudaOccupancyMaxPotentialBlockSize"]                      = {"hipOccupancyMaxPotentialBlockSize", CONV_OCCUPANCY, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaOccupancyMaxPotentialBlockSizeWithFlags"]             = {"hipOccupancyMaxPotentialBlockSizeWithFlags", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaOccupancyMaxActiveBlocksPerMultiprocessor"]           = {"hipOccupancyMaxActiveBlocksPerMultiprocessor", CONV_OCCUPANCY, API_RUNTIME};
    // unsupported yet by HIP
    cuda2hipRename["cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"]  = {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaOccupancyMaxPotentialBlockSizeVariableSMem"]          = {"hipOccupancyMaxPotentialBlockSizeVariableSMem", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags"] = {"hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED};

    // Peer2Peer
    cuda2hipRename["cudaDeviceCanAccessPeer"]        = {"hipDeviceCanAccessPeer", CONV_PEER, API_RUNTIME};
    cuda2hipRename["cudaDeviceDisablePeerAccess"]    = {"hipDeviceDisablePeerAccess", CONV_PEER, API_RUNTIME};
    cuda2hipRename["cudaDeviceEnablePeerAccess"]     = {"hipDeviceEnablePeerAccess", CONV_PEER, API_RUNTIME};

    cuda2hipRename["cudaMemcpyPeerAsync"]            = {"hipMemcpyPeerAsync", CONV_MEM, API_RUNTIME};
    cuda2hipRename["cudaMemcpyPeer"]                 = {"hipMemcpyPeer", CONV_MEM, API_RUNTIME};

    // #define cudaIpcMemLazyEnablePeerAccess 0x01
    cuda2hipRename["cudaIpcMemLazyEnablePeerAccess"] = {"hipIpcMemLazyEnablePeerAccess", CONV_TYPE, API_RUNTIME};              // 0x01 // API_Driver ANALOGUE (CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1)

    // Shared memory
    cuda2hipRename["cudaDeviceSetSharedMemConfig"]   = {"hipDeviceSetSharedMemConfig", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaDeviceGetSharedMemConfig"]   = {"hipDeviceGetSharedMemConfig", CONV_DEVICE, API_RUNTIME};
    // translate deprecated
    // cuda2hipRename["cudaThreadGetSharedMemConfig"] = {"hipDeviceGetSharedMemConfig", CONV_DEVICE, API_RUNTIME};
    // cuda2hipRename["cudaThreadSetSharedMemConfig"] = {"hipDeviceSetSharedMemConfig", CONV_DEVICE, API_RUNTIME};

    cuda2hipRename["cudaSharedMemConfig"]            = {"hipSharedMemConfig", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaSharedMemBankSizeDefault"]   = {"hipSharedMemBankSizeDefault", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaSharedMemBankSizeFourByte"]  = {"hipSharedMemBankSizeFourByte", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaSharedMemBankSizeEightByte"] = {"hipSharedMemBankSizeEightByte", CONV_TYPE, API_RUNTIME};

    // Limits
    cuda2hipRename["cudaLimit"]                             = {"hipLimit_t", CONV_TYPE, API_RUNTIME};                                                    // API_Driver ANALOGUE (CUlimit)
    cuda2hipRename["cudaLimitStackSize"]                    = {"hipLimitStackSize", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};                    // 0x00 // API_Driver ANALOGUE (CU_LIMIT_STACK_SIZE = 0x00)
    cuda2hipRename["cudaLimitPrintfFifoSize"]               = {"hipLimitPrintfFifoSize", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};               // 0x01 // API_Driver ANALOGUE (CU_LIMIT_PRINTF_FIFO_SIZE = 0x01)
    cuda2hipRename["cudaLimitMallocHeapSize"]               = {"hipLimitMallocHeapSize", CONV_TYPE, API_RUNTIME};                                // 0x02 // API_Driver ANALOGUE (CU_LIMIT_MALLOC_HEAP_SIZE = 0x02)
    cuda2hipRename["cudaLimitDevRuntimeSyncDepth"]          = {"hipLimitDevRuntimeSyncDepth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED};          // 0x03 // API_Driver ANALOGUE (CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03)
    cuda2hipRename["cudaLimitDevRuntimePendingLaunchCount"] = {"hipLimitDevRuntimePendingLaunchCount", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}; // 0x04 // API_Driver ANALOGUE (CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04)

    cuda2hipRename["cudaDeviceGetLimit"]                    = {"hipDeviceGetLimit", CONV_DEVICE, API_RUNTIME};

    // Profiler
    cuda2hipRename["cudaProfilerInitialize"]                = {"hipProfilerInitialize", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED};                       // API_Driver ANALOGUE (cuProfilerInitialize)
    cuda2hipRename["cudaProfilerStart"]                     = {"hipProfilerStart", CONV_OTHER, API_RUNTIME};                                             // API_Driver ANALOGUE (cuProfilerStart)
    cuda2hipRename["cudaProfilerStop"]                      = {"hipProfilerStop", CONV_OTHER, API_RUNTIME};                                              // API_Driver ANALOGUE (cuProfilerStop)

    // unsupported yet by HIP
    cuda2hipRename["cudaOutputMode"]                        = {"hipOutputMode", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaKeyValuePair"]                      = {"hipKeyValuePair", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaCSV"]                               = {"hipCSV", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED};

    // Texture Reference Management
    // enums
    cuda2hipRename["cudaTextureReadMode"]                   = {"hipTextureReadMode", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaReadModeElementType"]               = {"hipReadModeElementType", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaReadModeNormalizedFloat"]           = {"hipReadModeNormalizedFloat", CONV_TEX, API_RUNTIME};

    cuda2hipRename["cudaTextureFilterMode"]                 = {"hipTextureFilterMode", CONV_TEX, API_RUNTIME};                              // API_DRIVER ANALOGUE (CUfilter_mode)
    cuda2hipRename["cudaFilterModePoint"]                   = {"hipFilterModePoint", CONV_TEX, API_RUNTIME};                           // 0 // API_DRIVER ANALOGUE (CU_TR_FILTER_MODE_POINT = 0)
    cuda2hipRename["cudaFilterModeLinear"]                  = {"hipFilterModeLinear", CONV_TEX, API_RUNTIME};                          // 1 // API_DRIVER ANALOGUE (CU_TR_FILTER_MODE_POINT = 1)

    cuda2hipRename["cudaBindTexture"]                       = {"hipBindTexture", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaUnbindTexture"]                     = {"hipUnbindTexture", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaBindTexture2D"]                     = {"hipBindTexture2D", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaBindTextureToArray"]                = {"hipBindTextureToArray", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaBindTextureToMipmappedArray"]       = {"hipBindTextureToMipmappedArray", CONV_TEX, API_RUNTIME}; // Unsupported yet on NVCC path
    cuda2hipRename["cudaGetTextureAlignmentOffset"]         = {"hipGetTextureAlignmentOffset", CONV_TEX, API_RUNTIME};   // Unsupported yet on NVCC path
    cuda2hipRename["cudaGetTextureReference"]               = {"hipGetTextureReference", CONV_TEX, API_RUNTIME};         // Unsupported yet on NVCC path

    // Channel
    cuda2hipRename["cudaChannelFormatKind"]                 = {"hipChannelFormatKind", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaChannelFormatKindSigned"]           = {"hipChannelFormatKindSigned", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaChannelFormatKindUnsigned"]         = {"hipChannelFormatKindUnsigned", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaChannelFormatKindFloat"]            = {"hipChannelFormatKindFloat", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaChannelFormatKindNone"]             = {"hipChannelFormatKindNone", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaChannelFormatDesc"]                 = {"hipChannelFormatDesc", CONV_TEX, API_RUNTIME};

    cuda2hipRename["cudaCreateChannelDesc"]                 = {"hipCreateChannelDesc", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaGetChannelDesc"]                    = {"hipGetChannelDesc", CONV_TEX, API_RUNTIME};

    // Texture Object Management
    // structs
    cuda2hipRename["cudaResourceDesc"]                            = {"hipResourceDesc", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaResourceViewDesc"]                        = {"hipResourceViewDesc", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaTextureDesc"]                             = {"hipTextureDesc", CONV_TEX, API_RUNTIME};
    cuda2hipRename["surfaceReference"]                            = {"hipSurfaceReference", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};
    // Left unchanged
    // cuda2hipRename["textureReference"]                         = {"textureReference", CONV_TEX, API_RUNTIME};

    // typedefs
    cuda2hipRename["cudaTextureObject_t"]                         = {"hipTextureObject_t", CONV_TEX, API_RUNTIME};

    // enums
    // enum cudaResourceType
    cuda2hipRename["cudaResourceType"]                            = {"hipResourceType", CONV_TEX, API_RUNTIME};                                    // API_Driver ANALOGUE (CUresourcetype)
    cuda2hipRename["cudaResourceTypeArray"]                       = {"hipResourceTypeArray", CONV_TEX, API_RUNTIME};                       // 0x00 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_ARRAY = 0x00)
    cuda2hipRename["cudaResourceTypeMipmappedArray"]              = {"hipResourceTypeMipmappedArray", CONV_TEX, API_RUNTIME};              // 0x01 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01)
    cuda2hipRename["cudaResourceTypeLinear"]                      = {"hipResourceTypeLinear", CONV_TEX, API_RUNTIME};                      // 0x02 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_LINEAR = 0x02)
    cuda2hipRename["cudaResourceTypePitch2D"]                     = {"hipResourceTypePitch2D", CONV_TEX, API_RUNTIME};                     // 0x03 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_PITCH2D = 0x03)

    // enum cudaResourceViewFormat
    cuda2hipRename["cudaResourceViewFormat"]                      = {"hipResourceViewFormat", CONV_TEX, API_RUNTIME};                              // API_Driver ANALOGUE (CUresourceViewFormat)
    cuda2hipRename["cudaResViewFormatNone"]                       = {"hipResViewFormatNone", CONV_TEX, API_RUNTIME};                       // 0x00 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_NONE = 0x00)
    cuda2hipRename["cudaResViewFormatUnsignedChar1"]              = {"hipResViewFormatUnsignedChar1", CONV_TEX, API_RUNTIME};              // 0x01 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_1X8 = 0x01)
    cuda2hipRename["cudaResViewFormatUnsignedChar2"]              = {"hipResViewFormatUnsignedChar2", CONV_TEX, API_RUNTIME};              // 0x02 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_2X8 = 0x02)
    cuda2hipRename["cudaResViewFormatUnsignedChar4"]              = {"hipResViewFormatUnsignedChar4", CONV_TEX, API_RUNTIME};              // 0x03 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_4X8 = 0x03)
    cuda2hipRename["cudaResViewFormatSignedChar1"]                = {"hipResViewFormatSignedChar1", CONV_TEX, API_RUNTIME};                // 0x04 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_1X8 = 0x04)
    cuda2hipRename["cudaResViewFormatSignedChar2"]                = {"hipResViewFormatSignedChar2", CONV_TEX, API_RUNTIME};                // 0x05 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_2X8 = 0x05)
    cuda2hipRename["cudaResViewFormatSignedChar4"]                = {"hipResViewFormatSignedChar4", CONV_TEX, API_RUNTIME};                // 0x06 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_4X8 = 0x06)
    cuda2hipRename["cudaResViewFormatUnsignedShort1"]             = {"hipResViewFormatUnsignedShort1", CONV_TEX, API_RUNTIME};             // 0x07 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_1X16 = 0x07)
    cuda2hipRename["cudaResViewFormatUnsignedShort2"]             = {"hipResViewFormatUnsignedShort2", CONV_TEX, API_RUNTIME};             // 0x08 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_2X16 = 0x08)
    cuda2hipRename["cudaResViewFormatUnsignedShort4"]             = {"hipResViewFormatUnsignedShort4", CONV_TEX, API_RUNTIME};             // 0x09 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_4X16 = 0x09)
    cuda2hipRename["cudaResViewFormatSignedShort1"]               = {"hipResViewFormatSignedShort1", CONV_TEX, API_RUNTIME};               // 0x0a // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_1X16 = 0x0a)
    cuda2hipRename["cudaResViewFormatSignedShort2"]               = {"hipResViewFormatSignedShort2", CONV_TEX, API_RUNTIME};               // 0x0b // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_2X16 = 0x0b)
    cuda2hipRename["cudaResViewFormatSignedShort4"]               = {"hipResViewFormatSignedShort4", CONV_TEX, API_RUNTIME};               // 0x0c // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_4X16 = 0x0c)
    cuda2hipRename["cudaResViewFormatUnsignedInt1"]               = {"hipResViewFormatUnsignedInt1", CONV_TEX, API_RUNTIME};               // 0x0d // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_1X32 = 0x0d)
    cuda2hipRename["cudaResViewFormatUnsignedInt2"]               = {"hipResViewFormatUnsignedInt2", CONV_TEX, API_RUNTIME};               // 0x0e // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_2X32 = 0x0e)
    cuda2hipRename["cudaResViewFormatUnsignedInt4"]               = {"hipResViewFormatUnsignedInt4", CONV_TEX, API_RUNTIME};               // 0x0f // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_4X32 = 0x0f)
    cuda2hipRename["cudaResViewFormatSignedInt1"]                 = {"hipResViewFormatSignedInt1", CONV_TEX, API_RUNTIME};                 // 0x10 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_1X32 = 0x10)
    cuda2hipRename["cudaResViewFormatSignedInt2"]                 = {"hipResViewFormatSignedInt2", CONV_TEX, API_RUNTIME};                 // 0x11 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_2X32 = 0x11)
    cuda2hipRename["cudaResViewFormatSignedInt4"]                 = {"hipResViewFormatSignedInt4", CONV_TEX, API_RUNTIME};                 // 0x12 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_4X32 = 0x12)
    cuda2hipRename["cudaResViewFormatHalf1"]                      = {"hipResViewFormatHalf1", CONV_TEX, API_RUNTIME};                      // 0x13 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_1X16 = 0x13)
    cuda2hipRename["cudaResViewFormatHalf2"]                      = {"hipResViewFormatHalf2", CONV_TEX, API_RUNTIME};                      // 0x14 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_2X16 = 0x14)
    cuda2hipRename["cudaResViewFormatHalf4"]                      = {"hipResViewFormatHalf4", CONV_TEX, API_RUNTIME};                      // 0x15 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_4X16 = 0x15)
    cuda2hipRename["cudaResViewFormatFloat1"]                     = {"hipResViewFormatFloat1", CONV_TEX, API_RUNTIME};                     // 0x16 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_1X32 = 0x16)
    cuda2hipRename["cudaResViewFormatFloat2"]                     = {"hipResViewFormatFloat2", CONV_TEX, API_RUNTIME};                     // 0x17 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_2X32 = 0x17)
    cuda2hipRename["cudaResViewFormatFloat4"]                     = {"hipResViewFormatFloat4", CONV_TEX, API_RUNTIME};                     // 0x18 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_4X32 = 0x18)
    cuda2hipRename["cudaResViewFormatUnsignedBlockCompressed1"]   = {"hipResViewFormatUnsignedBlockCompressed1", CONV_TEX, API_RUNTIME};   // 0x19 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 0x19)
    cuda2hipRename["cudaResViewFormatUnsignedBlockCompressed2"]   = {"hipResViewFormatUnsignedBlockCompressed2", CONV_TEX, API_RUNTIME};   // 0x1a // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 0x1a)
    cuda2hipRename["cudaResViewFormatUnsignedBlockCompressed3"]   = {"hipResViewFormatUnsignedBlockCompressed3", CONV_TEX, API_RUNTIME};   // 0x1b // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 0x1b)
    cuda2hipRename["cudaResViewFormatUnsignedBlockCompressed4"]   = {"hipResViewFormatUnsignedBlockCompressed4", CONV_TEX, API_RUNTIME};   // 0x1c // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 0x1c)
    cuda2hipRename["cudaResViewFormatSignedBlockCompressed4"]     = {"hipResViewFormatSignedBlockCompressed4", CONV_TEX, API_RUNTIME};     // 0x1d // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SIGNED_BC4 = 0x1d)
    cuda2hipRename["cudaResViewFormatUnsignedBlockCompressed5"]   = {"hipResViewFormatUnsignedBlockCompressed5", CONV_TEX, API_RUNTIME};   // 0x1e // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 0x1e)
    cuda2hipRename["cudaResViewFormatSignedBlockCompressed5"]     = {"hipResViewFormatSignedBlockCompressed5", CONV_TEX, API_RUNTIME};     // 0x1f // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SIGNED_BC5 = 0x1f)
    cuda2hipRename["cudaResViewFormatUnsignedBlockCompressed6H"]  = {"hipResViewFormatUnsignedBlockCompressed6H", CONV_TEX, API_RUNTIME};  // 0x20 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20)
    cuda2hipRename["cudaResViewFormatSignedBlockCompressed6H"]    = {"hipResViewFormatSignedBlockCompressed6H", CONV_TEX, API_RUNTIME};    // 0x21 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SIGNED_BC6H = 0x21)
    cuda2hipRename["cudaResViewFormatUnsignedBlockCompressed7"]   = {"hipResViewFormatUnsignedBlockCompressed7", CONV_TEX, API_RUNTIME};   // 0x22 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 0x22)

    cuda2hipRename["cudaTextureAddressMode"]                      = {"hipTextureAddressMode", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaAddressModeWrap"]                         = {"hipAddressModeWrap", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaAddressModeClamp"]                        = {"hipAddressModeClamp", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaAddressModeMirror"]                       = {"hipAddressModeMirror", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaAddressModeBorder"]                       = {"hipAddressModeBorder", CONV_TEX, API_RUNTIME};

    // functions
    cuda2hipRename["cudaCreateTextureObject"]                     = {"hipCreateTextureObject", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaDestroyTextureObject"]                    = {"hipDestroyTextureObject", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaGetTextureObjectResourceDesc"]            = {"hipGetTextureObjectResourceDesc", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaGetTextureObjectResourceViewDesc"]        = {"hipGetTextureObjectResourceViewDesc", CONV_TEX, API_RUNTIME};
    cuda2hipRename["cudaGetTextureObjectTextureDesc"]             = {"hipGetTextureObjectTextureDesc", CONV_TEX, API_RUNTIME};

    // Surface Reference Management
    // unsupported yet by HIP
    cuda2hipRename["cudaBindSurfaceToArray"]                      = {"hipBindSurfaceToArray", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGetSurfaceReference"]                     = {"hipGetSurfaceReference", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};

    cuda2hipRename["cudaSurfaceBoundaryMode"]                     = {"hipSurfaceBoundaryMode", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaBoundaryModeZero"]                        = {"hipBoundaryModeZero", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaBoundaryModeClamp"]                       = {"hipBoundaryModeClamp", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaBoundaryModeTrap"]                        = {"hipBoundaryModeTrap", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};

    cuda2hipRename["cudaSurfaceFormatMode"]                       = {"hipSurfaceFormatMode", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaFormatModeForced"]                        = {"hipFormatModeForced", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaFormatModeAuto"]                          = {"hipFormatModeAuto", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};

    // Surface Object Management
    // unsupported yet by HIP
    cuda2hipRename["cudaCreateSurfaceObject"]                     = {"hipCreateSurfaceObject", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaDestroySurfaceObject"]                    = {"hipDestroySurfaceObject", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGetSurfaceObjectResourceDesc"]            = {"hipGetSurfaceObjectResourceDesc", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED};

    // Inter-Process Communications (IPC)
    // IPC types
    cuda2hipRename["cudaIpcEventHandle_t"]                        = {"hipIpcEventHandle_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaIpcEventHandle_st"]                       = {"hipIpcEventHandle_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaIpcMemHandle_t"]                          = {"hipIpcMemHandle_t", CONV_TYPE, API_RUNTIME};
    cuda2hipRename["cudaIpcMemHandle_st"]                         = {"hipIpcMemHandle_t", CONV_TYPE, API_RUNTIME};

    // IPC functions
    cuda2hipRename["cudaIpcCloseMemHandle"]                       = {"hipIpcCloseMemHandle", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaIpcGetEventHandle"]                       = {"hipIpcGetEventHandle", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaIpcGetMemHandle"]                         = {"hipIpcGetMemHandle", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaIpcOpenEventHandle"]                      = {"hipIpcOpenEventHandle", CONV_DEVICE, API_RUNTIME};
    cuda2hipRename["cudaIpcOpenMemHandle"]                        = {"hipIpcOpenMemHandle", CONV_DEVICE, API_RUNTIME};

    // OpenGL Interoperability
    cuda2hipRename["cudaGLGetDevices"]                            = {"hipGLGetDevices", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGraphicsGLRegisterBuffer"]                = {"hipGraphicsGLRegisterBuffer", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGraphicsGLRegisterImage"]                 = {"hipGraphicsGLRegisterImage", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaWGLGetDevice"]                            = {"hipWGLGetDevice", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};

    // Graphics Interoperability
    cuda2hipRename["cudaGraphicsMapResources"]                    = {"hipGraphicsMapResources", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};                      // API_Driver ANALOGUE (cuGraphicsMapResources)
    cuda2hipRename["cudaGraphicsResourceGetMappedMipmappedArray"] = {"hipGraphicsResourceGetMappedMipmappedArray", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};   // API_Driver ANALOGUE (cuGraphicsResourceGetMappedMipmappedArray)
    cuda2hipRename["cudaGraphicsResourceGetMappedPointer"]        = {"hipGraphicsResourceGetMappedPointer", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};          // API_Driver ANALOGUE (cuGraphicsResourceGetMappedPointer)
    cuda2hipRename["cudaGraphicsResourceSetMapFlags"]             = {"hipGraphicsResourceSetMapFlags", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};               // API_Driver ANALOGUE (cuGraphicsResourceSetMapFlags)
    cuda2hipRename["cudaGraphicsSubResourceGetMappedArray"]       = {"hipGraphicsSubResourceGetMappedArray", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};         // API_Driver ANALOGUE (cuGraphicsSubResourceGetMappedArray)
    cuda2hipRename["cudaGraphicsUnmapResources"]                  = {"hipGraphicsUnmapResources", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};                    // API_Driver ANALOGUE (cuGraphicsUnmapResources)
    cuda2hipRename["cudaGraphicsUnregisterResource"]              = {"hipGraphicsUnregisterResource", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};                // API_Driver ANALOGUE (cuGraphicsUnregisterResource)

    cuda2hipRename["cudaGraphicsCubeFace"]                        = {"hipGraphicsCubeFace", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGraphicsCubeFacePositiveX"]               = {"hipGraphicsCubeFacePositiveX", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGraphicsCubeFaceNegativeX"]               = {"hipGraphicsCubeFaceNegativeX", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGraphicsCubeFacePositiveY"]               = {"hipGraphicsCubeFacePositiveY", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGraphicsCubeFaceNegativeY"]               = {"hipGraphicsCubeFaceNegativeY", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGraphicsCubeFacePositiveZ"]               = {"hipGraphicsCubeFacePositiveZ", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};
    cuda2hipRename["cudaGraphicsCubeFaceNegativeZ"]               = {"hipGraphicsCubeFaceNegativeZ", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};

    // enum cudaGraphicsMapFlags
    cuda2hipRename["cudaGraphicsMapFlags"]                        = {"hipGraphicsMapFlags", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};                           // API_Driver ANALOGUE (CUgraphicsMapResourceFlags)
    cuda2hipRename["cudaGraphicsMapFlagsNone"]                    = {"hipGraphicsMapFlagsNone", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};                  // 0 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00)
    cuda2hipRename["cudaGraphicsMapFlagsReadOnly"]                = {"hipGraphicsMapFlagsReadOnly", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};              // 1 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01)
    cuda2hipRename["cudaGraphicsMapFlagsWriteDiscard"]            = {"hipGraphicsMapFlagsWriteDiscard", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};          // 2 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02)

    // enum cudaGraphicsRegisterFlags
    cuda2hipRename["cudaGraphicsRegisterFlags"]                   = {"hipGraphicsRegisterFlags", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};                      // API_Driver ANALOGUE (CUgraphicsRegisterFlags)
    cuda2hipRename["cudaGraphicsRegisterFlagsNone"]               = {"hipGraphicsRegisterFlagsNone", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};             // 0 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00)
    cuda2hipRename["cudaGraphicsRegisterFlagsReadOnly"]           = {"hipGraphicsRegisterFlagsReadOnly", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};         // 1 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01)
    cuda2hipRename["cudaGraphicsRegisterFlagsWriteDiscard"]       = {"hipGraphicsRegisterFlagsWriteDiscard", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};     // 2 // API_Driver ANALOGUE (CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02)
    cuda2hipRename["cudaGraphicsRegisterFlagsSurfaceLoadStore"]   = {"hipGraphicsRegisterFlagsSurfaceLoadStore", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}; // 4 // API_Driver ANALOGUE (CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 0x04)
    cuda2hipRename["cudaGraphicsRegisterFlagsTextureGather"]      = {"hipGraphicsRegisterFlagsTextureGather", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED};    // 8 // API_Driver ANALOGUE (CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x08)

    // OpenGL Interoperability
    // enum cudaGLDeviceList
    cuda2hipRename["cudaGLDeviceList"]                          = {"hipGLDeviceList", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                                    // API_Driver ANALOGUE (CUGLDeviceList)
    cuda2hipRename["cudaGLDeviceListAll"]                       = {"HIP_GL_DEVICE_LIST_ALL", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                     // 0x01 // API_Driver ANALOGUE (CU_GL_DEVICE_LIST_ALL)
    cuda2hipRename["cudaGLDeviceListCurrentFrame"]              = {"HIP_GL_DEVICE_LIST_CURRENT_FRAME", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};           // 0x02 // API_Driver ANALOGUE (CU_GL_DEVICE_LIST_CURRENT_FRAME)
    cuda2hipRename["cudaGLDeviceListNextFrame"]                 = {"HIP_GL_DEVICE_LIST_NEXT_FRAME", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};              // 0x03 // API_Driver ANALOGUE (CU_GL_DEVICE_LIST_NEXT_FRAME)

    cuda2hipRename["cudaGLGetDevices"]                          = {"hipGLGetDevices", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                                     // API_Driver ANALOGUE (cuGLGetDevices)
    cuda2hipRename["cudaGraphicsGLRegisterBuffer"]              = {"hipGraphicsGLRegisterBuffer", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                         // API_Driver ANALOGUE (cuGraphicsGLRegisterBuffer)
    cuda2hipRename["cudaGraphicsGLRegisterImage"]               = {"hipGraphicsGLRegisterImage", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                          // API_Driver ANALOGUE (cuGraphicsGLRegisterImage)
    cuda2hipRename["cudaWGLGetDevice"]                          = {"hipWGLGetDevice", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                                     // API_Driver ANALOGUE (cuWGLGetDevice)

    // OpenGL Interoperability [DEPRECATED]
    // enum cudaGLMapFlags
    cuda2hipRename["cudaGLMapFlags"]                            = {"hipGLMapFlags", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                                       // API_Driver ANALOGUE (CUGLmap_flags)
    cuda2hipRename["cudaGLMapFlagsNone"]                        = {"HIP_GL_MAP_RESOURCE_FLAGS_NONE", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};              // 0x00 // API_Driver ANALOGUE (CU_GL_MAP_RESOURCE_FLAGS_NONE)
    cuda2hipRename["cudaGLMapFlagsReadOnly"]                    = {"HIP_GL_MAP_RESOURCE_FLAGS_READ_ONLY", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};         // 0x01 // API_Driver ANALOGUE (CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY)
    cuda2hipRename["cudaGLMapFlagsWriteDiscard"]                = {"HIP_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};     // 0x02 // API_Driver ANALOGUE (CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD)

    cuda2hipRename["cudaGLMapBufferObject"]                     = {"hipGLMapBufferObject__", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                              // Not equal to cuGLMapBufferObject due to different signatures
    cuda2hipRename["cudaGLMapBufferObjectAsync"]                = {"hipGLMapBufferObjectAsync__", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                         // Not equal to cuGLMapBufferObjectAsync due to different signatures
    cuda2hipRename["cudaGLRegisterBufferObject"]                = {"hipGLRegisterBufferObject", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                           // API_Driver ANALOGUE (cuGLRegisterBufferObject)
    cuda2hipRename["cudaGLSetBufferObjectMapFlags"]             = {"hipGLSetBufferObjectMapFlags", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                        // API_Driver ANALOGUE (cuGLSetBufferObjectMapFlags)
    cuda2hipRename["cudaGLSetGLDevice"]                         = {"hipGLSetGLDevice", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                                    // no API_Driver ANALOGUE
    cuda2hipRename["cudaGLUnmapBufferObject"]                   = {"hipGLUnmapBufferObject", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                              // API_Driver ANALOGUE (cuGLUnmapBufferObject)
    cuda2hipRename["cudaGLUnmapBufferObjectAsync"]              = {"hipGLUnmapBufferObjectAsync", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                         // API_Driver ANALOGUE (cuGLUnmapBufferObjectAsync)
    cuda2hipRename["cudaGLUnregisterBufferObject"]              = {"hipGLUnregisterBufferObject", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED};                         // API_Driver ANALOGUE (cuGLUnregisterBufferObject)

    // Direct3D 9 Interoperability
    // enum CUd3d9DeviceList
    cuda2hipRename["cudaD3D9DeviceList"]                        = {"hipD3D9DeviceList", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                                // API_Driver ANALOGUE (CUd3d9DeviceList)
    cuda2hipRename["cudaD3D9DeviceListAll"]                     = {"HIP_D3D9_DEVICE_LIST_ALL", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                    // 1 // API_Driver ANALOGUE (CU_D3D9_DEVICE_LIST_ALL)
    cuda2hipRename["cudaD3D9DeviceListCurrentFrame"]            = {"HIP_D3D9_DEVICE_LIST_CURRENT_FRAME", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};          // 2 // API_Driver ANALOGUE (CU_D3D9_DEVICE_LIST_CURRENT_FRAME)
    cuda2hipRename["cudaD3D9DeviceListNextFrame"]               = {"HIP_D3D9_DEVICE_LIST_NEXT_FRAME", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};             // 3 // API_Driver ANALOGUE (CU_D3D9_DEVICE_LIST_NEXT_FRAME)

    cuda2hipRename["cudaD3D9GetDevice"]                         = {"hipD3D9GetDevice", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                                 // API_Driver ANALOGUE (cuD3D9GetDevice)
    cuda2hipRename["cudaD3D9GetDevices"]                        = {"hipD3D9GetDevices", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                                // API_Driver ANALOGUE (cuD3D9GetDevices)
    cuda2hipRename["cudaD3D9GetDirect3DDevice"]                 = {"hipD3D9GetDirect3DDevice", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                         // API_Driver ANALOGUE (cuD3D9GetDirect3DDevice)
    cuda2hipRename["cudaD3D9SetDirect3DDevice"]                 = {"hipD3D9SetDirect3DDevice", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                         // no API_Driver ANALOGUE
    cuda2hipRename["cudaGraphicsD3D9RegisterResource"]          = {"hipGraphicsD3D9RegisterResource", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                  // API_Driver ANALOGUE (cuGraphicsD3D9RegisterResource)

    // Direct3D 9 Interoperability [DEPRECATED]
    // enum cudaD3D9MapFlags
    cuda2hipRename["cudaD3D9MapFlags"]                          = {"hipD3D9MapFlags", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                                   // API_Driver ANALOGUE (CUd3d9map_flags)
    cuda2hipRename["cudaD3D9MapFlagsNone"]                      = {"HIP_D3D9_MAPRESOURCE_FLAGS_NONE", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};              // 0 // API_Driver ANALOGUE (CU_D3D9_MAPRESOURCE_FLAGS_NONE)
    cuda2hipRename["cudaD3D9MapFlagsReadOnly"]                  = {"HIP_D3D9_MAPRESOURCE_FLAGS_READONLY", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};          // 1 // API_Driver ANALOGUE (CU_D3D9_MAPRESOURCE_FLAGS_READONLY)
    cuda2hipRename["cudaD3D9MapFlagsWriteDiscard"]              = {"HIP_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};      // 2 // API_Driver ANALOGUE (CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD)

    // enum cudaD3D9RegisterFlags
    cuda2hipRename["cudaD3D9RegisterFlags"]                     = {"hipD3D9RegisterFlags", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                              // API_Driver ANALOGUE (CUd3d9Register_flags)
    cuda2hipRename["cudaD3D9RegisterFlagsNone"]                 = {"HIP_D3D9_REGISTER_FLAGS_NONE", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                 // 0 // API_Driver ANALOGUE (CU_D3D9_REGISTER_FLAGS_NONE)
    cuda2hipRename["cudaD3D9RegisterFlagsArray"]                = {"HIP_D3D9_REGISTER_FLAGS_ARRAY", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                // 1 // API_Driver ANALOGUE (CU_D3D9_REGISTER_FLAGS_ARRAY)

    cuda2hipRename["cudaD3D9MapResources"]                      = {"hipD3D9MapResources", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                               // API_Driver ANALOGUE (cuD3D9MapResources)
    cuda2hipRename["cudaD3D9RegisterResource"]                  = {"hipD3D9RegisterResource", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                           // API_Driver ANALOGUE (cuD3D9RegisterResource)
    cuda2hipRename["cudaD3D9ResourceGetMappedArray"]            = {"hipD3D9ResourceGetMappedArray", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                     // API_Driver ANALOGUE (cuD3D9ResourceGetMappedArray)
    cuda2hipRename["cudaD3D9ResourceGetMappedPitch"]            = {"hipD3D9ResourceGetMappedPitch", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                     // API_Driver ANALOGUE (cudaD3D9ResourceGetMappedPitch)
    cuda2hipRename["cudaD3D9ResourceGetMappedPointer"]          = {"hipD3D9ResourceGetMappedPointer", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                   // API_Driver ANALOGUE (cuD3D9ResourceGetMappedPointer)
    cuda2hipRename["cudaD3D9ResourceGetMappedSize"]             = {"hipD3D9ResourceGetMappedSize", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                      // API_Driver ANALOGUE (cuD3D9ResourceGetMappedSize)
    cuda2hipRename["cudaD3D9ResourceGetSurfaceDimensions"]      = {"hipD3D9ResourceGetSurfaceDimensions", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};               // API_Driver ANALOGUE (cuD3D9ResourceGetSurfaceDimensions)
    cuda2hipRename["cudaD3D9ResourceSetMapFlags"]               = {"hipD3D9ResourceSetMapFlags", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                        // API_Driver ANALOGUE (cuD3D9ResourceSetMapFlags)
    cuda2hipRename["cudaD3D9UnmapResources"]                    = {"hipD3D9UnmapResources", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                             // API_Driver ANALOGUE (cuD3D9UnmapResources)
    cuda2hipRename["cudaD3D9UnregisterResource"]                = {"hipD3D9UnregisterResource", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED};                         // API_Driver ANALOGUE (cuD3D9UnregisterResource)

    // Direct3D 10 Interoperability
    // enum cudaD3D10DeviceList
    cuda2hipRename["cudaD3D10DeviceList"]                       = {"hipd3d10DeviceList", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                              // API_Driver ANALOGUE (CUd3d10DeviceList)
    cuda2hipRename["cudaD3D10DeviceListAll"]                    = {"HIP_D3D10_DEVICE_LIST_ALL", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                  // 1 // API_Driver ANALOGUE (CU_D3D10_DEVICE_LIST_ALL)
    cuda2hipRename["cudaD3D10DeviceListCurrentFrame"]           = {"HIP_D3D10_DEVICE_LIST_CURRENT_FRAME", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};        // 2 // API_Driver ANALOGUE (CU_D3D10_DEVICE_LIST_CURRENT_FRAME)
    cuda2hipRename["cudaD3D10DeviceListNextFrame"]              = {"HIP_D3D10_DEVICE_LIST_NEXT_FRAME", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};           // 3 // API_Driver ANALOGUE (CU_D3D10_DEVICE_LIST_NEXT_FRAME)

    cuda2hipRename["cudaD3D10GetDevice"]                        = {"hipD3D10GetDevice", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                               // API_Driver ANALOGUE (cuD3D10GetDevice)
    cuda2hipRename["cudaD3D10GetDevices"]                       = {"hipD3D10GetDevices", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                              // API_Driver ANALOGUE (cuD3D10GetDevices)
    cuda2hipRename["cudaGraphicsD3D10RegisterResource"]         = {"hipGraphicsD3D10RegisterResource", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                // API_Driver ANALOGUE (cuGraphicsD3D10RegisterResource)

    // Direct3D 10 Interoperability [DEPRECATED]
    // enum cudaD3D10MapFlags
    cuda2hipRename["cudaD3D10MapFlags"]                         = {"hipD3D10MapFlags", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                                // API_Driver ANALOGUE (CUd3d10map_flags)
    cuda2hipRename["cudaD3D10MapFlagsNone"]                     = {"HIP_D3D10_MAPRESOURCE_FLAGS_NONE", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};           // 0 // API_Driver ANALOGUE (CU_D3D10_MAPRESOURCE_FLAGS_NONE)
    cuda2hipRename["cudaD3D10MapFlagsReadOnly"]                 = {"HIP_D3D10_MAPRESOURCE_FLAGS_READONLY", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};       // 1 // API_Driver ANALOGUE (CU_D3D10_MAPRESOURCE_FLAGS_READONLY)
    cuda2hipRename["cudaD3D10MapFlagsWriteDiscard"]             = {"HIP_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};   // 2 // API_Driver ANALOGUE (CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD)

    // enum cudaD3D10RegisterFlags
    cuda2hipRename["cudaD3D10RegisterFlags"]                    = {"hipD3D10RegisterFlags", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                           // API_Driver ANALOGUE (CUd3d10Register_flags)
    cuda2hipRename["cudaD3D10RegisterFlagsNone"]                = {"HIP_D3D10_REGISTER_FLAGS_NONE", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};              // 0 // API_Driver ANALOGUE (CU_D3D10_REGISTER_FLAGS_NONE)
    cuda2hipRename["cudaD3D10RegisterFlagsArray"]               = {"HIP_D3D10_REGISTER_FLAGS_ARRAY", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};             // 1 // API_Driver ANALOGUE (CU_D3D10_REGISTER_FLAGS_ARRAY)

    cuda2hipRename["cudaD3D10GetDirect3DDevice"]                = {"hipD3D10GetDirect3DDevice", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                          // API_Driver ANALOGUE (cudaD3D10GetDirect3DDevice)
    cuda2hipRename["cudaD3D10MapResources"]                     = {"hipD3D10MapResources", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                               // API_Driver ANALOGUE (cuD3D10MapResources)
    cuda2hipRename["cudaD3D10RegisterResource"]                 = {"hipD3D10RegisterResource", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                           // API_Driver ANALOGUE (cuD3D10RegisterResource)
    cuda2hipRename["cudaD3D10ResourceGetMappedArray"]           = {"hipD3D10ResourceGetMappedArray", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                     // API_Driver ANALOGUE (cuD3D10ResourceGetMappedArray)
    cuda2hipRename["cudaD3D10ResourceGetMappedPitch"]           = {"hipD3D10ResourceGetMappedPitch", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                     // API_Driver ANALOGUE (cudaD3D10ResourceGetMappedPitch)
    cuda2hipRename["cudaD3D10ResourceGetMappedPointer"]         = {"hipD3D10ResourceGetMappedPointer", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                   // API_Driver ANALOGUE (cuD3D10ResourceGetMappedPointer)
    cuda2hipRename["cudaD3D10ResourceGetMappedSize"]            = {"hipD3D10ResourceGetMappedSize", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                      // API_Driver ANALOGUE (cuD3D10ResourceGetMappedSize)
    cuda2hipRename["cudaD3D10ResourceGetSurfaceDimensions"]     = {"hipD3D10ResourceGetSurfaceDimensions", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};               // API_Driver ANALOGUE (cuD3D10ResourceGetSurfaceDimensions)
    cuda2hipRename["cudaD3D10ResourceSetMapFlags"]              = {"hipD3D10ResourceSetMapFlags", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                        // API_Driver ANALOGUE (cuD3D10ResourceSetMapFlags)
    cuda2hipRename["cudaD3D10SetDirect3DDevice"]                = {"hipD3D10SetDirect3DDevice", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                          // no API_Driver ANALOGUE
    cuda2hipRename["cudaD3D10UnmapResources"]                   = {"hipD3D10UnmapResources", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                             // API_Driver ANALOGUE (cuD3D10UnmapResources)
    cuda2hipRename["cudaD3D10UnregisterResource"]               = {"hipD3D10UnregisterResource", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED};                         // API_Driver ANALOGUE (cuD3D10UnregisterResource)

    // Direct3D 11 Interoperability
    // enum cudaD3D11DeviceList
    cuda2hipRename["cudaD3D11DeviceList"]                       = {"hipd3d11DeviceList", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED};                              // API_Driver ANALOGUE (CUd3d11DeviceList)
    cuda2hipRename["cudaD3D11DeviceListAll"]                    = {"HIP_D3D11_DEVICE_LIST_ALL", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED};                  // 1 // API_Driver ANALOGUE (CU_D3D11_DEVICE_LIST_ALL)
    cuda2hipRename["cudaD3D11DeviceListCurrentFrame"]           = {"HIP_D3D11_DEVICE_LIST_CURRENT_FRAME", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED};        // 2 // API_Driver ANALOGUE (CU_D3D11_DEVICE_LIST_CURRENT_FRAME)
    cuda2hipRename["cudaD3D11DeviceListNextFrame"]              = {"HIP_D3D11_DEVICE_LIST_NEXT_FRAME", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED};           // 3 // API_Driver ANALOGUE (CU_D3D11_DEVICE_LIST_NEXT_FRAME)

    cuda2hipRename["cudaD3D11GetDevice"]                        = {"hipD3D11GetDevice", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED};                               // API_Driver ANALOGUE (cuD3D11GetDevice)
    cuda2hipRename["cudaD3D11GetDevices"]                       = {"hipD3D11GetDevices", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED};                              // API_Driver ANALOGUE (cuD3D11GetDevices)
    cuda2hipRename["cudaGraphicsD3D11RegisterResource"]         = {"hipGraphicsD3D11RegisterResource", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED};                // API_Driver ANALOGUE (cuGraphicsD3D11RegisterResource)

    // Direct3D 11 Interoperability [DEPRECATED]
    cuda2hipRename["cudaD3D11GetDevice"]                        = {"hipD3D11GetDevice", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED};                               // API_Driver ANALOGUE (cuD3D11GetDevice)
    cuda2hipRename["cudaD3D11GetDevices"]                       = {"hipD3D11GetDevices", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED};                              // API_Driver ANALOGUE (cuD3D11GetDevices)
    cuda2hipRename["cudaGraphicsD3D11RegisterResource"]         = {"hipGraphicsD3D11RegisterResource", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED};                // API_Driver ANALOGUE (cuGraphicsD3D11RegisterResource)

    // VDPAU Interoperability
    cuda2hipRename["cudaGraphicsVDPAURegisterOutputSurface"]    = {"hipGraphicsVDPAURegisterOutputSurface", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED};           // API_Driver ANALOGUE (cuGraphicsVDPAURegisterOutputSurface)
    cuda2hipRename["cudaGraphicsVDPAURegisterVideoSurface"]     = {"hipGraphicsVDPAURegisterVideoSurface", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED};            // API_Driver ANALOGUE (cuGraphicsVDPAURegisterVideoSurface)
    cuda2hipRename["cudaVDPAUGetDevice"]                        = {"hipVDPAUGetDevice", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED};                               // API_Driver ANALOGUE (cuVDPAUGetDevice)
    cuda2hipRename["cudaVDPAUSetVDPAUDevice"]                   = {"hipVDPAUSetDevice", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED};                               // no API_Driver ANALOGUE

    // EGL Interoperability
    cuda2hipRename["cudaEglStreamConnection"]                   = {"hipEglStreamConnection", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED};                            // API_Driver ANALOGUE (CUeglStreamConnection)

    cuda2hipRename["cudaEGLStreamConsumerAcquireFrame"]         = {"hipEGLStreamConsumerAcquireFrame", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED};                  // API_Driver ANALOGUE (cuEGLStreamConsumerAcquireFrame)
    cuda2hipRename["cudaEGLStreamConsumerConnect"]              = {"hipEGLStreamConsumerConnect", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED};                       // API_Driver ANALOGUE (cuEGLStreamConsumerConnect)
    cuda2hipRename["cudaEGLStreamConsumerConnectWithFlags"]     = {"hipEGLStreamConsumerConnectWithFlags", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED};              // API_Driver ANALOGUE (cuEGLStreamConsumerConnectWithFlags)
    cuda2hipRename["cudaEGLStreamConsumerReleaseFrame"]         = {"hipEGLStreamConsumerReleaseFrame", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED};                  // API_Driver ANALOGUE (cuEGLStreamConsumerReleaseFrame)
    cuda2hipRename["cudaEGLStreamProducerConnect"]              = {"hipEGLStreamProducerConnect", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED};                       // API_Driver ANALOGUE (cuEGLStreamProducerConnect)
    cuda2hipRename["cudaEGLStreamProducerDisconnect"]           = {"hipEGLStreamProducerDisconnect", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED};                    // API_Driver ANALOGUE (cuEGLStreamProducerDisconnect)
    cuda2hipRename["cudaEGLStreamProducerPresentFrame"]         = {"hipEGLStreamProducerPresentFrame", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED};                  // API_Driver ANALOGUE (cuEGLStreamProducerPresentFrame)
    cuda2hipRename["cudaEGLStreamProducerReturnFrame"]          = {"hipEGLStreamProducerReturnFrame", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED};                   // API_Driver ANALOGUE (cuEGLStreamProducerReturnFrame)
    cuda2hipRename["cudaGraphicsEGLRegisterImage"]              = {"hipGraphicsEGLRegisterImage", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED};                       // API_Driver ANALOGUE (cuGraphicsEGLRegisterImage)
    cuda2hipRename["cudaGraphicsResourceGetMappedEglFrame"]     = {"hipGraphicsResourceGetMappedEglFrame", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED};              // API_Driver ANALOGUE (cuGraphicsResourceGetMappedEglFrame)

    //---------------------------------------BLAS-------------------------------------//
    // Blas types
    cuda2hipRename["cublasHandle_t"]                 = {"hipblasHandle_t", CONV_TYPE, API_BLAS};
    // TODO: dereferencing: typedef struct cublasContext *cublasHandle_t;
    // cuda2hipRename["cublasContext"]                  = {"hipblasHandle_t", CONV_TYPE, API_BLAS};
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
    cuda2hipRename["cublasSaxpyBatched"]             = {"hipblasSaxpyBatched", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasDaxpy"]                    = {"hipblasDaxpy", CONV_MATH_FUNC, API_BLAS};
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
    cuda2hipRename["cublasDgemv"]                    = {"hipblasDgemv", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
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
    cuda2hipRename["cublasDger"]                     = {"hipblasDger", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
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
    cuda2hipRename["cublasDgemm"]                    = {"hipblasDgemm", CONV_MATH_FUNC, API_BLAS};

    cuda2hipRename["cublasCgemm"]                    = {"hipblasCgemm", CONV_MATH_FUNC, API_BLAS};
    // unsupported yet by hipblas/hcblas
    cuda2hipRename["cublasZgemm"]                    = {"hipblasZgemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};

    // BATCH GEMM
    cuda2hipRename["cublasSgemmBatched"]             = {"hipblasSgemmBatched", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasDgemmBatched"]             = {"hipblasDgemmBatched", CONV_MATH_FUNC, API_BLAS};

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

    cuda2hipRename["cublasGetVersion_v2"]            = {"hipblasGetVersion", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED};
    cuda2hipRename["cublasSetStream_v2"]             = {"hipblasSetStream", CONV_MATH_FUNC, API_BLAS};
    cuda2hipRename["cublasGetStream_v2"]             = {"hipblasGetStream", CONV_MATH_FUNC, API_BLAS};
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
    cuda2hipRename["cublasDaxpy_v2"]                 = {"hipblasDaxpy", CONV_MATH_FUNC, API_BLAS};
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
                             << MacroNameTok.getIdentifierInfo()->getName() << "\n"
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
    SourceLocation sl = kernelDecl->getNameInfo().getEndLoc();
    SourceLocation kernelArgListStart = Lexer::findLocationAfterToken(sl, tok::l_paren, *SM, DefaultLangOptions, true);
    DEBUG(dbgs() << kernelArgListStart.printToString(*SM));
    if (kernelDecl->getNumParams() > 0) {
      const ParmVarDecl *pvdFirst = kernelDecl->getParamDecl(0);
      const ParmVarDecl *pvdLast =  kernelDecl->getParamDecl(kernelDecl->getNumParams() - 1);
      SourceLocation kernelArgListStart(pvdFirst->getLocStart());
      SourceLocation kernelArgListEnd(pvdLast->getLocEnd());
      SourceLocation stop = Lexer::getLocForEndOfToken(kernelArgListEnd, 0, *SM, DefaultLangOptions);
      size_t repLength = SM->getCharacterData(stop) - SM->getCharacterData(kernelArgListStart);
      OS << StringRef(SM->getCharacterData(kernelArgListStart), repLength);
      Replacement Rep0(*(Result.SourceManager), kernelArgListStart, repLength, OS.str());
      FullSourceLoc fullSL(sl, *(Result.SourceManager));
      insertReplacement(Rep0, fullSL);
    }
  }

  bool cudaCall(const MatchFinder::MatchResult &Result) {
    if (const CallExpr *call = Result.Nodes.getNodeAs<CallExpr>("cudaCall")) {
      const FunctionDecl *funcDcl = call->getDirectCallee();
      std::string name = funcDcl->getDeclName().getAsString();
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
            updateCounters(found->second, name);
            Replacement Rep(*SM, sl, length, repName);
            FullSourceLoc fullSL(sl, *SM);
            insertReplacement(Rep, fullSL);
          }
        } else {
          updateCounters(found->second, name);
        }
      } else {
        std::string msg = "the following reference is not handled: '" + name + "' [function call].";
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
        calleeName = Twine("(" + calleeName + ")").toStringRef(tmpData);
      }
      OS << "hipLaunchKernelGGL(" << calleeName << ",";
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
      hipCounter counter = {"hipLaunchKernelGGL", CONV_KERN, API_RUNTIME};
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
      StringRef name = enumConstantRef->getDecl()->getName();
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

  bool cudaEnumDecl(const MatchFinder::MatchResult &Result) {
    if (const VarDecl *enumDecl = Result.Nodes.getNodeAs<VarDecl>("cudaEnumDecl")) {
      std::string name = enumDecl->getType()->getAsTagDecl()->getNameAsString();
      QualType QT = enumDecl->getType().getUnqualifiedType();
      std::string name_unqualified = QT.getAsString();
      if ((name_unqualified.find(' ') == std::string::npos && name.find(' ') == std::string::npos) || name.empty()) {
        name = name_unqualified;
      }
      // Workaround for enum VarDecl as param decl, declared with enum type specifier
      // Example: void func(enum cudaMemcpyKind kind);
      //-------------------------------------------------
      SourceManager *SM = Result.SourceManager;
      SourceLocation sl(enumDecl->getLocStart());
      SourceLocation end(enumDecl->getLocEnd());
      size_t repLength = SM->getCharacterData(end) - SM->getCharacterData(sl);
      StringRef sfull = StringRef(SM->getCharacterData(sl), repLength);
      size_t offset = sfull.find(name);
      if (offset > 0) {
        sl = sl.getLocWithOffset(offset);
      }
      //-------------------------------------------------
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        updateCounters(found->second, name);
        if (!found->second.unsupported) {
          StringRef repName = found->second.hipName;
          Replacement Rep(*SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, *SM);
          insertReplacement(Rep, fullSL);
        }
      } else {
        std::string msg = "the following reference is not handled: '" + name + "' [enum constant decl].";
        printHipifyMessage(*SM, sl, msg);
      }
      return true;
    }
    return false;
  }

  bool cudaEnumVarPtr(const MatchFinder::MatchResult &Result) {
    if (const VarDecl *enumVarPtr = Result.Nodes.getNodeAs<VarDecl>("cudaEnumVarPtr")) {
      const Type *t = enumVarPtr->getType().getTypePtrOrNull();
      if (t) {
        QualType QT = t->getPointeeType();
        std::string name = QT.getAsString();
        QT = enumVarPtr->getType().getUnqualifiedType();
        std::string name_unqualified = QT.getAsString();
        if ((name_unqualified.find(' ') == std::string::npos && name.find(' ') == std::string::npos) || name.empty()) {
          name = name_unqualified;
        }
        // Workaround for enum VarDecl as param decl, declared with enum type specifier
        // Example: void func(enum cudaMemcpyKind kind);
        //-------------------------------------------------
        SourceManager *SM = Result.SourceManager;
        TypeLoc TL = enumVarPtr->getTypeSourceInfo()->getTypeLoc();
        SourceLocation sl(TL.getUnqualifiedLoc().getLocStart());
        SourceLocation end(TL.getUnqualifiedLoc().getLocEnd());
        size_t repLength = SM->getCharacterData(end) - SM->getCharacterData(sl);
        StringRef sfull = StringRef(SM->getCharacterData(sl), repLength);
        size_t offset = sfull.find(name);
        if (offset > 0) {
          sl = sl.getLocWithOffset(offset);
        }
        //-------------------------------------------------
        const auto found = N.cuda2hipRename.find(name);
        if (found != N.cuda2hipRename.end()) {
          updateCounters(found->second, name);
          if (!found->second.unsupported) {
            StringRef repName = found->second.hipName;
            Replacement Rep(*SM, sl, name.size(), repName);
            FullSourceLoc fullSL(sl, *SM);
            insertReplacement(Rep, fullSL);
          }
        }
        else {
          std::string msg = "the following reference is not handled: '" + name + "' [enum var ptr].";
          printHipifyMessage(*SM, sl, msg);
        }
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
      std::string name = QT.getAsString();
      SourceLocation sl = typedefVar->getLocStart();
      SourceManager *SM = Result.SourceManager;
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        updateCounters(found->second, name);
        if (!found->second.unsupported) {
          StringRef repName = found->second.hipName;
          Replacement Rep(*SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, *SM);
          insertReplacement(Rep, fullSL);
        }
      } else {
        std::string msg = "the following reference is not handled: '" + name + "' [typedef var].";
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
        std::string name = QT.getAsString();
        const auto found = N.cuda2hipRename.find(name);
        if (found != N.cuda2hipRename.end()) {
          updateCounters(found->second, name);
          if (!found->second.unsupported) {
            StringRef repName = found->second.hipName;
            Replacement Rep(*SM, sl, name.size(), repName);
            FullSourceLoc fullSL(sl, *SM);
            insertReplacement(Rep, fullSL);
          }
        }
        else {
          std::string msg = "the following reference is not handled: '" + name + "' [typedef var ptr].";
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
        std::string name = QT.getTypePtr()->getAsStructureType()->getDecl()->getNameAsString();
        TypeLoc TL = structVar->getTypeSourceInfo()->getTypeLoc();
        SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
        SourceManager *SM = Result.SourceManager;
        const auto found = N.cuda2hipRename.find(name);
        if (found != N.cuda2hipRename.end()) {
          updateCounters(found->second, name);
          if (!found->second.unsupported) {
            StringRef repName = found->second.hipName;
            Replacement Rep(*SM, sl, name.size(), repName);
            FullSourceLoc fullSL(sl, *SM);
            insertReplacement(Rep, fullSL);
          }
        }
        else {
          std::string msg = "the following reference is not handled: '" + name + "' [struct var].";
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

  bool cudaNewOperatorDecl(const MatchFinder::MatchResult &Result) {
    if (const auto *newOperator = Result.Nodes.getNodeAs<CXXNewExpr>("cudaNewOperatorDecl")) {
      const Type *t = newOperator->getType().getTypePtrOrNull();
      if (t) {
        SourceManager *SM = Result.SourceManager;
        TypeLoc TL = newOperator->getAllocatedTypeSourceInfo()->getTypeLoc();
        SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
        QualType QT = t->getPointeeType();
        std::string name = QT.getAsString();
        const auto found = N.cuda2hipRename.find(name);
        if (found != N.cuda2hipRename.end()) {
          updateCounters(found->second, name);
          if (!found->second.unsupported) {
            StringRef repName = found->second.hipName;
            Replacement Rep(*SM, sl, name.size(), repName);
            FullSourceLoc fullSL(sl, *SM);
            insertReplacement(Rep, fullSL);
          }
        }
        else {
          std::string msg = "the following reference is not handled: '" + name + "' [new operator].";
          printHipifyMessage(*SM, sl, msg);
        }
      }
    }
    return false;
  }

  bool cudaFunctionReturn(const MatchFinder::MatchResult &Result) {
    if (const auto *ret = Result.Nodes.getNodeAs<FunctionDecl>("cudaFunctionReturn")) {
      QualType QT = ret->getReturnType();
      SourceManager *SM = Result.SourceManager;
      SourceRange sr = ret->getReturnTypeSourceRange();
      SourceLocation sl = sr.getBegin();
      std::string name = QT.getAsString();
      if (QT.getTypePtr()->isEnumeralType()) {
        name = QT.getTypePtr()->getAs<EnumType>()->getDecl()->getNameAsString();
      }
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        updateCounters(found->second, name);
        if (!found->second.unsupported) {
          StringRef repName = found->second.hipName;
          Replacement Rep(*SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, *SM);
          insertReplacement(Rep, fullSL);
        }
      }
      else {
        std::string msg = "the following reference is not handled: '" + name + "' [function return].";
        printHipifyMessage(*SM, sl, msg);
      }
    }
    return false;
  }

  bool cudaSharedIncompleteArrayVar(const MatchFinder::MatchResult &Result) {
    StringRef refName = "cudaSharedIncompleteArrayVar";
    if (const VarDecl *sharedVar = Result.Nodes.getNodeAs<VarDecl>(refName)) {
      // Example: extern __shared__ uint sRadix1[];
      if (sharedVar->hasExternalFormalLinkage()) {
        QualType QT = sharedVar->getType();
        std::string typeName;
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
          std::string varName = sharedVar->getNameAsString();
          std::string repName = "HIP_DYNAMIC_SHARED(" + typeName + ", " + varName + ")";
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
      std::string name = QT.getAsString();
      const Type *t = QT.getTypePtr();
      if (t->isStructureOrClassType()) {
        name = t->getAsCXXRecordDecl()->getName();
      }
      TypeLoc TL = paramDecl->getTypeSourceInfo()->getTypeLoc();
      SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
      SourceManager *SM = Result.SourceManager;
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        updateCounters(found->second, name);
        if (!found->second.unsupported) {
          StringRef repName = found->second.hipName;
          Replacement Rep(*SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, *SM);
          insertReplacement(Rep, fullSL);
        }
      } else {
        std::string msg = "the following reference is not handled: '" + name + "' [param decl].";
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
      if (cudaEnumDecl(Result)) break;
      if (cudaEnumVarPtr(Result)) break;
      if (cudaTypedefVar(Result)) break;
      if (cudaTypedefVarPtr(Result)) break;
      if (cudaStructVar(Result)) break;
      if (cudaStructVarPtr(Result)) break;
      if (cudaStructSizeOf(Result)) break;
      if (cudaParamDecl(Result)) break;
      if (cudaParamDeclPtr(Result)) break;
      if (cudaLaunchKernel(Result)) break;
      if (cudaNewOperatorDecl(Result)) break;
      if (cudaFunctionReturn(Result)) break;
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
                            .bind("cudaEnumDecl"),
                            Callback);
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(pointsTo(enumDecl(
                            matchesName("cu.*|CU.*")))))
                            .bind("cudaEnumVarPtr"),
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
  // Example:
  // CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
  // hipJitOption *jitOptions = new hipJitOption[jitNumOptions];
  Finder.addMatcher(cxxNewExpr(isExpansionInMainFile(),
                               hasType(pointsTo(namedDecl(matchesName("cu.*|CU.*")))))
                              .bind("cudaNewOperatorDecl"),
                               Callback);
  // Examples:
  // 1.
  // cudaStream_t cuda_memcpy_stream(...)
  // 2.
  // template<typename System1, typename System2> cudaMemcpyKind cuda_memcpy_kind(...)
  Finder.addMatcher(functionDecl(isExpansionInMainFile(),
                                 returns(hasDeclaration(namedDecl(matchesName("cu.*|CU.*")))))
                                .bind("cudaFunctionReturn"),
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
#if (LLVM_VERSION_MAJOR >= 3) && (LLVM_VERSION_MINOR >= 9)
  llvm::sys::PrintStackTraceOnErrorSignal(StringRef());
#else
  llvm::sys::PrintStackTraceOnErrorSignal();
#endif
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
