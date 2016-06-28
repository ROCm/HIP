/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

#define DEBUG_TYPE "cuda2hip"

enum ConvTypes {
  CONV_DEV = 0,
  CONV_MEM,
  CONV_KERN,
  CONV_COORD_FUNC,
  CONV_MATH_FUNC,
  CONV_SPECIAL_FUNC,
  CONV_STREAM,
  CONV_EVENT,
  CONV_ERR,
  CONV_DEF,
  CONV_TEX,
  CONV_OTHER,
  CONV_INCLUDE,
  CONV_LITERAL,
  CONV_BLAS,
  CONV_LAST
};

const char *counterNames[ConvTypes::CONV_LAST] = {
    "dev",          "mem",    "kern",    "coord_func", "math_func",
    "special_func", "stream", "event",   "err",        "def",
    "tex",          "other",  "include", "literal",    "blas"};

namespace {

struct cuda2hipMap {
  cuda2hipMap() {
    // Defines
    cuda2hipRename["__CUDACC__"] = {"__HIPCC__", CONV_DEF};

    // CUDA includes
    cuda2hipRename["cuda_runtime.h"]     = {"hip_runtime.h", CONV_INCLUDE};
    cuda2hipRename["cuda_runtime_api.h"] = {"hip_runtime_api.h", CONV_INCLUDE};

    // CUBLAS includes
    cuda2hipRename["cublas.h"]           = {"hipblas.h", CONV_INCLUDE};
    cuda2hipRename["cublas_v2.h"]        = {"hipblas.h", CONV_INCLUDE};

    // Error codes and return types
    cuda2hipRename["cudaError_t"]                    = {"hipError_t", CONV_ERR};
    cuda2hipRename["cudaError"]                      = {"hipError", CONV_ERR};
    cuda2hipRename["cudaSuccess"]                    = {"hipSuccess", CONV_ERR};
    cuda2hipRename["cudaErrorUnknown"]               = {"hipErrorUnknown", CONV_ERR};
    cuda2hipRename["cudaErrorMemoryAllocation"]      = {"hipErrorMemoryAllocation", CONV_ERR};
    cuda2hipRename["cudaErrorMemoryFree"]            = {"hipErrorMemoryFree", CONV_ERR};
    cuda2hipRename["cudaErrorUnknownSymbol"]         = {"hipErrorUnknownSymbol", CONV_ERR};
    cuda2hipRename["cudaErrorOutOfResources"]        = {"hipErrorOutOfResources", CONV_ERR};
    cuda2hipRename["cudaErrorInvalidValue"]          = {"hipErrorInvalidValue", CONV_ERR};
    cuda2hipRename["cudaErrorInvalidResourceHandle"] = {"hipErrorInvalidResourceHandle", CONV_ERR};
    cuda2hipRename["cudaErrorInvalidDevice"]         = {"hipErrorInvalidDevice", CONV_ERR};
    cuda2hipRename["cudaErrorNoDevice"]              = {"hipErrorNoDevice", CONV_ERR};
    cuda2hipRename["cudaErrorNotReady"]              = {"hipErrorNotReady", CONV_ERR};
    cuda2hipRename["cudaErrorUnknown"]               = {"hipErrorUnknown", CONV_ERR};

    // Error API
    cuda2hipRename["cudaGetLastError"]               = {"hipGetLastError", CONV_ERR};
    cuda2hipRename["cudaPeekAtLastError"]            = {"hipPeekAtLastError", CONV_ERR};
    cuda2hipRename["cudaGetErrorName"]               = {"hipGetErrorName", CONV_ERR};
    cuda2hipRename["cudaGetErrorString"]             = {"hipGetErrorString", CONV_ERR};

    // Memcpy
    cuda2hipRename["cudaMemcpy"]               = {"hipMemcpy", CONV_MEM};
    cuda2hipRename["cudaMemcpyHostToHost"]     = {"hipMemcpyHostToHost", CONV_MEM};
    cuda2hipRename["cudaMemcpyHostToDevice"]   = {"hipMemcpyHostToDevice", CONV_MEM};
    cuda2hipRename["cudaMemcpyDeviceToHost"]   = {"hipMemcpyDeviceToHost", CONV_MEM};
    cuda2hipRename["cudaMemcpyDeviceToDevice"] = {"hipMemcpyDeviceToDevice", CONV_MEM};
    cuda2hipRename["cudaMemcpyDefault"]        = {"hipMemcpyDefault", CONV_MEM};
    cuda2hipRename["cudaMemcpyToSymbol"]       = {"hipMemcpyToSymbol", CONV_MEM};
    cuda2hipRename["cudaMemset"]               = {"hipMemset", CONV_MEM};
    cuda2hipRename["cudaMemsetAsync"]          = {"hipMemsetAsync", CONV_MEM};
    cuda2hipRename["cudaMemcpyAsync"]          = {"hipMemcpyAsync", CONV_MEM};
    cuda2hipRename["cudaMemGetInfo"]           = {"hipMemGetInfo", CONV_MEM};
    cuda2hipRename["cudaMemcpyKind"]           = {"hipMemcpyKind", CONV_MEM};

    // Memory management
    cuda2hipRename["cudaMalloc"]     = {"hipMalloc", CONV_MEM};
    cuda2hipRename["cudaMallocHost"] = {"hipHostAlloc", CONV_MEM};
    cuda2hipRename["cudaFree"]       = {"hipFree", CONV_MEM};
    cuda2hipRename["cudaFreeHost"]   = {"hipHostFree", CONV_MEM};

    // Coordinate Indexing and Dimensions
    cuda2hipRename["threadIdx.x"] = {"hipThreadIdx_x", CONV_COORD_FUNC};
    cuda2hipRename["threadIdx.y"] = {"hipThreadIdx_y", CONV_COORD_FUNC};
    cuda2hipRename["threadIdx.z"] = {"hipThreadIdx_z", CONV_COORD_FUNC};

    cuda2hipRename["blockIdx.x"]  = {"hipBlockIdx_x", CONV_COORD_FUNC};
    cuda2hipRename["blockIdx.y"]  = {"hipBlockIdx_y", CONV_COORD_FUNC};
    cuda2hipRename["blockIdx.z"]  = {"hipBlockIdx_z", CONV_COORD_FUNC};

    cuda2hipRename["blockDim.x"]  = {"hipBlockDim_x", CONV_COORD_FUNC};
    cuda2hipRename["blockDim.y"]  = {"hipBlockDim_y", CONV_COORD_FUNC};
    cuda2hipRename["blockDim.z"]  = {"hipBlockDim_z", CONV_COORD_FUNC};

    cuda2hipRename["gridDim.x"]   = {"hipGridDim_x", CONV_COORD_FUNC};
    cuda2hipRename["gridDim.y"]   = {"hipGridDim_y", CONV_COORD_FUNC};
    cuda2hipRename["gridDim.z"]   = {"hipGridDim_z", CONV_COORD_FUNC};

    cuda2hipRename["blockIdx.x"]  = {"hipBlockIdx_x", CONV_COORD_FUNC};
    cuda2hipRename["blockIdx.y"]  = {"hipBlockIdx_y", CONV_COORD_FUNC};
    cuda2hipRename["blockIdx.z"]  = {"hipBlockIdx_z", CONV_COORD_FUNC};

    cuda2hipRename["blockDim.x"]  = {"hipBlockDim_x", CONV_COORD_FUNC};
    cuda2hipRename["blockDim.y"]  = {"hipBlockDim_y", CONV_COORD_FUNC};
    cuda2hipRename["blockDim.z"]  = {"hipBlockDim_z", CONV_COORD_FUNC};

    cuda2hipRename["gridDim.x"]   = {"hipGridDim_x", CONV_COORD_FUNC};
    cuda2hipRename["gridDim.y"]   = {"hipGridDim_y", CONV_COORD_FUNC};
    cuda2hipRename["gridDim.z"]   = {"hipGridDim_z", CONV_COORD_FUNC};

    cuda2hipRename["warpSize"]    = {"hipWarpSize", CONV_SPECIAL_FUNC};

    // Events
    cuda2hipRename["cudaEvent_t"]              = {"hipEvent_t", CONV_EVENT};
    cuda2hipRename["cudaEventCreate"]          = {"hipEventCreate", CONV_EVENT};
    cuda2hipRename["cudaEventCreateWithFlags"] = {"hipEventCreateWithFlags", CONV_EVENT};
    cuda2hipRename["cudaEventDestroy"]         = {"hipEventDestroy", CONV_EVENT};
    cuda2hipRename["cudaEventRecord"]          = {"hipEventRecord", CONV_EVENT};
    cuda2hipRename["cudaEventElapsedTime"]     = {"hipEventElapsedTime", CONV_EVENT};
    cuda2hipRename["cudaEventSynchronize"]     = {"hipEventSynchronize", CONV_EVENT};

    // Streams
    cuda2hipRename["cudaStream_t"]              = {"hipStream_t", CONV_STREAM};
    cuda2hipRename["cudaStreamCreate"]          = {"hipStreamCreate", CONV_STREAM};
    cuda2hipRename["cudaStreamCreateWithFlags"] = {"hipStreamCreateWithFlags", CONV_STREAM};
    cuda2hipRename["cudaStreamDestroy"]         = {"hipStreamDestroy", CONV_STREAM};
    cuda2hipRename["cudaStreamWaitEvent"]       = {"hipStreamWaitEven", CONV_STREAM};
    cuda2hipRename["cudaStreamSynchronize"]     = {"hipStreamSynchronize", CONV_STREAM};
    cuda2hipRename["cudaStreamDefault"]         = {"hipStreamDefault", CONV_STREAM};
    cuda2hipRename["cudaStreamNonBlocking"]     = {"hipStreamNonBlocking", CONV_STREAM};

    // Other synchronization
    cuda2hipRename["cudaDeviceSynchronize"] = {"hipDeviceSynchronize", CONV_DEV};
    // translate deprecated cudaThreadSynchronize
    cuda2hipRename["cudaThreadSynchronize"] = {"hipDeviceSynchronize", CONV_DEV};
    cuda2hipRename["cudaDeviceReset"]       = {"hipDeviceReset", CONV_DEV};
    // translate deprecated cudaThreadExit
    cuda2hipRename["cudaThreadExit"]        = {"hipDeviceReset", CONV_DEV};
    cuda2hipRename["cudaSetDevice"]         = {"hipSetDevice", CONV_DEV};
    cuda2hipRename["cudaGetDevice"]         = {"hipGetDevice", CONV_DEV};

    // Attribute
    cuda2hipRename["bcudaDeviceAttr"]         = {"hipDeviceAttribute_t", CONV_DEV};
    cuda2hipRename["bcudaDeviceGetAttribute"] = {"hipDeviceGetAttribute", CONV_DEV};

    // Device
    cuda2hipRename["cudaDeviceProp"]          = {"hipDeviceProp_t", CONV_DEV};
    cuda2hipRename["cudaGetDeviceProperties"] = {"hipGetDeviceProperties", CONV_DEV};

    // Cache config
    cuda2hipRename["cudaDeviceSetCacheConfig"]  = {"hipDeviceSetCacheConfig", CONV_DEV};
    // translate deprecated
    cuda2hipRename["cudaThreadSetCacheConfig"] = {"hipDeviceSetCacheConfig", CONV_DEV};
    cuda2hipRename["cudaDeviceGetCacheConfig"] = {"hipDeviceGetCacheConfig", CONV_DEV};
    // translate deprecated
    cuda2hipRename["cudaThreadGetCacheConfig"]  = {"hipDeviceGetCacheConfig", CONV_DEV};
    cuda2hipRename["cudaFuncCache"]             = {"hipFuncCache", CONV_DEV};
    cuda2hipRename["cudaFuncCachePreferNone"]   = {"hipFuncCachePreferNone", CONV_DEV};
    cuda2hipRename["cudaFuncCachePreferShared"] = {"hipFuncCachePreferShared", CONV_DEV};
    cuda2hipRename["cudaFuncCachePreferL1"]     = {"hipFuncCachePreferL1", CONV_DEV};
    cuda2hipRename["cudaFuncCachePreferEqual"]  = {"hipFuncCachePreferEqual", CONV_DEV};
    cuda2hipRename["cudaFuncSetCacheConfig"]    = {"hipFuncSetCacheConfig", CONV_DEV};

    // Driver/Runtime
    cuda2hipRename["cudaDriverGetVersion"] = {"hipDriverGetVersion", CONV_DEV};
    cuda2hipRename["cudaGetDeviceCount"]   = { "hipGetDeviceCount", CONV_DEV };
    // unsupported yet
    //cuda2hipRename["cudaRuntimeGetVersion"] = {"hipRuntimeGetVersion", CONV_DEV};

    // Peer2Peer
    cuda2hipRename["cudaDeviceCanAccessPeer"]     = {"hipDeviceCanAccessPeer", CONV_DEV};
    cuda2hipRename["cudaDeviceDisablePeerAccess"] = {"hipDeviceDisablePeerAccess", CONV_DEV};
    cuda2hipRename["cudaDeviceEnablePeerAccess"]  = {"hipDeviceEnablePeerAccess", CONV_DEV};
    cuda2hipRename["cudaMemcpyPeerAsync"]         = {"hipMemcpyPeerAsync", CONV_MEM};
    cuda2hipRename["cudaMemcpyPeer"]              = {"hipMemcpyPeer", CONV_MEM};

    // Shared memory
    cuda2hipRename["cudaDeviceSetSharedMemConfig"]   = {"hipDeviceSetSharedMemConfig", CONV_DEV};
    // translate deprecated
    cuda2hipRename["cudaThreadSetSharedMemConfig"]   = {"hipDeviceSetSharedMemConfig", CONV_DEV};
    cuda2hipRename["cudaDeviceGetSharedMemConfig"]   = {"hipDeviceGetSharedMemConfig", CONV_DEV};
    // translate deprecated
    cuda2hipRename["cudaThreadGetSharedMemConfig"]   = {"hipDeviceGetSharedMemConfig", CONV_DEV};
    cuda2hipRename["cudaSharedMemConfig"]            = {"hipSharedMemConfig", CONV_DEV};
    cuda2hipRename["cudaSharedMemBankSizeDefault"]   = {"hipSharedMemBankSizeDefault", CONV_DEV};
    cuda2hipRename["cudaSharedMemBankSizeFourByte"]  = {"hipSharedMemBankSizeFourByte", CONV_DEV};
    cuda2hipRename["cudaSharedMemBankSizeEightByte"] = {"hipSharedMemBankSizeEightByte", CONV_DEV};

    // Profiler
    // unsupported yet
    //cuda2hipRename["cudaProfilerInitialize"]  = {"hipProfilerInitialize", CONV_OTHER};
    cuda2hipRename["cudaProfilerStart"]       = {"hipProfilerStart", CONV_OTHER};
    cuda2hipRename["cudaProfilerStop"]        = {"hipProfilerStop", CONV_OTHER};
    cuda2hipRename["cudaChannelFormatDesc"]   = {"hipChannelFormatDesc", CONV_TEX};
    cuda2hipRename["cudaFilterModePoint"]     = {"hipFilterModePoint", CONV_TEX};
    cuda2hipRename["cudaReadModeElementType"] = {"hipReadModeElementType", CONV_TEX};

    // Channel descriptor
    cuda2hipRename["cudaCreateChannelDesc"]   = {"hipCreateChannelDesc", CONV_TEX};
    cuda2hipRename["cudaBindTexture"]         = {"hipBindTexture", CONV_TEX};
    cuda2hipRename["cudaUnbindTexture"]       = {"hipUnbindTexture", CONV_TEX};

    //---------------------------------------BLAS-------------------------------------//
    // Blas types
    cuda2hipRename["cublasHandle_t"]                 = {"hipblasHandle_t", CONV_BLAS};
    // Blas operations
    cuda2hipRename["cublasOperation_t"]              = {"hipblasOperation_t", CONV_BLAS};
    cuda2hipRename["CUBLAS_OP_N"]                    = {"HIPBLAS_OP_N", CONV_BLAS};
    cuda2hipRename["CUBLAS_OP_T"]                    = {"HIPBLAS_OP_T", CONV_BLAS};
    cuda2hipRename["CUBLAS_OP_C"]                    = {"HIPBLAS_OP_C", CONV_BLAS};
    // Blas statuses
    cuda2hipRename["cublasStatus_t"]                 = {"hipblasStatus_t", CONV_BLAS};
    cuda2hipRename["CUBLAS_STATUS_SUCCESS"]          = {"HIPBLAS_STATUS_SUCCESS", CONV_BLAS};
    cuda2hipRename["CUBLAS_STATUS_NOT_INITIALIZED"]  = {"HIPBLAS_STATUS_NOT_INITIALIZED", CONV_BLAS};
    cuda2hipRename["CUBLAS_STATUS_ALLOC_FAILED"]     = {"HIPBLAS_STATUS_ALLOC_FAILED", CONV_BLAS};
    cuda2hipRename["CUBLAS_STATUS_INVALID_VALUE"]    = {"HIPBLAS_STATUS_INVALID_VALUE", CONV_BLAS};
    cuda2hipRename["CUBLAS_STATUS_MAPPING_ERROR"]    = {"HIPBLAS_STATUS_MAPPING_ERROR", CONV_BLAS};
    cuda2hipRename["CUBLAS_STATUS_EXECUTION_FAILED"] = {"HIPBLAS_STATUS_EXECUTION_FAILED", CONV_BLAS};
    cuda2hipRename["CUBLAS_STATUS_INTERNAL_ERROR"]   = {"HIPBLAS_STATUS_INTERNAL_ERROR", CONV_BLAS};
    cuda2hipRename["CUBLAS_STATUS_NOT_SUPPORTED"]    = {"HIPBLAS_STATUS_INTERNAL_ERROR", CONV_BLAS};
    // Blas Fill Modes
    // unsupported yet by hipblas/hcblas
    //cuda2hipRename["cublasFillMode_t"]               = {"hipblasFillMode_t", CONV_BLAS};
    //cuda2hipRename["CUBLAS_FILL_MODE_LOWER"]         = {"HIPBLAS_FILL_MODE_LOWER", CONV_BLAS};
    //cuda2hipRename["CUBLAS_FILL_MODE_UPPER"]         = {"HIPBLAS_FILL_MODE_UPPER", CONV_BLAS};
    // Blas Diag Types
    // unsupported yet by hipblas/hcblas
    //cuda2hipRename["cublasDiagType_t"]               = {"hipblasDiagType_t", CONV_BLAS};
    //cuda2hipRename["CUBLAS_DIAG_NON_UNIT"]           = {"HIPBLAS_DIAG_NON_UNIT", CONV_BLAS};
    //cuda2hipRename["CUBLAS_DIAG_UNIT"]               = {"HIPBLAS_DIAG_UNIT", CONV_BLAS};
    // Blas Side Modes
    // unsupported yet by hipblas/hcblas
    //cuda2hipRename["cublasSideMode_t"]               = {"hipblasSideMode_t", CONV_BLAS};
    //cuda2hipRename["CUBLAS_SIDE_LEFT"]               = {"HIPBLAS_SIDE_LEFT", CONV_BLAS};
    //cuda2hipRename["CUBLAS_SIDE_RIGHT"]              = {"HIPBLAS_SIDE_RIGHT", CONV_BLAS};
    // Blas Pointer Modes
    // unsupported yet by hipblas/hcblas
    //cuda2hipRename["cublasPointerMode_t"]            = {"hipblasPointerMode_t", CONV_BLAS};
    //cuda2hipRename["CUBLAS_POINTER_MODE_HOST"]       = {"HIPBLAS_POINTER_MODE_HOST", CONV_BLAS};
    //cuda2hipRename["CUBLAS_POINTER_MODE_DEVICE"]     = {"HIPBLAS_POINTER_MODE_DEVICE", CONV_BLAS};
    // Blas Atomics Modes
    // unsupported yet by hipblas/hcblas
    //cuda2hipRename["cublasAtomicsMode_t"]            = {"hipblasAtomicsMode_t", CONV_BLAS};
    //cuda2hipRename["CUBLAS_ATOMICS_NOT_ALLOWED"]     = {"HIPBLAS_ATOMICS_NOT_ALLOWED", CONV_BLAS};
    //cuda2hipRename["CUBLAS_ATOMICS_ALLOWED"]         = {"HIPBLAS_ATOMICS_ALLOWED", CONV_BLAS};
    // Blas Data Type
    // unsupported yet by hipblas/hcblas
    //cuda2hipRename["cublasDataType_t"]               = {"hipblasDataType_t", CONV_BLAS};
    //cuda2hipRename["CUBLAS_DATA_FLOAT"]              = {"HIPBLAS_DATA_FLOAT", CONV_BLAS};
    //cuda2hipRename["CUBLAS_DATA_DOUBLE"]             = {"HIPBLAS_DATA_DOUBLE", CONV_BLAS};
    //cuda2hipRename["CUBLAS_DATA_HALF"]               = {"HIPBLAS_DATA_HALF", CONV_BLAS};
    //cuda2hipRename["CUBLAS_DATA_INT8"]               = {"HIPBLAS_DATA_INT8", CONV_BLAS};

    // Blas1 (v1) Routines
    cuda2hipRename["cublasCreate"]       = {"hipblasCreate", CONV_BLAS};
    cuda2hipRename["cublasDestroy"]      = {"hipblasDestroy", CONV_BLAS};

    cuda2hipRename["cublasSetVector"]    = {"hipblasSetVector", CONV_BLAS};
    cuda2hipRename["cublasGetVector"]    = {"hipblasGetVector", CONV_BLAS};
    cuda2hipRename["cublasSetMatrix"]    = {"hipblasSetMatrix", CONV_BLAS};
    cuda2hipRename["cublasGetMatrix"]    = {"hipblasGetMatrix", CONV_BLAS};

    // unsupported yet by hipblas/hcblas
    //cuda2hipRename["cublasGetMatrixAsync"] = {"hipblasGetMatrixAsync", CONV_BLAS};
    //cuda2hipRename["cublasSetMatrixAsync"] = {"hipblasSetMatrixAsync", CONV_BLAS};

    // NRM2
    //cuda2hipRename["cublasSnrm2"]  = {"hipblasSnrm2", CONV_BLAS};
    //cuda2hipRename["cublasDnrm2"]  = {"hipblasDnrm2", CONV_BLAS};
    //cuda2hipRename["cublasScnrm2"] = {"hipblasScnrm2", CONV_BLAS};
    //cuda2hipRename["cublasDznrm2"] = {"hipblasDznrm2", CONV_BLAS};

    // DOT
    cuda2hipRename["cublasSdot"]        = {"hipblasSdot", CONV_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasSdotBatched"] = {"hipblasSdotBatched",CONV_BLAS};
    cuda2hipRename["cublasDdot"]        = {"hipblasDdot", CONV_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasDdotBatched"] = {"hipblasDdotBatched", CONV_BLAS};
    //cuda2hipRename["cublasCdotu"] = {"hipblasCdotu", CONV_BLAS};
    //cuda2hipRename["cublasCdotc"] = {"hipblasCdotc", CONV_BLAS};
    //cuda2hipRename["cublasZdotu"] = {"hipblasZdotu", CONV_BLAS};
    //cuda2hipRename["cublasZdotc"] = {"hipblasZdotc", CONV_BLAS};

    // SCAL
    cuda2hipRename["cublasSscal"] = {"hipblasSscal", CONV_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasSscalBatched"] = {"hipblasSscalBatched", CONV_BLAS};
    cuda2hipRename["cublasDscal"] = {"hipblasDscal", CONV_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasDscalBatched"] = {"hipblasDscalBatched", CONV_BLAS};
    //cuda2hipRename["cublasCscal"]  = {"hipblasCscal", CONV_BLAS};
    //cuda2hipRename["cublasCsscal"] = {"hipblasCsscal", CONV_BLAS};
    //cuda2hipRename["cublasZscal"]  = {"hipblasZscal", CONV_BLAS};
    //cuda2hipRename["cublasZdscal"] = {"hipblasZdscal", CONV_BLAS};

    // AXPY
    cuda2hipRename["cublasSaxpy"] = {"hipblasSaxpy", CONV_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasSaxpyBatched"] = {"hipblasSaxpyBatched", CONV_BLAS};
    //cuda2hipRename["cublasDaxpy"] = {"hipblasDaxpy", CONV_BLAS};
    //cuda2hipRename["cublasCaxpy"] = {"hipblasCaxpy", CONV_BLAS};
    //cuda2hipRename["cublasZaxpy"] = {"hipblasZaxpy", CONV_BLAS};

    // COPY
    cuda2hipRename["cublasScopy"] = {"hipblasScopy", CONV_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasScopyBatched"] = {"hipblasScopyBatched", CONV_BLAS};
    cuda2hipRename["cublasDcopy"] = {"hipblasDcopy", CONV_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasDcopyBatched"] = {"hipblasDcopyBatched", CONV_BLAS};
    //cuda2hipRename["cublasCcopy"] = {"hipblasCcopy", CONV_BLAS};
    //cuda2hipRename["cublasZcopy"] = {"hipblasZcopy", CONV_BLAS};

    // SWAP
    //cuda2hipRename["cublasSswap"] = {"hipblasSswap", CONV_BLAS};
    //cuda2hipRename["cublasDswap"] = {"hipblasDswap", CONV_BLAS};
    //cuda2hipRename["cublasCswap"] = {"hipblasCswap", CONV_BLAS};
    //cuda2hipRename["cublasZswap"] = {"hipblasZswap", CONV_BLAS};

    // AMAX
    //cuda2hipRename["cublasIsamax"] = {"hipblasIsamax", CONV_BLAS};
    //cuda2hipRename["cublasIdamax"] = {"hipblasIdamax", CONV_BLAS};
    //cuda2hipRename["cublasIcamax"] = {"hipblasIcamax", CONV_BLAS};
    //cuda2hipRename["cublasIzamax"] = {"hipblasIzamax", CONV_BLAS};

    // AMIN
    //cuda2hipRename["cublasIsamin"] = {"hipblasIsamin", CONV_BLAS};
    //cuda2hipRename["cublasIdamin"] = {"hipblasIdamin", CONV_BLAS};
    //cuda2hipRename["cublasIcamin"] = {"hipblasIcamin", CONV_BLAS};
    //cuda2hipRename["cublasIzamin"] = {"hipblasIzamin", CONV_BLAS};

    // ASUM
    cuda2hipRename["cublasSasum"]        = {"hipblasSasum", CONV_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasSasumBatched"] = {"hipblasSasumBatched", CONV_BLAS};
    cuda2hipRename["cublasDasum"]        = {"hipblasDasum", CONV_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasDasumBatched"] = {"hipblasDasumBatched", CONV_BLAS};
    //cuda2hipRename["cublasScasum"] = {"hipblasScasum", CONV_BLAS};
    //cuda2hipRename["cublasDzasum"] = {"hipblasDzasum", CONV_BLAS};

    // ROT
    //cuda2hipRename["cublasSrot"]  = {"hipblasSrot", CONV_BLAS};
    //cuda2hipRename["cublasDrot"]  = {"hipblasDrot", CONV_BLAS};
    //cuda2hipRename["cublasCrot"]  = {"hipblasCrot", CONV_BLAS};
    //cuda2hipRename["cublasCsrot"] = {"hipblasCsrot", CONV_BLAS};
    //cuda2hipRename["cublasZrot"]  = {"hipblasZrot", CONV_BLAS};
    //cuda2hipRename["cublasZdrot"] = {"hipblasZdrot", CONV_BLAS};

    // ROTG
    //cuda2hipRename["cublasSrotg"] = {"hipblasSrotg", CONV_BLAS};
    //cuda2hipRename["cublasDrotg"] = {"hipblasDrotg", CONV_BLAS};
    //cuda2hipRename["cublasCrotg"] = {"hipblasCrotg", CONV_BLAS};
    //cuda2hipRename["cublasZrotg"] = {"hipblasZrotg", CONV_BLAS};

    // ROTM
    //cuda2hipRename["cublasSrotm"] = {"hipblasSrotm", CONV_BLAS};
    //cuda2hipRename["cublasDrotm"] = {"hipblasDrotm", CONV_BLAS};

    // ROTMG
    //cuda2hipRename["cublasSrotmg"] = {"hipblasSrotmg", CONV_BLAS};
    //cuda2hipRename["cublasDrotmg"] = {"hipblasDrotmg", CONV_BLAS};

    // GEMV
    cuda2hipRename["cublasSgemv"] = {"hipblasSgemv", CONV_BLAS};
    // there is no such a function in CUDA
    cuda2hipRename["cublasSgemvBatched"] = {"hipblasSgemvBatched", CONV_BLAS};
    //cuda2hipRename["cublasDgemv"] = {"hipblasDgemv", CONV_BLAS};
    //cuda2hipRename["cublasCgemv"] = {"hipblasCgemv", CONV_BLAS};
    //cuda2hipRename["cublasZgemv"] = {"hipblasZgemv", CONV_BLAS};

    // GBMV
    //cuda2hipRename["cublasSgbmv"] = {"hipblasSgbmv", CONV_BLAS};
    //cuda2hipRename["cublasDgbmv"] = {"hipblasDgbmv", CONV_BLAS};
    //cuda2hipRename["cublasCgbmv"] = {"hipblasCgbmv", CONV_BLAS};
    //cuda2hipRename["cublasZgbmv"] = {"hipblasZgbmv", CONV_BLAS};

    // TRMV
    //cuda2hipRename["cublasStrmv"] = {"hipblasStrmv", CONV_BLAS};
    //cuda2hipRename["cublasDtrmv"] = {"hipblasDtrmv", CONV_BLAS};
    //cuda2hipRename["cublasCtrmv"] = {"hipblasCtrmv", CONV_BLAS};
    //cuda2hipRename["cublasZtrmv"] = {"hipblasZtrmv", CONV_BLAS};

    // TBMV
    //cuda2hipRename["cublasStbmv"] = {"hipblasStbmv", CONV_BLAS};
    //cuda2hipRename["cublasDtbmv"] = {"hipblasDtbmv", CONV_BLAS};
    //cuda2hipRename["cublasCtbmv"] = {"hipblasCtbmv", CONV_BLAS};
    //cuda2hipRename["cublasZtbmv"] = {"hipblasZtbmv", CONV_BLAS};

    // TPMV
    //cuda2hipRename["cublasStpmv"] = {"hipblasStpmv", CONV_BLAS};
    //cuda2hipRename["cublasDtpmv"] = {"hipblasDtpmv", CONV_BLAS};
    //cuda2hipRename["cublasCtpmv"] = {"hipblasCtpmv", CONV_BLAS};
    //cuda2hipRename["cublasZtpmv"] = {"hipblasZtpmv", CONV_BLAS};

    // TRSV
    //cuda2hipRename["cublasStrsv"] = {"hipblasStrsv", CONV_BLAS};
    //cuda2hipRename["cublasDtrsv"] = {"hipblasDtrsv", CONV_BLAS};
    //cuda2hipRename["cublasCtrsv"] = {"hipblasCtrsv", CONV_BLAS};
    //cuda2hipRename["cublasZtrsv"] = {"hipblasZtrsv", CONV_BLAS};

    // TPSV
    //cuda2hipRename["cublasStpsv"] = {"hipblasStpsv", CONV_BLAS};
    //cuda2hipRename["cublasDtpsv"] = {"hipblasDtpsv", CONV_BLAS};
    //cuda2hipRename["cublasCtpsv"] = {"hipblasCtpsv", CONV_BLAS};
    //cuda2hipRename["cublasZtpsv"] = {"hipblasZtpsv", CONV_BLAS};

    // TBSV
    //cuda2hipRename["cublasStbsv"] = {"hipblasStbsv", CONV_BLAS};
    //cuda2hipRename["cublasDtbsv"] = {"hipblasDtbsv", CONV_BLAS};
    //cuda2hipRename["cublasCtbsv"] = {"hipblasCtbsv", CONV_BLAS};
    //cuda2hipRename["cublasZtbsv"] = {"hipblasZtbsv", CONV_BLAS};

    // SYMV/HEMV
    //cuda2hipRename["cublasSsymv"] = {"hipblasSsymv", CONV_BLAS};
    //cuda2hipRename["cublasDsymv"] = {"hipblasDsymv", CONV_BLAS};
    //cuda2hipRename["cublasCsymv"] = {"hipblasCsymv", CONV_BLAS};
    //cuda2hipRename["cublasZsymv"] = {"hipblasZsymv", CONV_BLAS};
    //cuda2hipRename["cublasChemv"] = {"hipblasChemv", CONV_BLAS};
    //cuda2hipRename["cublasZhemv"] = {"hipblasZhemv", CONV_BLAS};

    // SBMV/HBMV
    //cuda2hipRename["cublasSsbmv"] = {"hipblasSsbmv", CONV_BLAS};
    //cuda2hipRename["cublasDsbmv"] = {"hpiblasDsbmv", CONV_BLAS};
    //cuda2hipRename["cublasChbmv"] = {"hipblasChbmv", CONV_BLAS};
    //cuda2hipRename["cublasZhbmv"] = {"hipblasZhbmv", CONV_BLAS};

    // SPMV/HPMV
    //cuda2hipRename["cublasSspmv"] = {"hipblasSspmv", CONV_BLAS};
    //cuda2hipRename["cublasDspmv"] = {"hipblasDspmv", CONV_BLAS};
    //cuda2hipRename["cublasChpmv"] = {"hipblasChpmv", CONV_BLAS};
    //cuda2hipRename["cublasZhpmv"] = {"hipblasZhpmv", CONV_BLAS};

    // GER
    cuda2hipRename["cublasSger"] = {"hipblasSger", CONV_BLAS};
    //cuda2hipRename["cublasDger"]  = {"hipblasDger", CONV_BLAS};
    //cuda2hipRename["cublasCgeru"] = {"hipblasCgeru", CONV_BLAS};
    //cuda2hipRename["cublasCgerc"] = {"hipblasCgerc", CONV_BLAS};
    //cuda2hipRename["cublasZgeru"] = {"hipblasZgeru", CONV_BLAS};
    //cuda2hipRename["cublasZgerc"] = {"hipblasZgerc", CONV_BLAS};

    // SYR/HER
    //cuda2hipRename["cublasSsyr"] = {"hipblasSsyr", CONV_BLAS};
    //cuda2hipRename["cublasDsyr"] = {"hipblasDsyr", CONV_BLAS};
    //cuda2hipRename["cublasCher"] = {"hipblasCher", CONV_BLAS};
    //cuda2hipRename["cublasZher"] = {"hipblasZher", CONV_BLAS};

    // SPR/HPR
    //cuda2hipRename["cublasSspr"] = {"hipblasSspr", CONV_BLAS};
    //cuda2hipRename["cublasDspr"] = {"hipblasDspr", CONV_BLAS};
    //cuda2hipRename["cublasChpr"] = {"hipblasChpr", CONV_BLAS};
    //cuda2hipRename["cublasZhpr"] = {"hipblasZhpr", CONV_BLAS};

    // SYR2/HER2
    //cuda2hipRename["cublasSsyr2"] = {"hipblasSsyr2", CONV_BLAS};
    //cuda2hipRename["cublasDsyr2"] = {"hipblasDsyr2", CONV_BLAS};
    //cuda2hipRename["cublasCher2"] = {"hipblasCher2", CONV_BLAS};
    //cuda2hipRename["cublasZher2"] = {"hipblasZher2", CONV_BLAS};

    // SPR2/HPR2
    //cuda2hipRename["cublasSspr2"] = {"hipblasSspr2", CONV_BLAS};
    //cuda2hipRename["cublasDspr2"] = {"hipblasDspr2", CONV_BLAS};
    //cuda2hipRename["cublasChpr2"] = {"hipblasChpr2", CONV_BLAS};
    //cuda2hipRename["cublasZhpr2"] = {"hipblasZhpr2", CONV_BLAS};

    // Blas3 (v1) Routines
    // GEMM
    cuda2hipRename["cublasSgemm"] = {"hipblasSgemm", CONV_BLAS};
    //cuda2hipRename["cublasDgemm"] = {"hipblasDgemm", CONV_BLAS};
    cuda2hipRename["cublasCgemm"] = {"hipblasCgemm", CONV_BLAS};
    //cuda2hipRename["cublasZgemm"] = {"hipblasZgemm", CONV_BLAS};

    // BATCH GEMM
    cuda2hipRename["cublasSgemmBatched"] = {"hipblasSgemmBatched", CONV_BLAS};
    //cuda2hipRename["cublasDgemmBatched"] = {"hipblasDgemmBatched", CONV_BLAS};
    cuda2hipRename["cublasCgemmBatched"] = {"hipblasCgemmBatched", CONV_BLAS};
    //cuda2hipRename["cublasZgemmBatched"] = {"hipblasZgemmBatched", CONV_BLAS};

    // SYRK
    //cuda2hipRename["cublasSsyrk"] = {"hipblasSsyrk", CONV_BLAS};
    //cuda2hipRename["cublasDsyrk"] = {"hipblasDsyrk", CONV_BLAS};
    //cuda2hipRename["cublasCsyrk"] = {"hipblasCsyrk", CONV_BLAS};
    //cuda2hipRename["cublasZsyrk"] = {"hipblasZsyrk", CONV_BLAS};

    // HERK
    //cuda2hipRename["cublasCherk"] = {"hipblasCherk", CONV_BLAS};
    //cuda2hipRename["cublasZherk"] = {"hipblasZherk", CONV_BLAS};

    // SYR2K
    //cuda2hipRename["cublasSsyr2k"] = {"hipblasSsyr2k", CONV_BLAS};
    //cuda2hipRename["cublasDsyr2k"] = {"hipblasDsyr2k", CONV_BLAS};
    //cuda2hipRename["cublasCsyr2k"] = {"hipblasCsyr2k", CONV_BLAS};
    //cuda2hipRename["cublasZsyr2k"] = {"hipblasZsyr2k", CONV_BLAS};

    // SYRKX - eXtended SYRK
    // cublasSsyrkx
    // cublasDsyrkx
    // cublasCsyrkx
    // cublasZsyrkx

    // HER2K
    //cuda2hipRename["cublasCher2k"] = {"hipblasCher2k", CONV_BLAS};
    //cuda2hipRename["cublasZher2k"] = {"hipblasZher2k", CONV_BLAS};

    // HERKX - eXtended HERK
    // cublasCherkx
    // cublasZherkx

    // SYMM
    //cuda2hipRename["cublasSsymm"] = {"hipblasSsymm", CONV_BLAS};
    //cuda2hipRename["cublasDsymm"] = {"hipblasDsymm", CONV_BLAS};
    //cuda2hipRename["cublasCsymm"] = {"hipblasCsymm", CONV_BLAS};
    //cuda2hipRename["cublasZsymm"] = {"hipblasZsymm", CONV_BLAS};

    // HEMM
    //cuda2hipRename["cublasChemm"] = {"hipblasChemm", CONV_BLAS};
    //cuda2hipRename["cublasZhemm"] = {"hipblasZhemm", CONV_BLAS};

    // TRSM
    //cuda2hipRename["cublasStrsm"] = {"hipblasStrsm", CONV_BLAS};
    //cuda2hipRename["cublasDtrsm"] = {"hipblasDtrsm", CONV_BLAS};
    //cuda2hipRename["cublasCtrsm"] = {"hipblasCtrsm", CONV_BLAS};
    //cuda2hipRename["cublasZtrsm"] = {"hipblasZtrsm", CONV_BLAS};

    // TRSM - Batched Triangular Solver
    //cuda2hipRename["cublasStrsmBatched"] = {"hipblasStrsmBatched", CONV_BLAS};
    //cuda2hipRename["cublasDtrsmBatched"] = {"hipblasDtrsmBatched", CONV_BLAS};
    //cuda2hipRename["cublasCtrsmBatched"] = {"hipblasCtrsmBatched", CONV_BLAS};
    //cuda2hipRename["cublasZtrsmBatched"] = {"hipblasZtrsmBatched", CONV_BLAS};

    // TRMM
    //cuda2hipRename["cublasStrmm"] = {"hipblasStrmm", CONV_BLAS};
    //cuda2hipRename["cublasDtrmm"] = {"hipblasDtrmm", CONV_BLAS};
    //cuda2hipRename["cublasCtrmm"] = {"hipblasCtrmm", CONV_BLAS};
    //cuda2hipRename["cublasZtrmm"] = {"hipblasZtrmm", CONV_BLAS};


    // TO SUPPORT OR NOT? (cublas_api.h)
    // ------------------------ CUBLAS BLAS - like extension

    // GEAM
    // cublasSgeam
    // cublasDgeam
    // cublasCgeam
    // cublasZgeam

    // GETRF - Batched LU
    // cublasSgetrfBatched
    // cublasDgetrfBatched
    // cublasCgetrfBatched
    // cublasZgetrfBatched

    // Batched inversion based on LU factorization from getrf
    // cublasSgetriBatched
    // cublasDgetriBatched
    // cublasCgetriBatched
    // cublasZgetriBatched

    // Batched solver based on LU factorization from getrf
    // cublasSgetrsBatched
    // cublasDgetrsBatched
    // cublasCgetrsBatched
    // cublasZgetrsBatched

    // TRSM - Batched Triangular Solver
    // cublasStrsmBatched
    // cublasDtrsmBatched
    // cublasCtrsmBatched
    // cublasZtrsmBatched

    // MATINV - Batched
    // cublasSmatinvBatched
    // cublasDmatinvBatched
    // cublasCmatinvBatched
    // cublasZmatinvBatched

    // Batch QR Factorization
    // cublasSgeqrfBatched
    // cublasDgeqrfBatched
    // cublasCgeqrfBatched
    // cublasZgeqrfBatched

    // Least Square Min only m >= n and Non-transpose supported
    // cublasSgelsBatched
    // cublasDgelsBatched
    // cublasCgelsBatched
    // cublasZgelsBatched

    // DGMM
    // cublasSdgmm
    // cublasDdgmm
    // cublasCdgmm
    // cublasZdgmm

    // TPTTR - Triangular Pack format to Triangular format
    // cublasStpttr
    // cublasDtpttr
    // cublasCtpttr
    // cublasZtpttr

    // TRTTP - Triangular format to Triangular Pack format
    // cublasStrttp
    // cublasDtrttp
    // cublasCtrttp
    // cublasZtrttp

    // Blas2 (v2) Routines
    cuda2hipRename["cublasCreate_v2"] =  {"hipblasCreate", CONV_BLAS};
    cuda2hipRename["cublasDestroy_v2"] = {"hipblasDestroy", CONV_BLAS};

    // unsupported yet by hipblas/hcblas
    //cuda2hipRename["cublasGetVersion_v2"]     = {"hipblasGetVersion", CONV_BLAS};
    //cuda2hipRename["cublasSetStream_v2"]      = {"hipblasSetStream", CONV_BLAS};
    //cuda2hipRename["cublasGetStream_v2"]      = {"hipblasGetStream", CONV_BLAS};
    //cuda2hipRename["cublasGetPointerMode_v2"] = {"hipblasGetPointerMode", CONV_BLAS};
    //cuda2hipRename["cublasSetPointerMode_v2"] = {"hipblasSetPointerMode", CONV_BLAS};

    // GEMV
    cuda2hipRename["cublasSgemv_v2"] = {"hipblasSgemv", CONV_BLAS};
    //cuda2hipRename["cublasDgemv_v2"] = {"hipblasDgemv", CONV_BLAS};
    //cuda2hipRename["cublasCgemv_v2"] = {"hipblasCgemv", CONV_BLAS};
    //cuda2hipRename["cublasZgemv_v2"] = {"hipblasZgemv", CONV_BLAS};

    // GBMV
    //cuda2hipRename["cublasSgbmv_v2"] = {"hipblasSgbmv", CONV_BLAS};
    //cuda2hipRename["cublasDgbmv_v2"] = {"hipblasDgbmv", CONV_BLAS};
    //cuda2hipRename["cublasCgbmv_v2"] = {"hipblasCgbmv", CONV_BLAS};
    //cuda2hipRename["cublasZgbmv_v2"] = {"hipblasZgbmv", CONV_BLAS};

    // TRMV
    //cuda2hipRename["cublasStrmv_v2"] = {"hipblasStrmv", CONV_BLAS};
    //cuda2hipRename["cublasDtrmv_v2"] = {"hipblasDtrmv", CONV_BLAS};
    //cuda2hipRename["cublasCtrmv_v2"] = {"hipblasCtrmv", CONV_BLAS};
    //cuda2hipRename["cublasZtrmv_v2"] = {"hipblasZtrmv", CONV_BLAS};

    // TBMV
    //cuda2hipRename["cublasStbmv_v2"] = {"hipblasStbmv", CONV_BLAS};
    //cuda2hipRename["cublasDtbmv_v2"] = {"hipblasDtbmv", CONV_BLAS};
    //cuda2hipRename["cublasCtbmv_v2"] = {"hipblasCtbmv", CONV_BLAS};
    //cuda2hipRename["cublasZtbmv_v2"] = {"hipblasZtbmv", CONV_BLAS};

    // TPMV
    //cuda2hipRename["cublasStpmv_v2"] = {"hipblasStpmv", CONV_BLAS};
    //cuda2hipRename["cublasDtpmv_v2"] = {"hipblasDtpmv", CONV_BLAS};
    //cuda2hipRename["cublasCtpmv_v2"] = {"hipblasCtpmv", CONV_BLAS};
    //cuda2hipRename["cublasZtpmv_v2"] = {"hipblasZtpmv", CONV_BLAS};

    // TRSV
    //cuda2hipRename["cublasStrsv_v2"] = {"hipblasStrsv", CONV_BLAS};
    //cuda2hipRename["cublasDtrsv_v2"] = {"hipblasDtrsv", CONV_BLAS};
    //cuda2hipRename["cublasCtrsv_v2"] = {"hipblasCtrsv", CONV_BLAS};
    //cuda2hipRename["cublasZtrsv_v2"] = {"hipblasZtrsv", CONV_BLAS};

    // TPSV
    //cuda2hipRename["cublasStpsv_v2"] = {"hipblasStpsv", CONV_BLAS};
    //cuda2hipRename["cublasDtpsv_v2"] = {"hipblasDtpsv", CONV_BLAS};
    //cuda2hipRename["cublasCtpsv_v2"] = {"hipblasCtpsv", CONV_BLAS};
    //cuda2hipRename["cublasZtpsv_v2"] = {"hipblasZtpsv", CONV_BLAS};

    // TBSV
    //cuda2hipRename["cublasStbsv_v2"] = {"hipblasStbsv", CONV_BLAS};
    //cuda2hipRename["cublasDtbsv_v2"] = {"hipblasDtbsv", CONV_BLAS};
    //cuda2hipRename["cublasCtbsv_v2"] = {"hipblasCtbsv", CONV_BLAS};
    //cuda2hipRename["cublasZtbsv_v2"] = {"hipblasZtbsv", CONV_BLAS};

    // SYMV/HEMV
    //cuda2hipRename["cublasSsymv_v2"] = {"hipblasSsymv", CONV_BLAS};
    //cuda2hipRename["cublasDsymv_v2"] = {"hipblasDsymv", CONV_BLAS};
    //cuda2hipRename["cublasCsymv_v2"] = {"hipblasCsymv", CONV_BLAS};
    //cuda2hipRename["cublasZsymv_v2"] = {"hipblasZsymv", CONV_BLAS};
    //cuda2hipRename["cublasChemv_v2"] = {"hipblasChemv", CONV_BLAS};
    //cuda2hipRename["cublasZhemv_v2"] = {"hipblasZhemv", CONV_BLAS};

    // SBMV/HBMV
    //cuda2hipRename["cublasSsbmv_v2"] = {"hipblasSsbmv", CONV_BLAS};
    //cuda2hipRename["cublasDsbmv_v2"] = {"hpiblasDsbmv", CONV_BLAS};
    //cuda2hipRename["cublasChbmv_v2"] = {"hipblasChbmv", CONV_BLAS};
    //cuda2hipRename["cublasZhbmv_v2"] = {"hipblasZhbmv", CONV_BLAS};

    // SPMV/HPMV
    //cuda2hipRename["cublasSspmv_v2"] = {"hipblasSspmv", CONV_BLAS};
    //cuda2hipRename["cublasDspmv_v2"] = {"hipblasDspmv", CONV_BLAS};
    //cuda2hipRename["cublasChpmv_v2"] = {"hipblasChpmv", CONV_BLAS};
    //cuda2hipRename["cublasZhpmv_v2"] = {"hipblasZhpmv", CONV_BLAS};

    // GER
    cuda2hipRename["cublasSger_v2"]  = {"hipblasSger", CONV_BLAS};
    //cuda2hipRename["cublasDger_v2"]  = {"hipblasDger", CONV_BLAS};
    //cuda2hipRename["cublasCgeru_v2"] = {"hipblasCgeru", CONV_BLAS};
    //cuda2hipRename["cublasCgerc_v2"] = {"hipblasCgerc", CONV_BLAS};
    //cuda2hipRename["cublasZgeru_v2"] = {"hipblasZgeru", CONV_BLAS};
    //cuda2hipRename["cublasZgerc_v2"] = {"hipblasZgerc", CONV_BLAS};

    // SYR/HER
    //cuda2hipRename["cublasSsyr_v2"] = {"hipblasSsyr", CONV_BLAS};
    //cuda2hipRename["cublasDsyr_v2"] = {"hipblasDsyr", CONV_BLAS};
    //cuda2hipRename["cublasCsyr_v2"] = {"hipblasCsyr", CONV_BLAS};
    //cuda2hipRename["cublasZsyr_v2"] = {"hipblasZsyr", CONV_BLAS};
    //cuda2hipRename["cublasCher_v2"] = {"hipblasCher", CONV_BLAS};
    //cuda2hipRename["cublasZher_v2"] = {"hipblasZher", CONV_BLAS};

    // SPR/HPR
    //cuda2hipRename["cublasSspr_v2"] = {"hipblasSspr", CONV_BLAS};
    //cuda2hipRename["cublasDspr_v2"] = {"hipblasDspr", CONV_BLAS};
    //cuda2hipRename["cublasChpr_v2"] = {"hipblasChpr", CONV_BLAS};
    //cuda2hipRename["cublasZhpr_v2"] = {"hipblasZhpr", CONV_BLAS};

    // SYR2/HER2
    //cuda2hipRename["cublasSsyr2_v2"] = {"hipblasSsyr2", CONV_BLAS};
    //cuda2hipRename["cublasDsyr2_v2"] = {"hipblasDsyr2", CONV_BLAS};
    //cuda2hipRename["cublasCsyr2_v2"] = {"hipblasCsyr2", CONV_BLAS};
    //cuda2hipRename["cublasZsyr2_v2"] = {"hipblasZsyr2", CONV_BLAS};
    //cuda2hipRename["cublasCher2_v2"] = {"hipblasCher2", CONV_BLAS};
    //cuda2hipRename["cublasZher2_v2"] = {"hipblasZher2", CONV_BLAS};

    // SPR2/HPR2
    //cuda2hipRename["cublasSspr2_v2"] = {"hipblasSspr2", CONV_BLAS};
    //cuda2hipRename["cublasDspr2_v2"] = {"hipblasDspr2", CONV_BLAS};
    //cuda2hipRename["cublasChpr2_v2"] = {"hipblasChpr2", CONV_BLAS};
    //cuda2hipRename["cublasZhpr2_v2"] = {"hipblasZhpr2", CONV_BLAS};

    // Blas3 (v2) Routines
    // GEMM
    cuda2hipRename["cublasSgemm_v2"] = {"hipblasSgemm", CONV_BLAS};
    //cuda2hipRename["cublasDgemm_v2"] = {"hipblasDgemm", CONV_BLAS};
    cuda2hipRename["cublasCgemm_v2"] = {"hipblasCgemm", CONV_BLAS};
    //cuda2hipRename["cublasZgemm_v2"] = {"hipblasZgemm", CONV_BLAS};

    //IO in FP16 / FP32, computation in float
    // cublasSgemmEx

    // SYRK
    //cuda2hipRename["cublasSsyrk_v2"] = {"hipblasSsyrk", CONV_BLAS};
    //cuda2hipRename["cublasDsyrk_v2"] = {"hipblasDsyrk", CONV_BLAS};
    //cuda2hipRename["cublasCsyrk_v2"] = {"hipblasCsyrk", CONV_BLAS};
    //cuda2hipRename["cublasZsyrk_v2"] = {"hipblasZsyrk", CONV_BLAS};

    // HERK
    //cuda2hipRename["cublasCherk_v2"] = {"hipblasCherk", CONV_BLAS};
    //cuda2hipRename["cublasZherk_v2"] = {"hipblasZherk", CONV_BLAS};

    // SYR2K
    //cuda2hipRename["cublasSsyr2k_v2"] = {"hipblasSsyr2k", CONV_BLAS};
    //cuda2hipRename["cublasDsyr2k_v2"] = {"hipblasDsyr2k", CONV_BLAS};
    //cuda2hipRename["cublasCsyr2k_v2"] = {"hipblasCsyr2k", CONV_BLAS};
    //cuda2hipRename["cublasZsyr2k_v2"] = {"hipblasZsyr2k", CONV_BLAS};

    // HER2K
    //cuda2hipRename["cublasCher2k_v2"] = {"hipblasCher2k", CONV_BLAS};
    //cuda2hipRename["cublasZher2k_v2"] = {"hipblasZher2k", CONV_BLAS};

    // SYMM
    //cuda2hipRename["cublasSsymm_v2"] = {"hipblasSsymm", CONV_BLAS};
    //cuda2hipRename["cublasDsymm_v2"] = {"hipblasDsymm", CONV_BLAS};
    //cuda2hipRename["cublasCsymm_v2"] = {"hipblasCsymm", CONV_BLAS};
    //cuda2hipRename["cublasZsymm_v2"] = {"hipblasZsymm", CONV_BLAS};

    // HEMM
    //cuda2hipRename["cublasChemm_v2"] = {"hipblasChemm", CONV_BLAS};
    //cuda2hipRename["cublasZhemm_v2"] = {"hipblasZhemm", CONV_BLAS};

    // TRSM
    //cuda2hipRename["cublasStrsm_v2"] = {"hipblasStrsm", CONV_BLAS};
    //cuda2hipRename["cublasDtrsm_v2"] = {"hipblasDtrsm", CONV_BLAS};
    //cuda2hipRename["cublasCtrsm_v2"] = {"hipblasCtrsm", CONV_BLAS};
    //cuda2hipRename["cublasZtrsm_v2"] = {"hipblasZtrsm", CONV_BLAS};

    // TRMM
    //cuda2hipRename["cublasStrmm_v2"] = {"hipblasStrmm", CONV_BLAS};
    //cuda2hipRename["cublasDtrmm_v2"] = {"hipblasDtrmm", CONV_BLAS};
    //cuda2hipRename["cublasCtrmm_v2"] = {"hipblasCtrmm", CONV_BLAS};
    //cuda2hipRename["cublasZtrmm_v2"] = {"hipblasZtrmm", CONV_BLAS};

    // NRM2
    //cuda2hipRename["cublasSnrm2_v2"]  = {"hipblasSnrm2", CONV_BLAS};
    //cuda2hipRename["cublasDnrm2_v2"]  = {"hipblasDnrm2", CONV_BLAS};
    //cuda2hipRename["cublasScnrm2_v2"] = {"hipblasScnrm2", CONV_BLAS};
    //cuda2hipRename["cublasDznrm2_v2"] = {"hipblasDznrm2", CONV_BLAS};

    // DOT
    cuda2hipRename["cublasSdot_v2"]  = {"hipblasSdot", CONV_BLAS};
    cuda2hipRename["cublasDdot_v2"]  = {"hipblasDdot", CONV_BLAS};
    //cuda2hipRename["cublasCdotu_v2"] = {"hipblasCdotu", CONV_BLAS};
    //cuda2hipRename["cublasCdotc_v2"] = {"hipblasCdotc", CONV_BLAS};
    //cuda2hipRename["cublasZdotu_v2"] = {"hipblasZdotu", CONV_BLAS};
    //cuda2hipRename["cublasZdotc_v2"] = {"hipblasZdotc", CONV_BLAS};

    // SCAL
    cuda2hipRename["cublasSscal_v2"]  = {"hipblasSscal", CONV_BLAS};
    cuda2hipRename["cublasDscal_v2"]  = {"hipblasDscal", CONV_BLAS};
    //cuda2hipRename["cublasCscal_v2"]  = {"hipblasCscal", CONV_BLAS};
    //cuda2hipRename["cublasCsscal_v2"] = {"hipblasCsscal", CONV_BLAS};
    //cuda2hipRename["cublasZscal_v2"]  = {"hipblasZscal", CONV_BLAS};
    //cuda2hipRename["cublasZdscal_v2"] = {"hipblasZdscal", CONV_BLAS};

    // AXPY
    cuda2hipRename["cublasSaxpy_v2"] = {"hipblasSaxpy", CONV_BLAS};
    //cuda2hipRename["cublasDaxpy_v2"] = {"hipblasDaxpy", CONV_BLAS};
    //cuda2hipRename["cublasCaxpy_v2"] = {"hipblasCaxpy", CONV_BLAS};
    //cuda2hipRename["cublasZaxpy_v2"] = {"hipblasZaxpy", CONV_BLAS};

    // COPY
    cuda2hipRename["cublasScopy_v2"] = {"hipblasScopy", CONV_BLAS};
    cuda2hipRename["cublasDcopy_v2"] = {"hipblasDcopy", CONV_BLAS};
    //cuda2hipRename["cublasCcopy_v2"] = {"hipblasCcopy", CONV_BLAS};
    //cuda2hipRename["cublasZcopy_v2"] = {"hipblasZcopy", CONV_BLAS};

    // SWAP
    //cuda2hipRename["cublasSswap_v2"] = {"hipblasSswap", CONV_BLAS};
    //cuda2hipRename["cublasDswap_v2"] = {"hipblasDswap", CONV_BLAS};
    //cuda2hipRename["cublasCswap_v2"] = {"hipblasCswap", CONV_BLAS};
    //cuda2hipRename["cublasZswap_v2"] = {"hipblasZswap", CONV_BLAS};

    // AMAX
    //cuda2hipRename["cublasIsamax_v2"] = {"hipblasIsamax", CONV_BLAS};
    //cuda2hipRename["cublasIdamax_v2"] = {"hipblasIdamax", CONV_BLAS};
    //cuda2hipRename["cublasIcamax_v2"] = {"hipblasIcamax", CONV_BLAS};
    //cuda2hipRename["cublasIzamax_v2"] = {"hipblasIzamax", CONV_BLAS};

    // AMIN
    //cuda2hipRename["cublasIsamin_v2"] = {"hipblasIsamin", CONV_BLAS};
    //cuda2hipRename["cublasIdamin_v2"] = {"hipblasIdamin", CONV_BLAS};
    //cuda2hipRename["cublasIcamin_v2"] = {"hipblasIcamin", CONV_BLAS};
    //cuda2hipRename["cublasIzamin_v2"] = {"hipblasIzamin", CONV_BLAS};

    // ASUM
    cuda2hipRename["cublasSasum_v2"]  = {"hipblasSasum", CONV_BLAS};
    cuda2hipRename["cublasDasum_v2"]  = {"hipblasDasum", CONV_BLAS};
    //cuda2hipRename["cublasScasum_v2"] = {"hipblasScasum", CONV_BLAS};
    //cuda2hipRename["cublasDzasum_v2"] = {"hipblasDzasum", CONV_BLAS};

    // ROT
    //cuda2hipRename["cublasSrot_v2"]  = {"hipblasSrot", CONV_BLAS};
    //cuda2hipRename["cublasDrot_v2"]  = {"hipblasDrot", CONV_BLAS};
    //cuda2hipRename["cublasCrot_v2"]  = {"hipblasCrot", CONV_BLAS};
    //cuda2hipRename["cublasCsrot_v2"] = {"hipblasCsrot", CONV_BLAS};
    //cuda2hipRename["cublasZrot_v2"]  = {"hipblasZrot", CONV_BLAS};
    //cuda2hipRename["cublasZdrot_v2"] = {"hipblasZdrot", CONV_BLAS};

    // ROTG
    //cuda2hipRename["cublasSrotg_v2"] = {"hipblasSrotg", CONV_BLAS};
    //cuda2hipRename["cublasDrotg_v2"] = {"hipblasDrotg", CONV_BLAS};
    //cuda2hipRename["cublasCrotg_v2"] = {"hipblasCrotg", CONV_BLAS};
    //cuda2hipRename["cublasZrotg_v2"] = {"hipblasZrotg", CONV_BLAS};

    // ROTM
    //cuda2hipRename["cublasSrotm_v2"] = {"hipblasSrotm", CONV_BLAS};
    //cuda2hipRename["cublasDrotm_v2"] = {"hipblasDrotm", CONV_BLAS};

    // ROTMG
    //cuda2hipRename["cublasSrotmg_v2"] = {"hipblasSrotmg", CONV_BLAS};
    //cuda2hipRename["cublasDrotmg_v2"] = {"hipblasDrotmg", CONV_BLAS};
}

  struct HipNames {
    StringRef hipName;
    ConvTypes countType;
  };

  SmallDenseMap<StringRef, HipNames> cuda2hipRename;
};

StringRef unquoteStr(StringRef s) {
  if (s.size() > 1 && s.front() == '"' && s.back() == '"')
    return s.substr(1, s.size() - 2);
  return s;
}

static void processString(StringRef s, const cuda2hipMap &map,
                          Replacements *Replace, SourceManager &SM,
                          SourceLocation start,
                          int64_t countReps[ConvTypes::CONV_LAST]) {
  size_t begin = 0;
  while ((begin = s.find("cuda", begin)) != StringRef::npos ||
         (begin = s.find("cublas", begin)) != StringRef::npos) {
    const size_t end = s.find_first_of(" ", begin + 4);
    StringRef name = s.slice(begin, end);
    const auto found = map.cuda2hipRename.find(name);
    if (found != map.cuda2hipRename.end()) {
      countReps[CONV_LITERAL]++;
      StringRef repName = found->second.hipName;
      SourceLocation sl = start.getLocWithOffset(begin + 1);
      Replacement Rep(SM, sl, name.size(), repName);
      Replace->insert(Rep);
    }
    if (end == StringRef::npos)
      break;
    begin = end + 1;
  }
}

struct HipifyPPCallbacks : public PPCallbacks, public SourceFileCallbacks {
  HipifyPPCallbacks(Replacements *R)
      : SeenEnd(false), _sm(nullptr), _pp(nullptr), Replace(R) {}

  virtual bool handleBeginSource(CompilerInstance &CI,
                                 StringRef Filename) override {
    Preprocessor &PP = CI.getPreprocessor();
    SourceManager &SM = CI.getSourceManager();
    setSourceManager(&SM);
    PP.addPPCallbacks(std::unique_ptr<HipifyPPCallbacks>(this));
    PP.Retain();
    setPreprocessor(&PP);
    return true;
  }

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
          countReps[found->second.countType]++;
          StringRef repName = found->second.hipName;
          DEBUG(dbgs() << "Include file found: " << file_name << "\n"
                       << "SourceLocation:"
                       << filename_range.getBegin().printToString(*_sm) << "\n"
                       << "Will be replaced with " << repName << "\n");
          SourceLocation sl = filename_range.getBegin();
          SourceLocation sle = filename_range.getEnd();
          const char *B = _sm->getCharacterData(sl);
          const char *E = _sm->getCharacterData(sle);
          SmallString<128> tmpData;
          Replacement Rep(*_sm, sl, E - B,
                          Twine("<" + repName + ">").toStringRef(tmpData));
          Replace->insert(Rep);
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
            countReps[found->second.countType]++;
            StringRef repName = found->second.hipName;
            SourceLocation sl = T.getLocation();
            DEBUG(dbgs() << "Identifier " << name
                         << " found in definition of macro "
                         << MacroNameTok.getIdentifierInfo()->getName() << "\n"
                         << "will be replaced with: " << repName << "\n"
                         << "SourceLocation: " << sl.printToString(*_sm)
                         << "\n");
            Replacement Rep(*_sm, sl, name.size(), repName);
            Replace->insert(Rep);
          }
        }
      }
    }
  }

  virtual void MacroExpands(const Token &MacroNameTok,
                            const MacroDefinition &MD, SourceRange Range,
                            const MacroArgs *Args) override {
    if (_sm->isWrittenInMainFile(MacroNameTok.getLocation())) {
      for (unsigned int i = 0; Args && i < MD.getMacroInfo()->getNumArgs();
           i++) {
        StringRef macroName = MacroNameTok.getIdentifierInfo()->getName();
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
              countReps[found->second.countType]++;
              StringRef repName = found->second.hipName;
              DEBUG(dbgs()
                    << "Identifier " << name
                    << " found as an actual argument in expansion of macro "
                    << macroName << "\n"
                    << "will be replaced with: " << repName << "\n");
              SourceLocation sl = tok.getLocation();
              Replacement Rep(*_sm, sl, name.size(), repName);
              Replace->insert(Rep);
            }
          }
          if (tok.is(tok::string_literal)) {
            StringRef s(tok.getLiteralData(), tok.getLength());
            processString(unquoteStr(s), N, Replace, *_sm, tok.getLocation(),
                          countReps);
          }
        }
      }
    }
  }

  void EndOfMainFile() override {}

  bool SeenEnd;
  void setSourceManager(SourceManager *sm) { _sm = sm; }
  void setPreprocessor(Preprocessor *pp) { _pp = pp; }

  int64_t countReps[ConvTypes::CONV_LAST] = {0};

private:
  SourceManager *_sm;
  Preprocessor *_pp;

  Replacements *Replace;
  struct cuda2hipMap N;
};

class Cuda2HipCallback : public MatchFinder::MatchCallback {
public:
  Cuda2HipCallback(Replacements *Replace, ast_matchers::MatchFinder *parent)
      : Replace(Replace), owner(parent) {}

  void convertKernelDecl(const FunctionDecl *kernelDecl,
                         const MatchFinder::MatchResult &Result) {
    SourceManager *SM = Result.SourceManager;
    LangOptions DefaultLangOptions;

    SmallString<40> XStr;
    raw_svector_ostream OS(XStr);
    StringRef initialParamList;
    OS << "hipLaunchParm lp";
    size_t replacementLength = OS.str().size();
    SourceLocation sl = kernelDecl->getNameInfo().getEndLoc();
    SourceLocation kernelArgListStart = Lexer::findLocationAfterToken(
        sl, tok::l_paren, *SM, DefaultLangOptions, true);
    DEBUG(dbgs() << kernelArgListStart.printToString(*SM));
    if (kernelDecl->getNumParams() > 0) {
      const ParmVarDecl *pvdFirst = kernelDecl->getParamDecl(0);
      const ParmVarDecl *pvdLast =
          kernelDecl->getParamDecl(kernelDecl->getNumParams() - 1);
      SourceLocation kernelArgListStart(pvdFirst->getLocStart());
      SourceLocation kernelArgListEnd(pvdLast->getLocEnd());
      SourceLocation stop = Lexer::getLocForEndOfToken(
          kernelArgListEnd, 0, *SM, DefaultLangOptions);
      replacementLength +=
          SM->getCharacterData(stop) - SM->getCharacterData(kernelArgListStart);
      initialParamList = StringRef(SM->getCharacterData(kernelArgListStart),
                                   replacementLength);
      OS << ", " << initialParamList;
    }
    DEBUG(dbgs() << "initial paramlist: " << initialParamList << "\n"
                 << "new paramlist: " << OS.str() << "\n");
    Replacement Rep0(*(Result.SourceManager), kernelArgListStart,
                     replacementLength, OS.str());
    Replace->insert(Rep0);
  }

  void run(const MatchFinder::MatchResult &Result) override {
    SourceManager *SM = Result.SourceManager;
    LangOptions DefaultLangOptions;

    if (const CallExpr *call =
            Result.Nodes.getNodeAs<CallExpr>("cudaCall")) {
      const FunctionDecl *funcDcl = call->getDirectCallee();
      StringRef name = funcDcl->getDeclName().getAsString();
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        countReps[found->second.countType]++;
        StringRef repName = found->second.hipName;
        SourceLocation sl = call->getLocStart();
        size_t length = name.size();
        if (SM->isMacroArgExpansion(sl)) {
          sl = SM->getImmediateSpellingLoc(sl);
        }
        else if (SM->isMacroBodyExpansion(sl)) {
          sl = SM->getExpansionLoc(sl);
          SourceLocation sl_end =
            Lexer::getLocForEndOfToken(sl, 0, *SM, DefaultLangOptions);
          length = SM->getCharacterData(sl_end) - SM->getCharacterData(sl);
        }
        Replacement Rep(*SM, sl, length, repName);
        Replace->insert(Rep);
      }
    }

    if (const CUDAKernelCallExpr *launchKernel =
            Result.Nodes.getNodeAs<CUDAKernelCallExpr>("cudaLaunchKernel")) {
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
                                .bind("unresolvedTemplateName"),
                            this);
        }
      }

      XStr.clear();
      OS << "hipLaunchKernel(HIP_KERNEL_NAME(" << calleeName << "),";

      const CallExpr *config = launchKernel->getConfig();
      DEBUG(dbgs() << "Kernel config arguments:"
                   << "\n");
      for (unsigned argno = 0; argno < config->getNumArgs(); argno++) {
        const Expr *arg = config->getArg(argno);
        if (!isa<CXXDefaultArgExpr>(arg)) {
          const ParmVarDecl *pvd =
              config->getDirectCallee()->getParamDecl(argno);

          SourceLocation sl(arg->getLocStart());
          SourceLocation el(arg->getLocEnd());
          SourceLocation stop =
              Lexer::getLocForEndOfToken(el, 0, *SM, DefaultLangOptions);
          StringRef outs(SM->getCharacterData(sl),
                         SM->getCharacterData(stop) - SM->getCharacterData(sl));
          DEBUG(dbgs() << "args[ " << argno << "]" << outs << " <"
                       << pvd->getType().getAsString() << ">"
                       << "\n");
          if (pvd->getType().getAsString().compare("dim3") == 0)
            OS << " dim3(" << outs << "),";
          else
            OS << " " << outs << ",";
        } else
          OS << " 0,";
      }

      for (unsigned argno = 0; argno < launchKernel->getNumArgs(); argno++) {
        const Expr *arg = launchKernel->getArg(argno);
        SourceLocation sl(arg->getLocStart());
        SourceLocation el(arg->getLocEnd());
        SourceLocation stop =
            Lexer::getLocForEndOfToken(el, 0, *SM, DefaultLangOptions);
        std::string outs(SM->getCharacterData(sl),
                         SM->getCharacterData(stop) - SM->getCharacterData(sl));
        DEBUG(dbgs() << outs << "\n");
        OS << " " << outs << ",";
      }
      XStr.pop_back();
      OS << ")";
      size_t length =
          SM->getCharacterData(Lexer::getLocForEndOfToken(
              launchKernel->getLocEnd(), 0, *SM, DefaultLangOptions)) -
          SM->getCharacterData(launchKernel->getLocStart());
      Replacement Rep(*SM, launchKernel->getLocStart(), length, OS.str());
      Replace->insert(Rep);
      countReps[ConvTypes::CONV_KERN]++;
    }

    if (const FunctionTemplateDecl *templateDecl =
            Result.Nodes.getNodeAs<FunctionTemplateDecl>(
                "unresolvedTemplateName")) {
      FunctionDecl *kernelDecl = templateDecl->getTemplatedDecl();
      convertKernelDecl(kernelDecl, Result);
    }

    if (const MemberExpr *threadIdx =
            Result.Nodes.getNodeAs<MemberExpr>("cudaBuiltin")) {
      if (const OpaqueValueExpr *refBase =
              dyn_cast<OpaqueValueExpr>(threadIdx->getBase())) {
        if (const DeclRefExpr *declRef =
                dyn_cast<DeclRefExpr>(refBase->getSourceExpr())) {
          StringRef name = declRef->getDecl()->getName();
          StringRef memberName = threadIdx->getMemberDecl()->getName();
          size_t pos = memberName.find_first_not_of("__fetch_builtin_");
          memberName = memberName.slice(pos, memberName.size());
          SmallString<128> tmpData;
          name = Twine(name + "." + memberName).toStringRef(tmpData);
          const auto found = N.cuda2hipRename.find(name);
          if (found != N.cuda2hipRename.end()) {
            countReps[found->second.countType]++;
            StringRef repName = found->second.hipName;
            SourceLocation sl = threadIdx->getLocStart();
            Replacement Rep(*SM, sl, name.size(), repName);
            Replace->insert(Rep);
          }
        }
      }
    }

    if (const DeclRefExpr *cudaEnumConstantRef =
            Result.Nodes.getNodeAs<DeclRefExpr>("cudaEnumConstantRef")) {
      StringRef name = cudaEnumConstantRef->getDecl()->getNameAsString();
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        countReps[found->second.countType]++;
        StringRef repName = found->second.hipName;
        SourceLocation sl = cudaEnumConstantRef->getLocStart();
        Replacement Rep(*SM, sl, name.size(), repName);
        Replace->insert(Rep);
      }
    }

    if (const VarDecl *cudaEnumConstantDecl =
            Result.Nodes.getNodeAs<VarDecl>("cudaEnumConstantDecl")) {
      StringRef name =
          cudaEnumConstantDecl->getType()->getAsTagDecl()->getNameAsString();
      // anonymous typedef enum
      if (name.empty()) {
        QualType QT = cudaEnumConstantDecl->getType().getUnqualifiedType();
        name = QT.getAsString();
      }
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        countReps[found->second.countType]++;
        StringRef repName = found->second.hipName;
        SourceLocation sl = cudaEnumConstantDecl->getLocStart();
        Replacement Rep(*SM, sl, name.size(), repName);
        Replace->insert(Rep);
      }
    }

    if (const VarDecl *cudaTypedefVar =
      Result.Nodes.getNodeAs<VarDecl>("cudaTypedefVar")) {
      QualType QT = cudaTypedefVar->getType().getUnqualifiedType();
      StringRef name = QT.getAsString();
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        countReps[found->second.countType]++;
        StringRef repName = found->second.hipName;
        SourceLocation sl = cudaTypedefVar->getLocStart();
        Replacement Rep(*SM, sl, name.size(), repName);
        Replace->insert(Rep);
      }
    }

    if (const VarDecl *cudaStructVar =
            Result.Nodes.getNodeAs<VarDecl>("cudaStructVar")) {
      StringRef name = cudaStructVar->getType()
                           ->getAsStructureType()
                           ->getDecl()
                           ->getNameAsString();
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        countReps[found->second.countType]++;
        StringRef repName = found->second.hipName;
        TypeLoc TL = cudaStructVar->getTypeSourceInfo()->getTypeLoc();
        SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
        Replacement Rep(*SM, sl, name.size(), repName);
        Replace->insert(Rep);
      }
    }

    if (const VarDecl *cudaStructVarPtr =
            Result.Nodes.getNodeAs<VarDecl>("cudaStructVarPtr")) {
      const Type *t = cudaStructVarPtr->getType().getTypePtrOrNull();
      if (t) {
        StringRef name = t->getPointeeCXXRecordDecl()->getName();
        const auto found = N.cuda2hipRename.find(name);
        if (found != N.cuda2hipRename.end()) {
          countReps[found->second.countType]++;
          StringRef repName = found->second.hipName;
          TypeLoc TL = cudaStructVarPtr->getTypeSourceInfo()->getTypeLoc();
          SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
          Replacement Rep(*SM, sl, name.size(), repName);
          Replace->insert(Rep);
        }
      }
    }

    if (const ParmVarDecl *cudaParamDecl =
            Result.Nodes.getNodeAs<ParmVarDecl>("cudaParamDecl")) {
      QualType QT = cudaParamDecl->getOriginalType().getUnqualifiedType();
      StringRef name = QT.getAsString();
      const Type *t = QT.getTypePtr();
      if (t->isStructureOrClassType()) {
        name = t->getAsCXXRecordDecl()->getName();
      }
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        countReps[found->second.countType]++;
        StringRef repName = found->second.hipName;
        TypeLoc TL = cudaParamDecl->getTypeSourceInfo()->getTypeLoc();
        SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
        Replacement Rep(*SM, sl, name.size(), repName);
        Replace->insert(Rep);
      }
    }

    if (const ParmVarDecl *cudaParamDeclPtr =
            Result.Nodes.getNodeAs<ParmVarDecl>("cudaParamDeclPtr")) {
      const Type *pt = cudaParamDeclPtr->getType().getTypePtrOrNull();
      if (pt) {
        QualType QT = pt->getPointeeType();
        const Type *t = QT.getTypePtr();
        StringRef name = t->isStructureOrClassType()
                             ? t->getAsCXXRecordDecl()->getName()
                             : StringRef(QT.getAsString());
        const auto found = N.cuda2hipRename.find(name);
        if (found != N.cuda2hipRename.end()) {
          countReps[found->second.countType]++;
          StringRef repName = found->second.hipName;
          TypeLoc TL = cudaParamDeclPtr->getTypeSourceInfo()->getTypeLoc();
          SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
          Replacement Rep(*SM, sl, name.size(), repName);
          Replace->insert(Rep);
        }
      }
    }

    if (const StringLiteral *stringLiteral =
            Result.Nodes.getNodeAs<StringLiteral>("stringLiteral")) {
      if (stringLiteral->getCharByteWidth() == 1) {
        StringRef s = stringLiteral->getString();
        processString(s, N, Replace, *SM, stringLiteral->getLocStart(),
                      countReps);
      }
    }

    if (const UnaryExprOrTypeTraitExpr *expr =
            Result.Nodes.getNodeAs<UnaryExprOrTypeTraitExpr>(
                "cudaStructSizeOf")) {
      TypeSourceInfo *typeInfo = expr->getArgumentTypeInfo();
      QualType QT = typeInfo->getType().getUnqualifiedType();
      const Type *type = QT.getTypePtr();
      StringRef name = type->getAsCXXRecordDecl()->getName();
      const auto found = N.cuda2hipRename.find(name);
      if (found != N.cuda2hipRename.end()) {
        countReps[found->second.countType]++;
        StringRef repName = found->second.hipName;
        TypeLoc TL = typeInfo->getTypeLoc();
        SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
        Replacement Rep(*SM, sl, name.size(), repName);
        Replace->insert(Rep);
      }
    }
  }

  int64_t countReps[ConvTypes::CONV_LAST] = {0};

private:
  Replacements *Replace;
  ast_matchers::MatchFinder *owner;
  struct cuda2hipMap N;
};

} // end anonymous namespace

// Set up the command line options
static cl::OptionCategory ToolTemplateCategory("CUDA to HIP source translator options");

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
       cl::value_desc("filename"), cl::cat(ToolTemplateCategory));

static cl::opt<bool>
    Inplace("inplace",
            cl::desc("Modify input file inplace, replacing input with hipified "
                     "output, save backup in .prehip file. "),
            cl::value_desc("inplace"));

static cl::opt<bool>
    NoOutput("no-output",
             cl::desc("don't write any translated output to stdout"),
             cl::value_desc("no-output"));

static cl::opt<bool>
    PrintStats("print-stats", cl::desc("print the command-line, like a header"),
               cl::value_desc("print-stats"));

int main(int argc, const char **argv) {

  llvm::sys::PrintStackTraceOnErrorSignal();

  int Result;

  CommonOptionsParser OptionsParser(argc, argv, ToolTemplateCategory, llvm::cl::Required);

  std::vector<std::string> fileSources = OptionsParser.getSourcePathList();

  std::string dst = OutputFilename;
  if (dst.empty()) {
    dst = fileSources[0];
    if (!Inplace) {
      size_t pos = dst.rfind(".cu");
      if (pos != std::string::npos) {
        dst = dst.substr(0, pos) + ".hip.cu";
      } else {
        llvm::errs() << "Input .cu file was not specified.\n";
        return 1;
      }
    }
  } else {
    if (Inplace) {
      llvm::errs() << "Conflict: both -o and -inplace options are specified.";
    }
    dst += ".cu";
  }

  // copy source file since tooling makes changes "inplace"
  std::ifstream source(fileSources[0], std::ios::binary);
  std::ofstream dest(Inplace ? dst + ".prehip" : dst, std::ios::binary);
  dest << source.rdbuf();
  source.close();
  dest.close();

  RefactoringTool Tool(OptionsParser.getCompilations(), dst);
  ast_matchers::MatchFinder Finder;
  Cuda2HipCallback Callback(&Tool.getReplacements(), &Finder);
  HipifyPPCallbacks PPCallbacks(&Tool.getReplacements());
  Finder.addMatcher(callExpr(isExpansionInMainFile(),
                             callee(functionDecl(matchesName("cuda.*|cublas.*"))))
                             .bind("cudaCall"),
                             &Callback);
  Finder.addMatcher(cudaKernelCallExpr().bind("cudaLaunchKernel"), &Callback);
  Finder.addMatcher(memberExpr(isExpansionInMainFile(),
                               hasObjectExpression(hasType(cxxRecordDecl(
                               matchesName("__cuda_builtin_")))))
                               .bind("cudaBuiltin"),
                               &Callback);
  Finder.addMatcher(declRefExpr(isExpansionInMainFile(),
                                to(enumConstantDecl(
                                matchesName("cuda.*|cublas.*|CUDA.*|CUBLAS*"))))
                                .bind("cudaEnumConstantRef"),
                                &Callback);
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(enumDecl()))
                            .bind("cudaEnumConstantDecl"),
                            &Callback);
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(typedefDecl(matchesName("cuda.*|cublas.*"))))
                            .bind("cudaTypedefVar"),
                            &Callback);
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(cxxRecordDecl(matchesName("cuda.*|cublas.*"))))
                            .bind("cudaStructVar"),
                            &Callback);
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(pointsTo(cxxRecordDecl(
                                             matchesName("cuda.*|cublas.*")))))
                            .bind("cudaStructVarPtr"),
                            &Callback);
  Finder.addMatcher(parmVarDecl(isExpansionInMainFile(),
                                hasType(namedDecl(matchesName("cuda.*|cublas.*"))))
                                .bind("cudaParamDecl"),
                                &Callback);
  Finder.addMatcher(parmVarDecl(isExpansionInMainFile(),
                                hasType(pointsTo(namedDecl(
                                                 matchesName("cuda.*|cublas.*")))))
                                .bind("cudaParamDeclPtr"),
                                &Callback);
  Finder.addMatcher(expr(isExpansionInMainFile(),
                         sizeOfExpr(hasArgumentOfType(recordType(hasDeclaration(
                             cxxRecordDecl(matchesName("cuda.*|cublas.*")))))))
                        .bind("cudaStructSizeOf"),
                         &Callback);
  Finder.addMatcher(stringLiteral(isExpansionInMainFile()).bind("stringLiteral"),
                                  &Callback);

  auto action = newFrontendActionFactory(&Finder, &PPCallbacks);

  std::vector<const char *> compilationStages;
  compilationStages.push_back("--cuda-host-only");

  for (auto Stage : compilationStages) {
    Tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster(Stage, ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-std=c++11"));
#if defined(HIPIFY_CLANG_RES)
    Tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster("-resource-dir=" HIPIFY_CLANG_RES));
#endif
    Tool.appendArgumentsAdjuster(getClangSyntaxOnlyAdjuster());
    Result = Tool.run(action.get());

    Tool.clearArgumentsAdjusters();
  }

  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);
  SourceManager Sources(Diagnostics, Tool.getFiles());

  DEBUG(dbgs() << "Replacements collected by the tool:\n");
  for (const auto &r : Tool.getReplacements()) {
    DEBUG(dbgs() << r.toString() << "\n");
  }

  Rewriter Rewrite(Sources, DefaultLangOptions);

  if (!Tool.applyAllReplacements(Rewrite)) {
    DEBUG(dbgs() << "Skipped some replacements.\n");
  }

  Result = Rewrite.overwriteChangedFiles();

  if (!Inplace) {
    size_t pos = dst.rfind(".cu");
    if (pos != std::string::npos) {
      rename(dst.c_str(), dst.substr(0, pos).c_str());
    }
  }
  if (PrintStats) {
    int64_t sum = 0;
    for (int i = 0; i < ConvTypes::CONV_LAST; i++) {
      sum += Callback.countReps[i] + PPCallbacks.countReps[i];
    }
    llvm::outs() << "info: converted " << sum << " CUDA->HIP refs ( ";
    for (int i = 0; i < ConvTypes::CONV_LAST; i++) {
      llvm::outs() << counterNames[i] << ':'
                   << Callback.countReps[i] + PPCallbacks.countReps[i] << ' ';
    }
    llvm::outs() << ") in \'" << fileSources[0] << "\'\n";
  }
  return Result;
}
