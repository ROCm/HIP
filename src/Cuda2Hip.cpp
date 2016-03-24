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

namespace {

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
  CONV_INC,
  CONV_LAST
} ;

struct cuda2hipMap {
  cuda2hipMap() {
    // defines
    cuda2hipRename["__CUDACC__"] = {"__HIPCC__", CONV_DEF};

    // includes
    cuda2hipRename["cuda_runtime.h"] = {"hip_runtime.h", CONV_INC};
    cuda2hipRename["cuda_runtime_api.h"] = {"hip_runtime_api.h", CONV_INC};

    // Error codes and return types:
    cuda2hipRename["cudaError_t"] = {"hipError_t", CONV_ERR};
    cuda2hipRename["cudaError"] = {"hipError", CONV_ERR};
    cuda2hipRename["cudaSuccess"] = {"hipSuccess", CONV_ERR};

    cuda2hipRename["cudaErrorUnknown"] = {"hipErrorUnknown", CONV_ERR};
    cuda2hipRename["cudaErrorMemoryAllocation"] = {"hipErrorMemoryAllocation", CONV_ERR};
    cuda2hipRename["cudaErrorMemoryFree"] = {"hipErrorMemoryFree", CONV_ERR};
    cuda2hipRename["cudaErrorUnknownSymbol"] = {"hipErrorUnknownSymbol", CONV_ERR};
    cuda2hipRename["cudaErrorOutOfResources"] = {"hipErrorOutOfResources", CONV_ERR};
    cuda2hipRename["cudaErrorInvalidValue"] = {"hipErrorInvalidValue", CONV_ERR};
    cuda2hipRename["cudaErrorInvalidResourceHandle"] =
        {"hipErrorInvalidResourceHandle", CONV_ERR};
    cuda2hipRename["cudaErrorInvalidDevice"] = {"hipErrorInvalidDevice", CONV_ERR};
    cuda2hipRename["cudaErrorNoDevice"] = {"hipErrorNoDevice", CONV_ERR};
    cuda2hipRename["cudaErrorNotReady"] = {"hipErrorNotReady", CONV_ERR};
    cuda2hipRename["cudaErrorUnknown"] = {"hipErrorUnknown", CONV_ERR};

    // error APIs:
    cuda2hipRename["cudaGetLastError"] = {"hipGetLastError", CONV_ERR};
    cuda2hipRename["cudaPeekAtLastError"] = {"hipPeekAtLastError", CONV_ERR};
    cuda2hipRename["cudaGetErrorName"] = {"hipGetErrorName", CONV_ERR};
    cuda2hipRename["cudaGetErrorString"] = {"hipGetErrorString", CONV_ERR};

    // Memcpy
    cuda2hipRename["cudaMemcpy"] = {"hipMemcpy", CONV_MEM};
    cuda2hipRename["cudaMemcpyHostToHost"] = {"hipMemcpyHostToHost", CONV_MEM};
    cuda2hipRename["cudaMemcpyHostToDevice"] = {"hipMemcpyHostToDevice", CONV_MEM};
    cuda2hipRename["cudaMemcpyDeviceToHost"] = {"hipMemcpyDeviceToHost", CONV_MEM};
    cuda2hipRename["cudaMemcpyDeviceToDevice"] = {"hipMemcpyDeviceToDevice", CONV_MEM};
    cuda2hipRename["cudaMemcpyDefault"] = {"hipMemcpyDefault", CONV_MEM};
    cuda2hipRename["cudaMemcpyToSymbol"] = {"hipMemcpyToSymbol", CONV_MEM};
    cuda2hipRename["cudaMemset"] = {"hipMemset", CONV_MEM};
    cuda2hipRename["cudaMemsetAsync"] = {"hipMemsetAsync", CONV_MEM};
    cuda2hipRename["cudaMemcpyAsync"] = {"hipMemcpyAsync", CONV_MEM};
    cuda2hipRename["cudaMemGetInfo"] = {"hipMemGetInfo", CONV_MEM};
    cuda2hipRename["cudaMemcpyKind"] = {"hipMemcpyKind", CONV_MEM};

    // Memory management :
    cuda2hipRename["cudaMalloc"] = {"hipMalloc", CONV_MEM};
    cuda2hipRename["cudaMallocHost"] = {"hipHostAlloc", CONV_MEM};
    cuda2hipRename["cudaFree"] = {"hipFree", CONV_MEM};
    cuda2hipRename["cudaFreeHost"] = {"hipHostFree", CONV_MEM};

    // Coordinate Indexing and Dimensions:
    cuda2hipRename["threadIdx.x"] = {"hipThreadIdx_x", CONV_COORD_FUNC};
    cuda2hipRename["threadIdx.y"] = {"hipThreadIdx_y", CONV_COORD_FUNC};
    cuda2hipRename["threadIdx.z"] = {"hipThreadIdx_z", CONV_COORD_FUNC};

    cuda2hipRename["blockIdx.x"] = {"hipBlockIdx_x", CONV_COORD_FUNC};
    cuda2hipRename["blockIdx.y"] = {"hipBlockIdx_y", CONV_COORD_FUNC};
    cuda2hipRename["blockIdx.z"] = {"hipBlockIdx_z", CONV_COORD_FUNC};

    cuda2hipRename["blockDim.x"] = {"hipBlockDim_x", CONV_COORD_FUNC};
    cuda2hipRename["blockDim.y"] = {"hipBlockDim_y", CONV_COORD_FUNC};
    cuda2hipRename["blockDim.z"] = {"hipBlockDim_z", CONV_COORD_FUNC};

    cuda2hipRename["gridDim.x"] = {"hipGridDim_x", CONV_COORD_FUNC};
    cuda2hipRename["gridDim.y"] = {"hipGridDim_y", CONV_COORD_FUNC};
    cuda2hipRename["gridDim.z"] = {"hipGridDim_z", CONV_COORD_FUNC};

    cuda2hipRename["blockIdx.x"] = {"hipBlockIdx_x", CONV_COORD_FUNC};
    cuda2hipRename["blockIdx.y"] = {"hipBlockIdx_y", CONV_COORD_FUNC};
    cuda2hipRename["blockIdx.z"] = {"hipBlockIdx_z", CONV_COORD_FUNC};

    cuda2hipRename["blockDim.x"] = {"hipBlockDim_x", CONV_COORD_FUNC};
    cuda2hipRename["blockDim.y"] = {"hipBlockDim_y", CONV_COORD_FUNC};
    cuda2hipRename["blockDim.z"] = {"hipBlockDim_z", CONV_COORD_FUNC};

    cuda2hipRename["gridDim.x"] = {"hipGridDim_x", CONV_COORD_FUNC};
    cuda2hipRename["gridDim.y"] = {"hipGridDim_y", CONV_COORD_FUNC};
    cuda2hipRename["gridDim.z"] = {"hipGridDim_z", CONV_COORD_FUNC};

    cuda2hipRename["warpSize"] = {"hipWarpSize", CONV_SPECIAL_FUNC};

    // Events
    cuda2hipRename["cudaEvent_t"] = {"hipEvent_t", CONV_EVENT};
    cuda2hipRename["cudaEventCreate"] = {"hipEventCreate", CONV_EVENT};
    cuda2hipRename["cudaEventCreateWithFlags"] = {"hipEventCreateWithFlags", CONV_EVENT};
    cuda2hipRename["cudaEventDestroy"] = {"hipEventDestroy", CONV_EVENT};
    cuda2hipRename["cudaEventRecord"] = {"hipEventRecord", CONV_EVENT};
    cuda2hipRename["cudaEventElapsedTime"] = {"hipEventElapsedTime", CONV_EVENT};
    cuda2hipRename["cudaEventSynchronize"] = {"hipEventSynchronize", CONV_EVENT};

    // Streams
    cuda2hipRename["cudaStream_t"] = {"hipStream_t", CONV_STREAM};
    cuda2hipRename["cudaStreamCreate"] = {"hipStreamCreate", CONV_STREAM};
    cuda2hipRename["cudaStreamCreateWithFlags"] = {"hipStreamCreateWithFlags", CONV_STREAM};
    cuda2hipRename["cudaStreamDestroy"] = {"hipStreamDestroy", CONV_STREAM};
    cuda2hipRename["cudaStreamWaitEvent"] = {"hipStreamWaitEven", CONV_STREAM};
    cuda2hipRename["cudaStreamSynchronize"] = {"hipStreamSynchronize", CONV_STREAM};
    cuda2hipRename["cudaStreamDefault"] = {"hipStreamDefault", CONV_STREAM};
    cuda2hipRename["cudaStreamNonBlocking"] = {"hipStreamNonBlocking", CONV_STREAM};

    // Other synchronization
    cuda2hipRename["cudaDeviceSynchronize"] = {"hipDeviceSynchronize", CONV_DEV};
    cuda2hipRename["cudaThreadSynchronize"] =
        {"hipDeviceSynchronize", CONV_DEV}; // translate deprecated cudaThreadSynchronize
    cuda2hipRename["cudaDeviceReset"] = {"hipDeviceReset", CONV_DEV};
    cuda2hipRename["cudaThreadExit"] =
        {"hipDeviceReset", CONV_DEV}; // translate deprecated cudaThreadExit
    cuda2hipRename["cudaSetDevice"] = {"hipSetDevice", CONV_DEV};
    cuda2hipRename["cudaGetDevice"] = {"hipGetDevice", CONV_DEV};

    // Attribute
    cuda2hipRename["bcudaDeviceAttr"] = {"hipDeviceAttribute_t", CONV_DEV};
    cuda2hipRename["bcudaDeviceGetAttribute"] = {"hipDeviceGetAttribute", CONV_DEV};
    
    // Device
    cuda2hipRename["cudaDeviceProp"] = {"hipDeviceProp_t", CONV_DEV};
    cuda2hipRename["cudaGetDeviceProperties"] = {"hipDeviceGetProperties", CONV_DEV};

    // Cache config
    cuda2hipRename["cudaDeviceSetCacheConfig"] = {"hipDeviceSetCacheConfig", CONV_DEV};
    cuda2hipRename["cudaThreadSetCacheConfig"] =
        {"hipDeviceSetCacheConfig", CONV_DEV}; // translate deprecated
    cuda2hipRename["cudaDeviceGetCacheConfig"] = {"hipDeviceGetCacheConfig", CONV_DEV};
    cuda2hipRename["cudaThreadGetCacheConfig"] =
        {"hipDeviceGetCacheConfig", CONV_DEV}; // translate deprecated
    cuda2hipRename["cudaFuncCache"] = {"hipFuncCache", CONV_DEV};
    cuda2hipRename["cudaFuncCachePreferNone"] = {"hipFuncCachePreferNone", CONV_DEV};
    cuda2hipRename["cudaFuncCachePreferShared"] = {"hipFuncCachePreferShared", CONV_DEV};
    cuda2hipRename["cudaFuncCachePreferL1"] = {"hipFuncCachePreferL1", CONV_DEV};
    cuda2hipRename["cudaFuncCachePreferEqual"] = {"hipFuncCachePreferEqual", CONV_DEV};
    // function
    cuda2hipRename["cudaFuncSetCacheConfig"] = {"hipFuncSetCacheConfig", CONV_DEV};

    cuda2hipRename["cudaDriverGetVersion"] = {"hipDriverGetVersion", CONV_DEV};
//    cuda2hipRename["cudaRuntimeGetVersion"] = {"hipRuntimeGetVersion", CONV_DEV};

    // Peer2Peer
    cuda2hipRename["cudaDeviceCanAccessPeer"] = {"hipDeviceCanAccessPeer", CONV_DEV};
    cuda2hipRename["cudaDeviceDisablePeerAccess"] =
        {"hipDeviceDisablePeerAccess", CONV_DEV};
    cuda2hipRename["cudaDeviceEnablePeerAccess"] = {"hipDeviceEnablePeerAccess", CONV_DEV};
    cuda2hipRename["cudaMemcpyPeerAsync"] = {"hipMemcpyPeerAsync", CONV_MEM};
    cuda2hipRename["cudaMemcpyPeer"] = {"hipMemcpyPeer", CONV_MEM};

    // Shared mem:
    cuda2hipRename["cudaDeviceSetSharedMemConfig"] =
        {"hipDeviceSetSharedMemConfig", CONV_DEV};
    cuda2hipRename["cudaThreadSetSharedMemConfig"] =
        {"hipDeviceSetSharedMemConfig", CONV_DEV}; // translate deprecated
    cuda2hipRename["cudaDeviceGetSharedMemConfig"] =
        {"hipDeviceGetSharedMemConfig", CONV_DEV};
    cuda2hipRename["cudaThreadGetSharedMemConfig"] =
        {"hipDeviceGetSharedMemConfig", CONV_DEV}; // translate deprecated
    cuda2hipRename["cudaSharedMemConfig"] = {"hipSharedMemConfig", CONV_DEV};
    cuda2hipRename["cudaSharedMemBankSizeDefault"] =
        {"hipSharedMemBankSizeDefault", CONV_DEV};
    cuda2hipRename["cudaSharedMemBankSizeFourByte"] =
        {"hipSharedMemBankSizeFourByte", CONV_DEV};
    cuda2hipRename["cudaSharedMemBankSizeEightByte"] =
        {"hipSharedMemBankSizeEightByte", CONV_DEV};

    cuda2hipRename["cudaGetDeviceCount"] = {"hipGetDeviceCount", CONV_DEV};

    // Profiler
    // cuda2hipRename["cudaProfilerInitialize"] = "hipProfilerInitialize";  //
    // see if these are called anywhere.
    cuda2hipRename["cudaProfilerStart"] = {"hipProfilerStart", CONV_OTHER};
    cuda2hipRename["cudaProfilerStop"] = {"hipProfilerStop", CONV_OTHER};

    cuda2hipRename["cudaChannelFormatDesc"] = {"hipChannelFormatDesc", CONV_TEX};
    cuda2hipRename["cudaFilterModePoint"] = {"hipFilterModePoint", CONV_TEX};
    cuda2hipRename["cudaReadModeElementType"] = {"hipReadModeElementType", CONV_TEX};

    cuda2hipRename["cudaCreateChannelDesc"] = {"hipCreateChannelDesc", CONV_TEX};
    cuda2hipRename["cudaBindTexture"] = {"hipBindTexture", CONV_TEX};
    cuda2hipRename["cudaUnbindTexture"] = {"hipUnbindTexture", CONV_TEX};
  }
  
  struct HipNames {
    StringRef hipName;
    ConvTypes countType;
  };
  
  SmallDenseMap<StringRef, HipNames, 128> cuda2hipRename;
};

StringRef unquoteStr(StringRef s) {
  if (s.size() > 1 && s.front() == '"' && s.back() == '"')
    return s.substr(1, s.size() - 2);
  return s;
}

static void processString(StringRef s, struct cuda2hipMap &map, Replacements *Replace,
                   SourceManager &SM, SourceLocation start) {
  size_t begin = 0;
  while ((begin = s.find("cuda", begin)) != StringRef::npos) {
    const size_t end = s.find_first_of(" ", begin + 4);
    StringRef name = s.slice(begin, end);
    StringRef repName = map.cuda2hipRename[name].hipName;
    if (!repName.empty()) {
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
        if (N.cuda2hipRename.count(file_name)) {
          StringRef repName = N.cuda2hipRename[file_name].hipName;
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
          if (N.cuda2hipRename.count(name)) {
            StringRef repName = N.cuda2hipRename[name].hipName;
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
            if (N.cuda2hipRename.count(name)) {
              StringRef repName = N.cuda2hipRename[name].hipName;
              DEBUG(dbgs() << "Identifier " << name
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
            processString(unquoteStr(s), N, Replace, *_sm, tok.getLocation());
          }
        }
      }
    }
  }

  void EndOfMainFile() override {}

  bool SeenEnd;
  void setSourceManager(SourceManager *sm) { _sm = sm; }
  void setPreprocessor(Preprocessor *pp) { _pp = pp; }

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
    SourceLocation kernelArgListStart = clang::Lexer::findLocationAfterToken(
        sl, clang::tok::l_paren, *SM, DefaultLangOptions, true);
    DEBUG(dbgs() << kernelArgListStart.printToString(*SM));
    if (kernelDecl->getNumParams() > 0) {
      const ParmVarDecl *pvdFirst = kernelDecl->getParamDecl(0);
      const ParmVarDecl *pvdLast =
          kernelDecl->getParamDecl(kernelDecl->getNumParams() - 1);
      SourceLocation kernelArgListStart(pvdFirst->getLocStart());
      SourceLocation kernelArgListEnd(pvdLast->getLocEnd());
      SourceLocation stop = clang::Lexer::getLocForEndOfToken(
          kernelArgListEnd, 0, *SM, DefaultLangOptions);
      size_t replacementLength =
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
            Result.Nodes.getNodeAs<clang::CallExpr>("cudaCall")) {
      const FunctionDecl *funcDcl = call->getDirectCallee();
      StringRef name = funcDcl->getDeclName().getAsString();
      if (N.cuda2hipRename.count(name)) {
        StringRef repName = N.cuda2hipRename[name].hipName;
        SourceLocation sl = call->getLocStart();
        Replacement Rep(*SM, SM->isMacroArgExpansion(sl)
                                 ? SM->getImmediateSpellingLoc(sl)
                                 : sl,
                        name.size(), repName);
        Replace->insert(Rep);
      }
    }

    if (const CUDAKernelCallExpr *launchKernel =
            Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>(
                "cudaLaunchKernel")) {
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
      DEBUG(dbgs() << "Kernel config arguments:" << "\n");
      for (unsigned argno = 0; argno < config->getNumArgs(); argno++) {
        const Expr *arg = config->getArg(argno);
        if (!isa<CXXDefaultArgExpr>(arg)) {
          const ParmVarDecl *pvd =
              config->getDirectCallee()->getParamDecl(argno);

          SourceLocation sl(arg->getLocStart());
          SourceLocation el(arg->getLocEnd());
          SourceLocation stop =
              clang::Lexer::getLocForEndOfToken(el, 0, *SM, DefaultLangOptions);
          StringRef outs(SM->getCharacterData(sl),
                         SM->getCharacterData(stop) - SM->getCharacterData(sl));
          DEBUG(dbgs() << "args[ " << argno << "]" << outs << " <"
                       << pvd->getType().getAsString() << ">" << "\n");
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
            clang::Lexer::getLocForEndOfToken(el, 0, *SM, DefaultLangOptions);
        std::string outs(SM->getCharacterData(sl),
                         SM->getCharacterData(stop) - SM->getCharacterData(sl));
        DEBUG(dbgs() << outs << "\n");
        OS << " " << outs << ",";
      }
      XStr.pop_back();
      OS << ")";
      size_t length =
          SM->getCharacterData(clang::Lexer::getLocForEndOfToken(
              launchKernel->getLocEnd(), 0, *SM, DefaultLangOptions)) -
          SM->getCharacterData(launchKernel->getLocStart());
      Replacement Rep(*SM, launchKernel->getLocStart(), length, OS.str());
      Replace->insert(Rep);
    }

    if (const FunctionTemplateDecl *templateDecl =
            Result.Nodes.getNodeAs<clang::FunctionTemplateDecl>(
                "unresolvedTemplateName")) {
      FunctionDecl *kernelDecl = templateDecl->getTemplatedDecl();
      convertKernelDecl(kernelDecl, Result);
    }

    if (const MemberExpr *threadIdx =
            Result.Nodes.getNodeAs<clang::MemberExpr>("cudaBuiltin")) {
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
          StringRef repName = N.cuda2hipRename[name].hipName;
          SourceLocation sl = threadIdx->getLocStart();
          Replacement Rep(*SM, sl, name.size(), repName);
          Replace->insert(Rep);
        }
      }
    }

    if (const DeclRefExpr *cudaEnumConstantRef =
            Result.Nodes.getNodeAs<clang::DeclRefExpr>("cudaEnumConstantRef")) {
      StringRef name = cudaEnumConstantRef->getDecl()->getNameAsString();
      StringRef repName = N.cuda2hipRename[name].hipName;
      SourceLocation sl = cudaEnumConstantRef->getLocStart();
      Replacement Rep(*SM, sl, name.size(), repName);
      Replace->insert(Rep);
    }

    if (const VarDecl *cudaEnumConstantDecl =
            Result.Nodes.getNodeAs<clang::VarDecl>("cudaEnumConstantDecl")) {
      StringRef name =
          cudaEnumConstantDecl->getType()->getAsTagDecl()->getNameAsString();
      StringRef repName = N.cuda2hipRename[name].hipName;
      SourceLocation sl = cudaEnumConstantDecl->getLocStart();
      Replacement Rep(*SM, sl, name.size(), repName);
      Replace->insert(Rep);
    }

    if (const VarDecl *cudaStructVar =
            Result.Nodes.getNodeAs<clang::VarDecl>("cudaStructVar")) {
      StringRef name = cudaStructVar->getType()
                           ->getAsStructureType()
                           ->getDecl()
                           ->getNameAsString();
      StringRef repName = N.cuda2hipRename[name].hipName;
      TypeLoc TL = cudaStructVar->getTypeSourceInfo()->getTypeLoc();
      SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
      Replacement Rep(*SM, sl, name.size(), repName);
      Replace->insert(Rep);
    }

    if (const VarDecl *cudaStructVarPtr =
            Result.Nodes.getNodeAs<clang::VarDecl>("cudaStructVarPtr")) {
      const Type *t = cudaStructVarPtr->getType().getTypePtrOrNull();
      if (t) {
        StringRef name = t->getPointeeCXXRecordDecl()->getName();
        StringRef repName = N.cuda2hipRename[name].hipName;
        TypeLoc TL = cudaStructVarPtr->getTypeSourceInfo()->getTypeLoc();
        SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
        Replacement Rep(*SM, sl, name.size(), repName);
        Replace->insert(Rep);
      }
    }

    if (const ParmVarDecl *cudaParamDecl =
            Result.Nodes.getNodeAs<clang::ParmVarDecl>("cudaParamDecl")) {
      QualType QT = cudaParamDecl->getOriginalType().getUnqualifiedType();
      StringRef name = QT.getAsString();
      const Type *t = QT.getTypePtr();
      if (t->isStructureOrClassType()) {
        name = t->getAsCXXRecordDecl()->getName();
      }
      StringRef repName = N.cuda2hipRename[name].hipName;
      TypeLoc TL = cudaParamDecl->getTypeSourceInfo()->getTypeLoc();
      SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
      Replacement Rep(*SM, sl, name.size(), repName);
      Replace->insert(Rep);
    }

    if (const ParmVarDecl *cudaParamDeclPtr =
            Result.Nodes.getNodeAs<clang::ParmVarDecl>("cudaParamDeclPtr")) {
      const Type *pt = cudaParamDeclPtr->getType().getTypePtrOrNull();
      if (pt) {
        QualType QT = pt->getPointeeType();
        const Type *t = QT.getTypePtr();
        StringRef name = t->isStructureOrClassType()
                             ? t->getAsCXXRecordDecl()->getName()
                             : StringRef(QT.getAsString());
        StringRef repName = N.cuda2hipRename[name].hipName;
        TypeLoc TL = cudaParamDeclPtr->getTypeSourceInfo()->getTypeLoc();
        SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
        Replacement Rep(*SM, sl, name.size(), repName);
        Replace->insert(Rep);
      }
    }

    if (const StringLiteral *stringLiteral =
            Result.Nodes.getNodeAs<clang::StringLiteral>("stringLiteral")) {
      if (stringLiteral->getCharByteWidth() == 1) {
        StringRef s = stringLiteral->getString();
        processString(s, N, Replace, *SM, stringLiteral->getLocStart());
      }
    }

    if (const UnaryExprOrTypeTraitExpr *expr =
            Result.Nodes.getNodeAs<clang::UnaryExprOrTypeTraitExpr>(
                "cudaStructSizeOf")) {
      TypeSourceInfo *typeInfo = expr->getArgumentTypeInfo();
      QualType QT = typeInfo->getType().getUnqualifiedType();
      const Type *type = QT.getTypePtr();
      StringRef name = type->getAsCXXRecordDecl()->getName();
      StringRef repName = N.cuda2hipRename[name].hipName;
      TypeLoc TL = typeInfo->getTypeLoc();
      SourceLocation sl = TL.getUnqualifiedLoc().getLocStart();
      Replacement Rep(*SM, sl, name.size(), repName);
      Replace->insert(Rep);
    }
  }

private:
  Replacements *Replace;
  ast_matchers::MatchFinder *owner;
  struct cuda2hipMap N;
};

} // end anonymous namespace

// Set up the command line options
static cl::OptionCategory
    ToolTemplateCategory("CUDA to HIP source translator options");
static cl::extrahelp MoreHelp("<source0> specify the path of source file\n\n");

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::cat(ToolTemplateCategory));

static cl::opt<bool>
    Inplace("inplace",
            cl::desc("Modify input file inplace, replacing input with hipified "
                     "output, save backup in .prehip file. "),
            cl::value_desc("inplace"), cl::cat(ToolTemplateCategory));

static cl::opt<bool>
    NoOutput("no-output",
            cl::desc("don't write any translated output to stdout"),
            cl::value_desc("no-output"), cl::cat(ToolTemplateCategory));
static cl::opt<bool>
    PrintStats("print-stats",
            cl::desc("print the command-line, like a header"),
            cl::value_desc("print-stats"), cl::cat(ToolTemplateCategory));
 
int main(int argc, const char **argv) {

  llvm::sys::PrintStackTraceOnErrorSignal();

  int Result;

  CommonOptionsParser OptionsParser(argc, argv, ToolTemplateCategory,
                                    llvm::cl::Required);
  std::string dst = OutputFilename;
  std::vector<std::string> fileSources = OptionsParser.getSourcePathList();
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
                             callee(functionDecl(matchesName("cuda.*"))))
                        .bind("cudaCall"),
                    &Callback);
  Finder.addMatcher(cudaKernelCallExpr().bind("cudaLaunchKernel"), &Callback);
  Finder.addMatcher(memberExpr(isExpansionInMainFile(),
                               hasObjectExpression(hasType(cxxRecordDecl(
                                   matchesName("__cuda_builtin_")))))
                        .bind("cudaBuiltin"),
                    &Callback);
  Finder.addMatcher(declRefExpr(isExpansionInMainFile(),
                                to(enumConstantDecl(matchesName("cuda.*"))))
                        .bind("cudaEnumConstantRef"),
                    &Callback);
  Finder.addMatcher(
      varDecl(isExpansionInMainFile(), hasType(enumDecl(matchesName("cuda.*"))))
          .bind("cudaEnumConstantDecl"),
      &Callback);
  Finder.addMatcher(varDecl(isExpansionInMainFile(),
                            hasType(cxxRecordDecl(matchesName("cuda.*"))))
                        .bind("cudaStructVar"),
                    &Callback);
  Finder.addMatcher(
      varDecl(isExpansionInMainFile(),
              hasType(pointsTo(cxxRecordDecl(matchesName("cuda.*")))))
          .bind("cudaStructVarPtr"),
      &Callback);
  Finder.addMatcher(parmVarDecl(isExpansionInMainFile(),
                                hasType(namedDecl(matchesName("cuda.*"))))
                        .bind("cudaParamDecl"),
                    &Callback);
  Finder.addMatcher(
      parmVarDecl(isExpansionInMainFile(),
                  hasType(pointsTo(namedDecl(matchesName("cuda.*")))))
          .bind("cudaParamDeclPtr"),
      &Callback);
  Finder.addMatcher(expr(isExpansionInMainFile(),
                         sizeOfExpr(hasArgumentOfType(recordType(hasDeclaration(
                             cxxRecordDecl(matchesName("cuda.*")))))))
                        .bind("cudaStructSizeOf"),
                    &Callback);
  Finder.addMatcher(
      stringLiteral(isExpansionInMainFile()).bind("stringLiteral"), &Callback);

  auto action = newFrontendActionFactory(&Finder, &PPCallbacks);

  std::vector<const char *> compilationStages;
  compilationStages.push_back("--cuda-host-only");
  compilationStages.push_back("--cuda-device-only");

  for (auto Stage : compilationStages) {
    Tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster(Stage, ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-std=c++11"));
#if defined(HIPIFY_CLANG_RES)
    Tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster("-resource-dir=" HIPIFY_CLANG_RES));
#endif // defined(HIPIFY_CLANG_HEADERS)
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
  return Result;
}
