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

#include <algorithm>
#include <set>
#include "HipifyAction.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/HeaderSearch.h"
#include "LLVMCompat.h"
#include "CUDA2HIP.h"
#include "StringUtils.h"
#include "ArgParse.h"

const std::string sHIP = "HIP";
const std::string sROC = "ROC";
const std::string sCub = "cub";
const std::string sHipcub = "hipcub";
const std::string sHIP_DYNAMIC_SHARED = "HIP_DYNAMIC_SHARED";
const std::string sHIP_KERNEL_NAME = "HIP_KERNEL_NAME";
std::string sHIP_SYMBOL = "HIP_SYMBOL";
std::string s_reinterpret_cast = "reinterpret_cast<const void*>";
const std::string sHipLaunchKernelGGL = "hipLaunchKernelGGL";
const std::string sDim3 = "dim3(";
const std::string s_hiprand_kernel_h = "hiprand_kernel.h";
const std::string s_hiprand_h = "hiprand.h";
const std::string sOnce = "once";
const std::string s_string_literal = "[string literal]";
// CUDA identifiers, used in matchers
const std::string sCudaMemcpyToSymbol = "cudaMemcpyToSymbol";
const std::string sCudaMemcpyToSymbolAsync = "cudaMemcpyToSymbolAsync";
const std::string sCudaGetSymbolSize = "cudaGetSymbolSize";
const std::string sCudaGetSymbolAddress = "cudaGetSymbolAddress";
const std::string sCudaMemcpyFromSymbol = "cudaMemcpyFromSymbol";
const std::string sCudaMemcpyFromSymbolAsync = "cudaMemcpyFromSymbolAsync";
const std::string sCudaFuncSetCacheConfig = "cudaFuncSetCacheConfig";
const std::string sCudaFuncGetAttributes = "cudaFuncGetAttributes";
// Matchers' names
const StringRef sCudaSharedIncompleteArrayVar = "cudaSharedIncompleteArrayVar";
const StringRef sCudaLaunchKernel = "cudaLaunchKernel";
const StringRef sCudaHostFuncCall = "cudaHostFuncCall";
const StringRef sCudaDeviceFuncCall = "cudaDeviceFuncCall";
const StringRef sCubNamespacePrefix = "cubNamespacePrefix";
const StringRef sCubFunctionTemplateDecl = "cubFunctionTemplateDecl";
const StringRef sCubUsingNamespaceDecl = "cubUsingNamespaceDecl";

std::set<std::string> DeviceSymbolFunctions0 {
  {sCudaMemcpyToSymbol},
  {sCudaMemcpyToSymbolAsync}
};

std::set<std::string> DeviceSymbolFunctions1 {
  {sCudaGetSymbolSize},
  {sCudaGetSymbolAddress},
  {sCudaMemcpyFromSymbol},
  {sCudaMemcpyFromSymbolAsync}
};

std::set<std::string> ReinterpretFunctions{
  {sCudaFuncSetCacheConfig},
  {sCudaFuncGetAttributes}
};

std::set<std::string> ReinterpretFunctions0{
  {sCudaFuncSetCacheConfig}
};

std::set<std::string> ReinterpretFunctions1{
  {sCudaFuncGetAttributes}
};

void HipifyAction::RewriteString(StringRef s, clang::SourceLocation start) {
  auto &SM = getCompilerInstance().getSourceManager();
  size_t begin = 0;
  while ((begin = s.find("cu", begin)) != StringRef::npos) {
    const size_t end = s.find_first_of(" ", begin + 4);
    StringRef name = s.slice(begin, end);
    const auto found = CUDA_RENAMES_MAP().find(name);
    if (found != CUDA_RENAMES_MAP().end()) {
      StringRef repName = Statistics::isToRoc(found->second) ? found->second.rocName : found->second.hipName;
      hipCounter counter = {s_string_literal, "", ConvTypes::CONV_LITERAL, ApiTypes::API_RUNTIME, found->second.supportDegree};
      Statistics::current().incrementCounter(counter, name.str());
      if (!Statistics::isUnsupported(counter)) {
        clang::SourceLocation sl = start.getLocWithOffset(begin + 1);
        ct::Replacement Rep(SM, sl, name.size(), repName.str());
        clang::FullSourceLoc fullSL(sl, SM);
        insertReplacement(Rep, fullSL);
      }
    }
    if (end == StringRef::npos) break;
    begin = end + 1;
  }
}

clang::SourceLocation HipifyAction::GetSubstrLocation(const std::string &str, const clang::SourceRange &sr) {
  clang::SourceLocation sl(sr.getBegin());
  clang::SourceLocation end(sr.getEnd());
  auto &SM = getCompilerInstance().getSourceManager();
  size_t length = SM.getCharacterData(end) - SM.getCharacterData(sl);
  StringRef sfull = StringRef(SM.getCharacterData(sl), length);
  size_t offset = sfull.find(str);
  if (offset > 0) {
    sl = sl.getLocWithOffset(offset);
  }
  return sl;
}

/**
  * Look at, and consider altering, a given token.
  *
  * If it's not a CUDA identifier, nothing happens.
  * If it's an unsupported CUDA identifier, a warning is emitted.
  * Otherwise, the source file is updated with the corresponding hipification.
  */
void HipifyAction::RewriteToken(const clang::Token &t) {
  // String literals containing CUDA references need fixing.
  if (t.is(clang::tok::string_literal)) {
    StringRef s(t.getLiteralData(), t.getLength());
    RewriteString(unquoteStr(s), t.getLocation());
    return;
  } else if (!t.isAnyIdentifier()) {
    // If it's neither a string nor an identifier, we don't care.
    return;
  }
  StringRef name = t.getRawIdentifier();
  clang::SourceLocation sl = t.getLocation();
  FindAndReplace(name, sl, CUDA_RENAMES_MAP());
}

void HipifyAction::FindAndReplace(StringRef name,
                                  clang::SourceLocation sl,
                                  const std::map<StringRef, hipCounter> &repMap,
                                  bool bReplace) {
  const auto found = repMap.find(name);
  if (found == repMap.end()) {
    // So it's an identifier, but not CUDA? Boring.
    return;
  }
  Statistics::current().incrementCounter(found->second, name.str());
  clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
  // Warn the user about unsupported identifier.
  if (Statistics::isUnsupported(found->second)) {
    std::string sWarn;
    Statistics::isToRoc(found->second) ? sWarn = sROC : sWarn = sHIP;
    sWarn = "" + sWarn;
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "CUDA identifier is unsupported in %0.");
    DE.Report(sl, ID) << sWarn;
    return;
  }
  if (!bReplace) {
    return;
  }
  StringRef repName = Statistics::isToRoc(found->second) ? found->second.rocName : found->second.hipName;
  auto &SM = getCompilerInstance().getSourceManager();
  ct::Replacement Rep(SM, sl, name.size(), repName.str());
  clang::FullSourceLoc fullSL(sl, SM);
  insertReplacement(Rep, fullSL);
}

namespace {

clang::SourceRange getReadRange(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  clang::SourceLocation begin = exprRange.getBegin();
  clang::SourceLocation end = exprRange.getEnd();
  bool beginSafe = !SM.isMacroBodyExpansion(begin) || clang::Lexer::isAtStartOfMacroExpansion(begin, SM, clang::LangOptions{});
  bool endSafe = !SM.isMacroBodyExpansion(end) || clang::Lexer::isAtEndOfMacroExpansion(end, SM, clang::LangOptions{});
  if (beginSafe && endSafe) {
    return {SM.getFileLoc(begin), SM.getFileLoc(end)};
  } else {
    return {SM.getSpellingLoc(begin), SM.getSpellingLoc(end)};
  }
}

clang::SourceRange getWriteRange(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  clang::SourceLocation begin = exprRange.getBegin();
  clang::SourceLocation end = exprRange.getEnd();
  // If the range is contained within a macro, update the macro definition.
  // Otherwise, use the file location and hope for the best.
  if (!SM.isMacroBodyExpansion(begin) || !SM.isMacroBodyExpansion(end)) {
    return {SM.getExpansionLoc(begin), SM.getExpansionLoc(end)};
  }
  return {SM.getSpellingLoc(begin), SM.getSpellingLoc(end)};
}

StringRef readSourceText(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  return clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(getReadRange(SM, exprRange)), SM, clang::LangOptions(), nullptr);
}

/**
  * Get a string representation of the expression `arg`, unless it's a defaulting function
  * call argument, in which case get a 0. Used for building argument lists to kernel calls.
  */
std::string stringifyZeroDefaultedArg(clang::SourceManager &SM, const clang::Expr *arg) {
  if (clang::isa<clang::CXXDefaultArgExpr>(arg)) return "0";
  else return std::string(readSourceText(SM, arg->getSourceRange()));
}

} // anonymous namespace

bool HipifyAction::Exclude(const hipCounter &hipToken) {
  switch (hipToken.type) {
    case CONV_INCLUDE_CUDA_MAIN_H:
      switch (hipToken.apiType) {
        case API_DRIVER:
        case API_RUNTIME:
          if (insertedRuntimeHeader) return true;
          insertedRuntimeHeader = true;
          return false;
        case API_BLAS:
          if (insertedBLASHeader) return true;
          insertedBLASHeader = true;
          return false;
        case API_RAND:
          if (hipToken.hipName == s_hiprand_kernel_h) {
            if (insertedRAND_kernelHeader) return true;
            insertedRAND_kernelHeader = true;
            return false;
          } else if (hipToken.hipName == s_hiprand_h) {
            if (insertedRANDHeader) return true;
            insertedRANDHeader = true;
            return false;
          }
        case API_DNN:
          if (insertedDNNHeader) return true;
          insertedDNNHeader = true;
          return false;
        case API_FFT:
          if (insertedFFTHeader) return true;
          insertedFFTHeader = true;
          return false;
        case API_COMPLEX:
          if (insertedComplexHeader) return true;
          insertedComplexHeader = true;
          return false;
        case API_SPARSE:
          if (insertedSPARSEHeader) return true;
          insertedSPARSEHeader = true;
          return false;
        default:
          return false;
      }
      return false;
    case CONV_INCLUDE:
      if (hipToken.hipName.empty()) return true;
      switch (hipToken.apiType) {
        case API_RAND:
          if (hipToken.hipName == s_hiprand_kernel_h) {
            if (insertedRAND_kernelHeader) return true;
            insertedRAND_kernelHeader = true;
          }
          return false;
        default:
          return false;
      }
      return false;
    default:
      return false;
  }
  return false;
}

void HipifyAction::InclusionDirective(clang::SourceLocation hash_loc,
                                      const clang::Token&,
                                      StringRef file_name,
                                      bool is_angled,
                                      clang::CharSourceRange filename_range,
                                      const clang::FileEntry*, StringRef,
                                      StringRef, const clang::Module*) {
  auto &SM = getCompilerInstance().getSourceManager();
  if (!SM.isWrittenInMainFile(hash_loc)) return;
  if (!firstHeader) {
    firstHeader = true;
    firstHeaderLoc = hash_loc;
  }
  const auto found = CUDA_INCLUDE_MAP.find(file_name);
  if (found == CUDA_INCLUDE_MAP.end()) return;
  bool exclude = Exclude(found->second);
  Statistics::current().incrementCounter(found->second, file_name.str());
  clang::SourceLocation sl = filename_range.getBegin();
  if (Statistics::isUnsupported(found->second)) {
    clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
    DE.Report(sl, DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "Unsupported CUDA header."));
    return;
  }
  clang::StringRef newInclude;
  // Keep the same include type that the user gave.
  if (!exclude) {
    clang::SmallString<128> includeBuffer;
    llvm::StringRef name = Statistics::isToRoc(found->second) ? found->second.rocName : found->second.hipName;
    if (is_angled) newInclude = llvm::Twine("<" + name+ ">").toStringRef(includeBuffer);
    else           newInclude = llvm::Twine("\"" + name + "\"").toStringRef(includeBuffer);
  } else {
    // hashLoc is location of the '#', thus replacing the whole include directive by empty newInclude starting with '#'.
    sl = hash_loc;
  }
  const char *B = SM.getCharacterData(sl);
  const char *E = SM.getCharacterData(filename_range.getEnd());
  ct::Replacement Rep(SM, sl, E - B, newInclude.str());
  insertReplacement(Rep, clang::FullSourceLoc{sl, SM});
}

void HipifyAction::PragmaDirective(clang::SourceLocation Loc, clang::PragmaIntroducerKind Introducer) {
  if (pragmaOnce) return;
  auto &SM = getCompilerInstance().getSourceManager();
  if (!SM.isWrittenInMainFile(Loc)) return;
  clang::Preprocessor &PP = getCompilerInstance().getPreprocessor();
  clang::Token tok;
  PP.Lex(tok);
  StringRef Text(SM.getCharacterData(tok.getLocation()), tok.getLength());
  if (Text == sOnce) {
    pragmaOnce = true;
    pragmaOnceLoc = tok.getEndLoc();
  }
}

bool HipifyAction::cudaLaunchKernel(const mat::MatchFinder::MatchResult &Result) {
  auto *launchKernel = Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>(sCudaLaunchKernel);
  if (!launchKernel) return false;
  auto *calleeExpr = launchKernel->getCallee();
  if (!calleeExpr) return false;
  auto *caleeDecl = launchKernel->getDirectCallee();
  if (!caleeDecl) return false;
  auto *config = launchKernel->getConfig();
  if (!config) return false;
  clang::SmallString<40> XStr;
  llvm::raw_svector_ostream OS(XStr);
  clang::LangOptions DefaultLangOptions;
  auto *SM = Result.SourceManager;
  clang::SourceRange sr = calleeExpr->getSourceRange();
  std::string kern = readSourceText(*SM, sr).str();
  OS << sHipLaunchKernelGGL << "(";
  if (caleeDecl->isTemplateInstantiation()) {
    OS << sHIP_KERNEL_NAME << "(";
    std::string cub = sCub + "::";
    std::string hipcub;
    const auto found = CUDA_CUB_TYPE_NAME_MAP.find(sCub);
    if (found != CUDA_CUB_TYPE_NAME_MAP.end()) {
      hipcub = found->second.hipName.str() + "::";
    } else {
      hipcub = sHipcub + "::";
    }
    size_t pos = kern.find(cub);
    while (pos != std::string::npos) {
      kern.replace(pos, cub.size(), hipcub);
      pos = kern.find(cub, pos + hipcub.size());
    }
  }
  OS << kern;
  if (caleeDecl->isTemplateInstantiation()) OS << ")";
  OS << ", ";
  // Next up are the four kernel configuration parameters, the last two of which are optional and default to zero.
  // Copy the two dimensional arguments verbatim.
  for (unsigned int i = 0; i < 2; ++i) {
    const std::string sArg = readSourceText(*SM, config->getArg(i)->getSourceRange()).str();
    bool bDim3 = std::equal(sDim3.begin(), sDim3.end(), sArg.c_str());
    OS << (bDim3 ? "" : sDim3) << sArg << (bDim3 ? "" : ")") << ", ";
  }
  // The stream/memory arguments default to zero if omitted.
  OS << stringifyZeroDefaultedArg(*SM, config->getArg(2)) << ", ";
  OS << stringifyZeroDefaultedArg(*SM, config->getArg(3));
  // If there are ordinary arguments to the kernel, just copy them verbatim into our new call.
  int numArgs = launchKernel->getNumArgs();
  if (numArgs > 0) {
    OS << ", ";
    // Start of the first argument.
    clang::SourceLocation argStart = llcompat::getBeginLoc(launchKernel->getArg(0));
    // End of the last argument.
    clang::SourceLocation argEnd = llcompat::getEndLoc(launchKernel->getArg(numArgs - 1));
    OS << readSourceText(*SM, {argStart, argEnd});
  }
  OS << ")";
  clang::SourceLocation launchKernelExprLocBeg = launchKernel->getExprLoc();
  clang::SourceLocation launchKernelExprLocEnd = launchKernelExprLocBeg.isMacroID() ? llcompat::getEndOfExpansionRangeForLoc(*SM, launchKernelExprLocBeg) : llcompat::getEndLoc(launchKernel);
  clang::SourceLocation launchKernelEnd = llcompat::getEndLoc(launchKernel);
  clang::BeforeThanCompare<clang::SourceLocation> isBefore(*SM);
  launchKernelExprLocEnd = isBefore(launchKernelEnd, launchKernelExprLocEnd) ? launchKernelExprLocEnd : launchKernelEnd;
  clang::SourceRange replacementRange = getWriteRange(*SM, {launchKernelExprLocBeg, launchKernelExprLocEnd});
  clang::SourceLocation launchBeg = replacementRange.getBegin();
  clang::SourceLocation launchEnd = replacementRange.getEnd();
  if (isBefore(launchBeg, launchEnd)) {
    size_t length = SM->getCharacterData(clang::Lexer::getLocForEndOfToken(launchEnd, 0, *SM, DefaultLangOptions)) - SM->getCharacterData(launchBeg);
    ct::Replacement Rep(*SM, launchBeg, length, OS.str());
    clang::FullSourceLoc fullSL(launchBeg, *SM);
    insertReplacement(Rep, fullSL);
    hipCounter counter = {sHipLaunchKernelGGL, "", ConvTypes::CONV_KERNEL_LAUNCH, ApiTypes::API_RUNTIME};
    Statistics::current().incrementCounter(counter, sCudaLaunchKernel.str());
    return true;
  }
  return false;
}

bool HipifyAction::cudaSharedIncompleteArrayVar(const mat::MatchFinder::MatchResult &Result) {
  auto *sharedVar = Result.Nodes.getNodeAs<clang::VarDecl>(sCudaSharedIncompleteArrayVar);
  if (!sharedVar) return false;
  // Example: extern __shared__ uint sRadix1[];
  if (!sharedVar->hasExternalFormalLinkage()) return false;
  clang::QualType QT = sharedVar->getType();
  std::string typeName;
  if (QT->isIncompleteArrayType()) {
    const clang::ArrayType *AT = QT.getTypePtr()->getAsArrayTypeUnsafe();
    QT = AT->getElementType();
    if (QT.getTypePtr()->isBuiltinType()) {
      QT = QT.getCanonicalType();
      auto *BT = clang::dyn_cast<clang::BuiltinType>(QT);
      if (BT) {
        clang::LangOptions LO;
        LO.CUDA = true;
        clang::PrintingPolicy policy(LO);
        typeName = std::string(BT->getName(policy));
      }
    } else {
      typeName = QT.getAsString();
    }
  }
  if (!typeName.empty()) {
    clang::SourceLocation slStart = sharedVar->getOuterLocStart();
    clang::SourceLocation slEnd = llcompat::getEndLoc(sharedVar->getTypeSourceInfo()->getTypeLoc());
    auto *SM = Result.SourceManager;
    size_t repLength = SM->getCharacterData(slEnd) - SM->getCharacterData(slStart) + 1;
    std::string varName = sharedVar->getNameAsString();
    std::string repName = sHIP_DYNAMIC_SHARED + "(" + typeName + ", " + varName + ")";
    ct::Replacement Rep(*SM, slStart, repLength, repName);
    clang::FullSourceLoc fullSL(slStart, *SM);
    insertReplacement(Rep, fullSL);
    hipCounter counter = {sHIP_DYNAMIC_SHARED, "", ConvTypes::CONV_EXTERN_SHARED, ApiTypes::API_RUNTIME};
    Statistics::current().incrementCounter(counter, sCudaSharedIncompleteArrayVar.str());
    return true;
  }
  return false;
}

bool HipifyAction::cudaDeviceFuncCall(const mat::MatchFinder::MatchResult &Result) {
  if (const clang::CallExpr *call = Result.Nodes.getNodeAs<clang::CallExpr>(sCudaDeviceFuncCall)) {
    auto *funcDcl = call->getDirectCallee();
    if (!funcDcl) return false;
    FindAndReplace(funcDcl->getDeclName().getAsString(), llcompat::getBeginLoc(call), CUDA_DEVICE_FUNC_MAP, false);
    return true;
  }
  return false;
}

bool HipifyAction::cubNamespacePrefix(const mat::MatchFinder::MatchResult &Result) {
  if (auto *decl = Result.Nodes.getNodeAs<clang::TypedefNameDecl>(sCubNamespacePrefix)) {
    clang::QualType QT = decl->getUnderlyingType();
    auto *t = QT.getTypePtr();
    if (!t) return false;
    const clang::ElaboratedType *et = t->getAs<clang::ElaboratedType>();
    if (!et) return false;
    const clang::NestedNameSpecifier *nns = et->getQualifier();
    if (!nns) return false;
    const clang::NamespaceDecl *nsd = nns->getAsNamespace();
    if (!nsd) return false;
    const clang::TypeSourceInfo *si = decl->getTypeSourceInfo();
    const clang::TypeLoc tloc = si->getTypeLoc();
    const clang::SourceRange sr = tloc.getSourceRange();
    std::string name = nsd->getDeclName().getAsString();
    FindAndReplace(name, GetSubstrLocation(name, sr), CUDA_CUB_TYPE_NAME_MAP);
    return true;
  }
  return false;
}

bool HipifyAction::cubUsingNamespaceDecl(const mat::MatchFinder::MatchResult &Result) {
  if (auto *decl = Result.Nodes.getNodeAs<clang::UsingDirectiveDecl>(sCubUsingNamespaceDecl)) {
    if (auto nsd = decl->getNominatedNamespace()) {
      FindAndReplace(nsd->getDeclName().getAsString(), decl->getIdentLocation(), CUDA_CUB_TYPE_NAME_MAP);
      return true;
    }
  }
  return false;
}

bool HipifyAction::cubFunctionTemplateDecl(const mat::MatchFinder::MatchResult &Result) {
  if (auto *decl = Result.Nodes.getNodeAs<clang::FunctionTemplateDecl>(sCubFunctionTemplateDecl)) {
    auto *Tparams = decl->getTemplateParameters();
    bool ret = false;
    for (size_t I = 0; I < Tparams->size(); ++I) {
      const clang::ValueDecl *valueDecl = dyn_cast<clang::ValueDecl>(Tparams->getParam(I));
      if (!valueDecl) continue;
      clang::QualType QT = valueDecl->getType();
      auto *t = QT.getTypePtr();
      if (!t) continue;
      const clang::ElaboratedType *et = t->getAs<clang::ElaboratedType>();
      if (!et) continue;
      const clang::NestedNameSpecifier *nns = et->getQualifier();
      if (!nns) continue;
      const clang::NamespaceDecl *nsd = nns->getAsNamespace();
      if (!nsd) continue;
      const clang::SourceRange sr = valueDecl->getSourceRange();
      std::string name = nsd->getDeclName().getAsString();
      FindAndReplace(name, GetSubstrLocation(name, sr), CUDA_CUB_TYPE_NAME_MAP);
      ret = true;
    }
    return ret;
  }
  return false;
}

bool HipifyAction::cudaHostFuncCall(const mat::MatchFinder::MatchResult &Result) {
  if (auto *call = Result.Nodes.getNodeAs<clang::CallExpr>(sCudaHostFuncCall)) {
    if (!call->getNumArgs()) return false;
    auto *funcDcl = call->getDirectCallee();
    if (!funcDcl) return false;
    std::string sName = funcDcl->getDeclName().getAsString();
    unsigned int argNum = 0;
    bool b_reinterpret = (ReinterpretFunctions.find(sName) != ReinterpretFunctions.end()) ? true : false;
    if (DeviceSymbolFunctions0.find(sName) != DeviceSymbolFunctions0.end() || sCudaFuncSetCacheConfig == sName) {
      argNum = 0;
    } else if (call->getNumArgs() > 1 && (DeviceSymbolFunctions1.find(sName) != DeviceSymbolFunctions1.end() || sCudaFuncGetAttributes == sName)) {
      argNum = 1;
    } else {
      return false;
    }
    clang::SmallString<40> XStr;
    llvm::raw_svector_ostream OS(XStr);
    clang::SourceRange sr = call->getArg(argNum)->getSourceRange();
    auto *SM = Result.SourceManager;
    OS << (b_reinterpret ? s_reinterpret_cast : sHIP_SYMBOL) << "(" << readSourceText(*SM, sr) << ")";
    clang::SourceRange replacementRange = getWriteRange(*SM, { sr.getBegin(), sr.getEnd() });
    clang::SourceLocation s = replacementRange.getBegin();
    clang::SourceLocation e = replacementRange.getEnd();
    clang::LangOptions DefaultLangOptions;
    size_t length = SM->getCharacterData(clang::Lexer::getLocForEndOfToken(e, 0, *SM, DefaultLangOptions)) - SM->getCharacterData(s);
    ct::Replacement Rep(*SM, s, length, OS.str());
    clang::FullSourceLoc fullSL(s, *SM);
    insertReplacement(Rep, fullSL);
    return true;
  }
  return false;
}

void HipifyAction::insertReplacement(const ct::Replacement &rep, const clang::FullSourceLoc &fullSL) {
  llcompat::insertReplacement(*replacements, rep);
  if (PrintStats) {
    rep.getLength();
    Statistics::current().lineTouched(fullSL.getExpansionLineNumber());
    Statistics::current().bytesChanged(rep.getLength());
  }
}

std::unique_ptr<clang::ASTConsumer> HipifyAction::CreateASTConsumer(clang::CompilerInstance &CI, StringRef) {
  Finder.reset(new mat::MatchFinder);
  // Replace the <<<...>>> language extension with a hip kernel launch
  Finder->addMatcher(mat::cudaKernelCallExpr(mat::isExpansionInMainFile()).bind(sCudaLaunchKernel), this);
  Finder->addMatcher(
    mat::varDecl(
      mat::isExpansionInMainFile(),
      mat::allOf(
        mat::hasAttr(clang::attr::CUDAShared),
        mat::hasType(mat::incompleteArrayType())
      )
    ).bind(sCudaSharedIncompleteArrayVar),
    this
  );
  Finder->addMatcher(
    mat::callExpr(
      mat::isExpansionInMainFile(),
      mat::callee(
        mat::functionDecl(
          mat::hasAnyName(
            sCudaGetSymbolAddress,
            sCudaGetSymbolSize,
            sCudaMemcpyFromSymbol,
            sCudaMemcpyFromSymbolAsync,
            sCudaMemcpyToSymbol,
            sCudaMemcpyToSymbolAsync,
            sCudaFuncSetCacheConfig,
            sCudaFuncGetAttributes
          )
        )
      )
    ).bind(sCudaHostFuncCall),
    this
  );
  Finder->addMatcher(
    mat::callExpr(
      mat::isExpansionInMainFile(),
      mat::callee(
        mat::functionDecl(
          mat::anyOf(
            mat::hasAttr(clang::attr::CUDADevice),
            mat::hasAttr(clang::attr::CUDAGlobal)
          ),
          mat::unless(mat::hasAttr(clang::attr::CUDAHost))
        )
      )
    ).bind(sCudaDeviceFuncCall),
    this
  );
  Finder->addMatcher(
    mat::typedefDecl(
      mat::isExpansionInMainFile(),
      mat::hasType(
        mat::elaboratedType(
          mat::hasQualifier(
            mat::specifiesNamespace(
              mat::hasName(sCub)
            )
          )
        )
       )
    ).bind(sCubNamespacePrefix),
    this
  );
  // TODO: Maybe worth to make it more concrete based on final cubFunctionTemplateDecl
  Finder->addMatcher(
    mat::functionTemplateDecl(
      mat::isExpansionInMainFile()
    ).bind(sCubFunctionTemplateDecl),
    this
  );
  // TODO: Maybe worth to make it more concrete
  Finder->addMatcher(
    mat::usingDirectiveDecl(
      mat::isExpansionInMainFile()
    ).bind(sCubUsingNamespaceDecl),
    this
  );
  // Ownership is transferred to the caller.
  return Finder->newASTConsumer();
}

void HipifyAction::Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD) {
  auto &SM = getCompilerInstance().getSourceManager();
  if (!SM.isWrittenInMainFile(Loc)) return;
  StringRef Text(SM.getCharacterData(MacroNameTok.getLocation()), MacroNameTok.getLength());
  Ifndefs.insert(std::make_pair(Text.str(), MacroNameTok.getEndLoc()));
}

void HipifyAction::EndSourceFileAction() {
  // Insert the hip header, if we didn't already do it by accident during substitution.
  if (!insertedRuntimeHeader) {
    // It's not sufficient to just replace CUDA headers with hip ones, because numerous CUDA headers are
    // implicitly included by the compiler. Instead, we _delete_ CUDA headers, and unconditionally insert
    // one copy of the hip include into every file.
    bool placeForIncludeCalculated = false;
    clang::SourceLocation sl, controllingMacroLoc;
    auto &SM = getCompilerInstance().getSourceManager();
    clang::Preprocessor &PP = getCompilerInstance().getPreprocessor();
    clang::HeaderSearch &HS = PP.getHeaderSearchInfo();
    clang::ExternalPreprocessorSource *EPL = HS.getExternalLookup();
    const clang::FileEntry *FE = SM.getFileEntryForID(SM.getMainFileID());
    const clang::IdentifierInfo *controllingMacro = HS.getFileInfo(FE).getControllingMacro(EPL);
    if (controllingMacro) {
      auto found = Ifndefs.find(controllingMacro->getName().str());
      if (found != Ifndefs.end()) {
        controllingMacroLoc = found->second;
        placeForIncludeCalculated = true;
      }
    }
    if (pragmaOnce) {
      if (placeForIncludeCalculated) sl = pragmaOnceLoc < controllingMacroLoc ? pragmaOnceLoc : controllingMacroLoc;
      else                           sl = pragmaOnceLoc;
      placeForIncludeCalculated = true;
    }
    if (!placeForIncludeCalculated) {
      if (firstHeader)               sl = firstHeaderLoc;
      else                           sl = SM.getLocForStartOfFile(SM.getMainFileID());
    }
    clang::FullSourceLoc fullSL(sl, SM);
    ct::Replacement Rep(SM, sl, 0, "\n#include <hip/hip_runtime.h>\n");
    insertReplacement(Rep, fullSL);
  }
  clang::ASTFrontendAction::EndSourceFileAction();
}

namespace {

/**
  * A silly little class to proxy PPCallbacks back to the HipifyAction class.
  */
class PPCallbackProxy : public clang::PPCallbacks {
  HipifyAction &hipifyAction;

public:
  explicit PPCallbackProxy(HipifyAction &action): hipifyAction(action) {}

  void InclusionDirective(clang::SourceLocation hash_loc, const clang::Token &include_token,
                          StringRef file_name, bool is_angled, clang::CharSourceRange filename_range,
                          const clang::FileEntry *file, StringRef search_path, StringRef relative_path,
                          const clang::Module *imported
#if LLVM_VERSION_MAJOR > 6
                        , clang::SrcMgr::CharacteristicKind FileType
#endif
                         ) override {
    hipifyAction.InclusionDirective(hash_loc, include_token, file_name, is_angled, filename_range, file, search_path, relative_path, imported);
  }

  void PragmaDirective(clang::SourceLocation Loc, clang::PragmaIntroducerKind Introducer) override {
    hipifyAction.PragmaDirective(Loc, Introducer);
  }

  void Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD) override {
    hipifyAction.Ifndef(Loc, MacroNameTok, MD);
  }
};
}

bool HipifyAction::BeginInvocation(clang::CompilerInstance &CI) {
  llcompat::RetainExcludedConditionalBlocks(CI);
  return true;
}

void HipifyAction::ExecuteAction() {
  clang::Preprocessor &PP = getCompilerInstance().getPreprocessor();
  auto &SM = getCompilerInstance().getSourceManager();
  // Start lexing the specified input file.
  const llvm::MemoryBuffer *FromFile = SM.getBuffer(SM.getMainFileID());
  clang::Lexer RawLex(SM.getMainFileID(), FromFile, SM, PP.getLangOpts());
  RawLex.SetKeepWhitespaceMode(true);
  // Perform a token-level rewrite of CUDA identifiers to hip ones. The raw-mode lexer gives us enough
  // information to tell the difference between identifiers, string literals, and "other stuff". It also
  // ignores preprocessor directives, so this transformation will operate inside preprocessor-deleted code.
  clang::Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  while (RawTok.isNot(clang::tok::eof)) {
    RewriteToken(RawTok);
    RawLex.LexFromRawLexer(RawTok);
  }
  // Register yourself as the preprocessor callback, by proxy.
  PP.addPPCallbacks(std::unique_ptr<PPCallbackProxy>(new PPCallbackProxy(*this)));
  // Now we're done futzing with the lexer, have the subclass proceeed with Sema and AST matching.
  clang::ASTFrontendAction::ExecuteAction();
}

void HipifyAction::run(const mat::MatchFinder::MatchResult &Result) {
  if (cudaLaunchKernel(Result)) return;
  if (cudaSharedIncompleteArrayVar(Result)) return;
  if (cudaHostFuncCall(Result)) return;
  if (cudaDeviceFuncCall(Result)) return;
  if (cubNamespacePrefix(Result)) return;
  if (cubFunctionTemplateDecl(Result)) return;
  if (cubUsingNamespaceDecl(Result)) return;
}
