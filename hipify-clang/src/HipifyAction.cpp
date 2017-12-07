#include "HipifyAction.h"

#include <memory>

#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include "LLVMCompat.h"
#include "CUDA2HipMap.h"
#include "StringUtils.h"
#include "ArgParse.h"

namespace ct = clang::tooling;
namespace mat = clang::ast_matchers;

void HipifyAction::RewriteString(StringRef s, clang::SourceLocation start) {
    clang::SourceManager& SM = getCompilerInstance().getSourceManager();

    size_t begin = 0;
    while ((begin = s.find("cu", begin)) != StringRef::npos) {
        const size_t end = s.find_first_of(" ", begin + 4);
        StringRef name = s.slice(begin, end);
        const auto found = CUDA_RENAMES_MAP().find(name);
        if (found != CUDA_RENAMES_MAP().end()) {
            StringRef repName = found->second.hipName;
            hipCounter counter = {"[string literal]", ConvTypes::CONV_LITERAL, ApiTypes::API_RUNTIME, found->second.unsupported};
            Statistics::current().incrementCounter(counter, name.str());

            if (!counter.unsupported) {
                clang::SourceLocation sl = start.getLocWithOffset(begin + 1);
                ct::Replacement Rep(SM, sl, name.size(), repName);
                clang::FullSourceLoc fullSL(sl, SM);
                insertReplacement(Rep, fullSL);
            }
        }

        if (end == StringRef::npos) {
            break;
        }

        begin = end + 1;
    }
}

/**
 * Look at, and consider altering, a given token.
 *
 * If it's not a CUDA identifier, nothing happens.
 * If it's an unsupported CUDA identifier, a warning is emitted.
 * Otherwise, the source file is updated with the corresponding hipification.
 */
void HipifyAction::RewriteToken(const clang::Token& t) {
    clang::SourceManager& SM = getCompilerInstance().getSourceManager();

    // String literals containing CUDA references need fixing...
    if (t.is(clang::tok::string_literal)) {
        StringRef s(t.getLiteralData(), t.getLength());
        RewriteString(unquoteStr(s), t.getLocation());
        return;
    } else if (!t.isAnyIdentifier()) {
        // If it's neither a string nor an identifier, we don't care.
        return;
    }

    StringRef name = t.getRawIdentifier();
    const auto found = CUDA_RENAMES_MAP().find(name);
    if (found == CUDA_RENAMES_MAP().end()) {
        // So it's an identifier, but not CUDA? Boring.
        return;
    }

    Statistics::current().incrementCounter(found->second, name.str());

    clang::SourceLocation sl = t.getLocation();
    if (found->second.unsupported) {
        // An unsupported identifier? Curses! Warn the user.
        clang::DiagnosticsEngine& DE = getCompilerInstance().getDiagnostics();
        const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "CUDA identifier unsupported in hip");
        DE.Report(sl, ID);
        return;
    }

    StringRef repName = found->second.hipName;
    ct::Replacement Rep(SM, sl, name.size(), repName);
    clang::FullSourceLoc fullSL(sl, SM);
    insertReplacement(Rep, fullSL);
}

namespace {

clang::SourceRange getReadRange(clang::SourceManager& SM, const clang::SourceRange& exprRange) {
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


clang::SourceRange getWriteRange(clang::SourceManager& SM, const clang::SourceRange& exprRange) {
    clang::SourceLocation begin = exprRange.getBegin();
    clang::SourceLocation end = exprRange.getEnd();

    // If the range is contained within a macro, update the macro definition.
    // Otherwise, use the file location and hope for the best.
    if (!SM.isMacroBodyExpansion(begin) || !SM.isMacroBodyExpansion(end)) {
        return {SM.getFileLoc(begin), SM.getFileLoc(end)};
    }

    return {SM.getSpellingLoc(begin), SM.getSpellingLoc(end)};
}


StringRef readSourceText(clang::SourceManager& SM, const clang::SourceRange& exprRange) {
    return clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(getReadRange(SM, exprRange)), SM, clang::LangOptions(), nullptr);
}

/**
 * Get a string representation of the expression `arg`, unless it's a defaulting function
 * call argument, in which case get a 0. Used for building argument lists to kernel calls.
 */
std::string stringifyZeroDefaultedArg(clang::SourceManager& SM, const clang::Expr* arg) {
    if (clang::isa<clang::CXXDefaultArgExpr>(arg)) {
        return "0";
    } else {
        return readSourceText(SM, arg->getSourceRange());
    }
}

} // anonymous namespace


void HipifyAction::InclusionDirective(clang::SourceLocation hash_loc,
                                      const clang::Token&,
                                      StringRef file_name,
                                      bool is_angled,
                                      clang::CharSourceRange filename_range,
                                      const clang::FileEntry*, StringRef,
                                      StringRef, const clang::Module*) {
    clang::SourceManager& SM = getCompilerInstance().getSourceManager();
    if (!SM.isWrittenInMainFile(hash_loc)) {
        return;
    }

    const auto found = CUDA_INCLUDE_MAP.find(file_name);
    if (found == CUDA_INCLUDE_MAP.end()) {
        // Not a CUDA include - don't touch it.
        return;
    }

    // Special-casing to avoid duplication of the hip_runtime include.
    if (found->second.hipName == "hip/hip_runtime.h") {
        if (insertedRuntimeHeader) {
            return;
        }

        insertedRuntimeHeader = true;
    }

    Statistics::current().incrementCounter(found->second, file_name.str());

    clang::SourceLocation sl = filename_range.getBegin();
    if (found->second.unsupported) {
        // An unsupported CUDA header? Oh dear. Print a warning.
        clang::DiagnosticsEngine& DE = getCompilerInstance().getDiagnostics();
        DE.Report(sl, DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "Unsupported CUDA header"));
        return;
    }

    const char *B = SM.getCharacterData(sl);
    const char *E = SM.getCharacterData(filename_range.getEnd());
    clang::SmallString<128> includeBuffer;
    clang::StringRef newInclude;

    // Keep the same include type that the user gave.
    if (is_angled) {
        newInclude = llvm::Twine("<" + found->second.hipName + ">").toStringRef(includeBuffer);
    } else {
        newInclude = llvm::Twine("\"" + found->second.hipName + "\"").toStringRef(includeBuffer);
    }

    ct::Replacement Rep(SM, sl, E - B, newInclude);
    insertReplacement(Rep, clang::FullSourceLoc{sl, SM});
}


bool HipifyAction::cudaLaunchKernel(const clang::ast_matchers::MatchFinder::MatchResult& Result) {
    StringRef refName = "cudaLaunchKernel";

    const auto* launchKernel = Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>(refName);
    if (!launchKernel) {
        return false;
    }

    clang::SmallString<40> XStr;
    llvm::raw_svector_ostream OS(XStr);

    clang::LangOptions DefaultLangOptions;
    clang::SourceManager* SM = Result.SourceManager;

    const clang::Expr& calleeExpr = *(launchKernel->getCallee());
    OS << "hipLaunchKernelGGL(" << readSourceText(*SM, calleeExpr.getSourceRange()) << ", ";

    // Next up are the four kernel configuration parameters, the last two of which are optional and default to zero.
    const clang::CallExpr& config = *(launchKernel->getConfig());

    // Copy the two dimensional arguments verbatim.
    OS << "dim3(" << readSourceText(*SM, config.getArg(0)->getSourceRange()) << "), ";
    OS << "dim3(" << readSourceText(*SM, config.getArg(1)->getSourceRange()) << "), ";

    // The stream/memory arguments default to zero if omitted.
    OS << stringifyZeroDefaultedArg(*SM, config.getArg(2)) << ", ";
    OS << stringifyZeroDefaultedArg(*SM, config.getArg(3));

    // If there are ordinary arguments to the kernel, just copy them verbatim into our new call.
    int numArgs = launchKernel->getNumArgs();
    if (numArgs > 0) {
        OS << ", ";

        // Start of the first argument.
        clang::SourceLocation argStart = launchKernel->getArg(0)->getLocStart();

        // End of the last argument.
        clang::SourceLocation argEnd = launchKernel->getArg(numArgs - 1)->getLocEnd();

        OS << readSourceText(*SM, {argStart, argEnd});
    }

    OS << ")";

    clang::SourceRange replacementRange = getWriteRange(*SM, {launchKernel->getLocStart(), launchKernel->getLocEnd()});
    clang::SourceLocation launchStart = replacementRange.getBegin();
    clang::SourceLocation launchEnd = replacementRange.getEnd();

    size_t length = SM->getCharacterData(clang::Lexer::getLocForEndOfToken(launchEnd, 0, *SM, DefaultLangOptions)) - SM->getCharacterData(launchStart);

    ct::Replacement Rep(*SM, launchStart, length, OS.str());
    clang::FullSourceLoc fullSL(launchStart, *SM);
    insertReplacement(Rep, fullSL);
    hipCounter counter = {"hipLaunchKernelGGL", ConvTypes::CONV_KERN, ApiTypes::API_RUNTIME};
    Statistics::current().incrementCounter(counter, refName.str());

    return true;
}

bool HipifyAction::cudaSharedIncompleteArrayVar(const clang::ast_matchers::MatchFinder::MatchResult& Result) {
    StringRef refName = "cudaSharedIncompleteArrayVar";
    auto* sharedVar = Result.Nodes.getNodeAs<clang::VarDecl>(refName);
    if (!sharedVar) {
        return false;
    }

    // Example: extern __shared__ uint sRadix1[];
    if (!sharedVar->hasExternalFormalLinkage()) {
        return false;
    }

    clang::QualType QT = sharedVar->getType();
    std::string typeName;
    if (QT->isIncompleteArrayType()) {
        const clang::ArrayType* AT = QT.getTypePtr()->getAsArrayTypeUnsafe();
        QT = AT->getElementType();
        if (QT.getTypePtr()->isBuiltinType()) {
            QT = QT.getCanonicalType();
            const auto* BT = clang::dyn_cast<clang::BuiltinType>(QT);
            if (BT) {
                clang::LangOptions LO;
                LO.CUDA = true;
                clang::PrintingPolicy policy(LO);
                typeName = BT->getName(policy);
            }
        } else {
            typeName = QT.getAsString();
        }
    }

    if (!typeName.empty()) {
        clang::SourceLocation slStart = sharedVar->getLocStart();
        clang::SourceLocation slEnd = sharedVar->getLocEnd();
        clang::SourceManager* SM = Result.SourceManager;
        size_t repLength = SM->getCharacterData(slEnd) - SM->getCharacterData(slStart) + 1;
        std::string varName = sharedVar->getNameAsString();
        std::string repName = "HIP_DYNAMIC_SHARED(" + typeName + ", " + varName + ")";
        ct::Replacement Rep(*SM, slStart, repLength, repName);
        clang::FullSourceLoc fullSL(slStart, *SM);
        insertReplacement(Rep, fullSL);
        hipCounter counter = {"HIP_DYNAMIC_SHARED", ConvTypes::CONV_MEM, ApiTypes::API_RUNTIME};
        Statistics::current().incrementCounter(counter, refName.str());
    }

    return true;
}

void HipifyAction::insertReplacement(const ct::Replacement& rep, const clang::FullSourceLoc& fullSL) {
    llcompat::insertReplacement(*replacements, rep);
    if (PrintStats) {
        rep.getLength();
        Statistics::current().lineTouched(fullSL.getExpansionLineNumber());
        Statistics::current().bytesChanged(rep.getLength());
    }
}

std::unique_ptr<clang::ASTConsumer> HipifyAction::CreateASTConsumer(clang::CompilerInstance& CI, llvm::StringRef) {
    Finder.reset(new clang::ast_matchers::MatchFinder);

    // Replace the <<<...>>> language extension with a hip kernel launch
    Finder->addMatcher(mat::cudaKernelCallExpr(mat::isExpansionInMainFile()).bind("cudaLaunchKernel"), this);

    Finder->addMatcher(
        mat::varDecl(
            mat::isExpansionInMainFile(),
            mat::allOf(
                mat::hasAttr(clang::attr::CUDAShared),
                mat::hasType(mat::incompleteArrayType())
            )
        ).bind("cudaSharedIncompleteArrayVar"),
        this
    );

    // Ownership is transferred to the caller...
    return Finder->newASTConsumer();
}

void HipifyAction::EndSourceFileAction() {
    // Insert the hip header, if we didn't already do it by accident during substitution.
    if (!insertedRuntimeHeader) {
        // It's not sufficient to just replace CUDA headers with hip ones, because numerous CUDA headers are
        // implicitly included by the compiler. Instead, we _delete_ CUDA headers, and unconditionally insert
        // one copy of the hip include into every file.
        clang::SourceManager& SM = getCompilerInstance().getSourceManager();

        clang::SourceLocation sl = SM.getLocForStartOfFile(SM.getMainFileID());
        clang::FullSourceLoc fullSL(sl, SM);
        ct::Replacement Rep(SM, sl, 0, "#include <hip/hip_runtime.h>\n");
        insertReplacement(Rep, fullSL);
    }

    clang::ASTFrontendAction::EndSourceFileAction();
}


namespace {

/**
 * A silly little class to proxy PPCallbacks back to the HipifyAction class.
 */
class PPCallbackProxy : public clang::PPCallbacks {
    HipifyAction& hipifyAction;

public:
    explicit PPCallbackProxy(HipifyAction& action): hipifyAction(action) {}

    void InclusionDirective(clang::SourceLocation hash_loc, const clang::Token& include_token,
                            StringRef file_name, bool is_angled, clang::CharSourceRange filename_range,
                            const clang::FileEntry* file, StringRef search_path, StringRef relative_path,
                            const clang::Module* imported) override {
        hipifyAction.InclusionDirective(hash_loc, include_token, file_name, is_angled, filename_range, file, search_path, relative_path, imported);
    }
};

}

void HipifyAction::ExecuteAction() {
    clang::Preprocessor& PP = getCompilerInstance().getPreprocessor();
    clang::SourceManager& SM = getCompilerInstance().getSourceManager();

    // Start lexing the specified input file.
    const llvm::MemoryBuffer* FromFile = SM.getBuffer(SM.getMainFileID());
    clang::Lexer RawLex(SM.getMainFileID(), FromFile, SM, PP.getLangOpts());
    RawLex.SetKeepWhitespaceMode(true);

    // Perform a token-level rewrite of CUDA identifiers to hip ones. The raw-mode lexer gives us enough
    // information to tell the difference between identifiers, string literals, and "other stuff". It also
    // ignores preprocessor directives, so this transformation will operate inside preprocessor-deleted
    // code.
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

void HipifyAction::run(const clang::ast_matchers::MatchFinder::MatchResult& Result) {
    if (cudaLaunchKernel(Result)) return;
    if (cudaSharedIncompleteArrayVar(Result)) return;
}
