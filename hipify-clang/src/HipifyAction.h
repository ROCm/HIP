#pragma once

#include "clang/Lex/PPCallbacks.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "ReplacementsFrontendActionFactory.h"

namespace ct = clang::tooling;

/**
 * A FrontendAction that hipifies CUDA programs.
 */
class HipifyAction : public clang::ASTFrontendAction,
                     public clang::ast_matchers::MatchFinder::MatchCallback {
private:
    ct::Replacements* replacements;
    std::unique_ptr<clang::ast_matchers::MatchFinder> Finder;

    /// CUDA implicitly adds its runtime header. We rewrite explicitly-provided CUDA includes with equivalent
    // ones, and track - using this flag - if the result led to us including the hip runtime header. If it did
    // not, we insert it at the top of the file when we finish processing it.
    // This approach means we do the best it's possible to do w.r.t preserving the user's include order.
    bool insertedRuntimeHeader = false;

    /**
     * Rewrite a string literal to refer to hip, not CUDA.
     */
    void RewriteString(StringRef s, clang::SourceLocation start);

    /**
     * Replace a CUDA identifier with the corresponding hip identifier, if applicable.
     */
    void RewriteToken(const clang::Token &t);

public:
    explicit HipifyAction(ct::Replacements *replacements):
        clang::ASTFrontendAction(),
        replacements(replacements) {}

    // MatchCallback listeners
    bool cudaBuiltin(const clang::ast_matchers::MatchFinder::MatchResult& Result);
    bool cudaLaunchKernel(const clang::ast_matchers::MatchFinder::MatchResult& Result);
    bool cudaSharedIncompleteArrayVar(const clang::ast_matchers::MatchFinder::MatchResult& Result);

    /**
     * Called by the preprocessor for each include directive during the non-raw lexing pass.
     */
    void InclusionDirective(clang::SourceLocation hash_loc,
                            const clang::Token &include_token,
                            StringRef file_name,
                            bool is_angled,
                            clang::CharSourceRange filename_range,
                            const clang::FileEntry *file,
                            StringRef search_path,
                            StringRef relative_path,
                            const clang::Module *imported);

protected:
    /**
     * Add a Replacement for the current file. These will all be applied after executing the FrontendAction.
     */
    void insertReplacement(const ct::Replacement& rep, const clang::FullSourceLoc& fullSL);

    /**
     * FrontendAction entry point.
     */
    void ExecuteAction() override;

    /**
     * Called at the start of each new file to process.
     */
    void EndSourceFileAction() override;

    /**
     * MatchCallback API entry point. Called by the AST visitor while searching the AST for things we registered an
     * interest for.
     */
    void run(const clang::ast_matchers::MatchFinder::MatchResult& Result) override;

    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) override;
};
