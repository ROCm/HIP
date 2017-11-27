#pragma once

#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Core/Replacement.h"

namespace ct = clang::tooling;


/**
 * A FrontendActionFactory that propagates a set of Replacements into the FrontendAction.
 * This is necessary boilerplate for using a custom FrontendAction with a RefactoringTool.
 *
 * @tparam T The FrontendAction to create.
 */
template <typename T>
class ReplacementsFrontendActionFactory : public ct::FrontendActionFactory {
    ct::Replacements* replacements;

public:
    explicit ReplacementsFrontendActionFactory(ct::Replacements* r):
        ct::FrontendActionFactory(),
        replacements(r) {}

    clang::FrontendAction* create() override {
        return new T(replacements);
    }
};
