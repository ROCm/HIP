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
