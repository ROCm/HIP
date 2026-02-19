/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>
#include <hip_test_process.hh>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <sstream>
#include <vector>

TEST_CASE("Unit_printf_specifier") {
  // Single canonical reference; flexible comparison handles platform differences
  // (float precision, null %s/%p representation, hex case).
  std::string reference(R"here(xyzzy
%
hello % world
%s
%s0xf01dab1eca55e77e
%cxyzzy
sep
-42
42
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
x

(nil)
3.14159000    hello 0xf01dab1eca55e77e
)here");

#ifdef HIP_STANDALONE_PRINTF_PROC
  hip::SpawnProc proc("printfSpecifiers_proc", true);
#else
  hip::SpawnProc proc("printfSpecifiers", true);
#endif
  REQUIRE(0 == proc.run());

  std::string output = proc.getOutput();
  constexpr float tol = 1e-5f;
  const std::vector<std::string> accept_null = {"(nil)", "0x", "0x0", "nil", "(null)", "", "0000000000000000"};
  auto eq = [&](const std::string& r, const std::string& o) {
    if (r == o) return true;
    char *er, *eo;
    float fr = std::strtof(r.c_str(), &er), fo = std::strtof(o.c_str(), &eo);
    if (er != r.c_str() && eo != o.c_str() && std::fabs(fr - fo) <= tol) return true;
    if (std::find(accept_null.begin(), accept_null.end(), r) != accept_null.end() &&
        std::find(accept_null.begin(), accept_null.end(), o) != accept_null.end())
      return true;
    // Last line: float + "hello" + hex (case-insensitive)
    auto ciContains = [](const std::string& s, const std::string& sub) {
      auto it = std::search(s.begin(), s.end(), sub.begin(), sub.end(),
          [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b)); });
      return it != s.end();
    };
    if (ciContains(r, "hello") && ciContains(r, "f01dab1eca55e77e") &&
        ciContains(o, "hello") && ciContains(o, "f01dab1eca55e77e") &&
        std::fabs(std::strtof(r.c_str(), nullptr) - std::strtof(o.c_str(), nullptr)) <= tol)
      return true;
    return false;
  };
  std::istringstream rs(reference), os(output);
  std::string rl, ol;
  for (size_t i = 0; std::getline(rs, rl) && std::getline(os, ol); i++) {
    INFO("Line " << i << ": expected '" << rl << "' got '" << ol << "'");
    REQUIRE(eq(rl, ol));
  }
  REQUIRE(rs.eof());
  while (std::getline(os, ol)) {
    REQUIRE(ol.find_first_not_of(" \t\r\n") == std::string::npos);
  }
}
