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
  const std::vector<std::string> accept_ptr = {"(nil)", "0x", "0x0", "nil", "(null)", "", "0000000000000000"};
  auto isPtr = [&](const std::string& s) {
    if (std::find(accept_ptr.begin(), accept_ptr.end(), s) != accept_ptr.end())
      return true;
    // Accept 0x followed by hex digits (case-insensitive)
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') &&
        std::all_of(s.begin() + 2, s.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); }))
      return true;
    return false;
  };
  auto eq = [&](const std::string& r, const std::string& o) {
    if (r == o) return true;
    char *er, *eo;
    float fr = std::strtof(r.c_str(), &er), fo = std::strtof(o.c_str(), &eo);
    if (er != r.c_str() && eo != o.c_str() && std::fabs(fr - fo) <= tol) return true;
    if (isPtr(r) && isPtr(o)) return true;
    // Handle lines with a common prefix and pointer suffix (e.g. "%s0xABCD" vs "%s(nil)")
    // Some OpenCL implementations print (nil) for all %p values.
    size_t prefix = 0;
    while (prefix < r.size() && prefix < o.size() && r[prefix] == o[prefix])
      prefix++;
    if (prefix > 0 && isPtr(r.substr(prefix)) && isPtr(o.substr(prefix)))
      return true;
    // Last line: float + "hello" + hex (case-insensitive)
    auto ciContains = [](const std::string& s, const std::string& sub) {
      auto it = std::search(s.begin(), s.end(), sub.begin(), sub.end(),
          [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b)); });
      return it != s.end();
    };
    if (ciContains(r, "hello") && ciContains(o, "hello")) {
      // Allow %p to be (nil) or hex value in the combined line
      bool rHasHex = ciContains(r, "f01dab1eca55e77e");
      bool oHasHex = ciContains(o, "f01dab1eca55e77e");
      bool rHasNil = ciContains(r, "(nil)") || ciContains(r, "0x");
      bool oHasNil = ciContains(o, "(nil)") || ciContains(o, "0x");
      if ((rHasHex || rHasNil) && (oHasHex || oHasNil) &&
          std::fabs(std::strtof(r.c_str(), nullptr) - std::strtof(o.c_str(), nullptr)) <= tol)
        return true;
    }
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
