#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace hip_impl {
inline const std::unordered_set<std::string>& amdgpu_targets() {  // The evolving list lives at:
    // https://www.llvm.org/docs/AMDGPUUsage.html#processors.
    static const std::unordered_set<std::string> r{
        "gfx700", "gfx701",  "gfx702",  "gfx703",  "gfx704",  "gfx705",
        "gfx801", "gfx802", "gfx803", "gfx805",  "gfx810",
        "gfx900",  "gfx902",  "gfx904",  "gfx906", "gfx908", "gfx909",
        "gfx1010", "gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032"};
    return r;
}

inline const std::string& code_object_extension() {
    static const std::string r{".ffa"};

    return r;
}

inline const std::string& fat_binary_extension() {
    static const std::string r{".adipose"};

    return r;
}

inline bool file_exists(const std::string& path_to) {
    return static_cast<bool>(std::ifstream{path_to});
}

inline std::vector<std::string> tokenize_targets(
    const std::string&
        x) {  // TODO: move to regular expressions once we clarify the need to support
    //       ancient standard library implementations.
    if (x.empty()) return {};

    static constexpr const char valid_characters[] = "0123456789abcdefghijklmnopqrstuvwxyz,";

    if (x.find_first_not_of(valid_characters) != std::string::npos) {
        throw std::runtime_error{"Invalid target string: " + x};
    }

    std::vector<std::string> r;

    auto it = x.cbegin();
    do {
        auto it1 = std::find(it, x.cend(), ',');
        r.emplace_back(it, it1);

        if (it1 == x.cend()) break;

        it = ++it1;
    } while (true);

    return r;
}

inline void validate_targets(const std::vector<std::string>& x) {
    assert(!x.empty());

    for (auto&& t : x) {
        static const std::string sufficies{"0123456789abcdefghijklmnopqrstuvwxyz"};
        static const std::string prefix{"gfx"};

        if (t.find(prefix) != 0 || t.find_first_not_of(sufficies, prefix.size()) != std::string::npos) {
            throw std::runtime_error{"Invalid target: " + t};
        }

        if (amdgpu_targets().find(t) == amdgpu_targets().cend()) {
            std::cerr << "Warning: target " << t
                      << " has not been validated yet; it may be invalid." << std::endl;
        }
    }
}
}  // Namespace hip_impl.
