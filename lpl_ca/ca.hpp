#pragma once

#include "common.hpp"

#include "../include/hip/hcc_detail/code_object_bundle.hpp"

#include "clara/clara.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace hip_impl {
inline clara::Parser cmdline_parser(bool& help, std::vector<std::string>& inputs,
                                    std::string& targets) {
    return clara::Help{help} |
           clara::Arg{inputs, "a" + fat_binary_extension() + " etc."}(
               "fat binaries which contain the code objects to be extracted; "
               "the binary format of the file(s) is documented at: "
               "https://reviews.llvm.org/D13909; "
               "the code object format is documented at: "
               "https://www.llvm.org/docs/AMDGPUUsage.html#code-object.") |
           clara::Opt{targets, "gfx803,gfx900,gfx906 etc."}["-t"]["--targets"](
               "targets for which code objects are to be extracted from "
               "the fat binary; must be included in the set of processors "
               "with ROCm support from "
               "https://www.llvm.org/docs/AMDGPUUsage.html#processors.");
}

inline std::string make_code_object_file_name(const std::string& input, const std::string& target) {
    assert(!input.empty() && !target.empty());

    auto r = input.substr(0, input.find(fat_binary_extension()));
    r += '_' + target + code_object_extension();

    return r;
}

inline void extract_code_objects(const std::vector<std::string>& inputs,
                                 const std::vector<std::string>& targets) {
    for (auto&& input : inputs) {
        std::ifstream tmp{input};
        std::vector<char> bundle{std::istreambuf_iterator<char>{tmp},
                                 std::istreambuf_iterator<char>{}};

        Bundled_code_header tmp1{bundle};

        if (!valid(tmp1)) {
            throw std::runtime_error{input + " is not a valid fat binary."};
        }

        for (auto&& target : targets) {
            const auto it = std::find_if(
                bundles(tmp1).cbegin(), bundles(tmp1).cend(),
                [&](const Bundled_code& x) { return x.triple.find(target) != std::string::npos; });

            if (it == bundles(tmp1).cend()) {
                std::cerr << "Warning: " << input << " does not contain code for the " << target
                          << " target.";
                continue;
            }

            std::ofstream out{make_code_object_file_name(input, target)};
            std::copy_n(it->blob.cbegin(), it->blob.size(), std::ostreambuf_iterator<char>{out});
        }
    }
}

inline void validate_inputs(const std::vector<std::string>& inputs) {
    const auto it = std::find_if_not(inputs.cbegin(), inputs.cend(), file_exists);

    if (it != inputs.cend()) {
        throw std::runtime_error{"Non existent file " + *it + " passed as input."};
    }
}
}  // namespace hip_impl
