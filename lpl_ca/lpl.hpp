#pragma once

#include "common.hpp"

#include "clara/clara.hpp"
#include "pstreams/pstream.h"
#include "../src/elfio/elfio.hpp"

#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hip_impl {
inline const std::string& kernel_section() {
    static const std::string r{".kernel"};

    return r;
}

inline const std::string& path_to_self() {
    static constexpr const char self[] = "/proc/self/exe";

    static std::string r;
    static std::once_flag f;

    std::call_once(f, []() {
        using N = decltype(readlink(self, &r.front(), r.size()));

        constexpr decltype(r.size()) max_path_sz{PATH_MAX};
        N read_cnt;
        do {
            r.resize(std::max(2 * r.size(), max_path_sz));
            read_cnt = readlink(self, &r.front(), r.size());
        } while (read_cnt == -1 && r.size() < r.max_size());

        r.resize(std::max(read_cnt, N{0}));
    });

    return r;
}

inline const std::string& path_to_hipcc() {
    assert(!path_to_self().empty());

    static const auto r = path_to_self().substr(0, path_to_self().find_last_of('/')) += "/hipcc";

    return r;
}

inline std::string make_hipcc_call(const std::vector<std::string>& sources,
                                   const std::vector<std::string>& targets,
                                   const std::string& flags, const std::string& hipcc_output) {
    assert(!sources.empty() && !targets.empty() && !hipcc_output.empty());

    std::string r{path_to_hipcc() + ' '};

    for (auto&& x : sources) r += x + ' ';
    r += "-o " + hipcc_output + ' ';
    for (auto&& x : targets) r += "--amdgpu-target=" + x + ' ';
    r += flags + " -fPIC -shared";

    return r;
}

inline void copy_kernel_section_to_fat_binary(const std::string& tmp, const std::string& output) {
    ELFIO::elfio reader;
    if (!reader.load(tmp)) {
        throw std::runtime_error{"The result of the compilation is inaccessible."};
    }

    const auto it =
        std::find_if(reader.sections.begin(), reader.sections.end(),
                     [](const ELFIO::section* x) { return x->get_name() == kernel_section(); });

    std::ofstream out{output};

    if (it == reader.sections.end()) {
        std::cerr << "Warning: no kernels were generated; fat binary shall "
                     "be empty."
                  << std::endl;
    } else {
        std::copy_n((*it)->get_data(), (*it)->get_size(), std::ostreambuf_iterator<char>{out});
    }
}

inline void generate_fat_binary(const std::vector<std::string>& sources,
                                const std::vector<std::string>& targets, const std::string& flags,
                                const std::string& output) {
    static const auto d = [](const std::string* f) { remove(f->c_str()); };
    std::string temp_str = output + ".tmp";
    std::unique_ptr<const std::string, decltype(d)> tmp{&temp_str, d};

    redi::ipstream hipcc{make_hipcc_call(sources, targets, flags, *tmp), redi::pstream::pstderr};

    if (!hipcc.is_open()) {
        throw std::runtime_error{"Compiler invocation failed."};
    }

    std::string log;
    while (std::getline(hipcc, log)) std::cout << log << '\n';

    hipcc.close();

    if (hipcc.rdbuf()->exited() && hipcc.rdbuf()->status() != EXIT_SUCCESS) {
        throw std::runtime_error{"Compilation failed."};
    }

    copy_kernel_section_to_fat_binary(*tmp, output);
}

inline bool hipcc_and_lpl_colocated() {
    if (path_to_self().empty()) return false;

    return file_exists(path_to_hipcc());
}

inline clara::Parser cmdline_parser(bool& help, std::vector<std::string>& sources,
                                    std::string& targets, std::string& flags, std::string& output) {
    return clara::Opt{flags, "\"-v -DMACRO etc.\""}["-f"]["--flags"](
               "flags for compilation; must be valid for hipcc.") |
           clara::Help{help} |
           clara::Opt{output, "filename"}["-o"]["--output"](
               "name of fat-binary output file; the binary format of the "
               "file is documented at: https://reviews.llvm.org/D13909.") |
           clara::Arg{sources,
                      "a.cpp b.cpp etc."}("inputs for compilation; must contain valid C++ code.") |
           clara::Opt{targets, "gfx803,gfx900,gfx906 etc."}["-t"]["--targets"](
               "targets for AMDGPU lowering; must be included in the set "
               "of processors with ROCm support from "
               "https://www.llvm.org/docs/AMDGPUUsage.html#processors.");
}
}  // namespace hip_impl
