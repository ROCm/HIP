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
#include <unordered_set>
#include <vector>

namespace hip_impl
{
    inline
    const std::unordered_set<std::string>& amdgpu_targets()
    {   // The evolving list lives at:
        // https://www.llvm.org/docs/AMDGPUUsage.html#processors.
        static const std::unordered_set<std::string> r{
            "gfx701", "gfx801", "gfx802", "gfx803", "gfx900"};

        return r;
    }

    inline
    const std::string& fat_binary_extension()
    {
        static const std::string r{".adipose"};

        return r;
    }

    inline
    const std::string& kernel_section()
    {
        static const std::string r{".kernel"};

        return r;
    }

    inline
    const std::string& path_to_self()
    {
        static constexpr const char self[] = "/proc/self/exe";

        static std::string r(PATH_MAX, '\0');
        static std::once_flag f;

        std::call_once(f, []() {
            decltype(readlink(self, &r.front(), r.size())) read_cnt;
            do {
                read_cnt = readlink(self, &r.front(), r.size());
            } while (read_cnt == -1);

            r.resize(read_cnt);
        });

        return r;
    }

    inline
    const std::string& path_to_hipcc()
    {
        assert(!path_to_self().empty());

        static const auto r = path_to_self().substr(
            0, path_to_self().find_last_of('/')) += "/hipcc";

        return r;
    }

    inline
    std::string make_hipcc_call(
        const std::vector<std::string>& sources,
        const std::vector<std::string>& targets,
        const std::string& flags,
        const std::string& hipcc_output)
    {
        assert(!sources.empty() && !targets.empty() && !hipcc_output.empty());

        std::string r{path_to_hipcc() + ' '};

        for (auto&& x : sources) r += x + ' ';
        r += "-o " + hipcc_output + ' ';
        for (auto&& x : targets) r += "--amdgpu-target=" + x + ' ';
        r += flags + " -fPIC -shared";

        return r;
    }

    inline
    void copy_kernel_section_to_fat_binary(
        const std::string& tmp, const std::string& output)
    {
        ELFIO::elfio reader;
        if (!reader.load(tmp)) {
            throw std::runtime_error{
                "The result of the compilation is inaccessible."};
        }

        const auto it = std::find_if(
            reader.sections.begin(),
            reader.sections.end(),
            [](const ELFIO::section* x) {
                return x->get_name() == kernel_section();
        });

        std::ofstream out{output + fat_binary_extension()};

        if (it == reader.sections.end()) {
            std::cerr << "Warning: no kernels were generated; fat binary shall "
                "be empty." << std::endl;
        }
        else {
            std::copy_n(
                (*it)->get_data(),
                (*it)->get_size(),
                std::ostreambuf_iterator<char>{out});
        }
    }

    inline
    void generate_fat_binary(
        const std::vector<std::string>& sources,
        const std::vector<std::string>& targets,
        const std::string& flags,
        const std::string& output)
    {
        static const auto d = [](const std::string* f) { remove(f->c_str()); };

        std::unique_ptr<const std::string, decltype(d)> tmp{&output, d};

        redi::ipstream hipcc{
            make_hipcc_call(sources, targets, flags, *tmp),
            redi::pstream::pstderr};

        if (!hipcc.is_open()) {
            throw std::runtime_error{"Compiler invocation failed."};
        }

        std::string log;
        while (std::getline(hipcc, log)) std::cout << log << '\n';

        hipcc.close();

        if (hipcc.rdbuf()->exited() &&
            hipcc.rdbuf()->status() != EXIT_SUCCESS) {
            throw std::runtime_error{"Compilation failed."};
        }

        copy_kernel_section_to_fat_binary(*tmp, output);
    }

    inline
    bool file_exists(const std::string& path_to)
    {
        return static_cast<bool>(std::ifstream{path_to});
    }

    inline
    bool hipcc_and_lpl_colocated()
    {
        if (path_to_self().empty()) return false;

        return file_exists(path_to_hipcc());
    }

    inline
    std::vector<std::string> tokenize_targets(const std::string& x)
    {   // TODO: move to regular expressions once we clarify the need to support
        //       ancient standard library implementations.
        if (x.empty()) return {};

        static constexpr const char valid_characters[] = "gfx0123456789,";

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

    inline
    void validate_targets(const std::vector<std::string>& x)
    {
        assert(!x.empty());

        for (auto&& t : x) {
            static const std::string digits{"0123456789"};
            static const std::string pre{"gfx"};

            if (t.find(pre) != 0 ||
                t.find_first_not_of(digits, pre.size()) != std::string::npos) {
                throw std::runtime_error{"Invalid target: " + t};
            }

            if (amdgpu_targets().find(t) == amdgpu_targets().cend()) {
                std::cerr << "Warning: target " << t
                    << " has not been validated yet; it may be invalid."
                    << std::endl;
            }
        }
    }

    inline
    clara::Parser cmdline_parser(
        bool& help,
        std::vector<std::string>& sources,
        std::string& targets,
        std::string& flags,
        std::string& output)
    {
        return
            clara::Opt{flags, "\"-v -DMACRO etc.\""}
                ["-f"]["--flags"](
                    "flags for compilation; must be valid for hipcc.") |
            clara::Help{help} |
            clara::Opt{output, "filename"}
                ["-o"]["--output"](
                    "name of fat-binary output file; the binary format of the "
                    "file is documented at: https://reviews.llvm.org/D13909.") |
            clara::Arg{sources, "a.cpp b.cpp etc."}(
                "inputs for compilation; must contain valid C++ code.") |
            clara::Opt{targets, "gfx803,gfx900 etc."}
                ["-t"]["--targets"](
                    "targets for AMDGPU lowering; must be one of the processors"
                    " with ROCm support from "
                    "https://www.llvm.org/docs/AMDGPUUsage.html#processors.");
    }
}