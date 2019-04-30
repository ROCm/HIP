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

#include "../include/hip/hiprtc.h"
#include "../include/hip/hcc_detail/code_object_bundle.hpp"
#include "../include/hip/hcc_detail/elfio/elfio.hpp"
#include "../include/hip/hcc_detail/program_state.hpp"

#include "../lpl_ca/pstreams/pstream.h"

#include <cxxabi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <iostream>

const char* hiprtcGetErrorString(hiprtcResult x)
{
    switch (x) {
    case HIPRTC_SUCCESS:
        return "HIPRTC_SUCCESS";
    case HIPRTC_ERROR_OUT_OF_MEMORY:
        return "HIPRTC_ERROR_OUT_OF_MEMORY";
    case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:
        return "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case HIPRTC_ERROR_INVALID_INPUT:
        return "HIPRTC_ERROR_INVALID_INPUT";
    case HIPRTC_ERROR_INVALID_PROGRAM:
        return "HIPRTC_ERROR_INVALID_PROGRAM";
    case HIPRTC_ERROR_INVALID_OPTION:
        return "HIPRTC_ERROR_INVALID_OPTION";
    case HIPRTC_ERROR_COMPILATION:
        return "HIPRTC_ERROR_COMPILATION";
    case HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE:
        return "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
        return "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
        return "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
        return "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case HIPRTC_ERROR_INTERNAL_ERROR:
        return "HIPRTC_ERROR_INTERNAL_ERROR";
    default: throw std::logic_error{"Invalid HIPRTC result."};
    };
}

struct _hiprtcProgram {
    // DATA - STATICS
    static std::vector<std::unique_ptr<_hiprtcProgram>> programs;
    static std::mutex mtx;

    // DATA
    std::vector<std::pair<std::string, std::string>> headers;
    std::vector<std::pair<std::string, std::string>> names;
    std::vector<std::string> lowered_names;
    std::vector<char> elf;
    std::string source;
    std::string name;
    std::string log;
    bool compiled;

    // STATICS
    static
    hiprtcResult destroy(_hiprtcProgram* p)
    {
        using namespace std;

        lock_guard<mutex> lck{mtx};

        const auto it{find_if(programs.cbegin(), programs.cend(),
                              [=](const unique_ptr<_hiprtcProgram>& x) {
            return x.get() == p;
        })};

        if (it == programs.cend()) return HIPRTC_ERROR_INVALID_PROGRAM;

        return HIPRTC_SUCCESS;
    }

    static
    std::string handleMangledName(std::string name)
    {
        using namespace std;

        name = hip_impl::demangle(name.c_str());

        if (name.empty()) return name;

        if (name.find("void ") == 0) name.erase(0, strlen("void "));

        auto dx{name.find('<')};
        if (dx != string::npos) {
            auto cnt{1u};
            do {
                ++dx;
                cnt += (name[dx] == '<') ? 1 : ((name[dx] == '>') ? -1 : 0);
            } while (cnt);

            name.erase(++dx);
        }
        else if (name.find('(') != string::npos) name.erase(name.find('('));

        return name;
    }

    static
    _hiprtcProgram* make(std::string s, std::string n,
                         std::vector<std::pair<std::string, std::string>> h)
    {
        using namespace std;

        unique_ptr<_hiprtcProgram> tmp{new _hiprtcProgram{move(h), {}, {}, {},
                                                          move(s), move(n), {},
                                                          false}};

        lock_guard<mutex> lck{mtx};

        programs.push_back(move(tmp));

        return programs.back().get();
    }

    static
    bool isValid(_hiprtcProgram* p) noexcept
    {
        using namespace std;

        return find_if(programs.cbegin(), programs.cend(),
                       [=](const unique_ptr<_hiprtcProgram>& x) {
            return x.get() == p;
        }) != programs.cend();
    }

    // ACCESSORS
    bool compile(const std::vector<std::string>& args,
                 const std::experimental::filesystem::path& program_folder)
    {
        using namespace std;

        redi::pstream compile{args.front(), args};

        compile.close();

        ostringstream{log} << compile.rdbuf();

        if (compile.rdbuf()->status() != EXIT_SUCCESS) return false;

        cerr << log << endl;

        ifstream in{args.back()};
        elf.resize(experimental::filesystem::file_size(args.back()));
        in.read(elf.data(), elf.size());

        return true;
    }

    bool read_lowered_names()
    {
        using namespace hip_impl;
        using namespace std;

        if (names.empty()) return true;

        Bundled_code_header h{elf.data()};
        istringstream blob{string{bundles(h).back().blob.cbegin(),
                                  bundles(h).back().blob.cend()}};

        ELFIO::elfio reader;

        if (!reader.load(blob)) return false;

        const auto it{find_section_if(reader, [](const ELFIO::section* x) {
            return x->get_type() == SHT_SYMTAB;
        })};

        ELFIO::symbol_section_accessor symbols{reader, it};

        lowered_names.resize(names.size());

        auto n{symbols.get_symbols_num()};
        while (n--) {
            const auto tmp{read_symbol(symbols, n)};

            auto it{find_if(names.cbegin(), names.cend(),
                            [&](const pair<string, string>& x) {
                return x.second == tmp.name;
            })};

            if (it == names.cend()) {
                const auto name{handleMangledName(tmp.name)};

                if (name.empty()) continue;

                it = find_if(names.cbegin(), names.cend(),
                             [&](const pair<string, string>& x) {
                    return x.second == name;
                });

                if (it == names.cend()) continue;
            }

            lowered_names[distance(names.cbegin(), it)] = tmp.name;
        }

        return true;
    }

    std::experimental::filesystem::path write_temporary_files(
        const std::experimental::filesystem::path& program_folder)
    {
        using namespace std;

        vector<future<void>> fut{headers.size()};
        transform(headers.cbegin(), headers.cend(), begin(fut),
                  [&](const pair<string, string>& x) {
            return async([&]() {
                ofstream h{program_folder / x.first};
                h.write(x.second.data(), x.second.size());
            });
        });

        auto tmp{(program_folder / name).replace_extension(".cpp")};
        ofstream{tmp}.write(source.data(), source.size());

        return tmp;
    }


};
std::vector<std::unique_ptr<_hiprtcProgram>> _hiprtcProgram::programs{};
std::mutex _hiprtcProgram::mtx{};

namespace
{
    inline
    bool isValidProgram(const hiprtcProgram p)
    {
        if (!p) return false;

        std::lock_guard<std::mutex> lck{_hiprtcProgram::mtx};

        return _hiprtcProgram::isValid(p);
    }
} // Unnamed namespace.

hiprtcResult hiprtcAddNameExpression(hiprtcProgram p, const char* n)
{
    if (!n) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (p->compiled) return HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;

    const auto id{p->names.size()};

    p->names.emplace_back(n, n);

    if (p->names.back().second.back() == ')') {
        p->names.back().second.pop_back();
        p->names.back().second.erase(0, p->names.back().second.find('('));
    }
    if (p->names.back().second.front() == '&') p->names.back().second.erase(0, 1);

    const auto var{"__hiprtc_" + std::to_string(id)};
    p->source.append("\nextern \"C\" constexpr auto " + var + " = " + n + ';');

    return HIPRTC_SUCCESS;
}

namespace
{
    class Unique_temporary_path {
        // DATA
        std::experimental::filesystem::path path_{};
    public:
        // CREATORS
        Unique_temporary_path() : path_{std::tmpnam(nullptr)}
        {
            while (std::experimental::filesystem::exists(path_)) {
                path_ = std::tmpnam(nullptr);
            }
        }
        Unique_temporary_path(const std::string& extension)
            : Unique_temporary_path{}
        {
            path_.replace_extension(extension);
        }

        Unique_temporary_path(const Unique_temporary_path&) = default;
        Unique_temporary_path(Unique_temporary_path&&) = default;

        ~Unique_temporary_path() noexcept
        {
            std::experimental::filesystem::remove_all(path_);
        }

        // MANIPULATORS
        Unique_temporary_path& operator=(
            const Unique_temporary_path&) = default;
        Unique_temporary_path& operator=(Unique_temporary_path&&) = default;

        // ACCESSORS
        const std::experimental::filesystem::path& path() const noexcept
        {
            return path_;
        }
    };
} // Unnamed namespace.

namespace hip_impl
{
    inline
    std::string demangle(const char* x)
    {
        if (!x) return {};

        int s{};
        std::unique_ptr<char, decltype(std::free)*> tmp{
            abi::__cxa_demangle(x, nullptr, nullptr, &s), std::free};

        if (s != 0) return {};

        return tmp.get();
    }
} // Namespace hip_impl.

namespace
{
    inline
    void handle_target(std::vector<std::string>& args)
    {
        using namespace std;

        bool has_target{false};
        for (auto&& x : args) {
            const auto dx{x.find("--gpu-architecture")};
            const auto dy{(dx == string::npos) ? x.find("-arch")
                                               : string::npos};

            if (dx == dy) continue;

            x.replace(0, x.find('=', min(dx, dy)), "--targets");
            has_target = true;

            break;
        }
        if (!has_target) args.emplace_back("--targets=gfx900");
    }
} // Unnamed namespace.

hiprtcResult hiprtcCompileProgram(hiprtcProgram p, int n, const char** o)
{
    using namespace std;

    if (n && !o) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (p->compiled) return HIPRTC_ERROR_COMPILATION;

    static const string hipcc{
        getenv("HIP_PATH") ? (getenv("HIP_PATH") + string{"/bin/hipcc"})
                           : "/opt/rocm/bin/hipcc"};

    if (!experimental::filesystem::exists(hipcc)) {
        return HIPRTC_ERROR_INTERNAL_ERROR;
    }

    Unique_temporary_path tmp{};
    experimental::filesystem::create_directory(tmp.path());

    const auto src{p->write_temporary_files(tmp.path())};

    vector<string> args{hipcc, "--genco"};
    if (n) args.insert(args.cend(), o, o + n);

    handle_target(args);

    args.emplace_back(src);
    args.emplace_back("-o");
    args.emplace_back(tmp.path() / "hiprtc.out");

    const auto compile{p->compile(args, tmp.path())};

    if (!p->compile(args, tmp.path())) return HIPRTC_ERROR_INTERNAL_ERROR;
    if (!p->read_lowered_names()) return HIPRTC_ERROR_INTERNAL_ERROR;

    p->compiled = true;

    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcCreateProgram(hiprtcProgram* p, const char* src,
                                 const char* name, int n_hdr, const char** hdrs,
                                 const char** inc_names)
{
    using namespace std;

    if (!p) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (n_hdr < 0) return HIPRTC_ERROR_INVALID_INPUT;
    if (n_hdr && (!hdrs || !inc_names)) return HIPRTC_ERROR_INVALID_INPUT;

    string s{src};
    string n{name ? name : "default_name"};
    vector<pair<string, string>> h;

    for (auto i = 0; i != n_hdr; ++i) h.emplace_back(inc_names[i], hdrs[i]);

    *p = _hiprtcProgram::make(move(s), move(n), move(h));

    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcDestroyProgram(hiprtcProgram* p)
{
    if (!p) return HIPRTC_SUCCESS;

    return _hiprtcProgram::destroy(*p);
}

hiprtcResult hiprtcGetLoweredName(hiprtcProgram p, const char* n,
                                  const char** ln)
{
    using namespace std;

    if (!n || !ln) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (!p->compiled) return HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION;

    const auto it{find_if(p->names.cbegin(), p->names.cend(),
                          [=](const pair<string, string>& x) {
        return x.first == n;
    })};

    if (it == p->names.cend()) return HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID;

    *ln = p->lowered_names[distance(p->names.cbegin(), it)].c_str();

    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram p, char* l)
{
    if (!l) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (!p->compiled) return HIPRTC_ERROR_INVALID_PROGRAM;

    l = std::copy_n(p->log.data(), p->log.size(), l);
    *l = '\0';

    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram p, std::size_t* sz)
{
    if (!sz) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (!p->compiled) return HIPRTC_ERROR_INVALID_PROGRAM;

    *sz = p->log.size() + 1;

    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetCode(hiprtcProgram p, char* c)
{
    if (!c) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (!p->compiled) return HIPRTC_ERROR_INVALID_PROGRAM;

    std::copy_n(p->elf.data(), p->elf.size(), c);

    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetCodeSize(hiprtcProgram p, std::size_t* sz)
{
    if (!sz) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (!p->compiled) return HIPRTC_ERROR_INVALID_PROGRAM;

    *sz = p->elf.size();

    return HIPRTC_SUCCESS;
}