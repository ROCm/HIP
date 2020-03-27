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
#include "code_object_bundle.inl"
#include "../include/hip/hcc_detail/elfio/elfio.hpp"
#include "../include/hip/hcc_detail/program_state.hpp"

#include "../lpl_ca/pstreams/pstream.h"

#include <hsa/hsa.h>

#include <cxxabi.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <iostream>
#include <sys/stat.h>

extern "C" const char* hiprtcGetErrorString(hiprtcResult x)
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

namespace hip_impl {
inline bool create_directory(const std::string& path) {
    mode_t mode = 0755;
    int ret = mkdir(path.c_str(), mode);
    if (ret == 0) return true;
    return false;
}

inline bool fileExists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}
}  // namespace hip_impl

namespace
{
    char* demangle(const char* x)
    {
        if (!x) return nullptr;

        int s{};
        char* tmp = abi::__cxa_demangle(x, nullptr, nullptr, &s);

        if (s != 0) return nullptr;

        return tmp;
    }
} // Unnamed namespace.

namespace
{
    struct Symbol {
        std::string name;
        ELFIO::Elf64_Addr value = 0;
        ELFIO::Elf_Xword size = 0;
        ELFIO::Elf_Half sect_idx = 0;
        std::uint8_t bind = 0;
        std::uint8_t type = 0;
        std::uint8_t other = 0;
    };

    inline
    Symbol read_symbol(const ELFIO::symbol_section_accessor& section,
                    unsigned int idx) {
        assert(idx < section.get_symbols_num());

        Symbol r;
        section.get_symbol(
            idx, r.name, r.value, r.size, r.bind, r.type, r.sect_idx, r.other);

        return r;
    }
} // Unnamed namespace.

struct _hiprtcProgram {
    // DATA - STATICS
    static std::vector<std::unique_ptr<_hiprtcProgram>> programs;
    static std::mutex mtx;

    // DATA
    std::vector<std::pair<std::string, std::string>> headers;
    std::vector<std::pair<std::string, std::string>> names;
    std::vector<std::string> loweredNames;
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

        char* demangled = demangle(name.c_str());
        name.assign(demangled == nullptr ? "" : demangled);
        free(demangled);

        if (name.empty()) return name;

        if (name.find("void ") == 0) name.erase(0, strlen("void "));

        auto dx{name.find_first_of("(<")};

        if (dx == string::npos) return name;

        if (name[dx] == '<') {
            auto cnt{1u};
            do {
                ++dx;
                cnt += (name[dx] == '<') ? 1 : ((name[dx] == '>') ? -1 : 0);
            } while (cnt);

            name.erase(++dx);
        }
        else name.erase(dx);

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
        return std::find_if(programs.cbegin(), programs.cend(),
                            [=](const std::unique_ptr<_hiprtcProgram>& x) {
            return x.get() == p;
        }) != programs.cend();
    }

    // MANIPULATORS
    bool compile(const std::vector<std::string>& args)
    {
        using namespace ELFIO;
        using namespace redi;
        using namespace std;

        ipstream compile{args.front(), args, pstreambuf::pstderr};

        constexpr const auto tmp_size{1024u};
        char tmp[tmp_size]{};
        while (!compile.eof()) {
            log.append(tmp, tmp + compile.readsome(tmp, tmp_size));
        }

        compile.close();

        if (compile.rdbuf()->exited() &&
            compile.rdbuf()->status() != EXIT_SUCCESS) return false;

        elfio reader;
        if (!reader.load(args.back())) return false;

        const auto it{find_if(reader.sections.begin(), reader.sections.end(),
                              [](const section* x) {
            return x->get_name() == ".kernel";
        })};

        if (it == reader.sections.end()) return false;

        hip_impl::Bundled_code_header h{(*it)->get_data()};

        if (bundles(h).empty()) return false;

        elf.assign(bundles(h).back().blob.cbegin(),
                   bundles(h).back().blob.cend());

        return true;
    }

    bool readLoweredNames()
    {
        using namespace ELFIO;
        using namespace hip_impl;
        using namespace std;

        if (names.empty()) return true;

        istringstream blob{string{elf.cbegin(), elf.cend()}};

        elfio reader;

        if (!reader.load(blob)) return false;

        const auto it{find_if(reader.sections.begin(), reader.sections.end(),
                              [](const section* x) {
            return x->get_type() == SHT_SYMTAB;
        })};

        ELFIO::symbol_section_accessor symbols{reader, *it};

        auto n{symbols.get_symbols_num()};

        if (n < loweredNames.size()) return false;

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

            loweredNames[distance(names.cbegin(), it)] = tmp.name;
        }

        return true;
    }

    void replaceExtension(std::string& fileName, const std::string &ext) const {
        auto res = fileName.rfind('.');
        auto sloc = fileName.rfind('/'); // slash location
        if (res != std::string::npos && (res > sloc || sloc == std::string::npos)) {
            fileName.replace(fileName.begin() + res, fileName.end(), ext);
        } else {
            fileName += ext;
        }
    }

    // ACCESSORS
    std::string writeTemporaryFiles(
        const std::string& programFolder) const
    {
        using namespace std;

        vector<future<void>> fut{headers.size()};
        transform(headers.cbegin(), headers.cend(), begin(fut),
                  [&](const pair<string, string>& x) {
            return async([&]() {
                ofstream h{programFolder + '/' + x.first};
                h.write(x.second.data(), x.second.size());
            });
        });

        auto tmp{(programFolder + '/' + name)};
        replaceExtension(tmp, ".cpp");
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

extern "C" hiprtcResult hiprtcAddNameExpression(hiprtcProgram p, const char* n)
{
    if (!n) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (p->compiled) return HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;

    const auto id{p->names.size()};

    p->names.emplace_back(n, n);
    p->loweredNames.emplace_back();

    if (p->names.back().second.back() == ')') {
        p->names.back().second.pop_back();
        p->names.back().second.erase(0, p->names.back().second.find('('));
    }
    if (p->names.back().second.front() == '&') {
        p->names.back().second.erase(0, 1);
    }

    const auto var{"__hiprtc_" + std::to_string(id)};
    p->source.append("\nextern \"C\" constexpr auto " + var + " = " + n + ';');

    return HIPRTC_SUCCESS;
}

namespace
{
    class Unique_temporary_path {
        // DATA
        std::string path_{};
    public:
        // CREATORS
        Unique_temporary_path() : path_{std::tmpnam(nullptr)}
        {
            while (hip_impl::fileExists(path_)) {
                path_ = std::tmpnam(nullptr);
            }
        }

        Unique_temporary_path(const Unique_temporary_path&) = default;
        Unique_temporary_path(Unique_temporary_path&&) = default;

        ~Unique_temporary_path() noexcept
        {
            std::string s("rm -r " + path_);
            system(s.c_str());
        }

        // MANIPULATORS
        Unique_temporary_path& operator=(
            const Unique_temporary_path&) = default;
        Unique_temporary_path& operator=(Unique_temporary_path&&) = default;

        // ACCESSORS
        const std::string& path() const noexcept
        {
            return path_;
        }
    };
} // Unnamed namespace.

namespace
{
    const std::string& defaultTarget()
    {
        using namespace std;

        static string r{"gfx900"};
        static once_flag f{};

        call_once(f, []() {
            static hsa_agent_t a{};
            hsa_iterate_agents([](hsa_agent_t x, void*) {
                hsa_device_type_t t{};
                hsa_agent_get_info(x, HSA_AGENT_INFO_DEVICE, &t);

                if (t != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;

                a = x;

                return HSA_STATUS_INFO_BREAK;
            }, nullptr);

            if (!a.handle) return;

            hsa_agent_iterate_isas(a, [](hsa_isa_t x, void*){
                uint32_t n{};
                hsa_isa_get_info_alt(x, HSA_ISA_INFO_NAME_LENGTH, &n);

                if (n == 0) return HSA_STATUS_SUCCESS;

                r.resize(n);
                hsa_isa_get_info_alt(x, HSA_ISA_INFO_NAME, &r[0]);

                r.erase(0, r.find("gfx"));

                return HSA_STATUS_INFO_BREAK;
            }, nullptr);
        });

        return r;
    }

    inline
    void handleTarget(std::vector<std::string>& args)
    {
        using namespace std;

        bool hasTarget{false};
        for (auto&& x : args) {
            const auto dx{x.find("--gpu-architecture")};
            const auto dy{(dx == string::npos) ? x.find("-arch")
                                               : string::npos};

            if (dx == dy) continue;

            x.replace(0, x.find('=', min(dx, dy)), "--amdgpu-target");
            hasTarget = true;

            break;
        }
        if (!hasTarget) args.push_back("--amdgpu-target=" + defaultTarget());
    }
} // Unnamed namespace.

extern "C" hiprtcResult hiprtcCompileProgram(hiprtcProgram p, int n, const char** o)
{
    using namespace std;

    if (n && !o) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (p->compiled) return HIPRTC_ERROR_COMPILATION;

    static const string hipcc{
        getenv("HIP_PATH") ? (getenv("HIP_PATH") + string{"/bin/hipcc"})
                           : "/opt/rocm/bin/hipcc"};

    if (!hip_impl::fileExists(hipcc)) {
        return HIPRTC_ERROR_INTERNAL_ERROR;
    }

    Unique_temporary_path tmp{};
    hip_impl::create_directory(tmp.path());

    const auto src{p->writeTemporaryFiles(tmp.path())};

    vector<string> args{hipcc, "-shared"};
    if (n) args.insert(args.cend(), o, o + n);

    handleTarget(args);

    args.emplace_back(src);
    args.emplace_back("-o");
    args.emplace_back(tmp.path() + '/' + "hiprtc.out");

    if (!p->compile(args)) return HIPRTC_ERROR_INTERNAL_ERROR;
    if (!p->readLoweredNames()) return HIPRTC_ERROR_INTERNAL_ERROR;

    p->compiled = true;

    return HIPRTC_SUCCESS;
}

extern "C" hiprtcResult hiprtcCreateProgram(hiprtcProgram* p, const char* src,
                                 const char* name, int n, const char** hdrs,
                                 const char** incs)
{
    using namespace std;

    if (!p) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (n < 0) return HIPRTC_ERROR_INVALID_INPUT;
    if (n && (!hdrs || !incs)) return HIPRTC_ERROR_INVALID_INPUT;

    vector<pair<string, string>> h;
    for (auto i = 0; i != n; ++i) h.emplace_back(incs[i], hdrs[i]);

    *p = _hiprtcProgram::make(src, name ? name : "default_name", move(h));

    return HIPRTC_SUCCESS;
}

extern "C" hiprtcResult hiprtcDestroyProgram(hiprtcProgram* p)
{
    if (!p) return HIPRTC_SUCCESS;

    return _hiprtcProgram::destroy(*p);
}

extern "C" hiprtcResult hiprtcGetLoweredName(hiprtcProgram p, const char* n,
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

    *ln = p->loweredNames[distance(p->names.cbegin(), it)].c_str();

    return HIPRTC_SUCCESS;
}

extern "C" hiprtcResult hiprtcGetProgramLog(hiprtcProgram p, char* l)
{
    if (!l) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (!p->compiled) return HIPRTC_ERROR_INVALID_PROGRAM;

    l = std::copy_n(p->log.data(), p->log.size(), l);
    *l = '\0';

    return HIPRTC_SUCCESS;
}

extern "C" hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram p, std::size_t* sz)
{
    if (!sz) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (!p->compiled) return HIPRTC_ERROR_INVALID_PROGRAM;

    *sz = p->log.empty() ? 0 : p->log.size() + 1;

    return HIPRTC_SUCCESS;
}

extern "C" hiprtcResult hiprtcGetCode(hiprtcProgram p, char* c)
{
    if (!c) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (!p->compiled) return HIPRTC_ERROR_INVALID_PROGRAM;

    std::copy_n(p->elf.data(), p->elf.size(), c);

    return HIPRTC_SUCCESS;
}

extern "C" hiprtcResult hiprtcGetCodeSize(hiprtcProgram p, std::size_t* sz)
{
    if (!sz) return HIPRTC_ERROR_INVALID_INPUT;
    if (!isValidProgram(p)) return HIPRTC_ERROR_INVALID_PROGRAM;
    if (!p->compiled) return HIPRTC_ERROR_INVALID_PROGRAM;

    *sz = p->elf.size();

    return HIPRTC_SUCCESS;
}

extern "C" hiprtcResult hiprtcVersion(int* major, int* minor)
{
  if (major == nullptr || minor == nullptr) {
    return HIPRTC_ERROR_INVALID_INPUT;
  }

  *major = 9;
  *minor = 0;

  return HIPRTC_SUCCESS;
}
