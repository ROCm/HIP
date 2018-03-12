#include "lpl.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clara;
using namespace hip_impl;
using namespace std;

int main(int argc, char** argv) {
    try {
        if (!hipcc_and_lpl_colocated()) {
            throw runtime_error{"The LPL executable and hipcc must be in the same directory."};
        }

        bool help = false;
        string flags;
        string output;
        vector<string> sources;
        string targets;

        auto cmd = cmdline_parser(help, sources, targets, flags, output);

        const auto r = cmd.parse(Args{argc, argv});

        if (!r) throw runtime_error{r.errorMessage()};

        if (help)
            cout << cmd << endl;
        else {
            if (sources.empty()) throw runtime_error{"No inputs specified."};

            auto tmp = tokenize_targets(targets);
            if (tmp.empty()) {
                tmp.assign(amdgpu_targets().cbegin(), amdgpu_targets().cend());
            } else
                validate_targets(tmp);

            if (output.empty())
                for (auto&& x : tmp) output += x;

            generate_fat_binary(sources, tmp, flags, output);
        }
    } catch (const exception& ex) {
        cerr << ex.what() << endl;

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}