#include "ca.hpp"

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
        bool help = false;
        vector<string> inputs;
        string targets;

        auto cmd = cmdline_parser(help, inputs, targets);

        const auto r = cmd.parse(Args{argc, argv});

        if (!r) throw runtime_error{r.errorMessage()};

        if (help)
            cout << cmd << endl;
        else {
            if (inputs.empty()) throw runtime_error{"No inputs specified."};

            validate_inputs(inputs);

            auto tmp = tokenize_targets(targets);
            if (tmp.empty()) {
                tmp.assign(amdgpu_targets().cbegin(), amdgpu_targets().cend());
            } else
                validate_targets(tmp);

            extract_code_objects(inputs, tmp);
        }
    } catch (const exception& ex) {
        cerr << ex.what() << endl;

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}