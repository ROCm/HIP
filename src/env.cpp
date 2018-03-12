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

#include "hip_hcc_internal.h"
#include "trace_helper.h"
#include "env.h"

//---
// Read environment variables.
void ihipReadEnv_I(int* var_ptr, const char* var_name1, const char* var_name2,
                   const char* description) {
    char* env = getenv(var_name1);

    // Check second name if first not defined, used to allow HIP_ or CUDA_ env vars.
    if ((env == NULL) && strcmp(var_name2, "0")) {
        env = getenv(var_name2);
    }

    // Default is set when variable is initialized (at top of this file), so only override if we
    // find an environment variable.
    if (env) {
        long int v = strtol(env, NULL, 0);
        *var_ptr = (int)(v);
    }
    if (HIP_PRINT_ENV) {
        printf("%-30s = %2d : %s\n", var_name1, *var_ptr, description);
    }
}


void ihipReadEnv_S(std::string* var_ptr, const char* var_name1, const char* var_name2,
                   const char* description) {
    char* env = getenv(var_name1);

    // Check second name if first not defined, used to allow HIP_ or CUDA_ env vars.
    if ((env == NULL) && strcmp(var_name2, "0")) {
        env = getenv(var_name2);
    }

    if (env) {
        *static_cast<std::string*>(var_ptr) = env;
    }
    if (HIP_PRINT_ENV) {
        printf("%-30s = %s : %s\n", var_name1, var_ptr->c_str(), description);
    }
}


void ihipReadEnv_Callback(void* var_ptr, const char* var_name1, const char* var_name2,
                          const char* description,
                          std::string (*setterCallback)(void* var_ptr, const char* env)) {
    char* env = getenv(var_name1);

    // Check second name if first not defined, used to allow HIP_ or CUDA_ env vars.
    if ((env == NULL) && strcmp(var_name2, "0")) {
        env = getenv(var_name2);
    }

    std::string var_string = "0";
    if (env) {
        var_string = setterCallback(var_ptr, env);
    }
    if (HIP_PRINT_ENV) {
        printf("%-30s = %s : %s\n", var_name1, var_string.c_str(), description);
    }
}


void tokenize(const std::string& s, char delim, std::vector<std::string>* tokens) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (getline(ss, item, delim)) {
        item.erase(std::remove(item.begin(), item.end(), ' '), item.end());  // remove whitespace.
        tokens->push_back(item);
    }
}

void trim(std::string* s) {
    // trim whitespace from beginning and end:
    const char* t = "\t\n\r\f\v";
    s->erase(0, s->find_first_not_of(t));
    s->erase(s->find_last_not_of(t) + 1);
}

static void ltrim(std::string* s) {
    // trim whitespace from beginning
    const char* t = "\t\n\r\f\v";
    s->erase(0, s->find_first_not_of(t));
}
