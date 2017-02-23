#include "hip_hcc.h"
#include "trace_helper.h"
#include "env.h"

//---
// Read environment variables.
void ihipReadEnv_I(int *var_ptr, const char *var_name1, const char *var_name2, const char *description)
{
    char * env = getenv(var_name1);

    // Check second name if first not defined, used to allow HIP_ or CUDA_ env vars.
    if ((env == NULL) && strcmp(var_name2, "0")) {
        env = getenv(var_name2);
    }

    // Default is set when variable is initialized (at top of this file), so only override if we find
    // an environment variable.
    if (env) {
        long int v = strtol(env, NULL, 0);
        *var_ptr = (int) (v);
    }
    if (HIP_PRINT_ENV) {
        printf ("%-30s = %2d : %s\n", var_name1, *var_ptr, description);
    }
}


void ihipReadEnv_S(std::string *var_ptr, const char *var_name1, const char *var_name2, const char *description)
{
    char * env = getenv(var_name1);

    // Check second name if first not defined, used to allow HIP_ or CUDA_ env vars.
    if ((env == NULL) && strcmp(var_name2, "0")) {
        env = getenv(var_name2);
    }

    if (env) {
        *static_cast<std::string*>(var_ptr) = env;
    }
    if (HIP_PRINT_ENV) {
        printf ("%-30s = %s : %s\n", var_name1, var_ptr->c_str(), description);
    }
}


void ihipReadEnv_Callback(void *var_ptr, const char *var_name1, const char *var_name2, const char *description, std::string (*setterCallback)(void * var_ptr, const char * env))
{
    char * env = getenv(var_name1);

    // Check second name if first not defined, used to allow HIP_ or CUDA_ env vars.
    if ((env == NULL) && strcmp(var_name2, "0")) {
        env = getenv(var_name2);
    }

    std::string var_string = "0";
    if (env) {
        var_string = setterCallback(var_ptr, env);
    }
    if (HIP_PRINT_ENV) {
        printf ("%-30s = %s : %s\n", var_name1, var_string.c_str(), description);
    }
}




void tokenize(const std::string &s, char delim, std::vector<std::string> *tokens)
{
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (getline(ss, item, delim)) {
        item.erase (std::remove (item.begin(), item.end(), ' '), item.end()); // remove whitespace.
        tokens->push_back(item);
    }
}

void trim(std::string *s)
{
    // trim whitespace from beginning and end:
    const char *t =  "\t\n\r\f\v";
    s->erase(0, s->find_first_not_of(t));
    s->erase(s->find_last_not_of(t)+1);
}

static void ltrim(std::string *s)
{
    // trim whitespace from beginning
    const char *t =  "\t\n\r\f\v";
    s->erase(0, s->find_first_not_of(t));
}
