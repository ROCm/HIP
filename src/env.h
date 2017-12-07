#pragma once

extern void HipReadEnv();


#define READ_ENV_I(_build, _ENV_VAR, _ENV_VAR2, _description)                                      \
    ihipReadEnv_I(&_ENV_VAR, #_ENV_VAR, #_ENV_VAR2, _description);

#define READ_ENV_S(_build, _ENV_VAR, _ENV_VAR2, _description)                                      \
    ihipReadEnv_S(&_ENV_VAR, #_ENV_VAR, #_ENV_VAR2, _description);

#define READ_ENV_C(_build, _ENV_VAR, _ENV_VAR2, _description, _callback)                           \
    ihipReadEnv_Callback(&_ENV_VAR, #_ENV_VAR, #_ENV_VAR2, _description, _callback);


extern void ihipReadEnv_I(int* var_ptr, const char* var_name1, const char* var_name2,
                          const char* description);
extern void ihipReadEnv_S(std::string* var_ptr, const char* var_name1, const char* var_name2,
                          const char* description);
extern void ihipReadEnv_Callback(void* var_ptr, const char* var_name1, const char* var_name2,
                                 const char* description,
                                 std::string (*setterCallback)(void* var_ptr, const char* env));


// String functions:
extern void trim(std::string* s);
extern void tokenize(const std::string& s, char delim, std::vector<std::string>* tokens);
