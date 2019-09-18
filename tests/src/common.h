#ifndef HIP_COMMON_H
#define HIP_COMMON_H

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <stddef.h>

#define HC __attribute__((hc))

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"


#define passed()                                                                          \
        printf("%sPASSED!%s\n", KGRN, KNRM);                                              \
        exit(0);

#define failed(...)                                                                       \
        printf("%serror: ", KRED);                                                        \
        printf(__VA_ARGS__);                                                              \
        printf("\n");                                                                     \
        printf("error: TEST FAILED\n%s", KNRM);                                           \
        abort();

#define warn(...)                                                                         \
        printf("%swarn: ", KYEL);                                                         \
        printf(__VA_ARGS__);                                                              \
        printf("\n");                                                                     \
        printf("warn: TEST WARNING\n%s", KNRM);

#define HIP_PRINT_STATUS(status)                                                          \
        std::cout << hipGetErrorName(status) << " at line: " << __LINE__ << std::endl;

#define HIPCHECK(error)                                                                                  \
        {                                                                                                \
            hipError_t localError = error;                                                               \
            if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {        \
                 printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, hipGetErrorString(localError),   \
                              localError, #error, __FILE__, __LINE__, KNRM);                             \
                 failed("API returned error code.");                                                     \
            }                                                                                            \
        }

#define HIPASSERT(condition)                                                                             \
        if (!(condition)) {                                                                              \
            failed("%sassertion %s at %s:%d%s \n", KRED, #condition, __FILE__, __LINE__, KNRM);          \
        }


#define HIPCHECK_API(API_CALL, EXPECTED_ERROR)                                                           \
        {                                                                                                \
            hipError_t _e = (API_CALL);                                                                  \
            if (_e != (EXPECTED_ERROR)) {                                                                \
               failed("%sAPI '%s' returned %d(%s) but test expected %d(%s) at %s:%d%s \n", KRED,         \
                       #API_CALL, _e, hipGetErrorName(_e), EXPECTED_ERROR,                               \
                       hipGetErrorName(EXPECTED_ERROR), __FILE__, __LINE__, KNRM);                       \
            }                                                                                            \
        }

#ifdef _WIN64
#include <tchar.h>
#define aligned_alloc _aligned_malloc
#define popen(x,y) _popen(x,y)
#define pclose(x) _pclose(x)
#define setenv(x,y,z) _putenv_s(x,y)
#endif

// standard command-line variables:
extern size_t N;
extern char memsetval;
extern int memsetD32val;
extern int iterations;
extern unsigned blocksPerCU;
extern unsigned threadsPerBlock;
extern int p_gpuDevice;
extern unsigned p_verbose;
extern int p_tests;

#endif //HIP_COMMON_H
