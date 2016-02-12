#include <iostream>
#include <sys/time.h>
#include <stddef.h>

#include "hip_runtime.h"

#define HC __attribute__((hc))


#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define passed() \
    printf ("%sPASSED!%s\n",KGRN, KNRM);\
    exit(0);

#define failed(...) \
    printf ("%serror: ", KRED);\
    printf (__VA_ARGS__);\
    printf ("\n");\
    printf ("error: TEST FAILED\n%s", KNRM );\
    exit(EXIT_FAILURE);


#define HIPCHECK(error) \
{\
    hipError_t localError = error; \
    if (localError != hipSuccess) { \
      printf("%serror: '%s'(%d) at %s:%d%s\n", \
          KRED,hipGetErrorString(localError), localError,\
      __FILE__, __LINE__,KNRM); \
        failed("API returned error code.");\
    }\
}

#define HIPASSERT(condition) \
    if (! (condition) ) { \
        failed("%sassertion %s at %s:%d%s \n", \
               KRED, #condition,\
              __FILE__, __LINE__,KNRM); \
    }

// standard command-line variables:
extern size_t N;
extern char memsetval;
extern int iterations;
extern unsigned blocksPerCU;
extern unsigned threadsPerBlock;
extern int p_gpuDevice;
extern unsigned p_verbose;
extern int p_tests;

namespace HipTest {

// Returns the current system time in microseconds
inline long long get_time()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

double elapsed_time(long long startTimeUs, long long stopTimeUs);

int parseSize(const char *str, size_t *output);
int parseUInt(const char *str, unsigned int *output);
int parseInt(const char *str, int *output);
int parseStandardArguments(int argc, char *argv[], bool failOnUndefinedArg);

unsigned setNumBlocks(unsigned blocksPerCU, unsigned threadsPerBlock, size_t N);


template <typename T>
__global__ void
vectorADD(hipLaunchParm lp,
            const T *A_d,
            const T *B_d,
            T *C_d,
            size_t N)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<N; i+=stride) {
		C_d[i] = A_d[i] + B_d[i];
	}
}


template <typename T>
void initArrays(T **A_d, T **B_d, T **C_d,
                T **A_h, T **B_h, T **C_h, 
                size_t N, bool usePinnedHost=false) 
{
    size_t Nbytes = N*sizeof(T);

    if (A_d) {
        HIPCHECK ( hipMalloc(A_d, Nbytes) );
    }
    if (B_d) {
        HIPCHECK ( hipMalloc(B_d, Nbytes) );
    }
    if (C_d) {
        HIPCHECK ( hipMalloc(C_d, Nbytes) );
    }

    if (usePinnedHost) {
        if (A_h) {
            HIPCHECK ( hipMallocHost(A_h, Nbytes) );
        }
        if (B_h) {
            HIPCHECK ( hipMallocHost(B_h, Nbytes) );
        }
        if (C_h) {
            HIPCHECK ( hipMallocHost(C_h, Nbytes) );
        }
    } else {
        if (A_h) {
            *A_h = (T*)malloc(Nbytes);
            HIPASSERT(*A_h != NULL);
        }
        
        if (B_h) {
            *B_h = (T*)malloc(Nbytes);
            HIPASSERT(*B_h != NULL);
        }

        if (C_h) {
            *C_h = (T*)malloc(Nbytes);
            HIPASSERT(*C_h != NULL);
        }
    }


    // Initialize the host data:
    for (size_t i=0; i<N; i++) {
        if (A_h) 
            (*A_h)[i] = 3.146f + i; // Pi
        if (B_h) 
            (*B_h)[i] = 1.618f + i; // Phi
    }
}


template <typename T>
void freeArrays(T *A_d, T *B_d, T *C_d,
                T *A_h, T *B_h, T *C_h, bool usePinnedHost) 
{
    if (A_d) {
        HIPCHECK ( hipFree(A_d) );
    }
    if (B_d) {
        HIPCHECK ( hipFree(B_d) );
    }
    if (C_d) {
        HIPCHECK ( hipFree(C_d) );
    }

    if (usePinnedHost) {
        if (A_h) {
            HIPCHECK (hipFreeHost(A_h));
        }
        if (B_h) {
            HIPCHECK (hipFreeHost(B_h));
        }
        if (C_h) {
            HIPCHECK (hipFreeHost(C_h));
        }
    } else {
        if (A_h) {
            free (A_h);
        }
        if (B_h) {
            free (B_h);
        }
        if (C_h) {
            free (C_h);
        }
    }
    
}


// Assumes C_h contains vector add of A_h + B_h
// Calls the test "failed" macro if a mismatch is detected.
template <typename T>
void checkVectorADD(T* A_h, T* B_h, T* result_H, size_t N, bool expectMatch=true)
{
    size_t  mismatchCount = 0;
    size_t  firstMismatch = 0;
    size_t  mismatchesToPrint = 10;
    for (size_t i=0; i<N; i++) {
        T expected = A_h[i] + B_h[i];
        if (result_H[i] != expected) {
            if (mismatchCount == 0) {
                firstMismatch = i;
            }
            mismatchCount++;
            if ((mismatchCount <= mismatchesToPrint) && expectMatch) {
                std::cout << "At " << i << "  Computed:" << result_H[i] << ", expected:" << expected << std::endl;
            }
        }
    }

    if (expectMatch) {
        if (mismatchCount) {
            failed("%zu mismatches ; first at index:%zu\n", mismatchCount, firstMismatch);
        }
    } else {
        if (mismatchCount == 0) {
            failed("expected mismatches but did not detect any!");
        }
    }

}

}; // namespace HipTest
