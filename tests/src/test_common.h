/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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



#ifdef __HIP_PLATFORM_HCC
#define TYPENAME(T) typeid(T).name()
#else
#define TYPENAME(T) "?"
#endif



#define passed() \
    printf ("%sPASSED!%s\n",KGRN, KNRM);\
    exit(0);

#define failed(...) \
    printf ("%serror: ", KRED);\
    printf (__VA_ARGS__);\
    printf ("\n");\
    printf ("error: TEST FAILED\n%s", KNRM );\
    abort();


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
            size_t NELEM)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<NELEM; i+=stride) {
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


//---
struct Pinned {
	static const bool isPinned = true;
	static const char *str() { return "Pinned"; };

    static void *Alloc(size_t sizeBytes) 
	{
        void *p; 
        HIPCHECK(hipMallocHost(&p, sizeBytes));
        return p;
    };
};


//---
struct Unpinned 
{
	static const bool isPinned = false;
	static const char *str() { return "Unpinned"; };

    static void *Alloc(size_t sizeBytes) 
	{
        void *p  = malloc (sizeBytes);
        HIPASSERT(p);
		return p;
    };
};



struct Memcpy
{
	static const char *str() { return "Memcpy"; };
};

struct MemcpyAsync
{
	static const char *str() { return "MemcpyAsync"; };
};


template <typename C> struct MemTraits;


template<>
struct MemTraits<Memcpy>
{

    static void Copy(void *dest, const void *src, size_t sizeBytes, hipMemcpyKind kind,  hipStream_t stream) 
	{
		HIPCHECK(hipMemcpy(dest, src, sizeBytes, kind));
	}
};


template<>
struct MemTraits<MemcpyAsync>
{

    static void Copy(void *dest, const void *src, size_t sizeBytes, hipMemcpyKind kind,  hipStream_t stream) 
	{
		HIPCHECK(hipMemcpyAsync(dest, src, sizeBytes, kind, stream));
	}
};

}; // namespace HipTest
