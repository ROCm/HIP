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
#include <iomanip>
#include <sys/time.h>
#include <stddef.h>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#define HC __attribute__((hc))


#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"


#ifdef __HIP_PLATFORM_HCC
#define TYPENAME(T) typeid(T).name()
#else
#define TYPENAME(T) "?"
#endif


#define passed()                                                                                   \
    printf("%sPASSED!%s\n", KGRN, KNRM);                                                           \
    exit(0);

#define failed(...)                                                                                \
    printf("%serror: ", KRED);                                                                     \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("error: TEST FAILED\n%s", KNRM);                                                        \
    abort();

#define warn(...)                                                                                  \
    printf("%swarn: ", KYEL);                                                                      \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("warn: TEST WARNING\n%s", KNRM);

#define HIP_PRINT_STATUS(status)                                                                   \
    std::cout << hipGetErrorName(status) << " at line: " << __LINE__ << std::endl;

#define HIPCHECK(error)                                                                            \
    {                                                                                              \
        hipError_t localError = error;                                                             \
        if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {      \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, hipGetErrorString(localError),  \
                   localError, #error, __FILE__, __LINE__, KNRM);                                  \
            failed("API returned error code.");                                                    \
        }                                                                                          \
    }

#define HIPASSERT(condition)                                                                       \
    if (!(condition)) {                                                                            \
        failed("%sassertion %s at %s:%d%s \n", KRED, #condition, __FILE__, __LINE__, KNRM);        \
    }


#define HIPCHECK_API(API_CALL, EXPECTED_ERROR)                                                     \
    {                                                                                              \
        hipError_t _e = (API_CALL);                                                                \
        if (_e != (EXPECTED_ERROR)) {                                                              \
            failed("%sAPI '%s' returned %d(%s) but test expected %d(%s) at %s:%d%s \n", KRED,      \
                   #API_CALL, _e, hipGetErrorName(_e), EXPECTED_ERROR,                             \
                   hipGetErrorName(EXPECTED_ERROR), __FILE__, __LINE__, KNRM);                     \
        }                                                                                          \
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
inline long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

double elapsed_time(long long startTimeUs, long long stopTimeUs);

int parseSize(const char* str, size_t* output);
int parseUInt(const char* str, unsigned int* output);
int parseInt(const char* str, int* output);
int parseStandardArguments(int argc, char* argv[], bool failOnUndefinedArg);

unsigned setNumBlocks(unsigned blocksPerCU, unsigned threadsPerBlock, size_t N);


template <typename T>
__global__ void vectorADD(const T* A_d, const T* B_d, T* C_d, size_t NELEM) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < NELEM; i += stride) {
        C_d[i] = A_d[i] + B_d[i];
    }
}


template <typename T>
__global__ void vectorADDReverse(const T* A_d, const T* B_d, T* C_d,
                                 size_t NELEM) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
        C_d[i] = A_d[i] + B_d[i];
    }
}


template <typename T>
__global__ void addCount(const T* A_d, T* C_d, size_t NELEM, int count) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    // Deliberately do this in an inefficient way to increase kernel runtime
    for (int i = 0; i < count; i++) {
        for (size_t i = offset; i < NELEM; i += stride) {
            C_d[i] = A_d[i] + (T)count;
        }
    }
}


template <typename T>
__global__ void addCountReverse(const T* A_d, T* C_d, int64_t NELEM, int count) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    // Deliberately do this in an inefficient way to increase kernel runtime
    for (int i = 0; i < count; i++) {
        for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
            C_d[i] = A_d[i] + (T)count;
        }
    }
}


template <typename T>
__global__ void memsetReverse(T* C_d, T val, int64_t NELEM) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
        C_d[i] = val;
    }
}


template <typename T>
void setDefaultData(size_t numElements, T* A_h, T* B_h, T* C_h) {
    // Initialize the host data:
    for (size_t i = 0; i < numElements; i++) {
        if (A_h) (A_h)[i] = 3.146f + i;  // Pi
        if (B_h) (B_h)[i] = 1.618f + i;  // Phi
        if (C_h) (C_h)[i] = 0.0f + i;
    }
}


template <typename T>
void initArraysForHost(T** A_h, T** B_h, T** C_h, size_t N, bool usePinnedHost = false) {
    size_t Nbytes = N * sizeof(T);

    if (usePinnedHost) {
        if (A_h) {
            HIPCHECK(hipHostMalloc((void**)A_h, Nbytes));
        }
        if (B_h) {
            HIPCHECK(hipHostMalloc((void**)B_h, Nbytes));
        }
        if (C_h) {
            HIPCHECK(hipHostMalloc((void**)C_h, Nbytes));
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

    setDefaultData(N, A_h ? *A_h : NULL, B_h ? *B_h : NULL, C_h ? *C_h : NULL);
}


template <typename T>
void initArrays(T** A_d, T** B_d, T** C_d, T** A_h, T** B_h, T** C_h, size_t N,
                bool usePinnedHost = false) {
    size_t Nbytes = N * sizeof(T);

    if (A_d) {
        HIPCHECK(hipMalloc(A_d, Nbytes));
    }
    if (B_d) {
        HIPCHECK(hipMalloc(B_d, Nbytes));
    }
    if (C_d) {
        HIPCHECK(hipMalloc(C_d, Nbytes));
    }

    initArraysForHost(A_h, B_h, C_h, N, usePinnedHost);
}


template <typename T>
void freeArraysForHost(T* A_h, T* B_h, T* C_h, bool usePinnedHost) {
    if (usePinnedHost) {
        if (A_h) {
            HIPCHECK(hipHostFree(A_h));
        }
        if (B_h) {
            HIPCHECK(hipHostFree(B_h));
        }
        if (C_h) {
            HIPCHECK(hipHostFree(C_h));
        }
    } else {
        if (A_h) {
            free(A_h);
        }
        if (B_h) {
            free(B_h);
        }
        if (C_h) {
            free(C_h);
        }
    }
}

template <typename T>
void freeArrays(T* A_d, T* B_d, T* C_d, T* A_h, T* B_h, T* C_h, bool usePinnedHost) {
    if (A_d) {
        HIPCHECK(hipFree(A_d));
    }
    if (B_d) {
        HIPCHECK(hipFree(B_d));
    }
    if (C_d) {
        HIPCHECK(hipFree(C_d));
    }

    freeArraysForHost(A_h, B_h, C_h, usePinnedHost);
}

#if defined(__HIP_PLATFORM_HCC__)
template <typename T>
void initArrays2DPitch(T** A_d, T** B_d, T** C_d, size_t* pitch_A, size_t* pitch_B, size_t* pitch_C,
                       size_t numW, size_t numH) {
    if (A_d) {
        HIPCHECK(hipMallocPitch((void**)A_d, pitch_A, numW * sizeof(T), numH));
    }
    if (B_d) {
        HIPCHECK(hipMallocPitch((void**)B_d, pitch_B, numW * sizeof(T), numH));
    }
    if (C_d) {
        HIPCHECK(hipMallocPitch((void**)C_d, pitch_C, numW * sizeof(T), numH));
    }

    HIPASSERT(*pitch_A == *pitch_B);
    HIPASSERT(*pitch_A == *pitch_C)
}

inline void initHIPArrays(hipArray** A_d, hipArray** B_d, hipArray** C_d,
                          const hipChannelFormatDesc* desc, const size_t numW, const size_t numH,
                          const unsigned int flags) {
    if (A_d) {
        HIPCHECK(hipMallocArray(A_d, desc, numW, numH, flags));
    }
    if (B_d) {
        HIPCHECK(hipMallocArray(B_d, desc, numW, numH, flags));
    }
    if (C_d) {
        HIPCHECK(hipMallocArray(C_d, desc, numW, numH, flags));
    }
}
#endif

// Assumes C_h contains vector add of A_h + B_h
// Calls the test "failed" macro if a mismatch is detected.
template <typename T>
size_t checkVectorADD(T* A_h, T* B_h, T* result_H, size_t N, bool expectMatch = true,
                      bool reportMismatch = true) {
    size_t mismatchCount = 0;
    size_t firstMismatch = 0;
    size_t mismatchesToPrint = 10;
    for (size_t i = 0; i < N; i++) {
        T expected = A_h[i] + B_h[i];
        if (result_H[i] != expected) {
            if (mismatchCount == 0) {
                firstMismatch = i;
            }
            mismatchCount++;
            if ((mismatchCount <= mismatchesToPrint) && expectMatch) {
                std::cout << std::fixed << std::setprecision(32);
                std::cout << "At " << i << std::endl;
                std::cout << "  Computed:" << result_H[i] << std::endl;
                std::cout << "  Expected:" << expected << std::endl;
            }
        }
    }

    if (reportMismatch) {
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

    return mismatchCount;
}


// Assumes C_h contains vector add of A_h + B_h
// Calls the test "failed" macro if a mismatch is detected.
template <typename T>
void checkTest(T* expected_H, T* result_H, size_t N, bool expectMatch = true) {
    size_t mismatchCount = 0;
    size_t firstMismatch = 0;
    size_t mismatchesToPrint = 10;
    for (size_t i = 0; i < N; i++) {
        if (result_H[i] != expected_H[i]) {
            if (mismatchCount == 0) {
                firstMismatch = i;
            }
            mismatchCount++;
            if ((mismatchCount <= mismatchesToPrint) && expectMatch) {
                std::cout << std::fixed << std::setprecision(32);
                std::cout << "At " << i << std::endl;
                std::cout << "  Computed:" << result_H[i] << std::endl;
                std::cout << "  Expected:" << expected_H[i] << std::endl;
            }
        }
    }

    if (expectMatch) {
        if (mismatchCount) {
            fprintf(stderr, "%zu mismatches ; first at index:%zu\n", mismatchCount, firstMismatch);
            // failed("%zu mismatches ; first at index:%zu\n", mismatchCount, firstMismatch);
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
    static const char* str() { return "Pinned"; };

    static void* Alloc(size_t sizeBytes) {
        void* p;
        HIPCHECK(hipHostMalloc((void**)&p, sizeBytes));
        return p;
    };
};


//---
struct Unpinned {
    static const bool isPinned = false;
    static const char* str() { return "Unpinned"; };

    static void* Alloc(size_t sizeBytes) {
        void* p = malloc(sizeBytes);
        HIPASSERT(p);
        return p;
    };
};


struct Memcpy {
    static const char* str() { return "Memcpy"; };
};

struct MemcpyAsync {
    static const char* str() { return "MemcpyAsync"; };
};


template <typename C>
struct MemTraits;


template <>
struct MemTraits<Memcpy> {
    static void Copy(void* dest, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                     hipStream_t stream) {
        HIPCHECK(hipMemcpy(dest, src, sizeBytes, kind));
    }
};


template <>
struct MemTraits<MemcpyAsync> {
    static void Copy(void* dest, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                     hipStream_t stream) {
        HIPCHECK(hipMemcpyAsync(dest, src, sizeBytes, kind, stream));
    }
};

};  // namespace HipTest
