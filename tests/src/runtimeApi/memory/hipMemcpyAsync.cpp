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

// Test under-development.  Calls async mem-copy API, experiment with functionality.

#include "hip/hip_runtime.h"
#include "test_common.h"
unsigned p_streams = 2;


void simpleNegTest()
{
    printf ("testing: %s\n",__func__);
    hipError_t e;
    float *A_malloc, *A_pinned, *A_d;

    size_t Nbytes = N*sizeof(float);
    A_malloc = (float*)malloc(Nbytes);
    HIPCHECK(hipHostMalloc((void**)&A_pinned, Nbytes, hipHostMallocDefault));
    A_d = NULL;
    HIPCHECK(hipMalloc(&A_d, Nbytes));
    HIPASSERT(A_d != NULL);
    // Can't use default with async copy
    e = hipMemcpyAsync(A_pinned, A_d, Nbytes, hipMemcpyDefault, NULL);
//    HIPASSERT (e == hipSuccess);


    // Not sure what happens here, the memory must be pinned.
    e = hipMemcpyAsync(A_malloc, A_d, Nbytes, hipMemcpyHostToDevice, NULL);

    printf ("  async memcpy of A_malloc to A_d. Result=%d\n", e);
    //HIPASSERT (e==hipErrorInvalidValue);
}

class Pinned;
class Unpinned;

template <typename T> struct HostTraits;

template<>
struct HostTraits<Pinned>
{
    static const char *Name() { return "Pinned"; } ;

    static void *Alloc(size_t sizeBytes) {
        void *p; 
        HIPCHECK(hipHostMalloc((void**)&p, sizeBytes, hipHostMallocDefault));
        return p;
    };
};


template<typename T>
__global__ void 
addK (hipLaunchParm lp, T *A, T K, size_t numElements)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<numElements; i+=stride) {
        A[i] = A[i] + K;
	}
}



//---
//Tests propert dependency resolution between H2D and D2H commands in same stream:
//IN: numInflight : number of copies inflight at any time:
//IN: numPongs = number of iterations to run (iteration)
template<typename T, class AllocType>
void test_pingpong(hipStream_t stream, size_t numElements, int numInflight, int numPongs, bool doHostSide) 
{
    HIPASSERT(numElements % numInflight == 0); // Must be evenly divisible.
    size_t Nbytes = numElements*sizeof(T);
    size_t eachCopyElements = numElements / numInflight;
    size_t eachCopyBytes = eachCopyElements * sizeof(T);

    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    printf ("-----------------------------------------------------------------------------------------------\n");
    printf ("testing: %s<%s>  Nbytes=%zu (%6.1f MB) numPongs=%d numInflight=%d eachCopyElements=%zu eachCopyBytes=%zu\n", 
            __func__, HostTraits<AllocType>::Name(), Nbytes, (double)(Nbytes)/1024.0/1024.0, numPongs, numInflight, eachCopyElements, eachCopyBytes);

    T *A_h = NULL;
    T *A_d = NULL;

    A_h = (T*)(HostTraits<AllocType>::Alloc(Nbytes));
    HIPCHECK(hipMalloc(&A_d, Nbytes));

    // Initialize the host array:
    const T initValue = 13;
    const T deviceConst = 2;
    const T hostConst = 10000;
    for (size_t i=0; i<numElements; i++) {
        A_h[i] = initValue + i;
    }


    for (int k=0; k<numPongs; k++ ) {
        for (int i=0; i<numInflight; i++) {
            HIPASSERT(A_d + i*eachCopyElements < A_d + Nbytes);
            HIPCHECK(hipMemcpyAsync(&A_d[i*eachCopyElements], &A_h[i*eachCopyElements], eachCopyBytes, hipMemcpyHostToDevice, stream));
        }

        hipLaunchKernel(addK<T>, dim3(blocks), dim3(threadsPerBlock), 0, stream,   A_d, 2, numElements);

        for (int i=0; i<numInflight; i++ ) {
            HIPASSERT(A_d + i*eachCopyElements < A_d + Nbytes);
            HIPCHECK(hipMemcpyAsync(&A_h[i*eachCopyElements], &A_d[i*eachCopyElements], eachCopyBytes, hipMemcpyDeviceToHost, stream));
        }

        if (doHostSide) {
            assert(0);
#if 0
            hipEvent_t e;
            HIPCHECK(hipEventCreate(&e));
#endif
            HIPCHECK(hipDeviceSynchronize());
            for (size_t i=0; i<numElements; i++) {
                A_h[i] += hostConst;
            }
        }
    };

    HIPCHECK(hipDeviceSynchronize());


    // Verify we copied back all the data correctly:
    for (size_t i=0; i<numElements; i++) {
        T gold = initValue + i;
        // Perform calcs in same order as test above to replicate FP order-of-operations:
        for (int k=0; k<numPongs; k++) {
            gold += deviceConst;
            if (doHostSide) {
                gold += hostConst;
            }
        }

        if (gold != A_h[i]) {
            std::cout << i << ": gold=" << gold << " out=" << A_h[i] << std::endl;
            HIPASSERT(gold == A_h[i]);
        }
    }


    HIPCHECK(hipHostFree(A_h));
    HIPCHECK(hipFree(A_d));
}


//---
//Send many async copies to the same stream.
//This requires runtime to keep track of many outstanding commands, and in the case of HCC requires growing/tracking the signal pool:
template<typename T>
void test_manyInflightCopies(hipStream_t stream, int numElements, int numCopies, bool syncBetweenCopies)
{
    size_t Nbytes = numElements*sizeof(T);
    size_t eachCopyElements = numElements / numCopies;
    size_t eachCopyBytes = eachCopyElements * sizeof(T);

    printf ("-----------------------------------------------------------------------------------------------\n");
    printf ("testing: %s  Nbytes=%zu (%6.1f MB) numCopies=%d eachCopyElements=%zu eachCopyBytes=%zu\n", 
            __func__, Nbytes, (double)(Nbytes)/1024.0/1024.0, numCopies, eachCopyElements, eachCopyBytes);

    T *A_d;
    T *A_h1, *A_h2;

    HIPCHECK(hipHostMalloc((void**)&A_h1, Nbytes, hipHostMallocDefault));
    HIPCHECK(hipHostMalloc((void**)&A_h2, Nbytes, hipHostMallocDefault));
    HIPCHECK(hipMalloc(&A_d, Nbytes));

    for (int i=0; i<numElements; i++) {
        A_h1[i] = 3.14f + static_cast<T> (i);
    }


    //stream=0; // fixme TODO


    for (int i=0; i<numCopies; i++) 
    {
        HIPASSERT(A_d + i*eachCopyElements < A_d + Nbytes);
        HIPCHECK(hipMemcpyAsync(&A_d[i*eachCopyElements], &A_h1[i*eachCopyElements], eachCopyBytes, hipMemcpyHostToDevice, stream));
    }

    if (syncBetweenCopies) {
        HIPCHECK(hipDeviceSynchronize());
    }

    for (int i=0; i<numCopies; i++) 
    {
        HIPASSERT(A_d + i*eachCopyElements < A_d + Nbytes);
        HIPCHECK(hipMemcpyAsync(&A_h2[i*eachCopyElements], &A_d[i*eachCopyElements], eachCopyBytes, hipMemcpyDeviceToHost, stream));
    }

    HIPCHECK(hipDeviceSynchronize());


    // Verify we copied back all the data correctly:
    for (int i=0; i<numElements; i++) {
        HIPASSERT(A_h1[i] == A_h2[i]);
    }


    HIPCHECK(hipHostFree(A_h1));
    HIPCHECK(hipHostFree(A_h2));
    HIPCHECK(hipFree(A_d));
}


//---
//Classic example showing how to overlap data transfer with compute.
//We divide the work into "chunks" and create a stream for each chunk.
//Each chunk then runs a H2D copy, followed by kernel execution, followed by D2H copyback.
//Work in separate streams is independent which enables concurrency.

// IN: nStreams : number of streams to use for the test
// IN :useNullStream - use NULL stream.  Synchronizes everything.
// IN: useSyncMemcpyH2D - use sync memcpy (no overlap) for H2D
// IN: useSyncMemcpyD2H - use sync memcpy (no overlap) for D2H
void test_chunkedAsyncExample(int nStreams, bool useNullStream, bool useSyncMemcpyH2D, bool useSyncMemcpyD2H)
{

    size_t Nbytes = N*sizeof(int);
    printf ("testing: %s(useNullStream=%d, useSyncMemcpyH2D=%d, useSyncMemcpyD2H=%d)  ",__func__, useNullStream, useSyncMemcpyH2D, useSyncMemcpyD2H);
    printf ("Nbytes=%zu (%6.1f MB)\n", Nbytes, (double)(Nbytes)/1024.0/1024.0);

    int *A_d, *B_d, *C_d;
    int *A_h, *B_h, *C_h;

    HipTest::initArrays (&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, true);


    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);


    hipStream_t *stream = (hipStream_t*)malloc(sizeof(hipStream_t) * nStreams);
    if (useNullStream) { 
        nStreams = 1;
        stream[0] = NULL;
    } else  {
        for (int i = 0; i < nStreams; ++i) {
            HIPCHECK (hipStreamCreate(&stream[i]));
        }
    }


    size_t workLeft = N; 
    size_t workPerStream = N / nStreams;
    for (int i = 0; i < nStreams; ++i) {
        size_t work = (workLeft < workPerStream) ? workLeft : workPerStream;
        size_t workBytes = work * sizeof(int);

        size_t offset = i*workPerStream;
        HIPASSERT(A_d + offset < A_d + Nbytes);
        HIPASSERT(B_d + offset < B_d + Nbytes);
        HIPASSERT(C_d + offset < C_d + Nbytes);
        if (useSyncMemcpyH2D) {
            HIPCHECK ( hipMemcpy(&A_d[offset], &A_h[offset], workBytes, hipMemcpyHostToDevice));
            HIPCHECK ( hipMemcpy(&B_d[offset], &B_h[offset], workBytes, hipMemcpyHostToDevice));
        } else {
            HIPCHECK ( hipMemcpyAsync(&A_d[offset], &A_h[offset], workBytes, hipMemcpyHostToDevice, stream[i]));
            HIPCHECK ( hipMemcpyAsync(&B_d[offset], &B_h[offset], workBytes, hipMemcpyHostToDevice, stream[i]));
        };

        hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, stream[i], &A_d[offset], &B_d[offset], &C_d[offset], work);

        if (useSyncMemcpyD2H) {
            HIPCHECK ( hipMemcpy(&C_h[offset], &C_d[offset], workBytes, hipMemcpyDeviceToHost));
        } else {
            HIPCHECK ( hipMemcpyAsync(&C_h[offset], &C_d[offset], workBytes, hipMemcpyDeviceToHost, stream[i]));
        }
    } 


    HIPCHECK (hipDeviceSynchronize());


    HipTest::checkVectorADD(A_h, B_h, C_h, N);

    HipTest::freeArrays (A_d, B_d, C_d, A_h, B_h, C_h, true);

    free(stream);
};


//---
//Parse arguments specific to this test.
void parseMyArguments(int argc, char *argv[])
{
    int more_argc = HipTest::parseStandardArguments(argc, argv, false);

    // parse args for this test:
    for (int i = 1; i < more_argc; i++) {
        const char *arg = argv[i];

        if (!strcmp(arg, "--streams")) {
            if (++i >= argc || !HipTest::parseUInt(argv[i], &p_streams)) {
               failed("Bad streams argument");
            }
        } else {
            failed("Bad argument '%s'", arg);
        }
    };
};




int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, false);
    parseMyArguments(argc, argv);


    printf ("info: set device to %d  tests=%x\n", p_gpuDevice, p_tests);
    HIPCHECK(hipSetDevice(p_gpuDevice));

    if (p_tests & 0x01) {
        simpleNegTest();
    }

    if (p_tests & 0x02) {
        hipStream_t stream;
        HIPCHECK (hipStreamCreate(&stream));

        test_manyInflightCopies<float>(stream, 1024,   16,  true);
        test_manyInflightCopies<float>(stream, 1024,    4,  true); // verify we re-use the same entries instead of growing pool.
        test_manyInflightCopies<float>(stream, 1024*8, 64, false);

        HIPCHECK(hipStreamDestroy(stream));
    }


    if (p_tests & 0x04) {
        test_chunkedAsyncExample(p_streams, true, true, true); // Easy sync version
        test_chunkedAsyncExample(p_streams, false, true, true); // Easy sync version
        test_chunkedAsyncExample(p_streams, false, false, true); // Some async
        test_chunkedAsyncExample(p_streams, false, false, false); // All async
    }

    if (p_tests & 0x08) {
        hipStream_t stream;
        HIPCHECK (hipStreamCreate(&stream));

//        test_pingpong<int, Pinned>(stream, 1024*1024*32, 1, 1, false);
//        test_pingpong<int, Pinned>(stream, 1024*1024*32, 1, 10, false);

        HIPCHECK(hipStreamDestroy(stream));
    }


    passed();

}
