// Test under-development.  Calls async mem-copy API, experiment with functionality.

#include "hip_runtime.h"
#include "test_common.h"

unsigned p_streams = 2;


void simpleNegTest()
{
    printf ("testing: %s\n",__func__);
    hipError_t e;
    float *A_malloc, *A_pinned, *A_d;

    size_t Nbytes = N*sizeof(float);
    A_malloc = (float*)malloc(Nbytes);
    HIPCHECK(hipMallocHost(&A_pinned, Nbytes));
    HIPCHECK(hipMalloc(&A_d, Nbytes));


    // Can't use default with async copy
    e = hipMemcpyAsync(A_pinned, A_d, Nbytes, hipMemcpyDefault, NULL);
    HIPASSERT (e==hipErrorInvalidMemcpyDirection);


    // Not sure what happens here, the memory must be pinned.
    e = hipMemcpyAsync(A_malloc, A_d, Nbytes, hipMemcpyHostToDevice, NULL);
    HIPASSERT (e==hipErrorInvalidValue);


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
void chunkedAsyncExample(int nStreams, bool useNullStream, bool useSyncMemcpyH2D, bool useSyncMemcpyD2H)
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
    HipTest::parseStandardArguments(argc, argv, true);
    parseMyArguments(argc, argv);


    printf ("info: set device to %d\n", p_gpuDevice);
    HIPCHECK(hipSetDevice(p_gpuDevice));

    simpleNegTest();


    chunkedAsyncExample(p_streams, true, true, true); // Easy sync version
    chunkedAsyncExample(p_streams, false, true, true); // Easy sync version
    chunkedAsyncExample(p_streams, false, false, true); // Some async
    chunkedAsyncExample(p_streams, false, false, false); // All async



    passed();

}
