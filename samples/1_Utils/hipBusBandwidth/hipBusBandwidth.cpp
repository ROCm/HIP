#include <stdio.h>
#include <iostream>
#include <hip_runtime.h>

#include "ResultDatabase.h"

// Cmdline parms:
const bool          p_verbose = false;
const bool          p_pinned  = true;
const unsigned int  p_iters   = 10;

#define CHECK_HIP_ERROR()                                                    \
{                                                                             \
    hipError_t err = hipGetLastError();                                     \
    if (err != hipSuccess)                                                   \
    {                                                                         \
        printf("error=%d name=%s at "                                         \
               "ln: %d\n  ",err,hipGetErrorString(err),__LINE__);            \
        exit(EXIT_FAILURE);                                                  \
    }                                                                         \
}


// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   Measures the bandwidth of the bus connecting the host processor to the
//   OpenCL device.  This benchmark repeatedly transfers data chunks of various
//   sizes across the bus to the OpenCL device, and calculates the bandwidth.
//
//
// Arguments:
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
//    Jeremy Meredith, Wed Dec  1 17:05:27 EST 2010
//    Added calculation of latency estimate.
//    Ben Sander - moved to standalone test 
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB) 
{
    // Sizes are in kb
    int nSizes  = 20;
    int sizes[20] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
		     32768,65536,131072,262144,524288};
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;

    // Create some host memory pattern
    float *hostMem = NULL;
    if (p_pinned)
    {
        hipMallocHost((void**)&hostMem, sizeof(float) * numMaxFloats);
        while (hipGetLastError() != hipSuccess)
        {
 	    // drop the size and try again
	    if (p_verbose) std::cout << " - dropping size allocating pinned mem\n";
	    --nSizes;
	    if (nSizes < 1)
	    {
            std::cerr << "Error: Couldn't allocated any pinned buffer\n";
		return;
	    }
	    numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
            hipMallocHost((void**)&hostMem, sizeof(float) * numMaxFloats);
        }
    }
    else
    {
        hostMem = new float[numMaxFloats];
    }

    for (int i = 0; i < numMaxFloats; i++)
    {
        hostMem[i] = i % 77;
    }

    float *device;
    hipMalloc((void**)&device, sizeof(float) * numMaxFloats);
    while (hipGetLastError() != hipSuccess)
    {
	// drop the size and try again
	if (p_verbose) std::cout << " - dropping size allocating device mem\n";
	--nSizes;
	if (nSizes < 1)
	{
        std::cerr << "Error: Couldn't allocated any device buffer\n";
	    return;
	}
	numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
        hipMalloc((void**)&device, sizeof(float) * numMaxFloats);
    }


    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    CHECK_HIP_ERROR();

    // Three passes, forward and backward both
    for (int pass = 0; pass < p_iters; pass++)
    {
        // store the times temporarily to estimate latency
        //float times[nSizes];
        // Step through sizes forward on even passes and backward on odd
        for (int i = 0; i < nSizes; i++)
        {
            int sizeIndex;
            if ((pass % 2) == 0)
                sizeIndex = i;
            else
                sizeIndex = (nSizes - 1) - i;

            int nbytes = sizes[sizeIndex] * 1024;

            hipEventRecord(start, 0);
            hipMemcpy(device, hostMem, nbytes, hipMemcpyHostToDevice);
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            float t = 0;
            hipEventElapsedTime(&t, start, stop);
            //times[sizeIndex] = t;

            // Convert to GB/sec
            if (p_verbose)
            {
                std::cerr << "size " << sizes[sizeIndex] << "k took " << t <<
                        " ms\n";
            }

            double speed = (double(sizes[sizeIndex]) * 1024. / (1000*1000)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "% 7dkB", sizes[sizeIndex]);
            resultDB.AddResult("DownloadSpeed", sizeStr, "GB/sec", speed);
            resultDB.AddResult("DownloadTime", sizeStr, "ms", t);
        }
    }

    // Cleanup
    hipFree((void*)device);
    CHECK_HIP_ERROR();
    if (p_pinned)
    {
        hipFreeHost((void*)hostMem);
        CHECK_HIP_ERROR();
    }
    else
    {
        delete[] hostMem;
    }
    hipEventDestroy(start);
    hipEventDestroy(stop);
}



int main(int argc, char *argv[])
{
    ResultDatabase resultDB;
    RunBenchmark(resultDB);

    resultDB.DumpSummary(std::cout);

    resultDB.DumpDetailed(std::cout);
}
