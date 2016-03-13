#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <hip_runtime.h>

#include "ResultDatabase.h"

// Cmdline parms:
bool          p_verbose = false;
bool          p_pinned  = true;
int           p_iterations   = 10;
int           p_device  = 0;
int           p_detailed  = 0;

bool          p_h2d = true;
bool          p_d2h = true;


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
int sizeToBytes(int size) {
    return (size < 0) ? -size : size * 1024;
}


// ****************************************************************************
std::string sizeToString(int size) 
{
    using namespace std;
    stringstream ss;
    if (size < 0) {
        ss << char(0x1) << setfill('0') << setw(3) << -size <<  "B";
    } else {
        ss << size << "kB";
    }
    return ss.str();
}


// ****************************************************************************
// -sizes are in bytes, +sizes are in kb, last size must be largest
int sizes[] = {-64, -256, -512, 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384, 32768,65536,131072,262144,524288};
int nSizes  = sizeof(sizes) / sizeof(int);


// ****************************************************************************
// Function: RunBenchmark_H2D
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
void RunBenchmark_H2D(ResultDatabase &resultDB) 
{
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;

    hipSetDevice(p_device);

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
    for (int pass = 0; pass < p_iterations; pass++)
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

            int nbytes = sizeToBytes(sizes[sizeIndex]);

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
                std::cerr << "size " << sizeToString(sizes[sizeIndex]) << " took " << t <<
                        " ms\n";
            }

            double speed = (double(sizeToBytes(sizes[sizeIndex])) / (1000*1000)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "%9s", sizeToString(sizes[sizeIndex]).c_str());
            resultDB.AddResult("H2D_Bandwidth", sizeStr, "GB/sec", speed);
            resultDB.AddResult("H2D_Time", sizeStr, "ms", t);
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



// ****************************************************************************
void RunBenchmark_D2H(ResultDatabase &resultDB)
{
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;

    // Create some host memory pattern
    float *hostMem1;
    float *hostMem2;
    if (p_pinned)
    {
        hipMallocHost((void**)&hostMem1, sizeof(float)*numMaxFloats);
        hipError_t err1 = hipGetLastError();
        hipMallocHost((void**)&hostMem2, sizeof(float)*numMaxFloats);
        hipError_t err2 = hipGetLastError();
	while (err1 != hipSuccess || err2 != hipSuccess)
	{
	    // free the first buffer if only the second failed
	    if (err1 == hipSuccess)
	        hipFreeHost((void*)hostMem1);

	    // drop the size and try again
	    if (p_verbose) std::cout << " - dropping size allocating pinned mem\n";
	    --nSizes;
	    if (nSizes < 1)
	    {
            std::cerr << "Error: Couldn't allocated any pinned buffer\n";
		return;
	    }
	    numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
            hipMallocHost((void**)&hostMem1, sizeof(float)*numMaxFloats);
            err1 = hipGetLastError();
            hipMallocHost((void**)&hostMem2, sizeof(float)*numMaxFloats);
            err2 = hipGetLastError();
	}
   }
    else
    {
        hostMem1 = new float[numMaxFloats];
        hostMem2 = new float[numMaxFloats];
    }
    for (int i=0; i<numMaxFloats; i++)
        hostMem1[i] = i % 77;

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

    hipMemcpy(device, hostMem1,
               numMaxFloats*sizeof(float), hipMemcpyHostToDevice);
    hipDeviceSynchronize();

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    CHECK_HIP_ERROR();

    // Three passes, forward and backward both
    for (int pass = 0; pass < p_iterations; pass++)
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

            int nbytes = sizeToBytes(sizes[sizeIndex]);

            hipEventRecord(start, 0);
            hipMemcpy(hostMem2, device,
                       nbytes, hipMemcpyDeviceToHost);
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            float t = 0;
            hipEventElapsedTime(&t, start, stop);
            //times[sizeIndex] = t;

            // Convert to GB/sec
            if (p_verbose)
            {
                std::cerr << "size " <<sizeToString(sizes[sizeIndex]) << " took " << t <<
                        " ms\n";
            }

            double speed = (double(sizeToBytes(sizes[sizeIndex])) / (1000*1000)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "%9s", sizeToString(sizes[sizeIndex]).c_str());
            resultDB.AddResult("D2H_Bandwidth", sizeStr, "GB/sec", speed);
            resultDB.AddResult("D2H_Time", sizeStr, "ms", t);
        }
	//resultDB.AddResult("ReadbackLatencyEstimate", "1-2kb", "ms", times[0]-(times[1]-times[0])/1.);
	//resultDB.AddResult("ReadbackLatencyEstimate", "1-4kb", "ms", times[0]-(times[2]-times[0])/3.);
	//resultDB.AddResult("ReadbackLatencyEstimate", "2-4kb", "ms", times[1]-(times[2]-times[1])/1.);
    }

    // Cleanup
    hipFree((void*)device);
    CHECK_HIP_ERROR();
    if (p_pinned)
    {
        hipFreeHost((void*)hostMem1);
        CHECK_HIP_ERROR();
        hipFreeHost((void*)hostMem2);
        CHECK_HIP_ERROR();
    }
    else
    {
        delete[] hostMem1;
        delete[] hostMem2;
        hipEventDestroy(start);
	    hipEventDestroy(stop);
    }
}


#define failed(...) \
    printf ("error: ");\
    printf (__VA_ARGS__);\
    printf ("\n");\
    exit(EXIT_FAILURE);

int parseInt(const char *str, int *output)
{
    char *next;
    *output = strtol(str, &next, 0);
    return !strlen(next);
}

void help() {
};

int parseStandardArguments(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (!strcmp(arg, " ")) {
            // skip NULL args.
        } else if (!strcmp(arg, "--iterations") || (!strcmp(arg, "-i"))) {
            if (++i >= argc || !parseInt(argv[i], &p_iterations)) {
               failed("Bad iterations argument"); 
            }
        } else if (!strcmp(arg, "--device") || (!strcmp(arg, "-d"))) {
            if (++i >= argc || !parseInt(argv[i], &p_device)) {
               failed("Bad device argument"); 
            }
        } else if (!strcmp(arg, "--unpinned")) {
            p_pinned = 0;
        } else if (!strcmp(arg, "--h2d")) {
            p_h2d = true;
            p_d2h = false;

        } else if (!strcmp(arg, "--d2h")) {
            p_h2d = false;
            p_d2h = true;

        } else if (!strcmp(arg, "--help")  || (!strcmp(arg, "-h"))) {
            help();

        } else if (!strcmp(arg, "--verbose")) {
            p_verbose = 1;
        } else if (!strcmp(arg, "--detailed")) {
            p_detailed = 1;
        } else {
            failed("Bad argument '%s'", arg);
        }
    } 

    return 0;
};



int main(int argc, char *argv[])
{
    parseStandardArguments(argc, argv);

    if (p_h2d) {
        ResultDatabase resultDB;
        RunBenchmark_H2D(resultDB);

        resultDB.DumpSummary(std::cout);

        if (p_detailed) {
            resultDB.DumpDetailed(std::cout);
        }
    }

    if (p_d2h) {
        ResultDatabase resultDB;
        RunBenchmark_D2H(resultDB);

        resultDB.DumpSummary(std::cout);

        if (p_detailed) {
            resultDB.DumpDetailed(std::cout);
        }
    }
}
