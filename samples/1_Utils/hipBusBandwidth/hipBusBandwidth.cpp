#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "hip/hip_runtime.h"

#include "ResultDatabase.h"

// Cmdline parms:
bool          p_verbose = false;
bool          p_pinned  = true;
int           p_iterations   = 10;
int           p_beatsperiteration=1;
int           p_device  = 0;
int           p_detailed  = 0;
bool          p_async = 0; 
int           p_alignedhost = 0;  // align host allocs to this granularity, in bytes. 64 or 4096 are good values to try.
int           p_onesize = 0;  

bool          p_h2d   = true;
bool          p_d2h   = true;
bool          p_bidir = true;




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
        // char (-) lexically sorts before " " so will cause Byte values to be displayed before kB.
        ss << "+" << setfill('0') << setw(3) << -size <<  "By";
    } else {
        ss << size << "kB";
    }
    return ss.str();
}


// ****************************************************************************
hipError_t memcopy(void * dst, const void *src, size_t sizeBytes, enum hipMemcpyKind kind)
{
    if (p_async) {
        return hipMemcpyAsync(dst, src, sizeBytes, kind, NULL);
    } else {
        return hipMemcpy(dst, src, sizeBytes, kind);
    }
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
        hipHostMalloc((void**)&hostMem, sizeof(float) * numMaxFloats);
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
            hipHostMalloc((void**)&hostMem, sizeof(float) * numMaxFloats);
        }
    }
    else
    {
        if (p_alignedhost) {
            hostMem = (float*)aligned_alloc(p_alignedhost, numMaxFloats*sizeof(float));
        } else {
            hostMem = new float[numMaxFloats];
        }
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

            const int thisSize = p_onesize ? p_onesize : sizes[sizeIndex];
            const int nbytes = sizeToBytes(thisSize);

            hipEventRecord(start, 0);
            for (int j=0;j<p_beatsperiteration;j++) {
                memcopy(device, hostMem, nbytes, hipMemcpyHostToDevice);
            }
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            float t = 0;
            hipEventElapsedTime(&t, start, stop);
            //times[sizeIndex] = t;

            // Convert to GB/sec
            if (p_verbose)
            {
                std::cerr << "size " << sizeToString(thisSize) << " took " << t << " ms\n";
            }

            double speed = (double(sizeToBytes(thisSize) * p_beatsperiteration) / (1000*1000)) / t;
            char sizeStr[256];
            if (p_beatsperiteration>1) {
                sprintf(sizeStr, "%9sx%d", sizeToString(thisSize).c_str(), p_beatsperiteration);
            } else {
                sprintf(sizeStr, "%9s", sizeToString(thisSize).c_str());
            }
            resultDB.AddResult(std::string("H2D_Bandwidth") + (p_pinned ? "_Pinned" : "_Unpinned"), sizeStr, "GB/sec", speed);
            resultDB.AddResult(std::string("H2D_Time") + (p_pinned ? "_Pinned" : "_Unpinned"), sizeStr, "ms", t);

            if (p_onesize) {
                break;
            }
        }
    }

    if (p_onesize) {
        numMaxFloats = sizeToBytes(p_onesize) / sizeof(float);
    }

    // Check.  First reset the host memory, then copy-back result.  Then compare against original ref value.
    for (int i = 0; i < numMaxFloats; i++)
    {
        hostMem[i] = 0;
    }
    hipMemcpy(hostMem, device, numMaxFloats*sizeof(float), hipMemcpyDeviceToHost);
    for (int i = 0; i < numMaxFloats; i++)
    {
        float ref = i % 77;
        if (ref != hostMem[i]) {
            printf ("error: H2D. i=%d reference:%6.f != copyback:%6.2f\n", i, ref, hostMem[i]);
        }
    }


    // Cleanup
    hipFree((void*)device);
    CHECK_HIP_ERROR();
    if (p_pinned)
    {
        hipHostFree((void*)hostMem);
        CHECK_HIP_ERROR();
    }
    else
    {
        if (p_alignedhost) {
            delete[] hostMem;
        } else {
            free(hostMem);
        }
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
        hipHostMalloc((void**)&hostMem1, sizeof(float)*numMaxFloats);
        hipError_t err1 = hipGetLastError();
        hipHostMalloc((void**)&hostMem2, sizeof(float)*numMaxFloats);
        hipError_t err2 = hipGetLastError();
	while (err1 != hipSuccess || err2 != hipSuccess)
	{
	    // free the first buffer if only the second failed
	    if (err1 == hipSuccess)
	        hipHostFree((void*)hostMem1);

	    // drop the size and try again
	    if (p_verbose) std::cout << " - dropping size allocating pinned mem\n";
	    --nSizes;
	    if (nSizes < 1)
	    {
            std::cerr << "Error: Couldn't allocated any pinned buffer\n";
		return;
	    }
	    numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
            hipHostMalloc((void**)&hostMem1, sizeof(float)*numMaxFloats);
            err1 = hipGetLastError();
            hipHostMalloc((void**)&hostMem2, sizeof(float)*numMaxFloats);
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

    hipMemcpy(device, hostMem1, numMaxFloats*sizeof(float), hipMemcpyHostToDevice);
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

            const int thisSize = p_onesize ? p_onesize : sizes[sizeIndex];
            const int nbytes = sizeToBytes(thisSize);

            hipEventRecord(start, 0);
            for (int j=0;j<p_beatsperiteration;j++) {
                memcopy(hostMem2, device, nbytes, hipMemcpyDeviceToHost);
            }
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            float t = 0;
            hipEventElapsedTime(&t, start, stop);
            //times[sizeIndex] = t;

            // Convert to GB/sec
            if (p_verbose)
            {
                std::cerr << "size " <<sizeToString(thisSize) << " took " << t <<
                        " ms\n";
            }

            double speed = (double(sizeToBytes(thisSize)) / (1000*1000)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "%9s", sizeToString(thisSize).c_str());
            if (p_beatsperiteration>1) {
                sprintf(sizeStr, "%9sx%d", sizeToString(thisSize).c_str(), p_beatsperiteration);
            } else {
                sprintf(sizeStr, "%9s", sizeToString(thisSize).c_str());
            }
            resultDB.AddResult(std::string("D2H_Bandwidth") + (p_pinned ? "_Pinned" : "_Unpinned"), sizeStr, "GB/sec", speed);
            resultDB.AddResult(std::string("D2H_Time") + (p_pinned ? "_Pinned" : "_Unpinned"), sizeStr, "ms", t);
            if (p_onesize) {
                break;
            }
        }
    }

    if (p_onesize) {
        numMaxFloats = sizeToBytes(p_onesize) / sizeof(float);
    }
    // Check.  First reset the host memory, then copy-back result.  Then compare against original ref value.
    for (int i = 0; i < numMaxFloats; i++)
    {
        float ref = i % 77;
        if (ref != hostMem2[i]) {
            printf ("error: D2H. i=%d reference:%6.f != copyback:%6.2f\n", i, ref, hostMem2[i]);
        }
    }

    // Cleanup
    hipFree((void*)device);
    CHECK_HIP_ERROR();
    if (p_pinned)
    {
        hipHostFree((void*)hostMem1);
        CHECK_HIP_ERROR();
        hipHostFree((void*)hostMem2);
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


void RunBenchmark_Bidir(ResultDatabase &resultDB) 
{
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;

    hipSetDevice(p_device);

    hipStream_t stream[2];


    // Create some host memory pattern
    float *hostMem[2] = {NULL, NULL};
    if (p_pinned)
    {
        while (1) 
        {
            hipError_t e1 = hipHostMalloc((void**)&hostMem[0], sizeof(float) * numMaxFloats);
            hipError_t e2 = hipHostMalloc((void**)&hostMem[1], sizeof(float) * numMaxFloats);

            if ((e1 == hipSuccess) && (e2 == hipSuccess)) {
                break;
            } else {
                // drop the size and try again
                if (p_verbose) std::cout << " - dropping size allocating pinned mem\n";
                --nSizes;
                if (nSizes < 1)
                {
                    std::cerr << "Error: Couldn't allocated any pinned buffer\n";
                return;
                }
                numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
            }
        }
    }
    else
    {
        hostMem[0] = new float[numMaxFloats];
        hostMem[1] = new float[numMaxFloats];
    }

    for (int i = 0; i < numMaxFloats; i++)
    {
        hostMem[0][i] = i % 77;
    }

    float *deviceMem[2];
    while (1) {
        hipError_t e1 = hipMalloc((void**)&deviceMem[0], sizeof(float) * numMaxFloats);
        hipError_t e2 = hipMalloc((void**)&deviceMem[1], sizeof(float) * numMaxFloats);

        if ((e1 == hipSuccess) && (e2 == hipSuccess)) {
            break;
        } else {
            if (e1) {
                // First alloc succeeded, so free it before trying again
                hipFree(&deviceMem[0]);
            }
            // drop the size and try again
            if (p_verbose) std::cout << " - dropping size allocating device mem\n";
            --nSizes;
            if (nSizes < 1)
            {
                std::cerr << "Error: Couldn't allocated any device buffer\n";
                return;
            }
            numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
        }
    };


    hipMemset(deviceMem[1], 0xFA, numMaxFloats);


    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    CHECK_HIP_ERROR();
    hipStreamCreate(&stream[0]);
    hipStreamCreate(&stream[1]);

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

            const int thisSize = p_onesize ? p_onesize : sizes[sizeIndex];
            const int nbytes = sizeToBytes(thisSize);

            hipEventRecord(start, 0);
            hipMemcpyAsync(deviceMem[0], hostMem[0],   nbytes, hipMemcpyHostToDevice, stream[0]);
            hipMemcpyAsync(hostMem[1],   deviceMem[1], nbytes, hipMemcpyDeviceToHost, stream[1]);
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            float t = 0;
            hipEventElapsedTime(&t, start, stop);

            // Convert to GB/sec
            if (p_verbose)
            {
                std::cerr << "size " << sizeToString(thisSize) << " took " << t <<
                        " ms\n";
            }

            double speed = (double(sizeToBytes(thisSize)) / (1000*1000)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "%9s", sizeToString(thisSize).c_str());
            resultDB.AddResult(std::string("Bidir_Bandwidth") + (p_pinned ? "_Pinned" : "_Unpinned"), sizeStr, "GB/sec", speed);
            resultDB.AddResult(std::string("Bidir_Time") + (p_pinned ? "_Pinned" : "_Unpinned"), sizeStr, "ms", t);
        }
    }

    // Cleanup
    hipFree((void*)deviceMem[0]);
    hipFree((void*)deviceMem[1]);
    CHECK_HIP_ERROR();
    if (p_pinned)
    {
        hipHostFree((void*)hostMem[0]);
        hipHostFree((void*)hostMem[1]);
        CHECK_HIP_ERROR();
    }
    else
    {
        delete[] hostMem[0];
        delete[] hostMem[1];
    }
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipStreamDestroy(stream[0]);
    hipStreamDestroy(stream[1]);
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


void printConfig() {
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, p_device);

    printf ("Device:%s Mem=%.1fGB #CUs=%d Freq=%.0fMhz  Pinned=%s\n", props.name, props.totalGlobalMem/1024.0/1024.0/1024.0, props.multiProcessorCount, props.clockRate/1000.0, p_pinned ? "YES" : "NO");
}

void help() {
    printf ("Usage: hipBusBandwidth [OPTIONS]\n");
    printf ("  --iterations, -i         : Number of copy iterations to run.\n");
    printf ("  --beatsperiterations, -b : Number of beats (back-to-back copies of same size) per iteration to run.\n");
    printf ("  --device, -d             : Device ID to use (0..numDevices).\n");
    printf ("  --unpinned               : Use unpinned host memory.\n");
    printf ("  --d2h                    : Run only device-to-host test.\n");
    printf ("  --h2d                    : Run only host-to-device test.\n");
    printf ("  --bidir                  : Run only bidir copy test.\n");
    printf ("  --verbose                : Print verbose status messages as test is run.\n");
    printf ("  --detailed               : Print detailed report (including all trials).\n");

    printf ("  --async                  : Use hipMemcpyAsync(with NULL stream) for H2D/D2H.  Default uses hipMemcpy.\n");
    printf ("  --onesize, -o            : Only run one measurement, at specified size (in KB, or if negative in bytes)\n");

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
        } else if (!strcmp(arg, "--beatsperiteration") || (!strcmp(arg, "-b"))) {
            if (++i >= argc || !parseInt(argv[i], &p_beatsperiteration)) {
               failed("Bad beatsperiteration argument"); 
            }
        } else if (!strcmp(arg, "--device") || (!strcmp(arg, "-d"))) {
            if (++i >= argc || !parseInt(argv[i], &p_device)) {
               failed("Bad device argument"); 
            }
        } else if (!strcmp(arg, "--onesize") || (!strcmp(arg, "-o"))) {
            if (++i >= argc || !parseInt(argv[i], &p_onesize)) {
               failed("Bad onesize argument"); 
            }
        } else if (!strcmp(arg, "--unpinned")) {
            p_pinned = 0;
        } else if (!strcmp(arg, "--h2d")) {
            p_h2d   = true;
            p_d2h   = false;
            p_bidir = false;

        } else if (!strcmp(arg, "--d2h")) {
            p_h2d   = false;
            p_d2h   = true;
            p_bidir = false;

        } else if (!strcmp(arg, "--bidir")) {
            p_h2d   = false;
            p_d2h   = false;
            p_bidir = true;

        } else if (!strcmp(arg, "--help")  || (!strcmp(arg, "-h"))) {
            help();
            exit(EXIT_SUCCESS);


        } else if (!strcmp(arg, "--verbose")) {
            p_verbose = 1;
        } else if (!strcmp(arg, "--async")) {
            p_async = 1;
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

    printConfig();

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


    if (p_bidir) {
        ResultDatabase resultDB;
        RunBenchmark_Bidir(resultDB);

        resultDB.DumpSummary(std::cout);

        if (p_detailed) {
            resultDB.DumpDetailed(std::cout);
        }
    }
}
