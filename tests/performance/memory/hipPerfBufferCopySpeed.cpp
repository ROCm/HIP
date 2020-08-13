#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <complex>

#include "timer.h"
#include "test_common.h"

/* HIT_START
 * BUILD: %t %s ../../src/test_common.cpp ../../src/timer.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define NUM_SIZES 8
//4KB, 8KB, 64KB, 256KB, 1 MB, 4MB, 16 MB, 16MB+10
static const unsigned int Sizes[NUM_SIZES] = {4096, 8192, 65536, 262144, 1048576, 4194304, 16777216, 16777216+10};

static const unsigned int Iterations[2] = {1, 1000};

#define BUF_TYPES 4
//  16 ways to combine 4 different buffer types
#define NUM_SUBTESTS (BUF_TYPES*BUF_TYPES)

#define CHECK_RESULT(test, msg)         \
    if ((test))                         \
    {                                   \
        printf("\n%s\n", msg);          \
        abort();                        \
    }

void setData(void *ptr, unsigned int size, char value)
{
    char *ptr2 = (char *)ptr;
    for (unsigned int i = 0; i < size ; i++)
    {
        ptr2[i] = value;
    }
}

void checkData(void *ptr, unsigned int size, char value)
{
    char *ptr2 = (char *)ptr;
    for (unsigned int i = 0; i < size; i++)
    {
        if (ptr2[i] != value)
        {
            printf("Data validation failed at %d!  Got 0x%08x\n", i, ptr2[i]);
            printf("Expected 0x%08x\n", value);
            CHECK_RESULT(true, "Data validation failed!");
            break;
        }
    }
}


int main(int argc, char* argv[]) {
    HipTest::parseStandardArguments(argc, argv, true);

    hipError_t err = hipSuccess;
    hipDeviceProp_t props = {0};
    hipGetDeviceProperties(&props, p_gpuDevice);
    CHECK_RESULT(err != hipSuccess, "hipGetDeviceProperties failed" );
    printf("Set device to %d : %s\n", p_gpuDevice, props.name);
    printf("Legend: unp - unpinned(malloc), hM - hipMalloc(device)\n");
    printf("        hHR - hipHostRegister(pinned), hHM - hipHostMalloc(prePinned)\n");
    err = hipSetDevice(p_gpuDevice);
    CHECK_RESULT(err != hipSuccess, "hipSetDevice failed" );

    unsigned int bufSize_;
    bool hostMalloc[2] = {false};
    bool hostRegister[2] = {false};
    bool unpinnedMalloc[2] = {false};
    unsigned int numIter;
    void *memptr[2] = {NULL};
    void *alignedmemptr[2] = {NULL};
    void* srcBuffer = NULL;
    void* dstBuffer = NULL;

    int numTests = (p_tests == -1) ? (NUM_SIZES*NUM_SUBTESTS*2 - 1) : p_tests;
    int test = (p_tests == -1) ? 0 : p_tests;

    for(;test <= numTests; test++)
    {
        unsigned int srcTest = (test / NUM_SIZES) % BUF_TYPES;
        unsigned int dstTest = (test / (NUM_SIZES*BUF_TYPES)) % BUF_TYPES;
        bufSize_ = Sizes[test % NUM_SIZES];
        hostMalloc[0] = hostMalloc[1] = false;
        hostRegister[0] = hostRegister[1] = false;
        unpinnedMalloc[0] = unpinnedMalloc[1] = false;
        srcBuffer = dstBuffer = 0;
        memptr[0] = memptr[1] = NULL;
        alignedmemptr[0] = alignedmemptr[1] = NULL;

        if (srcTest == 3)
        {
            hostRegister[0] = true;
        }
        else if (srcTest == 2)
        {
            hostMalloc[0] = true;
        }
        else if (srcTest == 1)
        {
            unpinnedMalloc[0] = true;
        }

        if (dstTest == 1)
        {
            unpinnedMalloc[1] = true;
        }
        else if (dstTest == 2)
        {
            hostMalloc[1] = true;
        }
        else if (dstTest == 3)
        {
            hostRegister[1] = true;
        }

        numIter = Iterations[test / (NUM_SIZES * NUM_SUBTESTS)];

        if (hostMalloc[0])
        {
            err = hipHostMalloc((void**)&srcBuffer, bufSize_, 0);
            setData(srcBuffer, bufSize_, 0xd0);
            CHECK_RESULT(err != hipSuccess, "hipHostMalloc failed");
        }
        else if (hostRegister[0])
        {
            memptr[0] = malloc(bufSize_ + 4096);
            alignedmemptr[0] = (void*)(((size_t)memptr[0] + 4095) & ~4095);
            srcBuffer = alignedmemptr[0];
            setData(srcBuffer, bufSize_, 0xd0);
            err = hipHostRegister(srcBuffer, bufSize_, 0);
            CHECK_RESULT(err != hipSuccess, "hipHostRegister failed");
        }
        else if (unpinnedMalloc[0])
        {
            memptr[0] = malloc(bufSize_ + 4096);
            alignedmemptr[0] = (void*)(((size_t)memptr[0] + 4095) & ~4095);
            srcBuffer = alignedmemptr[0];
            setData(srcBuffer, bufSize_, 0xd0);
        }
        else
        {
            err = hipMalloc(&srcBuffer, bufSize_);
            CHECK_RESULT(err != hipSuccess, "hipMalloc failed");
            err = hipMemset(srcBuffer, 0xd0, bufSize_);
            CHECK_RESULT(err != hipSuccess, "hipMemset failed");
        }

        if (hostMalloc[1])
        {
            err = hipHostMalloc((void**)&dstBuffer, bufSize_, 0);
            CHECK_RESULT(err != hipSuccess, "hipHostMalloc failed");
        }
        else if (hostRegister[1])
        {
            memptr[1] = malloc(bufSize_ + 4096);
            alignedmemptr[1] = (void*)(((size_t)memptr[1] + 4095) & ~4095);
            dstBuffer = alignedmemptr[1];
            err = hipHostRegister(dstBuffer, bufSize_, 0);
            CHECK_RESULT(err != hipSuccess, "hipHostRegister failed");
        }
        else if (unpinnedMalloc[1])
        {
            memptr[1] = malloc(bufSize_ + 4096);
            alignedmemptr[1] = (void*)(((size_t)memptr[1] + 4095) & ~4095);
            dstBuffer = alignedmemptr[1];
        }
        else
        {
            err = hipMalloc(&dstBuffer, bufSize_);
            CHECK_RESULT(err != hipSuccess, "hipMalloc failed");
        }

        CPerfCounter timer;

        //warm up
        err = hipMemcpy(dstBuffer, srcBuffer, bufSize_, hipMemcpyDefault);
        CHECK_RESULT(err, "hipMemcpy failed");

        timer.Reset();
        timer.Start();
        for (unsigned int i = 0; i < numIter; i++)
        {
            err = hipMemcpyAsync(dstBuffer, srcBuffer, bufSize_, hipMemcpyDefault, NULL);
            CHECK_RESULT(err, "hipMemcpyAsync failed");
        }
        err = hipDeviceSynchronize();
        CHECK_RESULT(err, "hipDeviceSynchronize failed");
        timer.Stop();
        double sec = timer.GetElapsedTime();

        // Buffer copy bandwidth in GB/s
        double perf = ((double)bufSize_*numIter*(double)(1e-09)) / sec;

        const char *strSrc = NULL;
        const char *strDst = NULL;
         if (hostMalloc[0])
            strSrc = "hHM";
        else if (hostRegister[0])
            strSrc = "hHR";
        else if (unpinnedMalloc[0])
            strSrc = "unp";
        else
            strSrc = "hM";

        if (hostMalloc[1])
            strDst = "hHM";
        else if (hostRegister[1])
            strDst = "hHR";
        else if (unpinnedMalloc[1])
            strDst = "unp";
        else
            strDst = "hM";
        // Double results when src and dst are both on device
        if ((!hostMalloc[0] && !hostRegister[0] && !unpinnedMalloc[0]) &&
            (!hostMalloc[1] && !hostRegister[1] && !unpinnedMalloc[1]))
            perf *= 2.0;
        // Double results when src and dst are both in sysmem
        if ((hostMalloc[0] || hostRegister[0] || unpinnedMalloc[0]) &&
            (hostMalloc[1] || hostRegister[1] || unpinnedMalloc[1]))
            perf *= 2.0;

        char buf[256];
        SNPRINTF(buf, sizeof(buf), "HIPPerfBufferCopySpeed[%d]\t(%8d bytes)\ts:%s d:%s\ti:%4d\t(GB/s) perf\t%f",
                test, bufSize_, strSrc, strDst, numIter, (float)perf);
        printf("%s\n", buf);

        // Verification
        void* temp = malloc(bufSize_ + 4096);
        void* chkBuf = (void*)(((size_t)temp + 4095) & ~4095);
        err = hipMemcpy(chkBuf, dstBuffer, bufSize_, hipMemcpyDefault);
        CHECK_RESULT(err, "hipMemcpy failed");
        checkData(chkBuf, bufSize_, 0xd0);
        free(temp);

        //Free src
        if (hostMalloc[0])
        {
            hipHostFree(srcBuffer);
        }
        else if (hostRegister[0])
        {
            hipHostUnregister(srcBuffer);
            free(memptr[0]);
        }
        else if (unpinnedMalloc[0])
        {
            free(memptr[0]);
        }
        else
        {
            hipFree(srcBuffer);
        }

        //Free dst
        if (hostMalloc[1])
        {
            hipHostFree(dstBuffer);
        }
        else if (hostRegister[1])
        {
            hipHostUnregister(dstBuffer);
            free(memptr[1]);
        }
        else if (unpinnedMalloc[1])
        {
            free(memptr[1]);
        }
        else
        {
            hipFree(dstBuffer);
        }
    }

    passed();
}
