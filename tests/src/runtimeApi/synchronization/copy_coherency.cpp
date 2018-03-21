/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

// ROCM_TARGET=gfx900 hipcc --genco memcpyInt.device.cpp -o memcpyInt.hsaco
// hipcc copy_coherency.cpp  -I ~/X/HIP/tests/src/ ~/X/HIP/tests/src/test_common.cpp


// TODO - add code object support here.
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * RUN: %t
 * HIT_END
 */


// Test cache management (fences) and synchronization between kernel and copy commands.
// Exhaustively tests 3 command types (copy, kernel, module kernel),
// many sync types (see SyncType), followed by another command, across a sweep
// of data sizes designed to stress various levels of the memory hierarchy.

#include "hip/hip_runtime.h"
#include "test_common.h"

// TODO - turn this back on when test infra can copy the module files to use as test inputs.
#define SKIP_MODULE_KERNEL 1


class MemcpyFunction {
   public:
    MemcpyFunction(const char* fileName, const char* functionName) {
        load(fileName, functionName);
    };
    void load(const char* fileName, const char* functionName);
    void launch(int* dst, const int* src, size_t numElements, hipStream_t s);

   private:
    hipFunction_t _function;
    hipModule_t _module;
};


void MemcpyFunction::load(const char* fileName, const char* functionName) {
#if SKIP_MODULE_KERNEL != 1
    HIPCHECK(hipModuleLoad(&_module, fileName));
    HIPCHECK(hipModuleGetFunction(&_function, _module, functionName));
#endif
};


void MemcpyFunction::launch(int* dst, const int* src, size_t numElements, hipStream_t s) {
    struct {
        int* _dst;
        const int* _src;
        size_t _numElements;
    } args;

    args._dst = dst;
    args._src = src;
    args._numElements = numElements;

    size_t size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);
    HIPCHECK(hipModuleLaunchKernel(_function, blocks, 1, 1, threadsPerBlock, 1, 1,
                                   0 /*dynamicShared*/, s, NULL, (void**)&config));
};

bool g_warnOnFail = true;
// int g_elementSizes[] = {1, 16, 1024, 524288, 16*1000*1000}; // TODO
int g_elementSizes[] = {128 * 1000, 256 * 1000, 16 * 1000 * 1000};

MemcpyFunction g_moduleMemcpy("memcpyInt.hsaco", "memcpyIntKernel");


// Set value of array to specified 32-bit integer:
__global__ void memsetIntKernel(int* ptr, const int val, size_t numElements) {
    int gid = (blockIdx.x * blockDim.x + threadIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (size_t i = gid; i < numElements; i += stride) {
        ptr[i] = val;
    }
};

__global__ void memcpyIntKernel(int* dst, const int* src, size_t numElements) {
    int gid = (blockIdx.x * blockDim.x + threadIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (size_t i = gid; i < numElements; i += stride) {
        dst[i] = src[i];
    }
};


// CHeck arrays in reverse order, to more easily detect cases where
// the copy is "partially" done.
void checkReverse(const int* ptr, int numElements, int expected) {
    int mismatchCnt = 0;
    for (int i = numElements - 1; i >= 0; i--) {
        if (ptr[i] != expected) {
            fprintf(stderr, "%s**error: i=%d, ptr[i] == (%x) , does not equal expected (%x)\n%s",
                    KRED, i, ptr[i], expected, KNRM);
            if (!g_warnOnFail) {
                assert(ptr[i] == expected);
            }
            if (++mismatchCnt >= 10) {
                break;
            }
        }
    }

    fprintf(stderr, "test:   OK\n");
}

#define ENUM_CASE_STR(x)                                                                           \
    case x:                                                                                        \
        return #x

enum CmdType { COPY, KERNEL, MODULE_KERNEL, MAX_CmdType };


const char* CmdTypeStr(CmdType c) {
    switch (c) {
        ENUM_CASE_STR(COPY);
        ENUM_CASE_STR(KERNEL);
        ENUM_CASE_STR(MODULE_KERNEL);
        default:
            return "UNKNOWN";
    };
}


enum SyncType {
    NONE,
    EVENT_QUERY,
    EVENT_SYNC,
    STREAM_WAIT_EVENT,
    STREAM_QUERY,
    STREAM_SYNC,
    DEVICE_SYNC,
    MAX_SyncType
};


const char* SyncTypeStr(SyncType s) {
    switch (s) {
        ENUM_CASE_STR(NONE);
        ENUM_CASE_STR(EVENT_QUERY);
        ENUM_CASE_STR(EVENT_SYNC);
        ENUM_CASE_STR(STREAM_WAIT_EVENT);
        ENUM_CASE_STR(STREAM_QUERY);
        ENUM_CASE_STR(STREAM_SYNC);
        ENUM_CASE_STR(DEVICE_SYNC);
        default:
            return "UNKNOWN";
    };
};


void runCmd(CmdType cmd, int* dst, const int* src, hipStream_t s, size_t numElements) {
    switch (cmd) {
        case COPY:
            HIPCHECK(
                hipMemcpyAsync(dst, src, numElements * sizeof(int), hipMemcpyDeviceToDevice, s));
            break;
        case KERNEL: {
            unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);
            hipLaunchKernelGGL(memcpyIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, s, dst, src,
                               numElements);
        } break;
        case MODULE_KERNEL:
            g_moduleMemcpy.launch(dst, src, numElements, s);
            break;
        default:
            failed("unknown cmd=%d type", cmd);
    };
}

void resetInputs(int* Ad, int* Bd, int* Cd, int* Ch, size_t numElements, int expected) {
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);
    hipLaunchKernelGGL(memsetIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, hipStream_t(0), Ad,
                       expected, numElements);
    hipLaunchKernelGGL(memsetIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, hipStream_t(0), Bd,
                       0xDEADBEEF,
                       numElements);  // poison with bad value to ensure is overwritten correctly
    hipLaunchKernelGGL(memsetIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, hipStream_t(0), Bd,
                       0xF000BA55,
                       numElements);  // poison with bad value to ensure is overwritten correctly
    memset(Ch, 13,
           numElements * sizeof(int));  // poison with bad value to ensure is overwritten correctly
    HIPCHECK(hipDeviceSynchronize());
}

// Intended to test proper synchronization and cache flushing between CMDA and CMDB.
// CMD are of type CmdType. All command copy memory, using either hipMemcpyAsync or kernel
// implementations. CmdA copies from Ad to Bd, Some form of synchronization is applied. Then cmdB
// copies from Bd to Cd.
//
// Cd is then copied to host Ch using a memory copy.
//
// Correct result at the end is that Ch contains the contents originally in Ad (integer 0x42)
void runTestImpl(CmdType cmdAType, SyncType syncType, CmdType cmdBType, hipStream_t stream1,
                 hipStream_t stream2, int numElements, int* Ad, int* Bd, int* Cd, int* Ch,
                 int expected) {
    hipEvent_t e;
    HIPCHECK(hipEventCreateWithFlags(&e, 0));

    resetInputs(Ad, Bd, Cd, Ch, numElements, expected);

    const size_t sizeElements = numElements * sizeof(int);
    fprintf(stderr, "test: runTest with %zu bytes (%6.2f MB) cmdA=%s; sync=%s; cmdB=%s\n",
            sizeElements, (double)(sizeElements / 1024.0), CmdTypeStr(cmdAType),
            SyncTypeStr(syncType), CmdTypeStr(cmdBType));

    if (SKIP_MODULE_KERNEL && ((cmdAType == MODULE_KERNEL) || (cmdBType == MODULE_KERNEL))) {
        fprintf(stderr, "warn: skipping since test infra does not yet support modules\n");
        return;
    }


    // Step A:
    runCmd(cmdAType, Bd, Ad, stream1, numElements);


    // Sync in-between?
    switch (syncType) {
        case NONE:
            break;
        case EVENT_QUERY: {
            hipError_t st = hipErrorNotReady;
            HIPCHECK(hipEventRecord(e, stream1));
            do {
                st = hipEventQuery(e);
            } while (st == hipErrorNotReady);
            HIPCHECK(st);
        } break;
        case EVENT_SYNC:
            HIPCHECK(hipEventRecord(e, stream1));
            HIPCHECK(hipEventSynchronize(e));
            break;
        case STREAM_WAIT_EVENT:
            HIPCHECK(hipEventRecord(e, stream1));
            HIPCHECK(hipStreamWaitEvent(stream2, e, 0));
            break;
        case STREAM_QUERY: {
            hipError_t st = hipErrorNotReady;
            do {
                st = hipStreamQuery(stream1);
            } while (st == hipErrorNotReady);
            HIPCHECK(st);
        } break;
        case STREAM_SYNC:
            HIPCHECK(hipStreamSynchronize(stream1));
            break;
        case DEVICE_SYNC:
            HIPCHECK(hipDeviceSynchronize());
            break;
        default:
            fprintf(stderr, "warning: unknown sync type=%s", SyncTypeStr(syncType));
            return;  // FIXME, this doesn't clean up
                     // failed("unknown sync type=%s", SyncTypeStr(syncType));
    };


    runCmd(cmdBType, Cd, Bd, stream2, numElements);


    // Copy back to host, use async copy to avoid any extra synchronization that might mask issues.
    HIPCHECK(hipMemcpyAsync(Ch, Cd, sizeElements, hipMemcpyDeviceToHost, stream2));
    HIPCHECK(hipStreamSynchronize(stream2));

    checkReverse(Ch, numElements, expected);

    HIPCHECK(hipEventDestroy(e));
};


void testWrapper(size_t numElements) {
    const size_t sizeElements = numElements * sizeof(int);
    const int expected = 0x42;
    int *Ad, *Bd, *Cd, *Ch;

    HIPCHECK(hipMalloc(&Ad, sizeElements));
    HIPCHECK(hipMalloc(&Bd, sizeElements));
    HIPCHECK(hipMalloc(&Cd, sizeElements));
    HIPCHECK(hipHostMalloc(&Ch, sizeElements));  // Ch is the end array


    hipStream_t stream1, stream2;

    HIPCHECK(hipStreamCreate(&stream1));
    HIPCHECK(hipStreamCreate(&stream2));


    HIPCHECK(hipDeviceSynchronize());
    fprintf(stderr, "test: init complete, start running tests\n");


    runTestImpl(COPY, EVENT_SYNC, KERNEL, stream1, stream2, numElements, Ad, Bd, Cd, Ch, expected);

    for (int cmdA = 0; cmdA < MAX_CmdType; cmdA++) {
        for (int cmdB = 0; cmdB < MAX_CmdType; cmdB++) {
            for (int syncMode = 0; syncMode < MAX_SyncType; syncMode++) {
                switch (syncMode) {
                    // case NONE::
                    case EVENT_QUERY:
                    case EVENT_SYNC:
                    case STREAM_WAIT_EVENT:
                    // case STREAM_QUERY:
                    case STREAM_SYNC:
                    case DEVICE_SYNC:
                        runTestImpl(CmdType(cmdA), SyncType(syncMode), CmdType(cmdB), stream1,
                                    stream2, numElements, Ad, Bd, Cd, Ch, expected);
                        break;
                    default:
                        break;
                }
            }
        }
    }

#if 0
    runTestImpl(COPY, STREAM_SYNC, MODULE_KERNEL, stream1, stream2, numElements, Ad, Bd, Cd, Ch, expected);
    runTestImpl(COPY, STREAM_SYNC, KERNEL, stream1, stream2, numElements, Ad, Bd, Cd, Ch, expected);
    runTestImpl(COPY, STREAM_WAIT_EVENT, MODULE_KERNEL, stream1, stream2, numElements, Ad, Bd, Cd, Ch, expected);

    runTestImpl(COPY, STREAM_WAIT_EVENT, KERNEL, stream1, stream2, numElements, Ad, Bd, Cd, Ch, expected);
#endif

    HIPCHECK(hipFree(Ad));
    HIPCHECK(hipFree(Bd));
    HIPCHECK(hipFree(Cd));
    HIPCHECK(hipHostFree(Ch));

    HIPCHECK(hipStreamDestroy(stream1));
    HIPCHECK(hipStreamDestroy(stream2));
}


int main(int argc, char* argv[]) {
    for (int index = 0; index < sizeof(g_elementSizes) / sizeof(int); index++) {
        size_t numElements = g_elementSizes[index];
        testWrapper(numElements);
    }

    passed();
}


// TODO
// - test environment variables
