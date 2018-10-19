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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * RUN: %t
 * HIT_END
 */


#include "hip/hip_runtime.h"
#include "test_common.h"

#include <algorithm>
#include <vector>

unsigned p_streams = 16;
int p_repeat = 10;
int p_db = 0;

using namespace std;

template <typename T>
__global__ void vectorADDRepeat(const T* A_d, const T* B_d, T* C_d, size_t NELEM,
                                int repeat) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (int j = 1; j <= repeat; j++) {
        for (size_t i = offset; i < NELEM; i += stride) {
            C_d[i] = A_d[i] * j + B_d[i] * j;
        }
    };
}


//------
// Structure for one stream - includes the stream + data buffers that are used by the stream.
template <typename T>
class Streamer {
   public:
    Streamer(size_t numElements, bool useNullStream = false);
    ~Streamer();
    void enqueAsync();
    void queryUntilComplete();

    void reset();
    void H2D();
    void D2H();


   public:
    T* _A_h;
    T* _B_h;
    T* _C_h;

    T* _A_d;
    T* _B_d;
    T* _C_d;

    hipStream_t _stream;
    hipEvent_t _event;

    size_t _numElements;
};

template <typename T>
Streamer<T>::Streamer(size_t numElements, bool useNullStream) : _numElements(numElements) {
    HipTest::initArrays(&_A_d, &_B_d, &_C_d, &_A_h, &_B_h, &_C_h, numElements, true);

    if (useNullStream) {
        _stream = 0x0;
    } else {
        HIPCHECK(hipStreamCreate(&_stream));
    }
    HIPCHECK(hipEventCreate(&_event));

    H2D();
};

template <typename T>
void Streamer<T>::H2D() {
    HIPCHECK(hipMemcpy(_A_d, _A_h, _numElements * sizeof(T), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(_B_d, _B_h, _numElements * sizeof(T), hipMemcpyHostToDevice));
}

template <typename T>
void Streamer<T>::D2H() {
    HIPCHECK(hipMemcpy(_C_h, _C_d, _numElements * sizeof(T), hipMemcpyDeviceToHost));
}

template <typename T>
void Streamer<T>::reset() {
    HipTest::setDefaultData(_numElements, _A_h, _B_h, _C_h);
    H2D();
}


template <typename T>
void Streamer<T>::enqueAsync() {
    printf("testing: %s  numElements=%zu size=%6.2fMB\n", __func__, _numElements,
           _numElements * sizeof(T) / 1024.0 / 1024.0);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, _numElements);
    hipLaunchKernelGGL(vectorADDRepeat, dim3(blocks), dim3(threadsPerBlock), 0, _stream,
                    static_cast<const T*>(_A_d), static_cast<const T*>(_B_d), _C_d, _numElements,
                    p_repeat);
}

template <typename T>
void Streamer<T>::queryUntilComplete() {
    int numQueries = 0;
    hipError_t e = hipSuccess;
    do {
        numQueries++;
        e = hipStreamQuery(_stream);
    } while (e != hipSuccess);

    printf("completed after %d queries\n", numQueries);
};


//---
// Parse arguments specific to this test.
void parseMyArguments(int argc, char* argv[]) {
    int more_argc = HipTest::parseStandardArguments(argc, argv, false);

    // parse args for this test:
    for (int i = 1; i < more_argc; i++) {
        const char* arg = argv[i];

        if (!strcmp(arg, "--streams")) {
            if (++i >= argc || !HipTest::parseUInt(argv[i], &p_streams)) {
                failed("Bad streams argument");
            }
        } else if (!strcmp(arg, "--repeat") || (!strcmp(arg, "-r"))) {
            if (++i >= argc || !HipTest::parseInt(argv[i], &p_repeat)) {
                failed("Bad repeat argument");
            }
        } else {
            failed("Bad argument '%s'", arg);
        }
    };
};


void printBuffer(std::string name, int* f, size_t numElements) {
    std::cout << name << "\n";
    for (size_t i = 0; i < numElements; i++) {
        printf("%5zu: %d\n", i, f[i]);
    }
}


//---
int main(int argc, char* argv[]) {
    HipTest::parseStandardArguments(argc, argv, false);
    parseMyArguments(argc, argv);

    typedef Streamer<int> IntStreamer;

    std::vector<IntStreamer*> streamers;

    size_t numElements = N;

    int* expected_H = (int*)malloc(numElements * sizeof(int));


    auto nullStreamer = new IntStreamer(numElements, true);

    // Expected resultr - last streamer runs vectorADDRepeat, then nullstreamer adds
    // lastStreamer->_C_d + lastStreamer->_C_d
    for (size_t i = 0; i < numElements; i++) {
        expected_H[i] =
            ((nullStreamer->_A_h[i]) * p_repeat + (nullStreamer->_B_h[i]) * p_repeat) * 2;
    }


    for (int i = 0; i < p_streams; i++) {
        IntStreamer* s = new IntStreamer(numElements);
        streamers.push_back(s);
    }
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    for (int s = 1; s < p_streams; s++) {
        if (p_tests & (1 << s)) {
            printf("==> Test %x runAsnc, #streams=%d\n", (1 << s), s);
            nullStreamer->reset();

            for (int i = 0; i < s; i++) {
                streamers[i]->enqueAsync();
            }

            auto lastStreamer = streamers[s - 1];

            // Dispatch to NULL stream, should wait for prior async activity to complete before
            // beginning:
            hipLaunchKernelGGL(vectorADDRepeat, dim3(blocks), dim3(threadsPerBlock), 0,
                            0 /*nullstream*/, static_cast<const int*>(lastStreamer->_C_d),
                            static_cast<const int*>(lastStreamer->_C_d), nullStreamer->_C_d,
                            numElements, 1 /*repeat*/);


            if (p_db) {
                HIPCHECK(hipDeviceSynchronize());
                lastStreamer->D2H();
                printBuffer("lastStream _A_h", lastStreamer->_A_h, min(numElements, size_t(20)));
                printBuffer("lastStream _B_h", lastStreamer->_B_h, min(numElements, size_t(20)));
                printBuffer("lastStream _C_h", lastStreamer->_C_h, min(numElements, size_t(20)));
            }
            nullStreamer->D2H();
            HIPCHECK(hipDeviceSynchronize());

            HipTest::checkTest(expected_H, nullStreamer->_C_h, numElements);
        }
    }


    for (int s = 1; s < p_streams; s += 2) {
        unsigned tmask = (0x10000 | (1 << s));
        if (p_tests & tmask) {
            nullStreamer->reset();
            printf("==> Test %x runAsnc-odd-only, #streams=%d\n", tmask, s);
            for (int i = 0; i < s; i++) {
                // RUn just odd streams so we have some empty ones to examine/optimize:
                if (i & 0x1) {
                    streamers[i]->enqueAsync();
                }
            }
            auto lastStreamer = streamers[s - 1];

            // Dispatch to NULL stream, should wait for prior async activity to complete before
            // beginning:
            hipLaunchKernelGGL(vectorADDRepeat, dim3(blocks), dim3(threadsPerBlock), 0,
                            0 /*nullstream*/, static_cast<const int*>(lastStreamer->_C_d),
                            static_cast<const int*>(lastStreamer->_C_d), nullStreamer->_C_d,
                            numElements, 1 /*repeat*/);

            nullStreamer->D2H();

            HIPCHECK(hipDeviceSynchronize());

            HipTest::checkTest(expected_H, nullStreamer->_C_h, numElements);
        }
    }

    // Expected resultr - last streamer runs vectorADDRepeat
    for (size_t i = 0; i < numElements; i++) {
        expected_H[i] = ((nullStreamer->_A_h[i]) * p_repeat + (nullStreamer->_B_h[i]) * p_repeat);
    }

    if (p_tests & 0x20000) {
        assert(p_streams >= 2);  // need a couple streams in order to run this test.
        nullStreamer->reset();
        printf("\n==> Test hipStreamSynchronize with defaultStream \n");

        // Enqueue a long-running job to stream1
        streamers[0]->enqueAsync();

        // Check to see if synchronizing on a null stream synchronizes all other streams or just the
        // null stream. This function follows null stream semantics and will wait for all other
        // blocking streams before returning. This will wait on the host
        HIPCHECK(hipStreamSynchronize(0));

        // Copy with stream1, this could go async if the streamSync doesn't synchronize ALL the
        // streams.
        HIPCHECK(hipMemcpyAsync(streamers[0]->_C_h, streamers[0]->_C_d,
                                streamers[0]->_numElements * sizeof(int), hipMemcpyDeviceToHost,
                                streamers[1]->_stream));


        HIPCHECK(hipDeviceSynchronize());

        HipTest::checkTest(expected_H, streamers[0]->_C_h, numElements);
    }


    passed();
}
