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
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */


#include "hip/hip_runtime.h"
#include "test_common.h"
#include <vector>
unsigned p_streams = 6;
int      p_repeat = 10;


template <typename T>
__global__ void
vectorADDRepeat(hipLaunchParm lp,
            const T *A_d,
            const T *B_d,
            T *C_d,
            size_t NELEM,
            int repeat)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (int j=1; j<=repeat;j++) {
        for (size_t i=offset; i<NELEM; i+=stride) {
            C_d[i] = A_d[i]*j + B_d[i]*j;
        }
    };
}




//------
// Structure for one stream - includes the stream + data buffers that are used by the stream.
template <typename T>
class Streamer {
public:
    Streamer(size_t numElements, bool useNullStream=false);
    ~Streamer();
    void enqueAsync();
    void queryUntilComplete();


public:
    T *_A_h;
    T *_B_h;
    T *_C_h;

    T *_A_d;
    T *_B_d;
    T *_C_d;

    hipStream_t _stream;
    hipEvent_t  _event;

    size_t      _numElements;
};

template <typename T>
Streamer<T>::Streamer(size_t numElements, bool useNullStream) :
    _numElements(numElements)
{
    HipTest::initArrays (&_A_d, &_B_d, &_C_d, &_A_h, &_B_h, &_C_h, numElements, true);

    if (useNullStream) {
        _stream = 0x0;
    } else {
        HIPCHECK(hipStreamCreate(&_stream));
    }
    HIPCHECK(hipEventCreate(&_event));
};

template <typename T>
void Streamer<T>::enqueAsync()
{
    printf ("testing: %s  numElements=%zu size=%6.2fMB\n", __func__, _numElements, _numElements * sizeof(T) / 1024.0/1024.0);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, _numElements);
    hipLaunchKernel(vectorADDRepeat, dim3(blocks), dim3(threadsPerBlock), 0, _stream, _A_d, _B_d, _C_d, _numElements, p_repeat);

}

template <typename T>
void Streamer<T>::queryUntilComplete()
{
    int numQueries = 0;
    hipError_t e = hipSuccess;
    do {
        numQueries++;
        e = hipStreamQuery(_stream);
    } while (e != hipSuccess) ;

    printf ("completed after %d queries\n", numQueries);
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





//---
int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, false);
    parseMyArguments(argc, argv);

    typedef Streamer<float> FloatStreamer;

    std::vector<FloatStreamer *> streamers;

    size_t numElements = N;

    float *expected_H = (float*)malloc(numElements*sizeof(float));


    auto nullStreamer = new FloatStreamer(numElements, true);
    for (size_t i=0; i<numElements; i++) {
        expected_H[i] = nullStreamer->_A_h[i]*p_repeat + nullStreamer->_B_h[i] * p_repeat;
    }


    for (int i=0; i<p_streams; i++) {
        FloatStreamer * s = new FloatStreamer(numElements);
        streamers.push_back(s);
    }

    if (p_tests & 0x1) {
        printf ("==> Test 0x1 runAsnc\n");
        for (int i=0; i<p_streams; i++) {
            streamers[i]->enqueAsync();
        }

        auto lastStreamer = streamers[p_streams - 1];

        // Dispatch to NULL stream, should wait for prior async activity to complete.
        unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);
        hipLaunchKernel(vectorADDRepeat, dim3(blocks), dim3(threadsPerBlock), 0, 0/*nullstream*/, lastStreamer->_C_d, lastStreamer->_C_d, nullStreamer->_C_d, numElements, 1/*repeat*/);
        HIPCHECK(hipMemcpy(nullStreamer->_C_h, nullStreamer->_C_d, numElements*sizeof(float), hipMemcpyDeviceToHost));
        HIPCHECK(hipStreamSynchronize(0));


        HipTest::checkTest(expected_H, nullStreamer->_C_h, numElements); 
    }


    if (p_tests & 0x2) {
        printf ("==> Test 0x2 runAsnc-odd-only\n");
        for (int i=0; i<p_streams; i++) {
            if (i & 0x1) {
                streamers[i]->enqueAsync();
            }
        }
    }


    passed();
}
