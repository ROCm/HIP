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

// Test under-development.  Calls async mem-copy API, experiment with functionality.

#include "hip/hip_runtime.h"
#include "test_common.h"
#include <vector>
#include <limits>
unsigned p_streams = 6;
unsigned p_db = 0;


template <typename T>
__global__ void
addOne( const T *A_d,
        T *C_d,
        size_t NELEM)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<NELEM; i+=stride) {
        C_d[i] = A_d[i] + (T)1;
        //C_d[i] = (T)1;
	}
}


template <typename T>
__global__ void
addOneReverse( const T *A_d,
        T *C_d,
        int64_t NELEM)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (int64_t i=NELEM-stride+offset; i>=0; i-=stride) {
        C_d[i] = A_d[i] + (T)1;
        //C_d[i] = (T)1;
	}
}


//------
// Structure for one stream;
template <typename T>
class Streamer {
public:
    Streamer(T *input, size_t numElements, bool reverse);
    ~Streamer();
    void runAsyncAfter(Streamer<T> *depStreamer);
    void runAsyncWaitSameStream();
    void queryUntilComplete();

    void syncAndCheck(int streamerNum, T initValue, T expectedOffset);

    hipEvent_t event() { return _event; };

    T *C_d() { return _C_d; };


private:
    T *_C_h;

    T *_A_d;
    T *_C_d;

    hipStream_t _stream;
    hipEvent_t  _event;

    size_t      _numElements;
    bool        _reverse;
};


template <typename T>
Streamer<T>::Streamer(T * A_d, size_t numElements, bool reverse) :
    _A_d(A_d),
    _numElements(numElements),
    _reverse(reverse)
{
    size_t sizeElements = numElements * sizeof(int);

    HIPCHECK(hipMalloc(&_C_d, sizeElements));
    HIPCHECK(hipHostMalloc(&_C_h, sizeElements));

    HIPCHECK(hipMemset(_C_d, -1, sizeElements));
    HIPCHECK(hipMemset(_C_h, -2, sizeElements));

    HIPCHECK(hipStreamCreate(&_stream));
    HIPCHECK(hipEventCreate(&_event));
};


template <typename T>
void Streamer<T>::runAsyncAfter(Streamer<T> *depStreamer)
{
    if (p_db) {
      printf ("testing: %s  numElements=%zu size=%6.2fMB\n", __func__, _numElements, _numElements * sizeof(T) / 1024.0/1024.0);
    }

    if (depStreamer) {
        HIPCHECK(hipStreamWaitEvent(_stream, depStreamer->event(), 0));
    }

    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, _numElements);
    if (_reverse) {
        hipLaunchKernelGGL(addOneReverse , dim3(blocks), dim3(threadsPerBlock), 0, _stream,    _A_d, _C_d, _numElements);
    } else {
        hipLaunchKernelGGL(addOne,         dim3(blocks), dim3(threadsPerBlock), 0, _stream,    _A_d, _C_d, _numElements);
    }
    HIPCHECK(hipEventRecord(_event, _stream));
}


template <typename T>
void Streamer<T>::runAsyncWaitSameStream()
{
    printf ("testing: %s  numElements=%zu size=%6.2fMB\n", __func__, _numElements, _numElements * sizeof(T) / 1024.0/1024.0);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, _numElements);
    if (_reverse) {
        hipLaunchKernelGGL(addOneReverse , dim3(blocks), dim3(threadsPerBlock), 0, _stream,    _A_d, _C_d, _numElements);
    } else {
        hipLaunchKernelGGL(addOne,         dim3(blocks), dim3(threadsPerBlock), 0, _stream,    _A_d, _C_d, _numElements);
    }

    // Test case where hipStreamWaitEvent waits on same event we just placed into the queue.
    HIPCHECK(hipEventRecord(_event, _stream));
    HIPCHECK(hipStreamWaitEvent(_stream, _event, 0));
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

    printf ("info: hipStreamQuery completed after %d queries\n", numQueries);
};


template <typename T>
void Streamer<T>::syncAndCheck(int streamerNum, T initValue, T expectedOffset)
{
    HIPCHECK(hipMemcpyAsync(_C_h, _C_d, _numElements*sizeof(T), hipMemcpyDeviceToHost, _stream));
    HIPCHECK(hipStreamSynchronize(_stream));

    T expected = initValue + expectedOffset;

    for (size_t i=0; i<_numElements; i++) {
        if (_C_h[i] != expected) {
            failed("for streamer:%d  _C_h[%zu] (%d)  !=  expected(%d)\n", streamerNum, i, _C_h[i], expected);
        }
    }
}

   

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

    typedef Streamer<int> IntStreamer;

    std::vector<IntStreamer *> streamers;

    size_t numElements = N;
    size_t sizeElements = numElements * sizeof(int);

    assert (sizeElements <= std::numeric_limits<int64_t>::max());


    int initValue = 1000;

    int * initArray_d, *initArray_h;
    HIPCHECK(hipMalloc(&initArray_d, sizeElements));
    HIPCHECK(hipHostMalloc(&initArray_h, sizeElements));
    for (size_t i=0; i<numElements; i++) {
        initArray_h[i] = initValue;
    }
    HIPCHECK(hipMemcpy(initArray_d, initArray_h, sizeElements, hipMemcpyHostToDevice));
    


    for (int i=0; i<p_streams; i++) {
        IntStreamer * s = new IntStreamer(i ? streamers.back()->C_d() : initArray_d, numElements, i&1 /*reverse?*/);
        streamers.push_back(s);
    }

    if (p_tests & 0x1) {
        printf ("==> Test 0x1 runAsyncAfter\n");
        for (int i=0; i<p_streams; i++) {
            streamers[i]->runAsyncAfter(i ? streamers[i-1] : NULL);
        }
        HIPCHECK(hipDeviceSynchronize());

        for (int i=0; i<p_streams; i++) {
            streamers[i]->syncAndCheck(i+1, initValue, i+1);
        }
    }

    if (p_tests & 0x2) {
        printf ("==> Test 0x2 queryUntilComplete\n");
        for (int i=0; i<p_streams; i++) {
            streamers[i]->runAsyncAfter(i ? streamers[i-1] : NULL);
            streamers[i]->queryUntilComplete();
        }
        HIPCHECK(hipDeviceSynchronize());
    }

    if (p_tests & 0x4) {
        printf ("==> Test 0x4 try null stream"); 
        hipStreamQuery(0/* try null stream*/);

    }

    if (p_tests & 0x8) {
        printf ("==> Test 0x8 runAsyncWaitSameStream\n");
        for (int i=0; i<p_streams; i++) {
            streamers[i]->runAsyncWaitSameStream();
        }
        HIPCHECK(hipDeviceSynchronize());
    }


    passed();
}
