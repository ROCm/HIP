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

#include "hip_runtime.h"
#include "test_common.h"
unsigned p_streams = 6;


//------
// Structure for one stream;
template <typename T> 
class Streamer {
public:
    Streamer(size_t numElements);
    ~Streamer();
    void runAsync();
    void waitComplete();

private:
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
Streamer<T>::Streamer(size_t numElements) :
    _numElements(numElements)
{
    HipTest::initArrays (&_A_d, &_B_d, &_C_d, &_A_h, &_B_h, &_C_h, numElements, true);

    HIPCHECK(hipStreamCreate(&_stream));
    HIPCHECK(hipEventCreate(&_event));
};

template <typename T>
void Streamer<T>::runAsync()
{
    printf ("testing: %s  numElements=%zu size=%6.2fMB\n", __func__, _numElements, _numElements * sizeof(T) / 1024.0/1024.0);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, _numElements);
    hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, _stream, _A_d, _B_d, _C_d, _numElements);
    HIPCHECK(hipEventRecord(_event, _stream));
    HIPCHECK(hipStreamWaitEvent(_stream, _event, 0));

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
    HipTest::parseStandardArguments(argc, argv, true);
    parseMyArguments(argc, argv);

    typedef Streamer<float> FloatStreamer;

    std::vector<FloatStreamer *> streamers;

    size_t numElements = N;

    for (int i=0; i<p_streams; i++) {
        FloatStreamer * s = new FloatStreamer(numElements);
        streamers.push_back(s);
    }

    for (int i=0; i<p_streams; i++) {
        streamers[i]->runAsync();
    }

    HIPCHECK(hipDeviceSynchronize());

    passed();
}
