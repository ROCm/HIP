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

// Test under-development.  Calls async mem-copy API, experiment with functionality.

#include "hip/hip_runtime.h"
#include "test_common.h"
#include <vector>
#include <limits>
unsigned p_streams = 8;
unsigned p_db = 0;
unsigned p_count = 100;





//------
// Structure for one stream;
template <typename T>
class Streamer {

#define COMMAND_ADD_FORWARD 0
#define COMMAND_ADD_REVERSE 1
#define COMMAND_COPY        2


public:
    Streamer(int deviceId, T *input, size_t numElements, int commandType);
    ~Streamer();
    void runAsyncAfter(Streamer<T> *depStreamer, bool waitSameStream=false);
    void runAsyncWaitSameStream();
    void queryUntilComplete();

    size_t check(int streamerNum, T initValue, T expectedOffset, bool expectPass=true);
    void copyToHost(hipStream_t copyStream);

    hipEvent_t event() { return _event; };

    int deviceId() const { return _deviceId; };
    size_t mismatchCount() const { return _mismatchCount; };
    T *C_d() { return _C_d; };

    // How much does this streamer add to A[i] after running runAsyncAfter
    int expectedAdd() const { return (_commandType == COMMAND_COPY) ? 0 : p_count; };


    int         _commandType;  // 0=addReverse, 1=addFwd, 2=move
private:

    T *_C_h;

    T *_preA_d; // if input is on another device, this is pointer to that memory.
    T *_A_d;
    T *_C_d;

    hipStream_t _stream;
    hipEvent_t  _event;

    int         _deviceId;
    size_t      _numElements;

    size_t      _mismatchCount;
};


template <typename T>
Streamer<T>::Streamer(int deviceId, T * A_d, size_t numElements, int commandType) :
    _preA_d(NULL), 
    _A_d(A_d),
    _deviceId(deviceId),
    _numElements(numElements),
    _commandType(commandType)
{
    size_t sizeElements = numElements * sizeof(int);

    //if (commandType == 0) _commandType = 1; // TODO - remove me

    HIPCHECK(hipSetDevice(_deviceId));


    hipPointerAttribute_t attr;
    HIPCHECK(hipPointerGetAttributes(&attr, A_d));
    if (attr.device != deviceId) {
        // source is on another device, we will need to copy later.
        // So save original source pointer and allocate local space.
        printf ("info: source for streamer on another device, will insert memcpy\n");
        _preA_d = A_d;
        HIPCHECK(hipMalloc(&_A_d, sizeElements));
        HIPCHECK(hipMemset(_A_d, -3, sizeElements));
    }

    HIPCHECK(hipMalloc(&_C_d, sizeElements));
    HIPCHECK(hipHostMalloc(&_C_h, sizeElements));

    HIPCHECK(hipMemset(_C_d, -1, sizeElements));
    HIPCHECK(hipMemset(_C_h, -2, sizeElements));

    HIPCHECK(hipStreamCreate(&_stream));
    HIPCHECK(hipEventCreate(&_event));



};


template <typename T>
Streamer<T>::~Streamer()
{
    HIPCHECK(hipSetDevice(_deviceId));

    printf ("info: ~Streamer\n");
    if (_preA_d) {
        HIPCHECK(hipFree(_preA_d));
    }
    HIPCHECK(hipFree(_C_d));
    HIPCHECK(hipHostFree(_C_h));

    HIPCHECK(hipStreamDestroy(_stream));
    HIPCHECK(hipEventDestroy(_event));
}


template <typename T>
void Streamer<T>::runAsyncAfter(Streamer<T> *depStreamer, bool waitSameStream)
{
    HIPCHECK(hipSetDevice(_deviceId));
    if (p_db) {
      printf ("testing: %s  numElements=%zu size=%6.2fMB\n", __func__, _numElements, _numElements * sizeof(T) / 1024.0/1024.0);
    }

    if (depStreamer) {
        HIPCHECK(hipStreamWaitEvent(_stream, depStreamer->event(), 0));
    }

    if (_preA_d) {
        // _preA_d is on another device, so copy to local device so kernel can access it:
        HIPCHECK(hipMemcpyAsync(_A_d, _preA_d, _numElements * sizeof(T), hipMemcpyDeviceToDevice, _stream));
    }


    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, _numElements);
    if (_commandType == COMMAND_ADD_REVERSE) {
        hipLaunchKernelGGL(HipTest::addCountReverse , dim3(blocks), dim3(threadsPerBlock), 0, _stream,    _A_d, _C_d, _numElements, p_count);
    } else if (_commandType == COMMAND_ADD_FORWARD) {
        hipLaunchKernelGGL(HipTest::addCount,         dim3(blocks), dim3(threadsPerBlock), 0, _stream,    _A_d, _C_d, _numElements, p_count);
    } else if (_commandType == COMMAND_COPY) {
        HIPCHECK(hipMemcpyAsync(_C_d, _A_d, _numElements * sizeof(T), hipMemcpyDeviceToDevice, _stream));
    } else {
        assert(0); // bad command type
    }
    HIPCHECK(hipEventRecord(_event, _stream));

    if (waitSameStream) {
        HIPCHECK(hipStreamWaitEvent(_stream, _event, 0)); // this is essentially a no-op, but make sure it doesn't crash
    }
}



template <typename T>
void Streamer<T>::queryUntilComplete()
{
    HIPCHECK(hipSetDevice(_deviceId));
    int numQueries = 0;
    hipError_t e = hipSuccess;
    do {
        numQueries++;
        e = hipStreamQuery(_stream);
    } while (e != hipSuccess) ;

    printf ("info: hipStreamQuery completed after %d queries\n", numQueries);
};


// If copyStream is !nullptr it is used for the copy.
template <typename T>
void Streamer<T>::copyToHost(hipStream_t copyStream)
{
    if (p_db) {
        printf ("db: copy back to host\n");
    }
    HIPCHECK(hipSetDevice(_deviceId));
    HIPCHECK(hipMemcpyAsync(_C_h, _C_d, _numElements*sizeof(T), hipMemcpyDeviceToHost, copyStream ? copyStream : _stream));
    HIPCHECK(hipStreamSynchronize(copyStream ? copyStream:_stream));

}


template <typename T>
size_t Streamer<T>::check(int streamerNum, T initValue, T expectedOffset, bool expectPass)
{
    T expected = initValue + expectedOffset;
    if (p_db) {
        printf ("db: check\n");
    }

    _mismatchCount = 0;
    for (size_t i=0; i<_numElements; i++) {
        if (_C_h[i] != expected) {
            _mismatchCount++;
            if (expectPass) {
                fprintf(stderr, "for streamer:%d  _C_h[%zu] (%d)  !=  expected(%d)\n", streamerNum, i, _C_h[i], expected);
                if (_mismatchCount > 10) {
                    failed("for streamer:%d  _C_h[%zu] (%d)  !=  expected(%d)\n", streamerNum, i, _C_h[i], expected);
                }
            }
        }
    }

    if (!expectPass && (_mismatchCount ==0)) {
        // the test should run kernels long enough that if we don't correctly wait for them to finish then an error is reported.
        //failed("for streamer:%d  we expected inavalid synchronization to lead to mismatch but none was detected.  Increase --N to sensitize sync.\n", streamerNum);

    }

    return _mismatchCount;
}

   

//---
//Parse arguments specific to this test.
void parseMyArguments(int argc, char *argv[])
{
    N = 64*1024*1024;

    int more_argc = HipTest::parseStandardArguments(argc, argv, false);

    // parse args for this test:
    for (int i = 1; i < more_argc; i++) {
        const char *arg = argv[i];

        if (!strcmp(arg, "--streams")) {
            if (++i >= argc || !HipTest::parseUInt(argv[i], &p_streams)) {
               failed("Bad streams argument");
            }
        } else if (!strcmp(arg, "--count")) {
            if (++i >= argc || !HipTest::parseUInt(argv[i], &p_count)) {
               failed("Bad count argument");
            }
        } else if (!strcmp(arg, "--db")) {
            if (++i >= argc || !HipTest::parseUInt(argv[i], &p_db)) {
               failed("Bad db argument");
            }
        } else {
            failed("Bad argument '%s'", arg);
        }
    };
};


typedef Streamer<int> IntStreamer;




void runStreamerLoop(std::vector<IntStreamer *> &streamers)
{
    for (int i=0; i<streamers.size(); i++) {
        streamers[i]->runAsyncAfter(i ? streamers[i-1] : NULL);
    }
}


void checkAll(int initValue, std::vector<IntStreamer *> &streamers, std::vector<hipStream_t> &sideStreams, bool expectPass=true)
{
    size_t mismatchCount=0;

    // Copy in reverse order to catch anything not yet finished...
    for (int i=streamers.size()-1; i>=0; i--) {
        streamers[i]->copyToHost(sideStreams.empty() ? NULL : sideStreams[streamers[i]->deviceId()]);
    }


    int expected = 0;
    // Check in forward order so we can find first mismatch:
    for (int i=0; i<streamers.size(); i++) {

        expected += streamers[i]->expectedAdd();
        
        mismatchCount += streamers[i]->check(i+1, initValue, expected, expectPass);

    }
    if (!expectPass && (mismatchCount==0)) {
        // the test should run kernels long enough that if we don't correctly wait for them to finish then an error is reported.
        failed("we expected inavalid synchronization to lead to mismatch but none was detected.  Increase --count to sensitize sync.\n");
    }

}



#define RUN_SYNC_TEST(_enableBit, _streamers, _sync, _expectPass)\
    if (p_tests & (_enableBit)) {\
        printf ("==> Test %02x runAsyncAfter sync=%s\n", (_enableBit), #_sync);\
        runStreamerLoop(_streamers);\
        (_sync);\
        checkAll (initValue, _streamers, sideStreams, _expectPass);\
    }




//---
// A family of sync functions which somehow wait for inflight activity to finish:


void sync_none(void) {};

void sync_allDevices(int numDevices) 
{
    for (int d=0; d<numDevices; d++) {
        HIPCHECK(hipSetDevice(d));
        HIPCHECK(hipDeviceSynchronize());
    }
}


void sync_queryAllUntilComplete(std::vector<IntStreamer *> streamers) 
{
    for (int i=streamers.size()-1; i>=0; i--) {
        streamers[i]->queryUntilComplete();
    };
}


void sync_streamWaitEvent(hipEvent_t lastEvent, int sideDeviceId, hipStream_t sideStream, bool waitHere) 
{
    HIPCHECK(hipSetDevice(sideDeviceId));

    // wait on the last event in the stream of chained streamers:
    // This plants a marker which the subsquent copy for this device will wait on:
    HIPCHECK(hipStreamWaitEvent(sideStream, lastEvent, 0));

    if (waitHere) {
        HIPCHECK(hipStreamSynchronize(sideStream));
    }
}



//---
int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, false);
    parseMyArguments(argc, argv);




    size_t numElements = N;
    size_t sizeElements = numElements * sizeof(int);

    printf("info: sizeof arrays = %zu elements (%6.3f MB)\n", numElements, sizeElements / 1024.0/1024.0);
    printf("info: streams=%d count=%d\n", p_streams, p_count);

    assert (sizeElements <= std::numeric_limits<int64_t>::max());


    int initValue = 1000;

    int * initArray_d, *initArray_h;
    HIPCHECK(hipMalloc(&initArray_d, sizeElements));
    HIPCHECK(hipHostMalloc(&initArray_h, sizeElements));
    for (size_t i=0; i<numElements; i++) {
        initArray_h[i] = initValue;
    }
    HIPCHECK(hipMemcpy(initArray_d, initArray_h, sizeElements, hipMemcpyHostToDevice));
    

    int numDevices;
    HIPCHECK(hipGetDeviceCount(&numDevices));
    numDevices = min(2, numDevices); // multi-GPU to 2 device.

    std::vector<IntStreamer *> streamers;
    std::vector<IntStreamer *> streamersDev0; // streamers for first device.

    for (int d=0; d<numDevices/*TODO*/; d++) {
        for (int i=0; i<p_streams; i++) {
            int command = (i%2) ? COMMAND_ADD_FORWARD : COMMAND_ADD_REVERSE;
            IntStreamer * s = new IntStreamer(d, i ? streamers.back()->C_d() : initArray_d, numElements, command);
            streamers.push_back(s);
            if (d==0) {
                streamersDev0.push_back(s);
            }
        }
    }





    // A sideband stream channel that is independent from above.
    // Used to check to ensure the WaitEvent or other synchronization is working correctly since by default sideStream is 
    // asynchronous wrt the other streams.
    std::vector<hipStream_t> sideStreams;
    for (int d=0; d<numDevices; d++) {
        hipStream_t s;
        HIPCHECK(hipStreamCreate(&s));
        sideStreams.push_back(s);
    }


    // Tests on first GPU:
    //
    // This test has no synchronization - make sure it mismatches so we can ensure the other tests properyl prevent the mismatch:
    RUN_SYNC_TEST(0x01, streamersDev0, sync_none(), false);

    RUN_SYNC_TEST(0x02, streamersDev0, sync_allDevices(numDevices),  true);
    RUN_SYNC_TEST(0x04, streamersDev0, sync_queryAllUntilComplete(streamersDev0),  true);
    RUN_SYNC_TEST(0x08, streamersDev0, sync_streamWaitEvent(streamersDev0.back()->event(), 0, sideStreams[0], false),  true);

    if (numDevices > 1) {
        // Sync on second device for activity running on device 0:
        RUN_SYNC_TEST(0x10, streamersDev0, sync_streamWaitEvent(streamersDev0.back()->event(), 1, sideStreams[1], true),  true);
    }


    // Tests on all GPUs:
    // RUN_SYNC_TEST(0x100, streamers, sync_streamWaitEvent(streamers.back()->event(), 0, sideStreams[0], false),  true);




    if (p_tests & 0x1000) {
        printf ("==> Test 0x1000 simple null stream tests\n"); 

        // try some null stream:
        hipStreamQuery(0);


        hipStream_t s1;
        hipEvent_t e1;

        {
            // stream null waits on event in s1 stream:
            HIPCHECK(hipStreamCreate(&s1));
            HIPCHECK(hipEventCreate(&e1));

            HIPCHECK(hipEventRecord(e1, s1))

            HIPCHECK(hipStreamWaitEvent(hipStream_t(0), e1, 0/*flags*/));
            
            HIPCHECK(hipStreamDestroy(s1));
            HIPCHECK(hipEventDestroy(e1));
        }

        {
            // stream s1 waits on event in null stream:
            HIPCHECK(hipStreamCreate(&s1));
            HIPCHECK(hipEventCreate(&e1));

            HIPCHECK(hipEventRecord(e1, hipStream_t(0)))

            HIPCHECK(hipStreamWaitEvent(s1, e1, 0/*flags*/));
            
            HIPCHECK(hipStreamDestroy(s1));
            HIPCHECK(hipEventDestroy(e1));
        }
        
    }


    // Insert small wrinkle here, insert a wait on event just recorded, all in the same stream.
    if (p_tests & 0x2000) {
        printf ("==> Test 0x2000 runAsyncWaitSameStream\n");
        for (int i=0; i<streamersDev0.size(); i++) {
            streamersDev0[i]->runAsyncAfter(i ? streamersDev0[i-1] : NULL, true/*waitSameStream*/);
        }

        sync_streamWaitEvent(streamersDev0.back()->event(), 0, sideStreams[0], false);
        checkAll (initValue, streamersDev0, sideStreams);
    }


    // Change Adds to copies to stimulate different case with event followign copy:
    for (auto &s : streamers) {
        if (s->_commandType == COMMAND_ADD_FORWARD)
            s->_commandType = COMMAND_COPY;
    }


    if (p_tests & 0x4000 ) {
        printf ("test: %x alternating memcpy/count-reverse followed by event\n", p_tests);
        RUN_SYNC_TEST(0x4000, streamersDev0, sync_queryAllUntilComplete(streamersDev0),  true);
        RUN_SYNC_TEST(0x8000, streamersDev0, sync_streamWaitEvent(streamersDev0.back()->event(), 0, sideStreams[0], false),  true);
    }


    passed();
}
