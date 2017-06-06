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
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

int p_iters=10;

void printSep()
{
    printf ("======================================================================================\n");
}

//---
// Test simple H2D copies and back.
// Designed to stress a small number of simple smoke tests

template<
    typename T=float, 
	class P=HipTest::Unpinned, 
	class C=HipTest::Memcpy
>
void simpleVectorAdd(size_t numElements, int iters, hipStream_t stream)
{
	using HipTest::MemTraits;

	std::thread::id pid = std::this_thread::get_id();

    printf ("test: %s <%s> %s %s\n", __func__, TYPENAME(T), P::str(), C::str());
    size_t Nbytes = numElements*sizeof(T);
    printf ("numElements=%zu Nbytes=%6.2fMB\n", numElements, Nbytes/1024.0/1024.0);

    T *A_d, *B_d, *C_d;
    T *A_h, *B_h, *C_h;

    HipTest::initArrays (&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, P::isPinned);
	for (size_t i=0; i<numElements; i++) {
		A_h[i] = 1000.0f;
		B_h[i] = 2000.0f;
		C_h[i] = -1;
	}


	MemTraits<C>::Copy(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream);
	MemTraits<C>::Copy(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream);
	MemTraits<C>::Copy(C_d, C_h, Nbytes, hipMemcpyHostToDevice, stream);
    HIPCHECK (hipDeviceSynchronize());

	for (size_t i=0; i<numElements; i++) {
		A_h[i] = 1.0f;
		B_h[i] = 2.0f;
		C_h[i] = -1;
	}



	for (int i=0; i<iters; i++) {

		unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

		MemTraits<C>::Copy(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream);
		MemTraits<C>::Copy(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream);

		//HIPCHECK(hipStreamSynchronize(stream));

		// This is the null stream?
		//hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, B_d, C_d, numElements);
		hipLaunchKernel(HipTest::vectorADDReverse, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, B_d, C_d, numElements);

		MemTraits<C>::Copy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream);

		HIPCHECK (hipDeviceSynchronize());

		HipTest::checkVectorADD(A_h, B_h, C_h, numElements);
	}

    HipTest::freeArrays (A_d, B_d, C_d, A_h, B_h, C_h, P::isPinned);
	std::cout <<"  pid" << pid << " success\n";
    HIPCHECK (hipDeviceSynchronize());

}

template<typename T, class C>
void test_multiThread_1(std::string testName, hipStream_t stream0, hipStream_t stream1, bool serialize)
{
	printSep();
	printf ("%s\n", __func__);
	std::cout << testName << std::endl;

	size_t numElements = N;

	// Test 2 threads operating on same stream:
    std::thread t1 (simpleVectorAdd<T, HipTest::Pinned, C>, numElements, p_iters/*iters*/, stream0);
    if (serialize) {
        t1.join();
    }
    std::thread t2 (simpleVectorAdd<T, HipTest::Pinned, C>, numElements, p_iters/*iters*/, stream1);
    if (serialize) {
        t2.join();
    }

    if (!serialize) {
        t1.join();
        t2.join();
    }

	HIPCHECK(hipDeviceSynchronize());
};


int main(int argc, char *argv[])
{
	N = 8000000;
    HipTest::parseStandardArguments(argc, argv, true);

    printf ("info: set device to %d\n", p_gpuDevice);
    HIPCHECK(hipSetDevice(p_gpuDevice));


    if (p_tests & 0x1) {
        HIPCHECK ( hipDeviceReset() );

        hipStream_t stream;
        HIPCHECK (hipStreamCreate(&stream));

        simpleVectorAdd<float, HipTest::Pinned, HipTest::MemcpyAsync>	(N/*mb*/, 10/*iters*/, stream);
        simpleVectorAdd<float, HipTest::Pinned, HipTest::Memcpy>		(N/*mb*/, 10/*iters*/, stream);

        HIPCHECK(hipStreamDestroy(stream));
    }


    hipStream_t stream0, stream1;
    HIPCHECK (hipStreamCreate(&stream0));
    HIPCHECK (hipStreamCreate(&stream1));

    if (p_tests & 0x2) {
		// Easy tests to verify the test works - these don't allow overlap between the threads:
		test_multiThread_1<float, HipTest::MemcpyAsync> ("Multithread NULL with serialized", NULL, NULL, true);
		test_multiThread_1<float, HipTest::MemcpyAsync> ("Multithread two streams serialized", stream0, stream1, true);
    }

    if (p_tests & 0x4) {
		//test_multiThread_1<float, HipTest::MemcpyAsync> ("Multithread with NULL stream", NULL, NULL, false);
		//test_multiThread_1<float, HipTest::MemcpyAsync> ("Multithread with two streams", stream0, stream1, false);
		test_multiThread_1<float, HipTest::MemcpyAsync> ("Multithread with one stream",  stream0, stream0, false);
	}

    passed();

}
