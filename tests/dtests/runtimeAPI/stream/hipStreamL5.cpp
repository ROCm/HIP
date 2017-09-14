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

#include "test_common.h"
#include "hipStream.h"

/*
The naming of tests is done by assigning a number to
type of disptach possible on stream.
The following are possible stream dispatches:
1. H2H - hipMemcpyHostToHost :  indexed as 1
2. H2D - hipMemcpyHostToDevice : indexed as 2
3. Ker - Kernel Dispatch : indexed as 3
4. D2D - hipMemcpyDeviceToDevice : indexed as 4
5. D2H - hipMemcpyDeviceToHost : indexed as 5
For example,
a test for Ker, D2D, D2H, H2H, H2D is given as test34512();
Note that all memory copies are Async.

invalid{
*WARNING: The commented out assertions are failing cases.
According to my observation, they are happening with tests
which end in HostToHost and take data from previous
dispatch in the stream. This also include disjoint data passes.
The list of failing tests are:
test23451<float>();
test32451<float>();
test42351<float>();

For disjoint data passed:
test24513
test25134
test34512
}
*/

template<typename T>
void test12345(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = sizeof(T) * N;

	T *Ah, *Bh, *Ch;
	T *Ad, *Bd;
	initArrays(&Ad, &Ah, N, true);
	initArrays(&Bd, &Bh, N, true);
	initArrays(&Ch, N, false, true);

	setArray(Ah, N, T(1));

	H2HAsync(Bh, Ah, size, stream);
	H2DAsync(Ad, Bh, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Ad);
	D2DAsync(Bd, Ad, size, stream);
	D2HAsync(Ch, Bd, size, stream);
	HIPCHECK(hipDeviceSynchronize());

	HIPASSERT(Ah[10] + T(1)== Ch[10]);
	HIPCHECK(hipStreamDestroy(stream));
}

template<typename T>
void test13452(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = sizeof(T) * N;

	T *Ah, *Bh, *Ch;
	T *Dh, *Eh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, false);
	initArrays(&Eh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Ah, N, T(1));
	setArray(Dh, N, T(2));

	H2D(Ad, Dh, size);

	H2HAsync(Bh, Ah, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Ad);
	D2DAsync(Bd, Ad, size, stream);
	D2HAsync(Ch, Bd, size, stream);
	H2DAsync(Cd, Ch, size, stream);
	HIPCHECK(hipDeviceSynchronize());

	D2H(Eh,Cd,size);

	HIPASSERT(Ah[10] == Bh[10]);
	HIPASSERT(Eh[10] == Dh[10] + T(1));

}

template<typename T>
void test14523(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = sizeof(T) * N;

	T *Ah, *Bh, *Ch;
	T *Dh, *Eh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, false);
	initArrays(&Eh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Ah, N, T(1));
	setArray(Dh, N, T(2));

	H2D(Ad,Dh,size);

	H2HAsync(Bh, Ah, size, stream);
	D2DAsync(Bd, Ad, size, stream);
	D2HAsync(Ch, Bd, size, stream);
	H2DAsync(Cd, Ch, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Cd);

	HIPCHECK(hipDeviceSynchronize());

	D2H(Eh, Cd, size);

	HIPASSERT(Ah[10] == Bh[10]);
	HIPASSERT(Ch[10] + T(1) == Eh[10]);
}

template<typename T>
void test15234(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = sizeof(T) * N;

	T *Ah, *Bh, *Ch;
	T *Dh, *Eh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, false);
	initArrays(&Eh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N,  true, false);
	initArrays(&Cd, N, true, false);

	setArray(Ah, N, T(1));
	setArray(Dh, N, T(2));

	H2D(Ad, Dh, size);

	H2HAsync(Bh, Ah, size, stream);
	D2HAsync(Ch, Ad, size, stream);
	H2DAsync(Bd, Ch, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Bd);
	D2DAsync(Cd, Bd, size, stream);

	D2H(Eh, Cd, size);

	HIPASSERT(Ah[10] == Bh[10]);
	HIPASSERT(Eh[10] == Dh[10] + T(1));

}

template<typename T>
void test23451(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));
	const size_t size = sizeof(T) * N;

	T *Ah, *Bh, *Ch;
	T *Ad, *Bd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);

	setArray(Ah, N, T(1));

	H2DAsync(Ad, Ah, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Ad);
	D2DAsync(Bd, Ad, size, stream);
	D2HAsync(Bh, Bd, size, stream);
	H2HAsync(Ch, Bh, size, stream);
	HIPCHECK(hipDeviceSynchronize());
	HIPASSERT(Ah[10] + T(1) == Ch[10]);
}

template<typename T>
void test24513(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = sizeof(T) * N;

	T *Ah, *Bh, *Ch;
	T *Dh, *Eh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, false);
	initArrays(&Eh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Ah, N, T(1));
	setArray(Dh, N, T(2));

	H2D(Cd, Dh, size);

	H2DAsync(Ad, Ah, size, stream);
	D2DAsync(Bd, Ad, size, stream);
	D2HAsync(Bh, Bd, size, stream);
	H2HAsync(Ch, Bh, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Cd);
	HIPCHECK(hipDeviceSynchronize());

	D2H(Eh, Cd, size);

	HIPASSERT(Eh[0] == Dh[0] + T(1));
	HIPASSERT(Ah[0] == Ch[0]);
}

template<typename T>
void test25134(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = sizeof(T) * N;

	T *Ah, *Bh, *Ch;
	T *Dh, *Eh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, false);
	initArrays(&Eh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Ah, N, T(1));
	setArray(Dh, N, T(2));

	H2D(Bd, Dh, size);

	H2DAsync(Ad, Ah, size, stream);
	D2HAsync(Bh, Ad, size, stream);
	H2HAsync(Ch, Bh, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Bd);
	D2DAsync(Cd, Bd, size, stream);

	D2H(Eh, Cd, size);

	HIPCHECK(hipDeviceSynchronize());

	HIPASSERT(Ah[10] == Ch[10]);
	HIPASSERT(Dh[10] + T(1) == Eh[10]);
}

template<typename T>
void test21345(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = N * sizeof(T);

	T *Ah, *Bh, *Ch, *Dh;
	T *Ad, *Bd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, true);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);

	setArray(Ah, N, T(1));
	setArray(Bh, N, T(2));

	H2DAsync(Ad, Ah, size, stream);
	H2HAsync(Ch, Bh, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Ad);
	D2DAsync(Bd, Ad, size, stream);
	D2HAsync(Dh, Bd, size, stream);

	HIPCHECK(hipDeviceSynchronize());

	HIPASSERT( Bh[10] == Ch[10] );
	HIPASSERT( Ah[10] + T(1) == Dh[10]);
}

template<typename T>
void test34512(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = N * sizeof(T);

	T *Bh, *Ch, *Dh;
	T *Ah, *Eh;
	T *Ad, *Bd, *Cd;

	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, true);
	initArrays(&Ah, N, false, false);
	initArrays(&Eh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Ah, N, T(1));

	H2D(Ad, Ah, size);

	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Ad);
	D2DAsync(Bd, Ad, size, stream);
	D2HAsync(Bh, Bd, size, stream);
	H2HAsync(Ch, Bh, size, stream);
	H2DAsync(Cd, Ch, size, stream);

	D2H(Dh, Cd, size);

	HIPCHECK(hipDeviceSynchronize());
	HIPASSERT( Ah[10] + T(1) == Dh[10] );
}

template<typename T>
void test35124(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = N * sizeof(T);

	T *Ah, *Bh;
	T *Ch, *Dh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, false);
	initArrays(&Dh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Dh, N, T(1));

	H2D(Ad, Dh, size);

	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Ad);
	D2HAsync(Ah, Ad, size, stream);
	H2HAsync(Bh, Ah, size, stream);
	H2DAsync(Bd, Bh, size, stream);
	D2DAsync(Cd, Bd, size, stream);

	D2H(Ch, Cd, size);

	HIPCHECK(hipDeviceSynchronize());

	HIPASSERT(Dh[10] + T(1) == Ch[10]);
}

template<typename T>
void test31245(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = N * sizeof(T);
	T *Ah, *Bh, *Ch;
	T *Dh, *Eh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, false);
	initArrays(&Eh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Dh, N, T(1));
	setArray(Ah, N, T(2));

	H2D(Ad, Dh, size);

	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Ad);
	H2HAsync(Bh, Ah, size, stream);
	H2DAsync(Bd, Bh, size, stream);
	D2DAsync(Cd, Bd, size, stream);
	D2HAsync(Ch, Cd, size, stream);

	D2H(Eh, Ad, size);

	HIPCHECK(hipDeviceSynchronize());

	HIPASSERT(Dh[10] + T(1) == Eh[10]);
	HIPASSERT(Bh[10] == Ch[10]);
}


template<typename T>
void test32451(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = N * sizeof(T);

	T *Ah, *Bh, *Ch;
	T *Dh, *Eh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, false);
	initArrays(&Eh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Ah, N, T(1));
	setArray(Eh, N, T(2));

	H2D(Ad, Eh, size);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Ad);
	H2DAsync(Bd, Ah, size, stream);
	D2DAsync(Cd, Bd, size, stream);
	D2HAsync(Bh, Cd, size, stream);
	H2HAsync(Ch, Bh, size, stream);
	HIPCHECK(hipDeviceSynchronize());
	D2H(Dh, Ad, size);

	HIPASSERT(Ah[10] == Ch[10]);
	HIPASSERT(Eh[10] + T(1) == Dh[10]);

}

template<typename T>
void test45123(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));
	const size_t size = N * sizeof(T);

	T *Ah, *Bh;
	T *Ch, *Dh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, false);
	initArrays(&Dh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Dh, N, T(1));

	H2D(Ad, Dh, size);

	D2DAsync(Bd, Ad, size, stream);
	D2HAsync(Ah, Bd, size, stream);
	H2HAsync(Bh, Ah, size, stream);
	H2DAsync(Cd, Bh, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Cd);
	D2H(Ch, Cd, size);
	HIPCHECK(hipDeviceSynchronize());

	HIPASSERT(Dh[10] + T(1) == Ch[10]);
}


template<typename T>
void test41235(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));
	const size_t size = N * sizeof(T);

	T *Ah, *Bh;
	T *Ch;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Ch, N, T(1));

	H2D(Ad, Ch, size);

	D2DAsync(Bd, Ad, size, stream);
	D2HAsync(Ah, Bd, size, stream);
	H2DAsync(Cd, Ah, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Cd);
	D2HAsync(Bh, Cd, size, stream);

	HIPCHECK(hipDeviceSynchronize());

	HIPASSERT(Ch[10] + T(1) == Bh[10]);
}

template<typename T>
void test42351(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = N * sizeof(T);

	T *Ah, *Bh, *Ch;
	T *Dh, *Eh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, false);
	initArrays(&Eh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Dh, N, T(2));
	setArray(Ah, N, T(1));

	H2D(Ad, Dh, size);

	D2DAsync(Bd, Ad, size, stream);
	H2DAsync(Cd, Ah, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Cd);
	D2HAsync(Bh, Cd, size, stream);
	H2HAsync(Ch, Bh, size, stream);

	D2H(Eh, Bd, size);

	HIPCHECK(hipDeviceSynchronize());
	HIPASSERT(Dh[10] == Eh[10]);
	HIPASSERT(Ah[10] + T(1) == Ch[10]);
}

template<typename T>
void test43512(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = N * sizeof(T);

	T *Ah, *Bh;
	T *Ch, *Dh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, false);
	initArrays(&Dh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Dh, N, T(1));

	H2D(Ad, Dh, size);

	D2DAsync(Bd, Ad, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Bd);
	D2HAsync(Ah, Bd, size, stream);
	H2HAsync(Bh, Ah, size, stream);
	H2DAsync(Cd, Bh, size, stream);

	D2H(Ch, Cd, size);
	HIPCHECK(hipDeviceSynchronize());
	HIPASSERT( Dh[10] + T(1) == Ch[10]);
}

template<typename T>
void test51234(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = N * sizeof(T);

	T *Ah, *Bh;
	T *Ch, *Dh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, false);
	initArrays(&Dh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Dh, N, T(1));

	H2D(Ad, Dh, size);

	D2HAsync(Ah, Ad, size, stream);
	H2HAsync(Bh, Ah, size, stream);
	H2DAsync(Bd, Bh, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Bd);
	D2DAsync(Cd, Bd, size, stream);

	D2H(Ch, Cd, size);

	HIPCHECK(hipDeviceSynchronize());

	HIPASSERT(Ch[10] == Dh[10] + T(1));
}

template<typename T>
void test52341(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));
	const size_t size = N * sizeof(T);

	T *Ah, *Bh, *Ch;
	T *Dh, *Eh;
	T *Ad, *Bd, *Cd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, false);
	initArrays(&Eh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);

	setArray(Eh, N, T(1));
	setArray(Bh, N, T(2));

	H2D(Ad, Eh, size);

	D2HAsync(Ah, Ad, size, stream);
	H2DAsync(Bd, Ah, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Bd);
	D2DAsync(Cd, Bd, size, stream);
	H2HAsync(Ch, Bh, size, stream);

	D2H(Dh, Cd, size);

	HIPCHECK(hipDeviceSynchronize());

	HIPASSERT(Eh[10] + T(1) == Dh[10]);
	HIPASSERT(Ch[10] == Bh[10]);
}

template<typename T>
void test53412(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));
	const size_t size = sizeof(T) * N;

	T *Ah, *Bh, *Ch, *Dh;
	T *Eh, *Fh, *Gh;
	T *Ad, *Bd, *Cd, *Dd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, true);
	initArrays(&Eh, N, false, false);
	initArrays(&Fh, N, false, false);
	initArrays(&Gh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);
	initArrays(&Dd, N, true, false);

	setArray(Dh, N, T(1));
	setArray(Eh, N, T(2));
	setArray(Bh, N, T(3));

	H2D(Ad, Dh, size);
	H2D(Bd, Eh, size);

	D2HAsync(Ah, Ad, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Bd);
	D2DAsync(Cd, Bd, size, stream);
	H2HAsync(Ch, Bh, size, stream);
	H2DAsync(Dd, Ch, size, stream);

	D2H(Fh, Cd, size);
	D2H(Gh, Dd, size);

	HIPASSERT(Ah[10] == Dh[10]);
	HIPASSERT(Eh[10] + T(1) == Fh[10]);
	HIPASSERT(Bh[10] == Gh[10]);
}

template<typename T>
void test54123(){
	hipStream_t stream;
	HIPCHECK(hipStreamCreate(&stream));

	const size_t size = N * sizeof(T);

	T *Ah, *Bh, *Ch;
	T *Dh, *Eh, *Fh, *Gh;
	T *Ad, *Bd, *Cd, *Dd;

	initArrays(&Ah, N, false, true);
	initArrays(&Bh, N, false, true);
	initArrays(&Ch, N, false, true);
	initArrays(&Dh, N, false, false);
	initArrays(&Eh, N, false, false);
	initArrays(&Fh, N, false, false);
	initArrays(&Gh, N, false, false);
	initArrays(&Ad, N, true, false);
	initArrays(&Bd, N, true, false);
	initArrays(&Cd, N, true, false);
	initArrays(&Dd, N, true, false);

	setArray(Dh, N, T(1));
	setArray(Eh, N, T(1));
	setArray(Bh, N, T(1));

	H2D(Ad, Dh, size);
	H2D(Bd, Eh, size);

	D2HAsync(Ah, Ad, size, stream);
	D2DAsync(Cd, Bd, size, stream);
	H2HAsync(Ch, Bh, size, stream);
	H2DAsync(Dd, Ch, size, stream);
	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, stream, Dd);

	D2H(Fh, Cd, size);
	D2H(Gh, Dd, size);

	HIPCHECK(hipDeviceSynchronize());
	HIPASSERT(Dh[10] == Ah[10]);
	HIPASSERT(Eh[10] == Fh[10]);
	HIPASSERT(Bh[10] + T(1) == Gh[10]);
}

int main(int argc, char *argv[])
{
	HipTest::parseStandardArguments(argc, argv, true);

	test12345<float>();
	test13452<float>();
	test14523<float>();
	test15234<float>();

	test23451<float>();
	test24513<float>();
	test25134<float>();
	test21345<float>();

	test34512<float>();
	test35124<float>();
	test31245<float>();
	test32451<float>();

	test45123<float>();
	test41235<float>();
	test42351<float>();
	test43512<float>();

	test51234<float>();
	test52341<float>();
	test53412<float>();
	test54123<float>();

	passed();

}

