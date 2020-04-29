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

#include "test_common.h"
#include <iostream>
#include <time.h>

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#define NUM_SIZE 19  //size up to 16M
#define NUM_ITER 500 //Total GPU memory up to 16M*500=8G

void valSet(int* A, int val, size_t size) {
    size_t len = size / sizeof(int);
    for (int i = 0; i < len; i++) {
        A[i] = val;
    }
}

void setup(size_t *size, const int num, int **pA) {
    std::cout << "size: ";
    for (int i = 0; i < num; i++) {
        size[i] = 1 << (i + 6);
        std::cout << size[i] << " ";
    }
    std::cout << std::endl;
    *pA = (int*)malloc(size[num - 1]);
    valSet(*pA, 1, size[num - 1]);
}

void testInit(size_t size, int *A) {
    int *Ad;
    clock_t start = clock();
    hipMalloc(&Ad, size); //hip::init() will be called
    clock_t end = clock();
    double uS = (end - start) * 1000000. / CLOCKS_PER_SEC;
    std::cout << "Initial" << std::endl;
    std::cout << "hipMalloc(" << size << ") cost " << uS << "us" << std::endl;

    start = clock();
    hipMemcpy(Ad, A, size, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    end = clock();
    uS = (end - start) * 1000000. / CLOCKS_PER_SEC;
    std::cout << "hipMemcpy(" << size << ") cost " << uS << "us" << std::endl;

    start = clock();
    hipFree(Ad);
    end = clock();
    uS = (end - start) * 1000000. / CLOCKS_PER_SEC;
    std::cout << "hipFree(" << size << ") cost " << uS << "us" << std::endl;
}

int main() {
    double uS;
    clock_t start, end;
    size_t size[NUM_SIZE] = { 0 };
    int *Ad[NUM_ITER] = { nullptr };
    int *A;

    setup(size, NUM_SIZE, &A);
    testInit(size[0], A);

    for (int i = 0; i < NUM_SIZE; i++) {
        std::cout << size[i] << std::endl;
        start = clock();
        for (int j = 0; j < NUM_ITER; j++) {
            HIPCHECK(hipMalloc(&Ad[j], size[i]));
        }
        end = clock();
        uS = (end - start) * 1000000. / (NUM_ITER * CLOCKS_PER_SEC);
        std::cout << "hipMalloc(" << size[i] << ") cost " << uS << "us" << std::endl;

        start = clock();
        for (int j = 0; j < NUM_ITER; j++) {
            HIPCHECK(hipMemcpy(Ad[j], A, size[i], hipMemcpyHostToDevice));
        }
        hipDeviceSynchronize();
        end = clock();
        uS = (end - start) * 1000000. / (NUM_ITER * CLOCKS_PER_SEC);
        std::cout << "hipMemcpy(" << size[i] << ") cost " << uS << "us" << std::endl;

        start = clock();
        for (int j = 0; j < NUM_ITER; j++) {
            HIPCHECK(hipFree(Ad[j]));
            Ad[j] = nullptr;
        }
        end = clock();
        double uS = (end - start) * 1000000. / (NUM_ITER * CLOCKS_PER_SEC);
        std::cout << "hipFree(" << size[i] << ") cost " << uS << "us" << std::endl;
    }
    free(A);
    passed();
}
