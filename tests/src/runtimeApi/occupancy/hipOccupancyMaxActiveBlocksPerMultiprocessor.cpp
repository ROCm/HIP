/*
Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

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
// Test the Grid_Launch syntax.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM rocclr
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

__global__ void f1(float *a) { *a = 1.0; }

template <typename T>
__global__ void f2(T *a) { *a = 1; }



int main(int argc, char* argv[]) {

    // test case for using kernel function pointer
    int gridSize = 0;
    int blockSize = 0;
    hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, 0);
    assert(gridSize != 0 && blockSize != 0);

    int numBlock = 0;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, f1, blockSize, 0);
    assert(numBlock != 0);


    // test case for using kernel function pointer with template
    gridSize = 0;
    blockSize = 0;
    hipOccupancyMaxPotentialBlockSize<void(*)(int *)>(&gridSize, &blockSize, f2, 0, 0);
    assert(gridSize != 0 && blockSize != 0);

    numBlock = 0;
    hipOccupancyMaxActiveBlocksPerMultiprocessor<void(*)(int *)>(&numBlock, f2, blockSize, 0);
    assert(numBlock != 0);

    passed();
}
