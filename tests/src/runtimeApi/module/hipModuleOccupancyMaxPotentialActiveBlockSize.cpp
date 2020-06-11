/*
Copyright (c) 2019 - prsent Advanced Micro Devices, Inc. All rights reserved.
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
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

#define fileName "vcpy_kernel.code"
#define kernel_name "hello_world"

int main(int argc, char* argv[]) {

    int gridSize = 0;
    int blockSize = 0;
    int numBlock = 0;
    HIPCHECK(hipInit(0));

    hipDevice_t device;
    hipCtx_t context;
    HIPCHECK(hipDeviceGet(&device, 0));
    HIPCHECK(hipCtxCreate(&context, 0, device));

    hipModule_t Module;
    hipFunction_t Function;
    HIPCHECK(hipModuleLoad(&Module, fileName));
    HIPCHECK(hipModuleGetFunction(&Function, Module, kernel_name));
    HIPCHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, Function, 0, 0));
    assert(gridSize != 0 && blockSize != 0);
    HIPCHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, Function, blockSize, 0));
    assert(numBlock != 0);
    HIPCHECK(hipModuleUnload(Module));
    HIPCHECK(hipCtxDestroy(context));
    passed();
}
