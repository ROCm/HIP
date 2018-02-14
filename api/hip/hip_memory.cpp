/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_runtime.h>

#include "hip_internal.hpp"

hipError_t hipMalloc(void** ptr, size_t sizeBytes)
{
    HIP_INIT_API(ptr, sizeBytes);

    amd::Context* context = as_amd(g_currentCtx);

    if (sizeBytes == 0) {
        *ptr = nullptr;
        return hipSuccess;
    }
    else if (!is_valid(context) || !ptr) {
        return hipErrorInvalidValue;
    }

    auto deviceHandle = as_amd(g_deviceArray[0]);
    if ((deviceHandle->info().maxMemAllocSize_ < size)) {
        return hipErrorOutOfMemory;
    }

    amd::Memory* mem = new (*context) amd::Buffer(*context, 0, sizeBytes);
    if (!mem) {
        return hipErrorOutOfMemory;
    }

    if (!mem->create(nullptr)) {
        return hipErrorMemoryAllocation;
    }

    *ptr = reinterpret_cast<void*>(as_cl(mem));

    return hipSuccess;
}

hipError_t hipFree(void* ptr)
{
    if (!is_valid(reinterpret_cast<cl_mem>(ptr))) {
        return hipErrorInvalidValue;
    }
    as_amd(reinterpret_cast<cl_mem>(ptr))->release();
    return hipSuccess;
}


