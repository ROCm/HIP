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


#ifndef GXXHIPAPI_H
#define GXXHIPAPI_H

#include <stdlib.h>
#include "hip/hip_runtime_api.h"

class memManager {
   private:
    void* devPtr;
    void* hstPtr;
    size_t size;

   public:
    memManager(size_t size) : size(size) {}
    memManager() {}
    memManager(const memManager& obj);
    template <typename T>
    void setDevPtr(T* ptr) {
        devPtr = (void*)ptr;
    }

    template <typename T>
    T* getDevPtr() {
        return (T*)devPtr;
    }

    template <typename T>
    void setHstPtr(T* ptr) {
        hstPtr = (void*)ptr;
    }

    template <typename T>
    T* getHstPtr() {
        return (T*)hstPtr;
    }

    void H2D();
    void D2H();
    template <typename T>
    void hostMemSet(T val) {
        T* tmpPtr = (T*)hstPtr;
        for (int i = 0; i < size / sizeof(T); i++) {
            tmpPtr[i] = val;
        }
    }
    template <typename T>
    void memAlloc() {
        hipMalloc((void**)&devPtr, size);
    }
};

#endif
