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

// Test under-development. Call hipStreamAddCallback function and see if it works as expected.

#include "hip/hip_runtime.h"
#include "test_common.h"
#define HIPRT_CB

class CallbackClass
{
public:
    static void HIPRT_CB Callback(hipStream_t stream, hipError_t status, void *userData);

private:
    void callbackFunc(hipError_t status);
};

void HIPRT_CB CallbackClass::Callback(hipStream_t stream, hipError_t status, void *userData)
{
    CallbackClass* obj = (CallbackClass*) userData;
    obj->callbackFunc(status);
}

void CallbackClass::callbackFunc(hipError_t status)
{
     HIPASSERT(status==hipSuccess);
}

int main(){
    hipStream_t mystream;
    HIPCHECK(hipStreamCreate(&mystream));
    CallbackClass* obj = new CallbackClass;
    HIPCHECK(hipStreamAddCallback(mystream, CallbackClass::Callback, obj, 0));
    HIPCHECK(hipStreamAddCallback(NULL, CallbackClass::Callback, obj, 0));

	passed();
}
