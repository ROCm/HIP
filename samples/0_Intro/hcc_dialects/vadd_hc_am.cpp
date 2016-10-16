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

// Simple test showing how to use HC syntax with AM (accelerator memory).
// AM provides a set of c-style memory management routines for allocating,
// freeing, and copying memory.   am_alloc returns a device pointer
// which can only be used on the device.  The programmer has full control
// over when data is copied.  

#include <hc.hpp>
#include <hc_am.hpp>

int main(int argc, char *argv[])
{
    int sizeElements = 1000000;
    size_t sizeBytes = sizeElements * sizeof(float);
    bool pass = true;

    // Allocate host memory
    float *A_h = (float*)malloc(sizeBytes);
    float *B_h = (float*)malloc(sizeBytes);
    float *C_h = (float*)malloc(sizeBytes);

    // Allocate device pointers:
    // Unlike array_view, these must be explicitly managed by user:
    hc::accelerator acc;  // grab default accelerator where we want to allocate memory:
    hc::accelerator_view av = acc.get_default_view();

    float *A_d, *B_d, *C_d;
    A_d = hc::am_alloc(sizeBytes, acc, 0);
    B_d = hc::am_alloc(sizeBytes, acc, 0);
    C_d = hc::am_alloc(sizeBytes, acc, 0);

    // Initialize host data
    for (int i=0; i<sizeElements; i++) {
        A_h[i] = 1.618f * i; 
        B_h[i] = 3.142f * i;
        C_h[i] = 0;
    }

    av.copy(A_h, A_d, sizeBytes); // C++ copy H2D
    av.copy(B_h, B_d, sizeBytes); // C++ copy H2D

    // Launch kernel onto AV.  
    // Because the kernel PFE and the copies are submitted to same AV, they will execute in order
    // and we don't need additional synchronization to ensure the copies complete before the PFE begins.
    hc::completion_future cf=
    hc::parallel_for_each(av,  hc::extent<1> (sizeElements),
      [=] (hc::index<1> idx) [[hc]] { 
        int i = idx[0];
        C_d[i] = A_d[i] + B_d[i];
    });

   
    // This copy is in same AV as the kernel and thus will wait for the kernel to finish before executing.
    av.copy(C_d, C_h, sizeBytes); // C++ copy D2H


    for (int i=0; i<sizeElements; i++) {
        float ref= 1.618f * i + 3.142f * i;
        if (C_h[i] != ref) {
            printf ("error:%d computed=%6.2f, reference=%6.2f\n", i, C_h[i], ref);
            pass = false;
        }
    };
    if (pass) printf ("PASSED!\n");
}
