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

#include <hc.hpp>

int main(int argc, char *argv[])
{
    int size = 1000000;
    bool pass = true;

    // Allocate auto-managed host/device views of data:
    hc::array_view<float> A(size);
    hc::array_view<float> B(size);
    hc::array_view<float> C(size);

    // Initialize host data
    for (int i=0; i<size; i++) {
        A[i] = 1.618f * i; 
        B[i] = 3.142f * i;
    }
    C.discard_data(); // tell runtime not to copy CPU host data.


    // Launch kernel onto default accelerator:
    hc::parallel_for_each(hc::extent<1> (size),
      [=] (hc::index<1> idx) [[hc]] { 
        int i = idx[0];
        C[i] = A[i] + B[i];
    });

    for (int i=0; i<size; i++) {
        float ref= 1.618f * i + 3.142f * i;
        if (C[i] != ref) {
            printf ("error:%d computed=%6.2f, reference=%6.2f\n", i, C[i], ref);
            pass = false;
        }
    };
    if (pass) printf ("PASSED!\n");
}
