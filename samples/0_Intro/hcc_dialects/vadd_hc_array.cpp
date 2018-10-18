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

// Simple test showing how to use HC syntax with array.
// Array provides a type-safe C++ mechanism to allocate accelerator memory.
// Like array_view, hc::array provides multi-dimensional indexing capability,
// and is typed.  However, unlike array_view, hc::array does not provide
// automatic data management capabilities - instead the programmer
// takes the reins and controls when copies are executed.

#if defined(HC_NEXT)
    #include <hc/hc.hpp>
#else
    #include <hc.hpp>
#endif

int main(int argc, char* argv[]) {
    int sizeElements = 1000000;
    size_t sizeBytes = sizeElements * sizeof(float);
    bool pass = true;

    // Allocate host memory
    float* A_h = (float*)malloc(sizeBytes);
    float* B_h = (float*)malloc(sizeBytes);
    float* C_h = (float*)malloc(sizeBytes);

    // Allocate device arrays<>
    // Unlike array_view, these must be explicitly managed by user:
    hc::array<float> A_d(sizeElements);
    hc::array<float> B_d(sizeElements);
    hc::array<float> C_d(sizeElements);

    // Initialize host data
    for (int i = 0; i < sizeElements; i++) {
        A_h[i] = 1.618f * i;
        B_h[i] = 3.142f * i;
    }

    hc::copy(A_h, A_d);  // C++ copy H2D
    hc::copy(B_h, B_d);  // C++ copy H2D

    // Launch kernel onto default accelerator:
    // array<> types are not implicitly copied, so we performed copies above.
    hc::parallel_for_each(hc::extent<1>(sizeElements), [&](hc::index<1> idx)[[hc]] {
        int i = idx[0];
        C_d[i] = A_d[i] + B_d[i];
    });

    // HCC runtime knows that C_d depends on previous PFE and will force the copy to wait for the
    // PFE to complte.
    hc::copy(C_d, C_h);  // C++ copy D2H


    for (int i = 0; i < sizeElements; i++) {
        float ref = 1.618f * i + 3.142f * i;
        if (C_h[i] != ref) {
            printf("error:%d computed=%6.2f, reference=%6.2f\n", i, C_h[i], ref);
            pass = false;
        }
    };
    if (pass) printf("PASSED!\n");
}
