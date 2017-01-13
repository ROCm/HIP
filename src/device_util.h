/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef DEVICE_UTIL_H
#define DEVICE_UTIL_H

/*
 Heap size computation for malloc and free device functions.
*/

#define NUM_PAGES_PER_THREAD  16
#define SIZE_OF_PAGE          64
#define NUM_THREADS_PER_CU    64
#define NUM_CUS_PER_GPU       64
#define NUM_PAGES NUM_PAGES_PER_THREAD * NUM_THREADS_PER_CU * NUM_CUS_PER_GPU
#define SIZE_MALLOC NUM_PAGES * SIZE_OF_PAGE
#define SIZE_OF_HEAP SIZE_MALLOC

#endif
