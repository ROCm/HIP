/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#define INTERNAL_BUFFER_SIZE 8
// Test Type
#define TEST_MALLOC_FREE 1
#define TEST_NEW_DELETE 2
// Kernel Params
#define BLOCKSIZE 64
#define GRIDSIZE 32
// Code Obj
#define DEV_ALLOC_SINGKER_COBJ "kerDevAllocSingleKer.code"
#define DEV_ALLOC_SINGKER_COBJ_FUNC "ker_TestDynamicAllocInAllThreads_CodeObj"
#define DEV_ALLOC_MULCOBJ "kerDevAllocMultCO.code"
#define DEV_WRITE_MULCOBJ "kerDevWriteMultCO.code"
#define DEV_FREE_MULCOBJ "kerDevFreeMultCO.code"
#define DEV_ALLOC_MULCODEOBJ_ALLOC "ker_Alloc_MultCodeObj"
#define DEV_ALLOC_MULCODEOBJ_WRITE "ker_Write_MultCodeObj"
#define DEV_ALLOC_MULCODEOBJ_FREE "ker_Free_MultCodeObj"
