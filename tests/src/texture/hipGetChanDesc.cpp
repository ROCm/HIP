/*
Copyright (c) 2015-Present Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include <hip/hip_runtime.h>
#include "test_common.h"

using namespace std;
#define R 8 //rows, height
#define C 8 //columns, width

bool runTest(void);

int main(int argc, char** argv) {
    bool testResult=runTest();

    if (testResult) {
        passed();
    } else {
        exit(EXIT_FAILURE);
    }
}

bool runTest()
{
bool testResult=true;
hipChannelFormatDesc chan_test,chan_desc=hipCreateChannelDesc(32,0,0,0,hipChannelFormatKindSigned);
hipArray *hipArray;
HIPCHECK(hipMallocArray(&hipArray, &chan_desc,C,R,0));
HIPCHECK(hipGetChannelDesc(&chan_test,hipArray));

if((chan_test.x == 32)&&(chan_test.y == 0)&&(chan_test.z == 0)&&(chan_test.f == 0))
	testResult=true;
else
	testResult=false;

HIPCHECK(hipFreeArray(hipArray));
return testResult;
}
