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


#include"gxxHipApi.h"
#include<vector>
#include "hip/hip_runtime.h"

#define LEN 1024*1024
#define SIZE LEN * sizeof(float)

class memManager;

int main()
{
    std::vector<class memManager> Vec(4);
    for(int i=0;i<Vec.size();i++){
        Vec[i] = memManager(SIZE);
    }

    for(int i=0;i<4;i++)
    {
        Vec[i].setHstPtr(new float[LEN]);
    }

    for(int i=0;i<2;i++)
    {
        Vec[i].memAlloc<float>();
    }

    for(int i=0;i<2;i++)
    {
        Vec[i].hostMemSet((i+1)*1.0f);
        Vec[i].H2D();
    }

    Vec[2].setDevPtr(Vec[0].getDevPtr<float>());
    Vec[3].setDevPtr(Vec[1].getDevPtr<float>());

    for(int i=2;i<Vec.size();i++)
    {
        Vec[i].D2H();
    }
    
    assert(Vec[0].getHstPtr<float>()[10] == Vec[2].getHstPtr<float>()[10]);
    assert(Vec[1].getHstPtr<float>()[10] == Vec[3].getHstPtr<float>()[10]);
}
