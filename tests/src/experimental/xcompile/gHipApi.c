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


#include"gHipApi.h"
#include "hip/hip_runtime_api.h"
#include"stdio.h"

void _h2d(mem_manager *self){
    hipMemcpy(self->dev_ptr, self->hst_ptr, self->size, hipMemcpyHostToDevice);
}

void _d2h(mem_manager *self)
{
    hipMemcpy(self->hst_ptr, self->dev_ptr, self->size, hipMemcpyDeviceToHost);
}

void _malloc_hip(mem_manager *self)
{
      hipMalloc(&(self->dev_ptr), self->size);
}

void _malloc_hst(mem_manager *self)
{
     self->hst_ptr = malloc(self->size);
}

void memset_hst(mem_manager *mem, float val)
{
      float *tmp = (float*)mem->hst_ptr;
      int i;
      for(i=0;i<(mem->size)/sizeof(float);i++)
      {
          tmp[i] = val;
      }
}

mem_manager *mem_manager_start(size_t _size)
{
    mem_manager *tmp = (mem_manager*)malloc(sizeof(mem_manager));
    tmp->size = _size;
    tmp->h2d = _h2d;
    tmp->d2h = _d2h;
    tmp->malloc_hip = _malloc_hip;
    tmp->malloc_hst = _malloc_hst;
    return tmp;
}


