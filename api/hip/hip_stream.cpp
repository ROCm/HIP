/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_runtime.h>

#include "hip_internal.hpp"

hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags)
{
  HIP_INIT_API(stream, flags);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}


hipError_t hipStreamCreate(hipStream_t *stream)
{
  HIP_INIT_API(stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}


hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags)
{
  HIP_INIT_API(stream, flags);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}


hipError_t hipStreamSynchronize(hipStream_t stream)
{
  HIP_INIT_API(stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}


hipError_t hipStreamDestroy(hipStream_t stream)
{
  HIP_INIT_API(stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}


