/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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

//#include <hc_am.hpp>

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"

hipError_t hipLaunchKernel(
    const void* function_address,
    dim3 numBlocks,
    dim3 dimBlocks,
    void** args,
    size_t sharedMemBytes,
    hipStream_t stream,
    size_t szKernArg)
{
   hip_impl::hip_init();

   const hipFunction_t kd = hip_impl::get_program_state().kernel_descriptor((std::uintptr_t)function_address,
				                           hip_impl::target_agent(stream));
   void* config[]{
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &szKernArg,
        HIP_LAUNCH_PARAM_END};

	if(args == NULL)
	{
	   config[1] = NULL;
	}

   return hipModuleLaunchKernel(kd, numBlocks.x, numBlocks.y, numBlocks.z,
                          dimBlocks.x, dimBlocks.y, dimBlocks.z, sharedMemBytes,
                          stream,nullptr,(void**)&config);
}

