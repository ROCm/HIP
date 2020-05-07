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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11 EXCLUDE_HIP_PLATFORM rocclr
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>

#include "test_common.h"

#define LEN 64
#define SIZE LEN << 2
#define THREADS 2
#define MAX_THREADS 16

#define FILENAME "vcpy_kernel.code"
#define kernel_name "hello_world"

std::vector<char> load_file()
{
    std::ifstream file(FILENAME, std::ios::binary | std::ios::ate);
    std::streamsize fsize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fsize);
    if (!file.read(buffer.data(), fsize)) {
        failed("could not open code object '%s'\n", FILENAME);
    }
    return buffer;
}

void run(const std::vector<char>& buffer) {
    hipDevice_t device;
    HIPCHECK(hipDeviceGet(&device, 0));
    hipCtx_t context;
    HIPCHECK(hipCtxCreate(&context, 0, device));

    hipModule_t Module;
    hipFunction_t Function;
    HIPCHECK(hipModuleLoadData(&Module, &buffer[0]));
    HIPCHECK(hipModuleGetFunction(&Function, Module, kernel_name));
    
    float *A, *B, *Ad, *Bd;
    A = new float[LEN];
    B = new float[LEN];

    for (uint32_t i = 0; i < LEN; i++) {
        A[i] = i * 1.0f;
        B[i] = 0.0f;
    }

    HIPCHECK(hipMalloc((void**)&Ad, SIZE));
    HIPCHECK(hipMalloc((void**)&Bd, SIZE));

    HIPCHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));

    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));

    struct {
        void* _Ad;
        void* _Bd;
    } args;
    args._Ad = (void*) Ad;
    args._Bd = (void*) Bd;
    size_t size = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
    HIPCHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream, NULL, (void**)&config));

    HIPCHECK(hipStreamDestroy(stream));

    HIPCHECK(hipModuleUnload(Module));

    HIPCHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));

    for (uint32_t i = 0; i < LEN; i++) {
        assert(A[i] == B[i]);
    }
   
    hipFree(Ad); 
    hipFree(Bd); 
    delete[] A;
    delete[] B;
    hipCtxDestroy(context);
    
}

struct joinable_thread : std::thread
{
    template <class... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...) // NOLINT
    {
    }

    joinable_thread& operator=(joinable_thread&& other) = default;
    joinable_thread(joinable_thread&& other)            = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

void run_multi_threads(uint32_t n, const std::vector<char>& buffer) {
    
    std::vector<joinable_thread> threads;
    
    for (uint32_t i = 0; i < n; i++) {
        threads.emplace_back(std::thread{[&, buffer] {
        run(buffer);
        }});
    }
    
}

int main() {

    HIPCHECK(hipInit(0));
    auto buffer = load_file();
    run_multi_threads(min(THREADS * std::thread::hardware_concurrency(), MAX_THREADS), buffer);

    passed();
}
