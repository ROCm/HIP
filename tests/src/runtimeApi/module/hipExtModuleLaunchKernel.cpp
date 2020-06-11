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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD_CMD: matmul.code %hc --genco %S/matmul.cpp -o matmul.code EXCLUDE_HIP_PLATFORM nvcc
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc rocclr
 * TEST: %t
 * HIT_END
 */

#include<chrono>
#include "hip/hip_runtime.h"
#include "hip/hip_hcc.h"
#include "test_common.h"
using namespace std::chrono;

#define fileName "matmul.code"
#define kernel_name1 "matmulK"
#define kernel_name2 "WaitKernel"

int main()
{
    int N=16384;
    int SIZE=N*N;

    int *A=new int[N*N*sizeof(int)];
    int *B=new int[N*N*sizeof(int)];
    int *C;

    hipDeviceptr_t *Ad,*Bd;

    for(int i=0;i<N;i++)
    for(int j=0;j<N;j++) {
        A[i*N +j]=1;
        B[i*N +j]=1; 
    }

    hipStream_t stream1,stream2;
    HIPCHECK(hipStreamCreate(&stream1));
    HIPCHECK(hipMalloc((void**)&Ad, SIZE*sizeof(int)));
    HIPCHECK(hipMalloc((void**)&Bd, SIZE*sizeof(int)));
    HIPCHECK(hipHostMalloc((void**)&C, SIZE*sizeof(int)));
    HIPCHECK(hipMemcpy(Ad,A,SIZE*sizeof(int),hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(Bd,B,SIZE*sizeof(int),hipMemcpyHostToDevice));

    hipModule_t Module;
    hipFunction_t Function1,Function2;
    HIPCHECK(hipModuleLoad(&Module, fileName));
    HIPCHECK(hipModuleGetFunction(&Function1, Module, kernel_name1))
    HIPCHECK(hipModuleGetFunction(&Function2, Module, kernel_name2))

    struct {
        void* _Ad;
        void* _Bd;
        void* _Cd;
        int _n;
    } args1,args2;

    args1._Ad = Ad;
    args1._Bd = Bd;
    args1._Cd = C;
    args1._n  = N;

    args2._Ad = NULL;
    args2._Bd = NULL;
    args2._Cd = NULL;
    args2._n  = 0; 

    size_t size1=sizeof(args1);
    size_t size2=sizeof(args2);
    void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,HIP_LAUNCH_PARAM_END};
    void* config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,HIP_LAUNCH_PARAM_END};

    auto start=high_resolution_clock::now();
    HIPCHECK(hipExtModuleLaunchKernel(Function2, 1,1, 1, 1,1 ,1 , 0, stream1, NULL, (void**)&config2, NULL, NULL,0 ));
    HIPCHECK(hipExtModuleLaunchKernel(Function1, N,N, 1, 32,32 ,1 , 0, stream1, NULL, (void**)&config1, NULL, NULL,0 ));
    HIPCHECK(hipStreamSynchronize(stream1));

    auto stop=high_resolution_clock::now();
    auto duration1=duration_cast<microseconds>(stop-start);

    start=high_resolution_clock::now();
    HIPCHECK(hipExtModuleLaunchKernel(Function2, 1,1, 1, 1,1 ,1 , 0, stream1, NULL, (void**)&config2, NULL, NULL,1 ));
    HIPCHECK(hipExtModuleLaunchKernel(Function1, N,N, 1, 32,32 ,1 , 0, stream1, NULL, (void**)&config1, NULL, NULL,1 ));
    HIPCHECK(hipStreamSynchronize(stream1));

    stop=high_resolution_clock::now();
    auto duration2=duration_cast<microseconds>(stop-start);

    bool testStatus = true;

    if(! (duration2.count() < duration1.count())) {
        std::cout<<"Test failed as there was no time gain observed when two kernels were launched using hipExtModuleLaunchKernel() with flag 1."<<std::endl;
        testStatus = false; 
    }

    unsigned long int mismatch=0;
    for(int i=0;i<N;i++) {
        for(int j=0;j<N;j++) {
            if(C[i*N + j] != N)
                mismatch++;
        }
    }
    if(mismatch) {
        std::cout<<"Test failed as the result of matrix multiplication was found incorrect."<<std::endl;
        testStatus = false; 
    }

    delete[] A;
    delete[] B; 
    HIPCHECK(hipFree(Ad));
    HIPCHECK(hipFree(Bd));
    HIPCHECK(hipHostFree(C));
    HIPCHECK(hipModuleUnload(Module));
    if(testStatus)
        passed();
}
