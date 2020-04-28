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

/* HIT_START
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_RUNTIME HCC NVCC_OPTIONS -std=c++11
 * TEST: %t EXCLUDE_HIP_RUNTIME HCC
 * HIT_END
 */

/* 
 * Unit test for hipIpcGetMemHandle, hipIpcOpenMemHandle, hipIpcCloseMemHandle
 * This covers positive test cases. It includes kernel operations and memory copies on the IPC device pointer received through hipIpcOpenMemHandle in the child process
 * This also covers opening mem handle on another device and tests the above cases on multi gpu setup
 */

#include "test_common.h"
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <semaphore.h>

#define N 10000
#define Nbytes N*sizeof(int)

bool enablePeers(int *gpu0, int *gpu1, int devCount) {
 
   for(int dev0 = 0; dev0 < devCount; ++dev0){
     for(int dev1 = 1; dev1 < devCount; ++dev1){
  
        int canAccessPeer01, canAccessPeer10;
        HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer01, dev0, dev1));
        HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer10, dev1, dev0));
        if (!canAccessPeer01 || !canAccessPeer10) {
           break;
        }

        HIPCHECK(hipSetDevice(dev0));
        HIPCHECK(hipDeviceEnablePeerAccess(dev1, 0 /*flags*/));
        HIPCHECK(hipSetDevice(dev1));
        HIPCHECK(hipDeviceEnablePeerAccess(dev0, 0 /*flags*/));
    
        *gpu0 = dev0; *gpu1 = dev1;
        return true;
     }
   }
   return false;
}

//Kernel
__global__ void simple_kernel(int* A, int size) {

    int tx = blockDim.x*blockIdx.x+threadIdx.x;

    if (tx < size){
        A[tx] = A[tx]+1;
    }
}

bool Asynccopy(int* S_d, int* A_h, int* B_h, int* C_h, int* D_h)
{   
   int *R_d;
   HIPCHECK(hipMalloc((void **) &R_d, Nbytes));
   hipStream_t stream;
   HIPCHECK(hipStreamCreate(&stream));

   HIPCHECK(hipMemcpyAsync(R_d, S_d, Nbytes, hipMemcpyDeviceToDevice, stream));
   HIPCHECK(hipMemcpyAsync(B_h, R_d, Nbytes, hipMemcpyDeviceToHost,stream));
   
   HIPCHECK(hipMemcpyAsync(S_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
   HIPCHECK(hipMemcpyAsync(C_h, S_d, Nbytes, hipMemcpyDeviceToHost, stream));

   HIPCHECK(hipMemcpyAsync(S_d, R_d, Nbytes, hipMemcpyDeviceToDevice, stream));
   HIPCHECK(hipMemcpyAsync(D_h, S_d, Nbytes, hipMemcpyDeviceToHost, stream));

   hipStreamSynchronize(stream);
   hipFree((void*)R_d);

   return true;
}

bool Synccopy(int* S_d, int* A_h, int* B_h, int* C_h, int* D_h)
{
   int *R_d;
   HIPCHECK(hipMalloc((void **) &R_d, Nbytes));

   HIPCHECK(hipMemcpy(R_d, S_d, Nbytes, hipMemcpyDeviceToDevice));
   HIPCHECK(hipMemcpy(B_h, R_d, Nbytes, hipMemcpyDeviceToHost));
   
   HIPCHECK(hipMemcpy(S_d, A_h, Nbytes, hipMemcpyHostToDevice));
   HIPCHECK(hipMemcpy(C_h, S_d, Nbytes, hipMemcpyDeviceToHost));

   HIPCHECK(hipMemcpy(S_d, R_d, Nbytes, hipMemcpyDeviceToDevice));
   HIPCHECK(hipMemcpy(D_h, S_d, Nbytes, hipMemcpyDeviceToHost));

   hipFree((void*)R_d);
   return true;
}

typedef struct mem_handle {
    hipIpcMemHandle_t memHandle;
    int device0;
    int device1;
} hip_ipc_t;

bool runipcTest()
{
    sem_t *sem_ob1, *sem_ob2;
    
    if ((sem_ob1 = sem_open ("/my-sem-object1", O_CREAT|O_EXCL, 0660, 0)) == SEM_FAILED) {
        perror ("my-sem-object1");
	exit (1);
    }
    if ((sem_ob2 = sem_open ("/my-sem-object2", O_CREAT|O_EXCL, 0660, 0)) == SEM_FAILED) {
        perror ("my-sem-object2");
	exit (1);
    }

    hip_ipc_t *shrd_mem=(hip_ipc_t *)mmap(NULL, sizeof(hip_ipc_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
    if (shrd_mem == MAP_FAILED) {
        perror ("mmap failed");
        exit (1);
    }
    
    pid_t pid;
    int *A_d;
    int *A_h, *B_h, *C_h, *D_h;

    A_h = (int*)malloc(Nbytes);
    B_h = (int*)malloc(Nbytes);
    C_h = (int*)malloc(Nbytes);
    D_h = (int*)malloc(Nbytes);

    for (int i = 0; i < N; i++) {
        A_h[i] = i;
    }
     
    pid = fork();
    
    if(pid != 0) {
        int dev0 = 0, dev1 = 0, devCount;
        HIPCHECK(hipGetDeviceCount(&devCount));
        if (devCount > 1) {
           if (enablePeers(&dev0, &dev1, devCount) == false) {
              printf("warning : could not find peer gpus, only single device test is run\n");
           }
           else {
              shrd_mem->device1 = dev1;
           }
        } 
        
        shrd_mem->device0 = dev0;  
        HIPCHECK(hipSetDevice(shrd_mem->device0));
        HIPCHECK(hipMalloc((void **) &A_d, Nbytes));
   
        HIPCHECK(hipIpcGetMemHandle(&shrd_mem->memHandle, (void *) A_d));

        HIPCHECK(hipMemcpy((void *) A_d, (void *) A_h, Nbytes, hipMemcpyHostToDevice));

        HIPCHECK(hipDeviceSynchronize());
	
        if(sem_post(sem_ob1) == -1) {
            perror ("sem_post error!"); 
            exit (1);
        }
 
    } else {
 		
        if(sem_wait(sem_ob1) == -1) {
            perror ("sem_wait error!"); 
            exit (1);
	}
         
        HIPCHECK(hipSetDevice(shrd_mem->device0));
        int *S_d = NULL;
        // verifies hipIpcOpenMemHandle on the same device in the child process 
        HIPCHECK(hipIpcOpenMemHandle((void **) &S_d, shrd_mem->memHandle, hipIpcMemLazyEnablePeerAccess));
        if (S_d == NULL) {
            std::cout << "hipIpcOpenMemHandle failed" << std::endl;
            exit(1); 
        }
        
        unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N); 
        //Test kernel launch and memcopies on the dev ptr got thru IPC
        hipLaunchKernelGGL(simple_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, S_d, N);
        
        //Async copies
        Asynccopy(S_d, A_h, B_h, C_h, D_h);
     
        //Sync copies
        Synccopy(S_d, A_h, B_h, C_h, D_h);
        
        //To-do: CloseMemHandle currently fails due to HSA issue - to enable after the fix
        //HIPCHECK(hipIpcCloseMemHandle((void*)S_d));
        
        for (unsigned i = 0; i < N; i++) {
           assert(C_h[i] == A_h[i]);
           assert(D_h[i] == A_h[i]+1);
           assert(B_h[i] == A_h[i]+1);
        }
        // Multi Gpu Setup 
        if (shrd_mem->device1){
            HIPCHECK(hipSetDevice(shrd_mem->device1));
            int *X_d = NULL;  
        
            //verifies hipIpcOpenMemHandle on the different device in the child process 
            HIPCHECK(hipIpcOpenMemHandle((void **) &X_d, shrd_mem->memHandle, hipIpcMemLazyEnablePeerAccess));
            if (X_d == NULL) {
               std::cout<< "hipIpcOpenMemHandle failed" << std::endl;
               exit(1); 
            }
            
            //Test kernel launch and memcopies on the dev ptr got thru IPC
            hipLaunchKernelGGL(simple_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, X_d, N);
            
            //Async copies
            Asynccopy(X_d, A_h, B_h, C_h, D_h);
     
            //Sync copies
            Synccopy(X_d, A_h, B_h, C_h, D_h);
        
            //to-do: CloseMemHandle currently fails due to HSA issue - To enable after the fix
            //HIPCHECK(hipIpcCloseMemHandle((void*)X_d));
        
            for (unsigned i = 0; i < N; i++) {
               assert(C_h[i] == A_h[i]);
               assert(D_h[i] == A_h[i]+2);
               assert(B_h[i] == A_h[i]+2);
            }   
        } 

        if(sem_post(sem_ob2) == -1) {
            perror ("sem_post error!"); 
            exit (1); 
        }

        if(sem_close(sem_ob1) == -1) {
            perror ("sem_close error!");
            exit (1);
        } 

        if(sem_close(sem_ob2) == -1) {
            perror ("sem_close error!");
            exit (1);
        }
        exit(0);	
    }

    if(sem_wait(sem_ob2) == -1) {
        perror ("sem_wait error!");
        exit (1);
    }
     
    if(sem_unlink("/my-sem-object1") == -1) {
        perror ("sem_unlink error!");
	exit (1);
    }
    if(sem_unlink("/my-sem-object2") == -1) {
        perror ("sem_unlink error!");
        exit (1);
    }
  
    munmap(shrd_mem, sizeof(hip_ipc_t)); 
    HIPCHECK(hipFree((void*)A_d));
    free(A_h);
    free(B_h);
    free(C_h);
    free(D_h);

    return true;
}

int main(){
   
    bool status = true;
    status &= runipcTest();

    if(status){
       passed();
    }
  
    return 0;
}
