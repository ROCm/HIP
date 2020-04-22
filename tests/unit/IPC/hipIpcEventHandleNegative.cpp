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
 * BUILD: %t %s ../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <semaphore.h>

#define N 10000
#define Nbytes N*sizeof(int)

int enablePeers(int dev0, int dev1) {
    int canAccessPeer01, canAccessPeer10;
    HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer01, dev0, dev1));
    HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer10, dev1, dev0));
    if (!canAccessPeer01 || !canAccessPeer10) {
        return -1;
    }

    HIPCHECK(hipSetDevice(dev0));
    HIPCHECK(hipDeviceEnablePeerAccess(dev1, 0 /*flags*/));
    HIPCHECK(hipSetDevice(dev1));
    HIPCHECK(hipDeviceEnablePeerAccess(dev0, 0 /*flags*/));
    HIPCHECK(hipSetDevice(dev0));

    return 0;
};

typedef struct mem_handle {
    hipIpcEventHandle_t eventHandle;
    int devCount;
} hip_ipc_t;

bool runipcNegativeTest()
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

    pid = fork();
    
    if(pid != 0) {
    
        int dev0 = 0;
        int dev1 = 1;
        HIPCHECK(hipGetDeviceCount(&shrd_mem->devCount));
        if (shrd_mem->devCount > 1) {
           if (enablePeers(dev0, dev1) == -1) {
              printf("warning : could not find peer gpus, only single device test is run\n");
              shrd_mem->devCount = 1;
           }
        } 

        hipEvent_t start;
        HIPCHECK(hipEventCreateWithFlags(&start, hipEventDisableTiming));
        HIPASSERT(hipIpcGetEventHandle(&shrd_mem->eventHandle, start)==hipErrorInvalidHandle);
        
        //CUDA crashes, hence skipping this
        #ifndef __HIP_PLATFORM_NVCC__
        HIPCHECK(hipEventCreateWithFlags(&start, hipEventInterprocess));
        
        HIPASSERT(hipIpcGetEventHandle(&shrd_mem->eventHandle, NULL)==hipErrorInvalidHandle);
        #endif
        HIPASSERT(hipIpcGetEventHandle(&shrd_mem->eventHandle, start)==hipErrorInvalidHandle);
        
        shrd_mem = NULL;
        HIPASSERT(hipIpcGetEventHandle(&shrd_mem->eventHandle, start)==hipErrorInvalidHandle);

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
         
        hipEvent_t ipc_event;

        //CUDA crashes, hence skipping this 
        #ifndef __HIP_PLATFORM_NVCC__
        HIPASSERT(hipIpcOpenEventHandle(&ipc_event, shrd_mem->eventHandle)==hipErrorInvalidValue);
        #endif
        HIPASSERT(hipIpcOpenEventHandle(NULL, shrd_mem->eventHandle)==hipErrorInvalidValue);
        
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
    return true;
}

int main(){
   
    bool status = true;
    status &= runipcNegativeTest();

    if(status){
       passed();
    }
  
    return 0;
}
