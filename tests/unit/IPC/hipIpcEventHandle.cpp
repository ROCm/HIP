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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_RUNTIME VDI NVCC_OPTIONS -std=c++11
 * TEST: %t EXCLUDE_HIP_RUNTIME VDI
 * HIT_END
 */

/* 
 * Unit test for hipIpcGetEventHandle, hipIpcOpenEventHandle
 * This covers positive test cases and only verifies if the ipc event received through event handle in the child process is valid and whether this ipc event can be used with various event APIs successfully 
 * This also covers opening event handle on another device and tests the above cases on multi gpu setup
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

typedef struct mem_handle {
    hipIpcEventHandle_t eventHandle;
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
        hipEvent_t start;
        HIPCHECK(hipEventCreateWithFlags(&start, hipEventDisableTiming|hipEventInterprocess));
        // hipIpcGetEventHandle is used to get an interprocess handle for an existing event
        HIPCHECK(hipIpcGetEventHandle(&shrd_mem->eventHandle, start));

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
        hipEvent_t ipc_event;
        hipStream_t stream;
        HIPCHECK(hipStreamCreate(&stream));
   
        // verifies hipIpcOpenEventHandle on the same device in the child process
        HIPCHECK(hipIpcOpenEventHandle(&ipc_event, shrd_mem->eventHandle));
        
        // Below operations are to test the validity of the ipc_event received thru ipc open event handle and ensure it works with the event APIs
        HIPCHECK(hipEventRecord(ipc_event, NULL));

        HIPCHECK(hipEventRecord(ipc_event, stream))
        
        HIPCHECK(hipStreamWaitEvent(stream, ipc_event, 0));
        
        HIPCHECK(hipEventSynchronize(ipc_event));
        HIPCHECK(hipEventQuery(ipc_event));
     
        HIPCHECK(hipStreamDestroy(stream));
        // Destroys the ipc event receieved thru ipc open event handle
        HIPCHECK(hipEventDestroy(ipc_event));
        
        // Multi Gpu Setup 
        if (shrd_mem->device1){
            
            HIPCHECK(hipSetDevice(shrd_mem->device1));

            hipStream_t stream1;
            hipEvent_t ipc_event1;
   
            HIPCHECK(hipStreamCreate(&stream1));
            // verifies hipIpcOpenEventHandle on the different device in the child process
            HIPCHECK(hipIpcOpenEventHandle(&ipc_event1, shrd_mem->eventHandle));
        
            // Below operations are to test the validity of the ipc_event received thru ipc open event handle and ensure it works with the event APIs
            HIPCHECK(hipEventRecord(ipc_event1, NULL));

            HIPCHECK(hipEventRecord(ipc_event1, stream1))
        
            HIPCHECK(hipStreamWaitEvent(stream1, ipc_event1, 0));
        
            HIPCHECK(hipEventSynchronize(ipc_event1));
        
            HIPCHECK(hipEventQuery(ipc_event1));
        
            HIPCHECK(hipStreamDestroy(stream1));
            // Destroys the ipc event receieved thru ipc open event handle
            HIPCHECK(hipEventDestroy(ipc_event1));
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
