#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "hip_runtime.h"
#include "test_common.h"

#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1

int main() {

  int *hostA;
  int *hostB;

  int *deviceA;
  int *deviceB;

  int i;
  int errors;

  hostA = (int *)malloc(NUM * sizeof(int));
  hostB = (int *)malloc(NUM * sizeof(int));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = i;
  }

  HIPCHECK(hipMalloc((void**)&deviceA, NUM * sizeof(int)));
  HIPCHECK(hipMalloc((void**)&deviceB, NUM * sizeof(int)));

  hipStream_t s;
  HIPCHECK(hipStreamCreate(&s));


  // hostB -> deviceB -> hostA
#define ASYNC 1
#if ASYNC
  HIPCHECK(hipMemcpyAsync(deviceB, hostB, NUM*sizeof(int), hipMemcpyHostToDevice, s));
  HIPCHECK(hipMemcpyAsync(hostA, deviceB, NUM*sizeof(int), hipMemcpyDeviceToHost, s));
#else
  HIPCHECK(hipMemcpy(deviceB, hostB, NUM*sizeof(int), hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(hostA, deviceB, NUM*sizeof(int), hipMemcpyDeviceToHost));
#endif

  HIPCHECK(hipStreamSynchronize(s));
  HIPCHECK(hipDeviceSynchronize());

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i])) {
      errors++;
    }
  }

  HIPCHECK(hipStreamDestroy(s));

  HIPCHECK(hipFree(deviceA));
  HIPCHECK(hipFree(deviceB));

  free(hostA);
  free(hostB);

  //hipResetDefaultAccelerator();

  if(errors != 0){
    HIPASSERT(1 == 2);
  }else{
    passed();
  }

  return errors;
}
