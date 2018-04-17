#include <iostream>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <stddef.h>

#include "hip/hip_texture_types.h"


using namespace std;

#define HIPCHECK(error) \
{\
string strr=hipGetErrorString(error);\

}

int main()
{
    hipDevice_t device;
    size_t Nbytes = N*sizeof(int);
  int numDevices = 0;
    int *A_d, *B_d, *C_d, *X_d, *Y_d, *Z_d;
    int *A_h, *B_h, *C_h ;

    HIPCHECK(hipGetDeviceCount(&numDevices));
     if(numDevices <= 1)
     {
     // test is not applicable for single device configs
      exit(1);
     } 
     else
     {
        HIPCHECK(hipSetDevice(0));
        unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
       HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
        HIPCHECK(hipSetDevice(1));
        HIPCHECK(hipMalloc(&X_d,Nbytes));
        HIPCHECK(hipMalloc(&Y_d,Nbytes));
        HIPCHECK(hipMalloc(&Z_d,Nbytes));

        
        HIPCHECK(hipSetDevice(0));
        HIPCHECK ( hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
        HIPCHECK ( hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));
        hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d,B_d, C_d, N);
        HIPCHECK ( hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
        HIPCHECK (hipDeviceSynchronize());
        HipTest::checkVectorADD(A_h, B_h, C_h, N);
    
         
        HIPCHECK(hipSetDevice(1));
        hipMemcpyPeer(X_d, 1, A_d, 0, Nbytes); //this call is eqv to hipMemcpy(hipMemcpyD2D) which goes via stg bufs.
        hipMemcpyPeer(Y_d, 1, B_d, 0, Nbytes);

        hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, X_d,Y_d, Z_d, N);
        HIPCHECK ( hipMemcpy(C_h, Z_d, Nbytes, hipMemcpyDeviceToHost));
        HIPCHECK (hipDeviceSynchronize());
        HipTest::checkVectorADD(A_h, B_h, C_h, N);



       passed();
     }
