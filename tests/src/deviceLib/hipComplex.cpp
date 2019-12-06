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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t EXCLUDE_HIP_PLATFORM nvcc
 * HIT_END
 */

#include <iostream>
#include "test_common.h"
#include "hip/hcc_detail/hip_complex.h"

/*
 * Overall purpose is to test supported operations for hip*Complex number.
 * 1. testFloatComplexConstruct() / testDoubleComplexConstruct() :
 *       Tests cover default construct, copy construct, assignment operator, C/C++ style typecasting.
 *       These are compile time test to make sure compiler is happy.
 *       Added these test as there was issue reproted where typecasting was broken.
 *
 * 2. testComplexFloatOperations() / testComplexDoubleOperations() :
 *      Tests cover all supported CPU operations for hip*Complex
 *
 * 3. testComplexOperationsOnDevice() :
 *      Test complex operation on device and runs same operation on CPU in order to confirm results
*/
__global__ void getSqAbs(float* A, float* B, float* C) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    C[tx] = hipCsqabsf(make_hipFloatComplex(A[tx], B[tx]));
}

template<typename T>
void validateResult(T acutal, T result)
{
   if( acutal != result)
   {
      failed("Result did not match!");
   }
}

//Intension here to add compile time check to make sure all casting/constructs are fine.
void testFloatComplexConstruct()
{
   // Default construct
   hipFloatComplex fc1;
   hipFloatComplex fc2 = make_hipFloatComplex(2.0, 3.0);

   // Casting
   float2 fVar; fVar.x = 4.0; fVar.y = 5.0;
   hipFloatComplex fc3 = (hipFloatComplex)fVar; // c style casting
   hipFloatComplex fc4 = static_cast<hipFloatComplex>(fVar); // C++ casting

   // copy constructor & assignment operator
   hipFloatComplex fc5 = hipFloatComplex(fVar);
   fc4 = fc2;

   // pointer construct
   float2 *ptrfVar = new float2;
   ptrfVar->x = 2.0;
   ptrfVar->y = 3.0;

   hipFloatComplex *ptrfc = new hipFloatComplex;
   hipFloatComplex *ptrfc1 = (hipFloatComplex*)ptrfVar;// C style casting
   hipFloatComplex *ptrfc2 = static_cast<hipFloatComplex *>(ptrfVar); // C++ style casting
   hipFloatComplex *ptrfc3;
   ptrfc3 = ptrfc1;

   // Casting hipFloatComplex -> float2
   float2 f2Var1 = (float2)fc2; // C style casting
   float2 f2Var2 = static_cast<float2>(fc2); // C++ style casting
}

//Intension here to add compile time check to make sure all casting/constructs are fine.
void testDoubleComplexConstruct()
{
   // Default construct
   hipDoubleComplex dc1;
   hipDoubleComplex dc2 = make_hipDoubleComplex(2.0, 3.0);

   // Casting
   double2 dVar; dVar.x = 4.0; dVar.y = 5.0;
   hipDoubleComplex dc3 = (hipDoubleComplex)dVar; //C style casting
   hipDoubleComplex dc4 = static_cast<hipDoubleComplex>(dVar); // C++ style casting

   // copy constructor & assignment
   hipDoubleComplex dc5 = hipDoubleComplex(dVar);
   dc4 = dc2;

   // pointer construct
   double2 *ptrDVar = new double2;
   ptrDVar->x = 2.0;
   ptrDVar->y = 3.0;

   hipDoubleComplex *ptrDC = new hipDoubleComplex;
   hipDoubleComplex *ptrDC1 = (hipDoubleComplex*)ptrDVar;// C style casting
   hipDoubleComplex *ptrDC2 = static_cast<hipDoubleComplex *>(ptrDVar); // C++ style casting
   hipDoubleComplex *ptrDC3;
   ptrDC3 = ptrDC1;

   // Casting hipDoubleComplex -> double2
   double2 d2Var1 = (double2)dc2; // C style casting
   double2 d2Var2 = static_cast<double2>(dc2); // C++ style casting
}

void testComplexFloatOperations()
{
    // Test all operations for floatComplex
    hipFloatComplex f_x = make_hipFloatComplex(2.0, 3.0);
    hipFloatComplex f_y = make_hipFloatComplex(4.0, 2.0);

    // Conjugate of 2+3i
    hipFloatComplex actual = make_hipFloatComplex(2.0, -3.0);
    validateResult(actual, hipConjf(f_x));

    // abs of complex number
    float actualVal = 3.60555124;
    validateResult(actualVal, hipCabsf(f_x));

    // Complex Add
    actual = make_hipFloatComplex(6.0, 5.0);
    validateResult(actual, f_x + f_y);
    validateResult(actual, hipCaddf(f_x,f_y));
    //validateResult(actual, f_x += f_y); //TODO: currently broken

    // Complex sub
    actual = make_hipFloatComplex(2.0, -1.0);
    validateResult(actual, f_y - f_x);
    validateResult(actual, hipCsubf(f_y,f_x));
    //validateResult(f_b -= f_x, make_hipFloatComplex(2.0, -1.0)); //TODO: currently broken

    // Complex multiplication
    actual = make_hipFloatComplex(2.0, 16.0);
    validateResult(actual, f_x * f_y);
    validateResult(actual, hipCmulf(f_x, f_y));
    //validateResult(f_c *= f_x, make_hipFloatComplex(2.0, 16.0)); TODO: currently broken

    // Complex division
    actual = make_hipFloatComplex(0.700000, 0.400000);
    validateResult(actual , f_x / f_y);
    validateResult(actual, hipCdivf(f_x, f_y));
}

void testComplexDoubleOperations()
{
    // Test all operations for doubleComplex
    hipDoubleComplex f_x = make_hipDoubleComplex(2.0, 3.0);
    hipDoubleComplex f_y = make_hipDoubleComplex(4.0, 2.0);

    // Conjugate of 2+3i
    hipDoubleComplex actual = make_hipDoubleComplex(2.0, -3.0);
    validateResult(actual, hipConj(f_x));

    // abs of complex number
    double actualVal = 3.6055512428283691;
    validateResult(actualVal, hipCabs(f_x));

    // Complex Add
    actual = make_hipDoubleComplex(6.0, 5.0);
    validateResult(actual, f_x + f_y);
    validateResult(actual, hipCadd(f_x,f_y));
    // validateResult(actual, f_x += f_y); TODO: currently broken

    // Complex sub
    actual = make_hipDoubleComplex(2.0, -1.0);
    validateResult(actual, f_y - f_x);
    validateResult(actual, hipCsub(f_y,f_x));
    //validateResult(f_b -= f_x, make_hipDoubleComplex(2.0, -1.0)); TODO: currently broken

    // Complex multiplication
    actual = make_hipDoubleComplex(2.0, 16.0);
    validateResult(actual, f_x * f_y);
    validateResult(actual, hipCmul(f_x, f_y));
    //validateResult(f_c *= f_x, make_hipDoubleComplex(2.0, 16.0)); TODO: currently broken

    // Complex division
    actual = make_hipDoubleComplex(0.700000, 0.400000);
    validateResult(actual , f_x / f_y);
    validateResult(actual, hipCdiv(f_x, f_y));
}

#define LEN 64
#define SIZE 64 << 2
void verifyResult(float *A, float *B, float *C)
{
    if(A == nullptr || B == nullptr || C == nullptr){
        failed("Bad input!");
    }

    for(int i = 0; i< LEN; ++i)
    {
        validateResult(C[i], hipCsqabsf(make_hipFloatComplex(A[i],B[i])));
    }
}
void testComplexOperationsOnDevice()
{
    float *A, *Ad, *B, *Bd, *C, *Cd;
    A = new float[LEN];
    B = new float[LEN];
    C = new float[LEN];

    for (uint32_t i = 0; i < LEN; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 1.0f;
        C[i] = 0;
    }

    HIPCHECK(hipMalloc((void**)&Ad, SIZE));
    HIPCHECK(hipMalloc((void**)&Bd, SIZE));
    HIPCHECK(hipMalloc((void**)&Cd, SIZE));
    HIPCHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(getSqAbs, dim3(1), dim3(LEN), 0, 0, Ad, Bd, Cd);
    HIPCHECK(hipGetLastError())

    HIPCHECK(hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost));

    verifyResult(A, B, C);

    HIPCHECK(hipFree(Ad));
    HIPCHECK(hipFree(Bd));
    HIPCHECK(hipFree(Cd));
    delete []A;
    delete []B;
    delete []C;
}

int main() {

    // Compile time test to make sure all type of casting and constructs are valid
    testFloatComplexConstruct();
    testDoubleComplexConstruct();

    // Tests validate all supported operations on HOST
    testComplexFloatOperations();
    testComplexDoubleOperations();

    // Validate operation on device
    testComplexOperationsOnDevice();

    passed();
}
