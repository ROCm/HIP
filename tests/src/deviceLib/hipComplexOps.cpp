/*
 * Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
 
/* HIT_START
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t EXCLUDE_HIP_PLATFORM nvcc
 * HIT_END
 */

#include "test_common.h"
#include "hip/hip_complex.h"
#include <stdio.h>

template<typename T>
void validateResult(T a, T r)
{
   if( a != r)
   {  
      failed("Result did not match!");
   }
}

//Intension here to add compile time check to make sure all casting/constructs are fine.
void hipDoubleComplexConstruct()
{
   // Default construct
   hipDoubleComplex dc1;
   hipDoubleComplex dc2 = make_hipDoubleComplex(2.0, 3.0);

   // Casting
   double2 dVar; dVar.x = 4.0; dVar.y = 5.0;
   hipDoubleComplex dc3 = (hipDoubleComplex)dVar; // c style casting
   hipDoubleComplex dc4 = static_cast<hipDoubleComplex>(dVar); // c++ casting

   // copy constructor & assignment
   hipDoubleComplex dc5 = hipDoubleComplex(dVar);
   hipDoubleComplex dc6; dc6 = dc2;

   // pointer construct
   double2 *ptrDVar = new double2;
   ptrDVar->x = 2.0;
   ptrDVar->y = 3.0;

   hipDoubleComplex *ptrDC = new hipDoubleComplex;
   hipDoubleComplex *ptrDC1 = (hipDoubleComplex*)ptrDVar;// c casting
   hipDoubleComplex *ptrDC2 = static_cast<hipDoubleComplex *>(ptrDVar); // c++ casting
   hipDoubleComplex *ptrDC3;
   ptrDC3 = ptrDC1;

   // Casting hipDoubleComplex -> double2
   double2 dd1 = (double2)dc2;
   double2 dd2 = static_cast<double2>(dc2);
}

//Intension here to add compile time check to make sure all casting/constructs are fine.
void hipFloatComplexConstruct()
{
   // Default construct
   hipFloatComplex fc1;
   hipFloatComplex fc2 = make_hipFloatComplex(2.0, 3.0);

   //Casting
   float2 fVar; fVar.x = 4.0; fVar.y = 5.0;
   hipFloatComplex fc3 = (hipFloatComplex)fVar; // c style casting
   hipFloatComplex fc4 = static_cast<hipFloatComplex>(fVar); // c++ casting

   // copy constructor & assignment
   hipFloatComplex fc5 = hipFloatComplex(fVar);
   hipFloatComplex fc6; fc6 = fc2;

   // pointer construct
   float2 *ptrfVar = new float2;
   ptrfVar->x = 2.0;
   ptrfVar->y = 3.0;

   hipFloatComplex *ptrfc = new hipFloatComplex;
   hipFloatComplex *ptrfc1 = (hipFloatComplex*)ptrfVar;// c casting
   hipFloatComplex *ptrfc2 = static_cast<hipFloatComplex *>(ptrfVar); // c++ casting
   hipFloatComplex *ptrfc3;
   ptrfc3 = ptrfc1;

   // Casting hipFloatComplex -> float2
   float2 dd1 = (float2)fc2;
   float2 dd2 = static_cast<float2>(fc2);
}

int main(int argc, char ** argv) {

   // Test casting and constructs
   hipDoubleComplexConstruct();

   hipDoubleComplex x = make_hipDoubleComplex(2.0, 3.0);
   hipDoubleComplex y = make_hipDoubleComplex(4.0, 2.0);

   // Test all operations for doubleComplex
   validateResult(x+y, make_hipDoubleComplex(6.0, 5.0));
   validateResult(y-x, make_hipDoubleComplex(2.0, -1.0));
   validateResult(x*y, make_hipDoubleComplex(2.0, 16.0));
   validateResult(x/y, make_hipDoubleComplex(0.700000, 0.400000));

   hipDoubleComplex a,b,c;
   a=b=c=y;
   validateResult(a += x, make_hipDoubleComplex(6.0, 5.0));
   validateResult(b -= x, make_hipDoubleComplex(2.0, -1.0));
   validateResult(c *= x, make_hipDoubleComplex(2.0, 16.0));

   // Test casting and constructs
   hipFloatComplexConstruct();

   // Test all operations for floatComplex
   hipFloatComplex f_x = make_hipFloatComplex(2.0, 3.0);
   hipFloatComplex f_y = make_hipFloatComplex(4.0, 2.0);

   validateResult(f_x+f_y, make_hipFloatComplex(6.0, 5.0));
   validateResult(f_y-f_x, make_hipFloatComplex(2.0, -1.0));
   validateResult(f_x*f_y, make_hipFloatComplex(2.0, 16.0));
   validateResult(f_x/f_y, make_hipFloatComplex(0.700000, 0.400000));

   hipFloatComplex f_a,f_b,f_c;
   f_a = f_b = f_c = f_y;
   validateResult(f_a += f_x, make_hipFloatComplex(6.0, 5.0));
   validateResult(f_b -= f_x, make_hipFloatComplex(2.0, -1.0));
   validateResult(f_c *= f_x, make_hipFloatComplex(2.0, 16.0));

   passed();         
   return 0;               
}

