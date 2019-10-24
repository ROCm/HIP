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
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
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

int main(int argc, char ** argv) {
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

