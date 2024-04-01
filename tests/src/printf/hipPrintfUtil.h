/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef _PRINTFUTIL_H_
#define _PRINTFUTIL_H_

#include "printf_common.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <vector>
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream, std::stringbu
#include <iostream>
#include <cstring>

static std::vector<std::string> correctBufferInt;
static std::vector<std::string> correctBufferFloat;
static std::vector<std::string> correctBufferOctal;
static std::vector<std::string> correctBufferUnsigned;
static std::vector<std::string> correctBufferHexadecimal;

enum PrintfTestType {
  TYPE_INT,
  TYPE_FLOAT,
  TYPE_FLOAT_LIMITS,
  TYPE_OCTAL,
  TYPE_UNSIGNED,
  TYPE_HEXADEC,
  TYPE_CHAR,
  TYPE_STRING,
  TYPE_VECTOR,
  TYPE_ADDRESS_SPACE,
  TYPE_COUNT
};

typedef enum {
  kuchar = 0,
  kchar = 1,
  kushort = 2,
  kshort = 3,
  kuint = 4,
  kint = 5,
  kfloat = 6,
  kulong = 7,
  klong = 8,
  kdouble = 9,
  kvector = 10,
  kTypeCount  // always goes last
} Type;

struct printDataGenParameters {
  const char* genericFormat;
  const char* dataRepresentation;
  const char* vectorFormatFlag;
  const char* vectorFormatSpecifier;
  const char* dataType;
  const char* vectorSize;
  const char* addrSpaceArgumentTypeQualifier;
  const char* addrSpaceVariableTypeQualifier;
  const char* addrSpaceParameter;
  const char* addrSpacePAdd;
};

struct testCase {
  enum PrintfTestType _type;                           //(data)type for test
  std::vector<std::string>& _correctBuffer;            //look-up table for correct results for printf
  std::vector<printDataGenParameters>& _genParameters; //auxiliary data to build the code for kernel source
  void (*printFN)(printDataGenParameters&,
                    char*,
                    const size_t);                       //function pointer for generating reference results
  Type dataType;                                       //the data type that will be printed during reference result generation (used for setting rounding mode)
  int  numOfTests;                                     // num of test performed based on genparam array elements
};

static void intRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize) {
    snprintf(refResult, refSize, params.genericFormat, atoi(params.dataRepresentation));
}

static void floatRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize) {
    snprintf(refResult, refSize, params.genericFormat, strtof(params.dataRepresentation, NULL));
}

static void octalRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize) {
    const unsigned long int data = strtoul(params.dataRepresentation, NULL, 10);
    snprintf(refResult, refSize, params.genericFormat, data);
}

static void unsignedRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize) {
    const unsigned long int data = strtoul(params.dataRepresentation, NULL, 10);
    snprintf(refResult, refSize, params.genericFormat, data);
}

static void hexRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize) {
    const unsigned long int data = strtoul(params.dataRepresentation, NULL, 0);
    snprintf(refResult, refSize, params.genericFormat, data);
}

//==================================

// int

//==================================

//------------------------------------------------------

// [string] format  | [string] int-data representation |

//------------------------------------------------------

std::vector<printDataGenParameters> printIntGenParameters = {
  //(Minimum)Five-wide,default(right)-justified
  {"%5d","10"},
  //(Minimum)Five-wide,left-justified
  {"%-5d","10"},
  //(Minimum)Five-wide,default(right)-justified,zero-filled
  {"%05d","10"},
  //(Minimum)Five-wide,default(right)-justified,with sign
  {"%+5d","10"},
  //(Minimum)Five-wide ,left-justified,with sign
  {"%-+5d","10"},
  //(Minimum)Five-digit(zero-filled in absent digits),default(right)-justified
  {"%.5i","100"},
  //(Minimum)Six-wide,Five-digit(zero-filled in absent digits),default(right)-justified
  {"%6.5i","100"},
  //0 and - flag both apper ==>0 is ignored,left-justified,capital I
  {"%-06i","100"},
  //(Minimum)Six-wide,Five-digit(zero-filled in absent digits),default(right)-justified
  {"%06.5i","100"}
};

testCase testCaseInt = {
  TYPE_INT,
  correctBufferInt,
  printIntGenParameters,
  intRefBuilder,
  kint,
  9
};


//==============================================

// float

//==============================================



//--------------------------------------------------------

// [string] format |  [string] float-data representation |

//--------------------------------------------------------

std::vector<printDataGenParameters> printFloatGenParameters = {
  //Default(right)-justified
  {"%f","10.3456"},
  //One position after the decimal,default(right)-justified
  {"%.1f","10.3456"},
  //Two positions after the decimal,default(right)-justified
  {"%.2f","10.3456"},
  //(Minimum)Eight-wide,three positions after the decimal,default(right)-justified
  {"%8.3f","10.3456"},
  //(Minimum)Eight-wide,two positions after the decimal,zero-filled,default(right)-justified
  {"%08.2f","10.3456"},
  //(Minimum)Eight-wide,two positions after the decimal,left-justified
  {"%-8.2f","10.3456"},
  //(Minimum)Eight-wide,two positions after the decimal,with sign,default(right)-justified
  {"%+8.2f","-10.3456"},
  //Zero positions after the decimal([floor]rounding),default(right)-justified
  {"%.0f","0.1"},
  //Zero positions after the decimal([ceil]rounding),default(right)-justified
  {"%.0f","0.6"},
  //Zero-filled,default positions number after the decimal,default(right)-justified
  {"%0f","0.6"},
  //Double argument representing floating-point,used by f style,default(right)-justified
  {"%4g","12345.6789"},
  //Double argument representing floating-point,used by e style,default(right)-justified
  {"%4.2g","12345.6789"},
  //Double argument representing floating-point,used by f style,default(right)-justified
  {"%4G","0.0000023"},
  //Double argument representing floating-point,used by e style,default(right)-justified
  {"%4G","0.023"},
  //Double argument representing floating-point,with exponent,left-justified,default(right)-justified
  {"%-#20.15e","789456123.0"},
  //Double argument representing floating-point,with exponent,left-justified,with sign,capital E,default(right)-justified ????
  {"%+#21.15E","789456123.0"},
  //Double argument representing floating-point,in [-]xh.hhhhpAd style
  {"%.6a","0.1"},
  //(Minimum)Ten-wide,Double argument representing floating-point,in xh.hhhhpAd style,default(right)-justified
  {"%10.2a","9990.235"},
};

//---------------------------------------------------------

//Test case for float                                     |

//---------------------------------------------------------

testCase testCaseFloat = {
  TYPE_FLOAT,
  correctBufferFloat,
  printFloatGenParameters,
  floatRefBuilder,
  kfloat,
  18
};


//==============================================

// float limits

//==============================================



//--------------------------------------------------------

// [string] format |  [string] float-data representation |

//--------------------------------------------------------


std::vector<printDataGenParameters> printFloatLimitsGenParameters = {
  //Infinity (1.0/0.0)
  {"%f","1.0f/0.0f"},
  //NaN
  {"%f","sqrt(-1.0f)"},
  //NaN
  {"%f","acos(2.0f)"}
};
//--------------------------------------------------------

//  Lookup table - [string]float-correct buffer             |

//--------------------------------------------------------

std::vector<std::string> correctBufferFloatLimits = {
  "inf",
  "-nan",
  "nan"
};

//---------------------------------------------------------

//Test case for float                                     |

//---------------------------------------------------------

testCase testCaseFloatLimits = {
  TYPE_FLOAT_LIMITS,
  correctBufferFloatLimits,
  printFloatLimitsGenParameters,
  NULL,
  kfloat,
  3
};

//=========================================================

// octal

//=========================================================



//---------------------------------------------------------

// [string] format  | [string] octal-data representation  |

//---------------------------------------------------------

std::vector<printDataGenParameters> printOctalGenParameters = {
  //Default(right)-justified
  {"%o","10"},
  //Five-digit,default(right)-justified
  {"%.5o","10"},
  //Default(right)-justified,increase precision
  {"%#o","100000000"},
  //(Minimum)Four-wide,Five-digit,0-flag ignored(because of precision),default(right)-justified
  {"%04.5o","10"}
};

//-------------------------------------------------------

//Test case for octal                                   |

//-------------------------------------------------------

testCase testCaseOctal = {
  TYPE_OCTAL,
  correctBufferOctal,
  printOctalGenParameters,
  octalRefBuilder,
  kulong,
  4
};



//=========================================================

// unsigned

//=========================================================



//---------------------------------------------------------

// [string] format  | [string] unsined-data representation  |

//---------------------------------------------------------

std::vector<printDataGenParameters> printUnsignedGenParameters = {
  //Default(right)-justified
  {"%u","10"},
};

//-------------------------------------------------------

//Test case for octal                                   |

//-------------------------------------------------------

testCase testCaseUnsigned = {
  TYPE_UNSIGNED,
  correctBufferUnsigned,
  printUnsignedGenParameters,
  unsignedRefBuilder,
  kulong,
  1
};



//=======================================================

// hexadecimal

//=======================================================



//--------------------------------------------------------------

// [string] format  | [string] hexadecimal-data representation |

//--------------------------------------------------------------

std::vector<printDataGenParameters> printHexadecimalGenParameters = {
  //Add 0x,low x,default(right)-justified
  {"%#x","0xABCDEF"},
  //Add 0x,capital X,default(right)-justified
  {"%#X","0xABCDEF"},
  //Not add 0x,if zero,default(right)-justified
  {"%#X","0"},
  //(Minimum)Eight-wide,default(right)-justified
  {"%8x","399"},
  //(Minimum)Four-wide,zero-filled,default(right)-justified
  {"%04x","399"}
};

//--------------------------------------------------------------

//Test case for hexadecimal                                    |

//--------------------------------------------------------------

testCase testCaseHexadecimal = {
  TYPE_HEXADEC,
  correctBufferHexadecimal,
  printHexadecimalGenParameters,
  hexRefBuilder,
  kulong,
  5
};



//=============================================================

// char

//=============================================================



//-----------------------------------------------------------

// [string] format  | [string] string-data representation   |

//-----------------------------------------------------------

std::vector<printDataGenParameters> printCharGenParameters = {
  //Four-wide,zero-filled,default(right)-justified
  {"%4c","'1'"},
  //Four-wide,left-justified
  {"%-4c","\'1\'"},
  //(unsigned) int argument,default(right)-justified
  {"%c","66"}
};

//---------------------------------------------------------

// Lookup table -[string] char-correct buffer             |

//---------------------------------------------------------

std::vector<std::string> correctBufferChar = {
  "   1",
  "1   ",
  "B",
};




//----------------------------------------------------------

//Test case for char                                       |

//----------------------------------------------------------

testCase testCaseChar = {
  TYPE_CHAR,
  correctBufferChar,
  printCharGenParameters,
  NULL,
  kchar,
  3
};

//==========================================================

// string

//==========================================================



//--------------------------------------------------------

// [string]format | [string] string-data representation  |

//--------------------------------------------------------

std::vector<printDataGenParameters> printStringGenParameters = {
  //(Minimum)Four-wide,zero-filled,default(right)-justified
  {"%4s","\"foo\""},
  //One-digit(precision ignored),left-justified
  {"%.1s","\"foo\""},
  //%% specification
  {"%s","\"%%\""},
};

//---------------------------------------------------------

// Lookup table -[string] string-correct buffer           |

//---------------------------------------------------------

std::vector<std::string> correctBufferString = {
  " foo",
  "f",
  "%%",
};


//---------------------------------------------------------

//Test case for string                                    |

//---------------------------------------------------------

testCase testCaseString = {
  TYPE_STRING,
  correctBufferString,
  printStringGenParameters,
  NULL,
  kchar,
  3
};

std::vector<testCase*> allTestCase = {&testCaseInt, &testCaseFloat, &testCaseFloatLimits, &testCaseOctal, &testCaseUnsigned,
                                      &testCaseHexadecimal, &testCaseChar, &testCaseString};
#endif