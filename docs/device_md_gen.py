"""
Copyright (c) 2015-2021 Advanced Micro Devices, Inc. All rights reserved.

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
"""

"""
1. This files uses Python3 to run

List of device functions:
acosf
acoshf
asinf
asinhf
atan2f
atanf
atanhf
cbrtf
ceilf
copysignf
cosf
coshf
cospif
cyl_bessel_i0f
cyl_bessel_i1f
erfcf
erfcinvf
erfcxf
erff
erfinvf
exp10f
exp2f
expf
expm1f
fabsf
fdimf
fdividef
floorf
fmaf
fmaxf
fminf
fmodf
frexpf
hypotf
ilogbf
isfinite
isinf
isnan
j0f
j1f
jnf
ldexpf
lgammaf
llrintf
llroundf
log10f
log1pf
logbf
lrintf
lroundf
modff
nanf
nearbyintf
nextafterf
norm3df
norm4df
normcdff
normcdfinvf
normf
powf
rcbrtf
remainderf
remquof
rhypotf
rintf
rnorm3df
rnorm4df
rnormf
roundf
rsqrtf
scalblnf
scalbnf
signbit
sincosf
sincospif
sinf
sinhf
sinpif
sqrtf
tanf
tanhf
tgammaf
truncf
y0f
y1f
ynf
acos
acosh
asin
asinh
atan
atan2
atanh
cbrt
ceil
copysign
cos
cosh
cospi
cyl_bessel_i0
cyl_bessel_i1
erf
erfc
erfcinv
erfcx
erfinv
exp
exp10
exp2
expm1
fabs
fdim
floor
fma
fmax
fmin
fmod
frexp
hypot
ilogb
isfinite
isinf
isnan
j0
j1
jn
ldexp
lgamma
llrint
llround
log
log10
log1p
log2
logb
lrint
lround
modf
nan
nearbyint
nextafter
norm
norm3d
norm4d
normcdf
normcdfinv
pow
rcbrt
remainder
remquo
rhypot
rint
rnorm
rnorm3d
rnorm4d
round
rsqrt
scalbln
scalbn
signbit
sin
sincos
sincospi
sinh
sinpi
sqrt
tan
tanh
tgamma
trunc
y0
y1
yn
__cosf
__exp10f
__expf
__fadd_rd
__fadd_rn
__fadd_ru
__fadd_rz
__fdiv_rd
__fdiv_rn
__fdiv_ru
__fdiv_rz
__fdividef
__fmaf_rd
__fmaf_rn
__fmaf_ru
__fmaf_rz
__fmul_rd
__fmul_rn
__fmul_ru
__fmul_rz
__frcp_rd
__frcp_rn
__frcp_ru
__frcp_rz
__frsqrt_rn
__fsqrt_rd
__fsqrt_rn
__fsqrt_ru
__fsqrt_rz
__fsub_rd
__fsub_rn
__fsub_ru
__log10f
__log2f
__logf
__powf
__saturatef
__sincosf
__sinf
__tanf
__dadd_rd
__dadd_rn
__dadd_ru
__dadd_rz
__ddiv_rd
__ddiv_rn
__ddiv_ru
__ddiv_rz
__dmul_rd
__dmul_rn
__dmul_ru
__dmul_rz
__drcp_rd
__drcp_rn
__drcp_ru
__drcp_rz
__dsqrt_rd
__dsqrt_rn
__dsqrt_ru
__dsqrt_rz
__dsub_rd
__dsub_rn
__dsub_ru
__dsub_rz
__fma_rd
__fma_rn
__fma_ru
__fma_rz
__brev
__brevll
__byte_perm
__clz
__clzll
__ffs
__ffsll
__hadd
__mul24
__mul64hi
__mulhi
__popc
__popcll
__rhadd
__sad
__uhadd
__umul24
__umul64hi
__umulhi
__urhadd
__usad
__double2float_rd
__double2float_rn
__double2float_ru
__double2float_rz
__double2hiint
__double2int_rd
__double2int_rn
__double2int_ru
__double2int_rz
__double2ll_rd
__double2ll_rn
__double2ll_ru
__double2ll_rz
__double2loint
__double2uint_rd
__double2uint_rn
__double2uint_ru
__double2uint_rz
__double2ull_rd
__double2ull_rn
__double2ull_ru
__double2ull_rz
__double_as_longlong
__float2half_rn
__half2float
__float2half_rn
__half2float
__float2int_rd
__float2int_rn
__float2int_ru
__float2int_rz
__float2ll_rd
__float2ll_rn
__float2ll_ru
__float2ll_rz
__float2uint_rd
__float2uint_rn
__float2uint_ru
__float2uint_rz
__float2ull_rd
__float2ull_rn
__float2ull_ru
__float2ull_rz
__float_as_int
__float_as_uint
__hiloint2double
__int2double_rn
__int2float_rd
__int2float_rn
__int2float_ru
__int2float_rz
__int_as_float
__ll2double_rd
__ll2double_rn
__ll2double_ru
__ll2double_rz
__ll2float_rd
__ll2float_rn
__ll2float_ru
__ll2float_rz
__longlong_as_double
__uint2double_rn
__uint2float_rd
__uint2float_rn
__uint2float_ru
__uint2float_rz
__uint_as_float
__ull2double_rd
__ull2double_rn
__ull2double_ru
__ull2double_rz
__ull2float_rd
__ull2float_rn
__ull2float_ru
__ull2float_rz
__heq
__hge
__hgt
__hisinf
__hisnan
__hle
__hlt
__hne
__hbeq2
__hbge2
__hbgt2
__hble2
__hblt2
__hbne2
__heq2
__hge2
__hgt2
__hisnan2
__hle2
__hlt2
__hne2
__float22half2_rn
__float2half
__float2half2_rn
__float2half_rd
__float2half_rn
__float2half_ru
__float2half_rz
__floats2half2_rn
__half22float2
__half2float
half2half2
__half2int_rd
__half2int_rn
__half2int_ru
__half2int_rz
__half2ll_rd
__half2ll_rn
__half2ll_ru
__half2ll_rz
__half2short_rd
__half2short_rn
__half2short_ru
__half2short_rz
__half2uint_rd
__half2uint_rn
__half2uint_ru
__half2uint_rz
__half2ull_rd
__half2ull_rn
__half2ull_ru
__half2ull_rz
__half2ushort_rd
__half2ushort_rn
__half2ushort_ru
__half2ushort_rz
__half_as_short
__half_as_ushort
__halves2half2
__high2float
__high2half
__high2half2
__highs2half2
__int2half_rd
__int2half_rn
__int2half_ru
__int2half_rz
__ll2half_rd
__ll2half_rn
__ll2half_ru
__ll2half_rz
__low2float
__low2half
__low2half2
__low2half2
__lowhigh2highlow
__lows2half2
__short2half_rd
__short2half_rn
__short2half_ru
__short2half_rz
__uint2half_rd
__uint2half_rn
__uint2half_ru
__uint2half_rz
__ull2half_rd
__ull2half_rn
__ull2half_ru
__ull2half_rz
__ushort2half_rd
__ushort2half_rn
__ushort2half_ru
__ushort2half_rz
__ushort_as_half
"""
# The dictionary is to place description of each device function. Expand it to all the device functions
deviceFuncDesc = {'acosf': "This function returns floating point of arc cosine from a floating point input"}

fnames = ["../../include/hip/amd_detail/math_functions.h","../../include/hip/amd_detail/device_functions.h","../../include/hip/amd_detail/hip_fp16.h"]
markdownFileName = "./hip-math-api.md"

preamble = "# HIP MATH APIs Documentation \n"+\
"HIP supports most of the device functions supported by CUDA. Way to find the unsupported one is to search for the function and check its description\n" + \
"Note: This document is not human generated. Any changes to this file will be discarded. Please make changes to Python3 script docs/markdown/device_md_gen.py\n\n" + \
"## For Developers \n" + \
"If you add or fixed a device function, make sure to add a signature of the function and definition later.\n" + \
"For example, if you want to add `__device__ float __dotf(float4, float4)`, which does a dot product on 4 float vector components \n" + \
"The way to add to the header is, \n" + \
"```cpp \n" + \
"__device__ static float __dotf(float4, float4); \n" + \
"/*Way down in the file....*/\n" + \
"__device__ static inline float __dotf(float4 x, float4 y) { \n" + \
" /*implementation*/\n}\n" + \
"```\n\n" + \
"This helps python script to add the device function newly declared into markdown documentation (as it looks at functions with `;` at the end and `__device__` at the beginning)\n\n" + \
"The next step would be to add Description to  `deviceFuncDesc` dictionary in python script.\n" + \
"From the above example, it can be writtern as,\n`deviceFuncDesc['__dotf'] = 'This functions takes 2 4 component float vector and outputs dot product across them'`\n\n"

def generateSnippet(name, description, signature):
    return "### " + name + "\n" + \
    "```cpp \n" + signature + "\n```\n" + \
    "**Description:**  " + description + "\n\n\n"

def getName(line):
    l1 = line.split('(')
    l2 = l1[0].split(' ')
    return l2[-1]

with open(markdownFileName, 'w') as mdfd:
    mdfd.truncate()
    mdfd.write(preamble)
    for fname in fnames:
        with open(fname) as fd:
            lines = fd.readlines()
            for line in lines:
                if line.find('HIP_FAST_MATH') != -1:
                    break;
                if line.find('__device__') != -1 and line.find(';') != -1 and line.find('hip') == -1:
                    name = getName(line)
                    if line.find('//') == -1:
                        if name in deviceFuncDesc:
                            mdfd.write(generateSnippet(name, deviceFuncDesc[name], line))
                        else:
                            mdfd.write(generateSnippet(name, "Supported", line))
                    else:
                        mdfd.write(generateSnippet(name, "**NOT Supported**", line))
            fd.close()
    mdfd.close()
