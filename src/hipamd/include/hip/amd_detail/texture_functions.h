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

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_TEXTURE_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_TEXTURE_FUNCTIONS_H
#include <hip/amd_detail/hip_vector_types.h>
#include <hip/amd_detail/hip_texture_types.h>

#pragma push_macro("TYPEDEF_VECTOR_VALUE_TYPE")
#define TYPEDEF_VECTOR_VALUE_TYPE(SCALAR_TYPE) \
typedef SCALAR_TYPE __hip_##SCALAR_TYPE##2_vector_value_type __attribute__((ext_vector_type(2))); \
typedef SCALAR_TYPE __hip_##SCALAR_TYPE##3_vector_value_type __attribute__((ext_vector_type(3))); \
typedef SCALAR_TYPE __hip_##SCALAR_TYPE##4_vector_value_type __attribute__((ext_vector_type(4))); \
typedef SCALAR_TYPE __hip_##SCALAR_TYPE##8_vector_value_type __attribute__((ext_vector_type(8))); \
typedef SCALAR_TYPE __hip_##SCALAR_TYPE##16_vector_value_type __attribute__((ext_vector_type(16)));

TYPEDEF_VECTOR_VALUE_TYPE(float);
TYPEDEF_VECTOR_VALUE_TYPE(int);
TYPEDEF_VECTOR_VALUE_TYPE(uint);

#undef TYPEDEF_VECTOR_VALUE_TYPE
#pragma pop_macro("TYPEDEF_VECTOR_VALUE_TYPE")

union TData {
    __hip_float4_vector_value_type f;
    __hip_int4_vector_value_type i;
    __hip_uint4_vector_value_type u;
};

#define __TEXTURE_FUNCTIONS_DECL__ static inline __device__


#if __clang__
#define ADDRESS_SPACE_CONSTANT __attribute__((address_space(4)))
#else
#define ADDRESS_SPACE_CONSTANT __attribute__((address_space(2)))
#endif

#define TEXTURE_PARAMETERS_INIT                                                                    \
    unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)textureObject;  \
    unsigned int ADDRESS_SPACE_CONSTANT* s = i + HIP_SAMPLER_OBJECT_OFFSET_DWORD;                  \
    TData texel;
#define TEXTURE_REF_PARAMETERS_INIT                                                                      \
    unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)texRef.textureObject; \
    unsigned int ADDRESS_SPACE_CONSTANT* s = i + HIP_SAMPLER_OBJECT_OFFSET_DWORD;                        \
    TData texel;
#define TEXTURE_SET_FLOAT *retVal = texel.f.x;

#define TEXTURE_SET_SIGNED *retVal = texel.i.x;

#define TEXTURE_SET_UNSIGNED *retVal = texel.u.x;

#define TEXTURE_SET_FLOAT_X retVal->x = texel.f.x;

#define TEXTURE_SET_SIGNED_X retVal->x = texel.i.x;

#define TEXTURE_SET_UNSIGNED_X retVal->x = texel.u.x;

#define TEXTURE_SET_FLOAT_XY                                                                       \
    retVal->x = texel.f.x;                                                                         \
    retVal->y = texel.f.y;

#define TEXTURE_SET_SIGNED_XY                                                                      \
    retVal->x = texel.i.x;                                                                         \
    retVal->y = texel.i.y;

#define TEXTURE_SET_UNSIGNED_XY                                                                    \
    retVal->x = texel.u.x;                                                                         \
    retVal->y = texel.u.y;

#define TEXTURE_SET_FLOAT_XYZW                                                                     \
    retVal->x = texel.f.x;                                                                         \
    retVal->y = texel.f.y;                                                                         \
    retVal->z = texel.f.z;                                                                         \
    retVal->w = texel.f.w;

#define TEXTURE_SET_SIGNED_XYZW                                                                    \
    retVal->x = texel.i.x;                                                                         \
    retVal->y = texel.i.y;                                                                         \
    retVal->z = texel.i.z;                                                                         \
    retVal->w = texel.i.w;

#define TEXTURE_SET_UNSIGNED_XYZW                                                                  \
    retVal->x = texel.u.x;                                                                         \
    retVal->y = texel.u.y;                                                                         \
    retVal->z = texel.u.z;                                                                         \
    retVal->w = texel.u.w;

#define TEXTURE_RETURN_CHAR return texel.i.x;

#define TEXTURE_RETURN_UCHAR return texel.u.x;

#define TEXTURE_RETURN_SHORT return texel.i.x;

#define TEXTURE_RETURN_USHORT return texel.u.x;

#define TEXTURE_RETURN_INT return texel.i.x;

#define TEXTURE_RETURN_UINT return texel.u.x;

#define TEXTURE_RETURN_SIGNED return texel.i.x;

#define TEXTURE_RETURN_UNSIGNED return texel.u.x;

#define TEXTURE_RETURN_CHAR_X return make_char1(texel.i.x);

#define TEXTURE_RETURN_UCHAR_X return make_uchar1(texel.u.x);

#define TEXTURE_RETURN_SHORT_X return make_short1(texel.i.x);

#define TEXTURE_RETURN_USHORT_X return make_ushort1(texel.u.x);

#define TEXTURE_RETURN_INT_X return make_int1(texel.i.x);

#define TEXTURE_RETURN_UINT_X return make_uint1(texel.u.x);

#define TEXTURE_RETURN_CHAR_XY return make_char2(texel.i.x, texel.i.y);

#define TEXTURE_RETURN_UCHAR_XY return make_uchar2(texel.u.x, texel.u.y);

#define TEXTURE_RETURN_SHORT_XY return make_short2(texel.i.x, texel.i.y);

#define TEXTURE_RETURN_USHORT_XY return make_ushort2(texel.u.x, texel.u.y);

#define TEXTURE_RETURN_INT_XY return make_int2(texel.i.x, texel.i.y);

#define TEXTURE_RETURN_UINT_XY return make_uint2(texel.u.x, texel.u.y);

#define TEXTURE_RETURN_CHAR_XYZW return make_char4(texel.i.x, texel.i.y, texel.i.z, texel.i.w);

#define TEXTURE_RETURN_UCHAR_XYZW return make_uchar4(texel.u.x, texel.u.y, texel.u.z, texel.u.w);

#define TEXTURE_RETURN_SHORT_XYZW return make_short4(texel.i.x, texel.i.y, texel.i.z, texel.i.w);

#define TEXTURE_RETURN_USHORT_XYZW return make_ushort4(texel.u.x, texel.u.y, texel.u.z, texel.u.w);

#define TEXTURE_RETURN_INT_XYZW return make_int4(texel.i.x, texel.i.y, texel.i.z, texel.i.w);

#define TEXTURE_RETURN_UINT_XYZW return make_uint4(texel.u.x, texel.u.y, texel.u.z, texel.u.w);

#define TEXTURE_RETURN_FLOAT return texel.f.x;

#define TEXTURE_RETURN_FLOAT_X return make_float1(texel.f.x);

#define TEXTURE_RETURN_FLOAT_XY return make_float2(texel.f.x, texel.f.y);

#define TEXTURE_RETURN_FLOAT_XYZW return make_float4(texel.f.x, texel.f.y, texel.f.z, texel.f.w);

extern "C" {

__device__
__hip_float4_vector_value_type __ockl_image_sample_1D(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    float c);

__device__
__hip_float4_vector_value_type __ockl_image_sample_1Da(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float2_vector_value_type c);

__device__
__hip_float4_vector_value_type __ockl_image_sample_2D(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float2_vector_value_type c);


__device__
__hip_float4_vector_value_type __ockl_image_sample_2Da(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float4_vector_value_type c);

__device__
float __ockl_image_sample_2Dad(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float4_vector_value_type c);

__device__
float __ockl_image_sample_2Dd(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float2_vector_value_type c);

__device__
__hip_float4_vector_value_type __ockl_image_sample_3D(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float4_vector_value_type c);

__device__
__hip_float4_vector_value_type __ockl_image_sample_grad_1D(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    float c, float dx, float dy);

__device__
__hip_float4_vector_value_type __ockl_image_sample_grad_1Da(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float2_vector_value_type c, float dx, float dy);

__device__
__hip_float4_vector_value_type __ockl_image_sample_grad_2D(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float2_vector_value_type c, __hip_float2_vector_value_type dx, __hip_float2_vector_value_type dy);

__device__
__hip_float4_vector_value_type __ockl_image_sample_grad_2Da(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float4_vector_value_type c, __hip_float2_vector_value_type dx, __hip_float2_vector_value_type dy);

__device__
float __ockl_image_sample_grad_2Dad(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float4_vector_value_type c, __hip_float2_vector_value_type dx, __hip_float2_vector_value_type dy);

__device__
float __ockl_image_sample_grad_2Dd(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float2_vector_value_type c, __hip_float2_vector_value_type dx, __hip_float2_vector_value_type dy);

__device__
__hip_float4_vector_value_type __ockl_image_sample_grad_3D(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float4_vector_value_type c, __hip_float4_vector_value_type dx, __hip_float4_vector_value_type dy);

__device__
__hip_float4_vector_value_type __ockl_image_sample_lod_1D(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    float c, float l);

__device__
__hip_float4_vector_value_type __ockl_image_sample_lod_1Da(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float2_vector_value_type c, float l);

__device__
__hip_float4_vector_value_type __ockl_image_sample_lod_2D(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float2_vector_value_type c, float l);

__device__
__hip_float4_vector_value_type __ockl_image_sample_lod_2Da(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float4_vector_value_type c, float l);

__device__
float __ockl_image_sample_lod_2Dad(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float4_vector_value_type c, float l);

__device__
float __ockl_image_sample_lod_2Dd(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float2_vector_value_type c, float l);

__device__
__hip_float4_vector_value_type __ockl_image_sample_lod_3D(
    unsigned int ADDRESS_SPACE_CONSTANT* i, unsigned int ADDRESS_SPACE_CONSTANT* s,
    __hip_float4_vector_value_type c, float l);
}

////////////////////////////////////////////////////////////
// Texture object APIs
////////////////////////////////////////////////////////////

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(char* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(char1* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(char2* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(char4* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(unsigned char* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(uchar1* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(uchar2* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(uchar4* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(short* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(short1* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(short2* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(short4* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(unsigned short* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(ushort1* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(ushort2* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(ushort4* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(int* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(int1* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(int2* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(int4* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(unsigned int* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(uint1* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(uint2* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(uint4* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(float* retVal, hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(float1* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(float2* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1Dfetch(float4* retVal, hipTextureObject_t textureObject,
                                           int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex1Dfetch(hipTextureObject_t textureObject, int x) {
    T ret;
    tex1Dfetch(&ret, textureObject, x);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex1D(char* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(char1* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(char2* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(char4* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(unsigned char* retVal, hipTextureObject_t textureObject,
                                      float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(uchar1* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(uchar2* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(uchar4* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(short* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(short1* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(short2* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(short4* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(unsigned short* retVal, hipTextureObject_t textureObject,
                                      float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(ushort1* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(ushort2* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(ushort4* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(int* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(int1* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(int2* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(int4* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(unsigned int* retVal, hipTextureObject_t textureObject,
                                      float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(uint1* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(uint2* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(uint4* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(float* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(float1* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(float2* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1D(float4* retVal, hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_SET_FLOAT_XYZW;
}
template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex1D(hipTextureObject_t textureObject, float x) {
    T ret;
    tex1D(&ret, textureObject, x);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(char* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(char1* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(char2* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(char4* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(unsigned char* retVal, hipTextureObject_t textureObject,
                                         float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(uchar1* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(uchar2* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(uchar4* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(short* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(short1* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(short2* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(short4* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(unsigned short* retVal, hipTextureObject_t textureObject,
                                         float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(ushort1* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(ushort2* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(ushort4* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(int* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(int1* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(int2* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(int4* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(unsigned int* retVal, hipTextureObject_t textureObject,
                                         float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(uint1* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(uint2* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(uint4* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(float* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(float1* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(float2* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLod(float4* retVal, hipTextureObject_t textureObject, float x,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex1DLod(hipTextureObject_t textureObject, float x, float level) {
    T ret;
    tex1DLod(&ret, textureObject, x, level);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(char* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(char1* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(char2* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(char4* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(unsigned char* retVal, hipTextureObject_t textureObject,
                                          float x, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(uchar1* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(uchar2* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(uchar4* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(short* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(short1* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(short2* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(short4* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(unsigned short* retVal, hipTextureObject_t textureObject,
                                          float x, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(ushort1* retVal, hipTextureObject_t textureObject,
                                          float x, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(ushort2* retVal, hipTextureObject_t textureObject,
                                          float x, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(ushort4* retVal, hipTextureObject_t textureObject,
                                          float x, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(int* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(int1* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(int2* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(int4* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(unsigned int* retVal, hipTextureObject_t textureObject,
                                          float x, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(uint1* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(uint2* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(uint4* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(float* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(float1* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(float2* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DGrad(float4* retVal, hipTextureObject_t textureObject, float x,
                                          float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex1DGrad(hipTextureObject_t textureObject, float x, float dx,
                                       float dy) {
    T ret;
    tex1DLod(&ret, textureObject, x, dx, dy);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex2D(char* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(char1* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(char2* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(char4* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(unsigned char* retVal, hipTextureObject_t textureObject,
                                      float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(uchar1* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(uchar2* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(uchar4* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(short* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(short1* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(short2* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(short4* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(unsigned short* retVal, hipTextureObject_t textureObject,
                                      float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(ushort1* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(ushort2* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(ushort4* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(int* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(int1* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(int2* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(int4* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(unsigned int* retVal, hipTextureObject_t textureObject,
                                      float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(uint1* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(uint2* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(uint4* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(float* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(float1* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(float2* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2D(float4* retVal, hipTextureObject_t textureObject, float x,
                                      float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex2D(hipTextureObject_t textureObject, float x, float y) {
    T ret;
    tex2D(&ret, textureObject, x, y);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(char* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(char1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(char2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(char4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(unsigned char* retVal, hipTextureObject_t textureObject,
                                         float x, float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(uchar1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(uchar2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(uchar4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(short* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(short1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(short2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(short4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(unsigned short* retVal, hipTextureObject_t textureObject,
                                         float x, float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(ushort1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(ushort2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(ushort4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(int* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(int1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(int2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(int4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(unsigned int* retVal, hipTextureObject_t textureObject,
                                         float x, float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(uint1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(uint2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(uint4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(float* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(float1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(float2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLod(float4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex2DLod(hipTextureObject_t textureObject, float x, float y,
                                      float level) {
    T ret;
    tex2DLod(&ret, textureObject, x, y, level);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex3D(char* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(char1* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(char2* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(char4* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(unsigned char* retVal, hipTextureObject_t textureObject,
                                      float x, float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(uchar1* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(uchar2* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(uchar4* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(short* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(short1* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(short2* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(short4* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(unsigned short* retVal, hipTextureObject_t textureObject,
                                      float x, float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(ushort1* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(ushort2* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(ushort4* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(int* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(int1* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(int2* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(int4* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(unsigned int* retVal, hipTextureObject_t textureObject,
                                      float x, float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(uint1* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(uint2* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(uint4* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(float* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(float1* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(float2* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3D(float4* retVal, hipTextureObject_t textureObject, float x,
                                      float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex3D(hipTextureObject_t textureObject, float x, float y, float z) {
    T ret;
    tex3D(&ret, textureObject, x, y, z);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(char* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(char1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(char2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(char4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(unsigned char* retVal, hipTextureObject_t textureObject,
                                         float x, float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(uchar1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(uchar2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(uchar4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(short* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(short1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(short2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(short4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(unsigned short* retVal, hipTextureObject_t textureObject,
                                         float x, float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(ushort1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(ushort2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(ushort4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(int* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(int1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(int2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(int4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(unsigned int* retVal, hipTextureObject_t textureObject,
                                         float x, float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(uint1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(uint2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(uint4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(float* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(float1* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(float2* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex3DLod(float4* retVal, hipTextureObject_t textureObject, float x,
                                         float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex3DLod(hipTextureObject_t textureObject, float x, float y, float z,
                                      float level) {
    T ret;
    tex3DLod(&ret, textureObject, x, y, z, level);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(char* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(char1* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(char2* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED_XY;
}
__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(char4* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(unsigned char* retVal,
                                             hipTextureObject_t textureObject, float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(uchar1* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(uchar2* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED_XY;
}
__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(uchar4* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(short* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(short1* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(short2* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED_XY;
}
__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(short4* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(unsigned short* retVal,
                                             hipTextureObject_t textureObject, float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(ushort1* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(ushort2* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED_XY;
}
__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(ushort4* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(int* retVal, hipTextureObject_t textureObject, float x,
                                             int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(int1* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(int2* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED_XY;
}
__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(int4* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(unsigned int* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(uint1* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(uint2* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED_XY;
}
__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(uint4* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(float* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(float1* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(float2* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_FLOAT_XY;
}
__TEXTURE_FUNCTIONS_DECL__ void tex1DLayered(float4* retVal, hipTextureObject_t textureObject,
                                             float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex1DLayered(hipTextureObject_t textureObject, float x, int layer) {
    T ret;
    tex1DLayered(&ret, textureObject, x, layer);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(char* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(char1* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(char2* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(char4* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(unsigned char* retVal,
                                                hipTextureObject_t textureObject, float x,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(uchar1* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(uchar2* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(uchar4* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(short* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(short1* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(short2* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(short4* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(unsigned short* retVal,
                                                hipTextureObject_t textureObject, float x,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(ushort1* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(ushort2* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(ushort4* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(int* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(int1* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(int2* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(int4* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(unsigned int* retVal,
                                                hipTextureObject_t textureObject, float x,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(uint1* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(uint2* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(uint4* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(float* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(float1* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(float2* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredLod(float4* retVal, hipTextureObject_t textureObject,
                                                float x, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex1DLayeredLod(hipTextureObject_t textureObject, float x, int layer,
                                             float level) {
    T ret;
    tex1DLayeredLod(&ret, textureObject, x, layer, level);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(char* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(char1* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(char2* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(char4* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(unsigned char* retVal,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(uchar1* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(uchar2* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(uchar4* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(short* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(short1* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(short2* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(short4* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(unsigned short* retVal,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(ushort1* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(ushort2* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(ushort4* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(int* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(int1* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(int2* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(int4* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(unsigned int* retVal,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(uint1* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(uint2* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(uint4* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(float* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(float1* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(float2* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex1DLayeredGrad(float4* retVal, hipTextureObject_t textureObject,
                                                 float x, int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex1DLayeredGrad(hipTextureObject_t textureObject, float x, int layer,
                                              float dx, float dy) {
    T ret;
    tex1DLayeredGrad(&ret, textureObject, x, layer, dx, dy);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(char* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(char1* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(char2* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(char4* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(unsigned char* retVal,
                                             hipTextureObject_t textureObject, float x, float y,
                                             int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(uchar1* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(uchar2* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(uchar4* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(short* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(short1* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(short2* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(short4* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(unsigned short* retVal,
                                             hipTextureObject_t textureObject, float x, float y,
                                             int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(ushort1* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(ushort2* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(ushort4* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(int* retVal, hipTextureObject_t textureObject, float x,
                                             float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(int1* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(int2* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(int4* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(unsigned int* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(uint1* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(uint2* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(uint4* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(float* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(float1* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(float2* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayered(float4* retVal, hipTextureObject_t textureObject,
                                             float x, float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex2DLayered(hipTextureObject_t textureObject, float x, float y,
                                          int layer) {
    T ret;
    tex2DLayered(&ret, textureObject, x, y, layer);
    return ret;
}

////////////////////////////////////////////////////////////
__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(char* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(char1* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(char2* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(char4* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(unsigned char* retVal,
                                                hipTextureObject_t textureObject, float x, float y,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(uchar1* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(uchar2* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(uchar4* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(short* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(short1* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(short2* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(short4* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(unsigned short* retVal,
                                                hipTextureObject_t textureObject, float x, float y,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(ushort1* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(ushort2* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(ushort4* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(int* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(int1* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(int2* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(int4* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_SIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(unsigned int* retVal,
                                                hipTextureObject_t textureObject, float x, float y,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(uint1* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(uint2* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(uint4* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_UNSIGNED_XYZW;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(float* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_FLOAT;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(float1* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_FLOAT_X;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(float2* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_FLOAT_XY;
}

__TEXTURE_FUNCTIONS_DECL__ void tex2DLayeredLod(float4* retVal, hipTextureObject_t textureObject,
                                                float x, float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_SET_FLOAT_XYZW;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex2DLayeredLod(hipTextureObject_t textureObject, float x, float y,
                                             int layer, float level) {
    T ret;
    tex2DLayeredLod(&ret, textureObject, x, y, layer, level);
    return ret;
}

////////////////////////////////////////////////////////////
// Texture Reference APIs
////////////////////////////////////////////////////////////
template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1Dfetch(texture<char, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1Dfetch(texture<char1, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1Dfetch(texture<char2, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1Dfetch(texture<char4, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1Dfetch(texture<unsigned char, texType, mode> texRef,
                                                    int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1Dfetch(texture<uchar1, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1Dfetch(texture<uchar2, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1Dfetch(texture<uchar4, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1Dfetch(texture<short, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1Dfetch(texture<short1, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1Dfetch(texture<short2, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1Dfetch(texture<short4, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1Dfetch(texture<ushort1, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1Dfetch(texture<unsigned short, texType, mode> texRef,
                                                     int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1Dfetch(texture<ushort2, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1Dfetch(texture<ushort4, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1Dfetch(texture<int1, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1Dfetch(texture<int, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1Dfetch(texture<int2, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1Dfetch(texture<int4, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1Dfetch(texture<unsigned int, texType, mode> texRef,
                                                   int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1Dfetch(texture<uint1, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1Dfetch(texture<uint2, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1Dfetch(texture<uint4, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1Dfetch(texture<float, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1Dfetch(texture<float1, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1Dfetch(texture<float2, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1Dfetch(texture<float4, texType, mode> texRef, int x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1Dfetch(texture<char, texType, mode> texRef,
                                           hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1Dfetch(texture<char1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1Dfetch(texture<char2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1Dfetch(texture<char4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1Dfetch(texture<unsigned char, texType, mode> texRef,
                                                    hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1Dfetch(texture<uchar1, texType, mode> texRef,
                                             hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1Dfetch(texture<uchar2, texType, mode> texRef,
                                             hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1Dfetch(texture<uchar4, texType, mode> texRef,
                                             hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1Dfetch(texture<short, texType, mode> texRef,
                                            hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1Dfetch(texture<short1, texType, mode> texRef,
                                             hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1Dfetch(texture<short2, texType, mode> texRef,
                                             hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1Dfetch(texture<short4, texType, mode> texRef,
                                             hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1Dfetch(texture<ushort1, texType, mode> texRef,
                                              hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1Dfetch(texture<unsigned short, texType, mode> texRef,
                                                     hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1Dfetch(texture<ushort2, texType, mode> texRef,
                                              hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1Dfetch(texture<ushort4, texType, mode> texRef,
                                              hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1Dfetch(texture<int1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1Dfetch(texture<int, texType, mode> texRef,
                                          hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1Dfetch(texture<int2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1Dfetch(texture<int4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1Dfetch(texture<unsigned int, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1Dfetch(texture<uint1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1Dfetch(texture<uint2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1Dfetch(texture<uint4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1Dfetch(texture<float, texType, mode> texRef,
                                            hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1Dfetch(texture<float1, texType, mode> texRef,
                                             hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1Dfetch(texture<float2, texType, mode> texRef,
                                             hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1Dfetch(texture<float4, texType, mode> texRef,
                                             hipTextureObject_t textureObject, int x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////
template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1D(texture<char, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1D(texture<char1, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1D(texture<char2, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1D(texture<char4, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1D(texture<unsigned char, texType, mode> texRef,
                                               float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1D(texture<uchar1, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1D(texture<uchar2, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1D(texture<uchar4, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1D(texture<short, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1D(texture<short1, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1D(texture<short2, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1D(texture<short4, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1D(texture<unsigned short, texType, mode> texRef,
                                                float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1D(texture<ushort1, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1D(texture<ushort2, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1D(texture<ushort4, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1D(texture<int, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1D(texture<int1, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1D(texture<int2, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1D(texture<int4, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1D(texture<unsigned int, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1D(texture<uint1, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1D(texture<uint2, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1D(texture<uint4, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1D(texture<float1, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1D(texture<float2, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1D(texture<float4, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////
template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1D(texture<char, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1D(texture<char1, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1D(texture<char2, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1D(texture<char4, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1D(texture<unsigned char, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1D(texture<uchar1, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1D(texture<uchar2, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1D(texture<uchar4, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1D(texture<short, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1D(texture<short1, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1D(texture<short2, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1D(texture<short4, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1D(texture<unsigned short, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1D(texture<ushort1, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1D(texture<ushort2, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1D(texture<ushort4, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1D(texture<int, texType, mode> texRef,
                                     hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1D(texture<int1, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1D(texture<int2, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1D(texture<int4, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1D(texture<unsigned int, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1D(texture<uint1, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1D(texture<uint2, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1D(texture<uint4, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1D(texture<float, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT;
}
//////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1D(texture<float, texType, mode> texRef, float x) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1D(texture<float1, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1D(texture<float2, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1D(texture<float4, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1D(i, s, x);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1DLod(texture<char, texType, mode> texRef, float x,
                                         float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLod(texture<char1, texType, mode> texRef, float x,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_CHAR_X;
}
template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLod(texture<char2, texType, mode> texRef, float x,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLod(texture<char4, texType, mode> texRef, float x,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLod(texture<unsigned char, texType, mode> texRef,
                                                  float x, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLod(texture<uchar1, texType, mode> texRef, float x,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLod(texture<uchar2, texType, mode> texRef, float x,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLod(texture<uchar4, texType, mode> texRef, float x,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1DLod(texture<short, texType, mode> texRef, float x,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLod(texture<short1, texType, mode> texRef, float x,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLod(texture<short2, texType, mode> texRef, float x,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLod(texture<short4, texType, mode> texRef, float x,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLod(texture<unsigned short, texType, mode> texRef,
                                                   float x, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLod(texture<ushort1, texType, mode> texRef, float x,
                                            float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLod(texture<ushort2, texType, mode> texRef, float x,
                                            float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLod(texture<ushort4, texType, mode> texRef, float x,
                                            float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1DLod(texture<int, texType, mode> texRef, float x, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLod(texture<int1, texType, mode> texRef, float x,
                                         float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLod(texture<int2, texType, mode> texRef, float x,
                                         float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLod(texture<int4, texType, mode> texRef, float x,
                                         float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLod(texture<unsigned int, texType, mode> texRef,
                                                 float x, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLod(texture<uint1, texType, mode> texRef, float x,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLod(texture<uint2, texType, mode> texRef, float x,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLod(texture<uint4, texType, mode> texRef, float x,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1DLod(texture<float, texType, mode> texRef, float x,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLod(texture<float1, texType, mode> texRef, float x,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLod(texture<float2, texType, mode> texRef, float x,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLod(texture<float4, texType, mode> texRef, float x,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1DLod(texture<char, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLod(texture<char1, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_CHAR_X;
}
template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLod(texture<char2, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLod(texture<char4, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLod(texture<unsigned char, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLod(texture<uchar1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLod(texture<uchar2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLod(texture<uchar4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1DLod(texture<short, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLod(texture<short1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLod(texture<short2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLod(texture<short4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLod(texture<unsigned short, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLod(texture<ushort1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x,
                                            float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLod(texture<ushort2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x,
                                            float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLod(texture<ushort4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x,
                                            float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1DLod(texture<int, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLod(texture<int1, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLod(texture<int2, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLod(texture<int4, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLod(texture<unsigned int, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLod(texture<uint1, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLod(texture<uint2, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLod(texture<uint4, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1DLod(texture<float, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLod(texture<float1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLod(texture<float2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLod(texture<float4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_1D(i, s, x, level);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1DGrad(texture<char, texType, mode> texRef, float x, float dx,
                                          float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1DGrad(texture<char1, texType, mode> texRef, float x, float dx,
                                           float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1DGrad(texture<char2, texType, mode> texRef, float x, float dx,
                                           float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1DGrad(texture<char4, texType, mode> texRef, float x, float dx,
                                           float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DGrad(texture<unsigned char, texType, mode> texRef,
                                                   float x, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DGrad(texture<uchar1, texType, mode> texRef, float x,
                                            float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DGrad(texture<uchar2, texType, mode> texRef, float x,
                                            float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DGrad(texture<uchar4, texType, mode> texRef, float x,
                                            float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1DGrad(texture<short, texType, mode> texRef, float x, float dx,
                                           float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1DGrad(texture<short1, texType, mode> texRef, float x,
                                            float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1DGrad(texture<short2, texType, mode> texRef, float x,
                                            float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1DGrad(texture<short4, texType, mode> texRef, float x,
                                            float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DGrad(texture<unsigned short, texType, mode> texRef,
                                                    float x, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DGrad(texture<ushort1, texType, mode> texRef, float x,
                                             float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DGrad(texture<ushort2, texType, mode> texRef, float x,
                                             float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DGrad(texture<ushort4, texType, mode> texRef, float x,
                                             float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1DGrad(texture<int, texType, mode> texRef, float x, float dx,
                                         float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1DGrad(texture<int1, texType, mode> texRef, float x, float dx,
                                          float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1DGrad(texture<int2, texType, mode> texRef, float x, float dx,
                                          float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1DGrad(texture<int4, texType, mode> texRef, float x, float dx,
                                          float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DGrad(texture<unsigned int, texType, mode> texRef,
                                                  float x, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DGrad(texture<uint1, texType, mode> texRef, float x, float dx,
                                           float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DGrad(texture<uint2, texType, mode> texRef, float x, float dx,
                                           float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DGrad(texture<uint4, texType, mode> texRef, float x, float dx,
                                           float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1DGrad(texture<float, texType, mode> texRef, float x, float dx,
                                           float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1DGrad(texture<float1, texType, mode> texRef, float x,
                                            float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1DGrad(texture<float2, texType, mode> texRef, float x,
                                            float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1DGrad(texture<float4, texType, mode> texRef, float x,
                                            float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1DGrad(texture<char, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float dx,
                                          float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1DGrad(texture<char1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float dx,
                                           float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1DGrad(texture<char2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float dx,
                                           float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1DGrad(texture<char4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float dx,
                                           float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DGrad(texture<unsigned char, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DGrad(texture<uchar1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float dx,
                                            float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DGrad(texture<uchar2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float dx,
                                            float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DGrad(texture<uchar4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float dx,
                                            float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1DGrad(texture<short, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float dx,
                                           float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1DGrad(texture<short1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float dx,
                                            float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1DGrad(texture<short2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float dx,
                                            float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1DGrad(texture<short4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float dx,
                                            float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DGrad(texture<unsigned short, texType, mode> texRef,
                                                    hipTextureObject_t textureObject, float x,
                                                    float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DGrad(texture<ushort1, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float dx,
                                             float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DGrad(texture<ushort2, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float dx,
                                             float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DGrad(texture<ushort4, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float dx,
                                             float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1DGrad(texture<int, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float dx,
                                         float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1DGrad(texture<int1, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float dx,
                                          float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1DGrad(texture<int2, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float dx,
                                          float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1DGrad(texture<int4, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float dx,
                                          float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DGrad(texture<unsigned int, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DGrad(texture<uint1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float dx,
                                           float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DGrad(texture<uint2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float dx,
                                           float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DGrad(texture<uint4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float dx,
                                           float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1DGrad(texture<float, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float dx,
                                           float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1DGrad(texture<float1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float dx,
                                            float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1DGrad(texture<float2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float dx,
                                            float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1DGrad(texture<float4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float dx,
                                            float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_1D(i, s, x, dx, dy);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2D(texture<char, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2D(texture<char1, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2D(texture<char2, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2D(texture<char4, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2D(texture<unsigned char, texType, mode> texRef,
                                               float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2D(texture<uchar1, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2D(texture<uchar2, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2D(texture<uchar4, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2D(texture<short, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2D(texture<short1, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2D(texture<short2, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2D(texture<short4, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2D(texture<unsigned short, texType, mode> texRef,
                                                float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2D(texture<ushort1, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2D(texture<ushort2, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2D(texture<ushort4, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2D(texture<int, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2D(texture<int1, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2D(texture<int2, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2D(texture<int4, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2D(texture<unsigned int, texType, mode> texRef, float x,
                                              float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2D(texture<uint1, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2D(texture<uint2, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2D(texture<uint4, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UINT_XYZW;
}


////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2D(texture<char, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2D(texture<char1, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2D(texture<char2, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2D(texture<char4, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2D(texture<unsigned char, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2D(texture<uchar1, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2D(texture<uchar2, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2D(texture<uchar4, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2D(texture<short, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2D(texture<short1, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2D(texture<short2, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2D(texture<short4, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2D(texture<unsigned short, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x,
                                                float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2D(texture<ushort1, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2D(texture<ushort2, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2D(texture<ushort4, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2D(texture<int, texType, mode> texRef,
                                     hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2D(texture<int1, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2D(texture<int2, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2D(texture<int4, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2D(texture<unsigned int, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2D(texture<uint1, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2D(texture<uint2, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2D(texture<uint4, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2D(texture<float, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2D(texture<float, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2D(texture<float1, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2D(texture<float1, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2D(texture<float2, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2D(texture<float2, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2D(texture<float4, texType, mode> texRef, float x, float y) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2D(texture<float4, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_2D(i, s, float2(x, y).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2DLod(texture<char, texType, mode> texRef, float x, float y,
                                         float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLod(texture<char1, texType, mode> texRef, float x, float y,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLod(texture<char2, texType, mode> texRef, float x, float y,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLod(texture<char4, texType, mode> texRef, float x, float y,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLod(texture<unsigned char, texType, mode> texRef,
                                                  float x, float y, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLod(texture<uchar1, texType, mode> texRef, float x, float y,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLod(texture<uchar2, texType, mode> texRef, float x, float y,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLod(texture<uchar4, texType, mode> texRef, float x, float y,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2DLod(texture<short, texType, mode> texRef, float x, float y,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLod(texture<short1, texType, mode> texRef, float x, float y,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLod(texture<short2, texType, mode> texRef, float x, float y,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLod(texture<short4, texType, mode> texRef, float x, float y,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLod(texture<unsigned short, texType, mode> texRef,
                                                   float x, float y, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLod(texture<ushort1, texType, mode> texRef, float x,
                                            float y, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLod(texture<ushort2, texType, mode> texRef, float x,
                                            float y, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLod(texture<ushort4, texType, mode> texRef, float x,
                                            float y, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2DLod(texture<int, texType, mode> texRef, float x, float y,
                                        float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLod(texture<int1, texType, mode> texRef, float x, float y,
                                         float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLod(texture<int2, texType, mode> texRef, float x, float y,
                                         float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLod(texture<int4, texType, mode> texRef, float x, float y,
                                         float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLod(texture<unsigned int, texType, mode> texRef,
                                                 float x, float y, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLod(texture<uint1, texType, mode> texRef, float x, float y,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLod(texture<uint2, texType, mode> texRef, float x, float y,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLod(texture<uint4, texType, mode> texRef, float x, float y,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2DLod(texture<float, texType, mode> texRef, float x, float y,
                                          float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLod(texture<float1, texType, mode> texRef, float x, float y,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLod(texture<float2, texType, mode> texRef, float x, float y,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLod(texture<float4, texType, mode> texRef, float x, float y,
                                           float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2DLod(texture<char, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLod(texture<char1, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLod(texture<char2, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLod(texture<char4, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLod(texture<unsigned char, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLod(texture<uchar1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLod(texture<uchar2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLod(texture<uchar4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2DLod(texture<short, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLod(texture<short1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLod(texture<short2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLod(texture<short4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLod(texture<unsigned short, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLod(texture<ushort1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLod(texture<ushort2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLod(texture<ushort4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2DLod(texture<int, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y,
                                        float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLod(texture<int1, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLod(texture<int2, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLod(texture<int4, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLod(texture<unsigned int, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLod(texture<uint1, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLod(texture<uint2, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLod(texture<uint4, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2DLod(texture<float, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLod(texture<float1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLod(texture<float2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLod(texture<float4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2DGrad(texture<char, texType, mode> texRef, float x, float y,
                                          float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2DGrad(texture<char1, texType, mode> texRef, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2DGrad(texture<char2, texType, mode> texRef, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2DGrad(texture<char4, texType, mode> texRef, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DGrad(texture<unsigned char, texType, mode> texRef,
                                                   float x, float y, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DGrad(texture<uchar1, texType, mode> texRef, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DGrad(texture<uchar2, texType, mode> texRef, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DGrad(texture<uchar4, texType, mode> texRef, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2DGrad(texture<short, texType, mode> texRef, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2DGrad(texture<short1, texType, mode> texRef, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2DGrad(texture<short2, texType, mode> texRef, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2DGrad(texture<short4, texType, mode> texRef, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DGrad(texture<unsigned short, texType, mode> texRef,
                                                    float x, float y, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DGrad(texture<ushort1, texType, mode> texRef, float x,
                                             float y, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DGrad(texture<ushort2, texType, mode> texRef, float x,
                                             float y, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DGrad(texture<ushort4, texType, mode> texRef, float x,
                                             float y, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2DGrad(texture<int, texType, mode> texRef, float x, float y,
                                         float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2DGrad(texture<int1, texType, mode> texRef, float x, float y,
                                          float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2DGrad(texture<int2, texType, mode> texRef, float x, float y,
                                          float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2DGrad(texture<int4, texType, mode> texRef, float x, float y,
                                          float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DGrad(texture<unsigned int, texType, mode> texRef,
                                                  float x, float y, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DGrad(texture<uint1, texType, mode> texRef, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DGrad(texture<uint2, texType, mode> texRef, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DGrad(texture<uint4, texType, mode> texRef, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2DGrad(texture<float, texType, mode> texRef, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2DGrad(texture<float1, texType, mode> texRef, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2DGrad(texture<float2, texType, mode> texRef, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2DGrad(texture<float4, texType, mode> texRef, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2DGrad(texture<char, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2DGrad(texture<char1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2DGrad(texture<char2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2DGrad(texture<char4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DGrad(texture<unsigned char, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DGrad(texture<uchar1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DGrad(texture<uchar2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DGrad(texture<uchar4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2DGrad(texture<short, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2DGrad(texture<short1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2DGrad(texture<short2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2DGrad(texture<short4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DGrad(texture<unsigned short, texType, mode> texRef,
                                                    hipTextureObject_t textureObject, float x,
                                                    float y, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DGrad(texture<ushort1, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float y,
                                             float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DGrad(texture<ushort2, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float y,
                                             float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DGrad(texture<ushort4, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float y,
                                             float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2DGrad(texture<int, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2DGrad(texture<int1, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2DGrad(texture<int2, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2DGrad(texture<int4, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DGrad(texture<unsigned int, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DGrad(texture<uint1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DGrad(texture<uint2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DGrad(texture<uint4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2DGrad(texture<float, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2DGrad(texture<float1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2DGrad(texture<float2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2DGrad(texture<float4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_grad_2D(i, s, float2(x, y).data,
                                          float2(dx.x, dx.y).data,
                                          float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex3D(texture<char, texType, mode> texRef, float x, float y,
                                      float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex3D(texture<char1, texType, mode> texRef, float x, float y,
                                       float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex3D(texture<char2, texType, mode> texRef, float x, float y,
                                       float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex3D(texture<char4, texType, mode> texRef, float x, float y,
                                       float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex3D(texture<unsigned char, texType, mode> texRef,
                                               float x, float y, float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex3D(texture<uchar1, texType, mode> texRef, float x, float y,
                                        float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex3D(texture<uchar2, texType, mode> texRef, float x, float y,
                                        float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex3D(texture<uchar4, texType, mode> texRef, float x, float y,
                                        float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex3D(texture<short, texType, mode> texRef, float x, float y,
                                       float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex3D(texture<short1, texType, mode> texRef, float x, float y,
                                        float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex3D(texture<short2, texType, mode> texRef, float x, float y,
                                        float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex3D(texture<short4, texType, mode> texRef, float x, float y,
                                        float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex3D(texture<unsigned short, texType, mode> texRef,
                                                float x, float y, float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex3D(texture<ushort1, texType, mode> texRef, float x, float y,
                                         float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex3D(texture<ushort2, texType, mode> texRef, float x, float y,
                                         float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex3D(texture<ushort4, texType, mode> texRef, float x, float y,
                                         float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex3D(texture<int, texType, mode> texRef, float x, float y,
                                     float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex3D(texture<int1, texType, mode> texRef, float x, float y,
                                      float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex3D(texture<int2, texType, mode> texRef, float x, float y,
                                      float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex3D(texture<int4, texType, mode> texRef, float x, float y,
                                      float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex3D(texture<unsigned int, texType, mode> texRef, float x,
                                              float y, float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex3D(texture<uint1, texType, mode> texRef, float x, float y,
                                       float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex3D(texture<uint2, texType, mode> texRef, float x, float y,
                                       float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex3D(texture<uint4, texType, mode> texRef, float x, float y,
                                       float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex3D(texture<float, texType, mode> texRef, float x, float y,
                                       float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex3D(texture<float1, texType, mode> texRef, float x, float y,
                                        float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex3D(texture<float2, texType, mode> texRef, float x, float y,
                                        float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex3D(texture<float4, texType, mode> texRef, float x, float y,
                                        float z) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex3D(texture<char, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x, float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex3D(texture<char1, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y,
                                       float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex3D(texture<char2, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y,
                                       float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex3D(texture<char4, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y,
                                       float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex3D(texture<unsigned char, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y,
                                               float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex3D(texture<uchar1, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y,
                                        float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex3D(texture<uchar2, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y,
                                        float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex3D(texture<uchar4, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y,
                                        float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex3D(texture<short, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y,
                                       float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex3D(texture<short1, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y,
                                        float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex3D(texture<short2, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y,
                                        float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex3D(texture<short4, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y,
                                        float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex3D(texture<unsigned short, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x, float y,
                                                float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex3D(texture<ushort1, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex3D(texture<ushort2, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex3D(texture<ushort4, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex3D(texture<int, texType, mode> texRef,
                                     hipTextureObject_t textureObject, float x, float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex3D(texture<int1, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x, float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex3D(texture<int2, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x, float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex3D(texture<int4, texType, mode> texRef,
                                      hipTextureObject_t textureObject, float x, float y, float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex3D(texture<unsigned int, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x, float y,
                                              float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex3D(texture<uint1, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y,
                                       float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex3D(texture<uint2, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y,
                                       float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex3D(texture<uint4, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y,
                                       float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex3D(texture<float, texType, mode> texRef,
                                       hipTextureObject_t textureObject, float x, float y,
                                       float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex3D(texture<float1, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y,
                                        float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex3D(texture<float2, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y,
                                        float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex3D(texture<float4, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y,
                                        float z) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex3DLod(texture<char, texType, mode> texRef, float x, float y,
                                         float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex3DLod(texture<char1, texType, mode> texRef, float x, float y,
                                          float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex3DLod(texture<char2, texType, mode> texRef, float x, float y,
                                          float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex3DLod(texture<char4, texType, mode> texRef, float x, float y,
                                          float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex3DLod(texture<unsigned char, texType, mode> texRef,
                                                  float x, float y, float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex3DLod(texture<uchar1, texType, mode> texRef, float x, float y,
                                           float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex3DLod(texture<uchar2, texType, mode> texRef, float x, float y,
                                           float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex3DLod(texture<uchar4, texType, mode> texRef, float x, float y,
                                           float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex3DLod(texture<int, texType, mode> texRef, float x, float y,
                                        float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex3DLod(texture<int1, texType, mode> texRef, float x, float y,
                                         float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex3DLod(texture<int2, texType, mode> texRef, float x, float y,
                                         float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex3DLod(texture<int4, texType, mode> texRef, float x, float y,
                                         float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex3DLod(texture<unsigned int, texType, mode> texRef,
                                                 float x, float y, float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex3DLod(texture<uint1, texType, mode> texRef, float x, float y,
                                          float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex3DLod(texture<uint2, texType, mode> texRef, float x, float y,
                                          float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex3DLod(texture<uint4, texType, mode> texRef, float x, float y,
                                          float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex3DLod(texture<float, texType, mode> texRef, float x, float y,
                                          float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex3DLod(texture<float1, texType, mode> texRef, float x, float y,
                                           float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex3DLod(texture<float2, texType, mode> texRef, float x, float y,
                                           float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex3DLod(texture<float4, texType, mode> texRef, float x, float y,
                                           float z, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex3DLod(texture<char, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex3DLod(texture<char1, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex3DLod(texture<char2, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex3DLod(texture<char4, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex3DLod(texture<unsigned char, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex3DLod(texture<uchar1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex3DLod(texture<uchar2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex3DLod(texture<uchar4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex3DLod(texture<int, texType, mode> texRef,
                                        hipTextureObject_t textureObject, float x, float y, float z,
                                        float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex3DLod(texture<int1, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex3DLod(texture<int2, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex3DLod(texture<int4, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex3DLod(texture<unsigned int, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex3DLod(texture<uint1, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex3DLod(texture<uint2, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex3DLod(texture<uint4, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex3DLod(texture<float, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex3DLod(texture<float1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex3DLod(texture<float2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex3DLod(texture<float4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data,
                                         level);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex3DGrad(texture<char, texType, mode> texRef, float x, float y,
                                          float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex3DGrad(texture<char1, texType, mode> texRef, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex3DGrad(texture<char2, texType, mode> texRef, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex3DGrad(texture<char4, texType, mode> texRef, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex3DGrad(texture<unsigned char, texType, mode> texRef,
                                                   float x, float y, float z, float4 dx,
                                                   float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex3DGrad(texture<uchar1, texType, mode> texRef, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex3DGrad(texture<uchar2, texType, mode> texRef, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex3DGrad(texture<uchar4, texType, mode> texRef, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex3DGrad(texture<short, texType, mode> texRef, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex3DGrad(texture<short1, texType, mode> texRef, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex3DGrad(texture<short2, texType, mode> texRef, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex3DGrad(texture<short4, texType, mode> texRef, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex3DGrad(texture<unsigned short, texType, mode> texRef,
                                                    float x, float y, float z, float4 dx,
                                                    float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex3DGrad(texture<ushort1, texType, mode> texRef, float x,
                                             float y, float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex3DGrad(texture<ushort2, texType, mode> texRef, float x,
                                             float y, float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex3DGrad(texture<ushort4, texType, mode> texRef, float x,
                                             float y, float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex3DGrad(texture<int, texType, mode> texRef, float x, float y,
                                         float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex3DGrad(texture<int1, texType, mode> texRef, float x, float y,
                                          float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex3DGrad(texture<int2, texType, mode> texRef, float x, float y,
                                          float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex3DGrad(texture<int4, texType, mode> texRef, float x, float y,
                                          float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex3DGrad(texture<unsigned int, texType, mode> texRef,
                                                  float x, float y, float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex3DGrad(texture<uint1, texType, mode> texRef, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex3DGrad(texture<uint2, texType, mode> texRef, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex3DGrad(texture<uint4, texType, mode> texRef, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex3DGrad(texture<float, texType, mode> texRef, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex3DGrad(texture<float1, texType, mode> texRef, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex3DGrad(texture<float2, texType, mode> texRef, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex3DGrad(texture<float4, texType, mode> texRef, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////
template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex3DGrad(texture<char, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex3DGrad(texture<char1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex3DGrad(texture<char2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex3DGrad(texture<char4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex3DGrad(texture<unsigned char, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex3DGrad(texture<uchar1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex3DGrad(texture<uchar2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex3DGrad(texture<uchar4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex3DGrad(texture<short, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex3DGrad(texture<short1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex3DGrad(texture<short2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex3DGrad(texture<short4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex3DGrad(texture<unsigned short, texType, mode> texRef,
                                                    hipTextureObject_t textureObject, float x,
                                                    float y, float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex3DGrad(texture<ushort1, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float y,
                                             float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex3DGrad(texture<ushort2, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float y,
                                             float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex3DGrad(texture<ushort4, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float y,
                                             float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex3DGrad(texture<int, texType, mode> texRef,
                                         hipTextureObject_t textureObject, float x, float y,
                                         float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex3DGrad(texture<int1, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex3DGrad(texture<int2, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex3DGrad(texture<int4, texType, mode> texRef,
                                          hipTextureObject_t textureObject, float x, float y,
                                          float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex3DGrad(texture<unsigned int, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex3DGrad(texture<uint1, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex3DGrad(texture<uint2, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex3DGrad(texture<uint4, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex3DGrad(texture<float, texType, mode> texRef,
                                           hipTextureObject_t textureObject, float x, float y,
                                           float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex3DGrad(texture<float1, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex3DGrad(texture<float2, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex3DGrad(texture<float4, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            float z, float4 dx, float4 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data,
                                    float4(dx.x, dx.y, dx.z, dx.w).data,
                                    float4(dy.x, dy.y, dy.z, dy.w).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1DLayered(texture<char, texType, mode> texRef, float x,
                                             int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLayered(texture<char1, texType, mode> texRef, float x,
                                              int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLayered(texture<char2, texType, mode> texRef, float x,
                                              int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLayered(texture<char4, texType, mode> texRef, float x,
                                              int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLayered(texture<unsigned char, texType, mode> texRef,
                                                      float x, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLayered(texture<uchar1, texType, mode> texRef, float x,
                                               int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLayered(texture<uchar2, texType, mode> texRef, float x,
                                               int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLayered(texture<uchar4, texType, mode> texRef, float x,
                                               int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1DLayered(texture<short, texType, mode> texRef, float x,
                                              int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLayered(texture<short1, texType, mode> texRef, float x,
                                               int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLayered(texture<short2, texType, mode> texRef, float x,
                                               int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLayered(texture<short4, texType, mode> texRef, float x,
                                               int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLayered(
    texture<unsigned short, texType, mode> texRef, float x, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLayered(texture<ushort1, texType, mode> texRef, float x,
                                                int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLayered(texture<ushort2, texType, mode> texRef, float x,
                                                int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLayered(texture<ushort4, texType, mode> texRef, float x,
                                                int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1DLayered(texture<int, texType, mode> texRef, float x,
                                            int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLayered(texture<int1, texType, mode> texRef, float x,
                                             int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLayered(texture<int2, texType, mode> texRef, float x,
                                             int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLayered(texture<int4, texType, mode> texRef, float x,
                                             int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLayered(texture<unsigned int, texType, mode> texRef,
                                                     float x, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLayered(texture<uint1, texType, mode> texRef, float x,
                                              int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLayered(texture<uint2, texType, mode> texRef, float x,
                                              int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLayered(texture<uint4, texType, mode> texRef, float x,
                                              int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1DLayered(texture<float, texType, mode> texRef, float x,
                                              int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayered(texture<float1, texType, mode> texRef, float x,
                                               int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayered(texture<float2, texType, mode> texRef, float x,
                                               int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayered(texture<float4, texType, mode> texRef, float x,
                                               int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1DLayered(texture<char, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLayered(texture<char1, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLayered(texture<char2, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLayered(texture<char4, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLayered(texture<unsigned char, texType, mode> texRef,
                                                      hipTextureObject_t textureObject, float x,
                                                      int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLayered(texture<uchar1, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLayered(texture<uchar2, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLayered(texture<uchar4, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1DLayered(texture<short, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLayered(texture<short1, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLayered(texture<short2, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLayered(texture<short4, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLayered(
    texture<unsigned short, texType, mode> texRef, hipTextureObject_t textureObject, float x,
    int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLayered(texture<ushort1, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x,
                                                int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLayered(texture<ushort2, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x,
                                                int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLayered(texture<ushort4, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x,
                                                int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1DLayered(texture<int, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLayered(texture<int1, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLayered(texture<int2, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLayered(texture<int4, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLayered(texture<unsigned int, texType, mode> texRef,
                                                     hipTextureObject_t textureObject, float x,
                                                     int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLayered(texture<uint1, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLayered(texture<uint2, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLayered(texture<uint4, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1DLayered(texture<float, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayered(texture<float1, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayered(texture<float2, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayered(texture<float4, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1DLayeredLod(texture<char, texType, mode> texRef, float x,
                                                int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLayeredLod(texture<char1, texType, mode> texRef, float x,
                                                 int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLayeredLod(texture<char2, texType, mode> texRef, float x,
                                                 int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLayeredLod(texture<char4, texType, mode> texRef, float x,
                                                 int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLayeredLod(
    texture<unsigned char, texType, mode> texRef, float x, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLayeredLod(texture<uchar1, texType, mode> texRef, float x,
                                                  int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLayeredLod(texture<uchar2, texType, mode> texRef, float x,
                                                  int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLayeredLod(texture<uchar4, texType, mode> texRef, float x,
                                                  int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1DLayeredLod(texture<short, texType, mode> texRef, float x,
                                                 int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLayeredLod(texture<short1, texType, mode> texRef, float x,
                                                  int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLayeredLod(texture<short2, texType, mode> texRef, float x,
                                                  int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLayeredLod(texture<short4, texType, mode> texRef, float x,
                                                  int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLayeredLod(
    texture<unsigned short, texType, mode> texRef, float x, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLayeredLod(texture<ushort1, texType, mode> texRef, float x,
                                                   int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLayeredLod(texture<ushort2, texType, mode> texRef, float x,
                                                   int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLayeredLod(texture<ushort4, texType, mode> texRef, float x,
                                                   int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1DLayeredLod(texture<int, texType, mode> texRef, float x,
                                               int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLayeredLod(texture<int1, texType, mode> texRef, float x,
                                                int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLayeredLod(texture<int2, texType, mode> texRef, float x,
                                                int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLayeredLod(texture<int4, texType, mode> texRef, float x,
                                                int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLayeredLod(texture<unsigned int, texType, mode> texRef,
                                                        float x, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLayeredLod(texture<uint1, texType, mode> texRef, float x,
                                                 int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLayeredLod(texture<uint2, texType, mode> texRef, float x,
                                                 int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLayeredLod(texture<uint4, texType, mode> texRef, float x,
                                                 int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredLod(texture<float, texType, mode> texRef, float x,
                                                 int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredLod(texture<float1, texType, mode> texRef, float x,
                                                  int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredLod(texture<float2, texType, mode> texRef, float x,
                                                  int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredLod(texture<float4, texType, mode> texRef, float x,
                                                  int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1DLayeredLod(texture<char, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLayeredLod(texture<char1, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLayeredLod(texture<char2, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLayeredLod(texture<char4, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLayeredLod(
    texture<unsigned char, texType, mode> texRef, hipTextureObject_t textureObject, float x,
    int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLayeredLod(texture<uchar1, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLayeredLod(texture<uchar2, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLayeredLod(texture<uchar4, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1DLayeredLod(texture<short, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLayeredLod(texture<short1, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLayeredLod(texture<short2, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLayeredLod(texture<short4, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLayeredLod(
    texture<unsigned short, texType, mode> texRef, hipTextureObject_t textureObject, float x,
    int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLayeredLod(texture<ushort1, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLayeredLod(texture<ushort2, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLayeredLod(texture<ushort4, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1DLayeredLod(texture<int, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, int layer,
                                               float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLayeredLod(texture<int1, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLayeredLod(texture<int2, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLayeredLod(texture<int4, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLayeredLod(texture<unsigned int, texType, mode> texRef,
                                                        hipTextureObject_t textureObject, float x,
                                                        int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLayeredLod(texture<uint1, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLayeredLod(texture<uint2, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLayeredLod(texture<uint4, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredLod(texture<float, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredLod(texture<float1, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredLod(texture<float2, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredLod(texture<float4, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1DLayeredGrad(texture<char, texType, mode> texRef, float x,
                                                 int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex1DLayeredGrad(texture<char, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLayeredGrad(texture<char1, texType, mode> texRef, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLayeredGrad(texture<char1, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLayeredGrad(texture<char2, texType, mode> texRef, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLayeredGrad(texture<char2, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLayeredGrad(texture<char4, texType, mode> texRef, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLayeredGrad(texture<char4, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLayeredGrad(
    texture<unsigned char, texType, mode> texRef, float x, int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLayeredGrad(
    texture<unsigned char, texType, mode> texRef, hipTextureObject_t textureObject, float x,
    int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLayeredGrad(texture<uchar1, texType, mode> texRef, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLayeredGrad(texture<uchar1, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLayeredGrad(texture<uchar2, texType, mode> texRef, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLayeredGrad(texture<uchar2, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLayeredGrad(texture<uchar4, texType, mode> texRef, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLayeredGrad(texture<uchar4, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1DLayeredGrad(texture<short, texType, mode> texRef, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex1DLayeredGrad(texture<short, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLayeredGrad(texture<short1, texType, mode> texRef, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLayeredGrad(texture<short1, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLayeredGrad(texture<short2, texType, mode> texRef, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLayeredGrad(texture<short2, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLayeredGrad(texture<short4, texType, mode> texRef, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLayeredGrad(texture<short4, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLayeredGrad(
    texture<unsigned short, texType, mode> texRef, float x, int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLayeredGrad(
    texture<unsigned short, texType, mode> texRef, hipTextureObject_t textureObject, float x,
    int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLayeredGrad(texture<ushort1, texType, mode> texRef, float x,
                                                    int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLayeredGrad(texture<ushort1, texType, mode> texRef,
                                                    hipTextureObject_t textureObject, float x,
                                                    int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLayeredGrad(texture<ushort2, texType, mode> texRef, float x,
                                                    int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLayeredGrad(texture<ushort2, texType, mode> texRef,
                                                    hipTextureObject_t textureObject, float x,
                                                    int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLayeredGrad(texture<ushort4, texType, mode> texRef, float x,
                                                    int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLayeredGrad(texture<ushort4, texType, mode> texRef,
                                                    hipTextureObject_t textureObject, float x,
                                                    int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1DLayeredGrad(texture<int, texType, mode> texRef, float x,
                                                int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex1DLayeredGrad(texture<int, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x,
                                                int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLayeredGrad(texture<int1, texType, mode> texRef, float x,
                                                 int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLayeredGrad(texture<int1, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLayeredGrad(texture<int2, texType, mode> texRef, float x,
                                                 int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLayeredGrad(texture<int2, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLayeredGrad(texture<int4, texType, mode> texRef, float x,
                                                 int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLayeredGrad(texture<int4, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x,
                                                 int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLayeredGrad(
    texture<unsigned int, texType, mode> texRef, float x, int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLayeredGrad(
    texture<unsigned int, texType, mode> texRef, hipTextureObject_t textureObject, float x,
    int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLayeredGrad(texture<uint1, texType, mode> texRef, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLayeredGrad(texture<uint1, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLayeredGrad(texture<uint2, texType, mode> texRef, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLayeredGrad(texture<uint2, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLayeredGrad(texture<uint4, texType, mode> texRef, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLayeredGrad(texture<uint4, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredGrad(texture<float, texType, mode> texRef, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredGrad(texture<float, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredGrad(texture<float1, texType, mode> texRef, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredGrad(texture<float1, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredGrad(texture<float2, texType, mode> texRef, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredGrad(texture<float2, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredGrad(texture<float4, texType, mode> texRef, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_FLOAT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredGrad(texture<float4, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   int layer, float dx, float dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dx, dy);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2DLayered(texture<char, texType, mode> texRef, float x, float y,
                                             int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2DLayered(texture<char, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float y,
                                             int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLayered(texture<char1, texType, mode> texRef, float x,
                                              float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLayered(texture<char1, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x, float y,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLayered(texture<char2, texType, mode> texRef, float x,
                                              float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLayered(texture<char2, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x, float y,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLayered(texture<char4, texType, mode> texRef, float x,
                                              float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLayered(texture<char4, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x, float y,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLayered(texture<unsigned char, texType, mode> texRef,
                                                      float x, float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLayered(texture<unsigned char, texType, mode> texRef,
                                                      hipTextureObject_t textureObject, float x,
                                                      float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLayered(texture<uchar1, texType, mode> texRef, float x,
                                               float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLayered(texture<uchar1, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLayered(texture<uchar2, texType, mode> texRef, float x,
                                               float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLayered(texture<uchar2, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLayered(texture<uchar4, texType, mode> texRef, float x,
                                               float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLayered(texture<uchar4, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2DLayered(texture<short, texType, mode> texRef, float x,
                                              float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2DLayered(texture<short, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x, float y,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLayered(texture<short1, texType, mode> texRef, float x,
                                               float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLayered(texture<short1, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLayered(texture<short2, texType, mode> texRef, float x,
                                               float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLayered(texture<short2, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLayered(texture<short4, texType, mode> texRef, float x,
                                               float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLayered(texture<short4, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLayered(
    texture<unsigned short, texType, mode> texRef, float x, float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLayered(
    texture<unsigned short, texType, mode> texRef, hipTextureObject_t textureObject, float x,
    float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLayered(texture<ushort1, texType, mode> texRef, float x,
                                                float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLayered(texture<ushort1, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x, float y,
                                                int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLayered(texture<ushort2, texType, mode> texRef, float x,
                                                float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLayered(texture<ushort2, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x, float y,
                                                int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLayered(texture<ushort4, texType, mode> texRef, float x,
                                                float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLayered(texture<ushort4, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x, float y,
                                                int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2DLayered(texture<int, texType, mode> texRef, float x, float y,
                                            int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2DLayered(texture<int, texType, mode> texRef,
                                            hipTextureObject_t textureObject, float x, float y,
                                            int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLayered(texture<int1, texType, mode> texRef, float x, float y,
                                             int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLayered(texture<int1, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float y,
                                             int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLayered(texture<int2, texType, mode> texRef, float x, float y,
                                             int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLayered(texture<int2, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float y,
                                             int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLayered(texture<int4, texType, mode> texRef, float x, float y,
                                             int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLayered(texture<int4, texType, mode> texRef,
                                             hipTextureObject_t textureObject, float x, float y,
                                             int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLayered(texture<unsigned int, texType, mode> texRef,
                                                     float x, float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLayered(texture<unsigned int, texType, mode> texRef,
                                                     hipTextureObject_t textureObject, float x,
                                                     float y, int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLayered(texture<uint1, texType, mode> texRef, float x,
                                              float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLayered(texture<uint1, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x, float y,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLayered(texture<uint2, texType, mode> texRef, float x,
                                              float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLayered(texture<uint2, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x, float y,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLayered(texture<uint4, texType, mode> texRef, float x,
                                              float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLayered(texture<uint4, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x, float y,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2DLayered(texture<float, texType, mode> texRef, float x,
                                              float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2DLayered(texture<float, texType, mode> texRef,
                                              hipTextureObject_t textureObject, float x, float y,
                                              int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayered(texture<float1, texType, mode> texRef, float x,
                                               float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayered(texture<float1, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayered(texture<float2, texType, mode> texRef, float x,
                                               float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayered(texture<float2, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayered(texture<float4, texType, mode> texRef, float x,
                                               float y, int layer) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayered(texture<float4, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y,
                                               int layer) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2DLayeredLod(texture<char, texType, mode> texRef, float x,
                                                float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2DLayeredLod(texture<char, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x, float y,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLayeredLod(texture<char1, texType, mode> texRef, float x,
                                                 float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLayeredLod(texture<char1, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLayeredLod(texture<char2, texType, mode> texRef, float x,
                                                 float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLayeredLod(texture<char2, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLayeredLod(texture<char4, texType, mode> texRef, float x,
                                                 float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLayeredLod(texture<char4, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLayeredLod(
    texture<unsigned char, texType, mode> texRef, float x, float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLayeredLod(
    texture<unsigned char, texType, mode> texRef, hipTextureObject_t textureObject, float x,
    float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLayeredLod(texture<uchar1, texType, mode> texRef, float x,
                                                  float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLayeredLod(texture<uchar1, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLayeredLod(texture<uchar2, texType, mode> texRef, float x,
                                                  float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLayeredLod(texture<uchar2, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLayeredLod(texture<uchar4, texType, mode> texRef, float x,
                                                  float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLayeredLod(texture<uchar4, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2DLayeredLod(texture<short, texType, mode> texRef, float x,
                                                 float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2DLayeredLod(texture<short, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLayeredLod(texture<short1, texType, mode> texRef, float x,
                                                  float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLayeredLod(texture<short1, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLayeredLod(texture<short2, texType, mode> texRef, float x,
                                                  float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLayeredLod(texture<short2, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLayeredLod(texture<short4, texType, mode> texRef, float x,
                                                  float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLayeredLod(texture<short4, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLayeredLod(
    texture<unsigned short, texType, mode> texRef, float x, float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLayeredLod(
    texture<unsigned short, texType, mode> texRef, hipTextureObject_t textureObject, float x,
    float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLayeredLod(texture<ushort1, texType, mode> texRef, float x,
                                                   float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLayeredLod(texture<ushort1, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLayeredLod(texture<ushort2, texType, mode> texRef, float x,
                                                   float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLayeredLod(texture<ushort2, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLayeredLod(texture<ushort4, texType, mode> texRef, float x,
                                                   float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLayeredLod(texture<ushort4, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2DLayeredLod(texture<int, texType, mode> texRef, float x, float y,
                                               int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2DLayeredLod(texture<int, texType, mode> texRef,
                                               hipTextureObject_t textureObject, float x, float y,
                                               int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLayeredLod(texture<int1, texType, mode> texRef, float x,
                                                float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLayeredLod(texture<int1, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x, float y,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLayeredLod(texture<int2, texType, mode> texRef, float x,
                                                float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLayeredLod(texture<int2, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x, float y,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLayeredLod(texture<int4, texType, mode> texRef, float x,
                                                float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLayeredLod(texture<int4, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x, float y,
                                                int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLayeredLod(texture<unsigned int, texType, mode> texRef,
                                                        float x, float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLayeredLod(texture<unsigned int, texType, mode> texRef,
                                                        hipTextureObject_t textureObject, float x,
                                                        float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLayeredLod(texture<uint1, texType, mode> texRef, float x,
                                                 float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLayeredLod(texture<uint1, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLayeredLod(texture<uint2, texType, mode> texRef, float x,
                                                 float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLayeredLod(texture<uint2, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLayeredLod(texture<uint4, texType, mode> texRef, float x,
                                                 float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLayeredLod(texture<uint4, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredLod(texture<float, texType, mode> texRef, float x,
                                                 float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredLod(texture<float, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredLod(texture<float1, texType, mode> texRef, float x,
                                                  float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredLod(texture<float1, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredLod(texture<float2, texType, mode> texRef, float x,
                                                  float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredLod(texture<float2, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredLod(texture<float4, texType, mode> texRef, float x,
                                                  float y, int layer, float level) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_FLOAT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredLod(texture<float4, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float level) {
    TEXTURE_PARAMETERS_INIT;
    texel.f = __ockl_image_sample_lod_2Da(
        i, s, float4(x, y, layer, 0.0f).data, level);
    TEXTURE_RETURN_FLOAT_XYZW;
}

////////////////////////////////////////////////////////////

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2DLayeredGrad(texture<char, texType, mode> texRef, float x,
                                                 float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char tex2DLayeredGrad(texture<char, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLayeredGrad(texture<char1, texType, mode> texRef, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLayeredGrad(texture<char1, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLayeredGrad(texture<char2, texType, mode> texRef, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLayeredGrad(texture<char2, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLayeredGrad(texture<char4, texType, mode> texRef, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLayeredGrad(texture<char4, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_CHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLayeredGrad(
    texture<unsigned char, texType, mode> texRef, float x, float y, int layer, float2 dx,
    float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLayeredGrad(
    texture<unsigned char, texType, mode> texRef, hipTextureObject_t textureObject, float x,
    float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLayeredGrad(texture<uchar1, texType, mode> texRef, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLayeredGrad(texture<uchar1, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLayeredGrad(texture<uchar2, texType, mode> texRef, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLayeredGrad(texture<uchar2, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLayeredGrad(texture<uchar4, texType, mode> texRef, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLayeredGrad(texture<uchar4, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UCHAR_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2DLayeredGrad(texture<short, texType, mode> texRef, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short tex2DLayeredGrad(texture<short, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLayeredGrad(texture<short1, texType, mode> texRef, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLayeredGrad(texture<short1, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLayeredGrad(texture<short2, texType, mode> texRef, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLayeredGrad(texture<short2, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLayeredGrad(texture<short4, texType, mode> texRef, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLayeredGrad(texture<short4, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_SHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLayeredGrad(
    texture<unsigned short, texType, mode> texRef, float x, float y, int layer, float2 dx,
    float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLayeredGrad(
    texture<unsigned short, texType, mode> texRef, hipTextureObject_t textureObject, float x,
    float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLayeredGrad(texture<ushort1, texType, mode> texRef, float x,
                                                    float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLayeredGrad(texture<ushort1, texType, mode> texRef,
                                                    hipTextureObject_t textureObject, float x,
                                                    float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLayeredGrad(texture<ushort2, texType, mode> texRef, float x,
                                                    float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLayeredGrad(texture<ushort2, texType, mode> texRef,
                                                    hipTextureObject_t textureObject, float x,
                                                    float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLayeredGrad(texture<ushort4, texType, mode> texRef, float x,
                                                    float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLayeredGrad(texture<ushort4, texType, mode> texRef,
                                                    hipTextureObject_t textureObject, float x,
                                                    float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_USHORT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2DLayeredGrad(texture<int, texType, mode> texRef, float x,
                                                float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int tex2DLayeredGrad(texture<int, texType, mode> texRef,
                                                hipTextureObject_t textureObject, float x, float y,
                                                int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLayeredGrad(texture<int1, texType, mode> texRef, float x,
                                                 float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLayeredGrad(texture<int1, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLayeredGrad(texture<int2, texType, mode> texRef, float x,
                                                 float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLayeredGrad(texture<int2, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLayeredGrad(texture<int4, texType, mode> texRef, float x,
                                                 float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLayeredGrad(texture<int4, texType, mode> texRef,
                                                 hipTextureObject_t textureObject, float x, float y,
                                                 int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_INT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLayeredGrad(
    texture<unsigned int, texType, mode> texRef, float x, float y, int layer, float2 dx,
    float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLayeredGrad(
    texture<unsigned int, texType, mode> texRef, hipTextureObject_t textureObject, float x, float y,
    int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLayeredGrad(texture<uint1, texType, mode> texRef, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLayeredGrad(texture<uint1, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLayeredGrad(texture<uint2, texType, mode> texRef, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLayeredGrad(texture<uint2, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLayeredGrad(texture<uint4, texType, mode> texRef, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLayeredGrad(texture<uint4, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_UINT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredGrad(texture<float, texType, mode> texRef, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredGrad(texture<float, texType, mode> texRef,
                                                  hipTextureObject_t textureObject, float x,
                                                  float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredGrad(texture<float1, texType, mode> texRef, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredGrad(texture<float1, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_X;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredGrad(texture<float2, texType, mode> texRef, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredGrad(texture<float2, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_XY;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredGrad(texture<float4, texType, mode> texRef, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_REF_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}

template <int texType, enum hipTextureReadMode mode>
__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredGrad(texture<float4, texType, mode> texRef,
                                                   hipTextureObject_t textureObject, float x,
                                                   float y, int layer, float2 dx, float2 dy) {
    TEXTURE_PARAMETERS_INIT;
    texel.f =
        __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data,
                                     float2(dx.x, dx.y).data,
                                     float2(dy.x, dy.y).data);
    TEXTURE_RETURN_FLOAT_XYZW;
}
#endif
