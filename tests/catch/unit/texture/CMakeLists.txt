# Copyright (c) 2021 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipCreateTextureObject_ArgValidation.cc
    hipCreateTextureObject_Linear.cc
    hipCreateTextureObject_Pitch2D.cc
    hipCreateTextureObject_Array.cc
    hipTextureObjFetchVector.cc
    hipNormalizedFloatValueTex.cc
    hipTextureObj2D.cc
    hipSimpleTexture3D.cc
    hipTextureRef2D.cc
    hipSimpleTexture2DLayered.cc
    hipTextureMipmapObj2D.cc
    hipBindTex2DPitch.cc
    hipBindTexRef1DFetch.cc
    hipTex1DFetchCheckModes.cc
    hipGetChanDesc.cc
    hipTexObjPitch.cc
    hipTextureObj1DFetch.cc
    hipBindTex2DPitch.cc
    hipBindTexRef1DFetch.cc
    hipTex1DFetchCheckModes.cc
    hipTextureObj1DCheckModes.cc
    hipTextureObj2DCheckModes.cc
    hipTextureObj3DCheckModes.cc
    hipTextureObj1DCheckSRGBModes.cc
    hipTextureObj2DCheckSRGBModes.cc
    hipBindTextureToMipmappedArray.cc
)

hip_add_exe_to_target(NAME TextureTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
