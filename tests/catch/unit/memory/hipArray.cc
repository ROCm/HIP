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
#include <hip_test_common.hh>
TEST_CASE("Unit_hipArray_Valid") {
    hipArray* array = nullptr;
    HIP_ARRAY_DESCRIPTOR desc;
    desc.Format = HIP_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = 1024;
    desc.Height = 1024;
    HIP_CHECK(hipArrayCreate(&array, &desc));
    HIP_CHECK(hipFreeArray(array));
}
TEST_CASE("Unit_hipArray_Invalid") {
    void* data = malloc(sizeof(char));
    hipArray_t arrayPtr = static_cast<hipArray*>(data);
    REQUIRE(hipFreeArray(arrayPtr) == hipErrorContextIsDestroyed);
    free(data);
}
TEST_CASE("Unit_hipArray_Nullptr") {
    hipArray* array = nullptr;
    REQUIRE(hipFreeArray(array) == hipErrorInvalidValue);
}
TEST_CASE("Unit_hipArray_DoubleFree") {
    hipArray* array = nullptr;
    HIP_ARRAY_DESCRIPTOR desc;
    desc.Format = HIP_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = 1024;
    desc.Height = 1024;
    HIP_CHECK(hipArrayCreate(&array, &desc));
    HIP_CHECK(hipFreeArray(array));
    REQUIRE(hipFreeArray(array) == hipErrorContextIsDestroyed);
}
TEST_CASE("Unit_hipArray_TrippleDestroy") {
    hipArray* array = nullptr;
    HIP_ARRAY_DESCRIPTOR desc;
    desc.Format = HIP_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = 1024;
    desc.Height = 1024;
    HIP_CHECK(hipArrayCreate(&array, &desc));
    HIP_CHECK(hipArrayDestroy(array));
    REQUIRE(hipArrayDestroy(array) == hipErrorContextIsDestroyed);
    REQUIRE(hipArrayDestroy(array) == hipErrorContextIsDestroyed);
}
TEST_CASE("Unit_hipArray_DoubleNullptr") {
    hipArray* array = nullptr;
    REQUIRE(hipFreeArray(array) == hipErrorInvalidValue);
    REQUIRE(hipFreeArray(array) == hipErrorInvalidValue);
}
TEST_CASE("Unit_hipArray_DoubleInvalid") {
    void* data = malloc(sizeof(char));
    hipArray_t arrayPtr = static_cast<hipArray*>(data);
    REQUIRE(hipFreeArray(arrayPtr) == hipErrorContextIsDestroyed);
    REQUIRE(hipFreeArray(arrayPtr) == hipErrorContextIsDestroyed);
    free(data);
}


