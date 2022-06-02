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

#include "DriverContext.hh"
#include "hipArrayCommon.hh"

namespace {
void checkArrayIsExpected(const hiparray array, const HIP_ARRAY3D_DESCRIPTOR& expected_desc) {
  std::ignore = array;
  std::ignore = expected_desc;

  // hipArray3DGetDescriptor doesn't currently exist (EXSWCPHIPT-87)
  // HIP_ARRAY3D_DESCRIPTOR queried_desc;
  // hipArray3DGetDescriptor(queried_desc, array);

  // REQUIRE(queried_desc.Width == expected_desc.Width);
  // REQUIRE(queried_desc.Height == expected_desc.Height);
  // REQUIRE(queried_desc.Depth == expected_desc.Depth);
  // REQUIRE(queried_desc.Format == expected_desc.Format);
  // REQUIRE(queried_desc.NumChannels == expected_desc.NumChannels);
  // REQUIRE(queried_desc.Flags == expected_desc.Flags);
}
}  // namespace

TEMPLATE_TEST_CASE("Unit_hipArray3DCreate_happy", "", char, uchar2, uint2, int4, short4, float,
                   float2, float4) {
  using vec_info = vector_info<TestType>;
  DriverContext ctx;

  hiparray array;
  HIP_ARRAY3D_DESCRIPTOR desc{};
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
#if HT_AMD
  desc.Flags = 0;
#else
  desc.Flags = GENERATE(0, CUDA_ARRAY3D_SURFACE_LDST);
#endif

  constexpr size_t size = 64;

  std::vector<hipExtent> extents{
      {size, 0, 0},       // 1D array
      {size, size, 0},    // 2D array
      {size, size, size}  // 3D array
  };

  for (auto& extent : extents) {
    desc.Width = extent.width;
    desc.Height = extent.height;
    desc.Depth = extent.depth;

    CAPTURE(desc.Width, desc.Height, desc.Depth);

    HIP_CHECK(hipArray3DCreate(&array, &desc));
    checkArrayIsExpected(array, desc);
    HIP_CHECK(hipArrayDestroy(array));
  }
}

TEMPLATE_TEST_CASE("Unit_hipArray3DCreate_MaxTexture", "", int, uint4, short, ushort2,
                   unsigned char, float, float4) {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-97");
  return;
#endif

  using vec_info = vector_info<TestType>;
  DriverContext ctx;

  hiparray array;
  HIP_ARRAY3D_DESCRIPTOR desc{};
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
#if HT_AMD
  desc.Flags = 0;
#else
  desc.Flags = GENERATE(0, CUDA_ARRAY3D_SURFACE_LDST);
  if (desc.Flags == CUDA_ARRAY3D_SURFACE_LDST) {
    HipTest::HIP_SKIP_TEST("EXSWCPHIPT-58");
    return;
  }
#endif
  CAPTURE(desc.Flags);

  const Sizes sizes(desc.Flags);
  CAPTURE(sizes.max1D, sizes.max2D, sizes.max3D);

  const size_t s = 64;
  SECTION("Happy") {
    // stored in a vector so some values can be ifdef'd out
    std::vector<hipExtent> extentsToTest{
        make_hipExtent(sizes.max1D, 0, 0),                              // 1D max
        make_hipExtent(sizes.max2D[0], s, 0),                           // 2D max width
        make_hipExtent(s, sizes.max2D[1], 0),                           // 2D max height
        make_hipExtent(sizes.max2D[0], sizes.max2D[1], 0),              // 2D max
        make_hipExtent(sizes.max3D[0], s, s),                           // 3D max width
        make_hipExtent(s, sizes.max3D[1], s),                           // 3D max height
        make_hipExtent(s, s, sizes.max3D[2]),                           // 3D max depth
        make_hipExtent(s, sizes.max3D[1], sizes.max3D[2]),              // 3D max height and depth
        make_hipExtent(sizes.max3D[0], s, sizes.max3D[2]),              // 3D max width and depth
        make_hipExtent(sizes.max3D[0], sizes.max3D[1], s),              // 3D max width and height
        make_hipExtent(sizes.max3D[0], sizes.max3D[1], sizes.max3D[2])  // 3D max
    };
    const auto extent =
        GENERATE_COPY(from_range(std::begin(extentsToTest), std::end(extentsToTest)));

    desc.Width = extent.width;
    desc.Height = extent.height;
    desc.Depth = extent.depth;

    CAPTURE(desc.Width, desc.Height, desc.Depth);

    auto maxArrayCreateError = hipArray3DCreate(&array, &desc);
    // this can try to alloc many GB of memory, so out of memory is acceptable
    if (maxArrayCreateError == hipErrorOutOfMemory) return;
    HIP_CHECK(maxArrayCreateError);
    checkArrayIsExpected(array, desc);
    HIP_CHECK(hipArrayDestroy(array));
  }
  SECTION("Negative") {
    std::vector<hipExtent> extentsToTest {
      make_hipExtent(sizes.max1D + 1, 0, 0),                          // 1D max
          make_hipExtent(sizes.max2D[0] + 1, s, 0),                   // 2D max width
          make_hipExtent(s, sizes.max2D[1] + 1, 0),                   // 2D max height
          make_hipExtent(sizes.max2D[0] + 1, sizes.max2D[1] + 1, 0),  // 2D max
          make_hipExtent(sizes.max3D[0] + 1, s, s),                   // 3D max width
          make_hipExtent(s, sizes.max3D[1] + 1, s),                   // 3D max height
#if !HT_NVIDIA                                       // leads to hipSuccess on NVIDIA
          make_hipExtent(s, s, sizes.max3D[2] + 1),  // 3D max depth
#endif
          make_hipExtent(s, sizes.max3D[1] + 1, sizes.max3D[2] + 1),  // 3D max height and depth
          make_hipExtent(sizes.max3D[0] + 1, s, sizes.max3D[2] + 1),  // 3D max width and depth
          make_hipExtent(sizes.max3D[0] + 1, sizes.max3D[1] + 1, s),  // 3D max width and height
          make_hipExtent(sizes.max3D[0] + 1, sizes.max3D[1] + 1, sizes.max3D[2] + 1)  // 3D max
    };
    const auto extent =
        GENERATE_COPY(from_range(std::begin(extentsToTest), std::end(extentsToTest)));

    desc.Width = extent.width;
    desc.Height = extent.height;
    desc.Depth = extent.depth;

    CAPTURE(desc.Width, desc.Height, desc.Depth);

    HIP_CHECK_ERROR(hipArray3DCreate(&array, &desc), hipErrorInvalidValue);
  }
}
