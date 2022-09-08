#pragma once

#define HIP_SAMPLING_VERIFY_EPSILON     0.00001
// The internal precision varies by the GPU family and sometimes within the family.
// Thus the following threshold is subject to change.
#define HIP_SAMPLING_VERIFY_RELATIVE_THRESHOLD  0.05  // 5% for filter mode
#define HIP_SAMPLING_VERIFY_ABSOLUTE_THRESHOLD  0.1

template<typename type, hipTextureFilterMode fMode = hipFilterModePoint>
bool hipTextureSamplingVerify(const type outputData, const type expected) {
  bool testResult = false;
  if (fMode == hipFilterModePoint) {
    testResult = outputData == expected;
  } else if (fMode == hipFilterModeLinear) {
    const type mean = (fabs(outputData) + fabs(expected)) / 2;
    const type diff = fabs(outputData - expected);
    const type ratio = diff / (mean + HIP_SAMPLING_VERIFY_EPSILON);
    if (ratio <= HIP_SAMPLING_VERIFY_RELATIVE_THRESHOLD) {
      testResult = true;
    } else if (diff <= HIP_SAMPLING_VERIFY_ABSOLUTE_THRESHOLD) {
      // Some small outputs have big ratio due to float operation difference of ALU and GPU
      testResult = true;
    }
  }
  return testResult;
}

// Simulate CTS static AddressingTable sAddressingTable
template<hipTextureAddressMode addressMode>
void hipTextureGetAddress(int &value, const int maxValue)
{
  switch(addressMode)
  {
    case hipAddressModeClamp:
      value = value < 0 ? 0
                  : (value > maxValue - 1 ? maxValue - 1 : value);
      break;
    case hipAddressModeBorder:
      value = value < -1 ? -1
                  : (value > maxValue ? maxValue : value);
      break;
    default:
      break;
  }
}

// Simlate logics in CTS read_image_pixel_float().
// x, y and z must be returned by hipTextureGetAddress()
template<hipTextureAddressMode addressMode>
float hipTextureGetValue(const float *data, const int x, const int width,
         const int y = 0, const int height = 0,const int z = 0, const int depth = 0) {
  float result = std::numeric_limits<float>::lowest();
  switch (addressMode) {
    case hipAddressModeClamp:
      if (width > 0) {
        if (height == 0 && depth == 0) {
          result = data[x];  // 1D
        } else if (depth == 0) {
          result = data[y * width + x];  // 2D
        } else {
          result = data[z * width * height + y * width + x];  // 3D
        }
      }
      break;
    case hipAddressModeBorder:
      if (width > 0) {
        if (height == 0 && depth == 0) {
          result = (x >= 0 && x < width) ? data[x] : 0;  // 1D
        } else if (depth == 0) {
          result = (x >= 0 && x < width && y >= 0 && y < height) ?
              data[y * width + x] : 0;  // 2D
        } else {
          result = (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) ?
              data[z * width * height + y * width + x] : 0;  // 3D
        }
      }
      break;
    default:
      break;
  }
  return result;
}

template<hipTextureAddressMode addressMode, hipTextureFilterMode filterMode>
float getExpectedValue(const int width, float x, const float *data) {
  float result = std::numeric_limits<float>::lowest();
  switch (filterMode) {
    case hipFilterModePoint: {
      int i1 = static_cast<int>(floor(x));
      hipTextureGetAddress < addressMode > (i1, width);
      result = hipTextureGetValue < addressMode > (data, i1, width);
    }
      break;
    case hipFilterModeLinear: {
      x -= 0.5;
      int i1 = static_cast<int>(floor(x));
      int i2 = i1 + 1;
      float a = x - i1;
      hipTextureGetAddress < addressMode > (i1, width);
      hipTextureGetAddress < addressMode > (i2, width);

      float t1 = hipTextureGetValue < addressMode > (data, i1, width);
      float t2 = hipTextureGetValue < addressMode > (data, i2, width);

      return (1 - a) * t1 + a * t2;
    }
      break;
  }
  return result;
}

template<hipTextureAddressMode addressMode, hipTextureFilterMode filterMode>
float getExpectedValue(const int width, const int height, float x, float y, const float *data) {
  float result = std::numeric_limits<float>::lowest();
  switch (filterMode) {
    case hipFilterModePoint: {
      int i1 = static_cast<int>(floor(x));
      int j1 = static_cast<int>(floor(y));
      hipTextureGetAddress < addressMode > (i1, width);
      hipTextureGetAddress < addressMode > (j1, height);
      result = hipTextureGetValue < addressMode > (data, i1, width, j1, height);
    }
      break;
    case hipFilterModeLinear: {
      x -= 0.5;
      y -= 0.5;

      int i1 = static_cast<int>(floor(x));
      int j1 = static_cast<int>(floor(y));

      int i2 = i1 + 1;
      int j2 = j1 + 1;

      float a = x - i1;
      float b = y - j1;

      hipTextureGetAddress < addressMode > (i1, width);
      hipTextureGetAddress < addressMode > (i2, width);
      hipTextureGetAddress < addressMode > (j1, height);
      hipTextureGetAddress < addressMode > (j2, height);

      float t11 = hipTextureGetValue < addressMode
          > (data, i1, width, j1, height);
      float t21 = hipTextureGetValue < addressMode
          > (data, i2, width, j1, height);
      float t12 = hipTextureGetValue < addressMode
          > (data, i1, width, j2, height);
      float t22 = hipTextureGetValue < addressMode
          > (data, i2, width, j2, height);

      result = (1 - a) * (1 - b) * t11 + a * (1 - b) * t21 + (1 - a) * b * t12
          + a * b * t22;
    }
      break;
  }
  return result;
}

template<hipTextureAddressMode addressMode, hipTextureFilterMode filterMode>
float getExpectedValue(const int width, const int height, const int depth,
                      float x, float y, float z, const float *data) {
  float result = std::numeric_limits<float>::lowest();
  switch (filterMode) {
    case hipFilterModePoint: {
      int i1 = static_cast<int>(floor(x));
      int j1 = static_cast<int>(floor(y));
      int k1 = static_cast<int>(floor(z));

      hipTextureGetAddress < addressMode > (i1, width);
      hipTextureGetAddress < addressMode > (j1, height);
      hipTextureGetAddress < addressMode > (k1, depth);

      result = hipTextureGetValue < addressMode > (data, i1, width, j1, height, k1, depth);
    }
      break;
    case hipFilterModeLinear: {
      x -= 0.5;
      y -= 0.5;
      z -= 0.5;

      int i1 = static_cast<int>(floor(x));
      int j1 = static_cast<int>(floor(y));
      int k1 = static_cast<int>(floor(z));

      int i2 = i1 + 1;
      int j2 = j1 + 1;
      int k2 = k1 + 1;

      float a = x - i1;
      float b = y - j1;
      float c = z - k1;

      hipTextureGetAddress < addressMode > (i1, width);
      hipTextureGetAddress < addressMode > (i2, width);
      hipTextureGetAddress < addressMode > (j1, height);
      hipTextureGetAddress < addressMode > (j2, height);
      hipTextureGetAddress < addressMode > (k1, depth);
      hipTextureGetAddress < addressMode > (k2, depth);

      float t111 = hipTextureGetValue < addressMode
          > (data, i1, width, j1, height, k1, depth);
      float t211 = hipTextureGetValue < addressMode
          > (data, i2, width, j1, height, k1, depth);
      float t121 = hipTextureGetValue < addressMode
          > (data, i1, width, j2, height, k1, depth);
      float t112 = hipTextureGetValue < addressMode
          > (data, i1, width, j1, height, k2, depth);
      float t122 = hipTextureGetValue < addressMode
          > (data, i1, width, j2, height, k2, depth);
      float t212 = hipTextureGetValue < addressMode
          > (data, i2, width, j1, height, k2, depth);
      float t221 = hipTextureGetValue < addressMode
          > (data, i2, width, j2, height, k1, depth);
      float t222 = hipTextureGetValue < addressMode
          > (data, i2, width, j2, height, k2, depth);

      result =
                (1 - a) * (1 - b) * (1 - c) * t111 + a * (1 - b) * (1 - c) * t211 +
                (1 - a) * b * (1 - c) * t121 + a * b * (1 - c) * t221 +
                (1 - a) * (1 - b) * c * t112 + a * (1 - b) * c * t212 +
                (1 - a) * b * c * t122 + a * b * c * t222;

    }
      break;
  }
  return result;
}