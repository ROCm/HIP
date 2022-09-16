#pragma once
#include <math.h>

#define HIP_SAMPLING_VERIFY_EPSILON     0.00001
// The internal precision varies by the GPU family and sometimes within the family.
// Thus the following threshold is subject to change.
#define HIP_SAMPLING_VERIFY_RELATIVE_THRESHOLD  0.05  // 5% for filter mode
#define HIP_SAMPLING_VERIFY_ABSOLUTE_THRESHOLD  0.1

#if HT_NVIDIA
template<typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T::x)) == 4, T>::type
inline __host__ __device__ operator+(const T &a, const T &b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w};
}

template<typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T::x)) == 4, T>::type
inline __host__ __device__ operator-(const T &a, const T &b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w};
}

template<typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T::x)) == 4, bool>::type
inline __host__ __device__ operator==(const T &a, const T &b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template<typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T::x)) == 4, T>::type
inline __host__ __device__ operator*(const decltype(T::x) &a, const T &b)
{
  return {a * b.x, a * b.y, a * b.z, a * b.w};
}

template<typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T::x)) == 4, void>::type
inline __host__ __device__ operator*=(T &a, const decltype(T::x) &b)
{
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}
#endif // HT_NVIDIA

// See https://en.wikipedia.org/wiki/SRGB#Transformation
// From CIE 1931 color space to sRGB
inline float hipSRGBMap(float fc) {
  double c = static_cast<double>(fc);

#if !defined(_WIN32)
  if (std::isnan(c))
    c = 0.0;
#else
    if (_isnan(c)) c = 0.0;
#endif

  if (c > 1.0)
    c = 1.0;
  else if (c < 0.0)
    c = 0.0;
  else if (c < 0.0031308)
    c = 12.92 * c;
  else
    c = 1.055 * pow(c, 5.0 / 12.0) - 0.055;

  return static_cast<float>(c);
}

// From sRGB to CIE 1931 color space
inline float hipSRGBUnmap(float fc) {
  double c = static_cast<double>(fc);

  if (c <= 0.04045)
    c = c / 12.92;
  else
    c = pow((c + 0.055) / 1.055, 2.4);

  return static_cast<float>(c);
}

inline float4 hipSRGBMap(float4 fc) {
  fc.x = hipSRGBMap(fc.x);
  fc.y = hipSRGBMap(fc.y);
  fc.z = hipSRGBMap(fc.z);
  // Alpha channel will keep unchanged
  return fc;
}

inline float4 hipSRGBUnmap(float4 fc) {
  fc.x = hipSRGBUnmap(fc.x);
  fc.y = hipSRGBUnmap(fc.y);
  fc.z = hipSRGBUnmap(fc.z);
  // Alpha channel will keep unchanged
  return fc;
}

template<typename T>
typename std::enable_if<std::is_scalar<T>::value == true, double>::type
hipFabs(const T &t) {
  return fabs(t);
}

template<typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T::x)) == 1, double>::type
hipFabs(const T &t) {
  return fabs(t.x);
}

template<typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T::x)) == 2, double>::type
hipFabs(const T &t) {
  double x = static_cast<double>(t.x);
  double y = static_cast<double>(t.y);
  double s =  x * x +  y * y;
  return sqrt(s);
}

template<typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T::x)) == 3, double>::type
hipFabs(const T &t) {
  double x = static_cast<double>(t.x);
  double y = static_cast<double>(t.y);
  double z = static_cast<double>(t.z);
  double s =  x * x +  y * y + z * z;
  return sqrt(s);
}

template<typename T>
typename std::enable_if<sizeof(T) / sizeof(decltype(T::x)) == 4, double>::type
hipFabs(const T &t) {
  double x = static_cast<double>(t.x);
  double y = static_cast<double>(t.y);
  double z = static_cast<double>(t.z);
  double w = static_cast<double>(t.w);
  double s =  x * x +  y * y + z * z + w * w;
  return sqrt(s);
}

template<typename T, hipTextureFilterMode fMode = hipFilterModePoint, bool sRGB = false>
bool hipTextureSamplingVerify(T outputData, T expected) {
  bool testResult = false;
  if (fMode == hipFilterModePoint && !sRGB) {
    testResult = outputData == expected;
  } else {
    double mean = (hipFabs(outputData) + hipFabs(expected)) / 2;
    double diff = hipFabs(outputData - expected);
    double ratio = diff / (mean + HIP_SAMPLING_VERIFY_EPSILON);
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

// Simulate logics in CTS read_image_pixel_float().
// x, y and z must be returned by hipTextureGetAddress()
template<typename T, hipTextureAddressMode addressMode, bool sRGB = false>
T hipTextureGetValue(const T *data, const int x, const int width,
         const int y = 0, const int height = 0, const int z = 0, const int depth = 0) {
  T result;
  memset(&result, 0, sizeof(result));
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
          if (x >= 0 && x < width)
            result = data[x];  // 1D
        } else if (depth == 0) {
          if (x >= 0 && x < width && y >= 0 && y < height)
              result = data[y * width + x];  // 2D
        } else {
          if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth)
              result = data[z * width * height + y * width + x];  // 3D
        }
      }
      break;
    default:
      break;
  }
  if (sRGB && std::is_same<T, float4>::value) {
    result = hipSRGBUnmap(result);
  }
  return result;
}

template<typename T, hipTextureAddressMode addressMode, hipTextureFilterMode filterMode, bool sRGB = false>
T getExpectedValue(const int width, float x, const T *data) {
  T result;
  memset(&result, 0, sizeof(result));
  switch (filterMode) {
    case hipFilterModePoint: {
      int i1 = static_cast<int>(floor(x));
      hipTextureGetAddress < addressMode > (i1, width);
      result = hipTextureGetValue < T, addressMode, sRGB > (data, i1, width);
    }
      break;
    case hipFilterModeLinear: {
      x -= 0.5;
      int i1 = static_cast<int>(floor(x));
      int i2 = i1 + 1;
      float a = x - i1;
      hipTextureGetAddress < addressMode > (i1, width);
      hipTextureGetAddress < addressMode > (i2, width);

      T t1 = hipTextureGetValue < T, addressMode, sRGB> (data, i1, width);
      T t2 = hipTextureGetValue < T, addressMode, sRGB > (data, i2, width);

      return (1 - a) * t1 + a * t2;
    }
      break;
  }
  return result;
}

template<typename T, hipTextureAddressMode addressMode, hipTextureFilterMode filterMode, bool sRGB = false>
T getExpectedValue(const int width, const int height, float x, float y, const T *data) {
  T result;
  memset(&result, 0, sizeof(result));
  switch (filterMode) {
    case hipFilterModePoint: {
      int i1 = static_cast<int>(floor(x));
      int j1 = static_cast<int>(floor(y));
      hipTextureGetAddress < addressMode > (i1, width);
      hipTextureGetAddress < addressMode > (j1, height);
      result = hipTextureGetValue < T, addressMode, sRGB > (data, i1, width, j1, height);
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

      T t11 = hipTextureGetValue < T, addressMode, sRGB
          > (data, i1, width, j1, height);
      T t21 = hipTextureGetValue < T, addressMode, sRGB
          > (data, i2, width, j1, height);
      T t12 = hipTextureGetValue < T, addressMode, sRGB
          > (data, i1, width, j2, height);
      T t22 = hipTextureGetValue < T, addressMode, sRGB
          > (data, i2, width, j2, height);

      result = (1 - a) * (1 - b) * t11 + a * (1 - b) * t21 + (1 - a) * b * t12
          + a * b * t22;
    }
      break;
  }
  return result;
}

template<class T, hipTextureAddressMode addressMode, hipTextureFilterMode filterMode, bool sRGB = false>
T getExpectedValue(const int width, const int height, const int depth,
                      float x, float y, float z, const T *data) {
  T result;
  memset(&result, 0, sizeof(result));
  switch (filterMode) {
    case hipFilterModePoint: {
      int i1 = static_cast<int>(floor(x));
      int j1 = static_cast<int>(floor(y));
      int k1 = static_cast<int>(floor(z));

      hipTextureGetAddress < addressMode > (i1, width);
      hipTextureGetAddress < addressMode > (j1, height);
      hipTextureGetAddress < addressMode > (k1, depth);

      result = hipTextureGetValue < T, addressMode, sRGB > (data, i1, width, j1, height, k1, depth);
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

      T t111 = hipTextureGetValue < T, addressMode, sRGB
          > (data, i1, width, j1, height, k1, depth);
      T t211 = hipTextureGetValue < T, addressMode, sRGB
          > (data, i2, width, j1, height, k1, depth);
      T t121 = hipTextureGetValue < T, addressMode, sRGB
          > (data, i1, width, j2, height, k1, depth);
      T t112 = hipTextureGetValue < T, addressMode, sRGB
          > (data, i1, width, j1, height, k2, depth);
      T t122 = hipTextureGetValue < T, addressMode, sRGB
          > (data, i1, width, j2, height, k2, depth);
      T t212 = hipTextureGetValue < T, addressMode, sRGB
          > (data, i2, width, j1, height, k2, depth);
      T t221 = hipTextureGetValue < T, addressMode, sRGB
          > (data, i2, width, j2, height, k1, depth);
      T t222 = hipTextureGetValue < T, addressMode, sRGB
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
