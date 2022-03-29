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

#pragma once
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <kernel_mapping.hh>
#include <catch.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

namespace HipTest {

struct KernelArgument{
  const void* ptr;
  size_t sizeRequirements;
  size_t alignmentRequirement;
};

/**
 * @brief Reconstructs the name expression for the kernel.
 *
 * @param kernelName the name of the kernel (e.g. "HipTest::VectorADD")
 * @param typenames the typenames used by this kernel (e.g. "float").
 * @return std::string the reconstructed expression (e.g. "VectorADD<float>""). Returns kernelName instead if the kernel is not a template.
 */
inline std::string reconstructExpression(std::string& kernelName,
                                         std::vector<std::string>& typenames) {
  std::string kernelExpression = kernelName;
  if (typenames.size() > 0) {
    kernelExpression += "<" + typenames[0];
    for (size_t i = 1; i < typenames.size(); ++i) {
      kernelExpression += "," + typenames[i];
    }
    kernelExpression += ">";
  }

  return kernelExpression;
}

/**
 * @brief Packs the kernel arguments into the format expected by hipModuleLaunchKernel
 *
 * @param args list of arguments for the kernel and their alignemnt requirements.
 * @return std::vector<char> the packed arguments ready to be passed on to hipModuleLaunchKernel
 */
inline std::vector<char> alignArguments(std::vector<KernelArgument>& args) {
  std::vector<char> alignedArguments{};
  int count = 0;
  for (auto& arg : args) {
    const char* argPtr{reinterpret_cast<const char*>(arg.ptr)};

    /*
     * Details about the padding formula can be found at:
     * https://en.wikipedia.org/wiki/Data_structure_alignment#Data_structure_padding
     */
    int paddingNeeded = -count & (arg.alignmentRequirement - 1);
    alignedArguments.insert(std::end(alignedArguments), paddingNeeded, 0);
    count += paddingNeeded;

    alignedArguments.insert(std::end(alignedArguments), argPtr, argPtr + arg.sizeRequirements);
    count += arg.sizeRequirements;
  }
  return alignedArguments;
}

inline std::vector<char> getKernelCode(hiprtcProgram& rtcProgram) {
  size_t codeSize;
  REQUIRE(HIPRTC_SUCCESS == hiprtcGetCodeSize(rtcProgram, &codeSize));

  std::vector<char> code(codeSize);
  REQUIRE(HIPRTC_SUCCESS == hiprtcGetCode(rtcProgram, code.data()));

  return code;
}

/**
 * @brief Compiles a kernel using HIP RTC
 * 
 * @param rtcKernel the name of the kernel to compile.
 * @param kernelNameExpression the name expression to be added to the RTC program (e.g. HipTest::VectorADD<float>)
 * @return hiprtcProgram the compiled rtc program.
 */
inline hiprtcProgram compileRTC(std::string& rtcKernel, std::string& kernelNameExpression) {
  std::string fileName = mapKernelToFileName.at(rtcKernel);
  std::string filePath{KERNELS_PATH + fileName};

  INFO("Opening Kernel File: " << filePath);
  std::ifstream kernelFile{filePath};
  REQUIRE(kernelFile.is_open());

  std::stringstream stringStream;
  std::string line;
  while (getline(kernelFile, line)) {
    /* Skip the include directive since it is not part of the kernel */
    if (line.find("#include") != std::string::npos) {
      continue;
    }
    stringStream << line << '\n';
  }
  kernelFile.close();

  std::string kernelCode{stringStream.str()};
  INFO("RTC Kernel Code:\n" << kernelCode)

  hiprtcProgram rtcProgram;
  hiprtcCreateProgram(&rtcProgram, kernelCode.c_str(), (fileName + ".cu").c_str(), 0, nullptr,
                      nullptr);

#ifdef __HIP_PLATFORM_AMD__
  
  int deviceCount;
  REQUIRE(hipSuccess == hipGetDeviceCount(&deviceCount));
  
  std::set<std::string> architectures{};
  for (int i = 0; i < deviceCount; ++i) {
    hipDeviceProp_t props;
    REQUIRE(hipSuccess == hipGetDeviceProperties(&props, i));
    architectures.insert(std::string{"--gpu-architecture="} + props.gcnArchName);
  }

  std::string sarg{};
  for (const std::string& s: architectures) {
    sarg += s + " ";
  }
#else
  std::string sarg = std::string("--fmad=false");
#endif
  const char* options[] = {sarg.c_str()};

  REQUIRE(HIPRTC_SUCCESS == hiprtcAddNameExpression(rtcProgram, kernelNameExpression.c_str()));
  REQUIRE(HIPRTC_SUCCESS == hiprtcCompileProgram(rtcProgram, 1, options));

  return rtcProgram;
}

/**
 * @brief Get a typename as a string
 * 
 * @tparam T The typename
 * @return std::string the string representation of T
 */
template <typename T> std::string getTypeName() {
  std::string name, prefix, suffix;

#ifdef __clang__
  name = __PRETTY_FUNCTION__;
  prefix = "std::string HipTest::getTypeName() [T = ";
  suffix = "]";
#elif defined(__GNUC__)
  name = __PRETTY_FUNCTION__;
  prefix = "std::string HipTest::getTypeName() [with T = ";
  suffix = "]";
#elif defined(_MSC_VER)
  name = __FUNCSIG__;
  prefix = "std::string __cdecl HipTest::getTypeName<";
  suffix = ">(void)";
#endif

  return name.substr(prefix.size(), name.find_last_of(suffix) - prefix.size());
}

/**
 * @brief Compiles and launches a kernel using HIP RTC
 * 
 * @tparam Typenames A list of typenames used by the kernel (unused if the kernel is not a template).
 * @tparam Args A list of kernel arguments to be forwarded.
 * @param getKernelName A function wrapper that returns the name of the kernel to launch (check kernels.hh for more info)
 * @param numBlocks 
 * @param numThreads 
 * @param memPerBlock 
 * @param stream 
 * @param packedArgs A list of kernel arguments to be forwarded.
 */
template <typename... Typenames, typename... Args>
void launchRTCKernel(std::string (*getKernelName)(), dim3 numBlocks, dim3 numThreads,
                     std::uint32_t memPerBlock, hipStream_t stream, Args&&... packedArgs) {
  std::string kernelName = (*getKernelName)();
  std::vector<std::string> kernelTypenames{std::string(HipTest::getTypeName<Typenames>())...};
  std::string kernelExpression = reconstructExpression(kernelName, kernelTypenames);

  /* TODO Implement a caching mechanism so that kernels are only compiled once per test */
  hiprtcProgram rtcProgram{compileRTC(kernelName, kernelExpression)};
  std::vector<char> compiledCode{getKernelCode(rtcProgram)};

  const char* loweredName;
  REQUIRE(HIPRTC_SUCCESS ==
          hiprtcGetLoweredName(rtcProgram, kernelExpression.c_str(), &loweredName));

  hipModule_t module;
  hipFunction_t kernelFunction;
  REQUIRE(hipSuccess == hipModuleLoadData(&module, compiledCode.data()));
  REQUIRE(hipSuccess == hipModuleGetFunction(&kernelFunction, module, loweredName));

  std::vector<KernelArgument> args = {
      {reinterpret_cast<const void*>(&packedArgs), sizeof(Args), alignof(Args)}...};

  std::vector<char> alignedArguments{alignArguments(args)};
  size_t argumentsSize{alignedArguments.size()};

  void* config_array[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, alignedArguments.data(),
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, reinterpret_cast<void*>(&argumentsSize),
                          HIP_LAUNCH_PARAM_END};

  REQUIRE(hipSuccess ==
          hipModuleLaunchKernel(kernelFunction, numBlocks.x, numBlocks.y, numBlocks.z, numThreads.x,
                                numThreads.y, numThreads.z, memPerBlock, stream, nullptr,
                                config_array));
  REQUIRE(HIPRTC_SUCCESS == hiprtcDestroyProgram(&rtcProgram));
}

/**
 * @brief Template overload for when numBlocks and numThreads is an integer.
 * 
 */
template <typename... Typenames, typename... Args>
void launchRTCKernel(std::string kernelName, int numBlocks, int numThreads,
                     std::uint32_t memPerBlock, hipStream_t stream, Args&&... packedArgs) {
  launchRTCKernel<Typenames...>(kernelName, dim3(numBlocks), dim3(numThreads), memPerBlock, stream,
                                std::forward<Args>(packedArgs)...);
}

}  // namespace HipTest
