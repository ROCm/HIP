/*
Copyright (c) 2018 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_runtime.h>
#include "hiprtc_internal.hpp"
#include <hip/hiprtc.h>
#include "platform/program.hpp"
#include <algorithm>

namespace hiprtc {
thread_local hiprtcResult g_lastRtcError = HIPRTC_SUCCESS;
}

class ProgramState {
  amd::Monitor lock_;
private:
  static ProgramState* programState_;

  ProgramState() : lock_("Guards program state") {}
  ~ProgramState() {}

  std::unordered_map<amd::Program*,
                     std::pair<std::vector<std::string>, std::vector<std::string>>> progHeaders_;
  std::vector<std::string> nameExpresssion_;
public:
  static ProgramState& instance();
  void createProgramHeaders(amd::Program* program, int numHeaders,
                            const char** headers, const char** headerNames);
  void getProgramHeaders(amd::Program* program, int* numHeaders, char** headers, char ** headerNames);
  uint32_t addNameExpression(const char* name_expression);
};

ProgramState* ProgramState::programState_ = nullptr;

ProgramState& ProgramState::instance() {
  if (programState_ == nullptr) {
    programState_ = new ProgramState;
  }
  return *programState_;
}

void ProgramState::createProgramHeaders(amd::Program* program, int numHeaders,
                                        const char** headers, const char** headerNames) {
  amd::ScopedLock lock(lock_);
  std::vector<std::string> vHeaderNames;
  std::vector<std::string> vHeaders;
  for (auto i = 0; i != numHeaders; ++i) {
    vHeaders.emplace_back(headers[i]);
    vHeaderNames.emplace_back(headerNames[i]);
    progHeaders_[program] = std::make_pair(std::move(vHeaders), std::move(vHeaderNames));
  }
}

void ProgramState::getProgramHeaders(amd::Program* program, int* numHeaders,
                                     char** headers, char ** headerNames) {
  amd::ScopedLock lock(lock_);

  const auto it = progHeaders_.find(program);
  if (it != progHeaders_.cend()) {
    *numHeaders = it->second.first.size();
    *headers  = reinterpret_cast<char*>(it->second.first.data());
    *headerNames = reinterpret_cast<char*>(it->second.second.data());
  }
}


uint32_t ProgramState::addNameExpression(const char* name_expression) {
  amd::ScopedLock lock(lock_);
  nameExpresssion_.emplace_back(name_expression);
  return nameExpresssion_.size();
}


const char* hiprtcGetErrorString(hiprtcResult x) {
  switch (x) {
    case HIPRTC_SUCCESS:
      return "HIPRTC_SUCCESS";
    case HIPRTC_ERROR_OUT_OF_MEMORY:
      return "HIPRTC_ERROR_OUT_OF_MEMORY";
    case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:
      return "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case HIPRTC_ERROR_INVALID_INPUT:
      return "HIPRTC_ERROR_INVALID_INPUT";
    case HIPRTC_ERROR_INVALID_PROGRAM:
      return "HIPRTC_ERROR_INVALID_PROGRAM";
    case HIPRTC_ERROR_INVALID_OPTION:
      return "HIPRTC_ERROR_INVALID_OPTION";
    case HIPRTC_ERROR_COMPILATION:
      return "HIPRTC_ERROR_COMPILATION";
    case HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE:
      return "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
      return "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
      return "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
      return "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case HIPRTC_ERROR_INTERNAL_ERROR:
      return "HIPRTC_ERROR_INTERNAL_ERROR";
    default:
      throw std::logic_error{"Invalid HIPRTC result."};
  };
}

hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog, const char* src, const char* name,
                                 int numHeaders, const char** headers, const char** headerNames) {
  HIPRTC_INIT_API(prog, src, name, numHeaders, headers, headerNames);

  if (prog == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_PROGRAM);
  }
  if (numHeaders < 0) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  if (numHeaders && (headers == nullptr || headerNames == nullptr)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  amd::Program* program = new amd::Program(*hip::getCurrentContext(), src, amd::Program::HIP);
  if (program == NULL) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  if (CL_SUCCESS != program->addDeviceProgram(*hip::getCurrentContext()->devices()[0])) {
    program->release();
    HIPRTC_RETURN(HIPRTC_ERROR_PROGRAM_CREATION_FAILURE);
  }

  ProgramState::instance().createProgramHeaders(program, numHeaders, headers, headerNames);

  *prog = reinterpret_cast<hiprtcProgram>(as_cl(program));

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}


hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions, const char** options) {

  // FIXME[skudchad] Add headers to amd::Program::build and device::Program::build,
  // pass the saved from ProgramState to amd::Program::build
  HIPRTC_INIT_API(prog, numOptions, options);

  amd::Program* program = as_amd(reinterpret_cast<cl_program>(prog));

  std::ostringstream ostrstr;
  std::vector<const char*> oarr(&options[0], &options[numOptions]);
  std::copy(oarr.begin(), oarr.end(), std::ostream_iterator<std::string>(ostrstr, " "));

  std::vector<amd::Device*> devices{hip::getCurrentContext()->devices()[0]};
  if (CL_SUCCESS != program->build(devices, ostrstr.str().c_str(), nullptr, nullptr)) {
    HIPRTC_RETURN(HIPRTC_ERROR_COMPILATION);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog, const char* name_expression) {
  HIPRTC_INIT_API(prog, name_expression);

  if (name_expression == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  amd::Program* program = as_amd(reinterpret_cast<cl_program>(prog));

  uint32_t id = ProgramState::instance().addNameExpression(name_expression);

  const auto var{"__hiprtc_" + std::to_string(id)};
  const auto code{"\nextern \"C\" constexpr auto " + var + " = " + name_expression + ';'};

  program->appendToSource(code.c_str());

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetLoweredName(hiprtcProgram prog, const char* name_expression,
                                  const char** loweredNames) {
  HIPRTC_INIT_API(prog, name_expression, loweredNames);

  if (name_expression == nullptr || loweredNames == nullptr) {
     HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

   HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog) {
  HIPRTC_INIT_API(prog);

  if (prog == NULL) {
     HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  amd::Program* program = as_amd(reinterpret_cast<cl_program>(prog));

  program->release();

   HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetCode(hiprtcProgram prog, char* binaryMem) {
 HIPRTC_INIT_API(prog, binaryMem);


  amd::Program* program = as_amd(reinterpret_cast<cl_program>(prog));
  const device::Program::binary_t& binary =
      program->getDeviceProgram(*hip::getCurrentContext()->devices()[0])->binary();

  ::memcpy(binaryMem, binary.first, binary.second);

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t* binarySizeRet) {

  HIPRTC_INIT_API(prog, binarySizeRet);

  amd::Program* program = as_amd(reinterpret_cast<cl_program>(prog));

  *binarySizeRet =
      program->getDeviceProgram(*hip::getCurrentContext()->devices()[0])->binary().second;

   HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char* dst) {

  HIPRTC_INIT_API(prog, dst);
  amd::Program* program = as_amd(reinterpret_cast<cl_program>(prog));
  const device::Program* devProgram =
      program->getDeviceProgram(*hip::getCurrentContext()->devices()[0]);

  auto log = program->programLog() + devProgram->buildLog().c_str();

  log.copy(dst, log.size());
  dst[log.size()] = '\0';

   HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t* logSizeRet) {

  HIPRTC_INIT_API(prog, logSizeRet);

  amd::Program* program = as_amd(reinterpret_cast<cl_program>(prog));
  const device::Program* devProgram =
      program->getDeviceProgram(*hip::getCurrentContext()->devices()[0]);

  auto log = program->programLog() + devProgram->buildLog().c_str();

  *logSizeRet = log.size() + 1;

   HIPRTC_RETURN(HIPRTC_SUCCESS);
}
