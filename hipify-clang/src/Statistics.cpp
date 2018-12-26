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

#include "Statistics.h"
#include <assert.h>
#include <sstream>
#include <iomanip>
#include "ArgParse.h"

const char *counterNames[NUM_CONV_TYPES] = {
  "error", // CONV_ERROR
  "init", // CONV_INIT
  "version", // CONV_VERSION
  "device", // CONV_DEVICE
  "context", // CONV_CONTEXT
  "module", // CONV_MODULE
  "memory", // CONV_MEMORY
  "addressing", // CONV_ADDRESSING
  "stream", // CONV_STREAM
  "event", // CONV_EVENT
  "external_resource_interop", // CONV_EXT_RES
  "stream_memory", // CONV_STREAM_MEMORY
  "execution", // CONV_EXECUTION
  "graph", // CONV_GRAPH
  "occupancy", // CONV_OCCUPANCY
  "texture", // CONV_TEXTURE
  "surface", // CONV_SURFACE
  "peer", // CONV_PEER
  "graphics", // CONV_GRAPHICS
  "profiler", // CONV_PROFILER
  "openGL", // CONV_OPENGL
  "D3D9", // CONV_D3D9
  "D3D10", // CONV_D3D10
  "D3D11", // CONV_D3D11
  "VDPAU", // CONV_VDPAU
  "EGL", // CONV_EGL
  "thread", // CONV_THREAD
  "complex", // CONV_COMPLEX
  "library", // CONV_LIB_FUNC
  "device_library", // CONV_LIB_DEVICE_FUNC
  "include", // CONV_INCLUDE
  "include_cuda_main_header", // CONV_INCLUDE_CUDA_MAIN_H
  "type", // CONV_TYPE
  "literal", // CONV_LITERAL
  "numeric_literal", // CONV_NUMERIC_LITERAL
  "define" // CONV_DEFINE
};

const char *apiNames[NUM_API_TYPES] = {
  "CUDA Driver API",
  "CUDA RT API",
  "cuComplex API",
  "cuBLAS API",
  "cuRAND API",
  "cuDNN API",
  "cuFFT API",
  "cuSPARSE API"
};

namespace {

template<typename ST, typename ST2>
void conditionalPrint(ST *stream1,
                      ST2* stream2,
                      const std::string& s1,
                      const std::string& s2) {
  if (stream1) {
    *stream1 << s1;
  }
  if (stream2) {
    *stream2 << s2;
  }
}

// Print a named stat value to both the terminal and the CSV file.
template<typename T>
void printStat(std::ostream *csv, llvm::raw_ostream* printOut, const std::string &name, T value) {
  if (printOut) {
    *printOut << "  " << name << ": " << value << "\n";
  }
  if (csv) {
    *csv << name << ";" << value << "\n";
  }
}

} // Anonymous namespace

void StatCounter::incrementCounter(const hipCounter& counter, const std::string& name) {
  counters[name]++;
  apiCounters[(int) counter.apiType]++;
  convTypeCounters[(int) counter.type]++;
}

void StatCounter::add(const StatCounter& other) {
  for (const auto& p : other.counters) {
    counters[p.first] += p.second;
  }
  for (int i = 0; i < NUM_API_TYPES; i++) {
    apiCounters[i] += other.apiCounters[i];
  }
  for (int i = 0; i < NUM_CONV_TYPES; i++) {
    convTypeCounters[i] += other.convTypeCounters[i];
  }
}

int StatCounter::getConvSum() {
  int acc = 0;
  for (const int& i : convTypeCounters) {
    acc += i;
  }
  return acc;
}

void StatCounter::print(std::ostream* csv, llvm::raw_ostream* printOut, const std::string& prefix) {
  for (int i = 0; i < NUM_CONV_TYPES; i++) {
    if (convTypeCounters[i] > 0) {
      conditionalPrint(csv, printOut, "\nCUDA ref type;Count\n", "[HIPIFY] info: " + prefix + " refs by type:\n");
      break;
    }
  }
  for (int i = 0; i < NUM_CONV_TYPES; i++) {
    if (convTypeCounters[i] > 0) {
      printStat(csv, printOut, counterNames[i], convTypeCounters[i]);
    }
  }
  for (int i = 0; i < NUM_API_TYPES; i++) {
    if (apiCounters[i] > 0) {
      conditionalPrint(csv, printOut, "\nCUDA API;Count\n", "[HIPIFY] info: " + prefix + " refs by API:\n");
      break;
    }
  }
  for (int i = 0; i < NUM_API_TYPES; i++) {
    if (apiCounters[i] > 0) {
      printStat(csv, printOut, apiNames[i], apiCounters[i]);
    }
  }
  if (counters.size() > 0) {
    conditionalPrint(csv, printOut, "\nCUDA ref name;Count\n", "[HIPIFY] info: " + prefix + " refs by names:\n");
    for (const auto &it : counters) {
      printStat(csv, printOut, it.first, it.second);
    }
  }
}

Statistics::Statistics(const std::string& name): fileName(name) {
  // Compute the total bytes/lines in the input file.
  std::ifstream src_file(name, std::ios::binary | std::ios::ate);
  src_file.clear();
  src_file.seekg(0);
  totalLines = (unsigned) std::count(std::istreambuf_iterator<char>(src_file), std::istreambuf_iterator<char>(), '\n');
  totalBytes = (int) src_file.tellg();
  if (totalBytes < 0) {
    totalBytes = 0;
  }
  startTime = chr::steady_clock::now();
}

///////// Counter update routines //////////

void Statistics::incrementCounter(const hipCounter &counter, const std::string& name) {
  if ((!TranslateToRoc && (HIP_UNSUPPORTED == (counter.supportDegree & HIP_UNSUPPORTED))) ||
      (TranslateToRoc  && (ROC_UNSUPPORTED == (counter.supportDegree & ROC_UNSUPPORTED)))) {
    unsupported.incrementCounter(counter, name);
  } else {
    supported.incrementCounter(counter, name);
  }
}

void Statistics::add(const Statistics &other) {
  supported.add(other.supported);
  unsupported.add(other.unsupported);
  touchedBytes += other.touchedBytes;
  totalBytes += other.totalBytes;
  touchedLines += other.touchedLines;
  totalLines += other.totalLines;
  if (other.hasErrors && !hasErrors) {
    hasErrors = true;
  }
  if (startTime > other.startTime) {
    startTime = other.startTime;
  }
}

void Statistics::lineTouched(int lineNumber) {
  touchedLinesSet.insert(lineNumber);
  touchedLines = touchedLinesSet.size();
}

void Statistics::bytesChanged(int bytes) {
  touchedBytes += bytes;
}

void Statistics::markCompletion() {
  completionTime = chr::steady_clock::now();
}

///////// Output functions //////////

void Statistics::print(std::ostream* csv, llvm::raw_ostream* printOut, bool skipHeader) {
  if (!skipHeader) {
    std::string str = "file \'" + fileName + "\' statistics:\n";
    conditionalPrint(csv, printOut, "\n" + str, "\n[HIPIFY] info: " + str);
  }
  if (hasErrors || totalBytes <= 0 || totalLines <= 0) {
    std::string str = "\n  ERROR: Statistics is invalid due to failed hipification.\n\n";
    conditionalPrint(csv, printOut, str, str);
  }
  // Total number of (un)supported refs that were converted.
  int supportedSum = supported.getConvSum();
  int unsupportedSum = unsupported.getConvSum();
  int allSum = supportedSum + unsupportedSum;
  printStat(csv, printOut, "CONVERTED refs count", supportedSum);
  printStat(csv, printOut, "UNCONVERTED refs count", unsupportedSum);
  printStat(csv, printOut, "CONVERSION %", 100 - (0 == allSum ? 100 : std::lround(double(unsupportedSum * 100) / double(allSum))));
  printStat(csv, printOut, "REPLACED bytes", touchedBytes);
  printStat(csv, printOut, "TOTAL bytes", totalBytes);
  printStat(csv, printOut, "CHANGED lines of code", touchedLines);
  printStat(csv, printOut, "TOTAL lines of code", totalLines);
  printStat(csv, printOut, "CODE CHANGED (in bytes) %", 0 == totalBytes ? 0 : std::lround(double(touchedBytes * 100) / double(totalBytes)));
  printStat(csv, printOut, "CODE CHANGED (in lines) %", 0 == totalLines ? 0 : std::lround(double(touchedLines * 100) / double(totalLines)));
  typedef std::chrono::duration<double, std::milli> duration;
  duration elapsed = completionTime - startTime;
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << elapsed.count() / 1000;
  printStat(csv, printOut, "TIME ELAPSED s", stream.str());
  supported.print(csv, printOut, "CONVERTED");
  unsupported.print(csv, printOut, "UNCONVERTED");
}

void Statistics::printAggregate(std::ostream *csv, llvm::raw_ostream* printOut) {
  Statistics globalStats = getAggregate();
  // A file is considered "converted" if we made any changes to it.
  int convertedFiles = 0;
  for (const auto& p : stats) {
    if (p.second.touchedLines && p.second.totalBytes &&
        p.second.totalLines && !p.second.hasErrors) {
      convertedFiles++;
    }
  }
  globalStats.markCompletion();
  globalStats.print(csv, printOut);
  std::string str = "TOTAL statistics:";
  conditionalPrint(csv, printOut, "\n" + str + "\n", "\n[HIPIFY] info: " + str + "\n");
  printStat(csv, printOut, "CONVERTED files", convertedFiles);
  printStat(csv, printOut, "PROCESSED files", stats.size());
}

//// Static state management ////

Statistics Statistics::getAggregate() {
  Statistics globalStats("GLOBAL");
  for (const auto& p : stats) {
    globalStats.add(p.second);
  }
  return globalStats;
}

Statistics& Statistics::current() {
  assert(Statistics::currentStatistics);
  return *Statistics::currentStatistics;
}

void Statistics::setActive(const std::string& name) {
  stats.emplace(std::make_pair(name, Statistics{name}));
  Statistics::currentStatistics = &stats.at(name);
}

std::map<std::string, Statistics> Statistics::stats = {};
Statistics* Statistics::currentStatistics = nullptr;
