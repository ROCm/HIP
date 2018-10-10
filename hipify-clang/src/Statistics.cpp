#include "Statistics.h"
#include <assert.h>
#include <sstream>
#include <iomanip>


const char *counterNames[NUM_CONV_TYPES] = {
    "version", "init", "device", "mem", "kern", "coord_func", "math_func", "device_func",
    "special_func", "stream", "event", "occupancy", "ctx", "peer", "module",
    "cache", "exec", "external_resource_interop", "graph", "err", "def", "tex", "gl", "graphics",
    "surface", "jit", "d3d9", "d3d10", "d3d11", "vdpau", "egl", "complex",
    "thread", "other", "include", "include_cuda_main_header", "type", "literal",
    "numeric_literal"
};

const char *apiNames[NUM_API_TYPES] = {
    "CUDA Driver API", "CUDA RT API", "CUBLAS API", "CURAND API", "CUDNN API", "CUFFT API", "cuComplex API"
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


/**
 * Print a named stat value to both the terminal and the CSV file.
 */
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

void StatCounter::incrementCounter(const hipCounter& counter, std::string name) {
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

void StatCounter::print(std::ostream* csv, llvm::raw_ostream* printOut, std::string prefix) {
    conditionalPrint(csv, printOut, "\nCUDA ref type;Count\n", "[HIPIFY] info: " + prefix + " refs by type:\n");
    for (int i = 0; i < NUM_CONV_TYPES; i++) {
        if (convTypeCounters[i] > 0) {
            printStat(csv, printOut, counterNames[i], convTypeCounters[i]);
        }
    }

    conditionalPrint(csv, printOut, "\nCUDA API;Count\n", "[HIPIFY] info: " + prefix + " refs by API:\n");
    for (int i = 0; i < NUM_API_TYPES; i++) {
        printStat(csv, printOut, apiNames[i], apiCounters[i]);
    }

    conditionalPrint(csv, printOut, "\nCUDA ref name;Count\n", "[HIPIFY] info: " + prefix + " refs by names:\n");
    for (const auto &it : counters) {
        printStat(csv, printOut, it.first, it.second);
    }
}


Statistics::Statistics(std::string name): fileName(name) {
    // Compute the total bytes/lines in the input file.
    std::ifstream src_file(name, std::ios::binary | std::ios::ate);
    src_file.clear();
    src_file.seekg(0);
    totalLines = (int) std::count(std::istreambuf_iterator<char>(src_file), std::istreambuf_iterator<char>(), '\n');
    totalBytes = (int) src_file.tellg();

    // Mark the start time...
    startTime = chr::steady_clock::now();
};


///////// Counter update routines //////////

void Statistics::incrementCounter(const hipCounter &counter, std::string name) {
    if (counter.unsupported) {
        unsupported.incrementCounter(counter, name);
    } else {
        supported.incrementCounter(counter, name);
    }
}

void Statistics::add(const Statistics &other) {
    supported.add(other.supported);
    unsupported.add(other.unsupported);
    totalBytes += other.totalBytes;
    totalLines += other.totalLines;
    touchedBytes += other.touchedBytes;
}

void Statistics::lineTouched(int lineNumber) {
    touchedLines.insert(lineNumber);
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

   size_t changedLines = touchedLines.size();

   // Total number of (un)supported refs that were converted.
   int supportedSum = supported.getConvSum();
   int unsupportedSum = unsupported.getConvSum();

   printStat(csv, printOut, "CONVERTED refs count", supportedSum);
   printStat(csv, printOut, "UNCONVERTED refs count", unsupportedSum);
   printStat(csv, printOut, "CONVERSION %", 100 - std::lround(double(unsupportedSum * 100) / double(supportedSum + unsupportedSum)));
   printStat(csv, printOut, "REPLACED bytes", touchedBytes);
   printStat(csv, printOut, "TOTAL bytes", totalBytes);
   printStat(csv, printOut, "CHANGED lines of code", changedLines);
   printStat(csv, printOut, "TOTAL lines of code", totalLines);

   if (totalBytes > 0) {
       printStat(csv, printOut, "CODE CHANGED (in bytes) %", std::lround(double(touchedBytes * 100) / double(totalBytes)));
   }

   if (totalLines > 0) {
       printStat(csv, printOut, "CODE CHANGED (in lines) %", std::lround(double(changedLines * 100) / double(totalLines)));
   }

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

    conditionalPrint(csv, printOut, "\nTOTAL statistics:\n", "\n[HIPIFY] info: TOTAL statistics:\n");

    // A file is considered "converted" if we made any changes to it.
    int convertedFiles = 0;
    for (const auto& p : stats) {
        if (!p.second.touchedLines.empty()) {
            convertedFiles++;
        }
    }

    printStat(csv, printOut, "CONVERTED files", convertedFiles);
    printStat(csv, printOut, "PROCESSED files", stats.size());

    globalStats.print(csv, printOut);
}

//// Static state management ////

Statistics Statistics::getAggregate() {
    Statistics globalStats("global");

    for (const auto& p : stats) {
        globalStats.add(p.second);
    }

    return globalStats;
}

Statistics& Statistics::current() {
    assert(Statistics::currentStatistics);
    return *Statistics::currentStatistics;
}

void Statistics::setActive(std::string name) {
    stats.emplace(std::make_pair(name, Statistics{name}));
    Statistics::currentStatistics = &stats.at(name);
}

std::map<std::string, Statistics> Statistics::stats = {};
Statistics* Statistics::currentStatistics = nullptr;
