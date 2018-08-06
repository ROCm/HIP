#pragma once

#include <chrono>
#include <string>
#include <fstream>
#include <llvm/ADT/StringRef.h>
#include <map>
#include <set>
#include <llvm/Support/raw_ostream.h>

namespace chr = std::chrono;

enum ConvTypes {
    CONV_VERSION = 0,
    CONV_INIT,
    CONV_DEVICE,
    CONV_MEM,
    CONV_KERN,
    CONV_COORD_FUNC,
    CONV_MATH_FUNC,
    CONV_DEVICE_FUNC,
    CONV_SPECIAL_FUNC,
    CONV_STREAM,
    CONV_EVENT,
    CONV_OCCUPANCY,
    CONV_CONTEXT,
    CONV_PEER,
    CONV_MODULE,
    CONV_CACHE,
    CONV_EXEC,
    CONV_ERROR,
    CONV_DEF,
    CONV_TEX,
    CONV_GL,
    CONV_GRAPHICS,
    CONV_SURFACE,
    CONV_JIT,
    CONV_D3D9,
    CONV_D3D10,
    CONV_D3D11,
    CONV_VDPAU,
    CONV_EGL,
    CONV_THREAD,
    CONV_OTHER,
    CONV_INCLUDE,
    CONV_INCLUDE_CUDA_MAIN_H,
    CONV_TYPE,
    CONV_LITERAL,
    CONV_NUMERIC_LITERAL,
    CONV_LAST
};
constexpr int NUM_CONV_TYPES = (int) ConvTypes::CONV_LAST;

enum ApiTypes {
    API_DRIVER = 0,
    API_RUNTIME,
    API_BLAS,
    API_RAND,
    API_DNN,
    API_FFT,
    API_LAST
};
constexpr int NUM_API_TYPES = (int) ApiTypes::API_LAST;

// The names of various fields in in the statistics reports.
extern const char *counterNames[NUM_CONV_TYPES];
extern const char *apiNames[NUM_API_TYPES];


struct hipCounter {
    llvm::StringRef hipName;
    ConvTypes type;
    ApiTypes apiType;
    bool unsupported;
};


/**
 * Tracks a set of named counters, as well as counters for each of the type enums defined above.
 */
class StatCounter {
private:
    // Each thing we track is either "supported" or "unsupported"...
    std::map<std::string, int> counters;

    int apiCounters[NUM_API_TYPES] = {};
    int convTypeCounters[NUM_CONV_TYPES] = {};

public:
    void incrementCounter(const hipCounter& counter, std::string name);

    /**
     * Add the counters from `other` onto the counters of this object.
     */
    void add(const StatCounter& other);

    int getConvSum();

    void print(std::ostream* csv, llvm::raw_ostream* printOut, std::string prefix);
};

/**
 * Tracks the statistics for a single input file.
 */
class Statistics {
    StatCounter supported;
    StatCounter unsupported;

    std::string fileName;

    std::set<int> touchedLines = {};
    int touchedBytes = 0;

    int totalLines = 0;
    int totalBytes = 0;

    chr::steady_clock::time_point startTime;
    chr::steady_clock::time_point completionTime;

public:
    Statistics(std::string name);

    void incrementCounter(const hipCounter &counter, std::string name);

    /**
     * Add the counters from `other` onto the counters of this object.
     */
    void add(const Statistics &other);

    void lineTouched(int lineNumber);
    void bytesChanged(int bytes);

    /**
     * Set the completion timestamp to now.
     */
    void markCompletion();

    /////// Output functions ///////

public:
   /**
     * Pretty-print the statistics stored in this object.
     *
     * @param csv Pointer to an output stream for the CSV to write. If null, no CSV is written
     * @param printOut Pointer to an output stream to print human-readable textual stats to. If null, no
     *                 such stats are produced.
     */
    void print(std::ostream* csv, llvm::raw_ostream* printOut, bool skipHeader = false);

    /// Print aggregated statistics for all registered counters.
    static void printAggregate(std::ostream *csv, llvm::raw_ostream* printOut);

    /////// Static nonsense ///////

    // The Statistics for each input file.
    static std::map<std::string, Statistics> stats;

    // The Statistics objects for the currently-being-processed input file.
    static Statistics* currentStatistics;

    /**
     * Aggregate statistics over all entries in `stats` and return the resulting Statistics object.
     */
    static Statistics getAggregate();

    /**
     * Convenient global entry point for updating the "active" Statistics. Since we operate single-threadedly
     * processing one file at a time, this allows us to simply expose the stats for the current file globally,
     * simplifying things.
     */
    static Statistics& current();

    /**
     * Set the active Statistics object to the named one, creating it if necessary, and write the completion
     * timestamp into the currently active one.
     */
    static void setActive(std::string name);
};
