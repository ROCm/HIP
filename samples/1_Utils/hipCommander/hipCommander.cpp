#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <typeinfo>

#include <hip/hip_runtime.h>
#ifdef __HIP_PLATFORM_HCC__
#include <elf.h>
#include <hsa/hsa.h>
#include <hc.hpp>
#endif

#include <sys/time.h>

#include "ResultDatabase.h"
#include "nullkernel.hip.cpp"

bool g_printedTiming = false;

// Cmdline parms:
int p_device = 0;
const char* p_command = "setstream(1); H2D; NullKernel; D2H;";
const char* p_file = nullptr;
unsigned p_verbose = 0x0;
unsigned p_db = 0x0;
unsigned p_blockingSync = 0x0;

//---
int p_iterations = 1;

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"


#define failed(...)                                                                                \
    printf("error: ");                                                                             \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    abort();


#define HIPCHECK(error)                                                                            \
    {                                                                                              \
        hipError_t localError = error;                                                             \
        if (localError != hipSuccess) {                                                            \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, hipGetErrorString(localError),  \
                   localError, #error, __FILE__, __LINE__, KNRM);                                  \
            failed("API returned error code.");                                                    \
        }                                                                                          \
    }
#define HIPASSERT(condition, msg)                                                                  \
    if (!(condition)) {                                                                            \
        failed("%sassertion %s at %s:%d: %s%s\n", KRED, #condition, __FILE__, __LINE__, msg,       \
               KNRM);                                                                              \
    }


int parseInt(const char* str, int* output) {
    char* next;
    *output = strtol(str, &next, 0);
    return !strlen(next);
}


void printConfig() {
    hipDeviceProp_t props;
    HIPCHECK(hipGetDeviceProperties(&props, p_device));

    printf("Device:%s Mem=%.1fGB #CUs=%d Freq=%.0fMhz\n", props.name,
           props.totalGlobalMem / 1024.0 / 1024.0 / 1024.0, props.multiProcessorCount,
           props.clockRate / 1000.0);
}


void help() {
    printf("Usage: hipBusBandwidth [OPTIONS]\n");
    printf("  --file, -f               : Read string of commands from file\n");
    printf("  --command, -c            : String specifying commands to run.\n");
    printf("  --iterations, -i         : Number of copy iterations to run.\n");
    printf("  --device, -d             : Device ID to use (0..numDevices).\n");
    printf(
        "  --verbose, -v            : Verbose printing of status.  Fore more info, combine with "
        "HIP_TRACE_API on ROCm\n");
};


int parseStandardArguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];

        if (!strcmp(arg, " ")) {
            // skip NULL args.
        } else if (!strcmp(arg, "--iterations") || (!strcmp(arg, "-i"))) {
            if (++i >= argc || !parseInt(argv[i], &p_iterations)) {
                failed("Bad --iterations argument");
            }

        } else if (!strcmp(arg, "--device") || (!strcmp(arg, "-d"))) {
            if (++i >= argc || !parseInt(argv[i], &p_device)) {
                failed("Bad --device argument");
            }

        } else if (!strcmp(arg, "--file") || (!strcmp(arg, "-f"))) {
            if (++i >= argc) {
                failed("Bad --file argument");
            } else {
                p_file = argv[i];
            }

        } else if (!strcmp(arg, "--commands") || (!strcmp(arg, "-c"))) {
            if (++i >= argc) {
                failed("Bad --commands argument");
            } else {
                p_command = argv[i];
            }

        } else if (!strcmp(arg, "--verbose") || (!strcmp(arg, "-v"))) {
            p_verbose = 1;

        } else if (!strcmp(arg, "--blockingSync") || (!strcmp(arg, "-B"))) {
            p_blockingSync = 1;


        } else if (!strcmp(arg, "--help") || (!strcmp(arg, "-h"))) {
            help();
            exit(EXIT_SUCCESS);

        } else {
            failed("Bad argument '%s'", arg);
        }
    }

    return 0;
};

// Returns the current system time in microseconds
inline long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}


class Command;


//=================================================================================================
// A stream of commands , specified as a string.
class CommandStream {
   public:
    // State that is inherited by sub-blocks:
    struct CommandStreamState {
        hipStream_t _currentStream;
        std::vector<hipStream_t> _streams;
        vector<CommandStream*> _subBlocks;
    };

   public:
    CommandStream(std::string commandStreamString, int iterations);
    ~CommandStream();

    hipStream_t currentStream() const { return _state._currentStream; };

    void print(const std::string& indent = "") const;
    void printBrief(std::ostream& s = std::cout) const;
    void run();
    void recordTime();
    void printTiming(int iterations = 0);

    CommandStream* currentCommandStream() {
        return _parseInSubBlock ? _state._subBlocks.back() : this;
    };

    void enterSubBlock(CommandStream* commandStream) {
        _parseInSubBlock = true;
        _state._subBlocks.push_back(commandStream);
    };

    void exitSubBlock() { _parseInSubBlock = false; };


    void setParent(CommandStream* parentCmdStream) {
        _parentCommandStream = parentCmdStream;
        _state = parentCmdStream->_state;
    };
    CommandStream* getParent() { return _parentCommandStream; };

    void setStream(int streamIndex);

    CommandStreamState& getState() { return _state; };

   private:
    static void tokenize(const std::string& s, char delim, std::vector<std::string>& tokens);
    void parse(const std::string fullCmd);

   protected:
    CommandStreamState _state;

   private:
    // List of commands to run in this stream:
    std::vector<Command*> _commands;


    // Number of iterations to run the command loop
    int _iterations;


    // Us to run the the command-stream.  Only valid after run is called.
    long long _startTime;
    double _elapsedUs;

    // Track nested loop of command streams:
    CommandStream* _parentCommandStream;

    // Track if we are parsing commands in the subblock.
    bool _parseInSubBlock;
};


//=================================================================================================
class Command {
   public:
    // @p minArgs : Minimum arguments for command.  -1 = don't check.
    // @p maxArgs : Minimum arguments for command.  0 means min=max, ie exact #arguments expected.
    // -1 = don't check max.
    Command(CommandStream* cmdStream, const std::vector<std::string>& args, int minArgs = 0,
            int maxArgs = 0)
        : _commandStream(cmdStream), _args(args) {
        int numArgs = args.size() - 1;

        if ((minArgs != -1) && (numArgs < minArgs)) {
            // TODO - print full command here.
            failed("Not enough arguments for command %s.  (Expected %d, got %d)", args[0].c_str(),
                   minArgs, numArgs);
        }

        // Check for an exact number of arguments:
        if (maxArgs == 0) {
            maxArgs = minArgs;
        }
        if ((maxArgs != -1) && (numArgs > maxArgs)) {
            failed("Too many arguments for command %s.  (Expected %d, got %d)", args[0].c_str(),
                   maxArgs, numArgs);
        }
    };

    void printBrief(std::ostream& s = std::cout) const { s << _args[0]; }

    virtual ~Command(){};

    virtual void print(const std::string& indent = "") const {
        std::cout << indent << "[";
        std::for_each(_args.begin(), _args.end(), [](const std::string& s) { std::cout << s; });
        std::cout << "]";
    };


    virtual void run() = 0;

   protected:
    int readIntArg(int argIndex, const std::string& argName) {
        // TODO - catch references to non-existant arguments here.
        int argVal;
        try {
            argVal = std::stoi(_args[argIndex]);
        } catch (std::invalid_argument) {
            failed("Command %s has bad %s argument ('%s')", _args[0].c_str(), argName.c_str(),
                   _args[argIndex].c_str());
        }
        return argVal;
    }

   protected:
    CommandStream* _commandStream;
    std::vector<std::string> _args;
};


#define FILENAME "nullkernel.hsaco"
#define KERNEL_NAME "NullKernel"


#ifdef __HIP_PLATFORM_HCC__
//=================================================================================================
// Use Aql to launch the NULL kernel.
class AqlKernelCommand : public Command {
   public:
    AqlKernelCommand(CommandStream* cmdStream, const std::vector<std::string> args)
        : Command(cmdStream, args) {
        hc::accelerator_view* av;
        HIPCHECK(hipHccGetAcceleratorView(cmdStream->currentStream(), &av));

        hc::accelerator acc = av->get_accelerator();

        hsa_region_t systemRegion = *(hsa_region_t*)acc.get_hsa_am_system_region();

        _hsaAgent = *(hsa_agent_t*)acc.get_hsa_agent();

        std::ifstream file(FILENAME, std::ios::binary | std::ios::ate);
        std::streamsize fsize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(fsize);
        if (file.read(buffer.data(), fsize)) {
            uint64_t elfSize = ElfSize(&buffer[0]);

            assert(fsize == elfSize);

            // TODO - replace module load code with explicit module load and unload.

            hipModule_t module;
            HIPCHECK(hipModuleLoadData(&module, &buffer[0]));
            HIPCHECK(hipModuleGetFunction(&_function, module, KERNEL_NAME));

        } else {
            failed("could not open code object '%s'\n", FILENAME);
        }
    };

    ~AqlKernelCommand(){};

    void run() override {
#define LEN 64
        uint32_t len = LEN;
        uint32_t one = 1;

        float* Ad = NULL;

        size_t argSize = 36;
        char argBuffer[argSize];
        *(uint32_t*)(&argBuffer[0]) = len;
        *(uint32_t*)(&argBuffer[4]) = one;
        *(uint32_t*)(&argBuffer[8]) = one;
        *(uint32_t*)(&argBuffer[12]) = len;
        *(uint32_t*)(&argBuffer[16]) = one;
        *(uint32_t*)(&argBuffer[20]) = one;
        *(float**)(&argBuffer[24]) = Ad;  // Ad pointer argument


        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &argBuffer[0],
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize, HIP_LAUNCH_PARAM_END};

        hipModuleLaunchKernel(_function, len, 1, 1, LEN, 1, 1, 0, 0, NULL, (void**)&config);
    };


   public:
    hsa_queue_t _hsaQueue;
    hsa_agent_t _hsaAgent;

    hipFunction_t _function;

   private:
    static uint64_t ElfSize(const void* emi) {
        const Elf64_Ehdr* ehdr = (const Elf64_Ehdr*)emi;
        const Elf64_Shdr* shdr = (const Elf64_Shdr*)((char*)emi + ehdr->e_shoff);

        uint64_t max_offset = ehdr->e_shoff;
        uint64_t total_size = max_offset + ehdr->e_shentsize * ehdr->e_shnum;

        for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
            uint64_t cur_offset = static_cast<uint64_t>(shdr[i].sh_offset);
            if (max_offset < cur_offset) {
                max_offset = cur_offset;
                total_size = max_offset;
                if (SHT_NOBITS != shdr[i].sh_type) {
                    total_size += static_cast<uint64_t>(shdr[i].sh_size);
                }
            }
        }
        return total_size;
    }
};
#endif

//=================================================================================================
// HCC optimizes away fully NULL kernel calls, so run one that is nearly null:
class ModuleKernelCommand : public Command {
   public:
    ModuleKernelCommand(CommandStream* cmdStream, const std::vector<std::string> args)
        : Command(cmdStream, args), _stream(cmdStream->currentStream()) {
        hipModule_t module;
        HIPCHECK(hipModuleLoad(&module, FILENAME));
        HIPCHECK(hipModuleGetFunction(&_function, module, KERNEL_NAME));
    };
    ~ModuleKernelCommand(){};

    void run() override {
#define LEN 64
        uint32_t len = LEN;
        uint32_t one = 1;

        float* Ad = NULL;

        size_t argSize = 36;
        char argBuffer[argSize];
        *(uint32_t*)(&argBuffer[0]) = len;
        *(uint32_t*)(&argBuffer[4]) = one;
        *(uint32_t*)(&argBuffer[8]) = one;
        *(uint32_t*)(&argBuffer[12]) = len;
        *(uint32_t*)(&argBuffer[16]) = one;
        *(uint32_t*)(&argBuffer[20]) = one;
        *(float**)(&argBuffer[24]) = Ad;  // Ad pointer argument


        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &argBuffer[0],
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize, HIP_LAUNCH_PARAM_END};

        hipModuleLaunchKernel(_function, len, 1, 1, LEN, 1, 1, 0, 0, NULL, (void**)&config);
    };


   public:
    hipFunction_t _function;
    hipStream_t _stream;
};


class KernelCommand : public Command {
   public:
    enum Type { Null, VectorAdd };
    KernelCommand(CommandStream* cmdStream, const std::vector<std::string> args, Type kind)
        : Command(cmdStream, args), _kind(kind), _stream(cmdStream->currentStream()){};
    ~KernelCommand(){};


    void run() override {
        static const int gridX = 64;
        static const int groupX = 64;

        switch (_kind) {
            case Null:
                hipLaunchKernelGGL(NullKernel, dim3(gridX / groupX), dim3(gridX), 0, _stream, nullptr);
                break;
            case VectorAdd:
                assert(0);  // TODO
                break;
        };
    }

   private:
    Type _kind;
    hipStream_t _stream;
};


#ifdef __HIP_PLATFORM_HCC__
//=================================================================================================
class PfeCommand : public Command {
   public:
    PfeCommand(CommandStream* cmdStream, const std::vector<std::string> args,
               hipStream_t stream = 0)
        : Command(cmdStream, args) {
        HIPCHECK(hipHccGetAcceleratorView(stream, &_av));
    }

    ~PfeCommand() {}


    void run() override {
        static const int gridX = 64;
        static const int groupX = 64;
        auto cf = hc::parallel_for_each(*_av, hc::extent<1>(gridX).tile(groupX),
                                        [=](hc::index<1>& idx) __HC__ {});
    }

   private:
    hc::accelerator_view* _av;
};
#endif


//=================================================================================================
class CopyCommand : public Command {
    enum MemType { PinnedHost, UnpinnedHost, Device };

   public:
    CopyCommand(CommandStream* cmdStream, const std::vector<std::string>& args, hipMemcpyKind kind,
                bool isAsync, bool isPinnedHost);

    ~CopyCommand() {
        if (_dst) {
            dealloc(_dst, _dstType);
            _dst = NULL;
        };

        if (_src) {
            dealloc(_src, _srcType);
            _src = NULL;
        }
    }


    void run() override {
        if (_isAsync) {
            HIPCHECK(hipMemcpyAsync(_dst, _src, _sizeBytes, _kind, _stream));
        } else {
            HIPCHECK(hipMemcpy(_dst, _src, _sizeBytes, _kind));
        }
    };

   private:
    void* alloc(size_t size, MemType memType) {
        void* p;
        if (memType == Device) {
            HIPCHECK(hipMalloc(&p, size));

        } else if (memType == PinnedHost) {
            HIPCHECK(hipHostMalloc(&p, size));

        } else if (memType == UnpinnedHost) {
            p = (char*)malloc(size);
            HIPASSERT(p, "malloc failed");

        } else {
            HIPASSERT(0, "unsupported memType");
        }

        return p;
    };


    void dealloc(void* p, MemType memType) {
        if (memType == Device) {
            HIPCHECK(hipFree(p));
        } else if (memType == PinnedHost) {
            HIPCHECK(hipHostFree(p));
        } else if (memType == UnpinnedHost) {
            free(p);
        } else {
            HIPASSERT(0, "unsupported memType");
        }
    }


   private:
    bool _isAsync;
    hipStream_t _stream;
    hipMemcpyKind _kind;

    size_t _sizeBytes;
    void* _dst;
    MemType _dstType;

    void* _src;
    MemType _srcType;
};


//=================================================================================================
class DeviceSyncCommand : public Command {
   public:
    DeviceSyncCommand(CommandStream* cmdStream, const std::vector<std::string>& args)
        : Command(cmdStream, args){};

    void run() override { HIPCHECK(hipDeviceSynchronize()); };
};


//=================================================================================================
class StreamSyncCommand : public Command {
   public:
    StreamSyncCommand(CommandStream* cmdStream, const std::vector<std::string>& args)
        : Command(cmdStream, args), _stream(cmdStream->currentStream()){};

    const char* help() { return "synchronizes the current stream"; };


    void run() override { HIPCHECK(hipStreamSynchronize(_stream)); };

   private:
    hipStream_t _stream;
};


//=================================================================================================

//=================================================================================================
class LoopCommand : public Command {
   public:
    LoopCommand(CommandStream* parentCmdStream, const std::vector<std::string>& args)
        : Command(parentCmdStream, args, 1) {
        int loopCnt;
        try {
            loopCnt = std::stoi(args[1]);
        } catch (std::invalid_argument) {
            failed("bad LOOP_CNT=%s", args[1].c_str());
        }

        _commandStream = new CommandStream("", loopCnt);
        _commandStream->setParent(parentCmdStream);
        parentCmdStream->enterSubBlock(_commandStream);
    };


    void print(const std::string& indent = "") const override {
        Command::print();
        _commandStream->print(indent + "  ");
    };

    void run() override { _commandStream->run(); };
};


//=================================================================================================
class EndBlockCommand : public Command {
   public:
    EndBlockCommand(CommandStream* blockCmdStream, CommandStream* parentCmdStream,
                    const std::vector<std::string>& args)
        : Command(parentCmdStream, args, 0, 1), _blockCmdStream(blockCmdStream), _printTiming(0) {
        int argCnt = args.size() - 1;
        if (argCnt >= 1) {
            _printTiming = readIntArg(1, "PRINT_TIMING");
        }

        if (parentCmdStream == nullptr) {
            failed("%s without corresponding command to start block", args[0].c_str());
        }
        parentCmdStream->exitSubBlock();
    };

    void run() override {
        if (_printTiming) {
            _blockCmdStream->printTiming();
        }
    };

   private:
    CommandStream* _blockCmdStream;

    // print the stream when loop exits.
    int _printTiming;
};


//=================================================================================================
class SetStreamCommand : public Command {
   public:
    SetStreamCommand(CommandStream* cmdStream, const std::vector<std::string>& args)
        : Command(cmdStream, args, 1) {
        int streamIndex = readIntArg(1, "STREAM_INDEX");

        cmdStream->setStream(streamIndex);
    };

    void run() override{};
};


//=================================================================================================
class PrintTimingCommand : public Command {
   public:
    PrintTimingCommand(CommandStream* cmdStream, const std::vector<std::string>& args)
        : Command(cmdStream, args, 1) {
        _iterations = readIntArg(1, "ITERATIONS");
    };

    void run() override { _commandStream->printTiming(_iterations); };

   private:
    int _iterations;
};


//=================================================================================================
CopyCommand::CopyCommand(CommandStream* cmdStream, const std::vector<std::string>& args,
                         hipMemcpyKind kind, bool isAsync, bool isPinnedHost)
    : Command(cmdStream, args),
      _isAsync(isAsync),
      _kind(kind),
      _stream(cmdStream->currentStream()) {
    switch (kind) {
        case hipMemcpyDeviceToHost:
            _srcType = Device;
            _dstType = isPinnedHost ? PinnedHost : UnpinnedHost;
            break;
        case hipMemcpyHostToDevice:
            _srcType = isPinnedHost ? PinnedHost : UnpinnedHost;
            _dstType = Device;
            break;
        default:
            HIPASSERT(0, "Unknown hipMemcpyKind");
    };

    _sizeBytes = 64;  // TODO, support reading from arg.

    _dst = alloc(_sizeBytes, _dstType);
    _src = alloc(_sizeBytes, _srcType);
};


//=================================================================================================
//=================================================================================================
// Implementations:
//=================================================================================================

//=================================================================================================
CommandStream::CommandStream(std::string commandStreamString, int iterations)
    : _iterations(iterations),
      _startTime(0),
      _elapsedUs(0.0),
      _parentCommandStream(nullptr),
      _parseInSubBlock(false) {
    std::vector<std::string> tokens;
    tokenize(commandStreamString, ';', tokens);


    std::for_each(tokens.begin(), tokens.end(), [&](const std::string s) { this->parse(s); });

    setStream(0);
}


CommandStream::~CommandStream() {
    std::for_each(_state._streams.begin(), _state._streams.end(), [&](hipStream_t s) {
        if (s) {
            HIPCHECK(hipStreamDestroy(s));
        }
    });

    std::for_each(_commands.begin(), _commands.end(), [&](Command* c) { delete c; });
}


void CommandStream::setStream(int streamIndex) {
    if (streamIndex >= _state._streams.size()) {
        _state._streams.resize(streamIndex + 1);
    }

    if (streamIndex && (_state._streams[streamIndex] == nullptr)) {
        // Create new stream:
        hipStream_t stream;
        HIPCHECK(hipStreamCreate(&stream));
        _state._streams[streamIndex] = stream;
        _state._currentStream = stream;
    } else {
        // Use existing stream:

        _state._currentStream = _state._streams[streamIndex];
    }
}


void CommandStream::tokenize(const std::string& s, char delim, std::vector<std::string>& tokens) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (getline(ss, item, delim)) {
        item.erase(std::remove(item.begin(), item.end(), ' '), item.end());  // remove whitespace.
        tokens.push_back(item);
    }
}

void trim(std::string* s) {
    // trim whitespace from begin and end:
    const char* t = "\t\n\r\f\v";
    s->erase(0, s->find_first_not_of(t));
    s->erase(s->find_last_not_of(t) + 1);
}

void ltrim(std::string* s) {
    // trim whitespace from begin and end:
    const char* t = "\t\n\r\f\v";
    s->erase(0, s->find_first_not_of(t));
}

void CommandStream::parse(std::string fullCmd) {
    // convert to lower-case:
    std::transform(fullCmd.begin(), fullCmd.end(), fullCmd.begin(), ::tolower);
    trim(&fullCmd);

    if (p_db) {
        printf("parse: <%s>\n", fullCmd.c_str());
    }


    std::string c;
    std::vector<std::string> args;
    size_t leftParenZ = fullCmd.find_first_of('(');
    if (leftParenZ == string::npos) {
        c = fullCmd;
        args.push_back(c);
    } else {
        c = fullCmd.substr(0, leftParenZ);
        args.push_back(c);
        size_t rightParenZ = fullCmd.find_first_of(')', leftParenZ);
        std::string argStr = fullCmd.substr(leftParenZ + 1, rightParenZ - leftParenZ - 1);
        // printf ("c=%s argstr='%s' leftParenZ=%zu rightParenZ=%zu\n", c.c_str(), argStr.c_str(),
        // leftParenZ, rightParenZ);
        tokenize(argStr, ',', args);
    }


    if ((args.size() == 0) || (fullCmd.c_str()[0] == '#')) {
        if (p_db) {
            printf("  skip comment\n");
        }
        return;
    }


    Command* cmd = NULL;
    CommandStream* cmdStream = currentCommandStream();

    if (c == "h2d") {
        cmd = new CopyCommand(cmdStream, args, hipMemcpyHostToDevice, true /*isAsync*/,
                              true /*isPinned*/);
        //= h2d
        //= Performs an async host-to-device copy of array A_h to A_d.
        //= The size of these arrays may be set with the datasize command.

    } else if (c == "d2h") {
        cmd = new CopyCommand(cmdStream, args, hipMemcpyDeviceToHost, true /*isAsync*/,
                              true /*isPinned*/);
        //= d2h
        //= Performs an async device-to-host copy of array A_d to A_h.
        //= The size of these arrays may be set with the datasize command.

    } else if (c == "modulekernel") {
        cmd = new ModuleKernelCommand(cmdStream, args);

    } else if (c == "nullkernel") {
        cmd = new KernelCommand(cmdStream, args, KernelCommand::Null);
        //= nullkernel
        //= Dispatches a null kernel to the device.

    } else if (c == "vectoraddkernel") {
        cmd = new KernelCommand(cmdStream, args, KernelCommand::VectorAdd);

#ifdef __HIP_PLATFORM_HCC__
    } else if (c == "nullpfe") {
        cmd = new PfeCommand(cmdStream, args);

    } else if (c == "aqlkernel") {
        cmd = new AqlKernelCommand(cmdStream, args);
#endif

    } else if (c == "devicesync") {
        cmd = new DeviceSyncCommand(cmdStream, args);

    } else if (c == "streamsync") {
        //= streamsync
        //= Execute hipStreamSynchronize.
        //= This will cause the host thread to wait until the current stream
        //= completes all pending operations.
        cmd = new StreamSyncCommand(cmdStream, args);

    } else if (c == "setstream") {
        //= setstream(STREAM_INDEX);
        //= Set current stream used by subsequent commands.
        //= STREAM_INDEX is index starting from 0...N.
        //= This function will create new stream on first call to setstream or re-use previous
        //= stream if setstream has already been called with STREAM_INDEX.
        //= STREAM_INDEX=0 will use the default "null" stream associated with the device, and will
        //not create a new stream. =  The default stream has special, conservative synchronization
        //properties.

        cmd = new SetStreamCommand(cmdStream, args);

    } else if (c == "printtiming") {
        cmd = new PrintTimingCommand(cmdStream, args);

    } else if (c == "loop") {
        //= loop(LOOP_CNT)
        //= Loop over next set of commands (until 'endloop' command) for LOOP_CNT iterations.
        //= Loops can be nested.

        cmd = new LoopCommand(cmdStream, args);

    } else if (c == "endloop") {
        //= endloop
        //= End a looped sequence. Must be paired with a preceding loop command.
        //= Command between the `loop` and `endloop` must be executed

        CommandStream* parentCmdStream = cmdStream->getParent();
        cmd = new EndBlockCommand(cmdStream, parentCmdStream, args);
        cmdStream = parentCmdStream;

    } else {
        std::cerr << "error: Bad command '" << fullCmd << "\n";
        HIPASSERT(0, "bad command in command-stream");
    }

    if (cmd) {
        cmdStream->_commands.push_back(cmd);
    }
}


void CommandStream::print(const std::string& indent) const {
    for (auto cmdI = _commands.begin(); cmdI != _commands.end(); cmdI++) {
        (*cmdI)->print(indent);
    };
}


void CommandStream::printBrief(std::ostream& s) const {
    for (auto cmdI = _commands.begin(); cmdI != _commands.end(); cmdI++) {
        (*cmdI)->printBrief(s);
        s << ";";
    };
}

void CommandStream::run() {
    _startTime = get_time();
    for (int i = 0; i < _iterations; i++) {
        for (auto cmdI = _commands.begin(); cmdI != _commands.end(); cmdI++) {
            if (p_verbose) {
                (*cmdI)->print();
            }
            (*cmdI)->run();
        }
    }

    // Record time, if not already stored.  (an earlier printTime command will also store the time)
    recordTime();
};

void CommandStream::recordTime() {
    if (_elapsedUs == 0.0) {
        auto stopTime = get_time();
        _elapsedUs = stopTime - _startTime;
    }
}


void CommandStream::printTiming(int iterations) {
    if ((_state._subBlocks.size() == 1) && (_commands.size() == 1)) {
        // printf ("print just the loop\n");
        _state._subBlocks.front()->printTiming(iterations);
    } else {
        g_printedTiming = true;

        recordTime();
        if (iterations == 0) {
            iterations = _iterations;
        }
        std::cout << "command<";
        printBrief(std::cout);
        std::cout << ">,";
        printf("    iterations,%d,   total_time,%6.3f,  time/iteration,%6.3f\n", iterations,
               _elapsedUs, _elapsedUs / iterations);
    }
};


//=================================================================================================
int main(int argc, char* argv[]) {
    parseStandardArguments(argc, argv);

    printConfig();

    CommandStream* cs;

    if (p_blockingSync) {
#ifdef __HIP_PLATFORM_HCC__
        printf("setting BlockingSync for AMD\n");
        setenv("HIP_BLOCKING_SYNC", "1", 1);

#endif
#ifdef __HIP_PLATFORM_NVCC__
        printf("setting cudaDeviceBlockingSync\n");
        HIPCHECK(hipSetDeviceFlags(cudaDeviceBlockingSync));
#endif
    };


    if (p_file) {
        // TODO - catch exception on file IO here:
        std::ifstream file(p_file);
        std::string str;
        std::string file_contents;
        while (std::getline(file, str)) {
            file_contents += str;
        }

        cs = new CommandStream(file_contents, p_iterations);

    } else {
        cs = new CommandStream(p_command, p_iterations);
    }

    cs->print();
    printf("------\n");

    cs->run();
    if (!g_printedTiming) {
        cs->printTiming();
    }

    delete cs;
}


// TODO - add error checking for arguments.
