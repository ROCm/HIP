#ifndef COMMON_H
#define COMMON_H

bool debug_mode;
#define WHEN_DEBUG(xxx)                                                 \
    do {                                                                \
        if (debug_mode) {                                               \
            xxx;                                                        \
        }                                                               \
    } while (false)

static bool parse_options(int argc, char *argv[]) {
    for (int ii = 1; ii != argc; ++ii) {
        char *str = argv[ii];
        if (str[0] != '-')
            return false;
        if (str[2])
            return false;
        switch (str[1]) {
        case 'd':
            debug_mode = true;
            break;
        default:
            return false;
            break;
        }
    }

    return true;
}

static bool set_flags(int argc, char *argv[]) {
    debug_mode = false;

    if (!parse_options(argc, argv)) {
        std::cout << "invalid command-line arguments" << std::endl;
        return false;
    }

    return true;
}

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"

#define failed(...)                                                     \
    printf("%serror: ", KRED);                                          \
    printf(__VA_ARGS__);                                                \
    printf("\n");                                                       \
    printf("error: TEST FAILED\n%s", KNRM);                             \
    abort();

#define test_passed(test_name) printf("%s %s  PASSED!%s\n",             \
                                      KGRN, test_name, KNRM);

#define runTest(name)                                                   \
    do {                                                                \
        uint status = name();                                           \
        if (status != 0) {                                              \
            std::cout << "status: " << status << std::endl;             \
            failed(#name);                                              \
        }                                                               \
        hipDeviceReset();                                               \
    } while (false);                                                    \

#define HIPCHECK(error)                                                 \
    do {                                                                \
        hipError_t localError = error;                                  \
        if ((localError != hipSuccess) &&                               \
            (localError != hipErrorPeerAccessAlreadyEnabled)) {         \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n",            \
                   KRED, hipGetErrorString(localError),                 \
                   localError, #error, __FILE__, __LINE__, KNRM);       \
            failed("API returned error code.");                         \
        }                                                               \
    } while (false);

#define TEST_SERVICE 42

extern "C" __device__ HIP_vector_base<long, 2>::Native_vec_
__ockl_hostcall_internal(void *buffer, uint service_id,
                         ulong arg0, ulong arg1, ulong arg2, ulong arg3,
                         ulong arg4, ulong arg5, ulong arg6, ulong arg7);

extern "C" __device__ HIP_vector_base<long, 2>::Native_vec_
__ockl_hostcall_preview(uint service_id,
                        ulong arg0, ulong arg1, ulong arg2, ulong arg3,
                        ulong arg4, ulong arg5, ulong arg6, ulong arg7);

extern "C" __device__ HIP_vector_base<long, 2>::Native_vec_
__ockl_call_host_function(ulong fptr,
                          ulong arg0, ulong arg1, ulong arg2, ulong arg3,
                          ulong arg4, ulong arg5, ulong arg6);

enum {
    SIGNAL_INIT = UINT64_MAX,
    SIGNAL_DONE = UINT64_MAX - 1
};

typedef struct {
    ulong next;
    ulong activemask;
    uint service;
    uint control;
} header_t;

enum {
    PAYLOAD_ALIGNMENT = 64
};

typedef struct {
    // 64 slots of 8 ulongs each
    ulong slots[64][8];
} payload_t;

typedef struct {
    header_t *headers;
    payload_t *payloads;
    hsa_signal_t doorbell;
    ulong free_stack;
    ulong ready_stack;
    uint index_size;
} hostcall_buffer_t;

static void
work_done(hostcall_buffer_t *buffer)
{
    hsa_signal_store_release(buffer->doorbell, SIGNAL_DONE);
}

static ulong get_ptr_tag(ulong ptr, uint index_size) {
    return ptr >> index_size;
}

static ulong get_ptr_index(ulong ptr, uint index_size) {
    ulong mask = 1;
    mask = (mask << index_size) - 1;
    return ptr & mask;
}

static header_t* get_header(hostcall_buffer_t* buffer, ulong ptr) {
    return buffer->headers + get_ptr_index(ptr, buffer->index_size);
}

static payload_t* get_payload(hostcall_buffer_t* buffer, ulong ptr) {
    return buffer->payloads + get_ptr_index(ptr, buffer->index_size);
}

static uint get_ready_flag(uint control) { return control & 1; }

static uint reset_ready_flag(uint control) { return control & ~1; }

static uint set_ready_flag(uint control) { return control | 1; }

static ulong inc_ptr_tag(ulong ptr, uint index_size) {
    ulong index = get_ptr_index(ptr, index_size);
    ulong tag = get_ptr_tag(ptr, index_size);
    return ((tag + 1) << index_size) | index;
}

static ulong set_tag(ulong ptr, ulong value, uint index_size)
{
    ulong index = get_ptr_index(ptr, index_size);
    return (value <<= index_size) | index;
}

static uint align_to(uint value, uint alignment) {
    if (value % alignment == 0) return value;
    return value - (value % alignment) + alignment;
}

static uint
get_header_start() {
    return align_to(sizeof(hostcall_buffer_t), alignof(header_t));
}

static uint
get_payload_start(uint num_packets) {
    uint header_start = get_header_start();
    uint header_end = header_start + sizeof(header_t) * num_packets;
    return align_to(header_end, PAYLOAD_ALIGNMENT);
}

static uint
get_buffer_size(uint num_packets) {
    uint payload_start = get_payload_start(num_packets);
    uint payload_size = sizeof(payload_t) * num_packets;
    return payload_start + payload_size;
}

static hostcall_buffer_t* createBuffer(uint num_packets, hsa_signal_t signal) {
    if (num_packets == 0) return nullptr;

    void* buffer;
    size_t buffer_size = get_buffer_size(num_packets);
    WHEN_DEBUG(std::cout << "buffer_t size: " << sizeof(hostcall_buffer_t) << std::endl);
    WHEN_DEBUG(std::cout << "header alignment: " << alignof(header_t) << std::endl);
    WHEN_DEBUG(std::cout << "header start: " << get_header_start() << std::endl);
    WHEN_DEBUG(std::cout << "payload alignment: " << PAYLOAD_ALIGNMENT << std::endl);
    WHEN_DEBUG(std::cout << "payload start: " << get_payload_start(num_packets) << std::endl);
    WHEN_DEBUG(std::cout << "buffer size: " << buffer_size << std::endl);
    HIPCHECK(hipHostMalloc(&buffer, buffer_size, hipHostMallocCoherent));
    memset(buffer, 0, buffer_size);

    hostcall_buffer_t* retval = (hostcall_buffer_t*)buffer;
    retval->doorbell = signal;

    retval->headers = (header_t*)((uchar*)retval + get_header_start());
    retval->payloads = (payload_t*)((uchar*)retval + get_payload_start(num_packets));

    uint index_size = 1;
    if (num_packets > 2)
        index_size = 32 - __builtin_clz(num_packets);
    WHEN_DEBUG(std::cout << "num packets: " << num_packets
               << "; index size: " << index_size << std::endl);
    retval->index_size = index_size;
    retval->headers[0].next = 0;

    ulong next = 0;
    next = inc_ptr_tag(next, index_size);
    for (uint ii = 1; ii != num_packets; ++ii) {
        retval->headers[ii].next = next;
        next = ii;
    }
    retval->free_stack = next;
    WHEN_DEBUG(std::cout << "free stack: " << retval->free_stack << std::endl);
    WHEN_DEBUG(std::cout << "next free: " << retval->headers[0].next << std::endl);

    retval->ready_stack = 0;

    return retval;
}

static bool
timeout(hipEvent_t mark, uint millisecs) {
    using std::chrono::system_clock;
    system_clock::time_point start = system_clock::now();
    while (hipEventQuery(mark) != hipSuccess) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        system_clock::time_point now = system_clock::now();
        if (now - start > std::chrono::milliseconds(500)) {
            WHEN_DEBUG(std::cout << "host timed out" << std::endl);
            return true;
        }
    }
    return false;
}

static ulong
wait_on_signal(hsa_signal_t doorbell, ulong timeout, ulong old_value)
{
    WHEN_DEBUG(std::cout << std::endl << "old signal value: "
               << (int64_t)old_value << std::endl);

    while (true) {
        auto new_value =
            hsa_signal_wait_scacquire(doorbell, HSA_SIGNAL_CONDITION_NE,
                                      old_value, timeout,
                                      HSA_WAIT_STATE_BLOCKED);
        WHEN_DEBUG(std::cout << "\nnew signal value: "
                   << new_value << std::endl);
        if (new_value != old_value)
            return new_value;
    }
}

#endif
