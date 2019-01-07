/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * HIT_START
 * BUILD: %t %s ../test_common.cpp HIPCC_OPTIONS -std=c++11 EXCLUDE_HIP_PLATFORM nvcc
 * RUN: %t
 * HIT_END
 */

// Includes HIP Runtime
#include "hip/hip_runtime.h"
#include <test_common.h>

#include <set>
#include <unordered_map>

#define test_failed(test_name)                                                 \
    printf("%s %s  FAILED!%s\n", KRED, test_name, KNRM);
#define test_passed(test_name)                                                 \
    printf("%s %s  PASSED!%s\n", KGRN, test_name, KNRM);

/* BEGIN DUPLICATION */

// TODO: The host code below is duplicated, and should ideally reside
// in a common host-side library.

typedef enum {
    __OCKL_AS_PACKET_EMPTY = 0,
    __OCKL_AS_PACKET_READY = 1
} __ockl_as_packet_type_t;

#define __OCKL_AS_PAYLOAD_ALIGNMENT 4
#define __OCKL_AS_PAYLOAD_BYTES 48

typedef enum {
    __OCKL_AS_PACKET_HEADER_TYPE = 0, // corresponds to HSA_PACKET_HEADER_TYPE
    __OCKL_AS_PACKET_HEADER_RESERVED0 = 8,
    __OCKL_AS_PACKET_HEADER_FLAGS = 13,
    __OCKL_AS_PACKET_HEADER_BYTES = 16,
    __OCKL_AS_PACKET_HEADER_SERVICE = 24,
} __ockl_as_packet_header_t;

typedef enum {
    __OCKL_AS_PACKET_HEADER_WIDTH_TYPE = 8,
    __OCKL_AS_PACKET_HEADER_WIDTH_RESERVED0 = 5,
    __OCKL_AS_PACKET_HEADER_WIDTH_FLAGS = 3,
    __OCKL_AS_PACKET_HEADER_WIDTH_BYTES = 8,
    __OCKL_AS_PACKET_HEADER_WIDTH_SERVICE = 8
} __ockl_as_packet_header_width_t;

// A packet is 64 bytes long, and the payload starts at index 16.
struct __ockl_as_packet_t {
    uint header;
    uint reserved1;
    ulong connection_id;

    uchar payload[__OCKL_AS_PAYLOAD_BYTES];
};

typedef enum {
    __OCKL_AS_STATUS_SUCCESS,
    __OCKL_AS_STATUS_INVALID_REQUEST,
    __OCKL_AS_STATUS_OUT_OF_RESOURCES,
    __OCKL_AS_STATUS_BUSY,
    __OCKL_AS_STATUS_UNKNOWN_ERROR
} __ockl_as_status_t;

typedef enum {
    __OCKL_AS_CONNECTION_BEGIN = 1,
    __OCKL_AS_CONNECTION_END = 2,
} __ockl_as_flag_t;

typedef enum { __OCKL_AS_FEATURE_ASYNCHRONOUS = 1 } __ockl_as_feature_t;

typedef struct {
    // Opaque handle. The value 0 is reserved.
    ulong handle;
} __ockl_as_signal_t;

typedef struct __ockl_as_packet_t __ockl_as_packet_t;

typedef struct {
    ulong read_index;
    ulong write_index;
    __ockl_as_signal_t doorbell_signal;
    __ockl_as_packet_t *base_address;
    ulong size;
} __ockl_as_stream_t;

#define ATTR_GLOBAL __attribute__((address_space(1)))

extern "C" __device__ __ockl_as_status_t
__ockl_as_write_block(__ockl_as_stream_t ATTR_GLOBAL *stream, uchar service_id,
                      ulong *connection_id, const uchar *str, uint32_t len,
                      uchar flags);

__device__ __ockl_as_status_t
__hip_as_write_block(__ockl_as_stream_t *stream, uchar service_id,
                     ulong *connection_id, const uchar *str, uint32_t len,
                     uchar flags)
{
    __ockl_as_stream_t ATTR_GLOBAL *gstream =
        reinterpret_cast<__ockl_as_stream_t ATTR_GLOBAL *>(stream);
    return __ockl_as_write_block(gstream, service_id, connection_id, str, len,
                                 flags);
}

/* END DUPLICATION */

static __ockl_as_stream_t *
createStream(void *ptr, uint buffer_size, uint num_packets)
{
    memset(ptr, 0, buffer_size);
    __ockl_as_stream_t *r = (__ockl_as_stream_t *)ptr;
    r->base_address = (__ockl_as_packet_t *)(&r[1]);
    r->doorbell_signal = {0};
    r->size = num_packets;

    return r;
}

static uint8_t
get_header_field(uint32_t header, uint8_t offset, uint8_t size)
{
    return (header >> offset) & ((1 << size) - 1);
}

static uint8_t
get_packet_type(uint32_t header)
{
    return get_header_field(header, __OCKL_AS_PACKET_HEADER_TYPE,
                            __OCKL_AS_PACKET_HEADER_WIDTH_TYPE);
}

static uint8_t
get_packet_flags(uint32_t header)
{
    return get_header_field(header, __OCKL_AS_PACKET_HEADER_FLAGS,
                            __OCKL_AS_PACKET_HEADER_WIDTH_FLAGS);
}

static uint8_t
get_packet_bytes(uint32_t header)
{
    return get_header_field(header, __OCKL_AS_PACKET_HEADER_BYTES,
                            __OCKL_AS_PACKET_HEADER_WIDTH_BYTES);
}

static uint8_t
get_packet_service(uint32_t header)
{
    return get_header_field(header, __OCKL_AS_PACKET_HEADER_SERVICE,
                            __OCKL_AS_PACKET_HEADER_WIDTH_SERVICE);
}

const unsigned int __OCKL_AS_PACKET_SIZE = sizeof(__ockl_as_packet_t);

/* END DUPLICATION */

using namespace std;

#define STR_HELLO_WORLD "hello world"
#define STRLEN_HELLO_WORLD 11

const unsigned int THREADS_PER_BLOCK = 123; // include a partial warp
const unsigned int NUM_BLOCKS = 3; // because powers of two are too convenient
const unsigned int NUM_THREADS = NUM_BLOCKS * THREADS_PER_BLOCK;
const unsigned int NUM_PACKETS_INSUFFICIENT = NUM_THREADS - 23;
const unsigned int NUM_PACKETS_LARGE = NUM_THREADS * 4;
const unsigned int NUM_SERVICES = 7;
const unsigned int TEST_SERVICE = 42;

unsigned int
read_uint(const unsigned char *ptr)
{
    unsigned int value = 0;

    for (int ii = sizeof(unsigned int) - 1; ii >= 0; --ii) {
        value <<= 8;
        value |= ptr[ii];
    }

    return value;
}

__global__ void
singlePacketSingleProducer(__ockl_as_stream_t *stream)
{
    uint len = STRLEN_HELLO_WORLD;

    uint64_t connection_id;

    __hip_as_write_block(stream, TEST_SERVICE, &connection_id,
                         (const uint8_t *)STR_HELLO_WORLD, STRLEN_HELLO_WORLD,
                         __OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END);
}

bool
checkSinglePacketSingleProducer(__ockl_as_stream_t *stream)
{
    if (stream->write_index != 1)
        return false;

    __ockl_as_packet_t *packet = &stream->base_address[0];
    uint header = packet->header;
    if (get_packet_type(header) != __OCKL_AS_PACKET_READY)
        return false;

    if (get_packet_service(header) != TEST_SERVICE)
        return false;

    if (get_packet_bytes(header) != STRLEN_HELLO_WORLD)
        return false;

    if (get_packet_flags(header) !=
        (__OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END))
        return false;

    if (0 != strcmp(STR_HELLO_WORLD, (const char *)packet->payload))
        return false;

    return true;
}

bool
ocklAsSinglePacketSingleProducer()
{
    bool success = true;
    unsigned int numThreads = 1;
    unsigned int numBlocks = 1;

    unsigned int numPackets = 1;
    unsigned int bufferSize =
        sizeof(__ockl_as_stream_t) + numPackets * __OCKL_AS_PACKET_SIZE;

    void *buffer;
    HIPCHECK(hipHostMalloc(&buffer, bufferSize));

    __ockl_as_stream_t *stream = createStream(buffer, bufferSize, numPackets);

    hipLaunchKernelGGL(singlePacketSingleProducer, dim3(numBlocks),
                       dim3(numThreads), 0, 0, stream);

    HIPCHECK(hipDeviceSynchronize());

    if (!checkSinglePacketSingleProducer(stream)) {
        test_failed(__func__);
        success = false;
    }

    HIPCHECK(hipHostFree(buffer));
    return success;
}

__global__ void
multipleProducers(__ockl_as_stream_t *stream)
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned char data = (unsigned char)tid;

    uint64_t connection_id;

    __hip_as_write_block(stream, tid % NUM_SERVICES, &connection_id,
                         (const unsigned char *)&tid, sizeof(unsigned int),
                         __OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END);
}

bool
checkMultipleProducers(__ockl_as_stream_t *stream)
{
    int data[NUM_SERVICES] = {
        0,
    };

    if (stream->write_index != NUM_THREADS)
        return false;

    for (int ii = 0; ii != NUM_THREADS; ++ii) {
        __ockl_as_packet_t *packet = &stream->base_address[ii];
        uint header = packet->header;
        if (get_packet_type(header) != __OCKL_AS_PACKET_READY)
            return false;

        if (get_packet_bytes(header) != sizeof(unsigned int))
            return false;

        if (get_packet_flags(header) !=
            (__OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END))
            return false;

        unsigned char service = get_packet_service(header);
        unsigned int payload = read_uint(packet->payload);

        if (service != payload % NUM_SERVICES)
            return false;
        data[service]++;
    }

    int expected[NUM_SERVICES];
    for (int ii = 0; ii != NUM_SERVICES; ++ii) {
        expected[ii] = NUM_THREADS / NUM_SERVICES;
        if (ii < NUM_THREADS % NUM_SERVICES) {
            expected[ii]++;
        }
    }

    for (int ii = 0; ii != NUM_SERVICES; ++ii) {
        if (data[ii] != expected[ii])
            return false;
    }

    return true;
}

bool
ocklAsMultipleProducers()
{
    bool success = true;
    unsigned int numPackets = NUM_THREADS;
    unsigned int bufferSize =
        sizeof(__ockl_as_stream_t) + numPackets * __OCKL_AS_PACKET_SIZE;

    void *buffer;
    HIPCHECK(hipHostMalloc(&buffer, bufferSize));

    __ockl_as_stream_t *stream = createStream(buffer, bufferSize, numPackets);

    hipLaunchKernelGGL(multipleProducers, dim3(NUM_BLOCKS),
                       dim3(THREADS_PER_BLOCK), 0, 0, stream);

    HIPCHECK(hipDeviceSynchronize());

    if (!checkMultipleProducers(stream)) {
        test_failed(__func__);
        success = false;
    }

    HIPCHECK(hipHostFree(buffer));
    return success;
}

__global__ void
dropPackets(__ockl_as_stream_t *stream, unsigned int *status)
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    uint64_t connection_id;

    status[tid] =
        __hip_as_write_block(stream, TEST_SERVICE, &connection_id,
                             (const unsigned char *)&tid, sizeof(unsigned int),
                             __OCKL_AS_CONNECTION_BEGIN |
                                 __OCKL_AS_CONNECTION_END);
}

bool
checkDropPackets(__ockl_as_stream_t *stream, unsigned int *status)
{
    unsigned int errorsExpected = NUM_THREADS - NUM_PACKETS_INSUFFICIENT;
    for (int ii = 0; ii != NUM_THREADS; ++ii) {
        switch (status[ii]) {
        case __OCKL_AS_STATUS_OUT_OF_RESOURCES:
            if (errorsExpected == 0)
                return false;
            --errorsExpected;
            break;
        case __OCKL_AS_STATUS_SUCCESS:
            break;
        default:
            return false;
        }
    }
    if (errorsExpected != 0)
        return false;

    if (stream->write_index != NUM_PACKETS_INSUFFICIENT)
        return false;

    for (int ii = 0; ii != NUM_PACKETS_INSUFFICIENT; ++ii) {
        __ockl_as_packet_t *packet = &stream->base_address[ii];
        uint header = packet->header;
        if (get_packet_type(header) != __OCKL_AS_PACKET_READY)
            return false;

        if (get_packet_service(header) != TEST_SERVICE)
            return false;

        if (get_packet_bytes(header) != sizeof(unsigned int))
            return false;

        if (get_packet_flags(header) !=
            (__OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END))
            return false;

        unsigned int payload = read_uint(packet->payload);
        if (payload >= NUM_THREADS)
            return false;
    }

    return true;
}

bool
ocklAsDropPackets()
{
    bool success = true;

    unsigned int numPackets = NUM_PACKETS_INSUFFICIENT;
    unsigned int bufferSize =
        sizeof(__ockl_as_stream_t) + numPackets * __OCKL_AS_PACKET_SIZE;
    void *buffer;
    HIPCHECK(hipHostMalloc(&buffer, bufferSize));

    void *status;
    HIPCHECK(hipHostMalloc(&status, NUM_THREADS * sizeof(unsigned int)));

    __ockl_as_stream_t *stream = createStream(buffer, bufferSize, numPackets);

    hipEvent_t event;
    HIPCHECK(hipEventCreate(&event));
    hipLaunchKernelGGL(dropPackets, dim3(NUM_BLOCKS), dim3(THREADS_PER_BLOCK),
                       0, 0, stream, (unsigned int *)status);

    HIPCHECK(hipDeviceSynchronize());

    if (!checkDropPackets(stream, (unsigned int *)status)) {
        test_failed(__func__);
        success = false;
    }

    HIPCHECK(hipHostFree(buffer));
    return success;
}

#define STR30 "Cras nec volutpat mi, sed sed."
#define STR47 "Lorem ipsum dolor sit amet, consectetur nullam."
#define STR60 "Curabitur id maximus nibh. Donec quis porttitor nisl nullam."
#define STR95                                                                  \
    "In mollis imperdiet nibh nec ullamcorper."                                \
    " Suspendisse placerat massa iaculis ipsum viverra sed."
#define STR124                                                                 \
    "Proin ut diam sit amet erat mollis gravida ac non sem."                   \
    " Mauris viverra leo metus, id luctus metus feugiat sed. Morbi "           \
    "posuere."

#define DECLARE_TEST_DATA()                                                    \
    const char *str30 = STR30;                                                 \
    const char *str60 = STR60;                                                 \
    const char *str47 = STR47;                                                 \
    const char *str95 = STR95;                                                 \
    const char *str124 = STR124;                                               \
    const int numStr = 5;                                                      \
    const char *strArray[5] = {str30, str60, str47, str95, str124};            \
    unsigned char strLengths[5] = {30, 60, 47, 95, 124};

__global__ void
mixedProducers(__ockl_as_stream_t *stream)
{
    DECLARE_TEST_DATA();

    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int idx = tid % 5;
    uint64_t connection_id;

    __hip_as_write_block(stream, idx, &connection_id,
                         (unsigned const char *)strArray[idx], strLengths[idx],
                         __OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END);
}

bool
checkMixedProducers(__ockl_as_stream_t *stream)
{
    typedef std::unordered_map<uint64_t, std::string> stream_buffer_map_t;
    stream_buffer_map_t buffers;
    std::unordered_map<std::string, int> strRecd;

    DECLARE_TEST_DATA();

    for (unsigned long read_index = 0; read_index != stream->write_index;
         ++read_index) {
        __ockl_as_packet_t *packet = stream->base_address + read_index;

        uint header = packet->header;
        if (get_packet_type(header) != __OCKL_AS_PACKET_READY)
            return false;

        uint bytes = get_packet_bytes(header);
        unsigned char flags = get_packet_flags(header);
        unsigned char service = get_packet_service(header);
        unsigned long connection_id = packet->connection_id;
        unsigned char *payload = packet->payload;

        if ((flags & __OCKL_AS_CONNECTION_BEGIN) !=
            (buffers.count(connection_id) == 0))
            return false;

        std::string &buf = buffers[connection_id];
        buf.insert(buf.end(), payload, payload + bytes);

        if (flags & __OCKL_AS_CONNECTION_END) {
            if (buf != strArray[service])
                return false;
            strRecd[buf]++;
            buffers.erase(connection_id);
        }
    }

    int expected_counts[numStr];
    for (int ii = 0; ii != numStr; ++ii) {
        expected_counts[ii] = NUM_THREADS / numStr;
        if (ii < (NUM_THREADS % numStr)) {
            ++expected_counts[ii];
        }
    }

    if (strRecd.size() != numStr)
        return false;

    for (int ii = 0; ii != numStr; ++ii) {
        std::string mystr(strArray[ii]);
        if (strRecd[mystr] != expected_counts[ii])
            return false;
    }
    return true;
}

bool
ocklAsMixedProducers()
{
    bool success = true;

    unsigned int numPackets = NUM_PACKETS_LARGE;
    unsigned int bufferSize =
        sizeof(__ockl_as_stream_t) + numPackets * __OCKL_AS_PACKET_SIZE;

    void *buffer;
    HIPCHECK(hipHostMalloc(&buffer, bufferSize));

    __ockl_as_stream_t *stream = createStream(buffer, bufferSize, numPackets);

    hipLaunchKernelGGL(mixedProducers, dim3(NUM_BLOCKS),
                       dim3(THREADS_PER_BLOCK), 0, 0, stream);

    HIPCHECK(hipDeviceSynchronize());

    if (!checkMixedProducers(stream)) {
        test_failed(__func__);
        success = false;
    }

    HIPCHECK(hipHostFree(buffer));
    return success;
}

#define STR27 "In et consectetur mi metus."
#define STR64 "Praesent tempus arcu id ligula blandit, eget congue justo metus."
#define STR40 "Sed at dolor ipsum. Curabitur cras amet."

__global__ void
splitMessage(__ockl_as_stream_t *stream)
{
    const char *str27 = STR27;
    const char *str64 = STR64;
    const char *str40 = STR40;
    const int numStr = 3;
    const char *strArray[] = {str27, str64, str40};
    int strLengths[] = {27, 64, 40};

    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int service = tid % 3;
    int first = tid % 3;
    int second = (tid + 1) % 3;
    int third = (tid + 2) % 3;

    uint64_t connection_id;
    __hip_as_write_block(stream, service, &connection_id,
                         (unsigned const char *)strArray[first],
                         strLengths[first], __OCKL_AS_CONNECTION_BEGIN);
    __hip_as_write_block(stream, service, &connection_id,
                         (unsigned const char *)strArray[second],
                         strLengths[second], 0);
    __hip_as_write_block(stream, service, &connection_id,
                         (unsigned const char *)strArray[third],
                         strLengths[third], __OCKL_AS_CONNECTION_END);
}

bool
checkSplitMessage(__ockl_as_stream_t *stream)
{
    typedef std::unordered_map<uint64_t, std::string> stream_buffer_map_t;
    stream_buffer_map_t buffers;
    std::unordered_map<std::string, int> strRecd;

    static const int numExpected = 3;
    const char *strExpected[numExpected] = {STR27 STR64 STR40,
                                            STR64 STR40 STR27,
                                            STR40 STR27 STR64};

    for (unsigned long read_index = 0; read_index != stream->write_index;
         ++read_index) {
        __ockl_as_packet_t *packet = stream->base_address + read_index;

        uint header = packet->header;
        if (get_packet_type(header) != __OCKL_AS_PACKET_READY)
            return false;

        uint bytes = get_packet_bytes(header);
        unsigned char flags = get_packet_flags(header);
        unsigned long connection_id = packet->connection_id;
        unsigned char *payload = packet->payload;
        uint8_t service = get_packet_service(header);

        if ((flags & __OCKL_AS_CONNECTION_BEGIN) !=
            (buffers.count(connection_id) == 0))
            return false;

        std::string &buf = buffers[connection_id];
        buf.insert(buf.end(), payload, payload + bytes);

        if (flags & __OCKL_AS_CONNECTION_END) {
            if (buf != strExpected[service])
                return false;
            strRecd[buf] += 1;
            buffers.erase(connection_id);
        }
    }

    int expected_counts[numExpected];
    for (int ii = 0; ii != numExpected; ++ii) {
        expected_counts[ii] = NUM_THREADS / numExpected;
        if (ii < (NUM_THREADS % numExpected)) {
            ++expected_counts[ii];
        }
    }

    if (strRecd.size() != numExpected)
        return false;

    for (int ii = 0; ii != numExpected; ++ii) {
        std::string mystr(strExpected[ii]);
        if (strRecd[mystr] != expected_counts[ii])
            return false;
    }

    return true;
}

bool
ocklAsSplitMessage()
{
    bool success = true;

    unsigned int numPackets = NUM_PACKETS_LARGE;
    unsigned int bufferSize =
        sizeof(__ockl_as_stream_t) + numPackets * __OCKL_AS_PACKET_SIZE;

    void *buffer;
    HIPCHECK(hipHostMalloc(&buffer, bufferSize));

    __ockl_as_stream_t *stream = createStream(buffer, bufferSize, numPackets);

    hipLaunchKernelGGL(splitMessage, dim3(NUM_BLOCKS), dim3(THREADS_PER_BLOCK),
                       0, 0, stream);

    HIPCHECK(hipDeviceSynchronize());

    if (!checkSplitMessage(stream)) {
        test_failed(__func__);
        success = false;
    }

    HIPCHECK(hipHostFree(buffer));
    return success;
}

#define TESTNAME "hipAsynchronousStreams"
int
main(int argc, char **argv)
{
    bool success = true;

    success &= ocklAsSinglePacketSingleProducer();
    success &= ocklAsMultipleProducers();
    success &= ocklAsDropPackets();
    success &= ocklAsMixedProducers();
    success &= ocklAsSplitMessage();

    hipDeviceReset();

    if (success) {
        test_passed(TESTNAME);
        return 0;
    }

    failed(TESTNAME);
}
