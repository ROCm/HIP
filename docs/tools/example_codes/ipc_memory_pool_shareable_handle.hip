// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// [sphinx-exporter-start]
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#define HIP_CHECK(expression)                        \
{                                                    \
    const hipError_t status = expression;            \
    if (status != hipSuccess)                        \
    {                                                \
        std::cerr << "HIP error " << status          \
                << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":"         \
                << __LINE__ << std::endl;            \
        std::exit(EXIT_FAILURE);                     \
    }                                                \
}

constexpr std::size_t numElements  = 1024;
constexpr const char* sockPath     = "/tmp/hip_ipc_handle.sock";
constexpr const char* doneFifoPath = "/tmp/hip_ipc_handle_done_fifo";

__global__ void fillKernel(int* data, std::size_t n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < static_cast<int>(n))
        data[tid] = tid * 2;
}

// Sends an open file descriptor to another process via SCM_RIGHTS.
// A POSIX file descriptor is process-local: its integer value has no meaning
// in another process's fd table. SCM_RIGHTS is the POSIX mechanism that
// duplicates the underlying kernel file description into the receiving process.
static void sendFd(int fdToSend)
{
    unlink(sockPath);
    int listenSock = socket(AF_UNIX, SOCK_STREAM, 0);

    sockaddr_un addr = {};
    addr.sun_family  = AF_UNIX;
    strncpy(addr.sun_path, sockPath, sizeof(addr.sun_path) - 1);
    bind(listenSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    listen(listenSock, 1);

    std::cout << "Exporter: waiting for importer to connect..." << std::endl;
    int conn = accept(listenSock, nullptr, nullptr);

    char buf[1]              = {0};
    iovec iov                = {buf, sizeof(buf)};
    char cmsgBuf[CMSG_SPACE(sizeof(int))];
    msghdr msg               = {};
    msg.msg_iov              = &iov;
    msg.msg_iovlen           = 1;
    msg.msg_control          = cmsgBuf;
    msg.msg_controllen       = sizeof(cmsgBuf);

    cmsghdr* cmsg            = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level         = SOL_SOCKET;
    cmsg->cmsg_type          = SCM_RIGHTS;
    cmsg->cmsg_len           = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &fdToSend, sizeof(int));

    sendmsg(conn, &msg, 0);
    std::cout << "Exporter: handle sent via SCM_RIGHTS." << std::endl;

    // Wait for importer's ack before closing the connection.
    char ack;
    recv(conn, &ack, 1, 0);
    close(conn);
    close(listenSock);
    unlink(sockPath);
}

void runExporter()
{
    // Create an IPC-capable memory pool and export its handle before
    // any allocation so the importer can independently allocate from it.
    hipMemPoolProps poolProps = {};
    poolProps.allocType       = hipMemAllocationTypePinned;
    poolProps.handleTypes     = hipMemHandleTypePosixFileDescriptor;
    poolProps.location.type   = hipMemLocationTypeDevice;
    poolProps.location.id     = 0;

    hipMemPool_t memPool;
    HIP_CHECK(hipMemPoolCreate(&memPool, &poolProps));

    int descriptor = -1;
    HIP_CHECK(hipMemPoolExportToShareableHandle(reinterpret_cast<void*>(&descriptor),
                                                memPool,
                                                hipMemHandleTypePosixFileDescriptor,
                                                0));

    // Transfer the file descriptor to the importer via SCM_RIGHTS.
    sendFd(descriptor);

    // Keep the pool alive until the importer signals completion.
    mkfifo(doneFifoPath, 0666);
    char ack;
    std::ifstream doneStream(doneFifoPath, std::ios::in | std::ios::binary);
    doneStream.read(&ack, 1);
    doneStream.close();

    HIP_CHECK(hipMemPoolDestroy(memPool));
    ::close(descriptor);
    unlink(doneFifoPath);
    std::cout << "Exporter: done." << std::endl;
}
// [sphinx-exporter-end]

// Receives a file descriptor from another process via SCM_RIGHTS.
static int recvFd()
{
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);

    sockaddr_un addr = {};
    addr.sun_family  = AF_UNIX;
    strncpy(addr.sun_path, sockPath, sizeof(addr.sun_path) - 1);

    while(connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
        usleep(10000);

    char buf[1]              = {};
    iovec iov                = {buf, sizeof(buf)};
    char cmsgBuf[CMSG_SPACE(sizeof(int))];
    msghdr msg               = {};
    msg.msg_iov              = &iov;
    msg.msg_iovlen           = 1;
    msg.msg_control          = cmsgBuf;
    msg.msg_controllen       = sizeof(cmsgBuf);

    recvmsg(sock, &msg, 0);

    int receivedFd           = -1;
    cmsghdr* cmsg            = CMSG_FIRSTHDR(&msg);
    if(cmsg && cmsg->cmsg_type == SCM_RIGHTS)
        memcpy(&receivedFd, CMSG_DATA(cmsg), sizeof(int));

    // Ack so the exporter knows the fd has been received.
    char ack = 1;
    send(sock, &ack, 1, 0);
    close(sock);
    return receivedFd;
}

// [sphinx-importer-start]
void runImporter()
{
    // Receive the pool's file descriptor via SCM_RIGHTS and import the pool.
    std::cout << "Importer: connecting to exporter..." << std::endl;
    int descriptor = recvFd();

    hipMemPool_t memPool;
    void* handleValue = reinterpret_cast<void*>(static_cast<intptr_t>(descriptor));
    HIP_CHECK(hipMemPoolImportFromShareableHandle(&memPool,
                                                  handleValue,
                                                  hipMemHandleTypePosixFileDescriptor,
                                                  0));

    // Allocate from the imported pool and run a kernel to verify GPU access.
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    int* devPtr = nullptr;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&devPtr),
                                     numElements * sizeof(*devPtr),
                                     memPool,
                                     stream));

    dim3 blockSize(256);
    dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);
    fillKernel<<<gridSize, blockSize, 0, stream>>>(devPtr, numElements);
    HIP_CHECK(hipStreamSynchronize(stream));

    int* hostData = new int[numElements];
    HIP_CHECK(hipMemcpy(hostData,
                        devPtr,
                        numElements * sizeof(*hostData),
                        hipMemcpyDeviceToHost));

    bool passed = true;
    for(std::size_t i = 0; i < numElements; ++i)
    {
        if(hostData[i] != static_cast<int>(i * 2))
        {
            std::cerr << "Mismatch at [" << i << "]: expected " << i * 2
                      << ", got " << hostData[i] << std::endl;
            passed = false;
            break;
        }
    }
    delete[] hostData;

    // Signal the exporter that this process has finished.
    char ack = 1;
    std::ofstream doneStream(doneFifoPath, std::ios::out | std::ios::binary);
    doneStream.write(&ack, 1);
    doneStream.close();

    HIP_CHECK(hipFreeAsync(devPtr, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemPoolDestroy(memPool));
    HIP_CHECK(hipStreamDestroy(stream));
    ::close(descriptor);

    std::cout << "Importer: " << (passed ? "PASS" : "FAIL") << std::endl;
    std::exit(passed ? EXIT_SUCCESS : EXIT_FAILURE);
}
// [sphinx-importer-end]

int main(int argc, char* argv[])
{
    if(argc < 2 || (std::string(argv[1]) != "export" && std::string(argv[1]) != "import"))
    {
        std::cerr << "Usage: " << argv[0] << " export|import" << std::endl;
        return EXIT_FAILURE;
    }

    if(std::string(argv[1]) == "export")
        runExporter();
    else
        runImporter();

    return EXIT_SUCCESS;
}
