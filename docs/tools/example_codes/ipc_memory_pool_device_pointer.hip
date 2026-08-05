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
#include <fstream>
#include <iostream>
#include <sys/stat.h>
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
constexpr const char* fifoPath     = "/tmp/hip_ipc_devptr_fifo";
constexpr const char* doneFifoPath = "/tmp/hip_ipc_devptr_done_fifo";

__global__ void fillKernel(int* data, std::size_t n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < static_cast<int>(n))
        data[tid] = tid * 2;
}

void runExporter()
{
    // Allocate memory from an IPC-capable pool.
    hipMemPoolProps poolProps = {};
    poolProps.allocType       = hipMemAllocationTypePinned;
    poolProps.handleTypes     = hipMemHandleTypePosixFileDescriptor;
    poolProps.location.type   = hipMemLocationTypeDevice;
    poolProps.location.id     = 0;

    hipMemPool_t memPool;
    HIP_CHECK(hipMemPoolCreate(&memPool, &poolProps));

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

    // Export the memory pool pointer.
    hipMemPoolPtrExportData exportData;
    HIP_CHECK(hipMemPoolExportPointer(&exportData, devPtr));

    // Write the exported data to the named pipe.
    mkfifo(fifoPath, 0666);
    std::ofstream fifoStream(fifoPath, std::ios::out | std::ios::binary);
    fifoStream.write(reinterpret_cast<char*>(&exportData), sizeof(exportData));
    fifoStream.close();

    // Wait for the importer to signal completion before freeing.
    mkfifo(doneFifoPath, 0666);
    char ack;
    std::ifstream doneStream(doneFifoPath, std::ios::in | std::ios::binary);
    doneStream.read(&ack, 1);
    doneStream.close();

    HIP_CHECK(hipFreeAsync(devPtr, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemPoolDestroy(memPool));
    HIP_CHECK(hipStreamDestroy(stream));
    unlink(fifoPath);
    unlink(doneFifoPath);
    std::cout << "Exporter: done." << std::endl;
}
// [sphinx-exporter-end]

// [sphinx-importer-start]
void runImporter()
{
    // Read the exported data from the named pipe.
    hipMemPoolPtrExportData importData;
    std::ifstream fifoStream(fifoPath, std::ios::in | std::ios::binary);
    fifoStream.read(reinterpret_cast<char*>(&importData), sizeof(importData));
    fifoStream.close();

    // Create a memory pool with matching IPC-capable properties.
    hipMemPoolProps poolProps = {};
    poolProps.allocType       = hipMemAllocationTypePinned;
    poolProps.handleTypes     = hipMemHandleTypePosixFileDescriptor;
    poolProps.location.type   = hipMemLocationTypeDevice;
    poolProps.location.id     = 0;

    hipMemPool_t memPool;
    HIP_CHECK(hipMemPoolCreate(&memPool, &poolProps));

    // Import the device pointer from the exporter's pool.
    int* importedDevPtr = nullptr;
    HIP_CHECK(hipMemPoolImportPointer(reinterpret_cast<void**>(&importedDevPtr),
                                      memPool,
                                      &importData));

    // Verify the values written by the exporter's kernel.
    int* hostData = new int[numElements];
    HIP_CHECK(hipMemcpy(hostData,
                        importedDevPtr,
                        numElements * sizeof(*hostData),
                        hipMemcpyDeviceToHost));

    bool passed = true;
    for (std::size_t i = 0; i < numElements; ++i)
    {
        if (hostData[i] != static_cast<int>(i * 2))
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

    HIP_CHECK(hipFree(importedDevPtr));
    HIP_CHECK(hipMemPoolDestroy(memPool));

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
