// RUN: %run_test hipify "%s" "%t" %cuda_args

/*
* This software contains source code provided by NVIDIA Corporation.
*/

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates a combination of Peer-to-Peer (P2P) and
 * Unified Virtual Address Space (UVA) features new to SDK 4.0
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>

// CUDA includes
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

__global__ void SimpleKernel(float *src, float *dst)
{
    // Just a dummy kernel, doing enough for us to verify that everything
    // worked
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * 2.0f;
}


// CHECK: inline bool IsGPUCapableP2P(hipDeviceProp_t *pProp)
inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
#ifdef _WIN32
    return (bool)(pProp->tccDriver ? true : false);
#else
    return (bool)(pProp->major >= 2);
#endif
}

inline bool IsAppBuiltAs64()
{
    return sizeof(void*) == 8;
}

int main(int argc, char **argv)
{
    printf("[%s] - Starting...\n", argv[0]);

    if (!IsAppBuiltAs64())
    {
        printf("%s is only supported with on 64-bit OSs and the application must be built as a 64-bit target.  Test is being waived.\n", argv[0]);
        exit(EXIT_WAIVED);
    }

    // Number of GPUs
    printf("Checking for multiple GPUs...\n");
    int gpu_n;
    // CHECK: checkCudaErrors(hipGetDeviceCount(&gpu_n));
    checkCudaErrors(cudaGetDeviceCount(&gpu_n));
    printf("CUDA-capable device count: %i\n", gpu_n);

    if (gpu_n < 2)
    {
        printf("Two or more GPUs with SM 2.0 or higher capability are required for %s.\n", argv[0]);
        printf("Waiving test.\n");
        exit(EXIT_WAIVED);
    }

    // Query device properties
    // CHECK hipDeviceProp_t prop[64];
    cudaDeviceProp prop[64];
    int gpuid[64]; // we want to find the first two GPU's that can support P2P
    int gpu_count = 0;   // GPUs that meet the criteria

    for (int i=0; i < gpu_n; i++)
    {
        // CHECK: checkCudaErrors(hipGetDeviceProperties(&prop[i], i));
        checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

        // Only boards based on Fermi can support P2P
        if ((prop[i].major >= 2)
#ifdef _WIN32
            // on Windows (64-bit), the Tesla Compute Cluster driver for windows must be enabled
            && prop[i].tccDriver
#endif
           )
        {
            // This is an array of P2P capable GPUs
            gpuid[gpu_count++] = i;
        }

        printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, prop[i].name, (IsGPUCapableP2P(&prop[i]) ? "IS " : "NOT"));
    }

    // Check for TCC for Windows
    if (gpu_count < 2)
    {
        printf("\nTwo or more GPUs with SM 2.0 or higher capability are required for %s.\n", argv[0]);
#ifdef _WIN32
        printf("\nAlso, a TCC driver must be installed and enabled to run %s.\n", argv[0]);
#endif
        // CHECK: checkCudaErrors(hipSetDevice(0));
        checkCudaErrors(cudaSetDevice(0));

        exit(EXIT_WAIVED);
    }

#if CUDART_VERSION >= 4000
    // Check possibility for peer access
    printf("\nChecking GPU(s) for support of peer to peer memory access...\n");

    int can_access_peer;
    int p2pCapableGPUs[2]; // We take only 1 pair of P2P capable GPUs
    p2pCapableGPUs[0] = p2pCapableGPUs[1] = -1;

    // Show all the combinations of supported P2P GPUs
    for (int i = 0; i < gpu_count; i++)
    {
        for (int j = 0; j < gpu_count; j++)
        {
            if (gpuid[i] == gpuid[j])
            {
                continue;
            }
            // CHECK: checkCudaErrors(hipDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
            checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
            printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[gpuid[i]].name, gpuid[i],
                           prop[gpuid[j]].name, gpuid[j] ,
                           can_access_peer ? "Yes" : "No");
            if (can_access_peer && p2pCapableGPUs[0] == -1)
            {
                    p2pCapableGPUs[0] = gpuid[i];
                    p2pCapableGPUs[1] = gpuid[j];
            }
        }
    }

    if (p2pCapableGPUs[0] == -1 || p2pCapableGPUs[1] == -1)
    {
        printf("Two or more GPUs with SM 2.0 or higher capability are required for %s.\n", argv[0]);
        printf("Peer to Peer access is not available amongst GPUs in the system, waiving test.\n");

        for (int i=0; i < gpu_count; i++)
        {
            // CHECK: checkCudaErrors(hipSetDevice(gpuid[i]));
            checkCudaErrors(cudaSetDevice(gpuid[i]));
        }
        exit(EXIT_WAIVED);
    }

    // Use first pair of p2p capable GPUs detected.
    gpuid[0] = p2pCapableGPUs[0];
    gpuid[1] = p2pCapableGPUs[1];

    // Enable peer access
    printf("Enabling peer access between GPU%d and GPU%d...\n", gpuid[0], gpuid[1]);

    // CHECK: checkCudaErrors(hipSetDevice(gpuid[0]));
    // CHECK: checkCudaErrors(hipDeviceEnablePeerAccess(gpuid[1], 0));
    // CHECK: checkCudaErrors(hipSetDevice(gpuid[1]));
    // CHECK: checkCudaErrors(hipDeviceEnablePeerAccess(gpuid[0], 0));
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    checkCudaErrors(cudaDeviceEnablePeerAccess(gpuid[1], 0));
    checkCudaErrors(cudaSetDevice(gpuid[1]));
    checkCudaErrors(cudaDeviceEnablePeerAccess(gpuid[0], 0));

    // Check that we got UVA on both devices
    printf("Checking GPU%d and GPU%d for UVA capabilities...\n", gpuid[0], gpuid[1]);
    const bool has_uva = (prop[gpuid[0]].unifiedAddressing && prop[gpuid[1]].unifiedAddressing);

    printf("> %s (GPU%d) supports UVA: %s\n", prop[gpuid[0]].name, gpuid[0], (prop[gpuid[0]].unifiedAddressing ? "Yes" : "No"));
    printf("> %s (GPU%d) supports UVA: %s\n", prop[gpuid[1]].name, gpuid[1], (prop[gpuid[1]].unifiedAddressing ? "Yes" : "No"));

    if (has_uva)
    {
        printf("Both GPUs can support UVA, enabling...\n");
    }
    else
    {
        printf("At least one of the two GPUs does NOT support UVA, waiving test.\n");
        exit(EXIT_WAIVED);
    }

    // Allocate buffers
    const size_t buf_size = 1024 * 1024 * 16 * sizeof(float);
    printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n", int(buf_size / 1024 / 1024), gpuid[0], gpuid[1]);
    // CHECK: checkCudaErrors(hipSetDevice(gpuid[0]));
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    float *g0;
    // CHECK: checkCudaErrors(hipMalloc(&g0, buf_size));
    // CHECK: checkCudaErrors(hipSetDevice(gpuid[1]));
    checkCudaErrors(cudaMalloc(&g0, buf_size));
    checkCudaErrors(cudaSetDevice(gpuid[1]));
    float *g1;
    // CHECK: checkCudaErrors(hipMalloc(&g1, buf_size));
    checkCudaErrors(cudaMalloc(&g1, buf_size));
    float *h0;
    // CHECK: checkCudaErrors(hipHostMalloc(&h0, buf_size));
    checkCudaErrors(cudaMallocHost(&h0, buf_size)); // Automatically portable with UVA

    // Create CUDA event handles
    printf("Creating event handles...\n");
    // CHECK: hipEvent_t start_event, stop_event;
    cudaEvent_t start_event, stop_event;
    float time_memcpy;
    // CHECK: int eventflags = hipEventBlockingSync;
    int eventflags = cudaEventBlockingSync;
    // CHECK: checkCudaErrors(hipEventCreateWithFlags(&start_event, eventflags));
    // CHECK: checkCudaErrors(hipEventCreateWithFlags(&stop_event, eventflags));
    checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
    checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));

    // P2P memcopy() benchmark
    // CHECK: checkCudaErrors(hipEventRecord(start_event, 0));
    checkCudaErrors(cudaEventRecord(start_event, 0));

    for (int i=0; i<100; i++)
    {
        // With UVA we don't need to specify source and target devices, the
        // runtime figures this out by itself from the pointers

        // Ping-pong copy between GPUs
        if (i % 2 == 0)
        {
            // CHECK: checkCudaErrors(hipMemcpy(g1, g0, buf_size, hipMemcpyDefault));
            checkCudaErrors(cudaMemcpy(g1, g0, buf_size, cudaMemcpyDefault));
        }
        else
        {
            // CHECK: checkCudaErrors(hipMemcpy(g0, g1, buf_size, hipMemcpyDefault));
            checkCudaErrors(cudaMemcpy(g0, g1, buf_size, cudaMemcpyDefault));
        }
    }

    // CHECK: checkCudaErrors(hipEventRecord(stop_event, 0));
    // CHECK: checkCudaErrors(hipEventSynchronize(stop_event));
    // CHECK: checkCudaErrors(hipEventElapsedTime(&time_memcpy, start_event, stop_event));
    // CHECK: printf("hipMemcpyPeer / hipMemcpy between GPU%d and GPU%d: %.2fGB/s\n", gpuid[0], gpuid[1],
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&time_memcpy, start_event, stop_event));
    printf("cudaMemcpyPeer / cudaMemcpy between GPU%d and GPU%d: %.2fGB/s\n", gpuid[0], gpuid[1],
           (1.0f / (time_memcpy / 1000.0f)) * ((100.0f * buf_size)) / 1024.0f / 1024.0f / 1024.0f);

    // Prepare host buffer and copy to GPU 0
    printf("Preparing host buffer and memcpy to GPU%d...\n", gpuid[0]);

    for (int i=0; i<buf_size / sizeof(float); i++)
    {
        h0[i] = float(i % 4096);
    }

    // CHECK: checkCudaErrors(hipSetDevice(gpuid[0]));
    // CHECK: checkCudaErrors(hipMemcpy(g0, h0, buf_size, hipMemcpyDefault));
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    checkCudaErrors(cudaMemcpy(g0, h0, buf_size, cudaMemcpyDefault));

    // Kernel launch configuration
    const dim3 threads(512, 1);
    const dim3 blocks((buf_size / sizeof(float)) / threads.x, 1);

    // Run kernel on GPU 1, reading input from the GPU 0 buffer, writing
    // output to the GPU 1 buffer
    printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n",
           gpuid[1], gpuid[0], gpuid[1]);
    // CHECK: checkCudaErrors(hipSetDevice(gpuid[1]));
    checkCudaErrors(cudaSetDevice(gpuid[1]));
    // CHECK: hipLaunchKernelGGL(SimpleKernel, dim3(blocks), dim3(threads), 0, 0, g0, g1);
    SimpleKernel<<<blocks, threads>>>(g0, g1);

    // CHECK: checkCudaErrors(hipDeviceSynchronize());
    checkCudaErrors(cudaDeviceSynchronize());

    // Run kernel on GPU 0, reading input from the GPU 1 buffer, writing
    // output to the GPU 0 buffer
    printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n",
           gpuid[0], gpuid[1], gpuid[0]);
    // CHECK: checkCudaErrors(hipSetDevice(gpuid[0]));
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    // CHECK: hipLaunchKernelGGL(SimpleKernel, dim3(blocks), dim3(threads), 0, 0, g1, g0);
    SimpleKernel<<<blocks, threads>>>(g1, g0);
    // CHECK: checkCudaErrors(hipDeviceSynchronize());
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy data back to host and verify
    printf("Copy data back to host from GPU%d and verify results...\n", gpuid[0]);
    // CHECK: checkCudaErrors(hipMemcpy(h0, g0, buf_size, hipMemcpyDefault));
    checkCudaErrors(cudaMemcpy(h0, g0, buf_size, cudaMemcpyDefault));

    int error_count = 0;

    for (int i=0; i<buf_size / sizeof(float); i++)
    {
        // Re-generate input data and apply 2x '* 2.0f' computation of both
        // kernel runs
        if (h0[i] != float(i % 4096) * 2.0f * 2.0f)
        {
            printf("Verification error @ element %i: val = %f, ref = %f\n", i, h0[i], (float(i%4096)*2.0f*2.0f));

            if (error_count++ > 10)
            {
                break;
            }
        }
    }

    // Disable peer access (also unregisters memory for non-UVA cases)
    printf("Disabling peer access...\n");
    // CHECK: checkCudaErrors(hipSetDevice(gpuid[0]));
    // CHECK: checkCudaErrors(hipDeviceDisablePeerAccess(gpuid[1]));
    // CHECK: checkCudaErrors(hipSetDevice(gpuid[1]));
    // CHECK: checkCudaErrors(hipDeviceDisablePeerAccess(gpuid[0]));
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    checkCudaErrors(cudaDeviceDisablePeerAccess(gpuid[1]));
    checkCudaErrors(cudaSetDevice(gpuid[1]));
    checkCudaErrors(cudaDeviceDisablePeerAccess(gpuid[0]));

    // Cleanup and shutdown
    printf("Shutting down...\n");
    // CHECK: checkCudaErrors(hipEventDestroy(start_event));
    // CHECK: checkCudaErrors(hipEventDestroy(stop_event));
    // CHECK: checkCudaErrors(hipSetDevice(gpuid[0]));
    // CHECK: checkCudaErrors(hipFree(g0));
    // CHECK: checkCudaErrors(hipSetDevice(gpuid[1]));
    // CHECK: checkCudaErrors(hipFree(g1));
    // CHECK: checkCudaErrors(hipHostFree(h0));
    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    checkCudaErrors(cudaFree(g0));
    checkCudaErrors(cudaSetDevice(gpuid[1]));
    checkCudaErrors(cudaFree(g1));
    checkCudaErrors(cudaFreeHost(h0));

    for (int i=0; i<gpu_n; i++)
    {
        // CHECK: checkCudaErrors(hipSetDevice(i));
        checkCudaErrors(cudaSetDevice(i));
    }

    if (error_count != 0)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("Test passed\n");
        exit(EXIT_SUCCESS);
    }

#else // Using CUDA 3.2 or older
    printf("simpleP2P requires CUDA 4.0 to build and run, waiving testing\n");
    exit(EXIT_WAIVED);
#endif

}

