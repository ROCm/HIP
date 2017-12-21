// RUN: %run_test hipify "%s" "%t" %cuda_args
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

#include <cstdio>
#include <vector>

#include <helper_cuda.h>

using namespace std;

const char *sSampleName = "P2P (Peer-to-Peer) GPU Bandwidth Latency Test";

//Macro for checking cuda errors following a cuda launch or api call

// CHECK: #define cudaCheckError() {                                          \
// CHECK: hipError_t e=hipGetLastError();                                 \
// CHECK: if(e!=hipSuccess) {                                              \
// CHECK: printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));           \
// CHECK: exit(EXIT_FAILURE);                                           \
// CHECK: }                                                                 \
// CHECK: }
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }
__global__ void delay(int * null) {
  float j=threadIdx.x;
  for(int i=1;i<10000;i++)
      j=(j+1)/j;

  if(threadIdx.x == j) null[0] = j;
}

void checkP2Paccess(int numGPUs)
{
    for (int i=0; i<numGPUs; i++)
    {
        // CHECK: hipSetDevice(i);
        cudaSetDevice(i);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if (i!=j)
            {
                // CHECK: hipDeviceCanAccessPeer(&access,i,j);
                cudaDeviceCanAccessPeer(&access,i,j);
                printf("Device=%d %s Access Peer Device=%d\n", i, access ? "CAN" : "CANNOT", j);
            }
        }
    }
    printf("\n***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.\nSo you can see lesser Bandwidth (GB/s) in those cases.\n\n");
}

void outputBandwidthMatrix(int numGPUs, bool p2p)
{
    int numElems=10000000;
    int repeat=5;
    vector<int *> buffers(numGPUs);
    vector<int *> buffersD2D(numGPUs); // buffer for D2D, that is, intra-GPU copy
    // CHECK: vector<hipEvent_t> start(numGPUs);
    // CHECK: vector<hipEvent_t> stop(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        // CHECK: hipSetDevice(d);
        // CHECK: hipMalloc(&buffers[d],numElems*sizeof(int));
        // CHECK: hipMalloc(&buffersD2D[d],numElems*sizeof(int));
        cudaSetDevice(d);
        cudaMalloc(&buffers[d],numElems*sizeof(int));
        cudaMalloc(&buffersD2D[d],numElems*sizeof(int));
        cudaCheckError();
        // CHECK: hipEventCreate(&start[d]);
        cudaEventCreate(&start[d]);
        cudaCheckError();
        // CHECK: hipEventCreate(&stop[d]);
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    vector<double> bandwidthMatrix(numGPUs*numGPUs);

    for (int i=0; i<numGPUs; i++)
    {
        // CHECK: hipSetDevice(i);
        cudaSetDevice(i);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if(p2p) {
                // CHECK: hipDeviceCanAccessPeer(&access,i,j);
                cudaDeviceCanAccessPeer(&access,i,j);
                if (access)
                {
                    // CHECK: hipDeviceEnablePeerAccess(j,0 );
                    cudaDeviceEnablePeerAccess(j,0 );
                    cudaCheckError();
                    // CHECK: hipSetDevice(j);
                    cudaSetDevice(j);
                    // CHECK: hipDeviceEnablePeerAccess(i,0 );
                    cudaDeviceEnablePeerAccess(i,0 );
                    // CHECK: hipSetDevice(i);
                    cudaSetDevice(i);
                    cudaCheckError();
                }
            }
            // CHECK: hipDeviceSynchronize();
            cudaDeviceSynchronize();
            cudaCheckError();
            // CHECK: hipLaunchKernelGGL(delay, dim3(1), dim3(1), 0, 0, (int *)NULL);
            delay<<<1,1>>>((int *)NULL);
            // CHECK: hipEventRecord(start[i]);
            cudaEventRecord(start[i]);

            if (i==j)
            {
                // Perform intra-GPU, D2D copies
                for (int r=0; r<repeat; r++)
                {
                    // CHECK: hipMemcpyPeerAsync(buffers[i],i,buffersD2D[i],i,sizeof(int)*numElems);
                    cudaMemcpyPeerAsync(buffers[i],i,buffersD2D[i],i,sizeof(int)*numElems);
                }
            }
            else
            {
                for (int r=0; r<repeat; r++)
                {
                    // CHECK: hipMemcpyPeerAsync(buffers[i],i,buffers[j],j,sizeof(int)*numElems);
                    cudaMemcpyPeerAsync(buffers[i],i,buffers[j],j,sizeof(int)*numElems);
                }
            }
            // CHECK: hipEventRecord(stop[i]);
            // CHECK: hipDeviceSynchronize();
            cudaEventRecord(stop[i]);
            cudaDeviceSynchronize();
            cudaCheckError();

            float time_ms;
            // CHECK: hipEventElapsedTime(&time_ms,start[i],stop[i]);
            cudaEventElapsedTime(&time_ms,start[i],stop[i]);
            double time_s=time_ms/1e3;

            double gb=numElems*sizeof(int)*repeat/(double)1e9;
            if(i==j) gb*=2;  //must count both the read and the write here
            bandwidthMatrix[i*numGPUs+j]=gb/time_s;
            if (p2p && access)
            {
                // CHECK: hipDeviceDisablePeerAccess(j);
                // CHECK: hipSetDevice(j);
                // CHECK: hipDeviceDisablePeerAccess(i);
                // CHECK: hipSetDevice(i);
                cudaDeviceDisablePeerAccess(j);
                cudaSetDevice(j);
                cudaDeviceDisablePeerAccess(i);
                cudaSetDevice(i);
                cudaCheckError();
            }
        }
    }

    printf("   D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d ", j);
    }

    printf("\n");

    for (int i=0; i<numGPUs; i++)
    {
        printf("%6d ",i);

        for (int j=0; j<numGPUs; j++)
        {
            printf("%6.02f ", bandwidthMatrix[i*numGPUs+j]);
        }

        printf("\n");
    }

    for (int d=0; d<numGPUs; d++)
    {
        // CHECK: hipSetDevice(d);
        // CHECK: hipFree(buffers[d]);
        // CHECK: hipFree(buffersD2D[d]);
        cudaSetDevice(d);
        cudaFree(buffers[d]);
        cudaFree(buffersD2D[d]);
        cudaCheckError();
        // CHECK: hipEventDestroy(start[d]);
        cudaEventDestroy(start[d]);
        cudaCheckError();
        // CHECK: hipEventDestroy(stop[d]);
        cudaEventDestroy(stop[d]);
        cudaCheckError();
    }
}

void outputBidirectionalBandwidthMatrix(int numGPUs, bool p2p)
{
    int numElems=10000000;
    int repeat=5;
    vector<int *> buffers(numGPUs);
    vector<int *> buffersD2D(numGPUs);
    // CHECK: vector<hipEvent_t> start(numGPUs);
    // CHECK: vector<hipEvent_t> stop(numGPUs);
    // CHECK: vector<hipStream_t> stream0(numGPUs);
    // CHECK: vector<hipStream_t> stream1(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);
    vector<cudaStream_t> stream0(numGPUs);
    vector<cudaStream_t> stream1(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        // CHECK: hipSetDevice(d);
        // CHECK: hipMalloc(&buffers[d],numElems*sizeof(int));
        // CHECK: hipMalloc(&buffersD2D[d],numElems*sizeof(int));
        cudaSetDevice(d);
        cudaMalloc(&buffers[d],numElems*sizeof(int));
        cudaMalloc(&buffersD2D[d],numElems*sizeof(int));
        cudaCheckError();
        // CHECK: hipEventCreate(&start[d]);
        cudaEventCreate(&start[d]);
        cudaCheckError();
        // CHECK: hipEventCreate(&stop[d]);
        cudaEventCreate(&stop[d]);
        cudaCheckError();
        // CHECK: hipStreamCreate(&stream0[d]);
        cudaStreamCreate(&stream0[d]);
        cudaCheckError();
        // CHECK: hipStreamCreate(&stream1[d]);
        cudaStreamCreate(&stream1[d]);
        cudaCheckError();
    }

    vector<double> bandwidthMatrix(numGPUs*numGPUs);

    for (int i=0; i<numGPUs; i++)
    {
        // CHECK: hipSetDevice(i);
        cudaSetDevice(i);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if(p2p) {
                // CHECK: hipDeviceCanAccessPeer(&access,i,j);
                cudaDeviceCanAccessPeer(&access,i,j);
                if (access)
                {
                    // CHECK: hipSetDevice(i);
                    // CHECK: hipDeviceEnablePeerAccess(j,0);
                    cudaSetDevice(i);
                    cudaDeviceEnablePeerAccess(j,0);
                    cudaCheckError();
                    // CHECK: hipSetDevice(j);
                    // CHECK: hipDeviceEnablePeerAccess(i,0);
                    cudaSetDevice(j);
                    cudaDeviceEnablePeerAccess(i,0);
                    cudaCheckError();
                }
            }
            // CHECK: hipSetDevice(i);
            // CHECK: hipDeviceSynchronize();
            cudaSetDevice(i);
            cudaDeviceSynchronize();
            cudaCheckError();
            // CHECK: hipLaunchKernelGGL(delay, dim3(1), dim3(1), 0, 0, (int *)NULL);
            // CHECK: hipEventRecord(start[i]);
            delay<<<1,1>>>((int *)NULL);
            cudaEventRecord(start[i]);

            if (i==j)
            {
                // For intra-GPU perform 2 memcopies buffersD2D <-> buffers
                for (int r=0; r<repeat; r++)
                {
                    // CHECK: hipMemcpyPeerAsync(buffers[i], i, buffersD2D[i],i,sizeof(int)*numElems,stream0[i]);
                    // CHECK: hipMemcpyPeerAsync(buffersD2D[i], i, buffers[i],i,sizeof(int)*numElems,stream1[i]);
                    cudaMemcpyPeerAsync(buffers[i], i, buffersD2D[i],i,sizeof(int)*numElems,stream0[i]);
                    cudaMemcpyPeerAsync(buffersD2D[i], i, buffers[i],i,sizeof(int)*numElems,stream1[i]);
                }
            }
            else
            {
                for (int r=0; r<repeat; r++)
                {
                    // CHECK: hipMemcpyPeerAsync(buffers[i],i,buffers[j],j,sizeof(int)*numElems,stream0[i]);
                    // CHECK: hipMemcpyPeerAsync(buffers[j],j,buffers[i],i,sizeof(int)*numElems,stream1[i]);
                    cudaMemcpyPeerAsync(buffers[i],i,buffers[j],j,sizeof(int)*numElems,stream0[i]);
                    cudaMemcpyPeerAsync(buffers[j],j,buffers[i],i,sizeof(int)*numElems,stream1[i]);
                }
            }
            // CHECK: hipEventRecord(stop[i]);
            // CHECK: hipDeviceSynchronize();
            cudaEventRecord(stop[i]);
            cudaDeviceSynchronize();
            cudaCheckError();

            float time_ms;
            // CHECK: hipEventElapsedTime(&time_ms,start[i],stop[i]);
            cudaEventElapsedTime(&time_ms,start[i],stop[i]);
            double time_s=time_ms/1e3;

            double gb=2.0*numElems*sizeof(int)*repeat/(double)1e9;
            if(i==j) gb*=2;  //must count both the read and the write here
            bandwidthMatrix[i*numGPUs+j]=gb/time_s;
            if(p2p && access)
            {
                // CHECK: hipSetDevice(i);
                // CHECK: hipDeviceDisablePeerAccess(j);
                // CHECK: hipSetDevice(j);
                // CHECK: hipDeviceDisablePeerAccess(i);
                cudaSetDevice(i);
                cudaDeviceDisablePeerAccess(j);
                cudaSetDevice(j);
                cudaDeviceDisablePeerAccess(i);
            }
        }
    }

    printf("   D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d ", j);
    }

    printf("\n");

    for (int i=0; i<numGPUs; i++)
    {
        printf("%6d ",i);

        for (int j=0; j<numGPUs; j++)
        {
            printf("%6.02f ", bandwidthMatrix[i*numGPUs+j]);
        }

        printf("\n");
    }

    for (int d=0; d<numGPUs; d++)
    {
        // CHECK: hipSetDevice(d);
        // CHECK: hipFree(buffers[d]);
        // CHECK: hipFree(buffersD2D[d]);
        cudaSetDevice(d);
        cudaFree(buffers[d]);
        cudaFree(buffersD2D[d]);
        cudaCheckError();
        // CHECK: hipEventDestroy(start[d]);
        cudaEventDestroy(start[d]);
        cudaCheckError();
        // CHECK: hipEventDestroy(stop[d]);
        cudaEventDestroy(stop[d]);
        cudaCheckError();
        // CHECK: hipStreamDestroy(stream0[d]);
        cudaStreamDestroy(stream0[d]);
        cudaCheckError();
        // CHECK: hipStreamDestroy(stream1[d]);
        cudaStreamDestroy(stream1[d]);
        cudaCheckError();
    }
}

void outputLatencyMatrix(int numGPUs, bool p2p)
{
    int repeat=10000;
    vector<int *> buffers(numGPUs);
    vector<int *> buffersD2D(numGPUs);  // buffer for D2D, that is, intra-GPU copy
    // CHECK: vector<hipEvent_t> start(numGPUs);
    // CHECK: vector<hipEvent_t> stop(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        // CHECK: hipSetDevice(d);
        // CHECK: hipMalloc(&buffers[d],1);
        // CHECK: hipMalloc(&buffersD2D[d],1);
        cudaSetDevice(d);
        cudaMalloc(&buffers[d],1);
        cudaMalloc(&buffersD2D[d],1);
        cudaCheckError();
        // CHECK: hipEventCreate(&start[d]);
        cudaEventCreate(&start[d]);
        cudaCheckError();
        // CHECK: hipEventCreate(&stop[d]);
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    vector<double> latencyMatrix(numGPUs*numGPUs);

    for (int i=0; i<numGPUs; i++)
    {
        // CHECK: hipSetDevice(i);
        cudaSetDevice(i);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if(p2p) {
                // CHECK: hipDeviceCanAccessPeer(&access,i,j);
                cudaDeviceCanAccessPeer(&access,i,j);
                if (access)
                {
                    // CHECK: hipDeviceEnablePeerAccess(j,0);
                    cudaDeviceEnablePeerAccess(j,0);
                    cudaCheckError();
                    // CHECK: hipSetDevice(j);
                    // CHECK: hipDeviceEnablePeerAccess(i,0 );
                    // CHECK: hipSetDevice(i);
                    cudaSetDevice(j);
                    cudaDeviceEnablePeerAccess(i,0 );
                    cudaSetDevice(i);
                    cudaCheckError();
                }
            }
            // CHECK: hipDeviceSynchronize();
            cudaDeviceSynchronize();
            cudaCheckError();
            // CHECK: hipLaunchKernelGGL(delay, dim3(1), dim3(1), 0, 0, (int *)NULL);
            delay<<<1,1>>>((int *)NULL);
            // CHECK: hipEventRecord(start[i]);
            cudaEventRecord(start[i]);

            if (i==j)
            {
                // Perform intra-GPU, D2D copies
                for (int r=0; r<repeat; r++)
                {
                    // CHECK: hipMemcpyPeerAsync(buffers[i],i,buffersD2D[i],i,1);
                    cudaMemcpyPeerAsync(buffers[i],i,buffersD2D[i],i,1);
                }
            }
            else
            {
                for (int r=0; r<repeat; r++)
                {
                    // CHECK: hipMemcpyPeerAsync(buffers[j],j,buffers[i],i,1);
                    cudaMemcpyPeerAsync(buffers[j],j,buffers[i],i,1); // Peform P2P writes
                }
            }

            // CHECK: hipEventRecord(stop[i]);
            // CHECK: hipDeviceSynchronize();
            cudaEventRecord(stop[i]);
            cudaDeviceSynchronize();
            cudaCheckError();

            float time_ms;
            // CHECK: hipEventElapsedTime(&time_ms,start[i],stop[i]);
            cudaEventElapsedTime(&time_ms,start[i],stop[i]);

            latencyMatrix[i*numGPUs+j]=time_ms*1e3/repeat;
            if(p2p && access)
            {
                // CHECK: hipDeviceDisablePeerAccess(j);
                // CHECK: hipSetDevice(j);
                // CHECK: hipDeviceDisablePeerAccess(i);
                // CHECK: hipSetDevice(i);
                cudaDeviceDisablePeerAccess(j);
                cudaSetDevice(j);
                cudaDeviceDisablePeerAccess(i);
                cudaSetDevice(i);
                cudaCheckError();
            }
        }
    }

    printf("   D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d ", j);
    }

    printf("\n");

    for (int i=0; i<numGPUs; i++)
    {
        printf("%6d ",i);

        for (int j=0; j<numGPUs; j++)
        {
            printf("%6.02f ", latencyMatrix[i*numGPUs+j]);
        }

        printf("\n");
    }

    for (int d=0; d<numGPUs; d++)
    {
        // CHECK: hipSetDevice(d);
        // CHECK: hipFree(buffers[d]);
        // CHECK: hipFree(buffersD2D[d]);
        cudaSetDevice(d);
        cudaFree(buffers[d]);
        cudaFree(buffersD2D[d]);
        cudaCheckError();
        // CHECK: hipEventDestroy(start[d]);
        cudaEventDestroy(start[d]);
        cudaCheckError();
        // CHECK: hipEventDestroy(stop[d]);
        cudaEventDestroy(stop[d]);
        cudaCheckError();
    }
}

int main(int argc, char **argv)
{

    int numGPUs;
    // CHECK: hipGetDeviceCount(&numGPUs);
    cudaGetDeviceCount(&numGPUs);

    printf("[%s]\n", sSampleName);

    //output devices
    for (int i=0; i<numGPUs; i++)
    {
        // CHECK: hipDeviceProp_t prop;
        // CHECK: hipGetDeviceProperties(&prop,i);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);
        printf("Device: %d, %s, pciBusID: %x, pciDeviceID: %x, pciDomainID:%x\n",i,prop.name, prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
    }

    checkP2Paccess(numGPUs);

    //Check peer-to-peer connectivity
    printf("P2P Connectivity Matrix\n");
    printf("     D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d", j);
    }
    printf("\n");

    for (int i=0; i<numGPUs; i++)
    {
        printf("%6d\t", i);
        for (int j=0; j<numGPUs; j++)
        {
            if (i!=j)
            {
               int access;
               // CHECK: hipDeviceCanAccessPeer(&access,i,j);
               cudaDeviceCanAccessPeer(&access,i,j);
               printf("%6d", (access) ? 1 : 0);
            }
            else
            {
                printf("%6d", 1);
            }
        }
        printf("\n");
    }

    printf("Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)\n");
    outputBandwidthMatrix(numGPUs, false);
    printf("Unidirectional P2P=Enabled Bandwidth Matrix (GB/s)\n");
    outputBandwidthMatrix(numGPUs, true);
    printf("Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)\n");
    outputBidirectionalBandwidthMatrix(numGPUs, false);
    printf("Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)\n");
    outputBidirectionalBandwidthMatrix(numGPUs, true);


    printf("P2P=Disabled Latency Matrix (us)\n");
    outputLatencyMatrix(numGPUs, false);
    printf("P2P=Enabled Latency Matrix (us)\n");
    outputLatencyMatrix(numGPUs, true);

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    exit(EXIT_SUCCESS);
}
