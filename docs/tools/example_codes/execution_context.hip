// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#ifndef __HIP_PLATFORM_AMD__
    // On the CUDA backend, execution contexts map to CUDA green contexts, which
    // are provided by the CUDA driver API (this header requires linking the CUDA
    // driver library, -lcuda).
    #include <cuda.h>
#endif

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#ifndef __HIP_PLATFORM_AMD__
/// \brief Aborts on a failed CUDA driver API call, mirroring HIP_CHECK.
#define CU_CHECK(condition)                                                                    \
    {                                                                                          \
        const CUresult error = (condition);                                                    \
        if(error != CUDA_SUCCESS)                                                              \
        {                                                                                      \
            const char* name = nullptr;                                                        \
            const char* desc = nullptr;                                                        \
            cuGetErrorName(error, &name);                                                       \
            cuGetErrorString(error, &desc);                                                     \
            std::cerr << "CUDA driver error: " << (name ? name : "?") << " ("                   \
                      << (desc ? desc : "?") << ") at " << __FILE__ << ':' << __LINE__          \
                      << std::endl;                                                             \
            std::exit(error_exit_code);                                                         \
        }                                                                                      \
    }
#endif

// The device SM/CU resource type differs between backends but is used the same
// way, so alias it to keep the partitioning code uniform.
#ifdef __HIP_PLATFORM_AMD__
using device_sm_resource = hipDevResource;
#else
using device_sm_resource = CUdevResource;
#endif

// [busy-kernel-start]
/// \brief A kernel that keeps every thread block busy for a fixed number of
/// iterations. It is used as a proxy for a compute-bound workload. The volatile
/// accumulator prevents the compiler from optimizing the loop away.
__global__ void busy_kernel(unsigned int* out, const unsigned int iterations)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    volatile unsigned int acc = 0;
    for(unsigned int i = 0; i < iterations; ++i)
    {
        acc += i;
    }

    // Write the result so the loop has an observable side effect.
    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = acc;
    }
}
// [busy-kernel-end]

/// \brief Sizing of the two concurrent workloads. The background kernel is a
/// large, device-saturating job; the critical kernel is a small,
/// latency-sensitive job whose completion time is what the example measures.
struct workload
{
    unsigned int block_size;
    unsigned int background_grid_size;
    unsigned int critical_grid_size;
    unsigned int background_iterations;
    unsigned int critical_iterations;
};

/// \brief GPU time in milliseconds for each of the two concurrent kernels.
struct timings
{
    float background_ms;
    float critical_ms;
};

// [timing-helper-start]
/// \brief Launches the long-running background kernel and the shorter critical
/// kernel on two separate streams so they execute concurrently, and returns the
/// GPU time each kernel takes. The critical kernel's time is the latency of
/// interest: when the two streams share the device's compute units it contends
/// with the background kernel, but when the streams belong to execution contexts
/// backed by disjoint compute units it runs undisturbed.
timings time_critical_with_background(hipStream_t     background_stream,
                                      hipStream_t     critical_stream,
                                      unsigned int*   d_out_background,
                                      unsigned int*   d_out_critical,
                                      const workload& wl)
{
    hipEvent_t bg_start, bg_stop, crit_start, crit_stop;
    HIP_CHECK(hipEventCreate(&bg_start));
    HIP_CHECK(hipEventCreate(&bg_stop));
    HIP_CHECK(hipEventCreate(&crit_start));
    HIP_CHECK(hipEventCreate(&crit_stop));

    // Launch and time the long-running background kernel. The launch returns
    // immediately and the kernel keeps executing on its own stream, occupying
    // the device while the critical kernel below runs concurrently.
    HIP_CHECK(hipEventRecord(bg_start, background_stream));
    busy_kernel<<<dim3(wl.background_grid_size), dim3(wl.block_size), 0, background_stream>>>(
        d_out_background,
        wl.background_iterations);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(bg_stop, background_stream));

    // Time the critical kernel on its stream while the background kernel runs.
    // The elapsed time includes any wait for compute units, which is exactly the
    // latency the partitioned path is meant to remove.
    HIP_CHECK(hipEventRecord(crit_start, critical_stream));
    busy_kernel<<<dim3(wl.critical_grid_size), dim3(wl.block_size), 0, critical_stream>>>(
        d_out_critical,
        wl.critical_iterations);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(crit_stop, critical_stream));

    // Wait for both kernels to finish before reading the timers.
    HIP_CHECK(hipEventSynchronize(crit_stop));
    HIP_CHECK(hipEventSynchronize(bg_stop));

    timings result = {};
    HIP_CHECK(hipEventElapsedTime(&result.background_ms, bg_start, bg_stop));
    HIP_CHECK(hipEventElapsedTime(&result.critical_ms, crit_start, crit_stop));

    HIP_CHECK(hipEventDestroy(bg_start));
    HIP_CHECK(hipEventDestroy(bg_stop));
    HIP_CHECK(hipEventDestroy(crit_start));
    HIP_CHECK(hipEventDestroy(crit_stop));

    return result;
}
// [timing-helper-end]

#ifdef __HIP_PLATFORM_AMD__
// [partitioned-start]
/// \brief Runs one partitioned measurement. Splits the device's CUs into a
/// background group and a critical group of (about) requested_critical_cus CUs,
/// creates an execution context and stream for each, times the critical kernel
/// while the background kernel runs, then tears the contexts down. The actual CU
/// counts (which the split API may adjust for alignment) are written back
/// through out_background_cus and out_critical_cus.
timings time_partitioned_case(int             device_id,
                              hipDevResource& all_cu_resources,
                              unsigned int    total_cus,
                              unsigned int    requested_critical_cus,
                              const workload& wl,
                              unsigned int*   d_out_background,
                              unsigned int*   d_out_critical,
                              unsigned int&   out_background_cus,
                              unsigned int&   out_critical_cus)
{
    // Split the CUs into two groups: the rest for the background kernel and
    // requested_critical_cus for the critical kernel. The split API can adjust
    // the counts to satisfy architecture alignment requirements.
    hipDevResource              groups[2]       = {};
    hipDevSmResourceGroupParams group_params[2] = {
        {/*smCount=*/total_cus - requested_critical_cus, 0, 0, 0},
        {/*smCount=*/requested_critical_cus, 0, 0, 0}};

    HIP_CHECK(hipDevSmResourceSplit(&groups[0],
                                    2,
                                    &all_cu_resources,
                                    nullptr /* remainder */,
                                    0 /* flags */,
                                    &group_params[0]));

    out_background_cus = groups[0].sm.smCount;
    out_critical_cus   = groups[1].sm.smCount;

    // Wrap each group in a descriptor and create an execution context from it.
    hipDevResourceDesc_t desc_background = {};
    hipDevResourceDesc_t desc_critical   = {};
    HIP_CHECK(hipDevResourceGenerateDesc(&desc_background, &groups[0], 1));
    HIP_CHECK(hipDevResourceGenerateDesc(&desc_critical, &groups[1], 1));

    hipExecutionCtx_t ctx_background = {};
    hipExecutionCtx_t ctx_critical   = {};
    HIP_CHECK(hipGreenCtxCreate(&ctx_background, desc_background, device_id, 0));
    HIP_CHECK(hipGreenCtxCreate(&ctx_critical, desc_critical, device_id, 0));

    // A stream created on a context keeps its kernels inside that context's CUs.
    hipStream_t background_stream, critical_stream;
    HIP_CHECK(hipExecutionCtxStreamCreate(&background_stream, ctx_background, hipStreamDefault, 0));
    HIP_CHECK(hipExecutionCtxStreamCreate(&critical_stream, ctx_critical, hipStreamDefault, 0));

    const timings result = time_critical_with_background(background_stream,
                                                         critical_stream,
                                                         d_out_background,
                                                         d_out_critical,
                                                         wl);

    HIP_CHECK(hipStreamDestroy(background_stream));
    HIP_CHECK(hipStreamDestroy(critical_stream));
    HIP_CHECK(hipExecutionCtxDestroy(ctx_background));
    HIP_CHECK(hipExecutionCtxDestroy(ctx_critical));

    return result;
}
// [partitioned-end]
#else
/// \brief Creates a CUDA green context from a single SM resource group and a
/// non-blocking stream on it. The stream is returned as a cudaStream_t so the
/// runtime-API kernel launches (<<<>>>) and events used by the timing helper
/// work unchanged.
CUgreenCtx make_green_context(CUdevice cu_device, CUdevResource& group, cudaStream_t& out_stream)
{
    CUdevResourceDesc desc = {};
    CU_CHECK(cuDevResourceGenerateDesc(&desc, &group, 1));

    CUgreenCtx green_ctx = {};
    CU_CHECK(cuGreenCtxCreate(&green_ctx, desc, cu_device, CU_GREEN_CTX_DEFAULT_STREAM));

    CUstream cu_stream = {};
    CU_CHECK(cuGreenCtxStreamCreate(&cu_stream, green_ctx, CU_STREAM_NON_BLOCKING, 0));
    out_stream = reinterpret_cast<cudaStream_t>(cu_stream);

    return green_ctx;
}

/// \brief CUDA counterpart of the partitioned measurement. Splits the device's
/// SMs into two disjoint groups using green contexts - a critical group of
/// (about) requested_critical_cus SMs and the remainder for the background
/// kernel - times the critical kernel while the background kernel runs, then
/// tears the green contexts down. The actual SM counts (which the split API may
/// round for alignment) are written back through out_background_cus and
/// out_critical_cus.
timings time_partitioned_case(int             device_id,
                              CUdevResource&  all_cu_resources,
                              unsigned int    total_cus,
                              unsigned int    requested_critical_cus,
                              const workload& wl,
                              unsigned int*   d_out_background,
                              unsigned int*   d_out_critical,
                              unsigned int&   out_background_cus,
                              unsigned int&   out_critical_cus)
{
    (void)total_cus;

    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_id));

    // A single split carves the device into a critical group of the requested
    // size and a disjoint remainder for the background kernel, so the two green
    // contexts never share SMs.
    CUdevResource critical_group   = {};
    CUdevResource background_group = {};
    unsigned int  group_count      = 1;
    CU_CHECK(cuDevSmResourceSplitByCount(&critical_group,
                                         &group_count,
                                         &all_cu_resources,
                                         &background_group /* remaining */,
                                         0 /* flags */,
                                         requested_critical_cus));
    out_critical_cus   = critical_group.sm.smCount;
    out_background_cus = background_group.sm.smCount;

    cudaStream_t background_stream, critical_stream;
    CUgreenCtx   ctx_background = make_green_context(cu_device, background_group, background_stream);
    CUgreenCtx   ctx_critical   = make_green_context(cu_device, critical_group, critical_stream);

    const timings result = time_critical_with_background(background_stream,
                                                         critical_stream,
                                                         d_out_background,
                                                         d_out_critical,
                                                         wl);

    CU_CHECK(cuStreamDestroy(reinterpret_cast<CUstream>(background_stream)));
    CU_CHECK(cuStreamDestroy(reinterpret_cast<CUstream>(critical_stream)));
    CU_CHECK(cuGreenCtxDestroy(ctx_background));
    CU_CHECK(cuGreenCtxDestroy(ctx_critical));

    return result;
}
#endif

#ifdef __HIP_PLATFORM_AMD__
/// \brief Reads and prints a device's compute-unit resource and its work queue
/// configuration. These read-only queries are the starting point before
/// partitioning a device into execution contexts.
void demonstrate_reading_resources()
{
    // [read-cu-resource-start]
    int current_device = 0;
    HIP_CHECK(hipSetDevice(current_device));

    hipDevResource cu_resources = {};
    HIP_CHECK(hipDeviceGetDevResource(current_device, &cu_resources, hipDevResourceTypeSm));

    std::cout << "Device ID: " << current_device << "\n";
    std::cout << "Available CUs: " << cu_resources.sm.smCount << "\n";
    std::cout << "Min. partition size: " << cu_resources.sm.minSmPartitionSize << "\n";
    std::cout << "Co-scheduled alignment: " << cu_resources.sm.smCoscheduledAlignment << "\n";
    // [read-cu-resource-end]

    // [read-wq-config-start]
    // Reading the work queue configuration from a device is not supported on
    // every ROCm runtime. Capture the status instead of aborting, and only print
    // the configuration when the query succeeds.
    hipDevResource   wq_config = {};
    const hipError_t wq_status
        = hipDeviceGetDevResource(current_device, &wq_config, hipDevResourceTypeWorkqueueConfig);

    if(wq_status == hipSuccess)
    {
        std::cout << "WQ concurrency limit: " << wq_config.wqConfig.wqConcurrencyLimit << "\n";
        std::cout << "WQ sharing scope: " << wq_config.wqConfig.sharingScope << "\n";
    }
    else
    {
        std::cout << "Work queue configuration query is not supported on this runtime ("
                  << hipGetErrorString(wq_status) << ").\n";

        // A failed call leaves its error in the thread's sticky last-error state.
        // Clear it with hipGetLastError so a later hipGetLastError check does not
        // misattribute this error to an unrelated, successful launch.
        (void)hipGetLastError();
    }
    // [read-wq-config-end]
}
#endif

/// \brief Prints one row of the results table.
void print_row(const std::string& configuration,
               const std::string& critical_cus,
               const std::string& background_cus,
               float              critical_ms,
               float              background_ms,
               float              speedup)
{
    std::cout << "  " << std::left << std::setw(20) << configuration << std::setw(14)
              << critical_cus << std::setw(16) << background_cus << std::setw(16)
              << (double_precision(critical_ms, 2) + " ms") << std::setw(16)
              << (double_precision(background_ms, 2) + " ms") << double_precision(speedup, 2) << "x"
              << std::endl;
}

int main()
{
    constexpr int device_id = 0;
    HIP_CHECK(hipSetDevice(device_id));

    // Determine the number of compute units (CUs, called SMs on NVIDIA) on the
    // device. This count drives both how the resources are partitioned and how
    // the launch grids are sized.
    unsigned int total_cus = 0;

    // The device SM/CU resource that partitioning is carved from. On AMD the
    // field is named smCount for CUDA source compatibility; there it represents
    // compute units.
    device_sm_resource all_cu_resources = {};

#ifdef __HIP_PLATFORM_AMD__
    // Query the total number of CUs through the device resource.
    HIP_CHECK(hipDeviceGetDevResource(device_id, &all_cu_resources, hipDevResourceTypeSm));
    total_cus = all_cu_resources.sm.smCount;

    // Show the read-only resource queries that precede partitioning.
    demonstrate_reading_resources();
#else
    // The CUDA green-context flow uses the driver API. Initialize it, get a
    // driver device handle, and query its SM resource, which is the whole-device
    // resource that the partition sweep splits below.
    CU_CHECK(cuInit(0));
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_id));
    CU_CHECK(cuDeviceGetDevResource(cu_device, &all_cu_resources, CU_DEV_RESOURCE_TYPE_SM));
    total_cus = all_cu_resources.sm.smCount;

    std::cout << "Compute units on device " << device_id << ": " << total_cus << std::endl;
#endif

    constexpr unsigned int min_required_cus = 4;
    if(total_cus < min_required_cus)
    {
        std::cout << "Device has too few compute units (" << total_cus << ") for this example. "
                  << "At least " << min_required_cus << " are required." << std::endl;
        return 0;
    }

    // Partitions must be a multiple of the device's co-scheduled CU alignment.
    // Requesting an unaligned count makes the split fail, so snap requests to it.
#ifdef __HIP_PLATFORM_AMD__
    // On AMD this is the WGP granularity (typically 2), reported by the resource.
    unsigned int cu_alignment = all_cu_resources.sm.smCoscheduledAlignment;
#else
    // The CUDA SM resource does not expose an alignment field, and
    // cuDevSmResourceSplitByCount rounds internally, so use 1 (no snapping).
    unsigned int cu_alignment = 1;
#endif
    if(cu_alignment < 1)
    {
        cu_alignment = 1;
    }

    // Workload sizing. The background kernel launches many more blocks than the
    // device can run at once (deep oversubscription) and each block runs for a
    // long time, so it keeps every CU busy for the whole measurement. The
    // critical kernel is kept small - a couple of thread blocks - so it fits
    // inside even the smallest partition without oversubscribing it; that way its
    // runtime reflects how long it waits for CUs, not how many CUs it has.
    workload wl;
    wl.block_size            = 256;
    wl.background_grid_size   = total_cus * 16;
    wl.critical_grid_size     = 2 * cu_alignment; // small, fits the smallest partition
    wl.background_iterations  = 2000000;
    wl.critical_iterations    = 500000; // fixed critical work across all cases

    std::cout
        << "\nThis example runs a small, latency-sensitive \"critical\" kernel at the same time\n"
           "as a large \"background\" kernel that saturates the GPU, and measures how long the\n"
           "critical kernel takes to complete. It first runs a baseline where both kernels share\n"
           "all CUs, then sweeps several execution-context partitions that give the critical\n"
           "kernel a growing slice of dedicated CUs. As the critical kernel gets more of its own\n"
           "CUs, its latency drops well below the contended baseline.\n"
        << std::endl;

    // The critical buffer only needs one entry per critical block; the critical
    // grid size is fixed, so this size is reused across every case.
    unsigned int* d_out_background = nullptr;
    unsigned int* d_out_critical   = nullptr;
    HIP_CHECK(hipMalloc(&d_out_background, sizeof(unsigned int) * wl.background_grid_size));
    HIP_CHECK(hipMalloc(&d_out_critical, sizeof(unsigned int) * wl.critical_grid_size));

    // Warm up the GPU so one-time costs (module load, first-launch setup) are not
    // charged to the first measured case. Write into the background buffer, which
    // is large enough for a full-device grid. The result is discarded.
    {
        hipStream_t warmup;
        HIP_CHECK(hipStreamCreateWithFlags(&warmup, hipStreamNonBlocking));
        busy_kernel<<<dim3(total_cus), dim3(wl.block_size), 0, warmup>>>(d_out_background, 1000);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(warmup));
        HIP_CHECK(hipStreamDestroy(warmup));
    }

    // [baseline-start]
    // Baseline: both kernels share all of the device's CUs. Two ordinary
    // non-blocking streams let them run concurrently and contend for CUs.
    hipStream_t shared_background, shared_critical;
    HIP_CHECK(hipStreamCreateWithFlags(&shared_background, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&shared_critical, hipStreamNonBlocking));

    const timings baseline = time_critical_with_background(shared_background,
                                                           shared_critical,
                                                           d_out_background,
                                                           d_out_critical,
                                                           wl);

    HIP_CHECK(hipStreamDestroy(shared_background));
    HIP_CHECK(hipStreamDestroy(shared_critical));
    // [baseline-end]

    // Make sure the device is fully idle before the first partitioned case
    // creates and destroys execution contexts.
    HIP_CHECK(hipDeviceSynchronize());

    // Header for the results table.
    std::cout << "Kernel runtimes while both kernels run concurrently:\n" << std::endl;
    std::cout << "  " << std::left << std::setw(20) << "Configuration" << std::setw(14)
              << "Critical CUs" << std::setw(16) << "Background CUs" << std::setw(16)
              << "Critical" << std::setw(16) << "Background" << "Speedup" << std::endl;
    print_row("Baseline (shared)",
              "all (" + std::to_string(total_cus) + ")",
              "all (" + std::to_string(total_cus) + ")",
              baseline.critical_ms,
              baseline.background_ms,
              1.0f);

    // Sweep several partition sizes for the critical kernel: an eighth, a
    // quarter, and half of the device. Each request is snapped to the CU
    // alignment and clamped so both groups keep at least one aligned block of
    // CUs; duplicates (on small devices) are skipped.
    const unsigned int candidates[]  = {total_cus / 8, total_cus / 4, total_cus / 2};
    unsigned int       last_critical = 0;

    for(const unsigned int candidate : candidates)
    {
        // Round the request up to a multiple of the alignment, keeping at least
        // one aligned block for the critical group and one for the background
        // group.
        unsigned int requested = ((candidate + cu_alignment - 1) / cu_alignment) * cu_alignment;
        if(requested < cu_alignment)
        {
            requested = cu_alignment;
        }
        if(requested > total_cus - cu_alignment)
        {
            requested = total_cus - cu_alignment;
        }
        // Skip a request that resolves to the same size as the previous one.
        if(requested == last_critical)
        {
            continue;
        }
        last_critical = requested;

        unsigned int  background_cus = 0;
        unsigned int  critical_cus   = 0;
        const timings result         = time_partitioned_case(device_id,
                                                     all_cu_resources,
                                                     total_cus,
                                                     requested,
                                                     wl,
                                                     d_out_background,
                                                     d_out_critical,
                                                     background_cus,
                                                     critical_cus);

        const float speedup
            = result.critical_ms > 0.0f ? baseline.critical_ms / result.critical_ms : 0.0f;
        print_row("Partitioned",
                  std::to_string(critical_cus),
                  std::to_string(background_cus),
                  result.critical_ms,
                  result.background_ms,
                  speedup);

        // Return the device to an idle state before the next case tears down and
        // rebuilds execution contexts.
        HIP_CHECK(hipDeviceSynchronize());
    }

    std::cout << "\nThe baseline critical kernel waits behind the background kernel for the shared\n"
                 "CUs. With an execution context it runs on its own CUs and finishes sooner; the\n"
                 "more CUs the partition holds, the lower its latency. The background kernel is\n"
                 "confined to fewer CUs as the critical partition grows, so its runtime rises."
              << std::endl;

    HIP_CHECK(hipFree(d_out_background));
    HIP_CHECK(hipFree(d_out_critical));

    std::cout << "\nExecution context example completed successfully." << std::endl;

    return 0;
}
