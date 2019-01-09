// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <utility>
#include <type_traits>
#include <algorithm>

#include "cmdparser.hpp"
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include <hiprand.h>
#include <curand.h>
// CHECK: #include <hiprand_kernel.h>
#include <curand_kernel.h>
// CHECK-NOT: #include <curand_mtgp32_host.h>
// CHECK-NOT: #include <curand_mtgp32dc_p_11213.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

// CHECK: if ((x) != hipSuccess) {
#define CUDA_CALL(x)                                                                               \
    do {                                                                                           \
        if ((x) != cudaSuccess) {                                                                  \
            printf("Error at %s:%d\n", __FILE__, __LINE__);                                        \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
// CHECK: if ((x) != HIPRAND_STATUS_SUCCESS) {
#define CURAND_CALL(x)                                                                             \
    do {                                                                                           \
        if ((x) != CURAND_STATUS_SUCCESS) {                                                        \
            printf("Error at %s:%d\n", __FILE__, __LINE__);                                        \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#ifndef DEFAULT_RAND_N
const size_t DEFAULT_RAND_N = 1024 * 1024 * 128;
#endif

size_t next_power2(size_t x)
{
    size_t power = 1;
    while (power < x)
    {
        power *= 2;
    }
    return power;
}

template<typename GeneratorState>
__global__
void init_kernel(GeneratorState * states,
                 const unsigned long long seed,
                 const unsigned long long offset)
{
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    GeneratorState state;
    // CHECK: hiprand_init(seed, state_id, offset, &state);
    curand_init(seed, state_id, offset, &state);
    states[state_id] = state;
}

template<typename GeneratorState, typename T, typename GenerateFunc, typename Extra>
__global__
void generate_kernel(GeneratorState * states,
                     T * data,
                     const size_t size,
                     const GenerateFunc& generate_func,
                     const Extra extra)
{
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;

    GeneratorState state = states[state_id];
    unsigned int index = state_id;
    while(index < size)
    {
        data[index] = generate_func(&state, extra);
        index += stride;
    }
    states[state_id] = state;
}

template<typename GeneratorState>
struct runner
{
    GeneratorState * states;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long seed,
           const unsigned long long offset)
    {
        const size_t states_size = blocks * threads;
        // CHECK: CUDA_CALL(hipMalloc((void **)&states, states_size * sizeof(GeneratorState)));
        CUDA_CALL(cudaMalloc((void **)&states, states_size * sizeof(GeneratorState)));
        // CHECK: hipLaunchKernelGGL(init_kernel, dim3(blocks), dim3(threads), 0, 0, states, seed, offset);
        init_kernel<<<blocks, threads>>>(states, seed, offset);
        // CHECK: CUDA_CALL(hipPeekAtLastError());
        // CHECK: CUDA_CALL(hipDeviceSynchronize());
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    ~runner()
    {
        CUDA_CALL(cudaFree(states));
    }

    template<typename T, typename GenerateFunc, typename Extra>
    void generate(const size_t blocks,
                  const size_t threads,
                  T * data,
                  const size_t size,
                  const GenerateFunc& generate_func,
                  const Extra extra)
    {
        // CHECK: hipLaunchKernelGGL(generate_kernel, dim3(blocks), dim3(threads), 0, 0, states, data, size, generate_func, extra);
        generate_kernel<<<blocks, threads>>>(states, data, size, generate_func, extra);
    }
};

// CHECK: void generate_kernel(hiprandStateMtgp32_t * states,
template<typename T, typename GenerateFunc, typename Extra>
__global__
void generate_kernel(curandStateMtgp32_t * states,
                     T * data,
                     const size_t size,
                     const GenerateFunc& generate_func,
                     const Extra extra)
{
    const unsigned int state_id = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    // CHECK: __shared__ hiprandStateMtgp32_t state;
    __shared__ curandStateMtgp32_t state;

    if (thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r = size%blockDim.x;
    const size_t size_rounded_up = r == 0 ? size : size + (blockDim.x - r);
    while(index < size_rounded_up)
    {
        auto value = generate_func(&state, extra);
        if(index < size)
            data[index] = value;
        index += stride;
    }
    __syncthreads();

    if (thread_id == 0)
        states[state_id] = state;
}

// CHECK: struct runner<hiprandStateMtgp32_t>
template<>
struct runner<curandStateMtgp32_t>
{
    // CHECK: hiprandStateMtgp32_t * states;
    curandStateMtgp32_t * states;
    mtgp32_kernel_params_t * d_param;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long seed,
           const unsigned long long offset)
    {
        const size_t states_size = std::min((size_t)200, blocks);
        // CHECK: CUDA_CALL(hipMalloc((void **)&states, states_size * sizeof(hiprandStateMtgp32_t)));
        CUDA_CALL(cudaMalloc((void **)&states, states_size * sizeof(curandStateMtgp32_t)));
        // CHECK: CUDA_CALL(hipMalloc((void **)&d_param, sizeof(mtgp32_kernel_params)));
        CUDA_CALL(cudaMalloc((void **)&d_param, sizeof(mtgp32_kernel_params)));
        // curandMakeMTGP32Constants is yet unsupported by HIP
        // CHECK-NOT: CURAND_CALL(hiprandMakeMTGP32Constants(mtgp32dc_params_fast_11213, d_param));
        CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, d_param));
        // curandMakeMTGP32KernelState is yet unsupported by HIP
        // CHECK-NOT: CURAND_CALL(hiprandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213, d_param, states_size, seed));
        CURAND_CALL(curandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213, d_param, states_size, seed));
    }

    ~runner()
    {
        // CHECK: CUDA_CALL(hipFree(states));
        // CHECK: CUDA_CALL(hipFree(d_param));
        CUDA_CALL(cudaFree(states));
        CUDA_CALL(cudaFree(d_param));
    }

    template<typename T, typename GenerateFunc, typename Extra>
    void generate(const size_t blocks,
                  const size_t threads,
                  T * data,
                  const size_t size,
                  const GenerateFunc& generate_func,
                  const Extra extra)
    {
        // CHECK: hipLaunchKernelGGL(generate_kernel, dim3(std::min((size_t)200, blocks)), dim3(256), 0, 0, states, data, size, generate_func, extra);
        generate_kernel<<<std::min((size_t)200, blocks), 256>>>(states, data, size, generate_func, extra);
    }
};

// CHECK: void init_kernel(hiprandStateSobol32_t * states,
template<typename Directions>
__global__
void init_kernel(curandStateSobol32_t * states,
                 const Directions directions,
                 const unsigned long long offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    // CHECK: hiprandStateSobol32_t state;
    // CHECK: hiprand_init(directions[dimension], offset + state_id, &state);
    curandStateSobol32_t state;
    curand_init(directions[dimension], offset + state_id, &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

// CHECK: void generate_kernel(hiprandStateSobol32_t * states,
template<typename T, typename GenerateFunc, typename Extra>
__global__
void generate_kernel(curandStateSobol32_t * states,
                     T * data,
                     const size_t size,
                     const GenerateFunc& generate_func,
                     const Extra extra)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    // CHECK: hiprandStateSobol32_t state = states[gridDim.x * blockDim.x * dimension + state_id];
    curandStateSobol32_t state = states[gridDim.x * blockDim.x * dimension + state_id];
    const unsigned int offset = dimension * size;
    unsigned int index = state_id;
    while(index < size)
    {
        data[offset + index] = generate_func(&state, extra);
        skipahead(stride - 1, &state);
        index += stride;
    }
    state = states[gridDim.x * blockDim.x * dimension + state_id];
    skipahead(static_cast<unsigned int>(size), &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

// CHECK: struct runner<hiprandStateSobol32_t>
template<>
struct runner<curandStateSobol32_t>
{
    // CHECK: hiprandStateSobol32_t * states;
    curandStateSobol32_t * states;
    size_t dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long seed,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;
        // CHECK: CUDA_CALL(hipMalloc((void **)&states, states_size * sizeof(hiprandStateSobol32_t)));
        const size_t states_size = blocks * threads * dimensions;
        CUDA_CALL(cudaMalloc((void **)&states, states_size * sizeof(curandStateSobol32_t)));
        // CHECK: hiprandDirectionVectors32_t * directions;
        curandDirectionVectors32_t * directions;
        // CHECK: const size_t size = dimensions * sizeof(hiprandDirectionVectors32_t);
        const size_t size = dimensions * sizeof(curandDirectionVectors32_t);
        // CHECK: CUDA_CALL(hipMalloc((void **)&directions, size));
        CUDA_CALL(cudaMalloc((void **)&directions, size));
        // CHECK: hiprandDirectionVectors32_t * h_directions;
        curandDirectionVectors32_t * h_directions;
        // hiprandGetDirectionVectors32 and HIPRAND_DIRECTION_VECTORS_32_JOEKUO6 (of hiprandDirectionVectorSet_t) are yet unsupported by HIP
        // CHECK-NOT: CURAND_CALL(hiprandGetDirectionVectors32(&h_directions, HIPRAND_DIRECTION_VECTORS_32_JOEKUO6));
        CURAND_CALL(curandGetDirectionVectors32(&h_directions, CURAND_DIRECTION_VECTORS_32_JOEKUO6));
        // CHECK: CUDA_CALL(hipMemcpy(directions, h_directions, size, hipMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(directions, h_directions, size, cudaMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        // CHECK: hipLaunchKernelGGL(init_kernel, dim3(dim3(blocks_x, dimensions)), dim3(threads), 0, 0, states, directions, offset);
        init_kernel<<<dim3(blocks_x, dimensions), threads>>>(states, directions, offset);
        // CHECK: CUDA_CALL(hipPeekAtLastError());
        // CHECK: CUDA_CALL(hipDeviceSynchronize());
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        // CHECK: CUDA_CALL(hipFree(directions));
        CUDA_CALL(cudaFree(directions));
    }

    ~runner()
    {
        // CHECK: CUDA_CALL(hipFree(states));
        CUDA_CALL(cudaFree(states));
    }

    template<typename T, typename GenerateFunc, typename Extra>
    void generate(const size_t blocks,
                  const size_t threads,
                  T * data,
                  const size_t size,
                  const GenerateFunc& generate_func,
                  const Extra extra)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        // CHECK: hipLaunchKernelGGL(generate_kernel, dim3(dim3(blocks_x, dimensions)), dim3(threads), 0, 0, states, data, size / dimensions, generate_func, extra);
        generate_kernel<<<dim3(blocks_x, dimensions), threads>>>(states, data, size / dimensions, generate_func, extra);
    }
};

template<typename T, typename GeneratorState, typename GenerateFunc, typename Extra>
void run_benchmark(const cli::Parser& parser,
                   const GenerateFunc& generate_func,
                   const Extra extra)
{
    const size_t size = parser.get<size_t>("size");
    const size_t dimensions = parser.get<size_t>("dimensions");
    const size_t trials = parser.get<size_t>("trials");

    const size_t blocks = parser.get<size_t>("blocks");
    const size_t threads = parser.get<size_t>("threads");

    T * data;
    // CHECK: CUDA_CALL(hipMalloc((void **)&data, size * sizeof(T)));
    CUDA_CALL(cudaMalloc((void **)&data, size * sizeof(T)));

    runner<GeneratorState> r(dimensions, blocks, threads, 12345ULL, 6789ULL);

    // Warm-up
    for (size_t i = 0; i < 5; i++)
    {
        r.generate(blocks, threads, data, size, generate_func, extra);
        // CHECK: CUDA_CALL(hipPeekAtLastError());
        // CHECK: CUDA_CALL(hipDeviceSynchronize());
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }
    // CHECK: CUDA_CALL(hipDeviceSynchronize());
    CUDA_CALL(cudaDeviceSynchronize());

    // Measurement
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < trials; i++)
    {
        r.generate(blocks, threads, data, size, generate_func, extra);
    }
    // CHECK: CUDA_CALL(hipPeekAtLastError());
    // CHECK: CUDA_CALL(hipDeviceSynchronize());
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(3)
              << "      "
              << "Throughput = "
              << std::setw(8) << (trials * size * sizeof(T)) /
                    (elapsed.count() / 1e3 * (1 << 30))
              << " GB/s, Samples = "
              << std::setw(8) << (trials * size) /
                    (elapsed.count() / 1e3 * (1 << 30))
              << " GSample/s, AvgTime (1 trial) = "
              << std::setw(8) << elapsed.count() / trials
              << " ms, Time (all) = "
              << std::setw(8) << elapsed.count()
              << " ms, Size = " << size
              << std::endl;
    // CHECK: CUDA_CALL(hipFree(data));
    CUDA_CALL(cudaFree(data));
}

template<typename GeneratorState>
void run_benchmarks(const cli::Parser& parser,
                    const std::string& distribution)
{
    if (distribution == "uniform-uint")
    {
        // curandStateSobol64_t and curandStateScrambledSobol64_t are yet unsupported by HIP
        // CHECK-NOT: if (!std::is_same<GeneratorState, hiprandStateSobol64_t>::value &&
        // CHECK-NOT: !std::is_same<GeneratorState, hiprandStateScrambledSobol64_t>::value)
        if (!std::is_same<GeneratorState, curandStateSobol64_t>::value &&
            !std::is_same<GeneratorState, curandStateScrambledSobol64_t>::value)
        {
            run_benchmark<unsigned int, GeneratorState>(parser,
                [] __device__ (GeneratorState * state, int) {
                    // CHECK: return hiprand(state);
                    return curand(state);
                }, 0
            );
        }
    }
    if (distribution == "uniform-long-long")
    {
        // curandStateSobol64_t and curandStateScrambledSobol64_t are yet unsupported by HIP
        // CHECK-NOT: if (!std::is_same<GeneratorState, hiprandStateSobol64_t>::value &&
        // CHECK-NOT: !std::is_same<GeneratorState, hiprandStateScrambledSobol64_t>::value)
        if (std::is_same<GeneratorState, curandStateSobol64_t>::value ||
            std::is_same<GeneratorState, curandStateScrambledSobol64_t>::value)
        {
            run_benchmark<unsigned long long, GeneratorState>(parser,
                [] __device__ (GeneratorState * state, int) {
                    // CHECK: return hiprand(state);
                    return curand(state);
                }, 0
            );
        }
    }
    if (distribution == "uniform-float")
    {
        run_benchmark<float, GeneratorState>(parser,
            [] __device__ (GeneratorState * state, int) {
                // CHECK: return hiprand_uniform(state);
                return curand_uniform(state);
            }, 0
        );
    }
    if (distribution == "uniform-double")
    {
        run_benchmark<double, GeneratorState>(parser,
            [] __device__ (GeneratorState * state, int) {
                // CHECK: return hiprand_uniform_double(state);
                return curand_uniform_double(state);
            }, 0
        );
    }
    if (distribution == "normal-float")
    {
        run_benchmark<float, GeneratorState>(parser,
            [] __device__ (GeneratorState * state, int) {
                // CHECK: return hiprand_normal(state);
                return curand_normal(state);
            }, 0
        );
    }
    if (distribution == "normal-double")
    {
        run_benchmark<double, GeneratorState>(parser,
            [] __device__ (GeneratorState * state, int) {
                // CHECK: return hiprand_normal_double(state);
                return curand_normal_double(state);
            }, 0
        );
    }
    if (distribution == "log-normal-float")
    {
        run_benchmark<float, GeneratorState>(parser,
            [] __device__ (GeneratorState * state, int) {
                // CHECK: return hiprand_log_normal(state, 0.0f, 1.0f);
                return curand_log_normal(state, 0.0f, 1.0f);
            }, 0
        );
    }
    if (distribution == "log-normal-double")
    {
        run_benchmark<double, GeneratorState>(parser,
            [] __device__ (GeneratorState * state, int) {
                // CHECK: return hiprand_log_normal_double(state, 0.0, 1.0);
                return curand_log_normal_double(state, 0.0, 1.0);
            }, 0
        );
    }
    if (distribution == "poisson")
    {
        const auto lambdas = parser.get<std::vector<double>>("lambda");
        for (double lambda : lambdas)
        {
            std::cout << "    " << "lambda "
                 << std::fixed << std::setprecision(1) << lambda << std::endl;
            run_benchmark<unsigned int, GeneratorState>(parser,
                [] __device__ (GeneratorState * state, double lambda) {
                    // CHECK: return hiprand_poisson(state, lambda);
                    return curand_poisson(state, lambda);
                }, lambda
            );
        }
    }
    if (distribution == "discrete-poisson")
    {
        const auto lambdas = parser.get<std::vector<double>>("lambda");
        for (double lambda : lambdas)
        {
            std::cout << "    " << "lambda "
                 << std::fixed << std::setprecision(1) << lambda << std::endl;
            // CHECK: hiprandDiscreteDistribution_t discrete_distribution;
            curandDiscreteDistribution_t discrete_distribution;
            // CHECK: CURAND_CALL(hiprandCreatePoissonDistribution(lambda, &discrete_distribution));
            CURAND_CALL(curandCreatePoissonDistribution(lambda, &discrete_distribution));
            run_benchmark<unsigned int, GeneratorState>(parser,
                // CHECK: [] __device__ (GeneratorState * state, hiprandDiscreteDistribution_t discrete_distribution) {
                [] __device__ (GeneratorState * state, curandDiscreteDistribution_t discrete_distribution) {
                    // CHECK: return hiprand_discrete(state, discrete_distribution);
                    return curand_discrete(state, discrete_distribution);
                }, discrete_distribution
            );
            // CHECK: CURAND_CALL(hiprandDestroyDistribution(discrete_distribution));
            CURAND_CALL(curandDestroyDistribution(discrete_distribution));
        }
    }
}

const std::vector<std::string> all_engines = {
    "xorwow",
    "mrg32k3a",
    "mtgp32",
    // "mt19937",
    "philox",
    "sobol32",
    // "scrambled_sobol32",
    // "sobol64",
    // "scrambled_sobol64",
};

const std::vector<std::string> all_distributions = {
    "uniform-uint",
    // "uniform-long-long",
    "uniform-float",
    "uniform-double",
    "normal-float",
    "normal-double",
    "log-normal-float",
    "log-normal-double",
    "poisson",
    "discrete-poisson",
};

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);

    const std::string distribution_desc =
        "space-separated list of distributions:" +
        std::accumulate(all_distributions.begin(), all_distributions.end(), std::string(),
            [](std::string a, std::string b) {
                return a + "\n      " + b;
            }
        ) +
        "\n      or all";
    const std::string engine_desc =
        "space-separated list of random number engines:" +
        std::accumulate(all_engines.begin(), all_engines.end(), std::string(),
            [](std::string a, std::string b) {
                return a + "\n      " + b;
            }
        ) +
        "\n      or all";

    parser.set_optional<size_t>("size", "size", DEFAULT_RAND_N, "number of values");
    parser.set_optional<size_t>("dimensions", "dimensions", 1, "number of dimensions of quasi-random values");
    parser.set_optional<size_t>("trials", "trials", 20, "number of trials");
    parser.set_optional<size_t>("blocks", "blocks", 256, "number of blocks");
    parser.set_optional<size_t>("threads", "threads", 256, "number of threads in each block");
    parser.set_optional<std::vector<std::string>>("dis", "dis", {"uniform-uint"}, distribution_desc.c_str());
    parser.set_optional<std::vector<std::string>>("engine", "engine", {"philox"}, engine_desc.c_str());
    parser.set_optional<std::vector<double>>("lambda", "lambda", {10.0}, "space-separated list of lambdas of Poisson distribution");
    parser.run_and_exit_if_error();

    std::vector<std::string> engines;
    {
        auto es = parser.get<std::vector<std::string>>("engine");
        if (std::find(es.begin(), es.end(), "all") != es.end())
        {
            engines = all_engines;
        }
        else
        {
            for (auto e : all_engines)
            {
                if (std::find(es.begin(), es.end(), e) != es.end())
                    engines.push_back(e);
            }
        }
    }

    std::vector<std::string> distributions;
    {
        auto ds = parser.get<std::vector<std::string>>("dis");
        if (std::find(ds.begin(), ds.end(), "all") != ds.end())
        {
            distributions = all_distributions;
        }
        else
        {
            for (auto d : all_distributions)
            {
                if (std::find(ds.begin(), ds.end(), d) != ds.end())
                    distributions.push_back(d);
            }
        }
    }

    int version;
    // CHECK: CURAND_CALL(hiprandGetVersion(&version));
    CURAND_CALL(curandGetVersion(&version));
    int runtime_version;
    // cudaRuntimeGetVersion is yet unsupported by HIP
    // CHECK-NOT: CUDA_CALL(hipRuntimeGetVersion(&runtime_version));
    CUDA_CALL(cudaRuntimeGetVersion(&runtime_version));
    int device_id;
    // CHECK: CUDA_CALL(hipGetDevice(&device_id));
    // CHECK: hipDeviceProp_t props;
    // CHECK: CUDA_CALL(hipGetDeviceProperties(&props, device_id));
    CUDA_CALL(cudaGetDevice(&device_id));
    cudaDeviceProp props;
    CUDA_CALL(cudaGetDeviceProperties(&props, device_id));

    std::cout << "cuRAND: " << version << " ";
    std::cout << "Runtime: " << runtime_version << " ";
    std::cout << "Device: " << props.name;
    std::cout << std::endl << std::endl;

    for (auto engine : engines)
    {
        std::cout << engine << ":" << std::endl;
        for (auto distribution : distributions)
        {
            std::cout << "  " << distribution << ":" << std::endl;
            const std::string plot_name = engine + "-" + distribution;
            if (engine == "xorwow")
            {
                // CHECK: run_benchmarks<hiprandStateXORWOW_t>(parser, distribution);
                run_benchmarks<curandStateXORWOW_t>(parser, distribution);
            }
            else if (engine == "mrg32k3a")
            {
                // CHECK: run_benchmarks<hiprandStateMRG32k3a_t>(parser, distribution);
                run_benchmarks<curandStateMRG32k3a_t>(parser, distribution);
            }
            else if (engine == "philox")
            {
                // CHECK: run_benchmarks<hiprandStatePhilox4_32_10_t>(parser, distribution);
                run_benchmarks<curandStatePhilox4_32_10_t>(parser, distribution);
            }
            else if (engine == "sobol32")
            {
                // CHECK: run_benchmarks<hiprandStateSobol32_t>(parser, distribution);
                run_benchmarks<curandStateSobol32_t>(parser, distribution);
            }
            else if (engine == "mtgp32")
            {
                // CHECK: run_benchmarks<hiprandStateMtgp32_t>(parser, distribution);
                run_benchmarks<curandStateMtgp32_t>(parser, distribution);
            }
        }
    }

    return 0;
}
