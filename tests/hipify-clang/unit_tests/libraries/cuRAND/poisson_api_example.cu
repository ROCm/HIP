// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// Taken from: http://docs.nvidia.com/cuda/curand/device-api-overview.html#poisson-api-example
/*
 * This program uses CURAND library for Poisson distribution
 * to simulate queues in store for 16 hours. It shows the
 * difference of using 3 different APIs:
 * - HOST API -arrival of customers is described by Poisson(4)
 * - SIMPLE DEVICE API -arrival of customers is described by
 *     Poisson(4*(sin(x/100)+1)), where x is number of minutes
 *     from store opening time.
 * - ROBUST DEVICE API -arrival of customers is described by:
 *     - Poisson(2) for first 3 hours.
 *     - Poisson(1) for second 3 hours.
 *     - Poisson(3) after 6 hours.  
 */

#include <stdio.h>
#include <stdlib.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
// CHECK: #include <hiprand_kernel.h>
#include <curand_kernel.h>
// CHECK: #include <hiprand.h>
#include <curand.h>

// CHECK: #define CUDA_CALL(x) do { if((x) != hipSuccess) {
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)
// CHECK: #define CURAND_CALL(x) do { if((x)!=HIPRAND_STATUS_SUCCESS) {
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)


#define HOURS 16
#define OPENING_HOUR 7
#define CLOSING_HOUR (OPENING_HOUR + HOURS)

#define access_2D(type, ptr, row, column, pitch)\
    *((type*)((char*)ptr + (row) * pitch) + column)

enum API_TYPE {
    HOST_API = 0,
    SIMPLE_DEVICE_API = 1,
    ROBUST_DEVICE_API = 2,
};

/* global variables */
API_TYPE api;
int report_break;
int cashiers_load_h[HOURS];
__constant__ int cashiers_load[HOURS];
// CHECK: __global__ void setup_kernel(hiprandState_t *state)
__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    // CHECK: hiprand_init(1234, id, 0, &state[id]);
    curand_init(1234, id, 0, &state[id]);
}

__inline__ __device__
void update_queue(int id, int min, unsigned int new_customers,
                  unsigned int &queue_length,
                  unsigned int *queue_lengths, size_t pitch)
{
    int balance;
    balance = new_customers - 2 * cashiers_load[(min-1)/60];
    if (balance + (int)queue_length <= 0){
        queue_length = 0;
    }else{
        queue_length += balance;
    }
    /* Store results */
    access_2D(unsigned int, queue_lengths, min-1, id, pitch)
        = queue_length;
}

// CHECK: __global__ void simple_device_API_kernel(hiprandState_t *state,
__global__ void simple_device_API_kernel(curandState *state, 
                    unsigned int *queue_lengths, size_t pitch)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int new_customers;
    unsigned int queue_length = 0;
    /* Copy state to local memory for efficiency */
    // CHECK: hiprandState_t localState = state[id];
    curandState localState = state[id];
    /* Simulate queue in time */
    for(int min = 1; min <= 60 * HOURS; min++) {
        /* Draw number of new customers depending on API */
        // CHECK: new_customers = hiprand_poisson(&localState,
        new_customers = curand_poisson(&localState,
                                4*(sin((float)min/100.0)+1));
        /* Update queue */
        update_queue(id, min, new_customers, queue_length,
                     queue_lengths, pitch);       
    }
    /* Copy state back to global memory */
    state[id] = localState;
}


__global__ void host_API_kernel(unsigned int *poisson_numbers,
                    unsigned int *queue_lengths, size_t pitch)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int new_customers;
    unsigned int queue_length = 0;
    /* Simulate queue in time */
    for(int min = 1; min <= 60 * HOURS; min++) {
        /* Get random number from global memory */
        new_customers = poisson_numbers
                    [blockDim.x * gridDim.x * (min -1) + id];
        /* Update queue */
        update_queue(id, min, new_customers, queue_length,
                     queue_lengths, pitch);
    }
}
// CHECK: __global__ void robust_device_API_kernel(hiprandState_t *state,
// CHECK: hiprandDiscreteDistribution_t poisson_1,
// CHECK: hiprandDiscreteDistribution_t poisson_2,
// CHECK: hiprandDiscreteDistribution_t poisson_3,
__global__ void robust_device_API_kernel(curandState *state,
                   curandDiscreteDistribution_t poisson_1,
                   curandDiscreteDistribution_t poisson_2,
                   curandDiscreteDistribution_t poisson_3,
                   unsigned int *queue_lengths, size_t pitch)
{
    int id = threadIdx.x + blockIdx.x * 64;
    unsigned int new_customers;
    unsigned int queue_length = 0;
    /* Copy state to local memory for efficiency */
    // CHECK: hiprandState_t localState = state[id];
    curandState localState = state[id];
    /* Simulate queue in time */
    /* first 3 hours */
    for(int min = 1; min <= 60 * 3; min++) {
        /* draw number of new customers depending on API */
        new_customers =
                    // CHECK: hiprand_discrete(&localState, poisson_2);
                    curand_discrete(&localState, poisson_2);
        /* Update queue */
        update_queue(id, min, new_customers, queue_length,
                                    queue_lengths, pitch);
    }
    /* second 3 hours */
    for(int min = 60 * 3 + 1; min <= 60 * 6; min++) {
        /* draw number of new customers depending on API */
        new_customers =
                    // CHECK: hiprand_discrete(&localState, poisson_1);
                    curand_discrete(&localState, poisson_1);
        /* Update queue */
        update_queue(id, min, new_customers, queue_length,
                                    queue_lengths, pitch);       
    }
    /* after 6 hours */
    for(int min = 60 * 6 + 1; min <= 60 * HOURS; min++) {
        /* draw number of new customers depending on API */
        new_customers =
                    // CHECK: hiprand_discrete(&localState, poisson_3);
                    curand_discrete(&localState, poisson_3);
        /* Update queue */
        update_queue(id, min, new_customers, queue_length,
                                    queue_lengths, pitch);       
    }
    /* Copy state back to global memory */
    state[id] = localState;
}

/* Set time intervals between reports */
void report_settings()
{
    do{
        printf("Set time intervals between queue reports");
        printf("(in minutes > 0)\n");
        if (scanf("%d", &report_break) == 0) continue;
    }while(report_break <= 0);
}


/* Set number of cashiers each hour */
void add_cachiers(int *cashiers_load)
{
    int i, min, max, begin, end;
    printf("Cashier serves 2 customers per minute...\n");
    for (i = 0; i < HOURS; i++){
        cashiers_load_h[i] = 0;
    }
    while (true){
        printf("Adding cashier...\n");
        min = OPENING_HOUR;
        max = CLOSING_HOUR-1;
        do{
            printf("Set hour that cahier comes (%d-%d)",
                                                min, max);
            printf(" [type 0 to finish adding cashiers]\n");
            if (scanf("%d", &begin) == 0) continue;
        }while (begin > max || (begin < min && begin != 0));
        if (begin == 0) break;
        min = begin+1;
        max = CLOSING_HOUR;
        do{
            printf("Set hour that cahier leaves (%d-%d)",
                                                min, max);
            printf(" [type 0 to finish adding cashiers]\n");
            if (scanf("%d", &end) == 0) continue;
        }while (end > max || (end < min && end != 0));
        if (end == 0) break;
        for (i = begin - OPENING_HOUR;
             i < end   - OPENING_HOUR; i++){
            cashiers_load_h[i]++;
        }
    }
    for (i = OPENING_HOUR; i < CLOSING_HOUR; i++){
        printf("\n%2d:00 - %2d:00     %d cashier",
                i, i+1, cashiers_load_h[i-OPENING_HOUR]);
        if (cashiers_load[i-OPENING_HOUR] != 1) printf("s");
    }
    printf("\n");
}

/* Set API type */
API_TYPE set_API_type()
{
    printf("Choose API type:\n");
    int choose;
    do{
        printf("type 1 for HOST API\n");
        printf("type 2 for SIMPLE DEVICE API\n");
        printf("type 3 for ROBUST DEVICE API\n");
        if (scanf("%d", &choose) == 0) continue;
    }while( choose < 1 || choose > 3);
    switch(choose){
        case 1: return HOST_API;
        case 2: return SIMPLE_DEVICE_API;
        case 3: return ROBUST_DEVICE_API;
        default:
            fprintf(stderr, "wrong API\n");
            return HOST_API;
    }
}

void settings()
{
    add_cachiers(cashiers_load);
    // CHECK: hipMemcpyToSymbol("cashiers_load", cashiers_load_h,
    // CHECK: HOURS * sizeof(int), 0, hipMemcpyHostToDevice);
    cudaMemcpyToSymbol("cashiers_load", cashiers_load_h,
            HOURS * sizeof(int), 0, cudaMemcpyHostToDevice);
    report_settings();
    api = set_API_type();
}

void print_statistics(unsigned int *hostResults, size_t pitch)
{
    int min, i, hour, minute;
    unsigned int sum;
    for(min = report_break; min <= 60 * HOURS;
                            min += report_break) {
        sum = 0;
        for(i = 0; i < 64 * 64; i++) {
            sum += access_2D(unsigned int, hostResults,
                                        min-1, i, pitch);
        }
        hour = OPENING_HOUR + min/60;
        minute = min%60;
        printf("%2d:%02d   # of waiting customers = %10.4g |",
                    hour, minute, (float)sum/(64.0 * 64.0));
        printf("  # of cashiers = %d  |  ",
                    cashiers_load_h[(min-1)/60]);
        printf("# of new customers/min ~= ");
        switch (api){
            case HOST_API:
                printf("%2.2f\n", 4.0);
                break;
            case SIMPLE_DEVICE_API:
                printf("%2.2f\n",
                            4*(sin((float)min/100.0)+1));
                break;
            case ROBUST_DEVICE_API:
                if (min <= 3 * 60){
                    printf("%2.2f\n", 2.0);
                }else{
                    if (min <= 6 * 60){
                        printf("%2.2f\n", 1.0);
                    }else{
                        printf("%2.2f\n", 3.0);
                    }
                }
                break;
            default:
                fprintf(stderr, "Wrong API\n");
        }
    }
}


int main(int argc, char *argv[])
{
    int n;
    size_t pitch;
    // CHECK: hiprandState_t *devStates;
    curandState *devStates;
    unsigned int *devResults, *hostResults;
    unsigned int *poisson_numbers_d;
    // CHECK: hiprandDiscreteDistribution_t poisson_1, poisson_2;
    // CHECK: hiprandDiscreteDistribution_t poisson_3;
    // CHECK: hiprandGenerator_t gen;
    curandDiscreteDistribution_t poisson_1, poisson_2;
    curandDiscreteDistribution_t poisson_3;
    curandGenerator_t gen;

    /* Setting cashiers, report and API */
    settings();

    /* Allocate space for results on device */
    // CHECK: CUDA_CALL(hipMallocPitch((void **)&devResults, &pitch,
    CUDA_CALL(cudaMallocPitch((void **)&devResults, &pitch,
                64 * 64 * sizeof(unsigned int), 60 * HOURS));

    /* Allocate space for results on host */
    hostResults = (unsigned int *)calloc(pitch * 60 * HOURS,
                sizeof(unsigned int));

    /* Allocate space for prng states on device */
    // CHECK: CUDA_CALL(hipMalloc((void **)&devStates, 64 * 64 *
    // CHECK: sizeof(hiprandState_t)));
    CUDA_CALL(cudaMalloc((void **)&devStates, 64 * 64 * 
              sizeof(curandState)));

    /* Setup prng states */
    if (api != HOST_API){
        // CHECK: hipLaunchKernelGGL(setup_kernel, dim3(64), dim3(64), 0, 0, devStates);
        setup_kernel<<<64, 64>>>(devStates);
    }
    /* Simulate queue  */
    switch (api){
        case HOST_API:
            /* Create pseudo-random number generator */
            // CHECK: CURAND_CALL(hiprandCreateGenerator(&gen,
                                // CHECK: HIPRAND_RNG_PSEUDO_DEFAULT));
            CURAND_CALL(curandCreateGenerator(&gen,
                                CURAND_RNG_PSEUDO_DEFAULT));
            /* Set seed */
            // CHECK: CURAND_CALL(hiprandSetPseudoRandomGeneratorSeed(
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(
                                            gen, 1234ULL));
            /* compute n */
            n = 64 * 64 * HOURS * 60;
            /* Allocate n unsigned ints on device */
            // CHECK: CUDA_CALL(hipMalloc((void **)&poisson_numbers_d,
            CUDA_CALL(cudaMalloc((void **)&poisson_numbers_d,
                                n * sizeof(unsigned int)));
            /* Generate n unsigned ints on device */
            // CHECK: CURAND_CALL(hiprandGeneratePoisson(gen,
            CURAND_CALL(curandGeneratePoisson(gen,
                                poisson_numbers_d, n, 4.0));
            // CHECK: hipLaunchKernelGGL(host_API_kernel, dim3(64), dim3(64), 0, 0, poisson_numbers_d,
            host_API_kernel<<<64, 64>>>(poisson_numbers_d,
                                        devResults, pitch);
            /* Cleanup */
            // CHECK: CURAND_CALL(hiprandDestroyGenerator(gen));
            CURAND_CALL(curandDestroyGenerator(gen));
            break;
        case SIMPLE_DEVICE_API:
            // CHECK: hipLaunchKernelGGL(simple_device_API_kernel, dim3(64), dim3(64), 0, 0, devStates,
            simple_device_API_kernel<<<64, 64>>>(devStates,
                                        devResults, pitch);
            break;
        case ROBUST_DEVICE_API:
            /* Create histograms for Poisson(1) */
            // CHECK: CURAND_CALL(hiprandCreatePoissonDistribution(1.0,
            CURAND_CALL(curandCreatePoissonDistribution(1.0,
                                                &poisson_1));
            /* Create histograms for Poisson(2) */
            // CHECK: CURAND_CALL(hiprandCreatePoissonDistribution(2.0,
            CURAND_CALL(curandCreatePoissonDistribution(2.0,
                                                &poisson_2));
            /* Create histograms for Poisson(3) */
            // CHECK: CURAND_CALL(hiprandCreatePoissonDistribution(3.0,
            CURAND_CALL(curandCreatePoissonDistribution(3.0,
                                                &poisson_3));
            // CHECK: hipLaunchKernelGGL(robust_device_API_kernel, dim3(64), dim3(64), 0, 0, devStates,
            robust_device_API_kernel<<<64, 64>>>(devStates,
                            poisson_1, poisson_2, poisson_3,
                            devResults, pitch);
            /* Cleanup */
            // CHECK: CURAND_CALL(hiprandDestroyDistribution(poisson_1));
            // CHECK: CURAND_CALL(hiprandDestroyDistribution(poisson_2));
            // CHECK: CURAND_CALL(hiprandDestroyDistribution(poisson_3));
            CURAND_CALL(curandDestroyDistribution(poisson_1));
            CURAND_CALL(curandDestroyDistribution(poisson_2));
            CURAND_CALL(curandDestroyDistribution(poisson_3));
            break;
        default:
            fprintf(stderr, "Wrong API\n");
    }
    /* Copy device memory to host */
    // CHECK: CUDA_CALL(hipMemcpy2D(hostResults, pitch, devResults,
    // CHECK: 60 * HOURS, hipMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy2D(hostResults, pitch, devResults,
                pitch, 64 * 64 * sizeof(unsigned int),
                60 * HOURS, cudaMemcpyDeviceToHost));
    /* Show result */
    print_statistics(hostResults, pitch);
    /* Cleanup */
    // CHECK: CUDA_CALL(hipFree(devStates));
    // CHECK: CUDA_CALL(hipFree(devResults));
    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(devResults));
    free(hostResults);
    return EXIT_SUCCESS;
}
