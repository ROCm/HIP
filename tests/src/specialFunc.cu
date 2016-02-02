//Test to ensure hipify runs correctly.
// Hipify may report warnings for some missing/unsupported functions

void __global__
test_kernel(float *A) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float a = __ballot(tid < 16);
    float b = __shfl(tid < 16);
}
