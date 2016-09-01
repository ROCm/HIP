
extern "C" __global__ void hello_world(float *a, float *b)
{
	int tx = threadIdx.x;
	b[tx] = a[tx];
}
