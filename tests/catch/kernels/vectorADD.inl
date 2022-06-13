namespace HipTest {
template <typename T> __global__ void vectorADD(const T* A_d, const T* B_d, T* C_d, size_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < NELEM; i += stride) {
    C_d[i] = A_d[i] + B_d[i];
  }
}
}