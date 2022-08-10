# Emitting Static Library

This sample shows how to generate a static library for a simple HIP application. We will evaluate two types of static libraries: the first type exports host functions in a static library generated with --emit-static-lib and is compatible with host linkers, and second type exports device functions in a static library made with system ar.

Please refer to the hip_programming_guide for limitations.

## Static libraries with host functions

### Source files
The static library source files may contain host functions and kernel `__global__` and `__device__` functions. Here is an example (please refer to the directory host_functions).

hipOptLibrary.cpp:
```
#define HIP_ASSERT(status) assert(status == hipSuccess)
#define LEN 512

__global__ void copy(uint32_t* A, uint32_t* B) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    B[tid] = A[tid];
}

void run_test1() {
    uint32_t *A_h, *B_h, *A_d, *B_d;
    size_t valbytes = LEN * sizeof(uint32_t);

    A_h = (uint32_t*)malloc(valbytes);
    B_h = (uint32_t*)malloc(valbytes);
    for (uint32_t i = 0; i < LEN; i++) {
        A_h[i] = i;
        B_h[i] = 0;
    }

    HIP_ASSERT(hipMalloc((void**)&A_d, valbytes));
    HIP_ASSERT(hipMalloc((void**)&B_d, valbytes));

    HIP_ASSERT(hipMemcpy(A_d, A_h, valbytes, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(copy, dim3(LEN/64), dim3(64), 0, 0, A_d, B_d);
    HIP_ASSERT(hipMemcpy(B_h, B_d, valbytes, hipMemcpyDeviceToHost));

    for (uint32_t i = 0; i < LEN; i++) {
        assert(A_h[i] == B_h[i]);
    }

    HIP_ASSERT(hipFree(A_d));
    HIP_ASSERT(hipFree(B_d));
    free(A_h);
    free(B_h);
    std::cout << "Test Passed!\n";
}
```

The above source file can be compiled into a static library, libHipOptLibrary.a, using the --emit-static-lib flag, like so:
```
hipcc hipOptLibrary.cpp --emit-static-lib -fPIC -o libHipOptLibrary.a
```

### Main source files
The main() program source file may link with the above static library using either hipcc or a host compiler (such as g++). A simple source file that calls the host function inside libHipOptLibrary.a:

hipMain1.cpp:
```
extern void run_test1();

int main(){
  run_test1();
}
```

To link to the static library:

Using hipcc:
```
hipcc hipMain1.cpp -L. -lHipOptLibrary -o test_emit_static_hipcc_linker.out
```
Using g++:
```
ROCM_PATH is the path where ROCM is installed. default path is /opt/rocm.
g++ hipMain1.cpp -L. -lHipOptLibrary -L<ROCM_PATH>/hip/lib -lamdhip64 -o test_emit_static_host_linker.out
```

## Static libraries with device functions

### Source files
The static library source files which contain only `__device__` functions need to be created using ar. Here is an example (please refer to the directory device_functions).

hipDevice.cpp:
```
#include <hip/hip_runtime.h>

__device__ int square_me(int A) {
  return A*A;
}
```

The above source file may be compiled into a static library, libHipDevice.a, by first compiling into a relocatable object, and then placed in an archive using ar:
```
hipcc hipDevice.cpp -c -fgpu-rdc -fPIC -o hipDevice.o
ar rcsD libHipDevice.a hipDevice.o
```

### Main source files
The main() program source file can link with the static library using hipcc. A simple source file that calls the device function inside libHipDevice.a:

hipMain2.cpp:
```
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>

#define HIP_ASSERT(status) assert(status == hipSuccess)
#define LEN 512

extern __device__ int square_me(int);

__global__ void square_and_save(int* A, int* B) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    B[tid] = square_me(A[tid]);
}

void run_test2() {
    int *A_h, *B_h, *A_d, *B_d;
    A_h = new int[LEN];
    B_h = new int[LEN];
    for (unsigned i = 0; i < LEN; i++) {
        A_h[i] = i;
        B_h[i] = 0;
    }
    size_t valbytes = LEN*sizeof(int);

    HIP_ASSERT(hipMalloc((void**)&A_d, valbytes));
    HIP_ASSERT(hipMalloc((void**)&B_d, valbytes));

    HIP_ASSERT(hipMemcpy(A_d, A_h, valbytes, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(square_and_save, dim3(LEN/64), dim3(64),
                       0, 0, A_d, B_d);
    HIP_ASSERT(hipMemcpy(B_h, B_d, valbytes, hipMemcpyDeviceToHost));

    for (unsigned i = 0; i < LEN; i++) {
        assert(A_h[i]*A_h[i] == B_h[i]);
    }

    HIP_ASSERT(hipFree(A_d));
    HIP_ASSERT(hipFree(B_d));
    free(A_h);
    free(B_h);
    std::cout << "Test Passed!\n";
}

int main(){
  // Run test that generates static lib with ar
  run_test2();
}
```

To link to the static library:
```
hipcc libHipDevice.a hipMain2.cpp -fgpu-rdc -o test_device_static_hipcc.out
```

##  How to build and run this sample:
Use the make command to build the static libraries, link with it, and execute it.
- Change directory to either host or device functions folder.
- To build the static library and link the main executable, use `make all`.
- To execute, run the generated executable `./test_*.out`.

Alternatively, use these CMake commands.
```
cd device_functions
mkdir -p build
cd build
cmake ..
make
./test_*.out
```
It is recommended to use Visual Studio's command prompt for this sample due to requirement of MS Librarian tool - LIB.exe on windows platform.
Override CMAKE_C_COMPILER and CMAKE_CXX_COMPILER to hipcc as Visual Studio's compiler would use cl.exe as default compiler.
i.e. cmake.exe -GNinja -DCMAKE_CXX_COMPILER_ID=ROCMClang -DCMAKE_C_COMPILER_ID=ROCMClang -DCMAKE_PREFIX_PATH=%HIP_PATH% -DCMAKE_C_COMPILER=%HIP_PATH%/bin/hipcc.bat -DCMAKE_CXX_COMPILER=%HIP_PATH%/bin/hipcc.bat ..

## For More Infomation, please refer to the HIP FAQ.
