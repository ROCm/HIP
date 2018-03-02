#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
using namespace std;
#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    cout<<"the value of hipError_t at "<<__LINE__<<" is: "<<hipGetErrorString(error)<<endl;\
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
    }\
}
int main()
{
 int deviceId;

// CHECK(hipGetDevice(&deviceId));
// hipDeviceProp_t props;

// CHECK(hipGetDeviceProperties(&props, deviceId));
// printf ("info: running on device #%d %s\n", deviceId, props.name);

hipStream_t stream;

CHECK(hipStreamCreate(&stream));

CHECK(hipStreamQuery(stream));

CHECK(hipDeviceReset());
CHECK(hipStreamQuery(NULL));
CHECK(hipStreamQuery(stream));


}
