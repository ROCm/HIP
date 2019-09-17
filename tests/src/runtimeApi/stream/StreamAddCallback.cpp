#include <stdio.h>
#include <hip/hip_runtime.h>
#include <unistd.h>
#include "test_common.h"

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

struct UserData
{
    size_t size;
    int* ptr;
};

void myCallback(hipStream_t stream, hipError_t status, void* user_data)
{
    UserData* data = reinterpret_cast<UserData*>(user_data);
    printf("Callback called with arg.size = %lu ; arg.ptr = %p\nsleeping for 1 sec...\n", data->size, data->ptr);

    sleep(1);

    printf("Callback ending.\n");
}

bool test(int count)
{
    printf("\n============ Test iteration %d =============\n",count);
    // Stream
    hipStream_t stream;
    bool result = true;

    HIPCHECK(hipStreamCreate(&stream));

    // Array size
    size_t size = 10000;

    // Device array
    int *data = NULL;
    HIPCHECK(hipMalloc((void**)&data, sizeof(int) * size));

    // Initialize device array to -1
    HIPCHECK(hipMemset(data, -1, sizeof(int) * size));

    // Host array
    int *host = NULL;
    HIPCHECK(hipHostMalloc((void**)&host, sizeof(int) * size));

    // Print host ptr address
    printf("Host ptr address = %p\n", host);

    // Initialize user_data for callback
    UserData arg;
    arg.size = size;
    arg.ptr  = host;

    // Synchronize device
    HIPCHECK(hipDeviceSynchronize());

    // Asynchronous copy from device to host
    HIPCHECK(hipMemcpyAsync(host, data, sizeof(int) * size, hipMemcpyDeviceToHost, stream));

    // Asynchronous memset on device
    HIPCHECK(hipMemsetAsync(data, 0, sizeof(int) * size, stream));

    // Add callback - should happen after hipMemsetAsync()
    HIPCHECK(hipStreamAddCallback(stream, myCallback, &arg, 0));

    // This should happen before the callback actually gets called
    // when the size is large enough
    printf("This should happen before the callback (assuming sufficiently large size).\n");

    //This should synchronize the stream (including the callback)
    HIPCHECK(hipStreamSynchronize(stream));

    printf("This should happen after the callback, since we synchronized stream before.\n");

    if(host[size/2] != -1)
         result = false;

    // Print some host data that just got copied
    printf("Pseudo host data printing (should be -1): %d\n", host[size/2]);

    HIPCHECK(hipMemcpy(host, data, sizeof(int)*size, hipMemcpyDeviceToHost));

    if(host[size-1] != 0)
         result = false;

    printf("Pseudo host data printing (should be 0): %d\n", host[size-1]);

    HIPCHECK(hipFree(data));
    HIPCHECK(hipHostFree(host));
    HIPCHECK(hipStreamDestroy(stream));

    return result;
}

int main()
{
    // Test involves multithreading hence running multiple times
    // to make sure consitency in the behavior
    bool status = true;

    for(int i=0; i < 10; i++){
       status = test(i+1);
       if(status == false)
       {
          failed("Test Failed!\n");
          break;
       }
    }

    if(status == true) passed();
    return 0;
}