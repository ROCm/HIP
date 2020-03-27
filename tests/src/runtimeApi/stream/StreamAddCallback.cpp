#include <stdio.h>
#include <hip/hip_runtime.h>
#include <unistd.h>
#include "test_common.h"
#include <atomic>

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */

enum class ExecState
{
   EXEC_NOT_STARTED,
   EXEC_STARTED,
   EXEC_CB_STARTED,
   EXEC_CB_FINISHED,
   EXEC_FINISHED
};

struct UserData
{
    size_t size;
    int* ptr;
};

// Global variable to check exection order
std::atomic<ExecState> gData(ExecState::EXEC_NOT_STARTED);


void myCallback(hipStream_t stream, hipError_t status, void* user_data)
{
    if(gData.load() != ExecState::EXEC_STARTED)
        return; // Error hence return early

    gData.store(ExecState::EXEC_CB_STARTED);

    UserData* data = reinterpret_cast<UserData*>(user_data);
    printf("Callback started\n");

    sleep(1);

    printf("Callback ending.\n");
    gData.store(ExecState::EXEC_CB_FINISHED);
}

bool test(int count)
{
    printf("\n============ Test iteration %d =============\n",count);
    // Stream
    hipStream_t stream;
    bool result = true;

    gData.store(ExecState::EXEC_STARTED);

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
    printf("In main thread\n");

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

    printf("Will wait in main thread until callback completes\n");

    //This should synchronize the stream (including the callback)
    HIPCHECK(hipStreamSynchronize(stream));

    if(gData.load() != ExecState::EXEC_CB_FINISHED)
    {
        std::cout<<"Callback is not finished\n";
        return false;
    }
    printf("Callback completed will resume main thread execution\n");

    if(host[size/2] != -1)
    {
         // Print some host data that just got copied
         printf("Pseudo host data printing (should be -1): %d\n", host[size/2]);
         result = false;
    }

    HIPCHECK(hipMemcpy(host, data, sizeof(int)*size, hipMemcpyDeviceToHost));

    if(host[size-1] != 0)
    {
         printf("Pseudo host data printing (should be 0): %d\n", host[size-1]);
         result = false;
    }

    HIPCHECK(hipFree(data));
    HIPCHECK(hipHostFree(host));
    HIPCHECK(hipStreamDestroy(stream));

    gData.store(ExecState::EXEC_FINISHED);
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
