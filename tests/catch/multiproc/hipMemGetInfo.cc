/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <stdlib.h>
#include <stdio.h>
#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>

#define ReadEnd  0
#define WriteEnd 1
#define MAX_SIZE 32
#define FREE_MEM_TO_HIDE 4294967296
#define SIZE_TO_ALLOCATE 2147483648
/*
* In main process allocate 2 GB of device memory.
* Fork() a child process and verify that 2 GB has been
* allocated in parent process.
*/
TEST_CASE("Unit_hipMemGetInfo_Functional_Scenario1") {
  constexpr size_t size = 2147483648;  // 2GB
  int fd[2], fd1[2], status;
  status = pipe(fd);
  REQUIRE(status == 0);
  status = pipe(fd1);
  REQUIRE(status == 0);
  pid_t child_pid;
  child_pid = fork();   // Create a new child process
  if (child_pid < 0) {
    WARN("Fork failed!!!!");
  } else if (child_pid == 0) {    // child
    close(fd1[WriteEnd]);
    close(fd[ReadEnd]);
    int result;
    size_t free = 0, total = 0;
    // Wait for signal from parent
    int check_child;
    status = read(fd1[ReadEnd], &check_child, sizeof(check_child));
    REQUIRE(status != -1);
    close(fd1[ReadEnd]);
    // Check the total and free memory which is allocated in parent
    HIP_CHECK(hipMemGetInfo(&free, &total));
    if ((total - free) >= size) {
      result = 1;
    } else {
      result = 0;
    }
    // Write the result to parent
    status = write(fd[WriteEnd], &result, sizeof(result));
    REQUIRE(status != -1);
    close(fd[WriteEnd]);
    exit(0);
  } else {    // Parent
    close(fd1[ReadEnd]);
    close(fd[WriteEnd]);
    // Allocate memory
    char* A_d = nullptr;
    HIP_CHECK(hipMalloc(&A_d, size));
    // Signal the child
    int check = 0;
    status = write(fd1[WriteEnd], &check, sizeof(check));
    REQUIRE(status != -1);
    close(fd1[WriteEnd]);
    // Read the result from Child
    int read_result;
    status = read(fd[ReadEnd], &read_result, sizeof(read_result));
    REQUIRE(status != -1);
    close(fd[ReadEnd]);
    REQUIRE(read_result == 1);
    HIP_CHECK(hipFree(A_d));
    // wait for child exit
    wait(NULL);
  }
}
/**
* From main process Fork() a child process. In the child process allocate
* 2 GB of device memory. Signal the parent process. Verify from the parent
* process that 2 GB is allocated in the child process.
*/
TEST_CASE("Unit_hipMemGetInfo_Functional_Scenario2") {
  constexpr size_t size = 2147483648;  // 2GB
  int fd[2], fd2[2], status;
  status = pipe(fd);
  REQUIRE(status == 0);
  status = pipe(fd2);
  REQUIRE(status == 0);
  pid_t child_pid;
  child_pid = fork();   // Create a new child process
  if (child_pid < 0) {
    WARN("Fork failed!!!!");
  } else if (child_pid == 0) {    // Child
    close(fd[ReadEnd]);
    close(fd2[WriteEnd]);
    // Allocate memory
    float* A_d = nullptr;
    HIP_CHECK(hipMalloc(&A_d, size));
    // Signal the parent
    int data = 0;
    status = write(fd[WriteEnd], &data, sizeof(data));
    REQUIRE(status != -1);
    close(fd[WriteEnd]);
    int valid = 0;
    // Wait for Signal from parent before freeing memory and exiting
    status = read(fd2[ReadEnd], &valid, sizeof(valid));
    REQUIRE(status != -1);
    close(fd2[ReadEnd]);
    // Free allocated device memory
    HIP_CHECK(hipFree(A_d));
    exit(0);
  } else {    // Parent
    size_t free = 0, total = 0;
    close(fd[WriteEnd]);
    close(fd2[ReadEnd]);
    // Wait for child signal
    int data = 0;
    status = read(fd[ReadEnd], &data, sizeof(data));
    REQUIRE(status != -1);
    close(fd[ReadEnd]);
    // Verify the memory
    HIP_CHECK(hipMemGetInfo(&free , &total));
    REQUIRE((total - free) >= size);
    // Signal child that validation is over and child can free memory
    int valid = 0;
    status = write(fd2[WriteEnd], &valid, sizeof(valid));
    REQUIRE(status != -1);
    close(fd2[WriteEnd]);
    // wait for child exit
    wait(NULL);
  }
}
/*
* From main process Fork() a child process. In the child process
* allocate 2 GB of device memory. Free the memory and exit from
* child process. Verify from the parent process that 2 GB is
* freed in the child process.
*/
TEST_CASE("Unit_hipMemGetInfo_Functional_Scenario3") {
  constexpr size_t size = 2147483648;  // 2GB
  int fd[2], status;
  status = pipe(fd);
  REQUIRE(status == 0);
  pid_t child_pid;
  child_pid = fork();   // Create a new child process
  if (child_pid < 0) {
    WARN("Fork failed!!!!");
  } else if (child_pid == 0) {    // Child
    close(fd[ReadEnd]);
    // Allocate the memory
    void* A_d = nullptr;
    HIP_CHECK(hipMalloc(&A_d, size));
    // Free the allocated memory
    HIP_CHECK(hipFree(A_d));
    // Signal the parent about memory free
    int check = 0;
    status = write(fd[WriteEnd], &check, sizeof(check));
    REQUIRE(status != -1);
    close(fd[WriteEnd]);
    exit(0);
  } else {    // Parent
    close(fd[WriteEnd]);
    // Wait for the signal from child about memory free
    int check_parent;
    status = read(fd[ReadEnd], &check_parent, sizeof(check_parent));
    REQUIRE(status != -1);
    close(fd[ReadEnd]);
    size_t free = 0, total = 0;
    // Verify the memory
    HIP_CHECK(hipMemGetInfo(&free , &total));
    REQUIRE((total - free) >= 0);
    // wait for child exit
    wait(NULL);
  }
}
/*
* From main process Fork() a child process. In the child process allocate
* 2 GB of device memory. Exit from child process. Verify from the parent
* process that 2 GB is freed in the child process.
*/
TEST_CASE("Unit_hipMemGetInfo_Functional_scenario4") {
  constexpr size_t size = 2147483648;  // 2GB
  pid_t child_pid;
  child_pid = fork();   // Create a new child process
  if (child_pid < 0) {
    WARN("Fork failed!!!!");
  } else if (child_pid == 0) {    // Child
    // Allocate the memory
    void* A_d = nullptr;
    HIP_CHECK(hipMalloc(&A_d, size));
    exit(0);
  } else {    // Parent
    // wait for child exit
    wait(NULL);
    size_t free = 0, total = 0;
    // Verify the memory
    HIP_CHECK(hipMemGetInfo(&free , &total));
    REQUIRE((total-free) >= 0);
  }
}
/*
* Multidevice Scenario: In main process allocate 2 GB of device memory
* in every device. Verify that 2 GB is allocated using hipMemGetInfo.
* Fork() a child process and verify that 2 GB has been allocated from
* parent process in every device.
*/
TEST_CASE("Unit_hipMemGetInfo_Functional_MultiDevice_Scenario5") {
  constexpr size_t size = 2147483648;  // 2GB
  size_t free = 0, total = 0;
  int fd1[2], fd2[2], status;
  status = pipe(fd1);
  REQUIRE(status == 0);
  status = pipe(fd2);
  REQUIRE(status == 0);
  pid_t child_pid;
  child_pid = fork();   // Create a new child process
  if (child_pid < 0) {
    WARN("Fork failed!!!!");
  } else if (child_pid == 0) {    // Child
      close(fd1[WriteEnd]);
      close(fd2[ReadEnd]);
      // Wait for the signal from parent after memory allocatoin
      int check_child;
      status = read(fd1[ReadEnd], &check_child, sizeof(check_child));
      REQUIRE(status != -1);
      close(fd1[ReadEnd]);
      int num_devices, result, count = 0;
      // Get the device count
      HIP_CHECK(hipGetDeviceCount(&num_devices));
      for (int i = 0; i < num_devices; i++) {
        HIP_CHECK(hipSetDevice(i));
        // Check the memory
        HIP_CHECK(hipMemGetInfo(&free , &total));
        if ((total - free) >= size) {
          count+=1;
        }
    }
    if ( count == num_devices ) {
      result = 1;
    } else {
      result = 0;
    }
    // Write the result to Parent
    status = write(fd2[WriteEnd], &result, sizeof(result));
    REQUIRE(status != -1);
    close(fd2[WriteEnd]);
    exit(0);
  } else {    // Parent
    close(fd1[ReadEnd]);
    close(fd2[WriteEnd]);
    int num_devices;
    // Get the device count
    HIP_CHECK(hipGetDeviceCount(&num_devices));
    std::vector<void*>v(num_devices, nullptr);
    for (int i = 0; i < num_devices; i++) {
      HIP_CHECK(hipSetDevice(i));
      // verify the memory
      HIP_CHECK(hipMemGetInfo(&free , &total));
      // Allocate memory
      HIP_CHECK(hipMalloc(&v[i], size));
      // Verify the memory
      HIP_CHECK(hipMemGetInfo(&free , &total));
    }
    // Signal the child about memory allocation
    int check = 0;
    status = write(fd1[WriteEnd], &check, sizeof(check));
    REQUIRE(status != -1);
    close(fd1[WriteEnd]);
    // Read result from child
    int result_parent;
    status = read(fd2[ReadEnd], &result_parent, sizeof(result_parent));
    REQUIRE(status != -1);
    REQUIRE(result_parent == 1);
    close(fd2[ReadEnd]);
    // Free the allocated memory on each device
    for (int i = 0; i < num_devices; i++) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipFree(v[i]));
    }
    // wait for child exit
    wait(NULL);
  }
}
#endif

