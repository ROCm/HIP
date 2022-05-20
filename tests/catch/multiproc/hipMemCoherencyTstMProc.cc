/*
   Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

/* Test Case Description:
   Scenario 3: The test validates if fine grain
   behavior is observed or not with memory allocated using malloc()
   Scenario 4: The test validates if coarse grain memory
   behavior is observed or not with memory allocated using malloc()
   Scenario 5: The test validates if fine memory
   behavior is observed or not with memory allocated using mmap()
   Scenario 6: The test validates if coarse grain memory
   behavior is observed or not with memory allocated using mmap()
   Scenario:7 Test Case Description: The following test checks if the memory is
   accessible when HIP_HOST_COHERENT is set to 0
   Scenario:8 Test Case Description: The following test checks if the memory
   exhibits fine grain behavior when HIP_HOST_COHERENT is set to 1
   */

#include <hip_test_common.hh>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <chrono>

__global__  void CoherentTst(int *ptr, int PeakClk) {
  // Incrementing the value by 1
  int64_t GpuFrq = (PeakClk * 1000);
  int64_t StrtTck = clock64();
  atomicAdd(ptr, 1);
  // The following while loop checks the value in ptr for around 3-4 seconds
  while ((clock64() - StrtTck) <= (3 * GpuFrq)) {
    if (*ptr == 3) {
      atomicAdd(ptr, 1);
      return;
    }
  }
}

__global__  void SquareKrnl(int *ptr) {
  // ptr value squared here
  *ptr = (*ptr) * (*ptr);
}



// The variable below will work as signal to decide pass/fail
static bool YES_COHERENT = false;

// The function tests the coherency of allocated memory
static void TstCoherency(int *Ptr, bool HmmMem) {
  int *Dptr = nullptr, peak_clk;
  hipStream_t strm;
  HIP_CHECK(hipStreamCreate(&strm));
  // storing value 1 in the memory created above
  *Ptr = 1;
  // Getting gpu frequency
  HIP_CHECK(hipDeviceGetAttribute(&peak_clk, hipDeviceAttributeClockRate, 0));
  if (!HmmMem) {
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void **>(&Dptr), Ptr,
                                      0));
    CoherentTst<<<1, 1, 0, strm>>>(Dptr, peak_clk);
  } else {
    CoherentTst<<<1, 1, 0, strm>>>(Ptr, peak_clk);
  }
  // looping until the value is 2 for 3 seconds
  std::chrono::steady_clock::time_point start =
               std::chrono::steady_clock::now();
  while (std::chrono::duration_cast<std::chrono::seconds>(
         std::chrono::steady_clock::now() - start).count() < 3) {
    if (*Ptr == 2) {
      *Ptr += 1;
      break;
    }
  }
  HIP_CHECK(hipStreamSynchronize(strm));
  HIP_CHECK(hipStreamDestroy(strm));
  if (*Ptr == 4) {
    YES_COHERENT = true;
  }
}

/* Test case description: The following test validates if fine grain
   behavior is observed or not with memory allocated using malloc()*/
// The following test is failing on Nvidia platform hence disabled it for now
#if HT_AMD
TEST_CASE("Unit_malloc_CoherentTst") {
  if ((setenv("HSA_XNACK", "1", 1)) != 0) {
    WARN("Unable to turn on HSA_XNACK, hence terminating the Test case!");
    REQUIRE(false);
  }
  // The following code block is used to check for gfx906/8 so as to skip if
  // any of the gpus available
  int fd1[2];  // Used to store two ends of first pipe
  pid_t p;
  if (pipe(fd1) == -1) {
    fprintf(stderr, "Pipe Failed");
    REQUIRE(false);
  }

  /* GpuId[0] for gfx90a exists--> 1 for yes and 0 for no*/
  int GpuId[1] = {0};
  p = fork();

  if (p < 0) {
    fprintf(stderr, "fork Failed");
    REQUIRE(false);
  } else if (p > 0) {  // parent process
    close(fd1[1]);  // Close writing end of first pipe
    // Wait for child to send a string
    wait(NULL);
    //  Read string from child and close reading end.
    read(fd1[0], GpuId, 2 * sizeof(int));
    close(fd1[0]);
    if (GpuId[0] == 0) {
      WARN("This test is applicable for MI200."
              "Skipping the test!!");
      exit(0);
    }
  } else {    // child process
    close(fd1[0]);  // Close read end of first pipe
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    char *p = NULL;
    p = strstr(prop.gcnArchName, "gfx90a");
    if (p) {
      WARN("gfx90a gpu found on this system!!");
      GpuId[0] = 1;
    }
    // Write concatenated string and close writing end
    write(fd1[1], GpuId, 2 * sizeof(int));
    close(fd1[1]);
    exit(0);
  }

  // Test Case execution begins from here
  int stat = 0;
  if (fork() == 0) {
    int managed = 0;
    HIPCHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                    0));
    if (managed == 1) {
      int *Ptr = nullptr, SIZE = sizeof(int);
      bool HmmMem = true;
      YES_COHERENT = false;
      // Allocating hipMallocManaged() memory
      Ptr = reinterpret_cast<int*>(malloc(SIZE));
      TstCoherency(Ptr, HmmMem);
      free(Ptr);
      if (YES_COHERENT) {
        // exit() with code 10 which indicates pass
        exit(10);
      } else {
        // exit() with code 9 which indicates fail
        exit(9);
      }
    } else {
      SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
             "attribute. Hence skipping the testing with Pass result.\n");
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif


/* Test case description: The following test validates if coarse grain memory
   behavior is observed or not with memory allocated using malloc()*/
// The following test is failing on Nvidia platform hence disabling it for now
#if HT_AMD
TEST_CASE("Unit_malloc_CoherentTstWthAdvise") {
  if ((setenv("HSA_XNACK", "1", 1)) != 0) {
      WARN("Unable to turn on HSA_XNACK, hence terminating the Test case!");
      REQUIRE(false);
  }
  // The following code block is used to check for gfx906/8 so as to skip if
  // any of the gpus available
  int fd1[2];  // Used to store two ends of first pipe
  pid_t p;
  if (pipe(fd1) == -1) {
    fprintf(stderr, "Pipe Failed");
    REQUIRE(false);
  }

  /* GpuId[0] for gfx90a exists--> 1 for yes and 0 for no */
  int GpuId[1] = {0};
  p = fork();

  if (p < 0) {
    fprintf(stderr, "fork Failed");
    REQUIRE(false);
  } else if (p > 0) {  // parent process
    close(fd1[1]);  // Close writing end of first pipe
    // Wait for child to send a string
    wait(NULL);
    //  Read string from child and close reading end.
    read(fd1[0], GpuId, 2 * sizeof(int));
    close(fd1[0]);
    if (GpuId[0] == 0) {
      WARN("This test is applicable for MI200."
              "Skipping the test!!");
      exit(0);
    }
  } else {    // child process
    close(fd1[0]);  // Close read end of first pipe
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    char *p = NULL;
    p = strstr(prop.gcnArchName, "gfx90a");
    if (p) {
      WARN("gfx90a gpu found on this system!!");
      GpuId[0] = 1;
    }
    // Write concatenated string and close writing end
    write(fd1[1], GpuId, 2 * sizeof(int));
    close(fd1[1]);
    exit(0);
  }
  int stat = 0;
  if (fork() == 0) {
    int managed = 0;
    HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                    0));
    if (managed == 1) {
      int *Ptr = nullptr, SIZE = sizeof(int);
      YES_COHERENT = false;
      // Allocating hipMallocManaged() memory
      Ptr = reinterpret_cast<int*>(malloc(SIZE));
      *Ptr = 4;
      hipStream_t strm;
      HIP_CHECK(hipStreamCreate(&strm));
      SquareKrnl<<<1, 1, 0, strm>>>(Ptr);
      HIP_CHECK(hipStreamSynchronize(strm));
      HIP_CHECK(hipStreamDestroy(strm));
      if (*Ptr == 16) {
        // exit() with code 10 which indicates pass
        free(Ptr);
        exit(10);
      } else {
        // exit() with code 9 which indicates fail
        free(Ptr);
        exit(9);
      }
    } else {
      SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
              "attribute. Hence skipping the testing with Pass result.\n");
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif

/* Test case description: The following test validates if fine memory
   behavior is observed or not with memory allocated using mmap()*/
// The following test is failing on Nvidia platform hence disabling it for now
#if HT_AMD
TEST_CASE("Unit_mmap_CoherentTst") {
  if ((setenv("HSA_XNACK", "1", 1)) != 0) {
    WARN("Unable to turn on HSA_XNACK, hence terminating the Test case!");
    REQUIRE(false);
  }
  // The following code block is used to check for gfx906/8 so as to skip if
  // any of the gpus available
  int fd1[2];  // Used to store two ends of first pipe
  pid_t p;
  if (pipe(fd1) == -1) {
    fprintf(stderr, "Pipe Failed");
    REQUIRE(false);
  }

  /* GpuId[0] for gfx90a exists--> 1 for yes and 0 for no */
  int GpuId[1] = {0};
  p = fork();

  if (p < 0) {
    fprintf(stderr, "fork Failed");
    REQUIRE(false);
  } else if (p > 0) {  // parent process
    close(fd1[1]);  // Close writing end of first pipe
    // Wait for child to send a string
    wait(NULL);
    //  Read string from child and close reading end.
    read(fd1[0], GpuId, 2 * sizeof(int));
    close(fd1[0]);
    if (GpuId[0] == 0) {
      WARN("This test is not applicable for MI200."
              "Skipping the test!!");
      exit(0);
    }
  } else {    // child process
    close(fd1[0]);  // Close read end of first pipe
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    char *p = NULL;
    p = strstr(prop.gcnArchName, "gfx90a");
    if (p) {
      WARN("gfx90a gpu found on this system!!");
      GpuId[0] = 1;
    }
    // Write concatenated string and close writing end
    write(fd1[1], GpuId, 2 * sizeof(int));
    close(fd1[1]);
    exit(0);
  }
  int stat = 0;
  if (fork() == 0) {
    int managed = 0;
    HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                    0));
    if (managed == 1) {
      bool HmmMem = true;
      int *Ptr = reinterpret_cast<int*>(mmap(NULL, sizeof(int),
                                        PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANONYMOUS, 0, 0));
      if (Ptr == MAP_FAILED) {
        WARN("Mapping Failed\n");
        REQUIRE(false);
      }
      // Initializing the value with 1
      *Ptr = 1;
      TstCoherency(Ptr, HmmMem);
      int err = munmap(Ptr, sizeof(int));
      if (err != 0) {
        WARN("munmap failed\n");
      }
      if (YES_COHERENT) {
        exit(10);
      } else {
        exit(9);
      }
    } else {
      SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
             "attribute. Hence skipping the testing with Pass result.\n");
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif

/* Test case description: The following test validates if coarse grain memory
   behavior is observed or not with memory allocated using mmap()*/
// The following test is failing on Nvidia platform hence disabling it for now
#if HT_AMD
TEST_CASE("Unit_mmap_CoherentTstWthAdvise") {
  if ((setenv("HSA_XNACK", "1", 1)) != 0) {
    WARN("Unable to turn on HSA_XNACK, hence terminating the Test case!");
    REQUIRE(false);
  }
  // The following code block is used to check for gfx906/8 so as to skip if
  // any of the gpus available
  int fd1[2];  // Used to store two ends of first pipe
  pid_t p;
  if (pipe(fd1) == -1) {
    fprintf(stderr, "Pipe Failed");
    REQUIRE(false);
  }

  /* GpuId[0] for gfx90a exists--> 1 for yes and 0 for no */
  int GpuId[1] = {0};
  p = fork();

  if (p < 0) {
    fprintf(stderr, "fork Failed");
    REQUIRE(false);
  } else if (p > 0) {  // parent process
    close(fd1[1]);  // Close writing end of first pipe
    // Wait for child to send a string
    wait(NULL);
    //  Read string from child and close reading end.
    read(fd1[0], GpuId, 2 * sizeof(int));
    close(fd1[0]);
    if (GpuId[0] == 0) {
      WARN("This test is applicable for MI200."
              "Skipping the test!!");
      exit(0);
    }
  } else {    // child process
    close(fd1[0]);  // Close read end of first pipe
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    char *p = NULL;
    p = strstr(prop.gcnArchName, "gfx90a");
    if (p) {
      WARN("gfx90a gpu found on this system!!");
      GpuId[0] = 1;
    }
    // Write concatenated string and close writing end
    write(fd1[1], GpuId, 2 * sizeof(int));
    close(fd1[1]);
    exit(0);
  }
  int stat = 0;
  if (fork() == 0) {
    int managed = 0;
    HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                    0));
    if (managed == 1) {
      int SIZE = sizeof(int);
      int *Ptr = reinterpret_cast<int*>(mmap(NULL, SIZE,
                                        PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANONYMOUS, 0, 0));
      if (Ptr == MAP_FAILED) {
        WARN("Mapping Failed\n");
        REQUIRE(false);
      }
      HIP_CHECK(hipMemAdvise(Ptr, SIZE, hipMemAdviseSetCoarseGrain, 0));
      // Initializing the value with 9
      *Ptr = 9;
      hipStream_t strm;
      HIP_CHECK(hipStreamCreate(&strm));
      SquareKrnl<<<1, 1, 0, strm>>>(Ptr);
      HIP_CHECK(hipStreamSynchronize(strm));
      bool IfTstPassed = false;
      if (*Ptr == 81) {
        IfTstPassed = true;
      }
      int err = munmap(Ptr, SIZE);
      if (err != 0) {
        WARN("munmap failed\n");
      }
      if (IfTstPassed) {
        exit(10);
      } else {
        exit(9);
      }
    } else {
      SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
             "attribute. Hence skipping the testing with Pass result.\n");
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif

/* Test Case Description: The following test checks if the memory is
   accessible when HIP_HOST_COHERENT is set to 0*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv0Flg1") {
  if ((setenv("HIP_HOST_COHERENT", "0", 1)) != 0) {
      WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
      REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {
    int *Ptr = nullptr, *PtrD = nullptr, SIZE = sizeof(int);
    YES_COHERENT = false;
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocPortable));
    *Ptr = 4;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&PtrD), Ptr, 0));
    SquareKrnl<<<1, 1, 0, strm>>>(PtrD);
    HIP_CHECK(hipStreamSynchronize(strm));
    HIP_CHECK(hipStreamDestroy(strm));
    if (*Ptr == 16) {
      // exit() with code 10 which indicates pass
      HIP_CHECK(hipHostFree(Ptr));
      exit(10);
    } else {
      // exit() with code 9 which indicates fail
      HIP_CHECK(hipHostFree(Ptr));
      exit(9);
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif

/* Test Case Description: The following test checks if the memory is
   accessible when HIP_HOST_COHERENT is set to 0*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv0Flg2") {
  if ((setenv("HIP_HOST_COHERENT", "0", 1)) != 0) {
      WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
      REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {
    int *Ptr = nullptr, *PtrD = nullptr, SIZE = sizeof(int);
    YES_COHERENT = false;
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocWriteCombined));
    *Ptr = 4;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&PtrD), Ptr, 0));
    SquareKrnl<<<1, 1, 0, strm>>>(PtrD);
    HIP_CHECK(hipStreamSynchronize(strm));
    HIP_CHECK(hipStreamDestroy(strm));
    if (*Ptr == 16) {
      // exit() with code 10 which indicates pass
      HIP_CHECK(hipHostFree(Ptr));
      exit(10);
    } else {
      // exit() with code 9 which indicates fail
      HIP_CHECK(hipHostFree(Ptr));
      exit(9);
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif

/* Test Case Description: The following test checks if the memory is
   accessible when HIP_HOST_COHERENT is set to 0*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv0Flg3") {
  if ((setenv("HIP_HOST_COHERENT", "0", 1)) != 0) {
      WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
      REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {
    int *Ptr = nullptr, *PtrD = nullptr, SIZE = sizeof(int);
    YES_COHERENT = false;
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocNumaUser));
    *Ptr = 4;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&PtrD), Ptr, 0));
    SquareKrnl<<<1, 1, 0, strm>>>(PtrD);
    HIP_CHECK(hipStreamSynchronize(strm));
    HIP_CHECK(hipStreamDestroy(strm));
    if (*Ptr == 16) {
      // exit() with code 10 which indicates pass
      HIP_CHECK(hipHostFree(Ptr));
      exit(10);
    } else {
      // exit() with code 9 which indicates fail
      HIP_CHECK(hipHostFree(Ptr));
      exit(9);
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif

/* Test Case Description: The following test checks if the memory is
   accessible when HIP_HOST_COHERENT is set to 0*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv0Flg4") {
  if ((setenv("HIP_HOST_COHERENT", "0", 1)) != 0) {
      WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
      REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {
    int *Ptr = nullptr, *PtrD = nullptr, SIZE = sizeof(int);
    YES_COHERENT = false;
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocNonCoherent));
    *Ptr = 4;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&PtrD), Ptr, 0));
    SquareKrnl<<<1, 1, 0, strm>>>(PtrD);
    HIP_CHECK(hipStreamSynchronize(strm));
    HIP_CHECK(hipStreamDestroy(strm));
    if (*Ptr == 16) {
      // exit() with code 10 which indicates pass
      HIP_CHECK(hipHostFree(Ptr));
      exit(10);
    } else {
      // exit() with code 9 which indicates fail
      HIP_CHECK(hipHostFree(Ptr));
      exit(9);
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif


/* Test Case Description: The following test checks if the memory exhibits
   fine grain behavior when HIP_HOST_COHERENT is set to 1*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv1") {
  if ((setenv("HIP_HOST_COHERENT", "1", 1)) != 0) {
      WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
      REQUIRE(false);
  }
  int stat = 0, Pageable = 0;

  HIP_CHECK(hipDeviceGetAttribute(&Pageable,
                                 hipDeviceAttributePageableMemoryAccess, 0));
  INFO("hipDeviceAttributePageableMemoryAccess: " << Pageable);

  if (Pageable) {
    if (fork() == 0) {  // child process
      int *Ptr = nullptr, SIZE = sizeof(int);
      bool HmmMem = false;
      YES_COHERENT = false;
      // Allocating hipHostMalloc() memory
      HIP_CHECK(hipHostMalloc(&Ptr, SIZE));
      *Ptr = 4;
      TstCoherency(Ptr, HmmMem);
      if (YES_COHERENT) {
        // exit() with code 10 which indicates pass
        HIP_CHECK(hipHostFree(Ptr));
        exit(10);
      } else {
        // exit() with code 9 which indicates fail
        HIP_CHECK(hipHostFree(Ptr));
        exit(9);
      }
    } else {  // parent process
      wait(&stat);
      int Result = WEXITSTATUS(stat);
      if (Result != 10) {
        REQUIRE(false);
      }
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributePageableMemoryAccess "
               "attribute. Hence skipping the test with Pass result.\n"); 
  }
}
#endif


/* Test Case Description: The following test checks if the memory exhibits
   fine grain behavior when HIP_HOST_COHERENT is set to 1*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv1Flg1") {
  if ((setenv("HIP_HOST_COHERENT", "1", 1)) != 0) {
      WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
      REQUIRE(false);
  }
  int stat = 0, Pageable = 0;

  HIP_CHECK(hipDeviceGetAttribute(&Pageable,
                                 hipDeviceAttributePageableMemoryAccess, 0));
  INFO("hipDeviceAttributePageableMemoryAccess: " << Pageable);

  if (Pageable) {
    if (fork() == 0) {  // child process
      int *Ptr = nullptr, SIZE = sizeof(int);
      bool HmmMem = false;
      YES_COHERENT = false;
      // Allocating hipHostMalloc() memory
      HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocPortable));
      *Ptr = 1;
      TstCoherency(Ptr, HmmMem);
      if (YES_COHERENT) {
        // exit() with code 10 which indicates pass
        HIP_CHECK(hipHostFree(Ptr));
        exit(10);
      } else {
        // exit() with code 9 which indicates fail
        HIP_CHECK(hipHostFree(Ptr));
        exit(9);
      }
    } else {  // parent process
      wait(&stat);
      int Result = WEXITSTATUS(stat);
      if (Result != 10) {
        REQUIRE(false);
      }
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributePageableMemoryAccess "
               "attribute. Hence skipping the test with Pass result.\n"); 
  }
}
#endif

/* Test Case Description: The following test checks if the memory exhibits
   fine grain behavior when HIP_HOST_COHERENT is set to 1*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv1Flg2") {
  if ((setenv("HIP_HOST_COHERENT", "1", 1)) != 0) {
      WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
      REQUIRE(false);
  }
  int stat = 0, Pageable = 0;

  HIP_CHECK(hipDeviceGetAttribute(&Pageable,
                                 hipDeviceAttributePageableMemoryAccess, 0));
  INFO("hipDeviceAttributePageableMemoryAccess: " << Pageable);

  if (Pageable) {
    if (fork() == 0) {  // child process
      int *Ptr = nullptr, SIZE = sizeof(int);
      bool HmmMem = false;
      YES_COHERENT = false;
      // Allocating hipHostMalloc() memory
      HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocWriteCombined));
      *Ptr = 4;
      TstCoherency(Ptr, HmmMem);
      if (YES_COHERENT) {
        // exit() with code 10 which indicates pass
        HIP_CHECK(hipHostFree(Ptr));
        exit(10);
      } else {
        // exit() with code 9 which indicates fail
        HIP_CHECK(hipHostFree(Ptr));
        exit(9);
      }
    } else {  // parent process
      wait(&stat);
      int Result = WEXITSTATUS(stat);
      if (Result != 10) {
        REQUIRE(false);
      }
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributePageableMemoryAccess "
               "attribute. Hence skipping the test with Pass result.\n"); 
  }
}
#endif


/* Test Case Description: The following test checks if the memory exhibits
   fine grain behavior when HIP_HOST_COHERENT is set to 1*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv1Flg3") {
  if ((setenv("HIP_HOST_COHERENT", "1", 1)) != 0) {
      WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
      REQUIRE(false);
  }
  int stat = 0, Pageable = 0;

  HIP_CHECK(hipDeviceGetAttribute(&Pageable,
                                 hipDeviceAttributePageableMemoryAccess, 0));
  INFO("hipDeviceAttributePageableMemoryAccess: " << Pageable);

  if (Pageable) {
    if (fork() == 0) {  // child process
      int *Ptr = nullptr, SIZE = sizeof(int);
      bool HmmMem = false;
      YES_COHERENT = false;
      // Allocating hipHostMalloc() memory
      HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocNumaUser));
      *Ptr = 1;
      TstCoherency(Ptr, HmmMem);
      if (YES_COHERENT) {
        // exit() with code 10 which indicates pass
        HIP_CHECK(hipHostFree(Ptr));
        exit(10);
      } else {
        // exit() with code 9 which indicates fail
        HIP_CHECK(hipHostFree(Ptr));
        exit(9);
      }
    } else {  // parent process
      wait(&stat);
      int Result = WEXITSTATUS(stat);
      if (Result != 10) {
        REQUIRE(false);
      }
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributePageableMemoryAccess "
               "attribute. Hence skipping the test with Pass result.\n"); 
  }
}
#endif


