/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
#ifdef __linux__
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif


/* Test Description: The following test case tests the working of
   hipMemAdvise() works with mmap() memory
*/

TEST_CASE("Unit_hipMemAdvise_MmapMem") {
  int managed = 0, NUM_ELMS = 212992, PageableMem = 0;
  INFO("The following are the attribute values related to HMM for"
         " device 0:\n");
  HIP_CHECK(hipDeviceGetAttribute(&managed,
              hipDeviceAttributeDirectManagedMemAccessFromHost, 0));
  INFO("hipDeviceAttributeDirectManagedMemAccessFromHost: " << managed);
  HIP_CHECK(hipDeviceGetAttribute(&managed,
                                 hipDeviceAttributeConcurrentManagedAccess, 0));
  INFO("hipDeviceAttributeConcurrentManagedAccess: " << managed);
  HIP_CHECK(hipDeviceGetAttribute(&PageableMem,
                                 hipDeviceAttributePageableMemoryAccess, 0));
  INFO("hipDeviceAttributePageableMemoryAccess: " << PageableMem);
  HIP_CHECK(hipDeviceGetAttribute(&managed,
              hipDeviceAttributePageableMemoryAccessUsesHostPageTables, 0));
  INFO("hipDeviceAttributePageableMemoryAccessUsesHostPageTables:"
        << managed);
  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                  0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);
  if ((managed == 1) && (PageableMem == 1)) {
#ifdef __linux__
    // For now this test is enabled only for linux platforms
    FILE *fptr;
    fptr = fopen("ForTest1.txt", "w");
    for (int m = 0; m < NUM_ELMS; ++m) {
      putw(m, fptr);
    }
    fclose(fptr);
    int fd = open("./ForTest1.txt", O_RDWR, S_IRUSR | S_IWUSR);
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      perror("couldn't get file size.\n");
      close(fd);
      REQUIRE(false);
    }
    void *MmpdFile = nullptr;
    MmpdFile = mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                    0);
    if (MmpdFile == nullptr) {
      INFO("mmap() call failed!\n. Cant proceed with the test.");
      REQUIRE(false);
    }
    HIP_CHECK(hipMemAdvise(MmpdFile, sb.st_size, hipMemAdviseSetReadMostly, 0));
    HIP_CHECK(hipMemAdvise(MmpdFile, sb.st_size,
                          hipMemAdviseSetPreferredLocation, 0));
    HIP_CHECK(hipMemAdvise(MmpdFile, sb.st_size, hipMemAdviseSetAccessedBy, 0));
    munmap(MmpdFile, sb.st_size);
    close(fd);
#endif
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributePageableMemoryAccess "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}
