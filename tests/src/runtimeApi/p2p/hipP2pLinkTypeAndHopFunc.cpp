/*
Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.

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

/* HIT_START
 * BUILD_CMD: %t %hc %S/%s -o %T/%t %S/../../test_common.cpp -I %S/../../ -L%rocm-path/rocm_smi/lib -lrocm_smi64 -ldl EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t --tests 0x1
 * TEST: %t --tests 0x2
 * TEST: %t --tests 0x3
 * TEST: %t --tests 0x4
 * TEST: %t --tests 0x5
 * TEST: %t --tests 0x6
 * HIT_END
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#include <dlfcn.h>
#endif
#include "test_common.h"
#include "hipP2pLinkTypeAndHopFunc.h"
/**
 * Fetches Gpu device count
 */
void getDeviceCount(int *pdevCnt) {
#ifdef __linux__
  int fd[2], val = 0;
  pid_t childpid;

  // create pipe descriptors
  pipe(fd);

  // disable visible_devices env from shell
  unsetenv("ROCR_VISIBLE_DEVICES");
  unsetenv("HIP_VISIBLE_DEVICES");

  childpid = fork();

  if (childpid > 0) {  // Parent
    close(fd[1]);
    // parent will wait to read the device cnt
    read(fd[0], &val, sizeof(val));

    // close the read-descriptor
    close(fd[0]);

    // wait for child exit
    wait(NULL);

    *pdevCnt = val;
  } else if (!childpid) {  // Child
    int devCnt = 1;
    // writing only, no need for read-descriptor
    close(fd[0]);

    HIPCHECK(hipGetDeviceCount(&devCnt));
    // send the value on the write-descriptor:
    write(fd[1], &devCnt, sizeof(devCnt));

    // close the write descriptor:
    close(fd[1]);
    exit(0);
  } else {  // failure
    *pdevCnt = 1;
    return;
  }
#else
  HIPCHECK(hipGetDeviceCount(pdevCnt));
#endif
}
/**
 * Validates negative scenarios for hipExtGetLinkTypeAndHopCount
 * Test Scenario: device1 is visible and device2 is masked
 */
#ifdef __linux__
#define MAX_SIZE 30
#define VISIBLE_DEVICE 0
bool testMaskedDevice(int actualNumGPUs) {
  bool testResult = true;
  int device;
  int fd[2];
  pipe(fd);

  pid_t cPid;
  cPid = fork();
  if (cPid == 0) {  // child
    hipError_t err;
    char visibleDeviceString[MAX_SIZE] = {};
    snprintf(visibleDeviceString, MAX_SIZE, "%d", VISIBLE_DEVICE);
    // disable visible_devices env from shell
    unsetenv("ROCR_VISIBLE_DEVICES");
    unsetenv("HIP_VISIBLE_DEVICES");
    setenv("ROCR_VISIBLE_DEVICES", visibleDeviceString, 1);
    setenv("HIP_VISIBLE_DEVICES", visibleDeviceString, 1);
    uint32_t linktype;
    uint32_t hopcount;
    for (int count = 1;
        count < actualNumGPUs; count++) {
      err = hipExtGetLinkTypeAndHopCount(VISIBLE_DEVICE,
            VISIBLE_DEVICE+count, &linktype, &hopcount);
      if (err == hipSuccess) {
        testResult &= false;
      } else {
        printf("testMaskedDevice: Error Code Returned: '%s'(%d)\n",
              hipGetErrorString(err), err);
      }
    }
    close(fd[0]);
    write(fd[1], &testResult, sizeof(testResult));
    close(fd[1]);
    exit(0);

  } else if (cPid > 0) {  // parent
    close(fd[1]);
    read(fd[0], &testResult, sizeof(testResult));
    close(fd[0]);
    wait(NULL);

  } else {
    printf("fork() failed\n");
    testResult = false;
  }
  return testResult;
}
#endif
/**
 * Validates negative scenarios for hipExtGetLinkTypeAndHopCount
 * Test Scenario: Invalid Device Number(s)
 */
bool testhipInvalidDevice(int numDevices) {
  bool TestPassed = true;
  hipError_t ret;
  uint32_t linktype;
  uint32_t hopcount;
  if ((ret = hipExtGetLinkTypeAndHopCount(-1, 0, &linktype, &hopcount))
       == hipSuccess) {
    TestPassed &= false;
  }
  if ((ret = hipExtGetLinkTypeAndHopCount(numDevices, 0, &linktype,
       &hopcount)) == hipSuccess) {
    TestPassed &= false;
  }
  if ((ret = hipExtGetLinkTypeAndHopCount(0, -1, &linktype, &hopcount))
       == hipSuccess) {
    TestPassed &= false;
  }
  if ((ret = hipExtGetLinkTypeAndHopCount(0, numDevices, &linktype,
       &hopcount)) == hipSuccess) {
    TestPassed &= false;
  }
  if ((ret = hipExtGetLinkTypeAndHopCount(-1, numDevices, &linktype,
      &hopcount)) == hipSuccess) {
    TestPassed &= false;
  }
  return TestPassed;
}

/**
 * Validates negative scenarios for hipExtGetLinkTypeAndHopCount
 * Test Scenario: linktype = NULL
 */
bool testhipInvalidLinkType() {
  uint32_t hopcount;
  if (hipSuccess != hipExtGetLinkTypeAndHopCount(0, 1, nullptr, &hopcount)) {
    return true;
  } else {
    printf("Test Failed as linktype = NULL returns hipSuccess \n");
  }
  return false;
}

/**
 * Validates negative scenarios for hipExtGetLinkTypeAndHopCount
 * Test Scenario: hopcount = NULL
 */
bool testhipInvalidHopcount() {
  uint32_t linktype;
  if (hipSuccess != hipExtGetLinkTypeAndHopCount(0, 1, &linktype, nullptr)) {
    return true;
  } else {
    printf("Test Failed as hopcount = NULL returns hipSuccess \n");
  }
  return false;
}

/**
 * Validates negative scenarios for hipExtGetLinkTypeAndHopCount
 * Test Scenario: device1 = device2
 */
bool testhipSameDevice(int numGPUs) {
  hipError_t ret;
  uint32_t linktype = 0;
  uint32_t hopcount = 0;
  for (int gpuId = 0; gpuId < numGPUs; gpuId++) {
    if ((ret = hipExtGetLinkTypeAndHopCount(gpuId, gpuId, &linktype,
        &hopcount)) == hipSuccess) {
      return false;
    }
  }
  return true;
}
/**
 * Validates negative scenarios for hipExtGetLinkTypeAndHopCount
 * Test Scenario: Verify (hopcount, linktype) values for (src= device1, dest = device2)
 * and (src = device2, dest = device1), where device1 and device2 are valid device numbers.
 */
bool testhipLinkTypeHopcountDeviceOrderRev(int numDevices) {
  bool TestPassed = true;
  // Get the unique pair of devices
  for (int x = 0; x < numDevices; x++) {
    for (int y = x+1; y < numDevices; y++) {
      uint32_t linktype1 = 0, linktype2 = 0;
      uint32_t hopcount1 = 0, hopcount2 = 0;
      HIPCHECK(hipExtGetLinkTypeAndHopCount(x, y,
                          &linktype1, &hopcount1));
      HIPCHECK(hipExtGetLinkTypeAndHopCount(y, x,
                          &linktype2, &hopcount2));
      if (hopcount1 != hopcount2) {
        TestPassed = false;
        break;
      }
    }
  }
  return TestPassed;
}

#ifdef __linux__
/**
 * Internal Function
 */
bool validateLinkType(uint32_t linktype_Hip,
                      RSMI_IO_LINK_TYPE linktype_RocmSmi) {
  bool TestPassed = false;

  if ((linktype_Hip == HSA_AMD_LINK_INFO_TYPE_PCIE) &&
     (linktype_RocmSmi == RSMI_IOLINK_TYPE_PCIEXPRESS)) {
    TestPassed = true;
  } else if ((linktype_Hip == HSA_AMD_LINK_INFO_TYPE_XGMI) &&
     (linktype_RocmSmi == RSMI_IOLINK_TYPE_XGMI)) {
    TestPassed = true;
  } else {
    printf("linktype Hip = %u, linktype RocmSmi = %u\n",
            linktype_Hip, linktype_RocmSmi);
    TestPassed = false;
  }
  return TestPassed;
}

/**
 * Validates negative scenarios for hipExtGetLinkTypeAndHopCount
 * Test Scenario: Verify (hopcount, linktype) values for all combination of
 * GPUs with the output of rocm_smi tool.
 */
bool testhipLinkTypeHopcountDevice(int numDevices) {
  bool TestPassed = true;
  // Opening and initializing rocm-smi library
  void *lib_rocm_smi_hdl;
  rsmi_status_t (*fntopo_get_link_type)(uint32_t, uint32_t, uint64_t*,
                      RSMI_IO_LINK_TYPE*);
  rsmi_status_t (*fntopo_init)(uint64_t);
  rsmi_status_t (*fntopo_shut_down)();

  lib_rocm_smi_hdl = dlopen("/opt/rocm/rocm_smi/lib/librocm_smi64.so",
                        RTLD_LAZY);
  if (!lib_rocm_smi_hdl) {
    printf("Error opening /opt/rocm/rocm_smi/lib/librocm_smi64.so\n");
    printf("Skipping this test\n");
    passed();
  }

  void* fnsym = dlsym(lib_rocm_smi_hdl, "rsmi_topo_get_link_type");
  if (!fnsym) {
    printf("Error getting rsmi_topo_get_link_type() function\n");
    printf("Skipping this test\n");
    dlclose(lib_rocm_smi_hdl);
    passed();
  }
  fntopo_get_link_type = reinterpret_cast<rsmi_status_t (*)(uint32_t,
            uint32_t, uint64_t*, RSMI_IO_LINK_TYPE*)>(fnsym);

  fnsym = dlsym(lib_rocm_smi_hdl, "rsmi_init");
  if (!fnsym) {
    printf("Error getting rsmi_init() function\n");
    printf("Skipping this test\n");
    dlclose(lib_rocm_smi_hdl);
    passed();
  }
  fntopo_init = reinterpret_cast<rsmi_status_t (*)(uint64_t)>(fnsym);

  fnsym = dlsym(lib_rocm_smi_hdl, "rsmi_shut_down");
  if (!fnsym) {
    printf("Error getting rsmi_shut_down() function\n");
    printf("Skipping this test\n");
    dlclose(lib_rocm_smi_hdl);
    passed();
  }
  fntopo_shut_down = reinterpret_cast<rsmi_status_t (*)()>(fnsym);

  uint64_t init_flags = 0;
  rsmi_status_t retsmi_init;
  retsmi_init = fntopo_init(init_flags);
  if (RSMI_STATUS_SUCCESS != retsmi_init) {
    printf("Error when initializing rocm_smi\n");
    printf("Skipping this test\n");
    dlclose(lib_rocm_smi_hdl);
    fntopo_shut_down();
    passed();
  }
  // Use rocm-smi API rsmi_topo_get_link_type() to validate
  struct devicePair {
    int device1;
    int device2;
  };
  std::vector<struct devicePair> devicePairList;
  // Get the unique pair of devices
  for (int x = 0; x < numDevices; x++) {
    for (int y = x+1; y < numDevices; y++) {
      devicePairList.push_back({x, y});
    }
  }
  for (auto pos=devicePairList.begin();
       pos != devicePairList.end(); pos++) {
    uint32_t linktype1 = 0;
    uint32_t hopcount1 = 0;
    RSMI_IO_LINK_TYPE linktype2 = RSMI_IOLINK_TYPE_UNDEFINED;
    uint64_t hopcount2 = 0;
    rsmi_status_t retsmi;
    HIPCHECK(hipExtGetLinkTypeAndHopCount((*pos).device1,
                (*pos).device2, &linktype1, &hopcount1));
    retsmi = fntopo_get_link_type((*pos).device1,
                (*pos).device2, &hopcount2, &linktype2);
    if (RSMI_STATUS_SUCCESS != retsmi) {
      printf("Error returned from rsmi_topo_get_link_type() function\n");
      printf("Skipping this test\n");
      fntopo_shut_down();
      dlclose(lib_rocm_smi_hdl);
      passed();
    }
    uint32_t hopcount32 = hopcount2;  // Convert uint64_t to uint32_t
    // Validate hopcount
    if (hopcount1 != hopcount2) {
      printf("device1=%u,device2=%u,hopcount hip=%u,hopcount smi=%u\n",
              (*pos).device1, (*pos).device2,
              hopcount1, hopcount32);
      TestPassed &= false;
    }
    // Validate linktype
    TestPassed &= validateLinkType(linktype1, linktype2);
  }
  fntopo_shut_down();
  dlclose(lib_rocm_smi_hdl);
  return TestPassed;
}
#endif

int main(int argc, char* argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  int numDevices = 0;
  getDeviceCount(&numDevices);
  if (numDevices < 2) {
    printf("No. GPUs found is less than 2. Skipping all Test Case. \n");
    passed();
  }
  bool TestPassed = true;
  if (p_tests == 0x1) {
    TestPassed = testhipInvalidDevice(numDevices);
  } else if (p_tests == 0x2) {
#ifdef __linux__
    TestPassed = testMaskedDevice(numDevices);
#else
    printf("This test is skipped due to non linux environment.\n");
#endif
  } else if (p_tests == 0x3) {
    TestPassed = testhipInvalidLinkType();
  } else if (p_tests == 0x4) {
    TestPassed = testhipInvalidHopcount();
  } else if (p_tests == 0x5) {
    TestPassed = testhipSameDevice(numDevices);
  } else if (p_tests == 0x6) {
    TestPassed = testhipLinkTypeHopcountDeviceOrderRev(numDevices);
  } else if (p_tests == 0x7) {
    /*TODO:This test is currently ommited from directed test due to existing issues
    in rocm-smi. Once rocm-smi issues are resolved, this test will be enabled. */
#ifdef __linux__
    TestPassed = testhipLinkTypeHopcountDevice(numDevices);
#else
    printf("This test is skipped due to non linux environment.\n");
#endif
  } else {
    printf("Invalid Test Case \n");
    exit(1);
  }
  if (TestPassed) {
    passed();
  } else {
    failed("Test Case %x Failed!", p_tests);
  }
}
