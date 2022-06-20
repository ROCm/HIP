# HIP Tests - with Catch2

## Intro and Motivation
HIP Tests were using HIT framework (a custom framework tailored for HIP) to add, build and run tests. As time progressed the frame got big and took substantial amount of effort to maintain and extend. It also took substantial amount of time to configure. We took this oppurtunity to rewrite the HIP's testing framework and porting the test infra to Catch2 format.

## How to write tests
Tests in Catch2 are declared via ```TEST_CASE```.

[Please read the Catch2 documentation on how to write test cases](https://github.com/catchorg/Catch2/blob/v2.13.6/docs/tutorial.md#top)

[Catch2 Detailed Reference](https://github.com/catchorg/Catch2/blob/v2.13.6/docs/Readme.md#top)

## Taking care of existing features
- Don’t build on platform: EXCLUDE_(HIP_PLATFORM/HIP_RUNTIME), can be done via CMAKE. Adding source in if(HIP_PLATFORM == amd/nvidia).
- HIPCC_OPTIONS/CLANG Options: Can be done via: set_source_files_properties(src.cc PROPERTIES COMPILE_FLAGS “…”).
- Additional libraries: Can be done via target_link_libraries()
- Multiple runs with different args: This can be done by Catch’s Feature: GENERATE(…)
Running Subtest: ctest –R “...” (Regex to match the subtest name)

## New Features
- Skip test without recompiling tests, by addition of a json file. Default name is ```config.json``` , this can be overridden by using the variable ```HT_CONFIG_FILE=some_config.json```.
- Json file supports regex. Ex: All tests which has the word ‘Memset’ can be skipped using ‘*Memset*’
- Support multiple skip test list which can be set via environment variable, so you can have multiple files containing different skip test lists and can pick and choose among them depending on your platform and os.
- Better CI integration via xunit compatible output

## Testing Context
HIP testing framework gives you a context for each test. This context will have useful information about the environment your test is running.

Some useful functions are:
- `bool isWindows()` : true if os is windows
- `bool isLinux()` : true if os is linux
- `bool isAmd()` : true if platform is AMD
- `bool isNvidia()` : true if platform is NVIDIA

This information can be accessed in any test via using: `TestContext::get().isAmd()`.

## Adding test for a specific platform
There might be some functionality which is not present on some platforms. Those tests can be hidden inside following macros.

- ```HT_AMD``` is 1 when tests are running on AMD platform and 0 on NVIDIA.
- ```HT_NVIDIA``` is 1 when tests are running on NVIDIA platform and 0 on AMD

Usage:

```cpp
#if HT_AMD
TEST_CASE("hipExtAPIs") {
  // ...
}
#endif
```

## Config file schema
Some tests can be skipped using a config file placed in hipTestMain/config folder. Multiple config files can be defined for different configurations.
The naming convention for the file needs to be "config_platform_os_archname.json"
Platform and os are mandatory, "all" for os if the tests needs to be skipped for all OS.
Arch name is optional and takes precedence while loading the json file.

example:
config_amd_windows.json
config_nvidia_windows.json

The schema of the json file is as follows:
```json
{
    "DisabledTests":
    [
        "TestName1",
        "TestName2",
        ...
    ]
}
```

## Environment Variables
- `HT_CONFIG_FILE` : This variable can be set to the config file name or full path. Disabled tests will be read from this.
- `HT_LOG_ENABLE` : This is for debugging the HIP Test Framework itself. Setting it to 1, all `LogPrintf` will be printed on screen

## Test Macros
### Single Thread Macros
These macros are to be used when your test is calling HIP APIs via the main thread.

- `HIP_CHECK` : This macro takes in a HIP API and tests for its result to be either ```hipSuccess``` or ```hipErrorPeerAccessAlreadyEnabled```.

  - Usage: ```HIP_CHECK(hipMalloc(&dPtr, 10));```

- ```HIP_CHECK_ERROR``` : This macro takes in a HIP API and tests its result against a provided result. This can be used when the API is expected to fail with a particular result.

  - Usage: ```HIP_CHECK_ERROR(hipMalloc(&dPtr, 0), hipErrorInvalidValue);```

- ```HIPRTC_CHECK``` : This macro takes in a HIPRTC API and tests its result against HIPRTC_SUCCESS.

  - Usage: ```HIPRTC_CHECK(hiprtcCompileProgram(prog, count, options));```

- ```HIP_ASSERT``` : This macro takes in a bool condition as input and does a ```REQUIRE``` on the condition.

  - Usage: ```HIP_ASSERT(result == 10);```

### Multi Thread Macros
These macros are to be used when you call HIP APIs in a multi threaded way. They exist because Catch2 ```REQUIRE``` and ```CHECK``` macros can not handle multi threaded calls. To solve this problem, two macros are added```HIP_CHECK_THREAD``` and ```REQUIRE_THREAD``` which can be used to check result of HIP APIs and test assertions respectively. The results can be validate after the threads join via ```HIP_CHECK_THREAD_FINALIZE```.

Note: These should used in ```std::thread``` only. For multi proc guidelines look at [MultiProc Macros](#multi-process-macros) and [SpawnProc Class](#multiproc-management-class)

- ```HIP_CHECK_THREAD``` : This macro takes in a HIP API and tests for its result to be either ```hipSuccess``` or ```hipErrorPeerAccessAlreadyEnabled```. It can also tell other threads if an error has occured in one of the HIP API and can prematurely stop the threads.

- ```REQUIRE_THREAD``` : This macro takes in a bool condition and tests for its result to be true. If this check fails, it can signal other threads to terminate early.

- ```HIP_CHECK_THREAD_FINALIZE``` : This macro checks for the results logged by ```HIP_CHECK_THREAD```. This needs to be called after the threads have joined.

Please also note that you can not return values in functions calling ```HIP_CHECK_THREAD``` or ```REQUIRE_THREAD``` macro.

  Usage:

  ```cpp
  auto threadFunc = []() {
      int *dPtr{nullptr};
      HIP_CHECK_THREAD(hipMalloc(&dPtr, 10));
      REQUIRE_THREAD(dPtr != nullptr);
      // Some other work
    };

    // Launch threads
    std::vector<std::thread> threadPool;
    for(...) {
        threadPool.emplace_back(std::thread(threadFunc));
    }

    // Join threads
    for(auto &i : threadPool) {
        i.join();
    }

    // Validate all results
    HIP_CHECK_THREAD_FINALIZE();
  ```

### Skipping Tests if certain criteria is not met
If there arises a condition where certain flag is disabled and due to which a test can not run at that time, the following macro can be of use. It will highlight the test in ctest report as well.

- ```HIP_SKIP_TEST``` : The api takes in an input of the reason as well and prints out the line HIP_SKIP_THIS_TEST. This causes ctest to mark the test as skipped and the test shows up in the report as skipped prompting proper response from the team.

  Usage:

  ```cpp
  TEST_CASE("TestOnlyOnXnack") {
    if(!XNACKEnabled) {
      HIP_SKIP_TEST("Test only runs on system with XNACK enabled");
      return;
    }
    // Rest of test functionality
  }
  ```

### Multi Process Macros
These macros are to be called in multi process tests, inside a process which gets spawned. The reasoning is the same, Catch2 does not support multi process checks.

- ```HIPCHECK``` : Same as ```HIP_CHECK``` but will not call Catch2's ```REQUIRE``` on the HIP API. It will print if there is a mismatch and exit the process.

- ```HIPASSERT``` : Same as ```HIP_ASSERT``` but will not call Catch2's ```REQUIRE``` on the HIP API. It will print if there is a mismatch and exit the process.

## MultiProc Management Class
There is a special interface available for process isolation. ```hip::SpawnProc``` in ```hip_test_process.hh```. Using this interface test can spawn a process and place passing conditions on its return value or its output to stdout. This can be useful for testing printf output.
Sample Usage:
```cpp
hip::SpawnProc proc(<relative path of exe with test folder>, <optional bool value, if output is to be recorded>);
REQUIRE(0 == proc.run()); // Test of return value of the proc
REQUIRE(exepctedOutput == proc.getOutput()); // Test on expected output of the process
```
The process can be a standalone exe (see tests/catch/unit/printfExe for more information).

## Enabling New Tests
Initially, the new tests can be enabled via using ```-DHIP_CATCH_TEST=1```. After porting existing tests, this will be turned on by default.

## Building a single test
```bash
hipcc <path_to_test.cpp> -I<HIP_SRC_DIR>/tests/catch/include <HIP_SRC_DIR>/tests/catch/hipTestMain/standalone_main.cc -I<HIP_SRC_DIR>/tests/catch/external/Catch2 -g -o <out_file_name>
```

## Debugging support
Catch2 allows multiple ways in which you can debug the test case.
- `-b` options breaks into a debugger as soon as there is a failure encountered [Catch2 Options Reference](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md#breaking-into-the-debugger)
- Catch2 provided [logging macro](https://github.com/catchorg/Catch2/blob/v2.13.6/docs/logging.md#top) that print useful information on test case failure 
- User can also call [CATCH_BREAK_INTO_DEBUGGER](https://github.com/catchorg/Catch2/blob/devel/docs/configuration.md#overriding-catchs-debug-break--b) macro to break at a certain point in a test case.
- User can also mention filename.cc:__LineNumber__ to break into a test case via gdb.

## External Libs being used
- [Catch2](https://github.com/catchorg/Catch2) - Testing framework
- [picojson](https://github.com/kazuho/picojson) - For config file parsing

# Testing Guidelines
Tests fall in 5 categories and its file name prefix are as follows:
 - Unit tests (Prefix: Unit_\*API\*_\*Optional Scenario\*, example : Unit_hipMalloc_Negative or Unit_hipMalloc): Unit Tests are simplest test for an API, the target here is to test the API with different types of input and different ways of calling.
 - Application Behavior Modelling tests (Prefix: ABM_\*Intent\*_\*Optional Scenario\*, example: ABM_ModuleLoadAndRun): ABM tests are used to model a specific use case of HIP APIs, either seen in a customer app or a general purpose app. It mimics the calling behavior seen in aforementioned app.
 - Stress/Scale tests (Prefix: Stress_\*API\*_\*Intent\*_\*Optional Scenario\*, example: Stress_hipMemset_ExhaustVRAM): These tests are used to see the behavior of HIP APIs in edge scenarios, for example what happens when we have exhausted vram and do a hipMalloc or run many instances of same API in parallel.
 - Multi Process tests (Prefix: MultiProc_\*API\*_\*Optional Scenario\*, example: MultiProc_hipIPCMemHandle_GetDataFromProc): These tests are multi process tests and will only run on linux. They are used to test HIP APIs in multi process environment
 - Performance tests(Prefix: Perf_\*Intent\*_\*Optional Scenario\*, example: Perf_DispatchLatenc  y): Performance tests are used to get results of HIP APIs.

# General Guidelines:
 - Do not use the catch2 tags. Tags wont be used for filtering
 - Add as many INFO() as you can in tests which prints state of the t est, this will help the debugger when the test fails (INFO macro only prints when the test fails)
 - Check return of each HIP API and fail whenever there is a misma    tch with hipSuccess or hiprtcSuccess.
 - Each Category of test will hav e its own exe and catch_discover_test macro will be called on it to discover its tests
 - Optional Scenario in test names are optional. For example you  can test all Scenarios of hipMalloc API in one file, you can name the file Unit_hipMalloc, if you are having a file just for negative scenarios you can name it as Unit_hipMalloc_Negative.
