# HIP Tests - with Catch2

## Intro and Motivation
HIP Tests were using HIT framework (a custom framework tailored for HIP) to add, build and run tests. As time progressed the frame got big and took substantial amount of effort to maintain and extend. It also took substantial amount of time to configure. We took this oppurtunity to rewrite the HIP's testing framework and porting the test infra to Catch2 format.

## How to write tests
Tests in Catch2 are declared via ```TEST_CASE```.

[Please read the Catch2 documentation on how to write test cases](https://github.com/catchorg/Catch2/blob/v2.13.6/docs/tutorial.md#top)

[Catch2 Detailed Reference](https://github.com/catchorg/Catch2/blob/v2.13.6/docs/Readme.md#top)

## Taking care of existing features
- Don’t build on platform: EXCLUDE_(HIP_PLATFORM/HIP_RUNTIME), can be done via CMAKE. Adding source in if(HIP_PLATFORM == amd/nvidia).
- HIPCC_OPTIONS/CLANG Options: Can be done via: set_source_files_properties(src.cc PROPERTIES COMPILE_FLAGS “…”).
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
- `std::vector<std::string> getDevices()` : returns a vector of strings that contains device names (eg: For AMD: gfx906, gfx908 etc / For NVIDIA: RTX 2070 Super)
- `std::vector<std::string> getTargetId()` : (AMD Only) returns target id for gpus (eg: gfx906:sramecc+:xnack- etc)

This information can be accessed in any test via using: `TestContext::get().isAmd()`.

## Config file schema
Some tests can be skipped using a config file placed in same directory as the exe.

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

## Env Variables
- `HT_CONFIG_FILE` : This variable can be set to the config file name or full path. Disabled tests will be read from this.
- `HT_LOG_ENABLE` : This is for debugging the HIP Test Framework itself. Setting it to 1, all `LogPrintf` will be printed on screen

## Enabling New Tests
Initially, the new tests can be enabled via using ```-DHIP_CATCH_TEST=ON```. After porting existing tests, this will be turned on by default.

## Building a single test
```bash
hipcc <path_to_test.cpp> -I<HIP_SRC_DIR>/tests/newTests/include <HIP_SRC_DIR>/tests/newTests/hipTestMain/standalone_main.cc -I<HIP_SRC_DIR>/tests/newTests/external/Catch2 -g -o <out_file_name>
```

## Debugging support
Catch2 allows multiple ways in which you can debug the test case.
- `-b` options breaks into a debugger as soon as there is a failure encountered [Catch2 Options Reference](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md#breaking-into-the-debugger)
- Catch2 provided [logging macro](https://github.com/catchorg/Catch2/blob/v2.13.6/docs/logging.md#top) that print useful information on test case failure 
- User can also call [CATCH_BREAK_INTO_DEBUGGER](https://github.com/catchorg/Catch2/blob/devel/docs/configuration.md#overriding-catchs-debug-break--b) macro to break at a certain point in a test case.
- User can also mention filename.cc:__LineNumber__ to break into a test case via gdb.

## External Libs being used
- [Catch2](https://github.com/catchorg/Catch2) - Testing framework
- [taocpp/json](https://github.com/taocpp/json) - For config file parsing
- [taocpp/PEGTL](https://github.com/taocpp/PEGTL) - Helper lib for taojson
