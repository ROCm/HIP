# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
Catch
-----

This module defines a function to help use the Catch test framework.

The :command:`catch_discover_tests` discovers tests by asking the compiled test
executable to enumerate its tests.  This does not require CMake to be re-run
when tests change.  However, it may not work in a cross-compiling environment,
and setting test properties is less convenient.

This command is intended to replace use of :command:`add_test` to register
tests, and will create a separate CTest test for each Catch test case.  Note
that this is in some cases less efficient, as common set-up and tear-down logic
cannot be shared by multiple test cases executing in the same instance.
However, it provides more fine-grained pass/fail information to CTest, which is
usually considered as more beneficial.  By default, the CTest test name is the
same as the Catch name; see also ``TEST_PREFIX`` and ``TEST_SUFFIX``.

.. command:: catch_discover_tests

  Automatically add tests with CTest by querying the compiled test executable
  for available tests::

    catch_discover_tests(target
                         [TEST_SPEC arg1...]
                         [EXTRA_ARGS arg1...]
                         [WORKING_DIRECTORY dir]
                         [TEST_PREFIX prefix]
                         [TEST_SUFFIX suffix]
                         [PROPERTIES name1 value1...]
                         [TEST_LIST var]
                         [REPORTER reporter]
                         [OUTPUT_DIR dir]
                         [OUTPUT_PREFIX prefix}
                         [OUTPUT_SUFFIX suffix]
    )

  ``catch_discover_tests`` sets up a post-build command on the test executable
  that generates the list of tests by parsing the output from running the test
  with the ``--list-test-names-only`` argument.  This ensures that the full
  list of tests is obtained.  Since test discovery occurs at build time, it is
  not necessary to re-run CMake when the list of tests changes.
  However, it requires that :prop_tgt:`CROSSCOMPILING_EMULATOR` is properly set
  in order to function in a cross-compiling environment.

  Additionally, setting properties on tests is somewhat less convenient, since
  the tests are not available at CMake time.  Additional test properties may be
  assigned to the set of tests as a whole using the ``PROPERTIES`` option.  If
  more fine-grained test control is needed, custom content may be provided
  through an external CTest script using the :prop_dir:`TEST_INCLUDE_FILES`
  directory property.  The set of discovered tests is made accessible to such a
  script via the ``<target>_TESTS`` variable.

  The options are:

  ``target``
    Specifies the Catch executable, which must be a known CMake executable
    target.  CMake will substitute the location of the built executable when
    running the test.

  ``TEST_SPEC arg1...``
    Specifies test cases, wildcarded test cases, tags and tag expressions to
    pass to the Catch executable with the ``--list-test-names-only`` argument.

  ``EXTRA_ARGS arg1...``
    Any extra arguments to pass on the command line to each test case.

  ``WORKING_DIRECTORY dir``
    Specifies the directory in which to run the discovered test cases.  If this
    option is not provided, the current binary directory is used.

  ``TEST_PREFIX prefix``
    Specifies a ``prefix`` to be prepended to the name of each discovered test
    case.  This can be useful when the same test executable is being used in
    multiple calls to ``catch_discover_tests()`` but with different
    ``TEST_SPEC`` or ``EXTRA_ARGS``.

  ``TEST_SUFFIX suffix``
    Similar to ``TEST_PREFIX`` except the ``suffix`` is appended to the name of
    every discovered test case.  Both ``TEST_PREFIX`` and ``TEST_SUFFIX`` may
    be specified.

  ``PROPERTIES name1 value1...``
    Specifies additional properties to be set on all tests discovered by this
    invocation of ``catch_discover_tests``.

  ``TEST_LIST var``
    Make the list of tests available in the variable ``var``, rather than the
    default ``<target>_TESTS``.  This can be useful when the same test
    executable is being used in multiple calls to ``catch_discover_tests()``.
    Note that this variable is only available in CTest.

  ``REPORTER reporter``
    Use the specified reporter when running the test case. The reporter will
    be passed to the Catch executable as ``--reporter reporter``.

  ``OUTPUT_DIR dir``
    If specified, the parameter is passed along as
    ``--out dir/<test_name>`` to Catch executable. The actual file name is the
    same as the test name. This should be used instead of
    ``EXTRA_ARGS --out foo`` to avoid race conditions writing the result output
    when using parallel test execution.

  ``OUTPUT_PREFIX prefix``
    May be used in conjunction with ``OUTPUT_DIR``.
    If specified, ``prefix`` is added to each output file name, like so
    ``--out dir/prefix<test_name>``.

  ``OUTPUT_SUFFIX suffix``
    May be used in conjunction with ``OUTPUT_DIR``.
    If specified, ``suffix`` is added to each output file name, like so
    ``--out dir/<test_name>suffix``. This can be used to add a file extension to
    the output e.g. ".xml".

#]=======================================================================]

#------------------------------------------------------------------------------
function(catch_discover_tests TARGET)
  cmake_parse_arguments(
    ""
    ""
    "TEST_PREFIX;TEST_SUFFIX;WORKING_DIRECTORY;TEST_LIST;REPORTER;OUTPUT_DIR;OUTPUT_PREFIX;OUTPUT_SUFFIX"
    "TEST_SPEC;EXTRA_ARGS;PROPERTIES"
    ${ARGN}
  )

  if(NOT _WORKING_DIRECTORY)
    set(_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
  if(NOT _TEST_LIST)
    set(_TEST_LIST ${TARGET}_TESTS)
  endif()

  ## Generate a unique name based on the extra arguments
  string(SHA1 args_hash "${_TEST_SPEC} ${_EXTRA_ARGS} ${_REPORTER} ${_OUTPUT_DIR} ${_OUTPUT_PREFIX} ${_OUTPUT_SUFFIX}")
  string(SUBSTRING ${args_hash} 0 7 args_hash)

  # Define rule to generate test list for aforementioned test executable
  set(ctest_include_file "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_include-${args_hash}.cmake")
  set(ctest_tests_file "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_tests-${args_hash}.cmake")
  file(RELATIVE_PATH ctestincludepath ${CMAKE_CURRENT_BINARY_DIR} ${ctest_include_file})
  file(RELATIVE_PATH ctestfilepath ${CMAKE_CURRENT_BINARY_DIR} ${ctest_tests_file})
  file(RELATIVE_PATH _workdir ${CMAKE_CURRENT_BINARY_DIR} ${_WORKING_DIRECTORY})
  file(RELATIVE_PATH _CATCH_ADD_TEST_SCRIPT ${CMAKE_CURRENT_BINARY_DIR} ${ADD_SCRIPT_PATH})

  get_property(crosscompiling_emulator
    TARGET ${TARGET}
    PROPERTY CROSSCOMPILING_EMULATOR
  )

  set(EXEC_NAME ${TARGET})
  if(WIN32)
    set(EXEC_NAME ${EXEC_NAME}.exe)
  endif()

  # uses catch_include.cmake.in file to generate the *_include.cmake file
  # *_include.cmake is used to generate the *_test.cmake during execution of ctest cmd
  configure_file(${CATCH2_INCLUDE} ${TARGET}_include-${args_hash}.cmake @ONLY)

  if(NOT ${CMAKE_VERSION} VERSION_LESS "3.10.0") 
    # Add discovered tests to directory TEST_INCLUDE_FILES
    set_property(DIRECTORY
      APPEND PROPERTY TEST_INCLUDE_FILES "${ctestincludepath}"
    )
  else()
    # Add discovered tests as directory TEST_INCLUDE_FILE if possible
    get_property(test_include_file_set DIRECTORY PROPERTY TEST_INCLUDE_FILE SET)
    if (NOT ${test_include_file_set})
      set_property(DIRECTORY
        PROPERTY TEST_INCLUDE_FILE "${ctestincludepath}"
      )
    else()
      message(FATAL_ERROR
        "Cannot set more than one TEST_INCLUDE_FILE"
      )
    endif()
  endif()

endfunction()

###############################################################################

set(_CATCH_DISCOVER_TESTS_SCRIPT
  ${CMAKE_CURRENT_LIST_DIR}/CatchAddTests.cmake
  CACHE INTERNAL "Catch2 full path to CatchAddTests.cmake helper file"
)

###############################################################################
# function to be called by all tests
function(hip_add_exe_to_target)
  set(options)
  set(args NAME TEST_TARGET_NAME PLATFORM COMPILE_OPTIONS)
  set(list_args TEST_SRC LINKER_LIBS PROPERTY)
  cmake_parse_arguments(
    PARSE_ARGV 0
    "" # variable prefix
    "${options}"
    "${args}"
    "${list_args}"
  )
  # Create shared lib of all tests
  if(NOT RTC_TESTING)
  add_executable(${_NAME} EXCLUDE_FROM_ALL ${_TEST_SRC} $<TARGET_OBJECTS:Main_Object> $<TARGET_OBJECTS:KERNELS>)
  else ()
  add_executable(${_NAME} EXCLUDE_FROM_ALL ${_TEST_SRC} $<TARGET_OBJECTS:Main_Object>)
    if(HIP_PLATFORM STREQUAL "amd")
      target_link_libraries(${_NAME} hiprtc)
    else()
      target_link_libraries(${_NAME} nvrtc)
    endif()
  endif()
  catch_discover_tests(${_NAME} PROPERTIES  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
  if(UNIX)
    set(_LINKER_LIBS ${_LINKER_LIBS} stdc++fs)
  else()
    # res files are built resource files using rc files.
    # use llvm-rc exe to build the res files
    # Thes are used to populate the properties of the built executables
    if(EXISTS "${PROP_RC}/catchProp.res")
      set(_LINKER_LIBS ${_LINKER_LIBS} "${PROP_RC}/catchProp.res")
    endif()
  endif()

  if(DEFINED _LINKER_LIBS)
    target_link_libraries(${_NAME} ${_LINKER_LIBS})
  endif()

  # Add dependency on build_tests to build it on this custom target
  add_dependencies(${_TEST_TARGET_NAME} ${_NAME})

  if (DEFINED _PROPERTY)
    set_property(TARGET ${_NAME} PROPERTY ${_PROPERTY})
  endif()

  if (DEFINED _COMPILE_OPTIONS)
    target_compile_options(${_NAME} PUBLIC ${_COMPILE_OPTIONS})
  endif()

  foreach(arg IN LISTS _UNPARSED_ARGUMENTS)
      message(WARNING "Unparsed arguments: ${arg}")
  endforeach()
endfunction()


