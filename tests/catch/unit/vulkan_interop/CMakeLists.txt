set(TEST_SRC
    vulkan_test.cc
    hipSignalExternalSemaphoresAsync.cc
   )

find_package(Vulkan)
if(NOT Vulkan_FOUND)
  if(EXISTS "${VULKAN_PATH}")
    message(STATUS "Vulkan SDK: ${VULKAN_PATH}")
  elseif (EXISTS "$ENV{VULKAN_SDK}")
    message(STATUS "FOUND VULKAN SDK: $ENV{VULKAN_SDK}")
    set(VULKAN_PATH  $ENV{VULKAN_SDK})
  else()
    message("Error: Unable to locate Vulkan SDK. please specify VULKAN_PATH")
    return()
  endif()
endif()

hip_add_exe_to_target(NAME VulkanInteropTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
if (WIN32)
  target_link_libraries(VulkanInteropTest vulkan-1)
else (WIN32)
  target_link_libraries(VulkanInteropTest vulkan)
endif (WIN32)