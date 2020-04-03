/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "cl_common.hpp"
#include "vdi_common.hpp"
#ifdef _WIN32
#include <d3d10_1.h>
#include "cl_d3d9_amd.hpp"
#include "cl_d3d10_amd.hpp"
#include "cl_d3d11_amd.hpp"
#endif  //_WIN32

#include <icd/loader/icd_dispatch.h>

#include <mutex>

amd::PlatformIDS amd::PlatformID::Platform =  //{ NULL };
    {amd::ICDDispatchedObject::icdVendorDispatch_};

static cl_int CL_API_CALL icdGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name,
                                             size_t param_value_size, void* param_value,
                                             size_t* param_value_size_ret) {
  return clGetPlatformInfo(NULL, param_name, param_value_size, param_value, param_value_size_ret);
}

static cl_int CL_API_CALL icdGetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
                                          cl_uint num_entries, cl_device_id* devices,
                                          cl_uint* num_devices) {
  return clGetDeviceIDs(NULL, device_type, num_entries, devices, num_devices);
}

static cl_int CL_API_CALL icdGetDeviceInfo(cl_device_id device, cl_device_info param_name,
                                           size_t param_value_size, void* param_value,
                                           size_t* param_value_size_ret) {
  if (param_name == CL_DEVICE_PLATFORM) {
    // Return the ICD platform instead of the default NULL platform.
    cl_platform_id platform = reinterpret_cast<cl_platform_id>(&amd::PlatformID::Platform);
    return amd::clGetInfo(platform, param_value_size, param_value, param_value_size_ret);
  }

  return clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_icd_dispatch amd::ICDDispatchedObject::icdVendorDispatch_[] = {
    {NULL /* should not get called */, icdGetPlatformInfo, icdGetDeviceIDs, icdGetDeviceInfo,
     clCreateContext, clCreateContextFromType, clRetainContext, clReleaseContext, clGetContextInfo,
     clCreateCommandQueue, clRetainCommandQueue, clReleaseCommandQueue, clGetCommandQueueInfo,
     clSetCommandQueueProperty, clCreateBuffer, clCreateImage2D, clCreateImage3D, clRetainMemObject,
     clReleaseMemObject, clGetSupportedImageFormats, clGetMemObjectInfo, clGetImageInfo,
     clCreateSampler, clRetainSampler, clReleaseSampler, clGetSamplerInfo,
     clCreateProgramWithSource, clCreateProgramWithBinary, clRetainProgram, clReleaseProgram,
     clBuildProgram, clUnloadCompiler, clGetProgramInfo, clGetProgramBuildInfo, clCreateKernel,
     clCreateKernelsInProgram, clRetainKernel, clReleaseKernel, clSetKernelArg, clGetKernelInfo,
     clGetKernelWorkGroupInfo, clWaitForEvents, clGetEventInfo, clRetainEvent, clReleaseEvent,
     clGetEventProfilingInfo, clFlush, clFinish, clEnqueueReadBuffer, clEnqueueWriteBuffer,
     clEnqueueCopyBuffer, clEnqueueReadImage, clEnqueueWriteImage, clEnqueueCopyImage,
     clEnqueueCopyImageToBuffer, clEnqueueCopyBufferToImage, clEnqueueMapBuffer, clEnqueueMapImage,
     clEnqueueUnmapMemObject, clEnqueueNDRangeKernel, clEnqueueTask, clEnqueueNativeKernel,
     clEnqueueMarker, clEnqueueWaitForEvents, clEnqueueBarrier, clGetExtensionFunctionAddress,
     clCreateFromGLBuffer, clCreateFromGLTexture2D, clCreateFromGLTexture3D,
     clCreateFromGLRenderbuffer, clGetGLObjectInfo, clGetGLTextureInfo, clEnqueueAcquireGLObjects,
     clEnqueueReleaseGLObjects, clGetGLContextInfoKHR,
     WINDOWS_SWITCH(clGetDeviceIDsFromD3D10KHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D10BufferKHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D10Texture2DKHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D10Texture3DKHR, NULL),
     WINDOWS_SWITCH(clEnqueueAcquireD3D10ObjectsKHR, NULL),
     WINDOWS_SWITCH(clEnqueueReleaseD3D10ObjectsKHR, NULL), clSetEventCallback, clCreateSubBuffer,
     clSetMemObjectDestructorCallback, clCreateUserEvent, clSetUserEventStatus,
     clEnqueueReadBufferRect, clEnqueueWriteBufferRect, clEnqueueCopyBufferRect,
     NULL, NULL, NULL, clCreateEventFromGLsyncKHR,

     /* OpenCL 1.2*/
     clCreateSubDevices, clRetainDevice, clReleaseDevice, clCreateImage,
     clCreateProgramWithBuiltInKernels, clCompileProgram, clLinkProgram, clUnloadPlatformCompiler,
     clGetKernelArgInfo, clEnqueueFillBuffer, clEnqueueFillImage, clEnqueueMigrateMemObjects,
     clEnqueueMarkerWithWaitList, clEnqueueBarrierWithWaitList,
     clGetExtensionFunctionAddressForPlatform, clCreateFromGLTexture,

     WINDOWS_SWITCH(clGetDeviceIDsFromD3D11KHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D11BufferKHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D11Texture2DKHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D11Texture3DKHR, NULL),
     WINDOWS_SWITCH(clCreateFromDX9MediaSurfaceKHR, NULL),
     WINDOWS_SWITCH(clEnqueueAcquireD3D11ObjectsKHR, NULL),
     WINDOWS_SWITCH(clEnqueueReleaseD3D11ObjectsKHR, NULL),

     WINDOWS_SWITCH(clGetDeviceIDsFromDX9MediaAdapterKHR,
                    NULL),  // KHRpfn_clGetDeviceIDsFromDX9MediaAdapterKHR
                            // clGetDeviceIDsFromDX9MediaAdapterKHR;
     WINDOWS_SWITCH(
         clEnqueueAcquireDX9MediaSurfacesKHR,
         NULL),  // KHRpfn_clEnqueueAcquireDX9MediaSurfacesKHR clEnqueueAcquireDX9MediaSurfacesKHR;
     WINDOWS_SWITCH(
         clEnqueueReleaseDX9MediaSurfacesKHR,
         NULL),  // KHRpfn_clEnqueueReleaseDX9MediaSurfacesKHR clEnqueueReleaseDX9MediaSurfacesKHR;

     NULL,
     NULL, NULL, NULL,

     clCreateCommandQueueWithProperties, clCreatePipe, clGetPipeInfo, clSVMAlloc, clSVMFree,
     clEnqueueSVMFree, clEnqueueSVMMemcpy, clEnqueueSVMMemFill, clEnqueueSVMMap, clEnqueueSVMUnmap,
     clCreateSamplerWithProperties, clSetKernelArgSVMPointer, clSetKernelExecInfo,
     clGetKernelSubGroupInfo,
     clCloneKernel,
     clCreateProgramWithIL,
     clEnqueueSVMMigrateMem,
     clGetDeviceAndHostTimer,
     clGetHostTimer,
     clGetKernelSubGroupInfo,
     clSetDefaultDeviceCommandQueue,

     clSetProgramReleaseCallback,
     clSetProgramSpecializationConstant }};

#if defined(ATI_OS_WIN)
#include <Shlwapi.h>

#pragma comment(lib, "shlwapi.lib")

static bool ShouldLoadPlatform() {
  // Get the OpenCL ICD registry values
  HKEY platformsKey = NULL;
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Khronos\\OpenCL\\Vendors", 0, KEY_READ,
                    &platformsKey) != ERROR_SUCCESS)
    return true;

  std::vector<std::string> registryValues;
  DWORD dwIndex = 0;
  while (true) {
    char cszLibraryName[1024] = {0};
    DWORD dwLibraryNameSize = sizeof(cszLibraryName);
    DWORD dwLibraryNameType = 0;
    DWORD dwValue = 0;
    DWORD dwValueSize = sizeof(dwValue);

    if (RegEnumValueA(platformsKey, dwIndex++, cszLibraryName, &dwLibraryNameSize, NULL,
                      &dwLibraryNameType, (LPBYTE)&dwValue, &dwValueSize) != ERROR_SUCCESS)
      break;
    // Require that the value be a DWORD and equal zero
    if (dwLibraryNameType != REG_DWORD || dwValue != 0) {
      continue;
    }
    registryValues.push_back(cszLibraryName);
  }
  RegCloseKey(platformsKey);

  HMODULE hm = NULL;
  if (!GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          (LPCSTR)&ShouldLoadPlatform, &hm))
    return true;

  char cszDllPath[1024] = {0};
  if (!GetModuleFileNameA(hm, cszDllPath, sizeof(cszDllPath))) return true;

  // If we are loaded from the DriverStore, then there should be a registry
  // value matching our current module absolute path.
  if (std::find(registryValues.begin(), registryValues.end(), cszDllPath) == registryValues.end())
    return true;

  LPSTR cszFileName;
  char buffer[1024] = {0};
  if (!GetFullPathNameA(cszDllPath, sizeof(buffer), buffer, &cszFileName)) return true;

  // We found an absolute path in the registry that matched this DLL, now
  // check if there is also an entry with the same filename.
  if (std::find(registryValues.begin(), registryValues.end(), cszFileName) == registryValues.end())
    return true;

  // Lastly, check if there is a DLL with the same name in the System folder.
  char cszSystemPath[1024] = {0};
#if defined(ATI_BITS_32)
  if (!GetSystemWow64DirectoryA(cszSystemPath, sizeof(cszSystemPath)))
#endif  // defined(ATI_BITS_32)
    if (!GetSystemDirectoryA(cszSystemPath, sizeof(cszSystemPath))) return true;

  std::string systemDllPath;
  systemDllPath.append(cszSystemPath).append("\\").append(cszFileName);
  if (!PathFileExistsA(systemDllPath.c_str())) {
    return true;
  }

  // If we get here, then all 3 conditions are true:
  // - An entry in the registry with an absolute path matches the current DLL
  // - An entry in the registry with a relative path matches the current DLL
  // - A DLL with the same name was found in the system directory
  //
  // We should not load this platform!

  return false;
}

#else

#include <dlfcn.h>

// If there is only one platform, load it.
// If there is more than one platform, only load platforms that have visible devices
// If all platforms have no devices available, only load the PAL platform
static bool ShouldLoadPlatform() {
  bool shouldLoad = true;

  if (!amd::Runtime::initialized()) {
    amd::Runtime::init();
  }
  const int numDevices = amd::Device::numDevices(CL_DEVICE_TYPE_GPU, false);

  void *otherPlatform = nullptr;
  if (amd::IS_LEGACY) {
    otherPlatform = dlopen("libamdocl64.so", RTLD_LAZY);
    if (otherPlatform != nullptr) { // Present platform exists
      shouldLoad = numDevices > 0;
    }
  } else {
    otherPlatform = dlopen("libamdocl-orca64.so", RTLD_LAZY);
    if (otherPlatform != nullptr) { // Legacy platform exists
      // gcc4.8 doesn't support casting void* to a function pointer
      // Work around this by creating a typedef untill we upgrade the compiler
      typedef void*(*clGetFunctionAddress_t)(const char *);
      typedef cl_int(*clIcdGetPlatformIDs_t)(cl_uint, cl_platform_id *, cl_uint *);

      clGetFunctionAddress_t legacyGetFunctionAddress =
        reinterpret_cast<clGetFunctionAddress_t>(dlsym(otherPlatform, "clGetExtensionFunctionAddress"));
      clIcdGetPlatformIDs_t legacyGetPlatformIDs =
        reinterpret_cast<clIcdGetPlatformIDs_t>(legacyGetFunctionAddress("clIcdGetPlatformIDsKHR"));

      cl_uint numLegacyPlatforms = 0;
      legacyGetPlatformIDs(0, nullptr, &numLegacyPlatforms);

      shouldLoad = (numDevices > 0) || (numLegacyPlatforms == 0);
    }
  }

  if (otherPlatform != nullptr) {
    dlclose(otherPlatform);
  }

  return shouldLoad;
}

#endif // defined(ATI_OS_WIN)

CL_API_ENTRY cl_int CL_API_CALL clIcdGetPlatformIDsKHR(cl_uint num_entries,
                                                       cl_platform_id* platforms,
                                                       cl_uint* num_platforms) {
  if (((num_entries > 0 || num_platforms == NULL) && platforms == NULL) ||
      (num_entries == 0 && platforms != NULL)) {
    return CL_INVALID_VALUE;
  }

  static bool shouldLoad = true;

  static std::once_flag initOnce;
  std::call_once(initOnce, [](){ shouldLoad = ShouldLoadPlatform(); });

  if (!shouldLoad) {
    *not_null(num_platforms) = 0;
    return CL_SUCCESS;
  }

  if (!amd::Runtime::initialized()) {
    amd::Runtime::init();
  }

  if (num_platforms != NULL && platforms == NULL) {
    *num_platforms = 1;
    return CL_SUCCESS;
  }

  assert(platforms != NULL && "check the code above");
  *platforms = reinterpret_cast<cl_platform_id>(&amd::PlatformID::Platform);

  *not_null(num_platforms) = 1;
  return CL_SUCCESS;
}
