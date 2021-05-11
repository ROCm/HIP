/* Copyright (c) 2009-present Advanced Micro Devices, Inc.

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

#ifdef _WIN32

#include "top.hpp"

#include "cl_common.hpp"
#include "cl_d3d10_amd.hpp"
#include "platform/command.hpp"

#include <cstring>
#include <utility>


/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_D3D10_Interops
 *
 *  This section discusses OpenCL functions that allow applications to use Direct3D 10
 * resources (buffers/textures) as OpenCL memory objects. This allows efficient sharing of
 * data between OpenCL and Direct3D 10. The OpenCL API can be used to execute kernels that
 * read and/or write memory objects that are also the Direct3D resources.
 * An OpenCL image object can be created from a D3D10 texture object. An
 * OpenCL buffer object can be created from a D3D10 buffer object (index/vertex).
 *
 *  @}
 *  \addtogroup clGetDeviceIDsFromD3D10KHR
 *  @{
 */

RUNTIME_ENTRY(cl_int, clGetDeviceIDsFromD3D10KHR,
              (cl_platform_id platform, cl_d3d10_device_source_khr d3d_device_source,
               void* d3d_object, cl_d3d10_device_set_khr d3d_device_set, cl_uint num_entries,
               cl_device_id* devices, cl_uint* num_devices)) {
  cl_int errcode;
  ID3D10Device* d3d10_device = NULL;
  cl_device_id* gpu_devices;
  cl_uint num_gpu_devices = 0;
  bool create_d3d10Device = false;
  static const bool VALIDATE_ONLY = true;
  HMODULE d3d10Module = NULL;

  if (platform != NULL && platform != AMD_PLATFORM) {
    LogWarning("\"platrform\" is not a valid AMD platform");
    return CL_INVALID_PLATFORM;
  }
  if (((num_entries > 0 || num_devices == NULL) && devices == NULL) ||
      (num_entries == 0 && devices != NULL)) {
    return CL_INVALID_VALUE;
  }
  // Get GPU devices
  errcode = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 0, NULL, &num_gpu_devices);
  if (errcode != CL_SUCCESS && errcode != CL_DEVICE_NOT_FOUND) {
    return CL_INVALID_VALUE;
  }

  if (!num_gpu_devices) {
    *not_null(num_devices) = 0;
    return CL_DEVICE_NOT_FOUND;
  }

  switch (d3d_device_source) {
    case CL_D3D10_DEVICE_KHR:
      d3d10_device = static_cast<ID3D10Device*>(d3d_object);
      break;
    case CL_D3D10_DXGI_ADAPTER_KHR: {
      typedef HRESULT(WINAPI * LPD3D10CREATEDEVICE)(IDXGIAdapter*, D3D10_DRIVER_TYPE, HMODULE, UINT,
                                                    UINT32, ID3D10Device**);
      static LPD3D10CREATEDEVICE dynamicD3D10CreateDevice = NULL;

      d3d10Module = LoadLibrary("D3D10.dll");
      if (d3d10Module == NULL) {
        return CL_INVALID_PLATFORM;
      }

      dynamicD3D10CreateDevice =
          (LPD3D10CREATEDEVICE)GetProcAddress(d3d10Module, "D3D10CreateDevice");

      IDXGIAdapter* dxgi_adapter = static_cast<IDXGIAdapter*>(d3d_object);
      HRESULT hr = dynamicD3D10CreateDevice(dxgi_adapter, D3D10_DRIVER_TYPE_HARDWARE, NULL, 0,
                                            D3D10_SDK_VERSION, &d3d10_device);
      if (SUCCEEDED(hr) && (NULL != d3d10_device)) {
        create_d3d10Device = true;
      } else {
        FreeLibrary(d3d10Module);
        return CL_INVALID_VALUE;
      }
    } break;
    default:
      LogWarning("\"d3d_device_source\" is invalid");
      return CL_INVALID_VALUE;
  }

  switch (d3d_device_set) {
    case CL_PREFERRED_DEVICES_FOR_D3D10_KHR:
    case CL_ALL_DEVICES_FOR_D3D10_KHR: {
      gpu_devices = (cl_device_id*)alloca(num_gpu_devices * sizeof(cl_device_id));

      errcode = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, num_gpu_devices, gpu_devices, NULL);
      if (errcode != CL_SUCCESS) {
        break;
      }

      void* external_device[amd::Context::DeviceFlagIdx::LastDeviceFlagIdx] = {};
      external_device[amd::Context::DeviceFlagIdx::D3D10DeviceKhrIdx] = d3d10_device;

      std::vector<amd::Device*> compatible_devices;
      for (cl_uint i = 0; i < num_gpu_devices; ++i) {
        cl_device_id device = gpu_devices[i];
        if (is_valid(device) &&
            as_amd(device)->bindExternalDevice(amd::Context::Flags::D3D10DeviceKhr, external_device,
                                               NULL, VALIDATE_ONLY)) {
          compatible_devices.push_back(as_amd(device));
        }
      }
      if (compatible_devices.size() == 0) {
        *not_null(num_devices) = 0;
        errcode = CL_DEVICE_NOT_FOUND;
        break;
      }

      auto it = compatible_devices.cbegin();
      cl_uint compatible_count = std::min(num_entries, (cl_uint)compatible_devices.size());

      while (compatible_count--) {
        *devices++ = as_cl(*it++);
        --num_entries;
      }
      while (num_entries--) {
        *devices++ = (cl_device_id)0;
      }

      *not_null(num_devices) = (cl_uint)compatible_devices.size();
    } break;

    default:
      LogWarning("\"d3d_device_set\" is invalid");
      errcode = CL_INVALID_VALUE;
  }

  if (create_d3d10Device) {
    d3d10_device->Release();
    FreeLibrary(d3d10Module);
  }
  return errcode;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromD3D10BufferKHR
 *  @{
 */

/*! \brief Creates an OpenCL buffer object from a Direct3D 10 resource.
 *
 *  \param context is a valid OpenCL context.
 *
 *  \param flags is a bit-field that is used to specify usage information.
 *  Only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY and CL_MEM_READ_WRITE values
 *  can be used.
 *
 *  \param pD3DResource is a valid pointer to a D3D10 resource of type ID3D10Buffer.
 *
 *  \return valid non-zero OpenCL buffer object and \a errcode_ret is set
 *  to CL_SUCCESS if the buffer object is created successfully. It returns a NULL
 *  value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context or if Direct3D 10
 *    interoperatbility has not been initialized between context and the ID3D10Device
 *    from which pD3DResource was created.
 *  - CL_INVALID_VALUE if values specified in \a clFlags are not valid.
 *  - CL_INVALID_D3D_RESOURCE if \a pD3DResource is not of type ID3D10Buffer.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33?
 */

RUNTIME_ENTRY_RET(cl_mem, clCreateFromD3D10BufferKHR,
                  (cl_context context, cl_mem_flags flags, ID3D10Buffer* pD3DResource,
                   cl_int* errcode_ret)) {
  cl_mem clMemObj = NULL;

  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return clMemObj;
  }
  if (!flags) flags = CL_MEM_READ_WRITE;
  if (!(((flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY) ||
        ((flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY) ||
        ((flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return clMemObj;
  }
  if (!pD3DResource) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("parameter \"pD3DResource\" is a NULL pointer");
    return clMemObj;
  }
  return (
      amd::clCreateBufferFromD3D10ResourceAMD(*as_amd(context), flags, pD3DResource, errcode_ret));
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateImageFromD3D10Resource
 *  @{
 */

/*! \brief Create an OpenCL 2D or 3D image object from a D3D10 resource.
 *
 *  \param context is a valid OpenCL context.
 *
 *  \param flags is a bit-field that is used to specify usage information.
 *  Only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY and CL_MEM_READ_WRITE values
 *  can be used.
 *
 *  \param pD3DResource is a valid pointer to a D3D10 resource of type
 *  ID3D10Texture2D, ID3D10Texture2D, or ID3D10Texture3D.
 * If pD3DResource is of type ID3D10Texture1D then the created image object
 * will be a 1D mipmapped image object.
 * If pD3DResource is of type ID3D10Texture2D and was not created with flag
 * D3D10_RESOURCE_MISC_TEXTURECUBE then the created image object will be a
 * 2D mipmapped image object.
 * If pD3DResource is of type ID3D10Texture2D and was created with flag
 * D3D10_RESOURCE_MISC_TEXTURECUBE then the created image object will be
 * a cubemap mipmapped image object.
 * errocde_ret returns CL_INVALID_D3D_RESOURCE if an OpenCL memory object has
 * already been created from pD3DResource in context.
 * If pD3DResource is of type ID3D10Texture3D then the created image object will
 * be a 3D mipmapped imageobject.
 *
 *  \return valid non-zero OpenCL image object and \a errcode_ret is set
 *  to CL_SUCCESS if the image object is created successfully. It returns a NULL
 *  value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context or if Direct3D 10
 *    interoperatbility has not been initialized between context and the ID3D10Device
 *    from which pD3DResource was created.
 *  - CL_INVALID_VALUE if values specified in \a flags are not valid.
 *  - CL_INVALID_D3D_RESOURCE if \a pD3DResource is not of type ID3D10Texture1D,
 *    ID3D10Texture2D, or ID3D10Texture3D.
 *  - CL_INVALID_D3D_RESOURCE if an OpenCL memory object has already been created
 *    from \a pD3DResource in context.
 *  - CL_INVALID_IMAGE_FORMAT if the Direct3D 10 texture format does not map
 *    to an appropriate OpenCL image format.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r48?
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateImageFromD3D10Resource,
                  (cl_context context, cl_mem_flags flags, ID3D10Resource* pD3DResource,
                   UINT subresource, int* errcode_ret, UINT dimension)) {
  cl_mem clMemObj = NULL;

  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return clMemObj;
  }
  if (!flags) flags = CL_MEM_READ_WRITE;
  if (!(((flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY) ||
        ((flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY) ||
        ((flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return clMemObj;
  }
  if (!pD3DResource) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("parameter \"pD3DResource\" is a NULL pointer");
    return clMemObj;
  }

  // Verify context init'ed for interop
  ID3D10Device* pDev;
  pD3DResource->GetDevice(&pDev);
  if (pDev == NULL) {
    *not_null(errcode_ret) = CL_INVALID_D3D10_DEVICE_KHR;
    LogWarning("Cannot retrieve D3D10 device from D3D10 resource");
    return (cl_mem)0;
  }
  pDev->Release();
  if (!((*as_amd(context)).info().flags_ & amd::Context::D3D10DeviceKhr)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("\"amdContext\" is not created from D3D10 device");
    return (cl_mem)0;
  }

  // Check for image support
  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  bool supportPass = false;
  bool sizePass = false;
  for (const auto& it : devices) {
    if (it->info().imageSupport_) {
      supportPass = true;
    }
  }
  if (!supportPass) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    LogWarning("there are no devices in context to support images");
    return (cl_mem)0;
  }

  switch (dimension) {
#if 0
    case 1:
        return(amd::clCreateImage1DFromD3D10ResourceAMD(
            *as_amd(context),
            flags,
            pD3DResource,
            subresource,
            errcode_ret));
#endif  // 0
    case 2:
      return (amd::clCreateImage2DFromD3D10ResourceAMD(*as_amd(context), flags, pD3DResource,
                                                       subresource, errcode_ret));
    case 3:
      return (amd::clCreateImage3DFromD3D10ResourceAMD(*as_amd(context), flags, pD3DResource,
                                                       subresource, errcode_ret));
    default:
      break;
  }

  *not_null(errcode_ret) = CL_INVALID_D3D10_RESOURCE_KHR;
  return (cl_mem)0;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromD3D10Texture2DKHR
 *  @{
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateFromD3D10Texture2DKHR,
                  (cl_context context, cl_mem_flags flags, ID3D10Texture2D* resource,
                   UINT subresource, cl_int* errcode_ret)) {
  return clCreateImageFromD3D10Resource(context, flags, resource, subresource, errcode_ret, 2);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromD3D10Texture3DKHR
 *  @{
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateFromD3D10Texture3DKHR,
                  (cl_context context, cl_mem_flags flags, ID3D10Texture3D* resource,
                   UINT subresource, cl_int* errcode_ret)) {
  return clCreateImageFromD3D10Resource(context, flags, resource, subresource, errcode_ret, 3);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clEnqueueAcquireD3D10ObjectsKHR
 *  @{
 */
RUNTIME_ENTRY(cl_int, clEnqueueAcquireD3D10ObjectsKHR,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clEnqueueAcquireExtObjectsAMD(command_queue, num_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event,
                                            CL_COMMAND_ACQUIRE_D3D10_OBJECTS_KHR);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clEnqueueReleaseD3D10ObjectsKHR
 *  @{
 */
RUNTIME_ENTRY(cl_int, clEnqueueReleaseD3D10ObjectsKHR,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clEnqueueReleaseExtObjectsAMD(command_queue, num_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event,
                                            CL_COMMAND_RELEASE_D3D10_OBJECTS_KHR);
}
RUNTIME_EXIT


//
//
//          namespace amd
//
//
namespace amd {
/*! @}
 *  \addtogroup CL-D3D10 interop helper functions
 *  @{
 */


//*******************************************************************
//
// Internal implementation of CL API functions
//
//*******************************************************************
//
//      clCreateBufferFromD3D10ResourceAMD
//
cl_mem clCreateBufferFromD3D10ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                          ID3D10Resource* pD3DResource, int* errcode_ret) {
  // Verify pD3DResource is a buffer
  D3D10_RESOURCE_DIMENSION rType;
  pD3DResource->GetType(&rType);
  if (rType != D3D10_RESOURCE_DIMENSION_BUFFER) {
    *not_null(errcode_ret) = CL_INVALID_D3D10_RESOURCE_KHR;
    return (cl_mem)0;
  }

  D3D10Object obj;
  int errcode = D3D10Object::initD3D10Object(amdContext, pD3DResource, 0, obj);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  BufferD3D10* pBufferD3D10 = new (amdContext) BufferD3D10(amdContext, flags, obj);
  if (!pBufferD3D10) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pBufferD3D10->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pBufferD3D10->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<Memory>(pBufferD3D10);
}
#if 0
// There is no support for 1D images in the base imagee code
//
//      clCreateImage1DFromD3D10ResourceAMD
//
cl_mem clCreateImage1DFromD3D10ResourceAMD(
    Context& amdContext,
    cl_mem_flags flags,
    ID3D10Resource* pD3DResource,
    UINT subresource,
    int* errcode_ret)
{

    // Verify the resource is a 1D texture
    D3D10_RESOURCE_DIMENSION rType;
    pD3DResource->GetType(&rType);
    if(rType != D3D10_RESOURCE_DIMENSION_TEXTURE1D) {
        *not_null(errcode_ret) = CL_INVALID_D3D10_RESOURCE_KHR;
        return (cl_mem) 0;
    }

    D3D10Object obj;
    int errcode = D3D10Object::initD3D10Object(pD3DResource, subresource, obj);
    if(CL_SUCCESS != errcode)
    {
        *not_null(errcode_ret) = errcode;
        return (cl_mem) 0;
    }

    Image1DD3D10 *pImage1DD3D10 = new Image1DD3D10(amdContext, flags, obj);
    if(!pImage1DD3D10) {
        *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
        return (cl_mem) 0;
    }
    if (!pImage1DD3D10->create()) {
        *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
        pImage1DD3D10->release();
        return (cl_mem) 0;
    }

    *not_null(errcode_ret) = CL_SUCCESS;
    return as_cl<Memory>(pImage1DD3D10);
}
#endif

//
//      clCreateImage2DFromD3D10ResourceAMD
//
cl_mem clCreateImage2DFromD3D10ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                           ID3D10Resource* pD3DResource, UINT subresource,
                                           int* errcode_ret) {
  // Verify the resource is a 2D texture
  D3D10_RESOURCE_DIMENSION rType;
  pD3DResource->GetType(&rType);
  if (rType != D3D10_RESOURCE_DIMENSION_TEXTURE2D) {
    *not_null(errcode_ret) = CL_INVALID_D3D10_RESOURCE_KHR;
    return (cl_mem)0;
  }

  D3D10Object obj;
  int errcode = D3D10Object::initD3D10Object(amdContext, pD3DResource, subresource, obj);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  Image2DD3D10* pImage2DD3D10 = new (amdContext) Image2DD3D10(amdContext, flags, obj);
  if (!pImage2DD3D10) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pImage2DD3D10->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pImage2DD3D10->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<Memory>(pImage2DD3D10);
}

//
//      clCreateImage2DFromD3D10ResourceAMD
//
cl_mem clCreateImage3DFromD3D10ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                           ID3D10Resource* pD3DResource, UINT subresource,
                                           int* errcode_ret) {
  // Verify the resource is a 2D texture
  D3D10_RESOURCE_DIMENSION rType;
  pD3DResource->GetType(&rType);
  if (rType != D3D10_RESOURCE_DIMENSION_TEXTURE3D) {
    *not_null(errcode_ret) = CL_INVALID_D3D10_RESOURCE_KHR;
    return (cl_mem)0;
  }

  D3D10Object obj;
  int errcode = D3D10Object::initD3D10Object(amdContext, pD3DResource, subresource, obj);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  Image3DD3D10* pImage3DD3D10 = new (amdContext) Image3DD3D10(amdContext, flags, obj);
  if (!pImage3DD3D10) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pImage3DD3D10->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pImage3DD3D10->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<Memory>(pImage3DD3D10);
}

//
// Helper function SyncD3D10Objects
//
void SyncD3D10Objects(std::vector<amd::Memory*>& memObjects) {
  Memory*& mem = memObjects.front();
  if (!mem) {
    LogWarning("\nNULL memory object\n");
    return;
  }
  InteropObject* interop = mem->getInteropObj();
  if (!interop) {
    LogWarning("\nNULL interop object\n");
    return;
  }
  D3D10Object* d3d10Obj = interop->asD3D10Object();
  if (!d3d10Obj) {
    LogWarning("\nNULL D3D10 object\n");
    return;
  }
  ID3D10Query* query = d3d10Obj->getQuery();
  if (!query) {
    LogWarning("\nNULL ID3D10Query\n");
    return;
  }
  query->End();
  BOOL data = FALSE;
  while (S_OK != query->GetData(&data, sizeof(BOOL), 0)) {
  }
}

//
// Class D3D10Object implementation
//
size_t D3D10Object::getElementBytes(DXGI_FORMAT dxgiFmt) {
  size_t bytesPerPixel;

  switch (dxgiFmt) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
      bytesPerPixel = 16;
      break;

    case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
      bytesPerPixel = 12;
      break;

    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
      bytesPerPixel = 8;
      break;

    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT:

    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
    case DXGI_FORMAT_R8G8_B8G8_UNORM:
    case DXGI_FORMAT_G8R8_G8B8_UNORM:

    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
      bytesPerPixel = 4;
      break;

    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_SINT:

    case DXGI_FORMAT_B5G6R5_UNORM:
    case DXGI_FORMAT_B5G5R5A1_UNORM:
      bytesPerPixel = 2;
      break;

    case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_A8_UNORM:
    case DXGI_FORMAT_R1_UNORM:
      bytesPerPixel = 1;
      break;


    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
      // Less than 1 byte per pixel - needs special consideration
      bytesPerPixel = 0;
      break;

    default:
      bytesPerPixel = 0;
      _ASSERT(FALSE);
      break;
  }
  return bytesPerPixel;
}

cl_image_format D3D10Object::getCLFormatFromDXGI(DXGI_FORMAT dxgiFmt) {
  cl_image_format fmt;

  //! @todo [odintsov]: add real fmt conversion from DXGI to CL
  fmt.image_channel_order = 0;      // CL_RGBA;
  fmt.image_channel_data_type = 0;  // CL_UNSIGNED_INT8;

  switch (dxgiFmt) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
      fmt.image_channel_order = CL_RGBA;
      break;

    case DXGI_FORMAT_R32G32B32A32_FLOAT:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_FLOAT;
      break;

    case DXGI_FORMAT_R32G32B32A32_UINT:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNSIGNED_INT32;
      break;

    case DXGI_FORMAT_R32G32B32A32_SINT:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_SIGNED_INT32;
      break;

    case DXGI_FORMAT_R32G32B32_TYPELESS:
      fmt.image_channel_order = CL_RGB;
      break;

    case DXGI_FORMAT_R32G32B32_FLOAT:
      fmt.image_channel_order = CL_RGB;
      fmt.image_channel_data_type = CL_FLOAT;
      break;

    case DXGI_FORMAT_R32G32B32_UINT:
      fmt.image_channel_order = CL_RGB;
      fmt.image_channel_data_type = CL_UNSIGNED_INT32;
      break;

    case DXGI_FORMAT_R32G32B32_SINT:
      fmt.image_channel_order = CL_RGB;
      fmt.image_channel_data_type = CL_SIGNED_INT32;
      break;

    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
      fmt.image_channel_order = CL_RGBA;
      break;

    case DXGI_FORMAT_R16G16B16A16_FLOAT:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_HALF_FLOAT;
      break;

    case DXGI_FORMAT_R16G16B16A16_UNORM:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNORM_INT16;
      break;

    case DXGI_FORMAT_R16G16B16A16_UINT:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNSIGNED_INT16;
      break;

    case DXGI_FORMAT_R16G16B16A16_SNORM:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_SNORM_INT16;
      break;

    case DXGI_FORMAT_R16G16B16A16_SINT:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_SIGNED_INT16;
      break;

    case DXGI_FORMAT_R32G32_TYPELESS:
      fmt.image_channel_order = CL_RG;
      break;

    case DXGI_FORMAT_R32G32_FLOAT:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_FLOAT;
      break;

    case DXGI_FORMAT_R32G32_UINT:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_UNSIGNED_INT32;
      break;

    case DXGI_FORMAT_R32G32_SINT:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_SIGNED_INT32;
      break;

    case DXGI_FORMAT_R32G8X24_TYPELESS:
      break;

    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
      break;

    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
      break;

    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
      break;

    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
      fmt.image_channel_order = CL_RGBA;
      break;

    case DXGI_FORMAT_R10G10B10A2_UNORM:
      fmt.image_channel_order = CL_RGBA;
      break;

    case DXGI_FORMAT_R10G10B10A2_UINT:
      fmt.image_channel_order = CL_RGBA;
      break;

    case DXGI_FORMAT_R11G11B10_FLOAT:
      fmt.image_channel_order = CL_RGB;
      break;

    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
      fmt.image_channel_order = CL_RGBA;
      break;

    case DXGI_FORMAT_R8G8B8A8_UNORM:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case DXGI_FORMAT_R8G8B8A8_UINT:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNSIGNED_INT8;
      break;

    case DXGI_FORMAT_R8G8B8A8_SNORM:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_SNORM_INT8;
      break;

    case DXGI_FORMAT_R8G8B8A8_SINT:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_SIGNED_INT8;
      break;

    case DXGI_FORMAT_R16G16_TYPELESS:
      fmt.image_channel_order = CL_RG;
      break;

    case DXGI_FORMAT_R16G16_FLOAT:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_HALF_FLOAT;
      break;

    case DXGI_FORMAT_R16G16_UNORM:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_UNORM_INT16;
      break;

    case DXGI_FORMAT_R16G16_UINT:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_UNSIGNED_INT16;
      break;

    case DXGI_FORMAT_R16G16_SNORM:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_SNORM_INT16;
      break;

    case DXGI_FORMAT_R16G16_SINT:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_SIGNED_INT16;
      break;

    case DXGI_FORMAT_R32_TYPELESS:
      fmt.image_channel_order = CL_R;
      break;

    case DXGI_FORMAT_D32_FLOAT:
      break;

    case DXGI_FORMAT_R32_FLOAT:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_FLOAT;
      break;

    case DXGI_FORMAT_R32_UINT:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_UNSIGNED_INT32;
      break;

    case DXGI_FORMAT_R32_SINT:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_SIGNED_INT32;
      break;

    case DXGI_FORMAT_R24G8_TYPELESS:
      fmt.image_channel_order = CL_RG;
      break;

    case DXGI_FORMAT_D24_UNORM_S8_UINT:
      break;

    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
      break;

    case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
      break;

    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
      break;

    case DXGI_FORMAT_R8G8_B8G8_UNORM:
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case DXGI_FORMAT_G8R8_G8B8_UNORM:
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case DXGI_FORMAT_B8G8R8A8_UNORM:
      fmt.image_channel_order = CL_BGRA;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case DXGI_FORMAT_B8G8R8X8_UNORM:
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case DXGI_FORMAT_R8G8_TYPELESS:
      fmt.image_channel_order = CL_RG;
      break;

    case DXGI_FORMAT_R8G8_UNORM:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case DXGI_FORMAT_R8G8_UINT:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_UNSIGNED_INT8;
      break;

    case DXGI_FORMAT_R8G8_SNORM:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_SNORM_INT8;
      break;

    case DXGI_FORMAT_R8G8_SINT:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_SIGNED_INT8;
      break;

    case DXGI_FORMAT_R16_TYPELESS:
      fmt.image_channel_order = CL_R;
      break;

    case DXGI_FORMAT_R16_FLOAT:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_HALF_FLOAT;
      break;

    case DXGI_FORMAT_D16_UNORM:
      fmt.image_channel_data_type = CL_UNORM_INT16;
      break;

    case DXGI_FORMAT_R16_UNORM:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_UNORM_INT16;
      break;

    case DXGI_FORMAT_R16_UINT:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_UNSIGNED_INT16;
      break;

    case DXGI_FORMAT_R16_SNORM:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_SNORM_INT16;
      break;

    case DXGI_FORMAT_R16_SINT:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_SIGNED_INT16;
      break;

    case DXGI_FORMAT_B5G6R5_UNORM:
      fmt.image_channel_data_type = CL_UNORM_SHORT_565;
      break;

    case DXGI_FORMAT_B5G5R5A1_UNORM:
      fmt.image_channel_order = CL_BGRA;
      break;

    case DXGI_FORMAT_R8_TYPELESS:
      fmt.image_channel_order = CL_R;
      break;

    case DXGI_FORMAT_R8_UNORM:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case DXGI_FORMAT_R8_UINT:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_UNSIGNED_INT8;
      break;

    case DXGI_FORMAT_R8_SNORM:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_SNORM_INT8;
      break;

    case DXGI_FORMAT_R8_SINT:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_SIGNED_INT8;
      break;

    case DXGI_FORMAT_A8_UNORM:
      fmt.image_channel_order = CL_A;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case DXGI_FORMAT_R1_UNORM:
      fmt.image_channel_order = CL_R;
      break;

    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
      break;

    default:
      _ASSERT(FALSE);
      break;
  }

  return fmt;
}

size_t D3D10Object::getResourceByteSize() {
  size_t bytes = 1;

  //! @todo [odintsov]: take into consideration the mip level?!

  switch (objDesc_.objDim_) {
    case D3D10_RESOURCE_DIMENSION_BUFFER:
      bytes = objDesc_.objSize_.ByteWidth;
      break;

    case D3D10_RESOURCE_DIMENSION_TEXTURE3D:
      bytes = objDesc_.objSize_.Depth;

    case D3D10_RESOURCE_DIMENSION_TEXTURE2D:
      bytes *= objDesc_.objSize_.Height;

    case D3D10_RESOURCE_DIMENSION_TEXTURE1D:
      bytes *= objDesc_.objSize_.Width * getElementBytes();
      break;

    default:
      LogError("getResourceByteSize: unknown type of D3D10 resource");
      bytes = 0;
      break;
  }
  return bytes;
}

int D3D10Object::initD3D10Object(const Context& amdContext, ID3D10Resource* pRes, UINT subres,
                                 D3D10Object& obj) {
  ID3D10Device* pDev;
  HRESULT hr;
  ScopedLock sl(resLock_);

  // Check if this ressource has already been used for interop
  for (const auto& it : resources_) {
    if (it.first == (void*)pRes && it.second == subres) {
      return CL_INVALID_D3D10_RESOURCE_KHR;
    }
  }

  (obj.pD3D10Res_ = pRes)->GetDevice(&pDev);

  if (!pDev) {
    return CL_INVALID_D3D10_DEVICE_KHR;
  }

  D3D10_QUERY_DESC desc = {D3D10_QUERY_EVENT, 0};
  pDev->CreateQuery(&desc, &obj.pQuery_);

#define SET_SHARED_FLAGS()                                                                         \
  {                                                                                                \
    obj.pD3D10ResOrig_ = obj.pD3D10Res_;                                                           \
    memcpy(&obj.objDescOrig_, &obj.objDesc_, sizeof(D3D10ObjDesc_t));                              \
    /* @todo - Check device type and select right usage for resource */                            \
    /* For now get only DPU path, CPU path for buffers */                                          \
    /* will not worl on DEFAUL resources */                                                        \
    /*desc.Usage = D3D10_USAGE_STAGING;*/                                                          \
    desc.Usage = D3D10_USAGE_DEFAULT;                                                              \
    desc.MiscFlags = D3D10_RESOURCE_MISC_SHARED;                                                   \
    desc.CPUAccessFlags = 0;                                                                       \
  }

#define STORE_SHARED_FLAGS(restype)                                                                \
  {                                                                                                \
    if (S_OK == hr && obj.pD3D10Res_) {                                                            \
      obj.objDesc_.objFlags_.d3d10Usage_ = desc.Usage;                                             \
      obj.objDesc_.objFlags_.bindFlags_ = desc.BindFlags;                                          \
      obj.objDesc_.objFlags_.miscFlags_ = desc.MiscFlags;                                          \
      obj.objDesc_.objFlags_.cpuAccessFlags_ = desc.CPUAccessFlags;                                \
    } else {                                                                                       \
      LogError("\nCannot create shared " #restype "\n");                                           \
      return CL_INVALID_D3D10_RESOURCE_KHR;                                                        \
    }                                                                                              \
  }

#define SET_BINDING()                                                                              \
  {                                                                                                \
    switch (desc.Format) {                                                                         \
      case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:                                                       \
      case DXGI_FORMAT_D32_FLOAT:                                                                  \
      case DXGI_FORMAT_D24_UNORM_S8_UINT:                                                          \
      case DXGI_FORMAT_D16_UNORM:                                                                  \
        desc.BindFlags = D3D10_BIND_DEPTH_STENCIL;                                                 \
        break;                                                                                     \
      default:                                                                                     \
        desc.BindFlags = D3D10_BIND_SHADER_RESOURCE | D3D10_BIND_RENDER_TARGET;                    \
        break;                                                                                     \
    }                                                                                              \
  }

  pRes->GetType(&obj.objDesc_.objDim_);

  // Init defaults
  obj.objDesc_.objSize_.Height = 1;
  obj.objDesc_.objSize_.Depth = 1;
  obj.objDesc_.mipLevels_ = 1;
  obj.objDesc_.arraySize_ = 1;
  obj.objDesc_.dxgiFormat_ = DXGI_FORMAT_UNKNOWN;
  obj.objDesc_.dxgiSampleDesc_ = dxgiSampleDescDefault;

  switch (obj.objDesc_.objDim_) {
    case D3D10_RESOURCE_DIMENSION_BUFFER:  // = 1,
    {
      D3D10_BUFFER_DESC desc;
      (reinterpret_cast<ID3D10Buffer*>(pRes))->GetDesc(&desc);
      obj.objDesc_.objSize_.ByteWidth = desc.ByteWidth;
      obj.objDesc_.objFlags_.d3d10Usage_ = desc.Usage;
      obj.objDesc_.objFlags_.bindFlags_ = desc.BindFlags;
      obj.objDesc_.objFlags_.cpuAccessFlags_ = desc.CPUAccessFlags;
      obj.objDesc_.objFlags_.miscFlags_ = desc.MiscFlags;
      // Handle D3D10Buffer without shared handle - create
      //  a duplicate with shared handle to provide for CAL
      if (!(obj.objDesc_.objFlags_.miscFlags_ & D3D10_RESOURCE_MISC_SHARED)) {
        SET_SHARED_FLAGS();
        desc.BindFlags = D3D10_BIND_SHADER_RESOURCE | D3D10_BIND_RENDER_TARGET;
        hr = pDev->CreateBuffer(&desc, NULL, (ID3D10Buffer**)&obj.pD3D10Res_);
        STORE_SHARED_FLAGS(ID3D10Buffer);
      }
    } break;

    case D3D10_RESOURCE_DIMENSION_TEXTURE1D:  // = 2,
    {
      D3D10_TEXTURE1D_DESC desc;
      (reinterpret_cast<ID3D10Texture1D*>(pRes))->GetDesc(&desc);

      if (subres) {
        // Calculate correct size of the subresource
        UINT miplevel = subres;
        if (desc.ArraySize > 1) {
          miplevel = subres % desc.ArraySize;
        }
        if (miplevel >= desc.MipLevels) {
          LogWarning("\nMiplevel >= number of miplevels\n");
        }
        if (subres >= desc.MipLevels * desc.ArraySize) {
          return CL_INVALID_VALUE;
        }
        desc.Width >>= miplevel;
        if (!desc.Width) {
          desc.Width = 1;
        }
      }
      obj.objDesc_.objSize_.Width = desc.Width;
      obj.objDesc_.mipLevels_ = desc.MipLevels;
      obj.objDesc_.arraySize_ = desc.ArraySize;
      obj.objDesc_.dxgiFormat_ = desc.Format;
      obj.objDesc_.objFlags_.d3d10Usage_ = desc.Usage;
      obj.objDesc_.objFlags_.bindFlags_ = desc.BindFlags;
      obj.objDesc_.objFlags_.cpuAccessFlags_ = desc.CPUAccessFlags;
      obj.objDesc_.objFlags_.miscFlags_ = desc.MiscFlags;
      // Handle D3D10Texture1D without shared handle - create
      //  a duplicate with shared handle and provide it for CAL
      // Workaround for subresource > 0 in shared resource
      if (subres) obj.objDesc_.objFlags_.miscFlags_ &= ~(D3D10_RESOURCE_MISC_SHARED);
      if (!(obj.objDesc_.objFlags_.miscFlags_ & D3D10_RESOURCE_MISC_SHARED)) {
        SET_SHARED_FLAGS();
        SET_BINDING();
        obj.objDesc_.mipLevels_ = desc.MipLevels = 1;
        obj.objDesc_.arraySize_ = desc.ArraySize = 1;
        hr = pDev->CreateTexture1D(&desc, NULL, (ID3D10Texture1D**)&obj.pD3D10Res_);
        STORE_SHARED_FLAGS(ID3D10Texture1D);
      }
    } break;

    case D3D10_RESOURCE_DIMENSION_TEXTURE2D:  // = 3,
    {
      D3D10_TEXTURE2D_DESC desc;
      (reinterpret_cast<ID3D10Texture2D*>(pRes))->GetDesc(&desc);

      if (subres) {
        // Calculate correct size of the subresource
        UINT miplevel = subres;
        if (desc.ArraySize > 1) {
          miplevel = subres % desc.MipLevels;
        }
        if (miplevel >= desc.MipLevels) {
          LogWarning("\nMiplevel >= number of miplevels\n");
        }
        if (subres >= desc.MipLevels * desc.ArraySize) {
          return CL_INVALID_VALUE;
        }
        desc.Width >>= miplevel;
        if (!desc.Width) {
          desc.Width = 1;
        }
        desc.Height >>= miplevel;
        if (!desc.Height) {
          desc.Height = 1;
        }
      }
      obj.objDesc_.objSize_.Width = desc.Width;
      obj.objDesc_.objSize_.Height = desc.Height;
      obj.objDesc_.mipLevels_ = desc.MipLevels;
      obj.objDesc_.arraySize_ = desc.ArraySize;
      obj.objDesc_.dxgiFormat_ = desc.Format;
      obj.objDesc_.dxgiSampleDesc_ = desc.SampleDesc;
      obj.objDesc_.objFlags_.d3d10Usage_ = desc.Usage;
      obj.objDesc_.objFlags_.bindFlags_ = desc.BindFlags;
      obj.objDesc_.objFlags_.cpuAccessFlags_ = desc.CPUAccessFlags;
      obj.objDesc_.objFlags_.miscFlags_ = desc.MiscFlags;
      // Handle D3D10Texture2D without shared handle - create
      //  a duplicate with shared handle and provide it for CAL
      // Workaround for subresource > 0 in shared resource
      if (subres) obj.objDesc_.objFlags_.miscFlags_ &= ~(D3D10_RESOURCE_MISC_SHARED);
      if (!(obj.objDesc_.objFlags_.miscFlags_ & D3D10_RESOURCE_MISC_SHARED)) {
        SET_SHARED_FLAGS();
        SET_BINDING();
        obj.objDesc_.mipLevels_ = desc.MipLevels = 1;
        obj.objDesc_.arraySize_ = desc.ArraySize = 1;
        hr = pDev->CreateTexture2D(&desc, NULL, (ID3D10Texture2D**)&obj.pD3D10Res_);
        STORE_SHARED_FLAGS(ID3D10Texture2D);
      }
    } break;

    case D3D10_RESOURCE_DIMENSION_TEXTURE3D:  // = 4
    {
      D3D10_TEXTURE3D_DESC desc;
      (reinterpret_cast<ID3D10Texture3D*>(pRes))->GetDesc(&desc);

      if (subres) {
        // Calculate correct size of the subresource
        UINT miplevel = subres;
        if (miplevel >= desc.MipLevels) {
          LogWarning("\nMiplevel >= number of miplevels\n");
        }
        if (subres >= desc.MipLevels) {
          return CL_INVALID_VALUE;
        }
        desc.Width >>= miplevel;
        if (!desc.Width) {
          desc.Width = 1;
        }
        desc.Height >>= miplevel;
        if (!desc.Height) {
          desc.Height = 1;
        }
        desc.Depth >>= miplevel;
        if (!desc.Depth) {
          desc.Depth = 1;
        }
      }
      obj.objDesc_.objSize_.Width = desc.Width;
      obj.objDesc_.objSize_.Height = desc.Height;
      obj.objDesc_.objSize_.Depth = desc.Depth;
      obj.objDesc_.mipLevels_ = desc.MipLevels;
      obj.objDesc_.dxgiFormat_ = desc.Format;
      obj.objDesc_.objFlags_.d3d10Usage_ = desc.Usage;
      obj.objDesc_.objFlags_.bindFlags_ = desc.BindFlags;
      obj.objDesc_.objFlags_.cpuAccessFlags_ = desc.CPUAccessFlags;
      obj.objDesc_.objFlags_.miscFlags_ = desc.MiscFlags;
      // Handle D3D10Texture3D without shared handle - create
      //  a duplicate with shared handle and provide it for CAL
      // Workaround for subresource > 0 in shared resource
      if (obj.objDesc_.mipLevels_ > 1)
        obj.objDesc_.objFlags_.miscFlags_ &= ~(D3D10_RESOURCE_MISC_SHARED);
      if (!(obj.objDesc_.objFlags_.miscFlags_ & D3D10_RESOURCE_MISC_SHARED)) {
        SET_SHARED_FLAGS();
        SET_BINDING();
        obj.objDesc_.mipLevels_ = desc.MipLevels = 1;
        hr = pDev->CreateTexture3D(&desc, NULL, (ID3D10Texture3D**)&obj.pD3D10Res_);
        STORE_SHARED_FLAGS(ID3D10Texture3D);
      }
    } break;

    default:
      LogError("unknown type of D3D10 resource");
      return CL_INVALID_D3D10_RESOURCE_KHR;
  }
  obj.subRes_ = subres;
  pDev->Release();
  // Check for CL format compatibilty
  if (obj.objDesc_.objDim_ != D3D10_RESOURCE_DIMENSION_BUFFER) {
    cl_image_format clFmt = obj.getCLFormatFromDXGI(obj.objDesc_.dxgiFormat_);
    amd::Image::Format imageFormat(clFmt);
    if (!imageFormat.isSupported(amdContext)) {
      return CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    }
  }
  resources_.push_back({pRes, subres});
  return CL_SUCCESS;
}

bool D3D10Object::copyOrigToShared() {
  // Don't copy if there is no orig
  if (NULL == getD3D10ResOrig()) return true;

  ID3D10Device* d3dDev;
  pD3D10Res_->GetDevice(&d3dDev);
  if (!d3dDev) {
    LogError("\nCannot get D3D10 device from D3D10 resource\n");
    return false;
  }
  // Any usage source can be read by GPU
  d3dDev->CopySubresourceRegion(pD3D10Res_, 0, 0, 0, 0, pD3D10ResOrig_, subRes_, NULL);

  // Flush D3D queues and make sure D3D stuff is finished
  d3dDev->Flush();
  pQuery_->End();
  BOOL data = FALSE;
  while ((S_OK != pQuery_->GetData(&data, sizeof(BOOL), 0)) || (data != TRUE)) {
  }

  d3dDev->Release();
  return true;
}

bool D3D10Object::copySharedToOrig() {
  // Don't copy if there is no orig
  if (NULL == getD3D10ResOrig()) return true;

  ID3D10Device* d3dDev;
  pD3D10Res_->GetDevice(&d3dDev);
  if (!d3dDev) {
    LogError("\nCannot get D3D10 device from D3D10 resource\n");
    return false;
  }

  d3dDev->CopySubresourceRegion(pD3D10ResOrig_, subRes_, 0, 0, 0, pD3D10Res_, 0, NULL);

  d3dDev->Release();
  return true;
}

std::vector<std::pair<void*, UINT>> D3D10Object::resources_;
Monitor D3D10Object::resLock_;

//
// Class BufferD3D10 implementation
//
void BufferD3D10::initDeviceMemory() {
  deviceMemories_ =
      reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(BufferD3D10));
  memset(deviceMemories_, 0, context_().devices().size() * sizeof(DeviceMemory));
}

//
// Class Image1DD3D10 implementation
//

void Image1DD3D10::initDeviceMemory() {
  deviceMemories_ =
      reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(Image1DD3D10));
  memset(deviceMemories_, 0, context_().devices().size() * sizeof(DeviceMemory));
}

//
// Class Image2DD3D10 implementation
//

void Image2DD3D10::initDeviceMemory() {
  deviceMemories_ =
      reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(Image2DD3D10));
  memset(deviceMemories_, 0, context_().devices().size() * sizeof(DeviceMemory));
}

//
// Class Image3DD3D10 implementation
//
void Image3DD3D10::initDeviceMemory() {
  deviceMemories_ =
      reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(Image3DD3D10));
  memset(deviceMemories_, 0, context_().devices().size() * sizeof(DeviceMemory));
}

}  // namespace amd

#endif  //_WIN32
