/* Copyright (c) 2012-present Advanced Micro Devices, Inc.

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

#include "cl_d3d9_amd.hpp"
#include "platform/command.hpp"

#include <cstring>
#include <utility>

#define D3DFMT_NV_12 static_cast<D3DFORMAT>(MAKEFOURCC('N', 'V', '1', '2'))
#define D3DFMT_P010  static_cast<D3DFORMAT>(MAKEFOURCC('P', '0', '1', '0'))
#define D3DFMT_YV_12 static_cast<D3DFORMAT>(MAKEFOURCC('Y', 'V', '1', '2'))
#define D3DFMT_YUY2  static_cast<D3DFORMAT>(MAKEFOURCC('Y', 'U', 'Y', '2'))


RUNTIME_ENTRY(cl_int, clGetDeviceIDsFromDX9MediaAdapterKHR,
              (cl_platform_id platform, cl_uint num_media_adapters,
               cl_dx9_media_adapter_type_khr* media_adapters_type, void* media_adapters,
               cl_dx9_media_adapter_set_khr media_adapter_set, cl_uint num_entries,
               cl_device_id* devices, cl_uint* num_devices)) {
  cl_int errcode;
  // Accept an array of DX9 devices here as the spec mention of array of num_media_adapters size.
  IDirect3DDevice9Ex** d3d9_device = static_cast<IDirect3DDevice9Ex**>(media_adapters);
  cl_device_id* gpu_devices = NULL;
  cl_uint num_gpu_devices = 0;
  static const bool VALIDATE_ONLY = true;

  if (platform != NULL && platform != AMD_PLATFORM) {
    LogWarning("\"platrform\" is not a valid AMD platform");
    return CL_INVALID_PLATFORM;
  }
  // check if input parameter are correct
  if ((num_media_adapters == 0) || (media_adapters_type == NULL) || (media_adapters == NULL) ||
      (media_adapter_set != CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR &&
       media_adapter_set != CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR) ||
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

  switch (media_adapter_set) {
    case CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR:
    case CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR: {
      gpu_devices = new cl_device_id[num_gpu_devices];
      errcode = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, num_gpu_devices, gpu_devices, NULL);
      if (errcode != CL_SUCCESS) {
        break;
      }

      std::vector<amd::Device*> compatible_devices;
      for (cl_uint i = 0; i < num_gpu_devices; ++i) {
        cl_device_id device = gpu_devices[i];
        amd::Context::Flags context_flag;
        amd::Context::DeviceFlagIdx devIdx;
        switch (media_adapters_type[i]) {
          case CL_ADAPTER_D3D9_KHR:
            context_flag = amd::Context::Flags::D3D9DeviceKhr;
            devIdx = amd::Context::DeviceFlagIdx::D3D9DeviceKhrIdx;
            break;
          case CL_ADAPTER_D3D9EX_KHR:
            context_flag = amd::Context::Flags::D3D9DeviceEXKhr;
            devIdx = amd::Context::DeviceFlagIdx::D3D9DeviceEXKhrIdx;
            break;
          case CL_ADAPTER_DXVA_KHR:
            context_flag = amd::Context::Flags::D3D9DeviceVAKhr;
            devIdx = amd::Context::DeviceFlagIdx::D3D9DeviceVAKhrIdx;
            break;
        }

        for (cl_uint j = 0; j < num_media_adapters; ++j) {
          // Since there can be multiple DX9 adapters passed in the array we need to validate
          // interopability with each.
          void* external_device[amd::Context::DeviceFlagIdx::LastDeviceFlagIdx] = {};
          external_device[devIdx] = d3d9_device[j];

          if (is_valid(device) && (media_adapters_type[j] == CL_ADAPTER_D3D9EX_KHR) &&
              as_amd(device)->bindExternalDevice(context_flag, external_device, NULL,
                                                 VALIDATE_ONLY)) {
            compatible_devices.push_back(as_amd(device));
          }
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
      LogWarning("\"d3d9_device_set\" is invalid");
      errcode = CL_INVALID_VALUE;
  }

  delete[] gpu_devices;
  return errcode;
}
RUNTIME_EXIT

RUNTIME_ENTRY_RET(cl_mem, clCreateFromDX9MediaSurfaceKHR,
                  (cl_context context, cl_mem_flags flags,
                   cl_dx9_media_adapter_type_khr adapter_type, void* surface_info, cl_uint plane,
                   cl_int* errcode_ret)) {
  cl_mem clMemObj = NULL;

  cl_dx9_surface_info_khr* cl_surf_info = NULL;

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

  if ((adapter_type != CL_ADAPTER_D3D9_KHR) && (adapter_type != CL_ADAPTER_D3D9EX_KHR) &&
      (adapter_type != CL_ADAPTER_DXVA_KHR)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return clMemObj;
  }

  if (!surface_info) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("parameter \"pD3DResource\" is a NULL pointer");
    return clMemObj;
  }

  cl_surf_info = (cl_dx9_surface_info_khr*)surface_info;
  IDirect3DSurface9* pD3D9Resource = cl_surf_info->resource;
  HANDLE shared_handle = cl_surf_info->shared_handle;

  if (!pD3D9Resource) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("parameter \"surface_info\" is a NULL pointer");
    return clMemObj;
  }

  D3DSURFACE_DESC Desc;
  pD3D9Resource->GetDesc(&Desc);

  if ((Desc.Format != D3DFMT_NV_12) &&
      (Desc.Format != D3DFMT_P010) &&
      (Desc.Format != D3DFMT_YV_12) && (plane != 0)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("The plane has to be Zero if the surface format is non-planar !");
    return clMemObj;
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
  // Verify the resource is a 2D image
  return amd::clCreateImage2DFromD3D9ResourceAMD(*as_amd(context), flags, adapter_type,
                                                 cl_surf_info, plane, errcode_ret);
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clEnqueueAcquireDX9MediaSurfacesKHR,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clEnqueueAcquireExtObjectsAMD(command_queue, num_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event,
                                            CL_COMMAND_ACQUIRE_DX9_MEDIA_SURFACES_KHR);
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clEnqueueReleaseDX9MediaSurfacesKHR,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clEnqueueReleaseExtObjectsAMD(command_queue, num_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event,
                                            CL_COMMAND_RELEASE_DX9_MEDIA_SURFACES_KHR);
}
RUNTIME_EXIT

//
//
//          namespace amd
//
//
namespace amd {
/*! @}
 *  \addtogroup CL-D3D9 interop helper functions
 *  @{
 */
//
// Class D3D9Object implementation
//
std::vector<std::pair<TD3D9RESINFO, TD3D9RESINFO>> D3D9Object::resources_;
Monitor D3D9Object::resLock_;

//
//      clCreateImage2DFromD3D9ResourceAMD
//
cl_mem clCreateImage2DFromD3D9ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                          cl_dx9_media_adapter_type_khr adapter_type,
                                          cl_dx9_surface_info_khr* surface_info, cl_uint plane,
                                          int* errcode_ret) {
  cl_dx9_surface_info_khr* cl_surf_info = reinterpret_cast<cl_dx9_surface_info_khr*>(surface_info);
  IDirect3DSurface9* pD3D9Resource = cl_surf_info->resource;
  HANDLE shared_handle = cl_surf_info->shared_handle;

  D3D9Object obj;
  cl_int errcode = D3D9Object::initD3D9Object(amdContext, adapter_type, surface_info, plane, obj);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  Image2DD3D9* pImage2DD3D9 = new (amdContext) Image2DD3D9(amdContext, flags, obj);
  if (!pImage2DD3D9) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pImage2DD3D9->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pImage2DD3D9->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<Memory>(pImage2DD3D9);
}

//
// Helper function SyncD3D9Objects
//
void SyncD3D9Objects(std::vector<amd::Memory*>& memObjects) {
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
  D3D9Object* d3d9Obj = interop->asD3D9Object();
  if (!d3d9Obj) {
    LogWarning("\nNULL D3D9 object\n");
    return;
  }
  IDirect3DQuery9* query = d3d9Obj->getQuery();
  if (!query) {
    LogWarning("\nNULL IDirect3DQuery9\n");
    return;
  }
  ScopedLock sl(d3d9Obj->getResLock());
  query->Issue(D3DISSUE_END);
  BOOL data = FALSE;
  while (S_OK != query->GetData(&data, sizeof(BOOL), D3DGETDATA_FLUSH)) {
  }
}

//
// Class D3D10Object implementation
//
size_t D3D9Object::getElementBytes(D3DFORMAT d3d9Format, cl_uint plane) {
  size_t bytesPerPixel;

  switch (d3d9Format) {
    case D3DFMT_UNKNOWN:
    case D3DFMT_UYVY:
    case D3DFMT_DXT1:
    case D3DFMT_DXT2:
    case D3DFMT_DXT3:
    case D3DFMT_DXT4:
    case D3DFMT_DXT5:
    case D3DFMT_VERTEXDATA:
    case D3DFMT_D32:
    case D3DFMT_D15S1:
    case D3DFMT_D24S8:
    case D3DFMT_D24X8:
    case D3DFMT_D24X4S4:
    case D3DFMT_D16:
    case D3DFMT_INDEX16:
    case D3DFMT_INDEX32:
    case D3DFMT_MULTI2_ARGB8:
    case D3DFMT_CxV8U8:
      // Less than 1 byte per pixel - needs special consideration
      bytesPerPixel = 0;
      break;

    case D3DFMT_R3G3B2:
    case D3DFMT_P8:
    case D3DFMT_A8:
    case D3DFMT_L8:
    case D3DFMT_A4L4:
      bytesPerPixel = 1;
      break;

    case D3DFMT_R16F:
    case D3DFMT_R5G6B5:
    case D3DFMT_X1R5G5B5:
    case D3DFMT_A1R5G5B5:
    case D3DFMT_A4R4G4B4:
    case D3DFMT_A8R3G3B2:
    case D3DFMT_X4R4G4B4:
    case D3DFMT_A8P8:
    case D3DFMT_A8L8:
    case D3DFMT_V8U8:
    case D3DFMT_L6V5U5:
    case D3DFMT_D16_LOCKABLE:
    case D3DFMT_L16:
      bytesPerPixel = 2;
      break;

    case D3DFMT_R8G8B8:
    case D3DFMT_D24FS8:
      bytesPerPixel = 3;
      break;

    case D3DFMT_D32F_LOCKABLE:
    case D3DFMT_A8R8G8B8:
    case D3DFMT_R32F:
    case D3DFMT_X8R8G8B8:
    case D3DFMT_A2B10G10R10:
    case D3DFMT_A8B8G8R8:
    case D3DFMT_X8B8G8R8:
    case D3DFMT_G16R16:
    case D3DFMT_A2R10G10B10:
    case D3DFMT_Q8W8V8U8:
    case D3DFMT_X8L8V8U8:
    case D3DFMT_V16U16:
    case D3DFMT_A2W10V10U10:
    case D3DFMT_R8G8_B8G8:
    case D3DFMT_G8R8_G8B8:
    case D3DFMT_G16R16F:
    case D3DFMT_YUY2:
      bytesPerPixel = 4;
      break;

    case D3DFMT_G32R32F:
    case D3DFMT_A16B16G16R16:
    case D3DFMT_A16B16G16R16F:
    case D3DFMT_Q16W16V16U16:
      bytesPerPixel = 8;
      break;
    case D3DFMT_A32B32G32R32F:
      bytesPerPixel = 16;
      break;
    //#if !defined(D3D_DISABLE_9EX)
    // case D3DFMT_D32_LOCKABLE:
    // case D3DFMT_S8_LOCKABLE:
    //#endif // !D3D_DISABLE_9EX
    case D3DFMT_NV_12:
      if (plane == 0) {
        bytesPerPixel = 1;
      } else if (plane == 1) {
        bytesPerPixel = 2;
      }  // plane != 0 or != 1 shouldn't happen here
      break;
    case D3DFMT_P010:
      if (plane == 0) {
        bytesPerPixel = 2;
      } else if (plane == 1) {
        bytesPerPixel = 4;
      }  // plane != 0 or != 1 shouldn't happen here
      break;
    case D3DFMT_YV_12:
      bytesPerPixel = 1;
      break;

    default:
      bytesPerPixel = 0;
      _ASSERT(FALSE);
      break;
  }
  return bytesPerPixel;
}

void setObjDesc(amd::D3D9ObjDesc_t& objDesc, D3DSURFACE_DESC& resDesc, cl_uint plane) {
  objDesc.d3dPool_ = resDesc.Pool;
  objDesc.resType_ = resDesc.Type;
  objDesc.usage_ = resDesc.Usage;
  objDesc.d3dFormat_ = resDesc.Format;
  switch (resDesc.Format) {
    case D3DFMT_NV_12:
    case D3DFMT_P010:
      objDesc.surfRect_.left = 0;
      objDesc.surfRect_.top = 0;
      if (plane == 0) {
        objDesc.objSize_.Height = resDesc.Height;
        objDesc.objSize_.Width = resDesc.Width;
        objDesc.surfRect_.right = resDesc.Width;  // resDesc.Width/2-1;
        objDesc.surfRect_.bottom = 3 * resDesc.Height / 2;
        ;  // 3*resDesc.Height/2-1;
      } else if (plane == 1) {
        objDesc.objSize_.Height = resDesc.Height / 2;
        objDesc.objSize_.Width = resDesc.Width / 2;
        objDesc.surfRect_.right = resDesc.Width;  // resDesc.Width/2-1;
        objDesc.surfRect_.bottom = 3 * resDesc.Height / 2;
        ;  // 3*resDesc.Height/2-1;
      }    // plane != 0 or != 1 shouldn't happen here
      break;
    case D3DFMT_YV_12:
      objDesc.surfRect_.left = 0;
      if (plane == 0) {
        objDesc.objSize_.Height = resDesc.Height;
        objDesc.objSize_.Width = resDesc.Width;
        objDesc.surfRect_.top = 0;
        objDesc.surfRect_.right = resDesc.Width - 1;
        objDesc.surfRect_.bottom = resDesc.Height - 1;
      } else if (plane == 1) {
        objDesc.objSize_.Height = resDesc.Height / 2;
        objDesc.objSize_.Width = resDesc.Width / 2;
        objDesc.surfRect_.top = resDesc.Height;
        objDesc.surfRect_.right = resDesc.Width / 2 - 1;
        objDesc.surfRect_.bottom = 3 * resDesc.Height / 2 - 1;
      } else if (plane == 2) {
        objDesc.objSize_.Height = resDesc.Height / 2;
        objDesc.objSize_.Width = resDesc.Width / 2;
        objDesc.surfRect_.top = 3 * resDesc.Height / 2;
        objDesc.surfRect_.right = resDesc.Width / 2 - 1;
        objDesc.surfRect_.bottom = 2 * resDesc.Height - 1;
      }  // plane > 0 or > 2 shouldn't happen here
      break;
    default:
      objDesc.objSize_.Height = resDesc.Height;
      objDesc.objSize_.Width = resDesc.Width;
      objDesc.surfRect_.left = 0;
      objDesc.surfRect_.top = 0;
      objDesc.surfRect_.right = resDesc.Width - 1;
      objDesc.surfRect_.bottom = resDesc.Height - 1;
      if (resDesc.Format == D3DFMT_YUY2) {
        objDesc.objSize_.Width >>= 1;
      }
      break;
  }
}

int D3D9Object::initD3D9Object(const Context& amdContext,
                               cl_dx9_media_adapter_type_khr adapter_type,
                               cl_dx9_surface_info_khr* cl_surf_info, cl_uint plane,
                               D3D9Object& obj) {
  ScopedLock sl(resLock_);

  IDirect3DDevice9Ex* pDev9Ex = NULL;
  cl_int errcode = CL_SUCCESS;

  // Check if this ressource has already been used for interop
  IDirect3DSurface9* pD3D9res = cl_surf_info->resource;
  HANDLE shared_handle = cl_surf_info->shared_handle;

  if ((adapter_type == CL_ADAPTER_D3D9_KHR) || (adapter_type == CL_ADAPTER_DXVA_KHR)) {
    return CL_INVALID_DX9_MEDIA_ADAPTER_KHR;  // Not supported yet
  }

  for (const auto& it : resources_) {
    if (it.first.surfInfo.resource == cl_surf_info->resource && it.first.surfPlane == plane) {
      return CL_INVALID_D3D9_RESOURCE_KHR;
    }
  }

  HRESULT hr;
  D3DQUERYTYPE desc = D3DQUERYTYPE_EVENT;

  D3DSURFACE_DESC resDesc;
  if (D3D_OK != pD3D9res->GetDesc(&resDesc)) {
    return CL_INVALID_D3D9_RESOURCE_KHR;
  }

  hr = pD3D9res->GetContainer(IID_IDirect3DDevice9Ex, (void**)&pDev9Ex);
  if (hr == D3D_OK) {
    pDev9Ex->CreateQuery(desc, &(obj.pQuery_));
  } else {
    return CL_INVALID_D3D9_RESOURCE_KHR;  // d3d9ex should be supported
  }

  obj.handleShared_ = shared_handle;
  obj.surfPlane_ = plane;
  obj.surfInfo_ = *cl_surf_info;
  obj.adapterType_ = adapter_type;

  // Init defaults
  setObjDesc(obj.objDescOrig_, resDesc, plane);
  obj.objDesc_ = obj.objDescOrig_;

  // shared handle cases if the shared_handle is NULL
  // first check if the format is NV12 or YV12, which we need special handling
  if (NULL == shared_handle) {
    bool found = false;
    for (const auto& it : resources_) {
      if (it.first.surfInfo.resource == cl_surf_info->resource &&
          it.first.surfPlane != plane) {
        obj.handleShared_ = it.second.surfInfo.shared_handle;
        obj.pD3D9Res_ = it.second.surfInfo.resource;
        obj.pD3D9Res_->AddRef();
        obj.objDesc_ = obj.objDescOrig_;
        found = true;
        break;
      }
    }
    if (!found) {
      obj.handleShared_ = 0;
      hr = pDev9Ex->CreateOffscreenPlainSurface(resDesc.Width, resDesc.Height, resDesc.Format,
                                                resDesc.Pool, &obj.pD3D9Res_, &obj.handleShared_);

      if (D3D_OK != hr) {
        errcode = CL_INVALID_D3D9_RESOURCE_KHR;
      }
    }

    // put the original info into the obj
    obj.pD3D9ResOrig_ = pD3D9res;
    obj.pD3D9ResOrig_->AddRef();  // addRef in case lost the resource
  } else {
    // Share the original resource
    obj.pD3D9ResOrig_ = NULL;
    obj.pD3D9Res_ = pD3D9res;
    obj.pD3D9Res_->AddRef();
  }

  // Release the Ex interface
  if (pDev9Ex) pDev9Ex->Release();

  // Check for CL format compatibilty
  if (obj.objDesc_.resType_ == D3DRTYPE_SURFACE) {
    cl_image_format clFmt = obj.getCLFormatFromD3D9(obj.objDesc_.d3dFormat_, plane);
    amd::Image::Format imageFormat(clFmt);
    if (!imageFormat.isSupported(amdContext)) {
      return CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    }
  }

  TD3D9RESINFO d3d9ObjOri = {*cl_surf_info, plane};
  TD3D9RESINFO d3d9ObjShared = {{obj.pD3D9Res_, obj.handleShared_}, plane};

  if (errcode == CL_SUCCESS) {
    resources_.push_back({d3d9ObjOri, d3d9ObjShared});
  }

  return errcode;
}
cl_uint D3D9Object::getMiscFlag() {
  switch (objDescOrig_.d3dFormat_) {
    case D3DFMT_NV_12:
    case D3DFMT_P010:
      return 1;
      break;
    case D3DFMT_YV_12:
      return 2;
      break;
    case D3DFMT_YUY2:
      return 3;
      break;
    default:
      return 0;
      break;
  }
}

cl_image_format D3D9Object::getCLFormatFromD3D9() {
  return getCLFormatFromD3D9(objDesc_.d3dFormat_, surfPlane_);
}

cl_image_format D3D9Object::getCLFormatFromD3D9(D3DFORMAT d3d9Fmt, cl_uint plane) {
  cl_image_format fmt;

  fmt.image_channel_order = 0;      // CL_RGBA;
  fmt.image_channel_data_type = 0;  // CL_UNSIGNED_INT8;

  switch (d3d9Fmt) {
    case D3DFMT_R32F:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_FLOAT;
      break;

    case D3DFMT_R16F:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_HALF_FLOAT;
      break;

    case D3DFMT_L16:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_UNORM_INT16;
      break;

    case D3DFMT_A8:
      fmt.image_channel_order = CL_A;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_L8:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_G32R32F:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_FLOAT;
      break;

    case D3DFMT_G16R16F:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_HALF_FLOAT;
      break;

    case D3DFMT_G16R16:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_UNORM_INT16;
      break;

    case D3DFMT_A8L8:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_A32B32G32R32F:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_FLOAT;
      break;

    case D3DFMT_A16B16G16R16F:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_HALF_FLOAT;
      break;

    case D3DFMT_A16B16G16R16:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNORM_INT16;
      break;

    case D3DFMT_A8B8G8R8:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_X8B8G8R8:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_A8R8G8B8:
      fmt.image_channel_order = CL_BGRA;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_X8R8G8B8:
      fmt.image_channel_order = CL_BGRA;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;
    case D3DFMT_NV_12:
      fmt.image_channel_data_type = CL_UNORM_INT8;
      if (plane == 0) {
        fmt.image_channel_order = CL_R;
      } else if (plane == 1) {
        fmt.image_channel_order = CL_RG;
      }
      break;
    case D3DFMT_P010:
      fmt.image_channel_data_type = CL_UNORM_INT16;
      if (plane == 0) {
        fmt.image_channel_order = CL_R;
      } else if (plane == 1) {
        fmt.image_channel_order = CL_RG;
      }
      break;
    case D3DFMT_YV_12:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;
    case D3DFMT_YUY2:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNSIGNED_INT8;
      break;
    case D3DFMT_UNKNOWN:
    case D3DFMT_R8G8B8:
    case D3DFMT_R5G6B5:
    case D3DFMT_X1R5G5B5:
    case D3DFMT_A1R5G5B5:
    case D3DFMT_A4R4G4B4:
    case D3DFMT_R3G3B2:
    case D3DFMT_A8R3G3B2:
    case D3DFMT_X4R4G4B4:
    case D3DFMT_A2B10G10R10:
    case D3DFMT_A2R10G10B10:
    case D3DFMT_A8P8:
    case D3DFMT_P8:
    case D3DFMT_A4L4:
    case D3DFMT_V8U8:
    case D3DFMT_L6V5U5:
    case D3DFMT_X8L8V8U8:
    case D3DFMT_Q8W8V8U8:
    case D3DFMT_V16U16:
    case D3DFMT_A2W10V10U10:
    case D3DFMT_UYVY:
    case D3DFMT_R8G8_B8G8:
    case D3DFMT_G8R8_G8B8:
    case D3DFMT_DXT1:
    case D3DFMT_DXT2:
    case D3DFMT_DXT3:
    case D3DFMT_DXT4:
    case D3DFMT_DXT5:
    case D3DFMT_D16_LOCKABLE:
    case D3DFMT_D32:
    case D3DFMT_D15S1:
    case D3DFMT_D24S8:
    case D3DFMT_D24X8:
    case D3DFMT_D24X4S4:
    case D3DFMT_D16:
    case D3DFMT_D32F_LOCKABLE:
    case D3DFMT_D24FS8:
    //#if !defined(D3D_DISABLE_9EX)
    case D3DFMT_D32_LOCKABLE:
    case D3DFMT_S8_LOCKABLE:
    //#endif // !D3D_DISABLE_9EX
    case D3DFMT_VERTEXDATA:
    case D3DFMT_INDEX16:
    case D3DFMT_INDEX32:
    case D3DFMT_Q16W16V16U16:
    case D3DFMT_MULTI2_ARGB8:
    case D3DFMT_CxV8U8:
    //#if !defined(D3D_DISABLE_9EX)
    case D3DFMT_A1:
    case D3DFMT_A2B10G10R10_XR_BIAS:
    case D3DFMT_BINARYBUFFER:
      _ASSERT(FALSE);  // NOT SURPPORTED
      break;
    //#endif // !D3D_DISABLE_9EX
    default:
      _ASSERT(FALSE);
      break;
  }

  return fmt;
}

bool D3D9Object::copyOrigToShared() {
  // Don't copy if there is no orig
  if (NULL == getD3D9ResOrig()) return true;

  IDirect3DDevice9Ex* d3dDev;
  HRESULT hr;
  ScopedLock sl(getResLock());

  IDirect3DSurface9* srcSurf = getD3D9ResOrig();
  IDirect3DSurface9* dstSurf = getD3D9Resource();

  hr = getD3D9Resource()->GetContainer(IID_IDirect3DDevice9Ex, (void**)&d3dDev);
  if (hr != D3D_OK || !d3dDev) {
    LogError("\nCannot get D3D9 device from D3D9 surface\n");
    return false;
  }

  hr = d3dDev->StretchRect(srcSurf, NULL, dstSurf, NULL, D3DTEXF_NONE);
  if (hr != D3D_OK) {
    LogError("\ncopy original surface to shared surface failed\n");
    return false;
  }
  // Flush D3D queues and make sure D3D stuff is finished
  pQuery_->Issue(D3DISSUE_END);
  BOOL data;
  while ((D3D_OK != pQuery_->GetData(&data, sizeof(BOOL), D3DGETDATA_FLUSH)) && (data != TRUE)) {
  }

  if (d3dDev) d3dDev->Release();
  return true;
}

bool D3D9Object::copySharedToOrig() {
  // Don't copy if there is no orig
  if (NULL == getD3D9ResOrig()) return true;

  IDirect3DDevice9Ex* d3dDev;
  HRESULT hr;
  ScopedLock sl(getResLock());

  hr = getD3D9Resource()->GetContainer(IID_IDirect3DDevice9Ex, (void**)&d3dDev);
  if (hr != D3D_OK || !d3dDev) {
    LogError("\nCannot get D3D9 device from D3D9 surface\n");
    return false;
  }

  hr = d3dDev->StretchRect(getD3D9Resource(), NULL, getD3D9ResOrig(), NULL, D3DTEXF_NONE);
  if (hr != D3D_OK) {
    LogError("\ncopy shared surface to original surface failed\n");
    return false;
  }

  if (d3dDev) d3dDev->Release();
  return true;
}

void Image2DD3D9::initDeviceMemory() {
  deviceMemories_ =
      reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(Image2DD3D9));
  memset(deviceMemories_, 0, context_().devices().size() * sizeof(DeviceMemory));
}

}  // namespace amd

#endif  //_WIN32
