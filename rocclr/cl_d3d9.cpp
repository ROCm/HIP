/* Copyright (c) 2010-present Advanced Micro Devices, Inc.

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

/* $Revision$ on $Date$ */

#ifndef __OPENCL_CL_D3D9_AMD_H
#define __OPENCL_CL_D3D9_AMD_H

#include "cl_common.hpp"
#include "platform/context.hpp"
#include "platform/memory.hpp"

#include <utility>

/* cl_amd_d3d9_sharing extension    */
#define cl_amd_d3d9_sharing 1

/* cl_amd_d3d9_sharing error codes */
#define CL_INVALID_D3D9_DEVICE_KHR              -1021
#define CL_INVALID_D3D9_RESOURCE_KHR            -1022

/* cl_amd_d3d9_sharing enumerations */
#define CL_CONTEXT_D3D9_DEVICE_KHR              0x4039

namespace amd
{
typedef struct
{
    union
    {
        UINT ByteWidth;
        UINT Width;
    };
    UINT Height;
    UINT Depth;
} D3D9ObjSize_t;

typedef struct
{
    D3D9ObjSize_t       objSize_;
    D3DFORMAT           d3dFormat_;
    D3DRESOURCETYPE     resType_;
    UINT                usage_;
    D3DPOOL             d3dPool_;
    D3DMULTISAMPLE_TYPE msType_;
    UINT                msQuality_;
    UINT                mipLevels_;
    UINT                fvf_;
    RECT                surfRect_;
} D3D9ObjDesc_t;

typedef struct d3d9ResInfo {
    cl_dx9_surface_info_khr surfInfo;
    cl_uint                 surfPlane;
} TD3D9RESINFO;


//typedef std::pair<cl_dx9_surface_info_khr, D3D9Object*> TD3D9OBJINFO;

//! Class D3D9Object keeps all the info about the D3D9 object
//! from which the CL object is created
class D3D9Object : public InteropObject
{
private:
    IDirect3DSurface9* pD3D9Aux_;
    cl_int  cliChecksum_;
    bool releaseResources_;
    static bool createSharedResource(D3D9Object& obj);
    static std::vector<std::pair<TD3D9RESINFO, TD3D9RESINFO>> resources_;

    //!Global lock
    static Monitor              resLock_;
    cl_uint                     surfPlane_;
    cl_dx9_surface_info_khr     surfInfo_;

protected:
    IDirect3DSurface9*  pD3D9Res_;
    IDirect3DSurface9*  pD3D9ResOrig_;
    IDirect3DQuery9*    pQuery_;
    D3D9ObjDesc_t       objDesc_;
    D3D9ObjDesc_t       objDescOrig_;
    HANDLE              handleOrig_;
    HANDLE              handleShared_;
    RECT                srcSurfRect;
    RECT                SharedSurfRect;
    cl_dx9_media_adapter_type_khr adapterType_;

public:
//! D3D9Object constructor initializes memeber variables
    D3D9Object()
        : releaseResources_(false),
        pQuery_(NULL)
    {
        // @todo Incorrect initialization!!!
        memset(this, 0, sizeof(D3D9Object));
    }
    //copy constructor
    D3D9Object(D3D9Object& d3d9obj)
        :pQuery_(NULL)
    {
        *this = d3d9obj;
        this->releaseResources_ = true;
    }

    //virtual destructor
    virtual ~D3D9Object()
    {
        ScopedLock sl(resLock_);
        if(releaseResources_) {
            if(pD3D9ResOrig_) pD3D9ResOrig_->Release();
            if(pD3D9Res_) pD3D9Res_->Release();
            if(pD3D9Aux_) pD3D9Aux_->Release();
            if(pQuery_) pQuery_->Release();
            //if the resouce is being used
            if(resources_.size()) {
                for(auto& it = resources_.cbegin(); it != resources_.cend(); it++) {
                    if( surfInfo_.resource && 
                        ((*it).first.surfInfo.resource == surfInfo_.resource) &&
                        ((*it).first.surfPlane == surfPlane_)) {
                            resources_.erase(it);
                            break;
                    }
                }
            }
        }
    }
    static int initD3D9Object(const Context& amdContext, cl_dx9_media_adapter_type_khr adapter_type, 
        cl_dx9_surface_info_khr* cl_surf_info, cl_uint plane, D3D9Object& obj);
    cl_uint getMiscFlag(void);

    D3D9Object* asD3D9Object() {return this;}
    IDirect3DSurface9* getD3D9Resource() const {return pD3D9Res_;}
    HANDLE getD3D9SharedHandle() const {return handleShared_;}
    IDirect3DSurface9* getD3D9ResOrig() const {return pD3D9ResOrig_;}
    RECT* getSrcSurfRect() {return &objDesc_.surfRect_;}
    RECT* getSharedSurfRect() {return &objDescOrig_.surfRect_;}
    void setD3D9AuxRes(IDirect3DSurface9* pAux) {pD3D9Aux_ = pAux;}
    IDirect3DSurface9* getD3D9AuxRes() {return pD3D9Aux_;}
    IDirect3DQuery9* getQuery() const {return pQuery_;}
    Monitor & getResLock() { return resLock_;}
    UINT getWidth() const {return objDesc_.objSize_.Width;}
    UINT getHeight() const {return objDesc_.objSize_.Height;}
    cl_uint getPlane() const {return surfPlane_;}
    cl_dx9_media_adapter_type_khr getAdapterType() const { return adapterType_;};
    const cl_dx9_surface_info_khr& getSurfInfo() const {return surfInfo_;};
    size_t getElementBytes(D3DFORMAT d3d9Format, cl_uint plane);
    size_t getElementBytes() {return getElementBytes(objDesc_.d3dFormat_, surfPlane_);}
    D3DFORMAT getD3D9Format() {return objDesc_.d3dFormat_;}
    D3D9ObjDesc_t* getObjDesc() {return &objDesc_;}
    cl_image_format getCLFormatFromD3D9();
    cl_image_format getCLFormatFromD3D9(D3DFORMAT d3d9Fmt, cl_uint plane);
    // On acquire copy data from original resource to shared resource
    virtual bool copyOrigToShared();
    // On release copy data from shared copy to the original resource
    virtual bool copySharedToOrig();
};

class Image2DD3D9 : public D3D9Object , public Image
{
protected:
    //! Initializes the device memory array which is nested
    // after'Image2DD3D9' object in memory layout.
    virtual void initDeviceMemory();
public:
//! Image2DD3D9 constructor just calls constructors of base classes
//! to pass down the parameters
    Image2DD3D9(
        Context&            amdContext,
        cl_mem_flags        clFlags,
        D3D9Object&         d3d9obj)
        : // Call base classes constructors
        D3D9Object(d3d9obj),
        Image(
            amdContext,
            CL_MEM_OBJECT_IMAGE2D,
            clFlags,
            d3d9obj.getCLFormatFromD3D9(),
            d3d9obj.getWidth(),
            d3d9obj.getHeight(),
            1,
            d3d9obj.getWidth() * d3d9obj.getElementBytes(), //rowPitch),
            0)
        {
            setInteropObj(this);
        }
    virtual ~Image2DD3D9() {}
};

cl_mem clCreateImage2DFromD3D9ResourceAMD(
    Context&        amdContext,
    cl_mem_flags    flags,
    cl_dx9_media_adapter_type_khr adapter_type,
    cl_dx9_surface_info_khr*  surface_info,
    cl_uint         plane,
    int*            errcode_ret);

void SyncD3D9Objects(std::vector<amd::Memory*>& memObjects);

} //namespace amd

#endif  /* __OPENCL_CL_D3D9_AMD_H   */
