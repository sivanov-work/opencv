// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/accelerators/accel_policy_dx11.hpp"
//#include "streaming/vpl/vpl_utils.hpp"
#include "streaming/onevpl/accelerators/surface/cpu_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/surface/dx11_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/onevpl_utils.hpp"
#include "logger.hpp"

#ifdef HAVE_DIRECTX
    #ifdef HAVE_D3D11
        #pragma comment(lib,"d3d11.lib")

        #define D3D11_NO_HELPERS
        #include <d3d11.h>
        #include <d3d11_4.h>
        #include <codecvt>
        #include "opencv2/core/directx.hpp"
        #ifdef HAVE_OPENCL
            #include <CL/cl_d3d11.h>
        #endif

namespace cv {
namespace gapi {
namespace wip {

VPLDX11AccelerationPolicy::VPLDX11AccelerationPolicy() :
    hw_handle(),
    allocator()
{
    // setup dx11 allocator
    memset(&allocator, 0, sizeof(mfxFrameAllocator));
    allocator.Alloc = alloc_cb;
    allocator.Lock = lock_cb;
    allocator.Unlock = unlock_cb;
    allocator.GetHDL = get_hdl_cb;
    allocator.Free = free_cb;
    allocator.pthis = this;

#ifdef CPU_ACCEL_ADAPTER
    adapter.reset(new VPLCPUAccelerationPolicy);
#endif
}

VPLDX11AccelerationPolicy::~VPLDX11AccelerationPolicy()
{
    for (auto& allocation_pair : allocation_table) {
        allocation_pair.second.clear();
    }

    if (hw_handle)
    {
        GAPI_LOG_INFO(nullptr, "VPLDX11AccelerationPolicy release ID3D11Device");
        hw_handle->Release();
    }
}

void VPLDX11AccelerationPolicy::init(session_t session) {
    //Create device
    UINT creationFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

#if defined(_DEBUG)
    // If the project is in a debug build, enable debugging via SDK Layers with this flag.
    creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    // This array defines the set of DirectX hardware feature levels this app will support.
    // Note the ordering should be preserved.
    // Don't forget to declare your application's minimum required feature level in its
    // description.  All applications are assumed to support 9.1 unless otherwise stated.
    D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3
    };
    D3D_FEATURE_LEVEL featureLevel;

    // Create the Direct3D 11 API device object and a corresponding context.
    ID3D11DeviceContext* context;
    HRESULT err =
        D3D11CreateDevice(
            nullptr, // Specify nullptr to use the default adapter.
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            creationFlags, // Set set debug and Direct2D compatibility flags.
            featureLevels, // List of feature levels this app can support.
            ARRAYSIZE(featureLevels),
            D3D11_SDK_VERSION, // Always set this to D3D11_SDK_VERSION.
            &hw_handle, // Returns the Direct3D device created.
            &featureLevel, // Returns feature level of device created.
            &context // Returns the device immediate context.
            );
    if(FAILED(err))
    {
        throw std::logic_error("Cannot create D3D11CreateDevice, error: " + std::to_string(HRESULT_CODE(err)));
    }

    // oneVPL recommendation
    ID3D11Multithread       *pD11Multithread;
    context->QueryInterface(IID_PPV_ARGS(&pD11Multithread));
    pD11Multithread->SetMultithreadProtected(true);

    mfxStatus sts = MFXVideoCORE_SetHandle(session, MFX_HANDLE_D3D11_DEVICE, (mfxHDL) hw_handle);
    if (sts != MFX_ERR_NONE)
    {
        throw std::logic_error("Cannot create VPLDX11AccelerationPolicy, MFXVideoCORE_SetHandle error: " +
                               mfxstatus_to_string(sts));
    }

    sts = MFXVideoCORE_SetFrameAllocator(session, &allocator);
    if (sts != MFX_ERR_NONE)
    {
        throw std::logic_error("Cannot create VPLDX11AccelerationPolicy, MFXVideoCORE_SetFrameAllocator error: " +
                               mfxstatus_to_string(sts));
    }
    GAPI_LOG_INFO(nullptr, "VPLDX11AccelerationPolicy initialized, session: " << session);
}

void VPLDX11AccelerationPolicy::deinit(session_t session) {
    (void)session;
    GAPI_LOG_INFO(nullptr, "deinitialize session: " << session);
}

VPLDX11AccelerationPolicy::pool_key_t
VPLDX11AccelerationPolicy::create_surface_pool(size_t pool_size, size_t surface_size_bytes,
                                               surface_ptr_ctr_t creator) {
    GAPI_LOG_DEBUG(nullptr, "pool size: " << pool_size << ", surface size bytes: " << surface_size_bytes);

#ifdef CPU_ACCEL_ADAPTER
    return adapter->create_surface_pool(pool_size, surface_size_bytes, creator);
#endif
    (void)pool_size;
    (void)surface_size_bytes;
    (void)creator;
    throw std::runtime_error("VPLDX11AccelerationPolicy::create_surface_pool() is not implemented");
}

VPLDX11AccelerationPolicy::surface_weak_ptr_t VPLDX11AccelerationPolicy::get_free_surface(pool_key_t key)
{
#ifdef CPU_ACCEL_ADAPTER
    return adapter->get_free_surface(key);
#endif
    (void)key;
    throw std::runtime_error("VPLDX11AccelerationPolicy::get_free_surface() is not implemented");
}

size_t VPLDX11AccelerationPolicy::get_free_surface_count(pool_key_t key) const {
#ifdef CPU_ACCEL_ADAPTER
    return adapter->get_free_surface_count(key);
#endif
    (void)key;
    throw std::runtime_error("get_free_surface_count() is not implemented");
}

size_t VPLDX11AccelerationPolicy::get_surface_count(pool_key_t key) const {
#ifdef CPU_ACCEL_ADAPTER
    return adapter->get_surface_count(key);
#endif
    (void)key;
    throw std::runtime_error("VPLDX11AccelerationPolicy::get_surface_count() is not implemented");
}

cv::MediaFrame::AdapterPtr VPLDX11AccelerationPolicy::create_frame_adapter(pool_key_t key,
                                                                           mfxFrameSurface1* surface) {

#ifdef CPU_ACCEL_ADAPTER
    return adapter->create_frame_adapter(key, surface);
#endif
    (void)key;
    (void)surface;
    throw std::runtime_error("VPLDX11AccelerationPolicy::create_frame_adapter() is not implemented");
}

mfxStatus VPLDX11AccelerationPolicy::alloc_cb(mfxHDL pthis, mfxFrameAllocRequest *request,
                                              mfxFrameAllocResponse *response) {
    if (!pthis) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    VPLDX11AccelerationPolicy *self = static_cast<VPLDX11AccelerationPolicy *>(pthis);
    return self->on_alloc(request, response);
}

mfxStatus VPLDX11AccelerationPolicy::lock_cb(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr) {
    if (!pthis) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    VPLDX11AccelerationPolicy *self = static_cast<VPLDX11AccelerationPolicy *>(pthis);
    return self->on_lock(mid, ptr);
}

mfxStatus VPLDX11AccelerationPolicy::unlock_cb(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr) {
    if (!pthis) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    VPLDX11AccelerationPolicy *self = static_cast<VPLDX11AccelerationPolicy *>(pthis);
    return self->on_unlock(mid, ptr);
}

mfxStatus VPLDX11AccelerationPolicy::get_hdl_cb(mfxHDL pthis, mfxMemId mid, mfxHDL *handle) {
    if (!pthis) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    VPLDX11AccelerationPolicy *self = static_cast<VPLDX11AccelerationPolicy *>(pthis);
    return self->on_get_hdl(mid, handle);
}

mfxStatus VPLDX11AccelerationPolicy::free_cb(mfxHDL pthis, mfxFrameAllocResponse *response) {
    if (!pthis) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    VPLDX11AccelerationPolicy *self = static_cast<VPLDX11AccelerationPolicy *>(pthis);
    return self->on_free(response);
}

mfxStatus VPLDX11AccelerationPolicy::on_alloc(mfxFrameAllocRequest *request,
                                              mfxFrameAllocResponse *response) {
    GAPI_LOG_DEBUG(nullptr, "Requestend allocation id: " << std::to_string(request->AllocId) <<
                            ", type: " << ext_mem_frame_type_to_cstr(request->Type) <<
                            ", size: " << request->Info.Width << "x" << request->Info.Height <<
                            ", frames sugested count: " << request->NumFrameSuggested);
    auto table_it = allocation_table.find(request->AllocId);
    if (allocation_table.end() != table_it) {
        GAPI_LOG_WARNING(nullptr, "Allocation already exist, id: " + std::to_string(request->AllocId) +
                                   ". Total allocation size: " + std::to_string(allocation_table.size()));

        // TODO cache
        return MFX_ERR_MEMORY_ALLOC;
    }

    DXGI_FORMAT colorFormat = VPLMediaFrameDX11Adapter::get_dx11_color_format(request->Info.FourCC);

    if (DXGI_FORMAT_UNKNOWN == colorFormat || colorFormat != DXGI_FORMAT_NV12) {
        GAPI_LOG_WARNING(nullptr, "Unsupported fourcc :" << request->Info.FourCC);
        return MFX_ERR_UNSUPPORTED;
    }

    D3D11_TEXTURE2D_DESC desc = { 0 };

    desc.Width = request->Info.Width;
    desc.Height = request->Info.Height;

    desc.MipLevels = 1;
    // single texture with subresources
    desc.ArraySize = request->NumFrameSuggested;
    desc.Format = colorFormat;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    desc.BindFlags = D3D11_BIND_DECODER;

    if (request->Type & MFX_MEMTYPE_SHARED_RESOURCE) {
        desc.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    }

    ID3D11Texture2D *pTexture2D;
    HRESULT err = hw_handle->CreateTexture2D(&desc, nullptr, &pTexture2D);
    if (FAILED(err)) {
        GAPI_LOG_WARNING(nullptr, "Cannot create texture, error: " + std::to_string(HRESULT_CODE(err)));
        return MFX_ERR_MEMORY_ALLOC;
    }

    // for multiple subresources initialize allocation array
    auto cand_resource_it = allocation_table.end();
    {
        allocation_t resources(request->NumFrameSuggested);
        for(decltype(request->NumFrameSuggested) i = 0; i < request->NumFrameSuggested; i++ ) {
            resources[i].texture_ptr = pTexture2D;
            resources[i].subresource_id = i;
        }

        // insert into global table
        auto inserted_it = allocation_table.emplace(request->AllocId, std::move(resources));
        if (!inserted_it.second) {
            GAPI_LOG_WARNING(nullptr, "Cannot assign allocation by id: " + std::to_string(request->AllocId) +
                                    " - aldeady exist. Total allocation size: " + std::to_string(allocation_table.size()));
            pTexture2D->Release();
            return MFX_ERR_MEMORY_ALLOC;
        }

        cand_resource_it = inserted_it.first;
    }

    //fill out response
    GAPI_DbgAssert(cand_resource_it != allocation_table.end() && "Invalid cand_resource_it");

    allocation_t &resources_array = cand_resource_it->second;
    response->NumFrameActual = request->NumFrameSuggested;
    response->mids = reinterpret_cast<mfxMemId *>(&resources_array);

    return MFX_ERR_NONE;
}

mfxStatus VPLDX11AccelerationPolicy::on_lock(mfxMemId mid, mfxFrameData *ptr) {
    GAPI_LOG_DEBUG(nullptr, __FUNCTION__);
    return MFX_ERR_MEMORY_ALLOC;
}

mfxStatus VPLDX11AccelerationPolicy::on_unlock(mfxMemId mid, mfxFrameData *ptr) {
    GAPI_LOG_DEBUG(nullptr, __FUNCTION__);
    return MFX_ERR_MEMORY_ALLOC;
}

mfxStatus VPLDX11AccelerationPolicy::on_get_hdl(mfxMemId mid, mfxHDL *handle) {
    GAPI_LOG_DEBUG(nullptr, __FUNCTION__);
    return MFX_ERR_MEMORY_ALLOC;
}

mfxStatus VPLDX11AccelerationPolicy::on_free(mfxFrameAllocResponse *response) {
    GAPI_LOG_DEBUG(nullptr, "Allocations count before: " << allocation_table.size());

    auto table_it = allocation_table.find(response->AllocId);
    if (allocation_table.end() == table_it) {
        GAPI_LOG_WARNING(nullptr, "Cannot find allocation id: " + std::to_string(response->AllocId) +
                                   ". Total allocation size: " + std::to_string(allocation_table.size()));
        return MFX_ERR_MEMORY_ALLOC;
    }

    table_it->second.begin()->texture_ptr->Release();
    allocation_table.erase(table_it);
    return MFX_ERR_MEMORY_ALLOC;
}
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_ONEVPL
