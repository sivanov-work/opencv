// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/accelerators/dx11_alloc_resource.hpp"
#include "streaming/onevpl/accelerators/utils/shared_lock.hpp"
#include "logger.hpp"

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
#pragma comment(lib,"d3d11.lib")

#define D3D11_NO_HELPERS
#define NOMINMAX
#include <d3d11.h>
#include <d3d11_4.h>
#include <codecvt>
#include "opencv2/core/directx.hpp"
#ifdef HAVE_OPENCL
#include <CL/cl_d3d11.h>
#endif // HAVE_OPENCL
#undef D3D11_NO_HELPERS
#undef NOMINMAX

namespace cv {
namespace gapi {
namespace wip {

size_t LockAdapter::read_lock() {
    if(!impl) return 0;
    return impl->shared_lock();
}
size_t LockAdapter::unlock_read() {
    if(!impl) return 0;
    return impl->unlock_shared();
}
void LockAdapter::write_lock() {
    if(!impl) return;
    return impl->lock();
}

bool LockAdapter::is_write_acquired() {
    if(!impl) return true;
    return impl->owns();
}

void LockAdapter::unlock_write() {
    if(!impl) return;
    return impl->unlock();
}

SharedLock* LockAdapter::set_adaptee(SharedLock* new_impl) {
    SharedLock* old_impl = impl;
    impl = new_impl;
    return old_impl;
}

SharedLock* LockAdapter::get_adaptee() {
    return impl;
}

DX11AllocationItem::DX11AllocationItem(std::weak_ptr<DX11AllocationRecord> parent,
                                       CComPtr<ID3D11DeviceContext> origin_ctx,
                                       mfxFrameAllocator origin_allocator,
                                       CComPtr<ID3D11Texture2D> tex_ptr,
                                       subresource_id_t subtex_id,
                                       CComPtr<ID3D11Texture2D> staging_tex_ptr) :
    shared_device_context(origin_ctx),
    shared_allocator_copy(origin_allocator),
    texture_ptr(tex_ptr),
    subresource_id(subtex_id),
    staging_texture_ptr(staging_tex_ptr),
    observer(parent) {
    GAPI_DbgAssert(texture_ptr &&
                   "Cannot create DX11AllocationItem for empty texture");
    GAPI_DbgAssert(staging_tex_ptr &&
                   "Cannot create DX11AllocationItem for empty staging texture");
    GAPI_DbgAssert(observer.lock() &&
                   "Cannot create DX11AllocationItem for empty parent");
}

DX11AllocationItem::~DX11AllocationItem() {
    release();
    observer.reset();
}

void DX11AllocationItem::release() {
    auto parent = observer.lock();
    GAPI_LOG_DEBUG(nullptr, "texture: " << texture_ptr <<
                            ", subresource id: " << subresource_id <<
                            ", parent: " << parent.get());
    cv::util::suppress_unused_warning(parent);
}

CComPtr<ID3D11Texture2D> DX11AllocationItem::get_texture() {
    return texture_ptr;
}

CComPtr<ID3D11Texture2D> DX11AllocationItem::get_staging_texture() {
    return staging_texture_ptr;
}

DX11AllocationItem::subresource_id_t DX11AllocationItem::get_subresource() const {
    return subresource_id;
}

CComPtr<ID3D11DeviceContext> DX11AllocationItem::get_device_ctx() {
    return shared_device_context;
}

void DX11AllocationItem::on_first_in_impl(mfxFrameData *ptr) {
    D3D11_MAP mapType = D3D11_MAP_READ;
    UINT mapFlags = D3D11_MAP_FLAG_DO_NOT_WAIT;

    shared_device_context->CopySubresourceRegion(get_staging_texture(), 0,
                                          0, 0, 0,
                                          get_texture(), get_subresource(),
                                          nullptr);
    HRESULT err = S_OK;
    D3D11_MAPPED_SUBRESOURCE lockedRect {};
    do {
        err = shared_device_context->Map(get_staging_texture(), 0, mapType, mapFlags, &lockedRect);
        if (S_OK != err && DXGI_ERROR_WAS_STILL_DRAWING != err) {
            GAPI_LOG_WARNING(nullptr, "Cannot Map staging texture in device context, error: " << std::to_string(HRESULT_CODE(err)));
            GAPI_Assert(false && "Cannot Map staging texture in device context");
        }
    } while (DXGI_ERROR_WAS_STILL_DRAWING == err);

    if (FAILED(err)) {
        GAPI_LOG_WARNING(nullptr, "Cannot lock frame");
        GAPI_Assert(false && "Cannot lock frame");
        return ;
    }

    D3D11_TEXTURE2D_DESC desc {};
    get_texture()->GetDesc(&desc);
    switch (desc.Format) {
        case DXGI_FORMAT_NV12:
            ptr->Pitch = (mfxU16)lockedRect.RowPitch;
            ptr->Y     = (mfxU8 *)lockedRect.pData;
            ptr->UV     = (mfxU8 *)lockedRect.pData + desc.Height * lockedRect.RowPitch;

            GAPI_Assert(ptr->Y && ptr->UV/* && ptr->V */&& "DXGI_FORMAT_NV12 locked frame data is nullptr");
            break;
        default:
            GAPI_LOG_WARNING(nullptr, "Unknown DXGI format: " << desc.Format);
            return;
    }
}

void DX11AllocationItem::on_last_out_impl(mfxFrameData *ptr) {
    shared_device_context->Unmap(get_staging_texture(), 0);
    if (ptr) {
        ptr->Pitch = 0;
        ptr->U = ptr->V = ptr->Y = 0;
        ptr->A = ptr->R = ptr->G = ptr->B = 0;
    }
}

DX11AllocationRecord::DX11AllocationRecord() = default;

DX11AllocationRecord::~DX11AllocationRecord() {
    GAPI_LOG_DEBUG(nullptr, "record: " << this <<
                            ", subresources count: " << resources.size());

    for (AllocationId id : resources) {
        delete id;
    }
    resources.clear();

    GAPI_LOG_DEBUG(nullptr, "release final referenced texture: " << texture_ptr);
    if(texture_ptr) {
        texture_ptr->Release();
    }
}

void DX11AllocationRecord::init(unsigned int items,
                                CComPtr<ID3D11DeviceContext> origin_ctx,
                                mfxFrameAllocator origin_allocator,
                                ID3D11Texture2D* texture,
                                std::vector<ID3D11Texture2D*> &&staging_textures) {
    GAPI_DbgAssert(items != 0 && "Cannot create DX11AllocationRecord with empty items");
    GAPI_DbgAssert(items == staging_textures.size() && "Allocation items count and staging size are not equal");
    GAPI_DbgAssert(origin_ctx &&
                   "Cannot create DX11AllocationItem for empty origin_ctx");
    auto shared_allocator_copy = origin_allocator;
    GAPI_DbgAssert((shared_allocator_copy.Lock && shared_allocator_copy.Unlock) &&
                   "Cannot create DX11AllocationItem for empty origin allocator");

    // abandon unusable c-allocator interfaces
    shared_allocator_copy.Alloc = nullptr;
    shared_allocator_copy.Free = nullptr;
    shared_allocator_copy.pthis = nullptr;


    GAPI_LOG_DEBUG(nullptr, "subresources count: " << items << ", text: " << texture)
    resources.reserve(items);
    texture_ptr = texture; // no AddRef here, because DX11AllocationRecord receive ownership it here
    for(unsigned int i = 0; i < items; i++ ) {
        resources.emplace_back(new DX11AllocationItem(get_ptr(), origin_ctx, shared_allocator_copy,
                                                      texture, i, staging_textures[i]));
    }
}

DX11AllocationRecord::Ptr DX11AllocationRecord::get_ptr() {
    return shared_from_this();
}

DX11AllocationRecord::AllocationId* DX11AllocationRecord::data() {
    return resources.data();
}

} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_ONEVPL
