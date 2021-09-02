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

size_t lockable::read_lock() {
    //GAPI_DbgAssert(impl && "No impl");
    if(!impl) return 0;
    return impl->shared_lock();
}
size_t lockable::unlock_read() {
    //GAPI_DbgAssert(impl && "No impl");
    if(!impl) return 0;
    return impl->unlock_shared();
}
void lockable::write_lock() {
    //GAPI_DbgAssert(impl && "No impl");
    if(!impl) return;
    return impl->lock();
}
void lockable::unlock_write() {
    //GAPI_DbgAssert(impl && "No impl");
    if(!impl) return;
    return impl->unlock();
}
SharedLock* lockable::set_locable_impl(SharedLock* new_impl) {
    SharedLock* old_impl = impl;
    impl = new_impl;
    return old_impl;
}
SharedLock* lockable::get_locable_impl() {
    return impl;
}

allocation_data_t::allocation_data_t(std::weak_ptr<allocation_record> parent,
                                     ID3D11Texture2D* tex_ptr,
                                     subresource_id_t subtex_id,
                                     ID3D11Texture2D* staging_tex_ptr) :
    texture_ptr(tex_ptr),
    subresource_id(subtex_id),
    staging_texture_ptr(staging_tex_ptr),
    observer(parent),
    incoming_requests(),
    outgoing_requests(),
    pending_requests(),
    reinit(false) {

    GAPI_DbgAssert(texture_ptr && "Cannot create allocation_data_t for empty texture");
    GAPI_DbgAssert(staging_tex_ptr && "Cannot create allocation_data_t for empty staging texture");
    GAPI_DbgAssert(observer.lock() && "Cannot create allocation_data_t for empty parent");

    // increase reference counter, cause allocation_data_t shares ownership
    texture_ptr->AddRef();

    // no need to increase reference to staging_tex_ptr, because
    // allocation_data_t owns it in exclusove way and receive ownership
}

allocation_data_t::~allocation_data_t() {
    release();
    observer.reset();
}

void allocation_data_t::release() {
    GAPI_LOG_DEBUG(nullptr, "texture: " << texture_ptr <<
                            ", subresource id: " << subresource_id <<
                            ", parent: " << observer.lock().get());
    if(texture_ptr) {
        texture_ptr->Release();
        texture_ptr = nullptr;
    }

    if(staging_texture_ptr) {
        staging_texture_ptr->Release();
        staging_texture_ptr = nullptr;
    }
}

ID3D11Texture2D* allocation_data_t::get_texture() {
    return texture_ptr;
}

ID3D11Texture2D* allocation_data_t::get_staging_texture() {
    return staging_texture_ptr;
}

allocation_data_t::subresource_id_t allocation_data_t::get_subresource() const {
    return subresource_id;
}

allocation_record::allocation_record() = default;
allocation_record::~allocation_record() {
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

void allocation_record::init(unsigned int items, ID3D11Texture2D* texture,
                             std::vector<ID3D11Texture2D*> &&staging_textures) {
    GAPI_DbgAssert(items != 0 && "Cannot create allocation_record with empty items");
    GAPI_DbgAssert(items == staging_textures.size() && "Allocation items count and staging size are not equal");

    GAPI_LOG_DEBUG(nullptr, "subresources count: " << items << ", text: " << texture)
    resources.reserve(items);
    texture_ptr = texture; // no AddRef here, because allocation_record receive ownership it here
    for(unsigned int i = 0; i < items; i++ ) {
        resources.emplace_back(new allocation_data_t(get_ptr(), texture, i, staging_textures[i]));
    }
}

allocation_record::Ptr allocation_record::get_ptr() {
    return shared_from_this();
}

allocation_record::AllocationId* allocation_record::data() {
    return resources.data();
}

VPLDX11AccelerationPolicy::VPLDX11AccelerationPolicy() :
    hw_handle(),
    device_context(),
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
        allocation_pair.second.reset();
    }

    if (device_context) {
        GAPI_LOG_INFO(nullptr, "release context: " << device_context);
        device_context->Release();
    }

    if (hw_handle)
    {
        GAPI_LOG_INFO(nullptr, "release ID3D11Device");
        hw_handle->Release();
    }
}

VPLAccelerationPolicy::AccelType VPLDX11AccelerationPolicy::get_accel_type() const {
    return AccelType::GPU;
}

void VPLDX11AccelerationPolicy::init(session_t session) {
    //Create device
    UINT creationFlags = 0;//D3D11_CREATE_DEVICE_BGRA_SUPPORT;

//#if defined(_DEBUG)
    // If the project is in a debug build, enable debugging via SDK Layers with this flag.
    creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
//#endif

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
    };
    D3D_FEATURE_LEVEL featureLevel;

    // Create the Direct3D 11 API device object and a corresponding context.
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
            &device_context // Returns the device immediate context.
            );
    if(FAILED(err))
    {
        throw std::logic_error("Cannot create D3D11CreateDevice, error: " + std::to_string(HRESULT_CODE(err)));
    }

    // oneVPL recommendation
    {
        ID3D11Multithread       *pD11Multithread;
        device_context->QueryInterface(IID_PPV_ARGS(&pD11Multithread));
        pD11Multithread->SetMultithreadProtected(true);
        pD11Multithread->Release();
    }

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

VPLDX11AccelerationPolicy::pool_key_t
VPLDX11AccelerationPolicy::create_surface_pool(const mfxFrameAllocRequest& alloc_req,
                                               mfxVideoParam& param) {

    param.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;

    // allocate textures by explicit request
    mfxFrameAllocResponse mfxResponse;
    //TODO
    mfxFrameAllocRequest alloc_request = alloc_req;
    alloc_request.NumFrameSuggested = alloc_request.NumFrameSuggested * 5;
    mfxStatus sts = on_alloc(&alloc_request, &mfxResponse);
    if (sts != MFX_ERR_NONE)
    {
        throw std::logic_error("Cannot create allocate memory for surfaces, error: " +
                               mfxstatus_to_string(sts));
    }

    // get reference pointer
    auto table_it = allocation_table.find(alloc_request.AllocId);
    GAPI_DbgAssert (allocation_table.end() != table_it);

    mfxU16 numSurfaces = alloc_request.NumFrameSuggested;

    // create pool
    pool_t pool;
    pool.reserve(numSurfaces);
    for (int i = 0; i < numSurfaces; i++) {
        std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1 {});
        handle->Info = param.mfx.FrameInfo;
        handle->Data.MemId = mfxResponse.mids[i];

        pool.push_back(Surface::create_surface(std::move(handle), table_it->second));
    }

    // remember pool by key
    pool_key_t key = reinterpret_cast<pool_key_t>(table_it->second.get());
    GAPI_LOG_INFO(nullptr, "New pool allocated, key: " << key <<
                           ", surface count: " << pool.total_size());
    try {
        if (!pool_table.emplace(key, std::move(pool)).second) {
            throw std::runtime_error(std::string("VPLDX11AccelerationPolicy::create_surface_pool - ") +
                                     "cannot insert pool, table size: " + std::to_string(pool_table.size()));
        }
    } catch (const std::exception&) {
        throw;
    }
    return key;
}

VPLDX11AccelerationPolicy::surface_weak_ptr_t VPLDX11AccelerationPolicy::get_free_surface(pool_key_t key)
{
#ifdef CPU_ACCEL_ADAPTER
    return adapter->get_free_surface(key);
#else
    auto pool_it = pool_table.find(key);
    if (pool_it == pool_table.end()) {
        std::stringstream ss;
        ss << "key is not found: " << key << ", table size: " << pool_table.size();
        const std::string& str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);
        throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
    }

    pool_t& requested_pool = pool_it->second;
    return requested_pool.find_free();
#endif
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
    auto pool_it = pool_table.find(key);
    if (pool_it == pool_table.end()) {
        std::stringstream ss;
        ss << "key is not found: " << key << ", table size: " << pool_table.size();
        const std::string& str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);
        throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
    }

    pool_t& requested_pool = pool_it->second;
    return cv::MediaFrame::AdapterPtr{new VPLMediaFrameDX11Adapter(requested_pool.find_by_handle(surface),
                                                                   allocator)};
}

mfxStatus VPLDX11AccelerationPolicy::alloc_cb(mfxHDL pthis, mfxFrameAllocRequest *request,
                                              mfxFrameAllocResponse *response) {
    if (!pthis) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    VPLDX11AccelerationPolicy *self = static_cast<VPLDX11AccelerationPolicy *>(pthis);

    request->NumFrameSuggested *= 5;
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

mfxStatus VPLDX11AccelerationPolicy::on_alloc(const mfxFrameAllocRequest *request,
                                              mfxFrameAllocResponse *response) {
    GAPI_LOG_DEBUG(nullptr, "Requestend allocation id: " << std::to_string(request->AllocId) <<
                            ", type: " << ext_mem_frame_type_to_cstr(request->Type) <<
                            ", size: " << request->Info.Width << "x" << request->Info.Height <<
                            ", frames minimum count: " << request->NumFrameMin <<
                            ", frames sugested count: " << request->NumFrameSuggested);
    auto table_it = allocation_table.find(request->AllocId);
    if (allocation_table.end() != table_it) {
        GAPI_LOG_WARNING(nullptr, "Allocation already exist, id: " + std::to_string(request->AllocId) +
                                   ". Total allocation size: " + std::to_string(allocation_table.size()));

        // TODO cache
        allocation_t &resources_array = table_it->second;
        response->AllocId = request->AllocId;
        response->NumFrameActual = request->NumFrameSuggested;
        response->mids = reinterpret_cast<mfxMemId *>(resources_array->data());

        return MFX_ERR_NONE;
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

    // create  staging texture to read it from
    desc.ArraySize      = 1;
    desc.Usage          = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.BindFlags      = 0;
    desc.MiscFlags      = 0;
    std::vector<ID3D11Texture2D*> staging_textures;
    staging_textures.reserve(request->NumFrameSuggested);
    for (int i = 0; i < request->NumFrameSuggested; i ++ ) {
        ID3D11Texture2D *staging_texture_2d = nullptr;
        err = hw_handle->CreateTexture2D(&desc, NULL, &staging_texture_2d);
        if (FAILED(err)) {
            GAPI_LOG_WARNING(nullptr, "Cannot create staging texture, error: " + std::to_string(HRESULT_CODE(err)));
            return MFX_ERR_MEMORY_ALLOC;
        }
        staging_textures.push_back(staging_texture_2d);
    }

    // for multiple subresources initialize allocation array
    auto cand_resource_it = allocation_table.end();
    {
        // insert into global table
        auto inserted_it = allocation_table.emplace(request->AllocId,
                                                    allocation_record::create(request->NumFrameSuggested,
                                                                              pTexture2D,
                                                                              std::move(staging_textures)));
        if (!inserted_it.second) {
            GAPI_LOG_WARNING(nullptr, "Cannot assign allocation by id: " + std::to_string(request->AllocId) +
                                    " - aldeady exist. Total allocation size: " + std::to_string(allocation_table.size()));
            pTexture2D->Release();
            return MFX_ERR_MEMORY_ALLOC;
        }

        GAPI_LOG_DEBUG(nullptr, "allocation by id: " << request->AllocId <<
                                " was created, total allocations count: " << allocation_table.size());
        cand_resource_it = inserted_it.first;
    }

    //fill out response
    GAPI_DbgAssert(cand_resource_it != allocation_table.end() && "Invalid cand_resource_it");

    allocation_t &resources_array = cand_resource_it->second;
    response->AllocId = request->AllocId;
    response->NumFrameActual = request->NumFrameSuggested;
    response->mids = reinterpret_cast<mfxMemId *>(resources_array->data());

    return MFX_ERR_NONE;
}

mfxStatus VPLDX11AccelerationPolicy::on_lock(mfxMemId mid, mfxFrameData *ptr) {
    allocation_record::AllocationId data = reinterpret_cast<allocation_record::AllocationId>(mid);
    if (!data) {
        GAPI_LOG_WARNING(nullptr, "Allocation record is empty");
        return MFX_ERR_LOCK_MEMORY;
    }

    GAPI_LOG_DEBUG(nullptr, "texture : " << data->get_texture() << ", sub id: " << data->get_subresource());

    // each `incoming` request must check in before acquire resource
    size_t thread_id = data->incoming_requests.fetch_add(1);
    if (thread_id == 0) {
        /*
         * only one `incoming` request is allowable to init resource
         * at first time
         * let's filter out the first one by `thread_id`
         *
         * The first one `incoming` request becomes main `incoming` request
         */
        if (data->outgoing_requests.load() == 0) {
            D3D11_MAP mapType = D3D11_MAP_READ;
            UINT mapFlags = D3D11_MAP_FLAG_DO_NOT_WAIT;

            device_context->CopySubresourceRegion(data->get_staging_texture(), 0,
                                                  0, 0, 0,
                                                  data->get_texture(), data->get_subresource(),
                                                  nullptr);
            HRESULT err = S_OK;
            D3D11_MAPPED_SUBRESOURCE lockedRect {};
            do {
                err = device_context->Map(data->get_staging_texture(), 0, mapType, mapFlags, &lockedRect);
                if (S_OK != err && DXGI_ERROR_WAS_STILL_DRAWING != err) {
                    GAPI_LOG_WARNING(nullptr, "Cannot Map staging texture in device context, error: " << std::to_string(HRESULT_CODE(err)));
                    return MFX_ERR_LOCK_MEMORY;
                }
            } while (DXGI_ERROR_WAS_STILL_DRAWING == err);

            if (FAILED(err)) {
                GAPI_LOG_WARNING(nullptr, "Cannot lock frame");
                return MFX_ERR_LOCK_MEMORY;
            }

            D3D11_TEXTURE2D_DESC desc {};
            data->get_texture()->GetDesc(&desc);
            switch (desc.Format) {
                case DXGI_FORMAT_NV12:
                    ptr->Pitch = (mfxU16)lockedRect.RowPitch;
                    ptr->Y     = (mfxU8 *)lockedRect.pData;
                    ptr->UV     = (mfxU8 *)lockedRect.pData + desc.Height * lockedRect.RowPitch;

                    GAPI_Assert(ptr->Y && ptr->UV/* && ptr->V */&& "DXGI_FORMAT_NV12 locked frame data is nullptr");
                    break;
                default:
                    GAPI_LOG_WARNING(nullptr, "Unknown DXGI format: " << desc.Format);
                    return MFX_ERR_LOCK_MEMORY;
            }

            /*
             * The main `incoming` request finished resource initialization
             * and became `outgoing`
             *
             * Non empty `outgoing` count means that
             * other further `incoming` (or busy-wait) requests
             * are getting on with its job without resource initialization,
             * because main `incoming` request has already initialized it at here
             */
            data->outgoing_requests.fetch_add(1);
            GAPI_Assert(ptr->Y && (ptr->UV || (ptr->U && ptr->V)) &&
                        "on_lock: data must be correct after first init");
            GAPI_LOG_DEBUG(nullptr, "charged, data: " << data);
            return MFX_ERR_NONE;
        }
        GAPI_Assert(false && "Unknown state");
        return MFX_ERR_NONE;
    } else {
        /*
         * CASE 1)
         *
         * busy wait for others `incoming` requests for resource initialization
         * besides main `incoming` request which are getting on
         * resource initialization at this point
         *
         */

        // OR

        /*
         * CASE 2)
         *
         * busy wait for ALL `incoming` request for resource initialization
         * including main `incoming` request. It will happen if
         * new `incoming` requests had came here while resource was getting on deinit
         * in `on_unlock` in another processing thread.
         * In this case no actual main `incoming` request is available and
         * all `incoming` requests must be in busy-wait stare
         *
         */

        // Each `incoming` request became `busy-wait` request
        size_t busy_thread_id = data->pending_requests.fetch_add(1);

        /*
         * CASE 1)
         *
         * Non empty `outgoing` requests count means that other further `incoming` or
         * `busy-wait` request are getting on with its job
         * without resource initialization because
         * main thread has already initialized it at here
         */
        while (data->outgoing_requests.load() == 0) {

            // OR

            /*
             * CASE 2)
             *
             * In case of NO master `incoming `request is available and doesn't
             * provide resource initialization. All `incoming` requests must be in
             * busy-wait state.
             * If it is not true then CASE 1) is going on
             *
             * OR
             *
             * `on_unlock` is in deinitialization phase in another thread.
             * Both cases mean busy-wait state here
             */
            if (data->pending_requests.load() == data->incoming_requests.load()) {
                /*
                 * CASE 2) ONLY
                 *
                 * It will happen if 'on_unlock` in another thread
                 * finishes its execution only
                 *
                 * `on_unlock` in another thread might finished with either
                 * deinitialization action or without deinitialization action
                 * (the call off deinitialization case)
                 *
                 * We must not continue at here (without reinit)
                 * if deinitialization happens in `on_unlock` in another thread.
                 * So try it on
                 */

                // only single `busy-wait` request must make sure about possible
                // deinitialization. So first `busy-wait` request becomes
                // main `busy-wait` request
                if (busy_thread_id == 0) {
                    bool expected_reinit = true;
                    if (!data->reinit.compare_exchange_strong(expected_reinit, false)) {
                        /*
                         * deinitialization called off in `on_unlock`
                         * because new `incoming` request had appeared at here before
                         * `on_unlock` started deinit procedure in another thread.
                         * So no reinit required because no deinit had happended
                         *
                         * main `busy-wait` request must break busy-wait state
                         * and become `outgoing` request.
                         * Non empty `outgoing` count means that other
                         * further `incoming` requests or
                         * `busy-wait` requests are getting on with its job
                         * without resource initialization/reinitialization
                         * because no deinit happened in `on_unlock`
                         * in another thread
                         */
                        break; //just quit busy loop
                    } else {
                        /* Deinitialization had happened in `on_unlock`
                         * in another thread right before
                         * new `incoming` requests appeared.
                         * So main `busy-wait` request must start reinit procedure
                         */
                        D3D11_MAP mapType = D3D11_MAP_READ;
                        UINT mapFlags = D3D11_MAP_FLAG_DO_NOT_WAIT;

                        device_context->CopySubresourceRegion(data->get_staging_texture(), 0,
                                                              0, 0, 0,
                                                              data->get_texture(), data->get_subresource(),
                                                              nullptr);
                        HRESULT err = S_OK;
                        D3D11_MAPPED_SUBRESOURCE lockedRect {};
                        do {
                            err = device_context->Map(data->get_staging_texture(), 0, mapType, mapFlags, &lockedRect);
                            if (S_OK != err && DXGI_ERROR_WAS_STILL_DRAWING != err) {
                                GAPI_LOG_WARNING(nullptr, "Cannot Map staging texture in device context, error: " << std::to_string(HRESULT_CODE(err)));
                                return MFX_ERR_LOCK_MEMORY;
                            }
                        } while (DXGI_ERROR_WAS_STILL_DRAWING == err);

                        if (FAILED(err)) {
                            GAPI_LOG_WARNING(nullptr, "Cannot lock frame");
                            return MFX_ERR_LOCK_MEMORY;
                        }

                        D3D11_TEXTURE2D_DESC desc {};
                        data->get_texture()->GetDesc(&desc);
                        switch (desc.Format) {
                            case DXGI_FORMAT_NV12:
                                ptr->Pitch = (mfxU16)lockedRect.RowPitch;
                                ptr->Y     = (mfxU8 *)lockedRect.pData;
                                ptr->UV     = (mfxU8 *)lockedRect.pData + desc.Height * lockedRect.RowPitch;

                                GAPI_Assert(ptr->Y && ptr->UV/* && ptr->V */&& "DXGI_FORMAT_NV12 locked frame data is nullptr");
                                break;
                            default:
                                GAPI_LOG_WARNING(nullptr, "Unknown DXGI format: " << desc.Format);
                                return MFX_ERR_LOCK_MEMORY;
                        }

                        /*
                         * Main `busy-wait` request has finished reinit procedure
                         * and becomes `outgong` request.
                         * Non empty `outgoing` count means that other
                         * further `incoming` requests or
                         * `busy-wait` requests are getting on with its job
                         * without resource initialization because
                         * main `busy-wait` request
                         * has already re-initialized it at here
                         */
                        data->outgoing_requests.fetch_add(1);
                        data->pending_requests.fetch_sub(1);
                        GAPI_LOG_DEBUG(nullptr, "REcharged, data: " << data);
                        GAPI_Assert(ptr->Y && (ptr->UV || (ptr->U && ptr->V)) &&
                                    "on_lock: data must be correct after subsequent reinit");
                        return MFX_ERR_NONE;
                    }
                }
            }
        }


        // All non main requests became `outgoing` and look at on initialized resource
        data->outgoing_requests++;

        // Each `busy-wait` request are not busy-wait now
        data->pending_requests.fetch_sub(1);
    }

    GAPI_Assert(ptr->Y && (ptr->UV || (ptr->U && ptr->V)) &&
                "on_lock: data must exist for charging `outgoing_requests`");
    return MFX_ERR_NONE;
}

mfxStatus VPLDX11AccelerationPolicy::on_unlock(mfxMemId mid, mfxFrameData *ptr) {

    allocation_record::AllocationId data = reinterpret_cast<allocation_record::AllocationId>(mid);
    if (!data) {
        return MFX_ERR_LOCK_MEMORY;
    }

    GAPI_LOG_DEBUG(nullptr, "texture: " << data->get_texture() << ", sub id: " << data->get_subresource());

    /*
     * Each released `outgoing` request checks out to doesn't use resource anymore.
     * The last `outgoing` request becomes main `outgoing` request and
     * must deinitialize resource if no `incoming` or `busy-wait` requests
     * are waiting for it
     */
    size_t thread_id = data->outgoing_requests.fetch_sub(1);
    if (thread_id == 1) {
        /*
         * Make sure that no another `incoming` (including `busy-wait)
         * exists.
         * But beforehand its must make sure that no `incoming` or `pending`
         * requests are exist.
         *
         * The main `outgoing` request is an one of `incoming` request
         * (it is the oldest one in the current `incoming` bunch) and still
         * holds resource in initialized state (thus we compare with 1).
         * We must not deinitialize resource before decrease
         * `incoming` requests counter because
         * after it has got 0 value in `on_lock` another thread
         * will start initialize resource procedure which will get conflict
         * with current deinitialize procedure
         *
         * From this point, all `on_lock` request in another thread would
         * become `busy-wait` without reaching main `incoming` state (CASE 2)
         * */
        if (data->incoming_requests.load() == 1) {
            /*
            * The main `outgoing` request is ready to deinit shared resource
            * in unconflicting manner.
            *
            * This is a critical section for single thread for main `outgoing`
            * request
            *
            * CASE 2 only available in `on_lock` thread
            * */
            device_context->Unmap(data->get_staging_texture(), 0);
            if (ptr) {
                ptr->Pitch = 0;
                ptr->U = ptr->V = ptr->Y = 0;
                ptr->A = ptr->R = ptr->G = ptr->B = 0;
            }

            /*
             * Before main `outgoinq` request become released it must notify
             * subsequent `busy-wait` requests in `on_lock` in another thread
             * that main `busy-wait` must start reinit resource procedure
             * */
            data->reinit.store(true);
            GAPI_Assert(!ptr->Y && !(ptr->UV || (ptr->U && ptr->V)) &&
                        "on_unlock: data must be cleared after last `incoming_requests`");

            /*
             * Deinitialize procedure is finished and main `outgoing` request
             * (it is the oldest one in `incoming` request) must become released
             *
             * Right after when we decrease `incoming` counter
             * the condition for equality
             * `busy-wait` and `incoming` counter will become true (CASE 2 only)
             * in `on_lock` in another threads. After that
             * a main `busy-wait` request would check `reinit` condition
             * */
            data->incoming_requests.fetch_sub(1);

            GAPI_LOG_DEBUG(nullptr, "UNcharged, data: " << data);
            return MFX_ERR_NONE;
        }

        /*
         * At this point we have guarantee that new `incoming` requests
         * had became increased in `on_lock` in another thread right before
         * current thread deinitialize resource.
         *
         * So call off deinitialization procedure here
         * */
    }

    GAPI_Assert(ptr->Y && (ptr->UV || (ptr->U && ptr->V)) &&
                "on_unlock: data must exist till last `outgoing_requests`");
    data->incoming_requests.fetch_sub(1);
    return MFX_ERR_NONE;
}

mfxStatus VPLDX11AccelerationPolicy::on_get_hdl(mfxMemId mid, mfxHDL *handle) {
    allocation_record::AllocationId data = reinterpret_cast<allocation_record::AllocationId>(mid);
    if (!data) {
        return MFX_ERR_INVALID_HANDLE;
    }

    mfxHDLPair *pPair = reinterpret_cast<mfxHDLPair *>(handle);

    pPair->first  = data->get_texture();
    pPair->second = (mfxHDL)reinterpret_cast<allocation_data_t::subresource_id_t *>(data->get_subresource());

    GAPI_LOG_DEBUG(nullptr, "texture : " << pPair->first << ", sub id: " << pPair->second);
    return MFX_ERR_NONE;
}

mfxStatus VPLDX11AccelerationPolicy::on_free(mfxFrameAllocResponse *response) {
    GAPI_LOG_DEBUG(nullptr, "Allocations count before: " << allocation_table.size() <<
                            ", requested id: " << response->AllocId);

    auto table_it = allocation_table.find(response->AllocId);
    if (allocation_table.end() == table_it) {
        GAPI_LOG_WARNING(nullptr, "Cannot find allocation id: " + std::to_string(response->AllocId) +
                                   ". Total allocation size: " + std::to_string(allocation_table.size()));
        return MFX_ERR_MEMORY_ALLOC;
    }

    allocation_table.erase(table_it);
    return MFX_ERR_NONE;
}
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_ONEVPL
