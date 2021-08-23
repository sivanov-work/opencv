// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_DX11_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_DX11_HPP

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS
//TODO
//#define  CPU_ACCEL_ADAPTER

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"

#ifdef CPU_ACCEL_ADAPTER
#include "streaming/onevpl/accelerators/accel_policy_cpu.hpp"
#endif

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
    #define D3D11_NO_HELPERS
    #define NOMINMAX
    #include <d3d11.h>
    #include <codecvt>
    #include "opencv2/core/directx.hpp"
    #ifdef HAVE_OPENCL
        #include <CL/cl_d3d11.h>
    #endif

namespace cv {
namespace gapi {
namespace wip {

struct VPLDX11AccelerationPolicy final: public VPLAccelerationPolicy
{
    // GAPI_EXPORTS for tests
    GAPI_EXPORTS VPLDX11AccelerationPolicy();
    GAPI_EXPORTS ~VPLDX11AccelerationPolicy();

    GAPI_EXPORTS void init(session_t session) override;
    GAPI_EXPORTS void deinit(session_t session) override;
    GAPI_EXPORTS pool_key_t create_surface_pool(size_t pool_size, size_t surface_size_bytes, surface_ptr_ctr_t creator) override;
    GAPI_EXPORTS surface_weak_ptr_t get_free_surface(pool_key_t key) override;
    GAPI_EXPORTS size_t get_free_surface_count(pool_key_t key) const override;
    GAPI_EXPORTS size_t get_surface_count(pool_key_t key) const override;

    GAPI_EXPORTS cv::MediaFrame::AdapterPtr create_frame_adapter(pool_key_t key,
                                                                 mfxFrameSurface1* surface) override;

private:
    ID3D11Device *hw_handle;

    mfxFrameAllocator allocator;
    static mfxStatus MFX_CDECL alloc_cb(mfxHDL pthis,
                                        mfxFrameAllocRequest *request,
                                        mfxFrameAllocResponse *response);
    static mfxStatus MFX_CDECL lock_cb(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL unlock_cb(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL get_hdl_cb(mfxHDL pthis, mfxMemId mid, mfxHDL *handle);
    static mfxStatus MFX_CDECL free_cb(mfxHDL pthis, mfxFrameAllocResponse *response);

    virtual mfxStatus on_alloc(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response);
    virtual mfxStatus on_lock(mfxMemId mid, mfxFrameData *ptr);
    virtual mfxStatus on_unlock(mfxMemId mid, mfxFrameData *ptr);
    virtual mfxStatus on_get_hdl(mfxMemId mid, mfxHDL *handle);
    virtual mfxStatus on_free(mfxFrameAllocResponse *response);

#ifdef CPU_ACCEL_ADAPTER
    std::unique_ptr<VPLCPUAccelerationPolicy> adapter;
#endif
};
} // namespace wip
} // namespace gapi
} // namespace cv
#undef NOMINMAX
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_DX11_HPP
