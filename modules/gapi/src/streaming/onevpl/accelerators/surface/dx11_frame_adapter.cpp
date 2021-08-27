// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "streaming/onevpl/accelerators/surface/dx11_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "logger.hpp"

#ifdef HAVE_ONEVPL

#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif

#include <vpl/mfx.h>

namespace cv {
namespace gapi {
namespace wip {

VPLMediaFrameDX11Adapter::VPLMediaFrameDX11Adapter(std::shared_ptr<Surface> surface,
                                                   mfxFrameAllocator alloc):
    parent_surface_ptr(surface),
    allocator(alloc) {

    GAPI_Assert(parent_surface_ptr && "Surface is nullptr");
    parent_surface_ptr->obtain_lock();


    const Surface::info_t& info = parent_surface_ptr->get_info();
    const Surface::data_t& data = parent_surface_ptr->get_data();

    GAPI_LOG_DEBUG(nullptr, "surface: " << parent_surface_ptr->get_handle() <<
                            ", w: " << info.Width << ", h: " << info.Height <<
                            ", p: " << data.Pitch);
}

VPLMediaFrameDX11Adapter::~VPLMediaFrameDX11Adapter() {

    // Each VPLMediaFrameDX11Adapter releases mfx surface counter
    // The last VPLMediaFrameDX11Adapter releases shared Surface pointer
    // The last surface pointer releases workspace memory
    parent_surface_ptr->release_lock();
}

cv::GFrameDesc VPLMediaFrameDX11Adapter::meta() const {
    GFrameDesc desc;
    const Surface::info_t& info = parent_surface_ptr->get_info();
    switch(info.FourCC)
    {
        case MFX_FOURCC_I420:
            throw std::runtime_error("MediaFrame doesn't support I420 type");
            break;
        case MFX_FOURCC_NV12:
            desc.fmt = MediaFormat::NV12;
            break;
        default:
            throw std::runtime_error("MediaFrame unknown 'fmt' type: " + std::to_string(info.FourCC));
    }

    desc.size = cv::Size{info.Width, info.Height};
    return desc;
}

MediaFrame::View VPLMediaFrameDX11Adapter::access(MediaFrame::Access) {

    Surface::data_t& data = parent_surface_ptr->get_data();
    const Surface::info_t& info = parent_surface_ptr->get_info();

    GAPI_LOG_DEBUG(nullptr, "START lock frame in surface: " << parent_surface_ptr->get_handle());
    mfxStatus sts = allocator.Lock(allocator.pthis, data.MemId, &data);
    if (sts != MFX_ERR_NONE) {
        throw std::runtime_error("Cannot lock frame");
    }
    GAPI_LOG_DEBUG(nullptr, "FINISH lock frame in surface: " << parent_surface_ptr->get_handle());
    using stride_t = typename cv::MediaFrame::View::Strides::value_type;
    GAPI_Assert(data.Pitch >= 0 && "Pitch is less than 0");

    stride_t pitch = static_cast<stride_t>(data.Pitch);


    //TODO
    auto parent_surface_ptr_copy = parent_surface_ptr;
    mfxFrameAllocator allocator_copy = allocator;
    switch(info.FourCC) {
        case MFX_FOURCC_I420:
        {
            cv::MediaFrame::View::Ptrs pp = {
                data.Y,
                data.U,
                data.V,
                nullptr
                };
            cv::MediaFrame::View::Strides ss = {
                    pitch,
                    pitch / 2,
                    pitch / 2, 0u
                };
            return cv::MediaFrame::View(std::move(pp), std::move(ss), [allocator_copy, parent_surface_ptr_copy] () {
                parent_surface_ptr_copy->obtain_lock();

                auto& data = parent_surface_ptr_copy->get_data();
                GAPI_LOG_DEBUG(nullptr, "START unlock frame in surface: " << parent_surface_ptr_copy->get_handle());
                allocator_copy.Unlock(allocator_copy.pthis, data.MemId, &data);
                GAPI_LOG_DEBUG(nullptr, "FINISH unlock frame in surface: " << parent_surface_ptr_copy->get_handle());

                parent_surface_ptr_copy->release_lock();
            });
        }
        case MFX_FOURCC_NV12:
        {
            cv::MediaFrame::View::Ptrs pp = {
                data.Y,
                data.UV, nullptr, nullptr
                };
            cv::MediaFrame::View::Strides ss = {
                    pitch,
                    pitch, 0u, 0u
                };
            return cv::MediaFrame::View(std::move(pp), std::move(ss), [allocator_copy, parent_surface_ptr_copy] () {
                parent_surface_ptr_copy->obtain_lock();

                auto& data = parent_surface_ptr_copy->get_data();
                GAPI_LOG_DEBUG(nullptr, "START unlock frame in surface: " << parent_surface_ptr_copy->get_handle());
                allocator_copy.Unlock(allocator_copy.pthis, data.MemId, &data);
                GAPI_LOG_DEBUG(nullptr, "FINISH unlock frame in surface: " << parent_surface_ptr_copy->get_handle());

                parent_surface_ptr_copy->release_lock();
            });
        }
            break;
        default:
            throw std::runtime_error("MediaFrame unknown 'fmt' type: " + std::to_string(info.FourCC));
    }
}

cv::util::any VPLMediaFrameDX11Adapter::blobParams() const {
    GAPI_Assert("VPLMediaFrameDX11Adapter::blobParams() is not implemented");
    return {};
}

void VPLMediaFrameDX11Adapter::serialize(cv::gapi::s11n::IOStream&) {
    GAPI_Assert("VPLMediaFrameDX11Adapter::serialize() is not implemented");
}

void VPLMediaFrameDX11Adapter::deserialize(cv::gapi::s11n::IIStream&) {
    GAPI_Assert("VPLMediaFrameDX11Adapter::deserialize() is not implemented");
}

DXGI_FORMAT VPLMediaFrameDX11Adapter::get_dx11_color_format(uint32_t mfx_fourcc) {
    switch (mfx_fourcc) {
        case MFX_FOURCC_NV12:
            return DXGI_FORMAT_NV12;

        case MFX_FOURCC_YUY2:
            return DXGI_FORMAT_YUY2;

        case MFX_FOURCC_RGB4:
            return DXGI_FORMAT_B8G8R8A8_UNORM;

        case MFX_FOURCC_P8:
        case MFX_FOURCC_P8_TEXTURE:
            return DXGI_FORMAT_P8;

        case MFX_FOURCC_ARGB16:
        case MFX_FOURCC_ABGR16:
            return DXGI_FORMAT_R16G16B16A16_UNORM;

        case MFX_FOURCC_P010:
            return DXGI_FORMAT_P010;

        case MFX_FOURCC_A2RGB10:
            return DXGI_FORMAT_R10G10B10A2_UNORM;

        case DXGI_FORMAT_AYUV:
        case MFX_FOURCC_AYUV:
            return DXGI_FORMAT_AYUV;

        default:
            return DXGI_FORMAT_UNKNOWN;
    }
}
} // namespace wip
} // namespace gapi
} // namespace cv
#endif HAVE_ONEVPL
