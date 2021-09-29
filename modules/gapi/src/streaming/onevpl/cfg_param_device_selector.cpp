// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>
#include <opencv2/gapi/util/variant.hpp>

#include "streaming/onevpl/cfg_param_device_selector.hpp"
#include "logger.hpp"

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
#pragma comment(lib,"d3d11.lib")

// get rid of generate macro max/min/etc from DX side
#define D3D11_NO_HELPERS
#define NOMINMAX
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
namespace onevpl {

// TODO Will be changed on generic function from `onevpl_param_parser` as soons as feature merges
static mfxVariant cfg_param_to_mfx_variant(const CfgParam& accel_param) {
    mfxVariant ret;
    const CfgParam::value_t& accel_val = accel_param.get_value();
    if (!cv::util::holds_alternative<std::string>(accel_val)) {
        // expected string or uint32_t as value
        if (!cv::util::holds_alternative<uint32_t>(accel_val)) {
            throw std::logic_error("Incorrect value type of \"mfxImplDescription.AccelerationMode\" "
                                   " std::string is expected" );
        }
        ret.Type = MFX_VARIANT_TYPE_U32;
        ret.Data.U32 = cv::util::get<uint32_t>(accel_val);
        return ret;
    }

    const std::string& accel_val_str = cv::util::get<std::string>(accel_val);
    ret.Type = MFX_VARIANT_TYPE_U32;
    if (accel_val_str == "MFX_ACCEL_MODE_NA") {
        ret.Data.U32 = MFX_ACCEL_MODE_NA;
    } else if (accel_val_str == "MFX_ACCEL_MODE_VIA_D3D9") {
        ret.Data.U32 = MFX_ACCEL_MODE_VIA_D3D9;
    } else if (accel_val_str == "MFX_ACCEL_MODE_VIA_D3D11") {
        ret.Data.U32 = MFX_ACCEL_MODE_VIA_D3D11;
    } else if (accel_val_str == "MFX_ACCEL_MODE_VIA_VAAPI") {
        ret.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI;
    } else if (accel_val_str == "MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET") {
        ret.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET;
    } else if (accel_val_str == "MFX_ACCEL_MODE_VIA_VAAPI_GLX") {
        ret.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI_GLX;
    } else if (accel_val_str == "MFX_ACCEL_MODE_VIA_VAAPI_X11") {
        ret.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI_X11;
    } else if (accel_val_str == "MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND") {
        ret.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND;
    } else if (accel_val_str == "MFX_ACCEL_MODE_VIA_HDDLUNITE") {
        ret.Data.U32 = MFX_ACCEL_MODE_VIA_HDDLUNITE;
    }
    return ret;
}

CfgParamDeviceSelector::CfgParamDeviceSelector(const CfgParams& cfg_params) :
    IDeviceSelector(),
    suggested_device(IDeviceSelector::create<Device>(nullptr, "CPU", AccelType::HOST)),
    suggested_context(IDeviceSelector::create<Context>(nullptr, AccelType::HOST)) {

    auto accel_mode_it =
        std::find_if(cfg_params.begin(), cfg_params.end(), [] (const CfgParam& value) {
            return value.get_name() == "mfxImplDescription.AccelerationMode";
        });
    if (accel_mode_it == cfg_params.end())
    {
        GAPI_LOG_DEBUG(nullptr, "No HW Accel requested. Use default CPU");
        return;
    }

    GAPI_LOG_DEBUG(nullptr, "Add HW acceleration support");
    mfxVariant accel_mode = cfg_param_to_mfx_variant(*accel_mode_it);

    switch(accel_mode.Data.U32) {
        case MFX_ACCEL_MODE_VIA_D3D11: {
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
            ID3D11Device *hw_handle = nullptr;
            ID3D11DeviceContext* device_context = nullptr;

            //Create device
            UINT creationFlags = 0;//D3D11_CREATE_DEVICE_BGRA_SUPPORT;

#if defined _DEBUG || defined CV_STATIC_ANALYSIS
            // If the project is in a debug build, enable debugging via SDK Layers with this flag.
            creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

            D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_1,
                                                  D3D_FEATURE_LEVEL_11_0,
                                                };
            D3D_FEATURE_LEVEL featureLevel;

            // Create the Direct3D 11 API device object and a corresponding context.
            HRESULT err = D3D11CreateDevice(
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
            if(FAILED(err)) {
                throw std::logic_error("Cannot create D3D11CreateDevice, error: " + std::to_string(HRESULT_CODE(err)));
            }

            // oneVPL recommendation
            {
                ID3D11Multithread       *pD11Multithread;
                device_context->QueryInterface(IID_PPV_ARGS(&pD11Multithread));
                pD11Multithread->SetMultithreadProtected(true);
                pD11Multithread->Release();
            }

            suggested_device = IDeviceSelector::create<Device>(hw_handle, "GPU", AccelType::DX11);
            suggested_context = IDeviceSelector::create<Context>(device_context, AccelType::DX11);
#else
            GAPI_LOG_WARNING(nullptr, "Unavailable \"mfxImplDescription.AccelerationMode: MFX_ACCEL_MODE_VIA_D3D11\""
                                      "was choosed for current project configuration");
            throw std::logic_error("Unsupported \"mfxImplDescription.AccelerationMode: MFX_ACCEL_MODE_VIA_D3D11\"");
#endif // HAVE_DIRECTX
#endif // HAVE_D3D11
            break;
        }
        case MFX_ACCEL_MODE_NA: {
            // nothing to do
            break;
        }
        default:
            throw std::logic_error("Unsupported \"mfxImplDescription.AccelerationMode\" requested: " +
                                   std::to_string(accel_mode.Data.U32));
            break;
    }
}

CfgParamDeviceSelector::CfgParamDeviceSelector(Device::Ptr device_ptr,
                                               const std::string& device_id,
                                               Context::Ptr ctx_ptr,
                                               const CfgParams& cfg_params) :
    IDeviceSelector(),
    suggested_device(IDeviceSelector::create<Device>(nullptr, "CPU", AccelType::HOST)),
    suggested_context(IDeviceSelector::create<Context>(nullptr, AccelType::HOST)) {
    auto accel_mode_it =
        std::find_if(cfg_params.begin(), cfg_params.end(), [] (const CfgParam& value) {
            return value.get_name() == "mfxImplDescription.AccelerationMode";
        });
    if (accel_mode_it == cfg_params.end()) {
        GAPI_LOG_WARNING(nullptr, "Cannot deternime \"device_ptr\" type. "
                         "Make sure a param \"mfxImplDescription.AccelerationMode\" "
                         "presents in configurations and has correct value according to "
                         "\"device_ptr\" type");
        throw std::logic_error("Missing \"mfxImplDescription.AccelerationMode\" param");
    }

    GAPI_LOG_DEBUG(nullptr, "Turn on HW acceleration support for device: " <<
                            device_ptr <<
                            ", context: " << ctx_ptr);
    if (!device_ptr) {
        GAPI_LOG_WARNING(nullptr, "Empty \"device_ptr\" is not allowed when "
                         "param \"mfxImplDescription.AccelerationMode\" existed");
        throw std::logic_error("Invalid param: \"device_ptr\"");
    }

     if (!ctx_ptr) {
        GAPI_LOG_WARNING(nullptr, "Empty \"ctx_ptr\" is not allowed");
        throw std::logic_error("Invalid  param: \"ctx_ptr\"");
    }
    mfxVariant accel_mode = cfg_param_to_mfx_variant(*accel_mode_it);

    switch(accel_mode.Data.U32) {
        case MFX_ACCEL_MODE_VIA_D3D11: {
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
            suggested_device = IDeviceSelector::create<Device>(device_ptr, device_id, AccelType::DX11);
            ID3D11Device* dx_device_ptr =
                reinterpret_cast<ID3D11Device*>(suggested_device.get_ptr());
            dx_device_ptr->AddRef();

            suggested_context = IDeviceSelector::create<Context>(ctx_ptr, AccelType::DX11);
            ID3D11DeviceContext* dx_ctx_ptr =
                reinterpret_cast<ID3D11DeviceContext*>(suggested_context.get_ptr());
            dx_ctx_ptr->AddRef();
#else
            GAPI_LOG_WARNING(nullptr, "Unavailable \"mfxImplDescription.AccelerationMode: MFX_ACCEL_MODE_VIA_D3D11\""
                                      "was choosed for current project configuration");
            throw std::logic_error("Unsupported \"mfxImplDescription.AccelerationMode: MFX_ACCEL_MODE_VIA_D3D11\"");
#endif // HAVE_DIRECTX
#endif // HAVE_D3D11
            break;
        }
        case MFX_ACCEL_MODE_NA: {
            GAPI_LOG_WARNING(nullptr, "Incompatible \"mfxImplDescription.AccelerationMode: MFX_ACCEL_MODE_NA\" with "
                                      "\"device_ptr\" and \"ctx_ptr\" arguments. "
                                      "You should not clarify these arguments with \"MFX_ACCEL_MODE_NA\" mode");
            throw std::logic_error("Incompatible param: MFX_ACCEL_MODE_NA");
        }
        default:
            throw std::logic_error("Unsupported \"mfxImplDescription.AccelerationMode\" requested: " +
                                   std::to_string(accel_mode.Data.U32));
            break;
    }
}

CfgParamDeviceSelector::~CfgParamDeviceSelector() {
    GAPI_LOG_INFO(nullptr, "release context: " << suggested_context.get_ptr());
    AccelType ctype = suggested_context.get_type();
    switch(ctype) {
        case AccelType::HOST:
            //nothing to do
            break;
        case AccelType::DX11: {
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
            ID3D11DeviceContext* device_ctx_ptr =
                reinterpret_cast<ID3D11DeviceContext*>(suggested_context.get_ptr());
            device_ctx_ptr->Release();
            device_ctx_ptr = nullptr;
#endif // HAVE_DIRECTX
#endif // HAVE_D3D11
            break;
        }
        default:
            break;
    }

    GAPI_LOG_INFO(nullptr, "release device by name: " <<
                           suggested_device.get_name() <<
                           ", ptr: " << suggested_device.get_ptr());
    AccelType dtype = suggested_device.get_type();
    switch(dtype) {
        case AccelType::HOST:
            //nothing to do
            break;
        case AccelType::DX11: {
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
            ID3D11Device* device_ptr = reinterpret_cast<ID3D11Device*>(suggested_device.get_ptr());
            device_ptr->Release();
            device_ptr = nullptr;
#endif // HAVE_DIRECTX
#endif // HAVE_D3D11
            break;
        }
        default:
            break;
    }
}

CfgParamDeviceSelector::DeviceScoreTable CfgParamDeviceSelector::select_devices() const {
    return {std::make_pair(Score::Max, suggested_device)};
}

CfgParamDeviceSelector::DeviceContexts CfgParamDeviceSelector::select_context() {
    return {suggested_context};
}

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#undef D3D11_NO_HELPERS
#undef NOMINMAX
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_ONEVPL
