// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>

#include "streaming/onevpl/cfg_param_device_selector.hpp"
#include "streaming/onevpl/onevpl_cfg_params_parser.hpp"
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

CfgParamDeviceSelector::CfgParamDeviceSelector(const onevpl_params_container_t& cfg_params) :
    IDeviceSelector(),
    suggested_device(IDeviceSelector::create<Device>(nullptr, AccelType::HOST)),
    suggested_context(IDeviceSelector::create<Context>(nullptr, AccelType::HOST)) {

    auto accel_mode_it =
        std::find_if(cfg_params.begin(), cfg_params.end(), [] (const oneVPL_cfg_param& value) {
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

            suggested_device = IDeviceSelector::create<Device>(hw_handle, AccelType::DX11);
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
                                               Context::Ptr ctx_ptr,
                                               const onevpl_params_container_t& cfg_params) :
    IDeviceSelector(),
    suggested_device(IDeviceSelector::create<Device>(nullptr, AccelType::HOST)),
    suggested_context(IDeviceSelector::create<Context>(nullptr, AccelType::HOST)) {
    auto accel_mode_it =
        std::find_if(cfg_params.begin(), cfg_params.end(), [] (const oneVPL_cfg_param& value) {
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
            suggested_device = IDeviceSelector::create<Device>(device_ptr, AccelType::DX11);
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

    GAPI_LOG_INFO(nullptr, "release device: " << suggested_device.get_ptr());
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

CfgParamDeviceSelector::DeviceScoreTable CfgParamDeviceSelector::select_spare_devices() const {
    return {};
}

Context CfgParamDeviceSelector::select_context(const DeviceScoreTable& selected_devices) {
    GAPI_Assert(selected_devices.size() == 1 && "Implementation must use only single device");
    GAPI_Assert(selected_devices.begin()->second.get_ptr() == suggested_device.get_ptr() &&
                   "Implementation must use suggested device ptr");
    GAPI_Assert(selected_devices.begin()->second.get_type() == suggested_device.get_type() &&
                   "Implementation must use suggested device type");
    return suggested_context;
}

Context CfgParamDeviceSelector::get_last_context() const {
    return suggested_context;
}
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_ONEVPL
