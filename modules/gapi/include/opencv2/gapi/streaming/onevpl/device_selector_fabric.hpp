// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_DEVICE_SELECTOR_FABRIC_HPP
#define GAPI_STREAMING_ONEVPL_DEVICE_SELECTOR_FABRIC_HPP

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include <opencv2/gapi/streaming/onevpl/source.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

GAPI_EXPORTS IDeviceSelector::Ptr
createCfgParamDeviceSelector(const CfgParams& param = {});

GAPI_EXPORTS IDeviceSelector::Ptr
createCfgParamDeviceSelector(Device::Ptr device_ptr, const std::string& device_id,
                             Context::Ptr ctx_ptr, const CfgParams& param);
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_DEVICE_SELECTOR_FABRIC_HPP
