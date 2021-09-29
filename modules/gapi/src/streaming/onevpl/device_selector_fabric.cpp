// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/streaming/onevpl/device_selector_fabric.hpp>

#include "streaming/onevpl/cfg_param_device_selector.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

IDeviceSelector::Ptr createCfgParamDeviceSelector(const CfgParams& param) {
    return std::shared_ptr<CfgParamDeviceSelector>(new CfgParamDeviceSelector(param));
}

IDeviceSelector::Ptr createCfgParamDeviceSelector(Device::Ptr device_ptr,
                                                  const std::string& device_id,
                                                  Context::Ptr ctx_ptr,
                                                  const CfgParams& param) {
    return std::shared_ptr<CfgParamDeviceSelector>(
                        new CfgParamDeviceSelector(device_ptr, device_id,
                                                   ctx_ptr, param));
}

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
