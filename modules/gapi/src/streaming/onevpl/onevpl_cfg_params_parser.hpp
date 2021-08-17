// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ONEVPL_CFG_PARAM_PARSER_HPP
#define GAPI_STREAMING_ONEVPL_ONEVPL_CFG_PARAM_PARSER_HPP

#ifdef HAVE_ONEVPL
#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif // MFX_VERSION

#include <vpl/mfx.h>
#include <vpl/mfxvideo.h>

#include <map>
#include <string>

#include <opencv2/gapi/streaming/onevpl/onevpl_source.hpp>

namespace cv {
namespace gapi {
namespace wip {

template<typename ValueType>
std::vector<ValueType> get_params_from_string(const std::string& str);

template <typename ReturnType>
struct ParamCreator {
    template<typename ValueType>
    ReturnType create(const std::string& name, ValueType&& value);
};

mfxVariant cfg_param_to_mfx_variant(const oneVPL_cfg_param& value);
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ONEVPL_CFG_PARAM_PARSER_HPP
