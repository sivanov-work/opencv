// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_HPP

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/streaming/onevpl/onevpl_cfg_params.hpp>
#include <opencv2/gapi/streaming/onevpl/onevpl_data_provider_interface.hpp>

namespace cv {
namespace gapi {
namespace wip {

using onevpl_params_container_t = std::vector<oneVPL_cfg_param>;

class GAPI_EXPORTS OneVPLSource : public IStreamSource
{
public:
    struct Priv;

    OneVPLSource(const std::string& filePath,
                 const onevpl_params_container_t& cfg_params = onevpl_params_container_t{});
    OneVPLSource(std::shared_ptr<IDataProvider> source,
                 const onevpl_params_container_t& cfg_params = onevpl_params_container_t{});
    ~OneVPLSource() override;

    bool pull(cv::gapi::wip::Data& data) override;
    GMetaArg descr_of() const override;

private:
    explicit OneVPLSource(std::unique_ptr<Priv>&& impl);
    std::unique_ptr<Priv> m_priv;
};

template<class... Args>
GAPI_EXPORTS_W cv::Ptr<IStreamSource> inline make_vpl_src(Args&&... args)
{
    return make_src<OneVPLSource>(std::forward<Args>(args)...);
}

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_HPP
