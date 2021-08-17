// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation


#include "../perf_precomp.hpp"
#include "../../test/common/gapi_tests_common.hpp"
#include <opencv2/gapi/streaming/onevpl/onevpl_source.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

namespace opencv_test
{
using namespace perf;

const std::string files[] = {
    "highgui/video/big_buck_bunny.avi",
    "highgui/video/big_buck_bunny.mov",
    "highgui/video/big_buck_bunny.mp4",
    "highgui/video/big_buck_bunny.wmv"
};

const std::string codec[] = {
    "MFX_CODEC_HEVC",
    "MFX_CODEC_AVC"
};

using source_t = std::string;
using codec_t = std::string;
using source_description_t = std::tuple<source_t, codec_t>;

class OneVPLSourcePerfTest : public TestPerfParams<source_description_t> {};
class VideoCapSourcePerfTest : public TestPerfParams<source_t> {};

PERF_TEST_P_(OneVPLSourcePerfTest, TestPerformance)
{
  using namespace cv::gapi::wip;

  const auto params = GetParam();
  const source_t& src = get<0>(params);
  codec_t type = get<1>(params);

  std::vector<oneVPL_cfg_param> cfg_params {
      oneVPL_cfg_param::create<std::string>("mfxImplDescription.Impl", "MFX_IMPL_TYPE_HARDWARE"),
      oneVPL_cfg_param::create("mfxImplDescription.mfxDecoderDescription.decoder.CodecID", type),
  };

  auto source_ptr = make_vpl_src(src, cfg_params);
  Data out;
  TEST_CYCLE()
  {
      source_ptr->pull(out);
  }

  SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(VideoCapSourcePerfTest, TestPerformance)
{
  using namespace cv::gapi::wip;

  const source_t& src = GetParam();
  auto source_ptr = make_src<GCaptureSource>(src);
  Data out;
  TEST_CYCLE()
  {
      source_ptr->pull(out);
  }

  SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(OneVPLSourcePerfTest, OneVPLSourcePerfTest,
                        Values(source_description_t(files[0], codec[0])));

INSTANTIATE_TEST_CASE_P(VideoCapSourcePerfTest, VideoCapSourcePerfTest,
                        Values(files[0]));
} // namespace opencv_test
