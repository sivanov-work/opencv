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
    "highgui/video/big_buck_bunny.h265",
    "highgui/video/Putin.raw",
    "highgui/video/big_buck_bunny.h264",
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
  source_t src = findDataFile(get<0>(params));
  codec_t type = get<1>(params);

  std::vector<oneVPL_cfg_param> cfg_params {
      oneVPL_cfg_param::create<std::string>("mfxImplDescription.Impl", "MFX_IMPL_TYPE_HARDWARE"),
      oneVPL_cfg_param::create("mfxImplDescription.mfxDecoderDescription.decoder.CodecID", type),
  };

  auto source_ptr = make_vpl_src(src, cfg_params);
  //auto def_source_ptr = make_src<GCaptureSource>(src);
  Data out;
  //Data def_out;
  TEST_CYCLE()
  {
      source_ptr->pull(out);
      //def_source_ptr->pull(def_out);
  }

  SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(VideoCapSourcePerfTest, TestPerformance)
{
  using namespace cv::gapi::wip;

  source_t src = findDataFile(GetParam());
  auto source_ptr = make_src<GCaptureSource>(src);
  Data out;
  TEST_CYCLE()
  {
      source_ptr->pull(out);
  }

  SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(Streaming, OneVPLSourcePerfTest,
                        Values(source_description_t(files[0], codec[0]),
                               source_description_t(files[1], codec[0]),
                               source_description_t(files[2], codec[1])));

INSTANTIATE_TEST_CASE_P(Streaming, VideoCapSourcePerfTest,
                        Values(files[0],
                               files[1],
                               files[2]));
} // namespace opencv_test
