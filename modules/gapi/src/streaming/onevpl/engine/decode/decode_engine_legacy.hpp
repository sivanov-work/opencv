// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONVPL_ENGINE_DECODE_DECODE_ENGINE_ASYNC_HPP
#define GAPI_STREAMING_ONVPL_ENGINE_DECODE_DECODE_ENGINE_ASYNC_HPP
#include <stdio.h>
#include <memory>

#include "streaming/onevpl/engine/processing_engine_base.hpp"

#ifdef HAVE_ONEVPL
#if (MFX_VERSION >= 2000)
    #include <vpl/mfxdispatcher.h>
#endif
#include <vpl/mfx.h>

namespace cv {
namespace gapi {
namespace wip {


class LegacyDecodeSessionAsync;
struct DecoderParams;
struct IDataProvider;
struct VPLAccelerationPolicy;

class VPLLegacyDecodeEngineAsync : public ProcessingEngineBase {
public:

    VPLLegacyDecodeEngineAsync(std::unique_ptr<VPLAccelerationPolicy>&& accel);
    void initialize_session(mfxSession mfx_session, DecoderParams&& decoder_param,
                            std::shared_ptr<IDataProvider> provider) override;

private:
    ExecutionStatus execute_op(operation_t& op, EngineSession& sess) override;
    ExecutionStatus process_error(mfxStatus status, LegacyDecodeSessionAsync& sess);

    void on_frame_ready(LegacyDecodeSessionAsync& sess, mfxFrameSurface1* ready_surface);
};
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONVPL_ENGINE_DECODE_DECODE_ENGINE_ASYNC_HPP
