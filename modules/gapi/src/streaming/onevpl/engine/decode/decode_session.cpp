// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <chrono>
#include <exception>

#include "streaming/onevpl/engine/decode/decode_session.hpp"
#include "streaming/onevpl/engine/decode/decode_engine_legacy.hpp"
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/onevpl_utils.hpp"

#include "logger.hpp"
namespace cv {
namespace gapi {
namespace wip {
LegacyDecodeSessionAsync::LegacyDecodeSessionAsync(mfxSession sess,
                                         DecoderParams&& decoder_param,
                                         std::shared_ptr<IDataProvider> provider) :
    EngineSession(sess, std::move(decoder_param.stream)),
    mfx_decoder_param(std::move(decoder_param.param)),
    data_provider(std::move(provider)),
    procesing_surface_ptr(),
    sync_queue(),
    decoded_frames_count()
{
}

LegacyDecodeSessionAsync::~LegacyDecodeSessionAsync()
{
    GAPI_LOG_INFO(nullptr, "Close Decode for session: " << session);
    MFXVideoDECODE_Close(session);
}

void LegacyDecodeSessionAsync::swap_surface(VPLLegacyDecodeEngineAsync& engine) {
    VPLAccelerationPolicy* acceleration_policy = engine.get_accel();
    GAPI_Assert(acceleration_policy && "Empty acceleration_policy");
    try {
        auto cand = acceleration_policy->get_free_surface(decoder_pool_id).lock();

        GAPI_LOG_DEBUG(nullptr, "[" << session << "] swap surface"
                                ", old: " << (!procesing_surface_ptr.expired()
                                              ? procesing_surface_ptr.lock()->get_handle()
                                              : nullptr) <<
                                ", new: "<< cand->get_handle());

        procesing_surface_ptr = cand;
    } catch (const std::exception& ex) {
        GAPI_LOG_WARNING(nullptr, "[" << session << "] error: " << ex.what() <<
                                   "Abort");
    }
}

void LegacyDecodeSessionAsync::init_surface_pool(VPLAccelerationPolicy::pool_key_t key) {
    GAPI_Assert(key && "Init decode pull with empty key");
    decoder_pool_id = key;
}

Data::Meta LegacyDecodeSessionAsync::generate_frame_meta() {
    const auto now = std::chrono::system_clock::now();
    const auto dur = std::chrono::duration_cast<std::chrono::microseconds>
                (now.time_since_epoch());
    Data::Meta meta {
                        {cv::gapi::streaming::meta_tag::timestamp, int64_t{dur.count()} },
                        {cv::gapi::streaming::meta_tag::seq_id, int64_t{decoded_frames_count++}}
                    };
    return meta;
}

const mfxVideoParam& LegacyDecodeSessionAsync::get_video_param() const {
    return mfx_decoder_param;
}
} // namespace wip
} // namespace gapi
} // namespace cv
