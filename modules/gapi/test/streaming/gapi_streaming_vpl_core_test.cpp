// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_streaming_tests_common.hpp"

#include <chrono>
#include <future>

#include <opencv2/gapi/media.hpp>
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>

#include <opencv2/gapi/ocl/core.hpp>
#include <opencv2/gapi/ocl/imgproc.hpp>

#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/gapi/streaming/format.hpp>

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/accelerators/surface/cpu_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/accel_policy_cpu.hpp"
#include "streaming/onevpl/accelerators/accel_policy_dx11.hpp"
#include <opencv2/gapi/streaming/onevpl/onevpl_data_provider_interface.hpp>
#include "streaming/onevpl/engine/processing_engine_base.hpp"
#include "streaming/onevpl/engine/engine_session.hpp"
#include "streaming/onevpl/cfg_param_device_selector.hpp"

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

#endif
#endif

namespace opencv_test
{
namespace
{

struct EmptyDataProvider : public cv::gapi::wip::IDataProvider {

    size_t provide_data(size_t, void*) override {
        return 0;
    }
    bool empty() const override {
        return true;
    }
};

struct TestProcessingSession : public cv::gapi::wip::EngineSession {
    TestProcessingSession(mfxSession mfx_session) :
        EngineSession(mfx_session, {}) {
    }

    const mfxVideoParam& get_video_param() const override {
        static mfxVideoParam empty;
        return empty;
    }
};

struct TestProcessingEngine: public cv::gapi::wip::ProcessingEngineBase {

    size_t pipeline_stage_num = 0;

    TestProcessingEngine(std::unique_ptr<cv::gapi::wip::VPLAccelerationPolicy>&& accel) :
        cv::gapi::wip::ProcessingEngineBase(std::move(accel)) {
        using cv::gapi::wip::EngineSession;
        create_pipeline(
            // 0)
            [this] (EngineSession&) -> ExecutionStatus
            {
                pipeline_stage_num = 0;
                return ExecutionStatus::Continue;
            },
            // 1)
            [this] (EngineSession&) -> ExecutionStatus
            {
                pipeline_stage_num = 1;
                return ExecutionStatus::Continue;
            },
            // 2)
            [this] (EngineSession&) -> ExecutionStatus
            {
                pipeline_stage_num = 2;
                return ExecutionStatus::Continue;
            },
            // 3)
            [this] (EngineSession&) -> ExecutionStatus
            {
                pipeline_stage_num = 3;
                ready_frames.emplace(cv::MediaFrame());
                return ExecutionStatus::Processed;
            }
        );
    }

    std::shared_ptr<cv::gapi::wip::EngineSession>
            initialize_session(mfxSession mfx_session,
                               const std::vector<cv::gapi::wip::oneVPL_cfg_param>&,
                               std::shared_ptr<cv::gapi::wip::IDataProvider>) override {

        return register_session<TestProcessingSession>(mfx_session);
    }
};

void test_eq(const typename cv::gapi::wip::IDeviceSelector::DeviceScoreTable::value_type &scored_device,
             cv::gapi::wip::IDeviceSelector::Score expected_score,
             cv::gapi::wip::AccelType expected_type,
             cv::gapi::wip::Device::Ptr expected_ptr) {
    EXPECT_EQ(std::get<0>(scored_device), expected_score);
    EXPECT_EQ(std::get<1>(scored_device).get_type(), expected_type);
    EXPECT_EQ(std::get<1>(scored_device).get_ptr(), expected_ptr);
}

void test_host_dev_eq(const typename cv::gapi::wip::IDeviceSelector::DeviceScoreTable::value_type &scored_device,
                      cv::gapi::wip::IDeviceSelector::Score expected_score) {
    test_eq(scored_device, expected_score,
            cv::gapi::wip::AccelType::HOST, nullptr);
}

TEST(OneVPL_Source_Surface, InitSurface)
{
    using namespace cv::gapi::wip;

    // create raw MFX handle
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
    memset(handle.get(), 0, sizeof(mfxFrameSurface1));
    mfxFrameSurface1 *mfx_core_handle = handle.get();

    // create preallocate surface memory: empty for test
    std::shared_ptr<void> associated_memory {};
    auto surf = Surface::create_surface(std::move(handle), associated_memory);

    // check self consistency
    EXPECT_EQ(reinterpret_cast<void*>(surf->get_handle()),
              reinterpret_cast<void*>(mfx_core_handle));
    EXPECT_EQ(surf->get_locks_count(), 0);
    EXPECT_EQ(surf->obtain_lock(), 0);
    EXPECT_EQ(surf->get_locks_count(), 1);
    EXPECT_EQ(surf->release_lock(), 1);
    EXPECT_EQ(surf->get_locks_count(), 0);
}

TEST(OneVPL_Source_Surface, ConcurrentLock)
{
    using namespace cv::gapi::wip;

    // create raw MFX handle
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
    memset(handle.get(), 0, sizeof(mfxFrameSurface1));

    // create preallocate surface memory: empty for test
    std::shared_ptr<void> associated_memory {};
    auto surf = Surface::create_surface(std::move(handle), associated_memory);

    // check self consistency
    EXPECT_EQ(surf->get_locks_count(), 0);

    // MFX internal limitation: do not exceede U16 range
    // so I16 is using here
    int16_t lock_counter = std::numeric_limits<int16_t>::max() - 1;
    std::promise<void> barrier;
    std::future<void> sync = barrier.get_future();


    std::thread worker_thread([&barrier, surf, lock_counter] () {
        barrier.set_value();

        // concurrent lock
        for (int16_t i = 0; i < lock_counter; i ++) {
            surf->obtain_lock();
        }
    });
    sync.wait();

    // concurrent lock
    for (int16_t i = 0; i < lock_counter; i ++) {
            surf->obtain_lock();
    }

    worker_thread.join();
    EXPECT_EQ(surf->get_locks_count(), lock_counter * 2);
}

TEST(OneVPL_Source_Surface, MemoryLifeTime)
{
    using namespace cv::gapi::wip;

    // create preallocate surface memory
    std::unique_ptr<char> preallocated_memory_ptr(new char);
    std::shared_ptr<void> associated_memory (preallocated_memory_ptr.get(),
                                             [&preallocated_memory_ptr] (void* ptr) {
                                                    EXPECT_TRUE(preallocated_memory_ptr);
                                                    EXPECT_EQ(preallocated_memory_ptr.get(), ptr);
                                                    preallocated_memory_ptr.reset();
                                            });

    // generate surfaces
    constexpr size_t surface_num = 10000;
    std::vector<std::shared_ptr<Surface>> surfaces(surface_num);
    std::generate(surfaces.begin(), surfaces.end(), [surface_num, associated_memory](){
        std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
        memset(handle.get(), 0, sizeof(mfxFrameSurface1));
        return Surface::create_surface(std::move(handle), associated_memory);
    });

    // destroy surfaces
    {
        std::thread deleter_thread([&surfaces]() {
            surfaces.clear();
        });
        deleter_thread.join();
    }

    // workspace memory must be alive
    EXPECT_EQ(surfaces.size(), 0);
    EXPECT_TRUE(associated_memory != nullptr);
    EXPECT_TRUE(preallocated_memory_ptr.get() != nullptr);

    // generate surfaces again + 1
    constexpr size_t surface_num_plus_one = 10001;
    surfaces.resize(surface_num_plus_one);
    std::generate(surfaces.begin(), surfaces.end(), [surface_num_plus_one, associated_memory](){
        std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
        memset(handle.get(), 0, sizeof(mfxFrameSurface1));
        return Surface::create_surface(std::move(handle), associated_memory);
    });

    // remember one surface
    std::shared_ptr<Surface> last_surface = surfaces.back();

    // destroy another surfaces
    surfaces.clear();

    // destroy associated_memory
    associated_memory.reset();

    // workspace memory must be still alive
    EXPECT_EQ(surfaces.size(), 0);
    EXPECT_TRUE(associated_memory == nullptr);
    EXPECT_TRUE(preallocated_memory_ptr.get() != nullptr);

    // destroy last surface
    last_surface.reset();

    // workspace memory must be freed
    EXPECT_TRUE(preallocated_memory_ptr.get() == nullptr);
}

TEST(OneVPL_Source_CPU_FrameAdapter, InitFrameAdapter)
{
    using namespace cv::gapi::wip;

    // create raw MFX handle
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
    memset(handle.get(), 0, sizeof(mfxFrameSurface1));

    // create preallocate surface memory: empty for test
    std::shared_ptr<void> associated_memory {};
    auto surf = Surface::create_surface(std::move(handle), associated_memory);

    // check consistency
    EXPECT_EQ(surf->get_locks_count(), 0);

    {
        VPLMediaFrameCPUAdapter adapter(surf);
        EXPECT_EQ(surf->get_locks_count(), 1);
    }
    EXPECT_EQ(surf->get_locks_count(), 0);
}

cv::gapi::wip::surface_ptr_t create_test_surface(std::shared_ptr<void> out_buf_ptr,
                                  size_t, size_t) {
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
    memset(handle.get(), 0, sizeof(mfxFrameSurface1));

    return cv::gapi::wip::Surface::create_surface(std::move(handle), out_buf_ptr);
}

TEST(OneVPL_Source_CPU_Accelerator, InitDestroy)
{
    using cv::gapi::wip::VPLCPUAccelerationPolicy;
    using cv::gapi::wip::VPLAccelerationPolicy;

    auto acceleration_policy = std::make_shared<VPLCPUAccelerationPolicy>();

    size_t surface_count = 10;
    size_t surface_size_bytes = 1024;
    size_t pool_count = 3;
    std::vector<VPLAccelerationPolicy::pool_key_t> pool_export_keys;
    pool_export_keys.reserve(pool_count);

    // create several pools
    for (size_t i = 0; i < pool_count; i++)
    {
        VPLAccelerationPolicy::pool_key_t key =
                acceleration_policy->create_surface_pool(surface_count,
                                                         surface_size_bytes,
                                                         create_test_surface);
        // check consistency
        EXPECT_EQ(acceleration_policy->get_surface_count(key), surface_count);
        EXPECT_EQ(acceleration_policy->get_free_surface_count(key), surface_count);

        pool_export_keys.push_back(key);
    }

    EXPECT_NO_THROW(acceleration_policy.reset());
}

TEST(OneVPL_Source_CPU_Accelerator, PoolProduceConsume)
{
    using cv::gapi::wip::VPLCPUAccelerationPolicy;
    using cv::gapi::wip::VPLAccelerationPolicy;
    using cv::gapi::wip::Surface;

    auto acceleration_policy = std::make_shared<VPLCPUAccelerationPolicy>();

    size_t surface_count = 10;
    size_t surface_size_bytes = 1024;

    VPLAccelerationPolicy::pool_key_t key =
                acceleration_policy->create_surface_pool(surface_count,
                                                         surface_size_bytes,
                                                         create_test_surface);
    // check consistency
    EXPECT_EQ(acceleration_policy->get_surface_count(key), surface_count);
    EXPECT_EQ(acceleration_policy->get_free_surface_count(key), surface_count);

    // consume available surfaces
    std::vector<std::shared_ptr<Surface>> surfaces;
    surfaces.reserve(surface_count);
    for (size_t i = 0; i < surface_count; i++) {
        std::shared_ptr<Surface> surf = acceleration_policy->get_free_surface(key).lock();
        EXPECT_TRUE(surf.get() != nullptr);
        EXPECT_EQ(surf->obtain_lock(), 0);
        surfaces.push_back(std::move(surf));
    }

    // check consistency (no free surfaces)
    EXPECT_EQ(acceleration_policy->get_surface_count(key), surface_count);
    EXPECT_EQ(acceleration_policy->get_free_surface_count(key), 0);

    // fail consume non-free surfaces
    for (size_t i = 0; i < surface_count; i++) {
        EXPECT_THROW(acceleration_policy->get_free_surface(key), std::runtime_error);
    }

    // release surfaces
    for (auto& surf : surfaces) {
        EXPECT_EQ(surf->release_lock(), 1);
    }
    surfaces.clear();

    // check consistency
    EXPECT_EQ(acceleration_policy->get_surface_count(key), surface_count);
    EXPECT_EQ(acceleration_policy->get_free_surface_count(key), surface_count);

    //check availability after release
    for (size_t i = 0; i < surface_count; i++) {
        std::shared_ptr<Surface> surf = acceleration_policy->get_free_surface(key).lock();
        EXPECT_TRUE(surf.get() != nullptr);
        EXPECT_EQ(surf->obtain_lock(), 0);
    }
}

TEST(OneVPL_Source_CPU_Accelerator, PoolProduceConcurrentConsume)
{
    using cv::gapi::wip::VPLCPUAccelerationPolicy;
    using cv::gapi::wip::VPLAccelerationPolicy;
    using cv::gapi::wip::Surface;

    auto acceleration_policy = std::make_shared<VPLCPUAccelerationPolicy>();

    size_t surface_count = 10;
    size_t surface_size_bytes = 1024;

    VPLAccelerationPolicy::pool_key_t key =
                acceleration_policy->create_surface_pool(surface_count,
                                                         surface_size_bytes,
                                                         create_test_surface);

    // check consistency
    EXPECT_EQ(acceleration_policy->get_surface_count(key), surface_count);
    EXPECT_EQ(acceleration_policy->get_free_surface_count(key), surface_count);

    // consume available surfaces
    std::vector<std::shared_ptr<Surface>> surfaces;
    surfaces.reserve(surface_count);
    for (size_t i = 0; i < surface_count; i++) {
        std::shared_ptr<Surface> surf = acceleration_policy->get_free_surface(key).lock();
        EXPECT_TRUE(surf.get() != nullptr);
        EXPECT_EQ(surf->obtain_lock(), 0);
        surfaces.push_back(std::move(surf));
    }

    std::promise<void> launch_promise;
    std::future<void> sync = launch_promise.get_future();
    std::promise<size_t> surface_released_promise;
    std::future<size_t> released_result = surface_released_promise.get_future();
    std::thread worker_thread([&launch_promise, &surface_released_promise, &surfaces] () {
        launch_promise.set_value();

        // concurrent release surfaces
        size_t surfaces_count = surfaces.size();
        for (auto& surf : surfaces) {
            EXPECT_EQ(surf->release_lock(), 1);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        surfaces.clear();

        surface_released_promise.set_value(surfaces_count);
    });
    sync.wait();

    // check free surface concurrently
    std::future_status status;
    size_t free_surface_count = 0;
    size_t free_surface_count_prev = 0;
    do {
        status = released_result.wait_for(std::chrono::seconds(1));
        free_surface_count = acceleration_policy->get_free_surface_count(key);
        EXPECT_TRUE(free_surface_count >= free_surface_count_prev);
        free_surface_count_prev = free_surface_count;
    } while (status != std::future_status::ready);
    std::cerr<< "Ready" << std::endl;
    free_surface_count = acceleration_policy->get_free_surface_count(key);
    worker_thread.join();
    EXPECT_TRUE(free_surface_count >= free_surface_count_prev);
}

TEST(OneVPL_Source_ProcessingEngine, Init)
{
    using namespace cv::gapi::wip;
    std::unique_ptr<cv::gapi::wip::VPLAccelerationPolicy> accel;
    TestProcessingEngine engine(std::move(accel));

    mfxSession mfx_session{};
    engine.initialize_session(mfx_session, {}, std::shared_ptr<IDataProvider>{});

    EXPECT_EQ(engine.get_ready_frames_count(), 0);
    ProcessingEngineBase::ExecutionStatus ret = engine.process(mfx_session);
    EXPECT_EQ(ret, ProcessingEngineBase::ExecutionStatus::Continue);
    EXPECT_EQ(engine.pipeline_stage_num, 0);

    ret = engine.process(mfx_session);
    EXPECT_EQ(ret, ProcessingEngineBase::ExecutionStatus::Continue);
    EXPECT_EQ(engine.pipeline_stage_num, 1);

    ret = engine.process(mfx_session);
    EXPECT_EQ(ret, ProcessingEngineBase::ExecutionStatus::Continue);
    EXPECT_EQ(engine.pipeline_stage_num, 2);

    ret = engine.process(mfx_session);
    EXPECT_EQ(ret, ProcessingEngineBase::ExecutionStatus::Processed);
    EXPECT_EQ(engine.pipeline_stage_num, 3);
    EXPECT_EQ(engine.get_ready_frames_count(), 1);

    Data frame;
    engine.get_frame(frame);
}

TEST(OneVPL_Source_DX11_Accel, Init)
{
    using namespace cv::gapi::wip;
    VPLDX11AccelerationPolicy accel;

    mfxLoader mfx_handle = MFXLoad();

    mfxConfig cfg_inst_0 = MFXCreateConfig(mfx_handle);
    EXPECT_TRUE(cfg_inst_0);
    mfxVariant mfx_param_0;
    mfx_param_0.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_0.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_0,(mfxU8 *)"mfxImplDescription.Impl",
                                                    mfx_param_0), MFX_ERR_NONE);

    mfxConfig cfg_inst_1 = MFXCreateConfig(mfx_handle);
    EXPECT_TRUE(cfg_inst_1);
    mfxVariant mfx_param_1;
    mfx_param_1.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_1.Data.U32 = MFX_ACCEL_MODE_VIA_D3D11;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_1,(mfxU8 *)"mfxImplDescription.AccelerationMode",
                                                    mfx_param_1), MFX_ERR_NONE);

    mfxConfig cfg_inst_2 = MFXCreateConfig(mfx_handle);
    EXPECT_TRUE(cfg_inst_2);
    mfxVariant mfx_param_2;
    mfx_param_2.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_2.Data.U32 = MFX_CODEC_HEVC;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_2,(mfxU8 *)"mfxImplDescription.mfxDecoderDescription.decoder.CodecID",
                                                    mfx_param_2), MFX_ERR_NONE);

    // create session
    mfxSession mfx_session{};
    mfxStatus sts = MFXCreateSession(mfx_handle, 0, &mfx_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // assign acceleration
    EXPECT_NO_THROW(accel.init(mfx_session));

    // create proper bitstream
    mfxBitstream bitstream{};
    const int BITSTREAM_BUFFER_SIZE = 2000000;
    bitstream.MaxLength = BITSTREAM_BUFFER_SIZE;
    bitstream.Data = (mfxU8 *)calloc(bitstream.MaxLength, sizeof(mfxU8));
    EXPECT_TRUE(bitstream.Data);

    // simulate read stream
    bitstream.DataOffset = 0;
    bitstream.DataLength = sizeof(streaming::onevpl::hevc_header) * sizeof(streaming::onevpl::hevc_header[0]);
    memcpy(bitstream.Data, streaming::onevpl::hevc_header, bitstream.DataLength);
    bitstream.CodecId = MFX_CODEC_HEVC;

    // prepare dec params
    mfxVideoParam mfxDecParams {};
    mfxDecParams.mfx.CodecId = bitstream.CodecId;
    mfxDecParams.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    sts = MFXVideoDECODE_DecodeHeader(mfx_session, &bitstream, &mfxDecParams);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    mfxFrameAllocRequest request{};
    memset(&request, 0, sizeof(request));
    sts = MFXVideoDECODE_QueryIOSurf(mfx_session, &mfxDecParams, &request);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // Allocate surfaces for decoder
    VPLAccelerationPolicy::pool_key_t key = accel.create_surface_pool(request,
                                                                      mfxDecParams);
    auto cand_surface = accel.get_free_surface(key).lock();

    sts = MFXVideoDECODE_Init(mfx_session, &mfxDecParams);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    MFXVideoDECODE_Close(mfx_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    EXPECT_NO_THROW(accel.deinit(mfx_session));
    MFXClose(mfx_session);
    MFXUnload(mfx_handle);
}

TEST(OneVPL_Source_Device_Selector_CfgParam, DefaultDevice)
{
    using namespace cv::gapi::wip;
    CfgParamDeviceSelector selector;
    IDeviceSelector::DeviceScoreTable devs = selector.select_devices();
    EXPECT_EQ(devs.size(), 1);
    test_host_dev_eq(*devs.begin(), IDeviceSelector::Score::Max);

    Context ctx = selector.select_context(devs);
    EXPECT_EQ(ctx.get_ptr(), nullptr);
}

TEST(OneVPL_Source_Device_Selector_CfgParam, DefaultDeviceFromCfgParam)
{
    using namespace cv::gapi::wip;

    using cfg_param = cv::gapi::wip::oneVPL_cfg_param;

    {
        std::vector<cfg_param> empty_params;
        CfgParamDeviceSelector selector(empty_params);
        IDeviceSelector::DeviceScoreTable devs = selector.select_devices();
        EXPECT_EQ(devs.size(), 1);
        test_host_dev_eq(*devs.begin(), IDeviceSelector::Score::Max);
    }
    {
        std::vector<cfg_param> cfg_params_w_no_accel;
        cfg_params_w_no_accel.push_back(cfg_param::create<uint32_t>("mfxImplDescription.AccelerationMode",
                                                                    MFX_ACCEL_MODE_NA));
        CfgParamDeviceSelector selector(cfg_params_w_no_accel);
        IDeviceSelector::DeviceScoreTable devs = selector.select_devices();
        EXPECT_EQ(devs.size(), 1);
        test_host_dev_eq(*devs.begin(), IDeviceSelector::Score::Max);
    }

    {
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
        std::vector<cfg_param> empty_params;
        CfgParamDeviceSelector selector(empty_params);
        IDeviceSelector::DeviceScoreTable devs = selector.select_devices();
        EXPECT_EQ(devs.size(), 1);
        test_host_dev_eq(*devs.begin(), IDeviceSelector::Score::Max);
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
    }

    {
#ifndef HAVE_DIRECTX
#ifndef HAVE_D3D11
        std::vector<cfg_param> cfg_params_w_non_existed_dx11;
        cfg_params_w_not_existed_dx11.push_back(cfg_param::create<uint32_t>("mfxImplDescription.AccelerationMode",
                                                                            MFX_ACCEL_MODE_VIA_D3D11));
        EXPECT_THROW(CfgParamDeviceSelector{cfg_params_w_non_existed_dx11},
                     std::logic_error);
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
    }

    {
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
        std::vector<cfg_param> cfg_params_w_dx11;
        cfg_params_w_dx11.push_back(cfg_param::create<uint32_t>("mfxImplDescription.AccelerationMode",
                                                                MFX_ACCEL_MODE_VIA_D3D11));
        std::unique_ptr<CfgParamDeviceSelector> selector_ptr;
        EXPECT_NO_THROW(selector_ptr.reset(new CfgParamDeviceSelector(cfg_params_w_dx11)));
        IDeviceSelector::DeviceScoreTable devs = selector_ptr->select_devices();

        EXPECT_EQ(devs.size(), 1);
        test_eq(*devs.begin(), IDeviceSelector::Score::Max, AccelType::DX11,
                std::get<1>(*devs.begin()).get_ptr() /* compare just type */);

        Context ctx = selector_ptr->select_context(devs);
        EXPECT_TRUE(ctx.get_ptr());
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
    }
}

TEST(OneVPL_Source_Device_Selector_CfgParam, PtrDeviceFromCfgParam)
{
    using namespace cv::gapi::wip;

    using cfg_param = cv::gapi::wip::oneVPL_cfg_param;

    {
        std::vector<cfg_param> empty_params;
        Device::Ptr empty_device_ptr = nullptr;
        Context::Ptr empty_ctx_ptr = nullptr;
        EXPECT_THROW(CfgParamDeviceSelector sel(empty_device_ptr, empty_ctx_ptr,
                                            empty_params),
                     std::logic_error); // params must describe device_ptr explicitly
    }

    {
#ifndef HAVE_DIRECTX
#ifndef HAVE_D3D11
        std::vector<cfg_param> cfg_params_w_non_existed_dx11;
        cfg_params_w_not_existed_dx11.push_back(cfg_param::create<uint32_t>("mfxImplDescription.AccelerationMode",
                                                                            MFX_ACCEL_MODE_VIA_D3D11));
        EXPECT_THROW(CfgParamDeviceSelector{cfg_params_w_non_existed_dx11},
                     std::logic_error); // the same as default dx11
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
    }

    {
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
        std::vector<cfg_param> cfg_params_w_dx11;
        cfg_params_w_dx11.push_back(cfg_param::create<uint32_t>("mfxImplDescription.AccelerationMode",
                                                                MFX_ACCEL_MODE_VIA_D3D11));
        Device::Ptr empty_device_ptr = nullptr;
        Context::Ptr empty_ctx_ptr = nullptr;
        EXPECT_THROW(CfgParamDeviceSelector sel(empty_device_ptr, empty_ctx_ptr,
                                                cfg_params_w_dx11),
                     std::logic_error); // empty_device_ptr must be valid
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
    }

    {
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
        ID3D11Device *device = nullptr;
        ID3D11DeviceContext* device_context = nullptr;
        {
            //Create device
            UINT creationFlags = 0;//D3D11_CREATE_DEVICE_BGRA_SUPPORT;

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
                    &device, // Returns the Direct3D device created.
                    &featureLevel, // Returns feature level of device created.
                    &device_context // Returns the device immediate context.
                    );
            EXPECT_FALSE(FAILED(err));
        }
        std::unique_ptr<CfgParamDeviceSelector> selector_ptr;
        std::vector<cfg_param> cfg_params_w_dx11;
        cfg_params_w_dx11.push_back(cfg_param::create<uint32_t>("mfxImplDescription.AccelerationMode",
                                                                MFX_ACCEL_MODE_VIA_D3D11));
        EXPECT_NO_THROW(selector_ptr.reset(new CfgParamDeviceSelector(device, device_context,
                                                                      cfg_params_w_dx11)));
        IDeviceSelector::DeviceScoreTable devs = selector_ptr->select_devices();

        EXPECT_EQ(devs.size(), 1);
        test_eq(*devs.begin(), IDeviceSelector::Score::Max,
                AccelType::DX11, device);

        Context ctx = selector_ptr->select_context(devs);
        EXPECT_EQ(ctx.get_ptr(), device_context);
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
    }
}
}
} // namespace opencv_test
#endif // HAVE_ONEVPL
