#include <algorithm>
#include <fstream>
#include <iostream>
#include <cctype>
#include <tuple>

#include <opencv2/imgproc.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/streaming/onevpl/source.hpp>
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include <opencv2/highgui.hpp> // CommandLineParser
#include <opencv2/gapi/infer/parsers.hpp>

#ifdef HAVE_INF_ENGINE
#include <inference_engine.hpp> // ParamMap

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
#pragma comment(lib,"d3d11.lib")

// get rid of generate macro max/min/etc from DX side
#define D3D11_NO_HELPERS
#define NOMINMAX
#include <cldnn/cldnn_config.hpp>
#include <d3d11.h>
#pragma comment(lib, "dxgi")
#undef NOMINMAX
#undef D3D11_NO_HELPERS

#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_INF_ENGINE

const std::string about =
    "This is an OpenCV-based version of oneVPLSource decoder example";
const std::string keys =
    "{ h help                       |                                           | Print this help message }"
    "{ input                        |                                           | Path to the input demultiplexed video file }"
    "{ output                       |                                           | Path to the output RAW video file. Use .avi extension }"
    "{ facem                        | face-detection-adas-0001.xml              | Path to OpenVINO IE face detection model (.xml) }"
    "{ faced                        | AUTO                                      | Target device for face detection model (e.g. AUTO, GPU, VPU, ...) }"
    "{ cfg_params                   | <prop name>:<value>;<prop name>:<value>   | Semicolon separated list of oneVPL mfxVariants which is used for configuring source (see `MFXSetConfigFilterProperty` by https://spec.oneapi.io/versions/latest/elements/oneVPL/source/index.html) }"
    "{ streaming_queue_capacity     | 1                                         | Streaming executor queue capacity. Calculated automaticaly if 0 }"
    "{ frames_pool_size             | 0                                         | OneVPL source applies this parameter as preallocated frames pool size}"
    "{ vpp_frames_pool_size         | 0                                         | OneVPL source applies this parameter as preallocated frames pool size for VPP preprocessing results}"
    "{ roi                          | -1,-1,-1,-1                               | Region of interest (ROI) to use for inference. Identified automatically when not set }";

namespace {
bool is_gpu(const std::string &device_name) {
    return device_name.find("GPU") != std::string::npos;
}

std::string get_weights_path(const std::string &model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    GAPI_Assert(sz > EXT_LEN);

    auto ext = model_path.substr(sz - EXT_LEN);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){
            return static_cast<unsigned char>(std::tolower(c));
        });
    GAPI_Assert(ext == ".xml");
    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}

// TODO: It duplicates infer_single_roi sample
cv::util::optional<cv::Rect> parse_roi(const std::string &rc) {
    cv::Rect rv;
    char delim[3];

    std::stringstream is(rc);
    is >> rv.x >> delim[0] >> rv.y >> delim[1] >> rv.width >> delim[2] >> rv.height;
    if (is.bad()) {
        return cv::util::optional<cv::Rect>(); // empty value
    }
    const auto is_delim = [](char c) {
        return c == ',';
    };
    if (!std::all_of(std::begin(delim), std::end(delim), is_delim)) {
        return cv::util::optional<cv::Rect>(); // empty value

    }
    if (rv.x < 0 || rv.y < 0 || rv.width <= 0 || rv.height <= 0) {
        return cv::util::optional<cv::Rect>(); // empty value
    }
    return cv::util::make_optional(std::move(rv));
}
} // anonymous namespace

namespace custom {
G_API_NET(FaceDetector,   <cv::GMat(cv::GMat)>, "face-detector");

using GDetections = cv::GArray<cv::Rect>;
using GRect       = cv::GOpaque<cv::Rect>;
using GSize       = cv::GOpaque<cv::Size>;
using GPrims      = cv::GArray<cv::gapi::wip::draw::Prim>;

G_API_OP(ParseSSD, <GDetections(cv::GMat, GRect, GSize)>, "sample.custom.parse-ssd") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GOpaqueDesc &, const cv::GOpaqueDesc &) {
        return cv::empty_array_desc();
    }
};

// TODO: It duplicates infer_single_roi sample
G_API_OP(LocateROI, <GRect(GSize)>, "sample.custom.locate-roi") {
    static cv::GOpaqueDesc outMeta(const cv::GOpaqueDesc &) {
        return cv::empty_gopaque_desc();
    }
};

G_API_OP(BBoxes, <GPrims(GDetections, GRect)>, "sample.custom.b-boxes") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc &, const cv::GOpaqueDesc &) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVLocateROI, LocateROI) {
    // This is the place where we can run extra analytics
    // on the input image frame and select the ROI (region
    // of interest) where we want to detect our objects (or
    // run any other inference).
    //
    // Currently it doesn't do anything intelligent,
    // but only crops the input image to square (this is
    // the most convenient aspect ratio for detectors to use)

    static void run(const cv::Size& in_size,
                    cv::Rect &out_rect) {

        // Identify the central point & square size (- some padding)
        const auto center = cv::Point{in_size.width/2, in_size.height/2};
        auto sqside = std::min(in_size.width, in_size.height);

        // Now build the central square ROI
        out_rect = cv::Rect{ center.x - sqside/2
                             , center.y - sqside/2
                             , sqside
                             , sqside
                            };
    }
};

GAPI_OCV_KERNEL(OCVBBoxes, BBoxes) {
    // This kernel converts the rectangles into G-API's
    // rendering primitives
    static void run(const std::vector<cv::Rect> &in_face_rcs,
                    const             cv::Rect  &in_roi,
                          std::vector<cv::gapi::wip::draw::Prim> &out_prims) {
        out_prims.clear();
        const auto cvt = [](const cv::Rect &rc, const cv::Scalar &clr) {
            return cv::gapi::wip::draw::Rect(rc, clr, 2);
        };
        out_prims.emplace_back(cvt(in_roi, CV_RGB(0,255,255))); // cyan
        for (auto &&rc : in_face_rcs) {
            out_prims.emplace_back(cvt(rc, CV_RGB(0,255,0)));   // green
        }
    }
};

GAPI_OCV_KERNEL(OCVParseSSD, ParseSSD) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Rect &in_roi,
                    const cv::Size &in_parent_size,
                    std::vector<cv::Rect> &out_objects) {
        const auto &in_ssd_dims = in_ssd_result.size;
        GAPI_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];
        GAPI_Assert(OBJECT_SIZE  == 7); // fixed SSD object size

        const cv::Size up_roi = in_roi.size();
        const cv::Rect surface({0,0}, in_parent_size);

        out_objects.clear();

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0];
            const float label      = data[i * OBJECT_SIZE + 1];
            const float confidence = data[i * OBJECT_SIZE + 2];
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];
            (void) label; // unused

            if (image_id < 0.f) {
                break;    // marks end-of-detections
            }
            if (confidence < 0.5f) {
                continue; // skip objects with low confidence
            }

            // map relative coordinates to the original image scale
            // taking the ROI into account
            cv::Rect rc;
            rc.x      = static_cast<int>(rc_left   * up_roi.width);
            rc.y      = static_cast<int>(rc_top    * up_roi.height);
            rc.width  = static_cast<int>(rc_right  * up_roi.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * up_roi.height) - rc.y;
            rc.x += in_roi.x;
            rc.y += in_roi.y;
            out_objects.emplace_back(rc & surface);
        }
    }
};

} // namespace custom

namespace cfg {
typename cv::gapi::wip::onevpl::CfgParam create_from_string(const std::string &line);
}

int main(int argc, char *argv[]) {

    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    // get file name
    const auto file_path = cmd.get<std::string>("input");
    const auto output = cmd.get<std::string>("output");
    const auto opt_roi = parse_roi(cmd.get<std::string>("roi"));
    const auto face_model_path = cmd.get<std::string>("facem");
    const auto streaming_queue_capacity = cmd.get<uint32_t>("streaming_queue_capacity");
    const auto source_decode_queue_capacity = cmd.get<uint32_t>("frames_pool_size");
    const auto source_vpp_queue_capacity = cmd.get<uint32_t>("vpp_frames_pool_size");
    const auto device_id = cmd.get<std::string>("faced");

    // check ouput file extension
    if (!output.empty()) {
        auto ext = output.find_last_of(".");
        if (ext == std::string::npos || (output.substr(ext + 1) != "avi")) {
            std::cerr << "Output file should have *.avi extension for output video" << std::endl;
            return -1;
        }
    }

    // get oneVPL cfg params from cmd
    std::stringstream params_list(cmd.get<std::string>("cfg_params"));
    std::vector<cv::gapi::wip::onevpl::CfgParam> source_cfgs;
    try {
        std::string line;
        while (std::getline(params_list, line, ';')) {
            source_cfgs.push_back(cfg::create_from_string(line));
        }
    } catch (const std::exception& ex) {
        std::cerr << "Invalid cfg parameter: " << ex.what() << std::endl;
        return -1;
    }

    if (source_decode_queue_capacity != 0) {
        source_cfgs.push_back(cv::gapi::wip::onevpl::CfgParam::create_frames_pool_size(source_decode_queue_capacity));
    }
    if (source_vpp_queue_capacity != 0) {
        source_cfgs.push_back(cv::gapi::wip::onevpl::CfgParam::create_vpp_frames_pool_size(source_vpp_queue_capacity));
    }

    if (is_gpu(device_id)) {
        // put accel type description for VPL source
        source_cfgs.push_back(cfg::create_from_string(
                                        "mfxImplDescription.AccelerationMode"
                                        ":"
                                        "MFX_ACCEL_MODE_VIA_D3D11"));
    }
    auto device_selector_ptr = cv::gapi::wip::create_device_selector_default(source_cfgs);

    auto face_net = cv::gapi::ie::Params<custom::FaceDetector> {
        face_model_path,                 // path to topology IR
        get_weights_path(face_model_path),   // path to weights
        device_id
    };

    // Turn on preproc
    face_net.cfgPreprocessingDeviceContext(device_selector_ptr);

    // Turn on Inference
    face_net.cfgInferenceDeviceContext(device_selector_ptr);

#ifdef HAVE_INF_ENGINE

    // set ctx_config for GPU device only - no need in case of CPU device type
    if (is_gpu(device_id)) {
        // NB: consider NV12 surface because it's one of native GPU image format
        face_net.pluginConfig({{"GPU_NV12_TWO_INPUTS", "YES" }});

        // TODO Will be removed when `cfgInferenceDeviceContext` is done
        InferenceEngine::ParamMap ctx_config({{"CONTEXT_TYPE", "VA_SHARED"},
                                              {"VA_DEVICE", device_selector_ptr->select_devices().begin()->second.get_ptr()} });
        face_net.cfgContextParams(ctx_config);
    }
#endif // HAVE_INF_ENGINE

    auto kernels = cv::gapi::kernels
        < custom::OCVLocateROI
        , custom::OCVParseSSD
        , custom::OCVBBoxes>();
    auto networks = cv::gapi::networks(face_net);
    auto face_detection_args = cv::compile_args(networks, kernels);
    if (streaming_queue_capacity != 0) {
        face_detection_args += cv::compile_args(cv::gapi::streaming::queue_capacity{ streaming_queue_capacity });
    }

    // Create source
    cv::gapi::wip::IStreamSource::Ptr cap;
    try {
        cap = cv::gapi::wip::make_onevpl_src(file_path, source_cfgs, device_selector_ptr);
        std::cout << "oneVPL source desription: " << cap->descr_of() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Cannot create source: " << ex.what() << std::endl;
        return -1;
    }

    cv::GMetaArg descr = cap->descr_of();
    auto frame_descr = cv::util::get<cv::GFrameDesc>(descr);
    auto inputs = cv::gin(cap);

    // Now build the graph
    cv::GFrame in;
    auto size = cv::gapi::streaming::size(in);
    cv::GStreamingCompiled pipeline;
    if (opt_roi.has_value()) {
        // Use the value provided by user
        std::cout << "Will run inference for static region "
                  << opt_roi.value()
                  << " only"
                  << std::endl;
        cv::GOpaque<cv::Rect> in_roi;
        auto blob = cv::gapi::infer<custom::FaceDetector>(in_roi, in);
        cv::GArray<cv::Rect> rcs = custom::ParseSSD::on(blob, in_roi, size);
        auto out_frame = cv::gapi::wip::draw::renderFrame(in, custom::BBoxes::on(rcs, in_roi));
        auto out = cv::gapi::streaming::BGR(out_frame);
        pipeline = cv::GComputation(cv::GIn(in, in_roi), cv::GOut(out))
            .compileStreaming(std::move(face_detection_args));

        // Since the ROI to detect is manual, make it part of the input vector
        inputs.push_back(cv::gin(opt_roi.value())[0]);
    } else {
        // Automatically detect ROI to infer. Make it output parameter
        std::cout << "ROI is not set or invalid. Locating it automatically"
                  << std::endl;
        cv::GOpaque<cv::Rect> roi = custom::LocateROI::on(size);
        auto blob = cv::gapi::infer<custom::FaceDetector>(roi, in);
        cv::GArray<cv::Rect> rcs = custom::ParseSSD::on(blob, roi, size);
        auto out_frame = cv::gapi::wip::draw::renderFrame(in, custom::BBoxes::on(rcs, roi));
        auto out = cv::gapi::streaming::BGR(out_frame);
        pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out))
                .compileStreaming(std::move(face_detection_args));
    }
    // The execution part
    pipeline.setSource(std::move(inputs));
    pipeline.start();

    size_t frames = 0u;
    cv::TickMeter tm;
    cv::VideoWriter writer;
    if (!output.empty() && !writer.isOpened()) {
        const auto sz = cv::Size{frame_descr.size.width, frame_descr.size.height};
        writer.open(output, cv::VideoWriter::fourcc('M','J','P','G'), 25.0, sz);
        GAPI_Assert(writer.isOpened());
    }

    cv::Mat outMat;
    tm.start();
    while (pipeline.pull(cv::gout(outMat))) {
        cv::imshow("Out", outMat);
        cv::waitKey(1);
        if (!output.empty()) {
            writer << outMat;
        }
        ++frames;
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames" << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}


namespace cfg {
typename cv::gapi::wip::onevpl::CfgParam create_from_string(const std::string &line) {
    using namespace cv::gapi::wip;

    if (line.empty()) {
        throw std::runtime_error("Cannot parse CfgParam from emply line");
    }

    std::string::size_type name_endline_pos = line.find(':');
    if (name_endline_pos == std::string::npos) {
        throw std::runtime_error("Cannot parse CfgParam from: " + line +
                                 "\nExpected separator \":\"");
    }

    std::string name = line.substr(0, name_endline_pos);
    std::string value = line.substr(name_endline_pos + 1);

    return cv::gapi::wip::onevpl::CfgParam::create(name, value,
                                                   /* vpp params strongly optional */
                                                   name.find("vpp.") == std::string::npos);
}
}
