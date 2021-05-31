#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/highgui.hpp>

const std::string keys =
    "{ h help |                                     | Print this help message }"
    "{ input  |                                     | Path to the input video file }"
    "{ output |                                     | Path to the output video file }"
    "{ ssm    | semantic-segmentation-adas-0001.xml | Path to OpenVINO IE semantic segmentation model (.xml) }";

// 20 colors for 20 classes of semantic-segmentation-adas-0001
const std::vector<cv::Vec3b> colors = {
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 255,  255,  255 },  // <- person
    { 255,  255,  255 },  // <- rider
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
};

namespace {
std::string get_weights_path(const std::string &model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    CV_Assert(sz > EXT_LEN);

    auto ext = model_path.substr(sz - EXT_LEN);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){
            return static_cast<unsigned char>(std::tolower(c));
        });
    CV_Assert(ext == ".xml");
    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}
} // anonymous namespace

namespace custom {
G_API_OP(PostProcessing, <cv::GMat(cv::GMat, cv::GMat)>, "sample.custom.post_processing") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in, const cv::GMatDesc &) {
        return in;
    }
};

GAPI_OCV_KERNEL(OCVPostProcessing, PostProcessing) {
    static void run(const cv::Mat &in, const cv::Mat &detected_classes, cv::Mat &out) {
        // This kernel constructs output image by class table and colors vector

        // The semantic-segmentation-adas-0001 output a blob with the shape
        // [B, C=1, H=1024, W=2048]
        const int outHeight = 1024;
        const int outWidth = 2048;
        cv::Mat maskPersonImg(outHeight, outWidth, CV_8UC3);
        cv::Mat maskBackgroundImg(outHeight, outWidth, CV_8UC3);

        cv::Vec3b person_cut_color{0, 0, 0};
        cv::Vec3b background_color{255, 255, 255};
        const int* const classes = detected_classes.ptr<int>();
        for (int rowId = 0; rowId < outHeight; ++rowId) {
            for (int colId = 0; colId < outWidth; ++colId) {
                size_t classId = static_cast<size_t>(classes[rowId * outWidth + colId]);
                cv::Vec3b detected_class_color = classId < colors.size()
                        ? colors[classId]
                        : person_cut_color;
                
                maskPersonImg.at<cv::Vec3b>(rowId, colId) = detected_class_color;
                if (detected_class_color != person_cut_color)
                {
                    maskBackgroundImg.at<cv::Vec3b>(rowId, colId) = person_cut_color;
                }
                else
                {
                    maskBackgroundImg.at<cv::Vec3b>(rowId, colId) = background_color;
                }
            }
        }
        cv::Mat background_mask, person_mask, person_out, background_out, background_out_blur;
        cv::resize(maskPersonImg, person_mask, in.size(), 0,0, cv::InterpolationFlags::INTER_NEAREST);
        cv::resize(maskBackgroundImg, background_mask, in.size(), 0,0, cv::InterpolationFlags::INTER_NEAREST);

        cv::bitwise_and(in, person_mask, person_out);
        cv::bitwise_and(in, background_mask, background_out);
        cv::blur(background_out, background_out_blur, cv::Size{20,20}/*in.size()*/);
        cv::bitwise_or(background_out_blur, person_out, out);
        /*
        const float blending = 0.3f;
        out = in * blending + out * (1 - blending);
        */
    }
};
} // namespace custom

int main(int argc, char *argv[]) {
    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    // Prepare parameters first
    const std::string input  = cmd.get<std::string>("input");
    const std::string output = cmd.get<std::string>("output");
    const auto model_path    = cmd.get<std::string>("ssm");
    const auto weights_path  = get_weights_path(model_path);
    const auto device        = "CPU";
    G_API_NET(SemSegmNet, <cv::GMat(cv::GMat)>, "semantic-segmentation");
    const auto net = cv::gapi::ie::Params<SemSegmNet> {
        model_path, weights_path, device
    };
    const auto kernels = cv::gapi::kernels<custom::OCVPostProcessing>();
    const auto networks = cv::gapi::networks(net);

    // Now build the graph
    cv::GMat in;
    cv::GMat detected_classes = cv::gapi::infer<SemSegmNet>(in);
    cv::GMat out = custom::PostProcessing::on(in, detected_classes);

    cv::GStreamingCompiled pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out))
        .compileStreaming(cv::compile_args(kernels, networks));
    auto inputs = cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input));

    // The execution part
    pipeline.setSource(std::move(inputs));
    pipeline.start();

    cv::VideoWriter writer;
    cv::Mat outMat;
    while (pipeline.pull(cv::gout(outMat))) {
        cv::imshow("Out", outMat);
        cv::waitKey(1);
        if (!output.empty()) {
            if (!writer.isOpened()) {
                const auto sz = cv::Size{outMat.cols, outMat.rows};
                writer.open(output, cv::VideoWriter::fourcc('M','J','P','G'), 25.0, sz);
                CV_Assert(writer.isOpened());
            }
            writer << outMat;
        }
    }
    return 0;
}
