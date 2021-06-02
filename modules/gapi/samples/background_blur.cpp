#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
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

//#define ENABLE_LEGACY

cv::Size blur_threshold{5, 5};
int blur_width_meter = blur_threshold.width;
int blur_height_meter = blur_threshold.height;

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
    
using GMat2 = std::tuple<cv::GMat, cv::GMat>;

G_API_OP(BlurPostProcessing, <cv::GMat(cv::GMat, cv::GMat)>, "sample.custom.post_processing") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in, const cv::GMatDesc &) {
        return in;
    }
};

G_API_OP(SemSegPostProcessing, <GMat2(cv::GMat, cv::GMat)>, "sample.custom.sem_seg_post_processing") {
    static std::tuple<cv::GMatDesc, cv::GMatDesc> outMeta(const cv::GMatDesc& in, const cv::GMatDesc &classes) {

        return std::make_tuple(in, in);
    }
};

G_API_OP(MeteredBlur, <cv::GMat(cv::GMat, std::reference_wrapper<cv::Size>, cv::Point, int, cv::Scalar)>,
         "org.opencv.imgproc.filters.metered_blur") {
    static cv::GMatDesc outMeta(const cv::GMatDesc& in, std::reference_wrapper<cv::Size>, cv::Point, int, cv::Scalar) {
        return in;
    }
};

GAPI_OCV_KERNEL(OCVBlurPostProcessing, BlurPostProcessing) {
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
        cv::blur(background_out, background_out_blur, blur_threshold/*in.size()*/);
        cv::bitwise_or(background_out_blur, person_out, out);
    }
};

GAPI_OCV_KERNEL(OCVSemSegPostProcessing, SemSegPostProcessing) {
    static void run(const cv::Mat &in, const cv::Mat &detected_classes, cv::Mat& out_person, cv::Mat& out_bg) {
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
        cv::resize(maskPersonImg, out_person, in.size(), 0,0, cv::InterpolationFlags::INTER_NEAREST);
        cv::resize(maskBackgroundImg, out_bg, in.size(), 0,0, cv::InterpolationFlags::INTER_NEAREST);
    }
};

GAPI_OCV_KERNEL(OCVMeteredBlur, MeteredBlur) {
        static void run(const cv::Mat &in, std::reference_wrapper<cv::Size> ksize_ref,
                        const cv::Point& anchor, int borderType, const cv::Scalar& bordVal, cv::Mat &out) {

            cv::Size ksize = ksize_ref.get();
        
            if( borderType == cv::BORDER_CONSTANT )
            {
                cv::Mat temp_in;
                int width_add = (ksize.width - 1) / 2;
                int height_add =  (ksize.height - 1) / 2;
                cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, borderType, bordVal);
                cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
                cv::blur(temp_in(rect), out, ksize, anchor, borderType);
            }
            else
                cv::blur(in, out, ksize, anchor, borderType);
        }
};
} // namespace custom

void thresh_callback(int meter, void* data)
{
    using value_type = cv::Size::value_type;
    value_type* threshold_value_ptr = static_cast<value_type*>(data);
    if (meter > 0)
    {
        *threshold_value_ptr = meter;
    }
}


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
    
#ifdef     ENABLE_LEGACY 
    const auto kernels = cv::gapi::kernels<custom::OCVBlurPostProcessing>();
    const auto networks = cv::gapi::networks(net);

    // Now build the graph
    cv::GMat in;
    cv::GMat detected_classes = cv::gapi::infer<SemSegmNet>(in);
    cv::GMat out = custom::BlurPostProcessing::on(in, detected_classes);

    cv::GStreamingCompiled pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out))
        .compileStreaming(cv::compile_args(kernels, networks));
#else

    const auto kernels = cv::gapi::kernels<custom::OCVSemSegPostProcessing, custom::OCVMeteredBlur>();
    const auto networks = cv::gapi::networks(net);

    // Now build the graph
    cv::GMat in;
    cv::GMat detected_classes = cv::gapi::infer<SemSegmNet>(in);

    cv::GMat person_resized, background_resized;
    std::tie(person_resized, background_resized) = custom::SemSegPostProcessing::on(in, detected_classes);
    
    // Extract objects
    cv::GMat person_extracted = cv::gapi::bitwise_and(in, person_resized);
    cv::GMat background_extracted = cv::gapi::bitwise_and(in, background_resized);

    // Blur BG
    cv::GMat blurred_background = custom::MeteredBlur::on(
                background_extracted, std::ref(blur_threshold), cv::Point(-1,-1),
                cv::BorderTypes::BORDER_DEFAULT, cv::Scalar(0));//cv::gapi::blur(background_extracted, cv::Size{blur_threshold_width, blur_threshold_height});
    
    cv::GMat out = cv::gapi::bitwise_or(blurred_background, person_extracted);
    
    cv::GStreamingCompiled pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out))
        .compileStreaming(cv::compile_args(kernels, networks));
#endif

    cv::gapi::wip::IStreamSource::Ptr src;
    if(input.empty())
    {
        src = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(0);
    }
    else
    {
        src = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input);
    }
    auto inputs = cv::gin(std::move(src));

    // The execution part
    pipeline.setSource(std::move(inputs));
    pipeline.start();

    cv::VideoWriter writer;
    cv::Mat outMat;
    const std::string source_window{"BlurBackground"};
    while (pipeline.pull(cv::gout(outMat))) {
        cv::imshow(source_window, outMat);
        cv::createTrackbar("Blur thresh width:", source_window,
                           &blur_width_meter,
                           outMat.size().width,
                           thresh_callback, &blur_threshold.width );
        cv::createTrackbar("Blur thresh height:", source_window,
                           &blur_height_meter,
                           outMat.size().height,
                           thresh_callback, &blur_threshold.height);
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
