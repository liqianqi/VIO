#include "vo/camera.h"

D435Camera::D435Camera(int width, int height, int fps)
    : width_(width), height_(height), fps_(fps)
{
    auto devices = ctx_.query_devices();
    if (devices.size() == 0) {
        std::cerr << "未检测到 RealSense 设备!" << std::endl;
        return ;
    }
    rs2::device dev = devices[0];
    std::cout << "设备: " << dev.get_info(RS2_CAMERA_INFO_NAME)
              << "  序列号: " << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)
              << "  固件: " << dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION)
              << std::endl;

    // VIO 需要关闭红外结构光投射器, 否则 IR 图上的散斑点会干扰特征提取/跟踪
    for (auto&& sensor : dev.query_sensors()) {
        if (sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
            sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f);
            std::cout << "已关闭 IR 投射器 (emitter)" << std::endl;
        }
    }

    cfg_.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
    cfg_.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
    cfg_.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);

    rs2::pipeline_profile profile = pipe_.start(cfg_);
    auto ir1_profile = profile.get_stream(RS2_STREAM_INFRARED, 1)
        .as<rs2::video_stream_profile>();
    auto ir2_profile = profile.get_stream(RS2_STREAM_INFRARED, 2)
        .as<rs2::video_stream_profile>();
    intr_left_ = ir1_profile.get_intrinsics();
    intr_right_ = ir2_profile.get_intrinsics();
    extr_ = ir1_profile.get_extrinsics_to(ir2_profile);

    std::printf("left IR 内参: fx=%.3f fy=%.3f cx=%.3f cy=%.3f\n",
        intr_left_.fx, intr_left_.fy, intr_left_.ppx, intr_left_.ppy);
    std::printf("right IR 内参: fx=%.3f fy=%.3f cx=%.3f cy=%.3f\n",
        intr_right_.fx, intr_right_.fy, intr_right_.ppx, intr_right_.ppy);
    std::printf("左右 IR 基线: %.4f m\n", -extr_.translation[0]);

    std::cout << "按 q 或 ESC 退出, 按 s 保存当前帧" << std::endl;

}

D435Camera::~D435Camera()
{
    pipe_.stop();
}

void D435Camera::getStereoFrame(StereoFrame& frame)
{
    rs2::frameset frames = pipe_.wait_for_frames();
    // 成员持有 frame 引用计数, 数据在下次 getStereoFrame 前不会释放,
    // 因此 Mat 可以零拷贝包装; 代价是返回的 Mat 在下次取帧后失效
    ir_left = frames.get_infrared_frame(1);
    ir_right = frames.get_infrared_frame(2);
    color = frames.get_color_frame();

    frame.left = cv::Mat(cv::Size(width_, height_), CV_8UC1,
                        const_cast<void*>(ir_left.get_data()), cv::Mat::AUTO_STEP);
    frame.right = cv::Mat(cv::Size(width_, height_), CV_8UC1,
                        const_cast<void*>(ir_right.get_data()), cv::Mat::AUTO_STEP);
    frame.timestamp = ir_left.get_timestamp();

    color_ = cv::Mat(cv::Size(width_, height_), CV_8UC3,
                     const_cast<void*>(color.get_data()), cv::Mat::AUTO_STEP);
}