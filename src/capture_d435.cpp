#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <iostream>

int main(int argc, char** argv) try {
    const int width = 640;
    const int height = 480;
    const int fps = 30;

    rs2::context ctx;
    auto devices = ctx.query_devices();
    if (devices.size() == 0) {
        std::cerr << "未检测到 RealSense 设备!" << std::endl;
        return EXIT_FAILURE;
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

    rs2::config cfg;
    // 左右红外相机, Y8 即 8 位灰度
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
    // 彩色相机, 直接要 BGR8 方便 OpenCV 使用
    cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);

    rs2::pipeline pipe;
    rs2::pipeline_profile profile = pipe.start(cfg);

    // 打印左右 IR 的内参和外参(基线), 后面做 VIO 会用到
    auto ir1_profile = profile.get_stream(RS2_STREAM_INFRARED, 1)
                           .as<rs2::video_stream_profile>();
    auto ir2_profile = profile.get_stream(RS2_STREAM_INFRARED, 2)
                           .as<rs2::video_stream_profile>();
    rs2_intrinsics intr = ir1_profile.get_intrinsics();
    rs2_extrinsics extr = ir1_profile.get_extrinsics_to(ir2_profile);
    std::printf("IR 内参: fx=%.3f fy=%.3f cx=%.3f cy=%.3f\n",
                intr.fx, intr.fy, intr.ppx, intr.ppy);
    std::printf("左右 IR 基线: %.4f m\n", -extr.translation[0]);

    std::cout << "按 q 或 ESC 退出, 按 s 保存当前帧" << std::endl;

    int saved = 0;
    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();

        rs2::video_frame ir_left = frames.get_infrared_frame(1);
        rs2::video_frame ir_right = frames.get_infrared_frame(2);
        rs2::video_frame color = frames.get_color_frame();

        // 包装为 cv::Mat (不拷贝数据, 生命周期由 frame 保证)
        cv::Mat left(cv::Size(width, height), CV_8UC1,
                     const_cast<void*>(ir_left.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat right(cv::Size(width, height), CV_8UC1,
                      const_cast<void*>(ir_right.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat bgr(cv::Size(width, height), CV_8UC3,
                    const_cast<void*>(color.get_data()), cv::Mat::AUTO_STEP);

        // 时间戳(毫秒), VIO 与 IMU 对齐时要用
        double ts = ir_left.get_timestamp();

        cv::Mat stereo;
        cv::hconcat(left, right, stereo);
        cv::putText(stereo, cv::format("ts: %.3f ms", ts), {10, 25},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {255}, 1);

        cv::imshow("IR left | right", stereo);
        cv::imshow("Color", bgr);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 's') {
            cv::imwrite(cv::format("left_%03d.png", saved), left);
            cv::imwrite(cv::format("right_%03d.png", saved), right);
            cv::imwrite(cv::format("color_%03d.png", saved), bgr);
            std::cout << "已保存第 " << saved << " 组图像" << std::endl;
            ++saved;
        }
    }

    pipe.stop();
    return EXIT_SUCCESS;
} catch (const rs2::error& e) {
    std::cerr << "RealSense 错误 (" << e.get_failed_function() << "): "
              << e.what() << std::endl;
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
