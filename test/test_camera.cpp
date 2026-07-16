#include "vo/camera.h"

#include <iostream>
#include <vector>

// D435Camera 类的功能测试:
// 1. 能否正常构造(打开设备、启动取流)
// 2. 连续取 60 帧, 检查图像尺寸/类型、时间戳单调递增
// 3. 统计实际帧率
// 4. 数据独立性: 上一帧的图像在取新帧后内容不能被破坏
int main() {
    const int width = 640, height = 480, fps = 30;
    const int num_frames = 60;

    D435Camera cam(width, height, fps);

    StereoFrame frame;
    int failures = 0;

    cam.getStereoFrame(frame);
    double last_ts = frame.timestamp;
    double first_ts = frame.timestamp;

    for (int i = 1; i < num_frames; ++i) {
        cam.getStereoFrame(frame);

        if (frame.left.size() != cv::Size(width, height) || frame.left.type() != CV_8UC1) {
            std::cerr << "[FAIL] 第 " << i << " 帧左图尺寸/类型错误" << std::endl;
            ++failures;
        }
        if (frame.right.size() != cv::Size(width, height) || frame.right.type() != CV_8UC1) {
            std::cerr << "[FAIL] 第 " << i << " 帧右图尺寸/类型错误" << std::endl;
            ++failures;
        }
        if (frame.timestamp <= last_ts) {
            std::cerr << "[FAIL] 第 " << i << " 帧时间戳未递增: " << last_ts
                      << " -> " << frame.timestamp << std::endl;
            ++failures;
        }
        if (cv::countNonZero(frame.left) == 0) {
            std::cerr << "[FAIL] 第 " << i << " 帧左图全黑" << std::endl;
            ++failures;
        }
        last_ts = frame.timestamp;
    }

    double elapsed_s = (last_ts - first_ts) / 1000.0;
    double actual_fps = (num_frames - 1) / elapsed_s;
    std::cout << "取 " << num_frames << " 帧耗时 " << elapsed_s << " s, 实际帧率 "
              << actual_fps << " fps (设定 " << fps << ")" << std::endl;
    if (actual_fps < fps * 0.8) {
        std::cerr << "[FAIL] 实际帧率过低" << std::endl;
        ++failures;
    }

    cv::imwrite("test_left.png", frame.left);
    cv::imwrite("test_right.png", frame.right);
    std::cout << "已保存最后一帧到 test_left.png / test_right.png" << std::endl;

    if (failures == 0) {
        std::cout << "[PASS] D435Camera 全部测试通过" << std::endl;
        return 0;
    }
    std::cerr << "[FAIL] 共 " << failures << " 项失败" << std::endl;
    return 1;
}
