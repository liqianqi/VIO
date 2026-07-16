#pragma once

#include <cmath>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <iostream>


struct StereoFrame
{
    cv::Mat left;
    cv::Mat right;
    double timestamp;
};

class D435Camera {
public:
    D435Camera(int width, int height, int fps);
    ~D435Camera();


    void getStereoFrame(StereoFrame& frame);
    
    float fx() const { return intr_left_.fx; }
    float fy() const { return intr_left_.fy; }
    float cx() const { return intr_left_.ppx; }
    float cy() const { return intr_left_.ppy; }
    float baseline() const { return -extr_.translation[0]; }  // 注意负号, 约 0.05

    // 3x3 内参矩阵 K, PnP (cv::solvePnP) 直接要这个格式
    cv::Mat K() const {
        return (cv::Mat_<double>(3, 3) << fx(), 0, cx(), 0, fy(), cy(), 0, 0, 1);
    }
    
private:
    rs2::context ctx_;
    rs2::config cfg_;
    rs2::pipeline pipe_;
    rs2_intrinsics intr_left_;
    rs2_intrinsics intr_right_;
    rs2_extrinsics extr_;

    // 持有最近一帧, 保证外部拿到的 cv::Mat 数据指针在下次取帧前有效
    // (rs2::frame 有默认构造, rs2::video_frame 没有)
    rs2::frame ir_left;
    rs2::frame ir_right;
    rs2::frame color;

    int width_ = 0;
    int height_ = 0;
    int fps_ = 0;

    cv::Mat color_;
}
;