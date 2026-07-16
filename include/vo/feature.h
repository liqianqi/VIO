#pragma once

#include "vo/camera.h"

class FeatureTracker{
public:
    FeatureTracker();
    ~FeatureTracker();
    // 输入当前左图, 返回当前帧的特征点 (带id)
    void track(const cv::Mat& img);

    void matchStereo(const cv::Mat& left, const cv::Mat& right,
        float fx, float baseline);

    std::vector<cv::Point2f> cur_pts_;   // 当前特征点位置
    std::vector<long> ids_;              // 每个点的全局 id, 与 cur_pts_ 一一对应
    std::vector<int> track_cnt_;         // 每个点被连续跟踪了多少帧
    std::vector<cv::Point2f> right_pts_; // 每个点在右图的匹配位置
    std::vector<float> depth_;           // 每个点的深度(米), <0 表示无效

private:
    cv::Mat img_prev_;
    std::vector<cv::Point2f> prev_pts_;
    long next_id_ = 0;                   // 发号器
    // 参数
    int max_cnt_ = 200;                  // 目标点数
    int min_dist_ = 25;                  // 点与点最小间距(像素), 兼做均匀化
};