#pragma once

#include "vo/frame.h"
#include "vo/map.h"

// 基于关键帧 + 局部地图的 PnP 视觉里程计:
// 地图点是世界系坐标, 锚定在关键帧上 (LocalMap::addPoints 不覆盖老 id);
// 普通帧用 FrameMatch 关联 "地图 3D 点 - 当前帧 2D 点", solvePnPRansac
// 直接解出绝对位姿 T_cw, 求逆得 T_wc —— 两个关键帧之间不累积误差。
// 跟踪变弱或运动超阈值时新建关键帧, 把当前帧的新点入库。
// 世界系 = 第一帧相机系。
class Odometry {
public:
    Odometry(float fx, float fy, float cx, float cy, float baseline);

    // 处理一帧; 返回 true 表示位姿被有效更新,
    // false 表示本帧被拒绝 (匹配/内点不足或运动异常), 位姿保持不变
    bool process(const StereoFrame& frame);

    const cv::Mat& pose() const { return T_wc_; }   // 4x4 CV_64F, 相机 -> 世界
    cv::Point3d position() const;                   // 当前相机在世界系的位置
    int lastInliers() const { return last_inliers_; }
    int keyframeCount() const { return kf_count_; }
    size_t mapSize() const { return map_.size(); }
    double lastBaMs() const { return last_ba_ms_; }  // 最近一次滑窗 BA 耗时
    const FrameMatch& matcher() const { return matcher_; }

private:
    // 新点入库 + 组装带观测的关键帧 + 滑窗局部 BA
    void insertKeyframe(double timestamp);

    FrameMatch matcher_;
    LocalMap map_;
    cv::Mat K_;
    double baseline_;
    cv::Mat T_wc_;                 // 当前位姿, 初始为单位阵
    cv::Mat T_wc_kf_;              // 最近一个关键帧的位姿 (关键帧判据用)

    int last_inliers_ = 0;
    int kf_count_ = 0;
    double last_ba_ms_ = 0;

    // 拒绝坏解的阈值
    int min_matches_ = 20;         // 匹配对少于此值不解 PnP
    int min_inliers_ = 15;         // RANSAC 内点少于此值拒绝
    double max_trans_ = 0.5;       // 相邻帧位姿跳变超 0.5m 视为坏解 (30fps 下不可能)
    double max_rot_deg_ = 30.0;    // 相邻帧位姿跳变超 30 度视为坏解

    // 关键帧判据
    int kf_min_inliers_ = 60;      // 跟地图的匹配内点少于此值 -> 新关键帧
    double kf_trans_ = 0.2;        // 距上个关键帧平移超 0.2m -> 新关键帧
    double kf_rot_deg_ = 15.0;     // 距上个关键帧旋转超 15 度 -> 新关键帧
};
