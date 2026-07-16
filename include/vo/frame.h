#pragma once

#include "vo/feature.h"
#include "vo/map.h"

// 帧匹配: 每帧运行特征跟踪 + 双目匹配, 然后按特征 id 把
// "局部地图里的世界系 3D 点" 和 "当前帧跟踪到的 2D 点" 关联起来,
// 输出 PnP (3D-2D) 位姿估计需要的匹配对
class FrameMatch {
public:
    FrameMatch(float fx, float fy, float cx, float cy, float baseline);
    ~FrameMatch();

    // 处理一帧; 返回后 obj_pts_ / img_pts_ 即为本帧的 3D-2D 匹配对
    void process(const StereoFrame& frame, const LocalMap& map);

    // 收集当前帧深度有效的点 (反投影成相机系 3D 点), 供关键帧插入地图用
    void collectDepthPoints(std::vector<long>& ids_out,
                            std::vector<cv::Point3f>& pts_cam_out) const;

    // PnP 输入 (下标 k 一一对应):
    // obj_pts_[k] 世界系 3D 点, img_pts_[k] 它在当前帧的像素位置
    std::vector<cv::Point3f> obj_pts_;
    std::vector<cv::Point2f> img_pts_;
    std::vector<long> match_ids_;    // 每对匹配的特征 id (调试/可视化用)

    const FeatureTracker& tracker() const { return feature_tracker_; }

private:
    // 像素 + 深度 -> 相机系 3D 点
    cv::Point3f backproject(const cv::Point2f& uv, float z) const;

    FeatureTracker feature_tracker_;

    // 相机参数
    float fx_, fy_, cx_, cy_, baseline_;
};

