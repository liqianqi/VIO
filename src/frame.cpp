#include "vo/frame.h"

FrameMatch::FrameMatch(float fx, float fy, float cx, float cy, float baseline)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline)
{
}

FrameMatch::~FrameMatch() {}

cv::Point3f FrameMatch::backproject(const cv::Point2f& uv, float z) const
{
    return { (uv.x - cx_) / fx_ * z,
             (uv.y - cy_) / fy_ * z,
             z };
}

void FrameMatch::process(const StereoFrame& frame, const LocalMap& map)
{
    feature_tracker_.track(frame.left);
    feature_tracker_.matchStereo(frame.left, frame.right, fx_, baseline_);

    const auto& ids = feature_tracker_.ids_;
    const auto& pts = feature_tracker_.cur_pts_;

    // 按 id 去局部地图查世界系 3D 点, 查得到的构成一对 3D-2D 匹配
    obj_pts_.clear();
    img_pts_.clear();
    match_ids_.clear();
    cv::Point3f pw;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!map.getPoint(ids[i], pw)) continue;   // 还没进地图的新点
        obj_pts_.push_back(pw);
        img_pts_.push_back(pts[i]);
        match_ids_.push_back(ids[i]);
    }
}

void FrameMatch::collectDepthPoints(std::vector<long>& ids_out,
                                    std::vector<cv::Point3f>& pts_cam_out) const
{
    const auto& ids = feature_tracker_.ids_;
    const auto& pts = feature_tracker_.cur_pts_;
    const auto& depth = feature_tracker_.depth_;

    ids_out.clear();
    pts_cam_out.clear();
    for (size_t i = 0; i < ids.size(); ++i) {
        if (depth[i] <= 0) continue;
        ids_out.push_back(ids[i]);
        pts_cam_out.push_back(backproject(pts[i], depth[i]));
    }
}
