#include "vo/odometry.h"
#include "vo/optimize.h"

#include <chrono>

namespace {

// 两个位姿间的相对平移 (米) 和旋转角 (度)
void poseDelta(const cv::Mat& T_a, const cv::Mat& T_b, double& trans, double& rot_deg)
{
    cv::Mat Ra = T_a(cv::Rect(0, 0, 3, 3)), Rb = T_b(cv::Rect(0, 0, 3, 3));
    cv::Mat ta = T_a(cv::Rect(3, 0, 1, 3)), tb = T_b(cv::Rect(3, 0, 1, 3));
    trans = cv::norm(ta - tb);
    // 相对旋转角: acos((trace(Ra^T * Rb) - 1) / 2)
    double tr = cv::trace(Ra.t() * Rb)[0];
    double c = std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0));
    rot_deg = std::acos(c) * 180.0 / CV_PI;
}

// SE3 求逆: [R|t]^-1 = [R^T | -R^T t]
cv::Mat invertT(const cv::Mat& T)
{
    cv::Mat Ti = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat Rt = T(cv::Rect(0, 0, 3, 3)).t();
    Rt.copyTo(Ti(cv::Rect(0, 0, 3, 3)));
    cv::Mat t = -Rt * T(cv::Rect(3, 0, 1, 3));
    t.copyTo(Ti(cv::Rect(3, 0, 1, 3)));
    return Ti;
}

}  // namespace

Odometry::Odometry(float fx, float fy, float cx, float cy, float baseline)
    : matcher_(fx, fy, cx, cy, baseline),
      K_((cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1)),
      baseline_(baseline),
      T_wc_(cv::Mat::eye(4, 4, CV_64F)),
      T_wc_kf_(cv::Mat::eye(4, 4, CV_64F))
{
}

cv::Point3d Odometry::position() const
{
    return { T_wc_.at<double>(0, 3), T_wc_.at<double>(1, 3), T_wc_.at<double>(2, 3) };
}

void Odometry::insertKeyframe(double timestamp)
{
    std::vector<long> ids;
    std::vector<cv::Point3f> pts_cam;
    matcher_.collectDepthPoints(ids, pts_cam);
    if (pts_cam.empty()) return;

    // p_w = R_wc * p_c + t_wc
    cv::Mat R = T_wc_(cv::Rect(0, 0, 3, 3));
    cv::Mat t = T_wc_(cv::Rect(3, 0, 1, 3));
    std::vector<cv::Point3f> pts_world(pts_cam.size());
    for (size_t i = 0; i < pts_cam.size(); ++i) {
        cv::Mat pc = (cv::Mat_<double>(3, 1) << pts_cam[i].x, pts_cam[i].y, pts_cam[i].z);
        cv::Mat pw = R * pc + t;
        pts_world[i] = { static_cast<float>(pw.at<double>(0)),
                         static_cast<float>(pw.at<double>(1)),
                         static_cast<float>(pw.at<double>(2)) };
    }
    map_.addPoints(ids, pts_world);            // 老 id 不覆盖, 锚定在旧关键帧

    // 组装关键帧观测: 当前帧跟踪到的、在地图里有坐标的点
    Keyframe kf;
    kf.kf_id = kf_count_;
    kf.timestamp = timestamp;
    kf.T_wc = T_wc_.clone();
    const auto& trk = matcher_.tracker();
    cv::Point3f pw;
    for (size_t i = 0; i < trk.ids_.size(); ++i) {
        if (!map_.getPoint(trk.ids_[i], pw)) continue;
        kf.pt_ids.push_back(trk.ids_[i]);
        kf.obs.push_back(trk.cur_pts_[i]);
        // 右目观测 (给 BA 钉尺度用), 无双目匹配标 -1
        kf.u_right.push_back(trk.depth_[i] > 0 ? trk.right_pts_[i].x : -1.0f);
    }
    map_.addKeyframe(std::move(kf));           // 滑窗淘汰 + 按窗口清理地图点
    ++kf_count_;

    // 滑窗局部 BA: 联合优化窗口内关键帧位姿和地图点, 然后取回本帧位姿
    auto t0 = std::chrono::steady_clock::now();
    localBA(map_, K_, baseline_);
    last_ba_ms_ = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();

    T_wc_ = map_.keyframes().back().T_wc.clone();
    T_wc_kf_ = T_wc_.clone();
}

bool Odometry::process(const StereoFrame& frame)
{
    matcher_.process(frame, map_);
    last_inliers_ = 0;

    // 启动: 地图还是空的 (第一帧, 或跟踪彻底丢失后地图被清空),
    // 直接把当前帧立为关键帧, 位姿沿用当前估计
    if (map_.size() == 0) {
        insertKeyframe(frame.timestamp);
        return kf_count_ > 0;
    }

    if (static_cast<int>(matcher_.obj_pts_.size()) < min_matches_)
        return false;

    // 用上一帧位姿做初值: rvec/tvec 是 T_cw = T_wc^-1
    cv::Mat T_cw = invertT(T_wc_);
    cv::Mat rvec, tvec = T_cw(cv::Rect(3, 0, 1, 3)).clone(), inliers;
    cv::Rodrigues(T_cw(cv::Rect(0, 0, 3, 3)), rvec);

    // RANSAC 只负责挑内点 (obj_pts_ 是世界系点, 解的是绝对位姿 T_cw)
    bool ok = cv::solvePnPRansac(matcher_.obj_pts_, matcher_.img_pts_, K_,
                                 cv::noArray(),   // IR 图出厂已去畸变
                                 rvec, tvec, /*useExtrinsicGuess=*/true,
                                 100, 2.0, 0.99, inliers);
    if (!ok || inliers.rows < min_inliers_)
        return false;

    // 手写 GN 在内点上精化位姿 (Huber 核继续压制漏网外点)
    std::vector<cv::Point3f> obj_in(inliers.rows);
    std::vector<cv::Point2f> img_in(inliers.rows);
    for (int i = 0; i < inliers.rows; ++i) {
        int idx = inliers.at<int>(i);
        obj_in[i] = matcher_.obj_pts_[idx];
        img_in[i] = matcher_.img_pts_[idx];
    }
    cv::Mat R_cw;
    cv::Rodrigues(rvec, R_cw);
    R_cw.copyTo(T_cw(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(T_cw(cv::Rect(3, 0, 1, 3)));
    poseGN(obj_in, img_in, K_, T_cw);

    cv::Mat T_wc_new = invertT(T_cw);

    // 运动合理性检查: 相邻帧 (33ms) 不可能有大跳变, 超限说明解飞了
    double trans, rot_deg;
    poseDelta(T_wc_, T_wc_new, trans, rot_deg);
    if (trans > max_trans_ || rot_deg > max_rot_deg_)
        return false;

    last_inliers_ = inliers.rows;
    T_wc_ = T_wc_new;

    // 关键帧判据: 跟地图的联系变弱, 或离上个关键帧走远/转多了
    double kf_trans, kf_rot;
    poseDelta(T_wc_kf_, T_wc_, kf_trans, kf_rot);
    if (last_inliers_ < kf_min_inliers_ || kf_trans > kf_trans_ || kf_rot > kf_rot_deg_)
        insertKeyframe(frame.timestamp);

    return true;
}
