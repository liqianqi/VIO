#include "vo/camera.h"
#include "vo/feature.h"

#include <deque>
#include <iostream>
#include <map>

// 深度 -> 伪彩色: 0.3m (红) ~ 5m (蓝), 无效返回灰色
cv::Scalar depthColor(float z)
{
    if (z < 0) return {128, 128, 128};
    const float z_min = 0.3f, z_max = 5.0f;
    float t = std::min(1.0f, std::max(0.0f, (z - z_min) / (z_max - z_min)));
    // 红(近) -> 绿 -> 蓝(远)
    if (t < 0.5f) return {0, 255 * t * 2, 255 * (1 - t * 2)};
    return {255 * (t - 0.5f) * 2, 255 * (1 - (t - 0.5f) * 2), 0};
}

// FeatureTracker 实时可视化测试, 按 q 或 ESC 退出:
// - 上图 (左目): 特征点按深度伪彩色 (近红远蓝, 灰=无深度) + 最近 10 帧轨迹线,
//   每隔几个点标注深度读数, 左上角显示点数/存活率/深度有效率
// - 下图 (右目): 双目匹配位置
int main() {
    D435Camera cam(640, 480, 30);
    FeatureTracker tracker;

    std::map<long, std::deque<cv::Point2f>> trails;  // 每个 id 最近若干帧的位置
    const size_t trail_len = 10;

    StereoFrame frame;
    while (true) {
        cam.getStereoFrame(frame);

        size_t prev_num = tracker.cur_pts_.size();
        tracker.track(frame.left);
        tracker.matchStereo(frame.left, frame.right, cam.fx(), cam.baseline());

        // 存活率 = 上一帧的点里活到这一帧的比例 (track_cnt > 1 的是老点)
        size_t survived = 0;
        for (int cnt : tracker.track_cnt_)
            if (cnt > 1) ++survived;
        double survival = prev_num > 0 ? 100.0 * survived / prev_num : 0.0;

        // 深度有效率 = 双目匹配成功拿到深度的点的比例
        size_t valid_depth = 0;
        for (float z : tracker.depth_)
            if (z > 0) ++valid_depth;
        double depth_rate = tracker.depth_.empty()
                                ? 0.0
                                : 100.0 * valid_depth / tracker.depth_.size();

        // 更新轨迹缓存, 并清掉已丢失的 id
        std::map<long, std::deque<cv::Point2f>> alive;
        for (size_t i = 0; i < tracker.cur_pts_.size(); ++i) {
            auto& dq = trails[tracker.ids_[i]];
            dq.push_back(tracker.cur_pts_[i]);
            if (dq.size() > trail_len) dq.pop_front();
            alive[tracker.ids_[i]] = dq;
        }
        trails.swap(alive);

        // ---- 左目: 轨迹 + 深度伪彩色点 + 深度读数 ----
        cv::Mat vis_left;
        cv::cvtColor(frame.left, vis_left, cv::COLOR_GRAY2BGR);

        for (const auto& [id, dq] : trails)
            for (size_t k = 1; k < dq.size(); ++k)
                cv::line(vis_left, dq[k - 1], dq[k], {0, 255, 0}, 1);

        for (size_t i = 0; i < tracker.cur_pts_.size(); ++i) {
            cv::circle(vis_left, tracker.cur_pts_[i], 3,
                       depthColor(tracker.depth_[i]), -1);
            // 每 8 个点标一个深度读数, 避免画面太乱
            if (tracker.depth_[i] > 0 && i % 8 == 0)
                cv::putText(vis_left, cv::format("%.2f", tracker.depth_[i]),
                            tracker.cur_pts_[i] + cv::Point2f(5, -5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
        }

        cv::putText(vis_left,
                    cv::format("pts: %zu  survival: %.1f%%  depth: %.1f%%",
                               tracker.cur_pts_.size(), survival, depth_rate),
                    {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 255}, 2);

        // ---- 右目: 匹配到的位置 ----
        cv::Mat vis_right;
        cv::cvtColor(frame.right, vis_right, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < tracker.cur_pts_.size(); ++i)
            if (tracker.depth_[i] > 0)
                cv::circle(vis_right, tracker.right_pts_[i], 3,
                           depthColor(tracker.depth_[i]), -1);

        cv::Mat vis;
        cv::vconcat(vis_left, vis_right, vis);
        cv::imshow("Stereo Tracking: left(top) right(bottom), q to quit", vis);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }
    return 0;
}
