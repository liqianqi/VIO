#include "vo/camera.h"
#include "vo/odometry.h"

#include <fstream>
#include <iostream>
#include <vector>

// 旋转矩阵 -> 四元数 (w, x, y, z), TUM 轨迹格式需要
static void rotToQuat(const cv::Mat& R, double q[4])
{
    double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
    if (trace > 0) {
        double s = std::sqrt(trace + 1.0) * 2;
        q[0] = 0.25 * s;
        q[1] = (R.at<double>(2, 1) - R.at<double>(1, 2)) / s;
        q[2] = (R.at<double>(0, 2) - R.at<double>(2, 0)) / s;
        q[3] = (R.at<double>(1, 0) - R.at<double>(0, 1)) / s;
    } else {
        // 对角元最大的分支 (相机不会倒着拿, 简单处理即可)
        int i = R.at<double>(1, 1) > R.at<double>(0, 0) ? 1 : 0;
        if (R.at<double>(2, 2) > R.at<double>(i, i)) i = 2;
        int j = (i + 1) % 3, k = (i + 2) % 3;
        double s = std::sqrt(R.at<double>(i, i) - R.at<double>(j, j) - R.at<double>(k, k) + 1.0) * 2;
        q[i + 1] = 0.25 * s;
        q[0] = (R.at<double>(k, j) - R.at<double>(j, k)) / s;
        q[j + 1] = (R.at<double>(j, i) + R.at<double>(i, j)) / s;
        q[k + 1] = (R.at<double>(k, i) + R.at<double>(i, k)) / s;
    }
}

// 帧对帧 PnP 视觉里程计实时演示, 按 q 或 ESC 退出:
// - 左窗口: 左目图像 + 特征点
// - 右窗口: 俯视图轨迹 (相机系 x 向右, z 向前), 网格间距 0.5m
// - 轨迹同时按 TUM 格式写入 traj_vo.txt, 可用 evo 工具分析
int main() {
    D435Camera cam(640, 480, 30);
    Odometry odom(cam.fx(), cam.fy(), cam.cx(), cam.cy(), cam.baseline());

    std::ofstream traj_file("traj_vo.txt");
    std::vector<cv::Point3d> path;

    const int canvas_size = 700;
    const double scale = 150.0;   // 像素/米, 视野约 ±2.3m

    StereoFrame frame;
    int frame_idx = 0, rejected = 0;
    while (true) {
        cam.getStereoFrame(frame);
        bool ok = odom.process(frame);
        if (!ok && frame_idx > 0) ++rejected;

        cv::Point3d pos = odom.position();
        path.push_back(pos);

        // TUM 格式: timestamp(s) tx ty tz qx qy qz qw
        double q[4];
        rotToQuat(odom.pose()(cv::Rect(0, 0, 3, 3)), q);
        traj_file << cv::format("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                frame.timestamp / 1000.0, pos.x, pos.y, pos.z,
                                q[1], q[2], q[3], q[0]);

        // ---- 左目 + 特征点 ----
        const auto& trk = odom.matcher().tracker();
        cv::Mat vis;
        cv::cvtColor(frame.left, vis, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < trk.cur_pts_.size(); ++i)
            cv::circle(vis, trk.cur_pts_[i], 2,
                       trk.depth_[i] > 0 ? cv::Scalar(0, 255, 0) : cv::Scalar(128, 128, 128), -1);
        cv::putText(vis,
                    cv::format("matches: %zu  inliers: %d  %s",
                               odom.matcher().obj_pts_.size(), odom.lastInliers(),
                               ok ? "OK" : "REJECT"),
                    {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    ok ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::imshow("VO camera", vis);

        // ---- 俯视图轨迹: 世界系 x 向右, z 向上(前方), 起点在画布中心 ----
        cv::Mat canvas(canvas_size, canvas_size, CV_8UC3, cv::Scalar(30, 30, 30));
        for (int m = -2; m <= 2; ++m) {   // 0.5m 网格
            int p = canvas_size / 2 + static_cast<int>(m * 0.5 * scale);
            cv::line(canvas, {p, 0}, {p, canvas_size}, {60, 60, 60});
            cv::line(canvas, {0, p}, {canvas_size, p}, {60, 60, 60});
        }
        auto toCanvas = [&](const cv::Point3d& p) {
            return cv::Point(canvas_size / 2 + static_cast<int>(p.x * scale),
                             canvas_size / 2 - static_cast<int>(p.z * scale));
        };
        for (size_t i = 1; i < path.size(); ++i)
            cv::line(canvas, toCanvas(path[i - 1]), toCanvas(path[i]), {0, 255, 255}, 2);
        cv::circle(canvas, toCanvas(pos), 5, {0, 0, 255}, -1);
        cv::putText(canvas,
                    cv::format("x: %+.3f  y: %+.3f  z: %+.3f m", pos.x, pos.y, pos.z),
                    {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 2);
        cv::putText(canvas,
                    cv::format("frames: %d  rejected: %d  KF: %d  map: %zu  BA: %.1fms",
                               frame_idx, rejected, odom.keyframeCount(), odom.mapSize(),
                               odom.lastBaMs()),
                    {10, 50}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 2);
        cv::imshow("VO trajectory (top-down, 0.5m grid)", canvas);

        if (frame_idx % 30 == 0)
            std::printf("帧 %4d: pos=(%+.3f, %+.3f, %+.3f) 匹配 %3zu 内点 %3d %s\n",
                        frame_idx, pos.x, pos.y, pos.z,
                        odom.matcher().obj_pts_.size(), odom.lastInliers(),
                        ok ? "" : "[拒绝]");

        ++frame_idx;
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    std::cout << "轨迹已保存到 traj_vo.txt (TUM 格式, 共 " << path.size() << " 帧)" << std::endl;
    return 0;
}
