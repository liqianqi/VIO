#include "vo/camera.h"
#include "vo/frame.h"

#include <iostream>

// FrameMatch 功能测试: 实机取 60 帧, 检查每帧输出的 3D-2D 匹配对:
// 数组长度一致、3D 点深度为正、匹配数量合理 (静止场景下应接近跟踪点数)
int main() {
    D435Camera cam(640, 480, 30);
    FrameMatch matcher(cam.fx(), cam.fy(), cam.cx(), cam.cy(), cam.baseline());
    LocalMap map;

    StereoFrame frame;
    int failures = 0;
    const int num_frames = 60;

    for (int i = 0; i < num_frames; ++i) {
        cam.getStereoFrame(frame);
        matcher.process(frame, map);

        // 模拟关键帧: 第 1 帧把有深度的点入库 (相机静止, 相机系 == 世界系)
        if (i == 1) {
            std::vector<long> ids;
            std::vector<cv::Point3f> pts;
            matcher.collectDepthPoints(ids, pts);
            map.addPoints(ids, pts);
            std::printf("第 1 帧入库 %zu 个地图点\n", map.size());
        }

        if (matcher.obj_pts_.size() != matcher.img_pts_.size() ||
            matcher.obj_pts_.size() != matcher.match_ids_.size()) {
            std::cerr << "[FAIL] 帧 " << i << ": 匹配对数组长度不一致" << std::endl;
            ++failures;
        }
        for (const auto& p : matcher.obj_pts_)
            if (p.z <= 0) {
                std::cerr << "[FAIL] 帧 " << i << ": 出现深度非正的 3D 点" << std::endl;
                ++failures;
                break;
            }

        // 入库前地图为空, 匹配数应为 0
        if (i <= 1 && !matcher.obj_pts_.empty()) {
            std::cerr << "[FAIL] 帧 " << i << ": 地图为空却有匹配" << std::endl;
            ++failures;
        }
        // 入库后 (相机静止) 应该有相当数量的匹配
        if (i >= 2 && matcher.obj_pts_.size() < 30) {
            std::cerr << "[FAIL] 帧 " << i << ": 匹配对过少 ("
                      << matcher.obj_pts_.size() << ")" << std::endl;
            ++failures;
        }

        if (i % 10 == 0)
            std::printf("帧 %3d: 跟踪点 %3zu, 3D-2D 匹配对 %3zu\n",
                        i, matcher.tracker().cur_pts_.size(),
                        matcher.obj_pts_.size());
    }

    std::cout << (failures == 0 ? "[PASS] FrameMatch 全部测试通过"
                                : "[FAIL] 存在失败项")
              << std::endl;
    return failures == 0 ? 0 : 1;
}
