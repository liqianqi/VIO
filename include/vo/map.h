#pragma once

#include <opencv2/core.hpp>
#include <deque>
#include <unordered_map>
#include <vector>

// 关键帧: 滑窗 BA 的优化节点。
// 除了位姿, 还要记录 "本帧看到了哪些地图点、在哪个像素位置" (观测),
// BA 的重投影残差就是逐条观测构建的。
struct Keyframe {
    int kf_id = 0;
    double timestamp = 0;
    cv::Mat T_wc;                        // 4x4 CV_64F, 相机 -> 世界 (BA 会修改)
    std::vector<long> pt_ids;            // 观测到的地图点 id
    std::vector<cv::Point2f> obs;        // 左目像素位置, 与 pt_ids 一一对应
    std::vector<float> u_right;          // 右目 u 坐标 (-1 表示无双目匹配)。
                                         // BA 用它约束绝对尺度, 否则纯左目
                                         // 重投影残差存在尺度零空间
};

class LocalMap{
public:
    LocalMap();
    ~LocalMap();

    // ---- 地图点 ----
    // 查询: id -> 世界系 3D 点, 查不到返回 false
    bool getPoint(long id, cv::Point3f& pw) const;
    // 关键帧插入: 只为地图里没有的新 id 添加点 (老点保持原坐标, 锚定在旧关键帧上)
    void addPoints(const std::vector<long>& ids,
                    const std::vector<cv::Point3f>& pts_world);
    // BA 优化结果写回
    void setPoint(long id, const cv::Point3f& pw);
    // 淘汰: 只保留还在跟踪中的 id (无关键帧窗口时的简易策略)
    void cull(const std::vector<long>& alive_ids);
    size_t size() const;

    // ---- 关键帧滑动窗口 ----
    // 插入关键帧; 超过窗口容量时滑出最老的, 并清掉只有它观测过的点
    void addKeyframe(Keyframe kf);
    std::deque<Keyframe>& keyframes() { return keyframes_; }
    const std::deque<Keyframe>& keyframes() const { return keyframes_; }

private:
    // 只保留窗口内关键帧观测过的点
    void cullByWindow();

    std::unordered_map<long, cv::Point3f> points_;  // id -> 世界系坐标
    std::deque<Keyframe> keyframes_;
    size_t window_size_ = 5;
};
