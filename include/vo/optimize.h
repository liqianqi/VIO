#pragma once

#include "vo/map.h"

#include <opencv2/core.hpp>
#include <vector>

// 手写高斯牛顿优化 (十四讲 7.8 / ch13 后端的核心练习)。
// 残差定义: r = proj(T_cw * p_w) - uv_观测, 目标 min Σ huber(|r|²)

// 位姿精化 (pose-only GN):
// 固定 3D 点, 只优化 T_cw (4x4 CV_64F, 输入作初值, 输出精化结果)。
// 用法: solvePnPRansac 挑出内点后, 在内点上调用本函数替代其内部精化。
// 返回 false 表示输入不足或数值异常, T_cw 保持原值。
bool poseGN(const std::vector<cv::Point3f>& pts_world,
            const std::vector<cv::Point2f>& pts_img,
            const cv::Mat& K, cv::Mat& T_cw,
            double huber_px = 2.0, int iters = 10);

// 滑动窗口局部 BA:
// 联合优化窗口内所有关键帧位姿 (第一个固定, 消除规范自由度) 和
// 被 >=2 个关键帧观测到的地图点, 高斯牛顿 + Schur 消元 + Huber 核。
// 残差: 左目重投影 (u,v) + 右目 u 坐标 (有双目匹配时), 后者钉住绝对尺度。
// 优化结果直接写回 map (关键帧 T_wc 和地图点坐标)。
void localBA(LocalMap& map, const cv::Mat& K, double baseline,
             double huber_px = 2.0, int iters = 8);
