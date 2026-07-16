#include "vo/feature.h"

#include <algorithm>
#include <numeric>

namespace {

// OpenCV 没有现成的 inBorder, 自己写: 距图像边缘至少 border 像素
bool inBorder(const cv::Point2f& pt, const cv::Size& size, int border = 3)
{
    return pt.x >= border && pt.x < size.width - border &&
           pt.y >= border && pt.y < size.height - border;
}

// 按 status 同步压缩数组: 保留 status[i] != 0 的元素, 保持相对顺序
template <typename T>
void reduceVector(std::vector<T>& v, const std::vector<uchar>& status)
{
    size_t j = 0;
    for (size_t i = 0; i < v.size(); ++i)
        if (status[i]) v[j++] = v[i];
    v.resize(j);
}

}  // namespace

FeatureTracker::FeatureTracker() {}

FeatureTracker::~FeatureTracker() {}

void FeatureTracker::track(const cv::Mat& img)
{
    // ========== ①② 光流跟踪老点 (第一帧或点全丢时跳过, 直接去④提点) ==========
    if (!img_prev_.empty() && !prev_pts_.empty()) {
        // ① LK 前向跟踪: prev_pts_ -> cur_pts_
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(img_prev_, img, prev_pts_, cur_pts_, status, err,
                                 cv::Size(21, 21), 3);   // 21x21 窗口, 3 层金字塔

        // ② 反向光流检查: cur_pts_ -> back_pts, 回不到原位的判为误跟踪
        std::vector<cv::Point2f> back_pts;
        std::vector<uchar> back_status;
        cv::calcOpticalFlowPyrLK(img, img_prev_, cur_pts_, back_pts, back_status, err,
                                 cv::Size(21, 21), 3);
        for (size_t i = 0; i < status.size(); ++i)
            status[i] = status[i] && back_status[i] &&
                        cv::norm(prev_pts_[i] - back_pts[i]) <= 0.5 &&
                        inBorder(cur_pts_[i], img.size());

        // ③ 四个数组按 status 同步压缩, 存活的点跟踪计数 +1
        reduceVector(prev_pts_, status);
        reduceVector(cur_pts_, status);
        reduceVector(ids_, status);
        reduceVector(track_cnt_, status);
        for (auto& cnt : track_cnt_) ++cnt;
    } else {
        cur_pts_.clear();
        ids_.clear();
        track_cnt_.clear();
    }

    // ========== ④ mask 均匀化 + 补点 ==========
    // 按 track_cnt 从大到小处理: 跟踪越久的点越稳定, 优先保留;
    // 每保留一个点就在 mask 上挖掉它周围 min_dist_ 的圆, 落进圆里的后来者被淘汰
    cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(255));

    std::vector<size_t> order(cur_pts_.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return track_cnt_[a] > track_cnt_[b]; });

    std::vector<cv::Point2f> kept_pts;
    std::vector<long> kept_ids;
    std::vector<int> kept_cnt;
    for (size_t idx : order) {
        const cv::Point2f& pt = cur_pts_[idx];
        if (mask.at<uchar>(cvRound(pt.y), cvRound(pt.x)) == 0) continue;  // 太挤, 淘汰
        kept_pts.push_back(pt);
        kept_ids.push_back(ids_[idx]);
        kept_cnt.push_back(track_cnt_[idx]);
        cv::circle(mask, pt, min_dist_, 0, -1);
    }
    cur_pts_ = std::move(kept_pts);
    ids_ = std::move(kept_ids);
    track_cnt_ = std::move(kept_cnt);

    // 点数不足则在 mask 白区补提新点, 新点发新 id
    int need = max_cnt_ - static_cast<int>(cur_pts_.size());
    if (need > 0) {
        std::vector<cv::Point2f> new_pts;
        cv::goodFeaturesToTrack(img, new_pts, need, 0.01, min_dist_, mask);
        for (const auto& pt : new_pts) {
            cur_pts_.push_back(pt);
            ids_.push_back(next_id_++);
            track_cnt_.push_back(1);
        }
    }

    // ========== ⑤ 滚动状态, 为下一帧做准备 ==========
    // img 可能是零拷贝 Mat (数据下次取帧就失效), 跨帧保留必须 clone
    img_prev_ = img.clone();
    prev_pts_ = cur_pts_;
}


void FeatureTracker::matchStereo(const cv::Mat& left, const cv::Mat& right,
    float fx, float baseline)
{
    right_pts_.clear();
    right_pts_ = cur_pts_;      // 用左图位置做初值(视差不大时能加速收敛)
    std::vector<uchar> status;  
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(left, right, cur_pts_, right_pts_, status, err,
                             cv::Size(21, 21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                              30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    
    // ② 反向光流检查: right_pts_ -> back_pts, 回不到原位的判为误跟踪
    std::vector<cv::Point2f> back_pts;
    std::vector<uchar> back_status;
    cv::calcOpticalFlowPyrLK(right, left, right_pts_, back_pts, back_status, err,
                                cv::Size(21, 21), 3);
    for (size_t i = 0; i < status.size(); ++i)
        status[i] = status[i] && back_status[i] &&
                cv::norm(cur_pts_[i] - back_pts[i]) <= 0.5 &&
                inBorder(right_pts_[i], right.size());

    // ③④ 几何检查 + 视差转深度
    // 左右点的对应关系就是数组下标: right_pts_[i] 是 cur_pts_[i] 在右图的匹配,
    // 这是 calcOpticalFlowPyrLK 保证的(输出与输入一一对应)。
    // 所以这里绝不能 reduceVector(只压缩一个数组就和 cur_pts_/ids_ 错位了),
    // 匹配失败的点保留原位, 深度标 -1 表示无效即可
    depth_.assign(cur_pts_.size(), -1.0f);
    for (size_t i = 0; i < cur_pts_.size(); ++i) {
        if (!status[i]) continue;

        float dv = std::abs(cur_pts_[i].y - right_pts_[i].y);   // 行对齐: 应在同一行
        float disparity = cur_pts_[i].x - right_pts_[i].x;      // 视差: 左图 u 更大
        if (dv < 2.0f && disparity > 1.0f)
            depth_[i] = fx * baseline / disparity;
    }
}

