#include "vo/map.h"
#include <algorithm>
#include <unordered_set>


LocalMap::LocalMap() {}
LocalMap::~LocalMap() {}

bool LocalMap::getPoint(long id, cv::Point3f& pw) const {
    auto it = points_.find(id);
    if (it == points_.end()) return false;
    pw = it->second;
    return true;
}

void LocalMap::addPoints(const std::vector<long>& ids, const std::vector<cv::Point3f>& pts_world) {
    for (size_t i = 0; i < ids.size(); ++i) {
        points_.emplace(ids[i], pts_world[i]);   // id 已存在时什么都不做
    }
}

void LocalMap::setPoint(long id, const cv::Point3f& pw) {
    auto it = points_.find(id);
    if (it != points_.end()) it->second = pw;
}

void LocalMap::cull(const std::vector<long>& alive_ids) {
    for (auto it = points_.begin(); it != points_.end();) {
        if (std::find(alive_ids.begin(), alive_ids.end(), it->first) == alive_ids.end()) {
            it = points_.erase(it);
        } else {
            ++it;
        }
    }
}

size_t LocalMap::size() const {
    return points_.size();
}

void LocalMap::addKeyframe(Keyframe kf) {
    keyframes_.push_back(std::move(kf));
    if (keyframes_.size() > window_size_)
        keyframes_.pop_front();
    cullByWindow();
}

void LocalMap::cullByWindow() {
    std::unordered_set<long> observed;
    for (const auto& kf : keyframes_)
        observed.insert(kf.pt_ids.begin(), kf.pt_ids.end());

    for (auto it = points_.begin(); it != points_.end();) {
        if (observed.count(it->first) == 0)
            it = points_.erase(it);
        else
            ++it;
    }
}
