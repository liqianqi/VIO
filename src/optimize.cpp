#include "vo/optimize.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <unordered_map>

namespace {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat3 = Eigen::Matrix3d;
using Mat26 = Eigen::Matrix<double, 2, 6>;
using Mat23 = Eigen::Matrix<double, 2, 3>;

Mat3 hat(const Vec3& w)
{
    Mat3 W;
    W <<     0, -w.z(),  w.y(),
         w.z(),      0, -w.x(),
        -w.y(),  w.x(),      0;
    return W;
}

// SO(3) 指数映射 (罗德里格斯公式): 旋转向量 -> 旋转矩阵
Mat3 expSO3(const Vec3& w)
{
    double th = w.norm();
    Mat3 W = hat(w);
    if (th < 1e-10) return Mat3::Identity() + W;
    return Mat3::Identity() + std::sin(th) / th * W +
           (1 - std::cos(th)) / (th * th) * W * W;
}

// 相机位姿 T_cw 的内部表示
struct Pose {
    Mat3 R;
    Vec3 t;
    Vec3 map(const Vec3& pw) const { return R * pw + t; }
    // 左乘扰动更新: T <- exp([dphi]) * T + drho,
    // 即 R <- exp(dphi) R, t <- exp(dphi) t + drho
    void update(const Vec6& dx)
    {
        Mat3 dR = expSO3(dx.tail<3>());
        R = dR * R;
        t = dR * t + dx.head<3>();
    }
};

Pose fromCv(const cv::Mat& T)
{
    Pose p;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) p.R(r, c) = T.at<double>(r, c);
        p.t(r) = T.at<double>(r, 3);
    }
    return p;
}

void toCv(const Pose& p, cv::Mat& T)
{
    T = cv::Mat::eye(4, 4, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T.at<double>(r, c) = p.R(r, c);
        T.at<double>(r, 3) = p.t(r);
    }
}

// 重投影残差 r = proj(P') - uv 及其对相机系点 P' 的雅可比 (2x3)
// 返回 false 表示点在相机后方, 该观测不可用
bool reproject(const Vec3& pc, const Vec2& uv, double fx, double fy,
               double cx, double cy, Vec2& r, Mat23& J_pc)
{
    if (pc.z() <= 1e-6) return false;
    double invz = 1.0 / pc.z();
    double invz2 = invz * invz;
    r << fx * pc.x() * invz + cx - uv.x(),
         fy * pc.y() * invz + cy - uv.y();
    J_pc << fx * invz, 0, -fx * pc.x() * invz2,
            0, fy * invz, -fy * pc.y() * invz2;
    return true;
}

// Huber 核权重: |r| <= delta 时为 1, 否则 delta/|r| (削弱大残差的话语权)
double huberWeight(double r_norm, double delta)
{
    return r_norm <= delta ? 1.0 : delta / r_norm;
}

}  // namespace

bool poseGN(const std::vector<cv::Point3f>& pts_world,
            const std::vector<cv::Point2f>& pts_img,
            const cv::Mat& K, cv::Mat& T_cw,
            double huber_px, int iters)
{
    if (pts_world.size() < 6 || pts_world.size() != pts_img.size())
        return false;

    const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);

    Pose pose = fromCv(T_cw);
    double last_chi2 = std::numeric_limits<double>::max();

    for (int it = 0; it < iters; ++it) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vec6 b = Vec6::Zero();
        double chi2 = 0;

        for (size_t i = 0; i < pts_world.size(); ++i) {
            Vec3 pw(pts_world[i].x, pts_world[i].y, pts_world[i].z);
            Vec2 uv(pts_img[i].x, pts_img[i].y);
            Vec3 pc = pose.map(pw);

            Vec2 r;
            Mat23 J_pc;
            if (!reproject(pc, uv, fx, fy, cx, cy, r, J_pc)) continue;

            // 位姿雅可比 (2x6): dP'/d[drho, dphi] = [I, -hat(P')]
            Mat26 J;
            J.leftCols<3>() = J_pc;
            J.rightCols<3>() = -J_pc * hat(pc);

            double w = huberWeight(r.norm(), huber_px);
            H += w * J.transpose() * J;
            b += -w * J.transpose() * r;
            chi2 += w * r.squaredNorm();
        }

        H.diagonal().array() += 1e-8;   // 数值保底
        Vec6 dx = H.ldlt().solve(b);
        if (!dx.allFinite()) return false;

        pose.update(dx);

        if (dx.norm() < 1e-8 || std::abs(last_chi2 - chi2) < 1e-6 * chi2)
            break;
        last_chi2 = chi2;
    }

    toCv(pose, T_cw);
    return true;
}

void localBA(LocalMap& map, const cv::Mat& K, double baseline,
             double huber_px, int iters)
{
    auto& kfs = map.keyframes();
    const int n_kf = static_cast<int>(kfs.size());
    if (n_kf < 2) return;

    const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);

    // ---- 组织优化变量 ----
    // 位姿: 窗口内全部关键帧的 T_cw, 第 0 个固定 (规范自由度), 其余各 6 维
    std::vector<Pose> poses(n_kf);
    for (int k = 0; k < n_kf; ++k) {
        Pose p_wc = fromCv(kfs[k].T_wc);
        poses[k].R = p_wc.R.transpose();          // T_cw = T_wc^-1
        poses[k].t = -poses[k].R * p_wc.t;
    }
    const int n_pose_var = n_kf - 1;              // 被优化的位姿数
    auto poseIdx = [](int k) { return k - 1; };   // 关键帧 k -> 变量块下标

    // 地图点: 只优化被 >=2 个关键帧观测的点 (单次观测无法约束 3 维坐标)
    std::unordered_map<long, int> obs_cnt;
    for (const auto& kf : kfs)
        for (long id : kf.pt_ids) ++obs_cnt[id];

    std::vector<long> pt_ids;                     // 变量块下标 -> 点 id
    std::unordered_map<long, int> pt_idx;         // 点 id -> 变量块下标
    std::vector<Vec3> pts;
    cv::Point3f pw_cv;
    for (const auto& [id, cnt] : obs_cnt) {
        if (cnt < 2 || !map.getPoint(id, pw_cv)) continue;
        pt_idx[id] = static_cast<int>(pt_ids.size());
        pt_ids.push_back(id);
        pts.emplace_back(pw_cv.x, pw_cv.y, pw_cv.z);
    }
    const int n_pt = static_cast<int>(pts.size());
    if (n_pt < 10) return;

    // ---- 高斯牛顿迭代, Schur 消元求解 ----
    // 正规方程 [Hpp Hpl; Hlp Hll] [dp; dl] = [bp; bl],
    // Hll 是逐点 3x3 块对角 -> 先消去点增量:
    //   S = Hpp - Hpl Hll^-1 Hlp,  bs = bp - Hpl Hll^-1 bl
    //   dp = S^-1 bs,  dl_j = Hll_j^-1 (bl_j - Hlp_j dp)
    const int dim_p = 6 * n_pose_var;
    for (int it = 0; it < iters; ++it) {
        Eigen::MatrixXd Hpp = Eigen::MatrixXd::Zero(dim_p, dim_p);
        Eigen::VectorXd bp = Eigen::VectorXd::Zero(dim_p);
        std::vector<Mat3> Hll(n_pt, Mat3::Zero());
        std::vector<Vec3> bl(n_pt, Vec3::Zero());
        // Hpl 的稀疏存储: 每个点 j -> {(位姿块下标, 6x3 块)}
        std::vector<std::vector<std::pair<int, Eigen::Matrix<double, 6, 3>>>> Hpl(n_pt);

        for (int k = 0; k < n_kf; ++k) {
            const Keyframe& kf = kfs[k];
            for (size_t o = 0; o < kf.pt_ids.size(); ++o) {
                auto it_pt = pt_idx.find(kf.pt_ids[o]);
                if (it_pt == pt_idx.end()) continue;
                int j = it_pt->second;

                Vec3 pc = poses[k].map(pts[j]);
                Vec2 uv(kf.obs[o].x, kf.obs[o].y);
                Vec2 r;
                Mat23 J_pc;
                if (!reproject(pc, uv, fx, fy, cx, cy, r, J_pc)) continue;

                double w = huberWeight(r.norm(), huber_px);

                // 右目 u 坐标残差 (有双目匹配时): 右相机在 +x 方向 baseline 处,
                // u_r = fx*(X-b)/Z + cx。它对深度 (即尺度) 敏感, 用来钉住
                // 纯左目重投影固有的尺度零空间
                bool has_right = o < kf.u_right.size() && kf.u_right[o] >= 0;
                double r_r = 0, w_r = 0;
                Eigen::RowVector3d Jr_pc;
                if (has_right) {
                    double invz = 1.0 / pc.z(), invz2 = invz * invz;
                    r_r = fx * (pc.x() - baseline) * invz + cx - kf.u_right[o];
                    Jr_pc << fx * invz, 0, -fx * (pc.x() - baseline) * invz2;
                    w_r = huberWeight(std::abs(r_r), huber_px);
                }

                Mat23 J_pt = J_pc * poses[k].R;   // dP'/dPw = R_cw
                Eigen::RowVector3d Jr_pt = has_right
                    ? Eigen::RowVector3d(Jr_pc * poses[k].R) : Eigen::RowVector3d::Zero();

                // 点块
                Hll[j] += w * J_pt.transpose() * J_pt;
                bl[j] += -w * J_pt.transpose() * r;
                if (has_right) {
                    Hll[j] += w_r * Jr_pt.transpose() * Jr_pt;
                    bl[j] += -w_r * Jr_pt.transpose() * r_r;
                }

                if (k == 0) continue;             // 第 0 帧位姿固定, 无位姿块
                int pi = poseIdx(k);
                Mat26 J_pose;
                J_pose.leftCols<3>() = J_pc;
                J_pose.rightCols<3>() = -J_pc * hat(pc);

                Hpp.block<6, 6>(6 * pi, 6 * pi) += w * J_pose.transpose() * J_pose;
                bp.segment<6>(6 * pi) += -w * J_pose.transpose() * r;
                Eigen::Matrix<double, 6, 3> Hpl_blk = w * J_pose.transpose() * J_pt;

                if (has_right) {
                    Eigen::Matrix<double, 1, 6> Jr_pose;
                    Jr_pose.leftCols<3>() = Jr_pc;
                    Jr_pose.rightCols<3>() = -Jr_pc * hat(pc);
                    Hpp.block<6, 6>(6 * pi, 6 * pi) += w_r * Jr_pose.transpose() * Jr_pose;
                    bp.segment<6>(6 * pi) += -w_r * Jr_pose.transpose() * r_r;
                    Hpl_blk += w_r * Jr_pose.transpose() * Jr_pt;
                }
                Hpl[j].push_back({pi, Hpl_blk});
            }
        }

        // Schur 补
        Eigen::MatrixXd S = Hpp;
        Eigen::VectorXd bs = bp;
        std::vector<Mat3> Hll_inv(n_pt);
        for (int j = 0; j < n_pt; ++j) {
            Mat3 Hj = Hll[j];
            Hj.diagonal().array() += 1e-6;
            Hll_inv[j] = Hj.inverse();
            for (const auto& [pi_a, Hpl_a] : Hpl[j]) {
                bs.segment<6>(6 * pi_a) -= Hpl_a * Hll_inv[j] * bl[j];
                for (const auto& [pi_b, Hpl_b] : Hpl[j])
                    S.block<6, 6>(6 * pi_a, 6 * pi_b) -=
                        Hpl_a * Hll_inv[j] * Hpl_b.transpose();
            }
        }

        S.diagonal().array() += 1e-8;
        Eigen::VectorXd dp = S.ldlt().solve(bs);
        if (!dp.allFinite()) return;

        // 回代求点增量并更新
        for (int j = 0; j < n_pt; ++j) {
            Vec3 rhs = bl[j];
            for (const auto& [pi, Hpl_j] : Hpl[j])
                rhs -= Hpl_j.transpose() * dp.segment<6>(6 * pi);
            pts[j] += Hll_inv[j] * rhs;
        }
        for (int k = 1; k < n_kf; ++k)
            poses[k].update(dp.segment<6>(6 * poseIdx(k)));

        if (dp.norm() < 1e-8) break;
    }

    // ---- 写回地图 ----
    for (int k = 1; k < n_kf; ++k) {
        Pose p_wc;
        p_wc.R = poses[k].R.transpose();          // T_wc = T_cw^-1
        p_wc.t = -p_wc.R * poses[k].t;
        toCv(p_wc, kfs[k].T_wc);
    }
    for (int j = 0; j < n_pt; ++j)
        map.setPoint(pt_ids[j], { static_cast<float>(pts[j].x()),
                                  static_cast<float>(pts[j].y()),
                                  static_cast<float>(pts[j].z()) });
}
