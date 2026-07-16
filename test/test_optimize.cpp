#include "vo/optimize.h"

#include <opencv2/calib3d.hpp>

#include <iostream>
#include <random>

// 手写优化器的合成数据测试 (不需要相机):
// 1. poseGN: 生成真值位姿和 3D 点, 投影加噪声得观测,
//    从扰动的初值出发优化, 检查恢复精度
// 2. localBA: 构造 3 个关键帧的滑窗, 扰动位姿和点后跑 BA,
//    检查位姿/点误差是否显著下降
// 全程与真值对比, 这是验证雅可比和 Schur 消元写对了的最直接手段

static std::mt19937 rng(42);

static cv::Mat makeT(double rx, double ry, double rz, double tx, double ty, double tz)
{
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << rx, ry, rz), R;
    cv::Rodrigues(rvec, R);
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    T.at<double>(0, 3) = tx;
    T.at<double>(1, 3) = ty;
    T.at<double>(2, 3) = tz;
    return T;
}

static cv::Point2f project(const cv::Mat& T_cw, const cv::Point3f& pw,
                           double fx, double fy, double cx, double cy)
{
    cv::Mat p = (cv::Mat_<double>(3, 1) << pw.x, pw.y, pw.z);
    cv::Mat pc = T_cw(cv::Rect(0, 0, 3, 3)) * p + T_cw(cv::Rect(3, 0, 1, 3));
    double invz = 1.0 / pc.at<double>(2);
    return { static_cast<float>(fx * pc.at<double>(0) * invz + cx),
             static_cast<float>(fy * pc.at<double>(1) * invz + cy) };
}

static double poseError(const cv::Mat& T_a, const cv::Mat& T_b)
{
    return cv::norm(T_a(cv::Rect(3, 0, 1, 3)) - T_b(cv::Rect(3, 0, 1, 3)));
}

int main() {
    const double fx = 387.9, fy = 387.9, cx = 326.0, cy = 235.6;
    const double baseline = 0.05;
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    std::uniform_real_distribution<double> ux(-2, 2), uy(-1.5, 1.5), uz(1.0, 6.0);
    std::normal_distribution<double> pix_noise(0, 0.2);   // 0.2px 观测噪声

    int failures = 0;

    // ---------- 测试 1: poseGN ----------
    {
        cv::Mat T_gt = makeT(0.05, -0.1, 0.03, 0.2, -0.1, 0.15);   // 真值 T_cw

        std::vector<cv::Point3f> pws;
        std::vector<cv::Point2f> uvs;
        for (int i = 0; i < 100; ++i) {
            cv::Point3f pw(ux(rng), uy(rng), uz(rng));
            cv::Point2f uv = project(T_gt, pw, fx, fy, cx, cy);
            pws.push_back(pw);
            uvs.push_back({ uv.x + static_cast<float>(pix_noise(rng)),
                            uv.y + static_cast<float>(pix_noise(rng)) });
        }

        // 初值: 真值上加 5cm / 3 度左右的扰动
        cv::Mat T_init = makeT(0.10, -0.05, 0.06, 0.25, -0.05, 0.10);
        double err_before = poseError(T_init, T_gt);

        cv::Mat T_opt = T_init.clone();
        bool ok = poseGN(pws, uvs, K, T_opt);
        double err_after = poseError(T_opt, T_gt);

        std::printf("poseGN:  平移误差 %.4f m -> %.4f m\n", err_before, err_after);
        if (!ok || err_after > 0.005) {   // 0.5px 噪声下应恢复到毫米级
            std::cerr << "[FAIL] poseGN 精度不达标" << std::endl;
            ++failures;
        }
    }

    // ---------- 测试 2: localBA ----------
    {
        // 3 个关键帧真值位姿 (T_wc): 沿 x 平移的双目小车
        std::vector<cv::Mat> T_wc_gt = {
            makeT(0, 0, 0, 0, 0, 0),
            makeT(0, 0.05, 0, 0.3, 0, 0.1),
            makeT(0, 0.10, 0, 0.6, 0, 0.2),
        };

        // 真值点 + 全部关键帧的无污染观测
        std::vector<cv::Point3f> pw_gt;
        for (int i = 0; i < 120; ++i)
            pw_gt.push_back({ static_cast<float>(ux(rng)),
                              static_cast<float>(uy(rng)),
                              static_cast<float>(uz(rng) + 2.0) });

        std::normal_distribution<double> pt_noise(0, 0.05);    // 点扰动 5cm
        std::normal_distribution<double> pose_noise(0, 0.03);  // 位姿扰动 3cm

        LocalMap map;
        std::vector<long> ids(pw_gt.size());
        std::vector<cv::Point3f> pw_noisy(pw_gt.size());
        for (size_t i = 0; i < pw_gt.size(); ++i) {
            ids[i] = static_cast<long>(i);
            pw_noisy[i] = { pw_gt[i].x + static_cast<float>(pt_noise(rng)),
                            pw_gt[i].y + static_cast<float>(pt_noise(rng)),
                            pw_gt[i].z + static_cast<float>(pt_noise(rng)) };
        }
        map.addPoints(ids, pw_noisy);

        for (int k = 0; k < 3; ++k) {
            Keyframe kf;
            kf.kf_id = k;
            // 第 0 帧位姿给真值 (它在 BA 里是固定的), 其余帧加扰动
            kf.T_wc = T_wc_gt[k].clone();
            if (k > 0) {
                kf.T_wc.at<double>(0, 3) += pose_noise(rng);
                kf.T_wc.at<double>(1, 3) += pose_noise(rng);
                kf.T_wc.at<double>(2, 3) += pose_noise(rng);
            }
            cv::Mat T_cw_gt = kf.T_wc.clone();   // 观测用真值位姿生成
            {
                cv::Mat R = T_wc_gt[k](cv::Rect(0, 0, 3, 3)).t();
                cv::Mat t = -R * T_wc_gt[k](cv::Rect(3, 0, 1, 3));
                T_cw_gt = cv::Mat::eye(4, 4, CV_64F);
                R.copyTo(T_cw_gt(cv::Rect(0, 0, 3, 3)));
                t.copyTo(T_cw_gt(cv::Rect(3, 0, 1, 3)));
            }
            for (size_t i = 0; i < pw_gt.size(); ++i) {
                cv::Point2f uv = project(T_cw_gt, pw_gt[i], fx, fy, cx, cy);
                kf.pt_ids.push_back(ids[i]);
                kf.obs.push_back({ uv.x + static_cast<float>(pix_noise(rng)),
                                   uv.y + static_cast<float>(pix_noise(rng)) });
                // 右目 u 观测: u_r = fx*(X_c - b)/Z + cx
                cv::Mat p = (cv::Mat_<double>(3, 1) << pw_gt[i].x, pw_gt[i].y, pw_gt[i].z);
                cv::Mat pc = T_cw_gt(cv::Rect(0, 0, 3, 3)) * p + T_cw_gt(cv::Rect(3, 0, 1, 3));
                double ur = fx * (pc.at<double>(0) - baseline) / pc.at<double>(2) + cx;
                kf.u_right.push_back(static_cast<float>(ur + pix_noise(rng)));
            }
            map.addKeyframe(std::move(kf));
        }

        // BA 前的误差
        double pose_err_before = 0;
        for (int k = 1; k < 3; ++k)
            pose_err_before += poseError(map.keyframes()[k].T_wc, T_wc_gt[k]);

        localBA(map, K, baseline);

        double pose_err_after = 0;
        for (int k = 1; k < 3; ++k)
            pose_err_after += poseError(map.keyframes()[k].T_wc, T_wc_gt[k]);

        double pt_err_before = 0, pt_err_after = 0;
        cv::Point3f p;
        for (size_t i = 0; i < pw_gt.size(); ++i) {
            pt_err_before += cv::norm(pw_noisy[i] - pw_gt[i]);
            map.getPoint(ids[i], p);
            pt_err_after += cv::norm(p - pw_gt[i]);
        }
        pt_err_before /= pw_gt.size();
        pt_err_after /= pw_gt.size();

        std::printf("localBA: 位姿误差 %.4f m -> %.4f m, 点平均误差 %.4f m -> %.4f m\n",
                    pose_err_before, pose_err_after, pt_err_before, pt_err_after);
        // 判据: 位姿误差显著下降; 点误差不劣化即可 ——
        // 0.2px 噪声下点的深度不确定度与 5cm 扰动同量级, 无法奢求大幅下降
        if (pose_err_after > 0.4 * pose_err_before || pt_err_after > pt_err_before) {
            std::cerr << "[FAIL] localBA 误差下降不足" << std::endl;
            ++failures;
        }
    }

    std::cout << (failures == 0 ? "[PASS] 优化器全部测试通过" : "[FAIL] 存在失败项")
              << std::endl;
    return failures == 0 ? 0 : 1;
}
