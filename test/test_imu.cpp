// HiwonderImu 类的功能测试: 实例化后直接取出加速度和角速度并打印。
//
// 用法: ./test_imu [串口设备]   默认 /dev/ttyUSB0

#include "vo/imu.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

// 6 维四元数姿态 EKF
//   状态  x_ = [q0, q1, q2, q3, bx, by]
//         q  : 机体系->世界系姿态四元数
//         bx, by : 陀螺 x/y 轴零偏 (z 轴指天, 零偏不可观, 不估计)
//   预测  用扣除零偏后的陀螺角速度积分四元数, 零偏按随机游走(常值模型)
//   量测  用加速度计的重力方向(归一化后)校正; 加速度不直接依赖零偏,
//         故量测雅可比 H 的零偏两列为 0
class EKF{
public:
    explicit EKF(double dt = 0.01) {
        dt_ = dt;

        // 状态: [q0..q3, bx, by], 初始为无旋转 + 零偏为 0
        x_ = Eigen::VectorXd::Zero(6);
        x_(0) = 1.0;

        // 状态协方差: 姿态与零偏分别给初值
        P_ = Eigen::MatrixXd::Identity(6, 6);
        P_.block<4,4>(0,0) *= 1e-2;   // 姿态不确定性
        P_.block<2,2>(4,4) *= 1e-4;   // 零偏不确定性

        // 过程噪声: 姿态积分噪声 + 零偏随机游走噪声
        Q_ = Eigen::MatrixXd::Zero(6, 6);
        Q_.block<4,4>(0,0) = Eigen::Matrix4d::Identity() * 1e-5;
        Q_.block<2,2>(4,4) = Eigen::Matrix2d::Identity() * 1e-8;

        // 量测噪声(加速度计): 调大 => 更信任陀螺积分
        R_ = Eigen::Matrix3d::Identity() * 1e-1;

        // 状态转移矩阵 / 卡尔曼增益, predict/update 里会重新填
        F_ = Eigen::MatrixXd::Identity(6, 6);
        K_ = Eigen::MatrixXd::Zero(6, 3);
    }
    ~EKF() {}

    void setDt(double dt) { dt_ = dt; }
    Eigen::Vector4d quaternion() const { return x_.head<4>(); }
    Eigen::Vector2d bias() const { return x_.segment<2>(4); }

    void predict(double wx, double wy, double wz)
    {
        // 扣除零偏 (z 轴无零偏估计)
        double cx = wx - x_(4);
        double cy = wy - x_(5);
        double cz = wz;

        // 四元数运动学: q_dot = 0.5 * Omega(w) * q
        Eigen::Matrix4d omega_q;
        omega_q << 0,  -cx, -cy, -cz,
                   cx,   0,  cz, -cy,
                   cy, -cz,   0,  cx,
                   cz,  cy, -cx,   0;

        Eigen::Vector4d q = x_.head<4>();          // 线性化点(积分前的 q)
        Eigen::Vector4d q_new = q + 0.5 * omega_q * q * dt_;
        q_new.normalize();                          // 保持单位四元数
        x_.head<4>() = q_new;
        // 零偏按常值模型, 不变

        // 状态转移矩阵 F (6x6):
        //   [ dq/dq (4x4)   dq/db (4x2) ]
        //   [   0 (2x4)      I2 (2x2)   ]
        F_ = Eigen::MatrixXd::Identity(6, 6);
        F_.block<4,4>(0,0) = Eigen::Matrix4d::Identity() + 0.5 * omega_q * dt_;

        // dq/db: 角速度里减了零偏, 对 bx,by 求导 (= -0.5*dt*d(Omega*q)/dw 的前两列)
        Eigen::MatrixXd G(4, 2);
        G <<  0.5 * dt_ * q[1],  0.5 * dt_ * q[2],
             -0.5 * dt_ * q[0],  0.5 * dt_ * q[3],
             -0.5 * dt_ * q[3], -0.5 * dt_ * q[0],
              0.5 * dt_ * q[2], -0.5 * dt_ * q[1];
        F_.block<4,2>(0,4) = G;

        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    void update(double ax, double ay, double az)
    {
        // 只用重力方向, 因此把加速度归一化
        Eigen::Vector3d z(ax, ay, az);
        if (z.norm() < 1e-9) return;
        z.normalize();
        
        double a_norm = z.norm();
        if (std::abs(a_norm - 9.80665) > 3.0) return;   // 幅值偏离重力过多, 判为动态/撞击
        
        Eigen::Vector4d q = x_.head<4>();

        // 量测预测: 世界系"上"方向 [0,0,1] 旋到机体系 = R(q)^T * [0,0,1]
        Eigen::Vector3d x_h;
        x_h << 2 * (q[1] * q[3] - q[0] * q[2]),
               2 * (q[2] * q[3] + q[0] * q[1]),
               1 - 2 * (q[1] * q[1] + q[2] * q[2]);
        Eigen::Vector3d y = z - x_h;

        // 量测雅可比 H (3x6): 前 4 列 = d(x_h)/dq, 后 2 列(零偏)= 0
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 6);
        H.block<3,4>(0,0) <<
            -2 * q[2],  2 * q[3], -2 * q[0],  2 * q[1],
             2 * q[1],  2 * q[0],  2 * q[3],  2 * q[2],
                    0, -4 * q[1], -4 * q[2],         0;

        Eigen::Matrix3d S = H * P_ * H.transpose() + R_;

        // 卡方检验(新息门限): 撞击/跳跃会让加速度瞬间偏离重力方向,
        // 此时新息 y 异常大。用马氏距离 d^2 = y^T S^-1 y 判定,
        // 自由度=3(量测维度), 超过阈值则认为是异常量测, 跳过本次 update。
        //   卡方 0.95 分位: 7.815   0.99 分位: 11.345
        constexpr double kChi2Thresh = 7.815;
        double d2 = y.transpose() * S.inverse() * y;
        if (d2 > kChi2Thresh) return;

        K_ = P_ * H.transpose() * S.inverse();   // 6x3
        x_ = x_ + K_ * y;

        // 更新后重新归一化四元数
        Eigen::Vector4d qn = x_.head<4>();
        qn.normalize();
        x_.head<4>() = qn;

        P_ = (Eigen::MatrixXd::Identity(6, 6) - K_ * H) * P_;
    }

private:
    Eigen::VectorXd x_; // 状态向量 [q0,q1,q2,q3, bx,by]
    Eigen::MatrixXd P_; // 状态协方差矩阵 6x6
    Eigen::MatrixXd Q_; // 过程噪声协方差矩阵 6x6
    Eigen::MatrixXd R_; // 量测噪声协方差矩阵 3x3
    Eigen::MatrixXd K_; // 卡尔曼增益 6x3
    Eigen::MatrixXd F_; // 状态转移矩阵 6x6
    double dt_;         // 时间间隔
};

// ---------------- 简易 3D 可视化 (类 RViz) ----------------
// 世界系: Z 轴朝上, 用一个固定视角把 3D 点正交投影到 2D 画布。
// 立方体代表 IMU 板, 随 EKF 估计的姿态旋转; 附带世界坐标系网格 + 三色轴。
class Viewer3D {
public:
    Viewer3D(int w = 720, int h = 720, double az_deg = 45, double el_deg = 22)
        : W_(w), H_(h) {
        double az = az_deg * M_PI / 180.0, el = el_deg * M_PI / 180.0;
        // 相机在世界系中的方向(从原点指向相机)
        Eigen::Vector3d camz(std::cos(el) * std::cos(az),
                             std::cos(el) * std::sin(az),
                             std::sin(el));
        Eigen::Vector3d world_up(0, 0, 1);
        right_ = world_up.cross(camz).normalized();
        up_ = camz.cross(right_).normalized();
    }

    // 世界坐标 -> 画布像素 (正交投影)
    cv::Point project(const Eigen::Vector3d& p) const {
        double x = right_.dot(p);
        double y = up_.dot(p);
        return cv::Point(static_cast<int>(W_ / 2.0 + scale_ * x),
                         static_cast<int>(H_ / 2.0 - scale_ * y));
    }

    void drawLine(cv::Mat& img, const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                  const cv::Scalar& c, int th = 1) const {
        cv::line(img, project(a), project(b), c, th, cv::LINE_AA);
    }

    // 世界地面网格 + 世界坐标轴
    void drawWorld(cv::Mat& img) const {
        const double s = 1.0, step = 0.2;
        cv::Scalar grid(60, 60, 60);
        for (double x = -s; x <= s + 1e-9; x += step) {
            drawLine(img, {x, -s, 0}, {x, s, 0}, grid);
            drawLine(img, {-s, x, 0}, {s, x, 0}, grid);
        }
        // 世界轴 (较暗), X 红 Y 绿 Z 蓝
        drawLine(img, {0, 0, 0}, {0.4, 0, 0}, cv::Scalar(80, 80, 160), 1);
        drawLine(img, {0, 0, 0}, {0, 0.4, 0}, cv::Scalar(80, 160, 80), 1);
        drawLine(img, {0, 0, 0}, {0, 0, 0.4}, cv::Scalar(160, 80, 80), 1);
    }

    // 用旋转矩阵 R(机体->世界) 画 IMU 立方体 + 机体三色轴
    void drawImu(cv::Mat& img, const Eigen::Matrix3d& R) const {
        const double hx = 0.30, hy = 0.20, hz = 0.05;  // 板半尺寸
        Eigen::Vector3d v[8] = {
            {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
            {-hx, -hy, hz},  {hx, -hy, hz},  {hx, hy, hz},  {-hx, hy, hz}};
        for (auto& p : v) p = R * p;

        int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                            {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        // 顶面(+Z)填充, 一眼看出朝向
        cv::Point top[4] = {project(v[4]), project(v[5]), project(v[6]), project(v[7])};
        cv::fillConvexPoly(img, top, 4, cv::Scalar(90, 70, 40), cv::LINE_AA);
        for (auto& e : edges)
            drawLine(img, v[e[0]], v[e[1]], cv::Scalar(230, 200, 150), 2);

        // 机体三色轴 X 红 Y 绿 Z 蓝
        Eigen::Vector3d o(0, 0, 0);
        drawLine(img, o, R * Eigen::Vector3d(0.5, 0, 0), cv::Scalar(60, 60, 255), 3);
        drawLine(img, o, R * Eigen::Vector3d(0, 0.5, 0), cv::Scalar(60, 255, 60), 3);
        drawLine(img, o, R * Eigen::Vector3d(0, 0, 0.5), cv::Scalar(255, 120, 60), 3);
    }

private:
    int W_, H_;
    double scale_ = 260.0;
    Eigen::Vector3d right_, up_;
};

int main(int argc, char** argv) {
    std::string dev = (argc > 1) ? argv[1] : "/dev/ttyUSB0";

    HiwonderImu imu(dev, 9600);
    if (!imu.isOpen()) {
        std::fprintf(stderr, "[FAIL] IMU 打开失败\n");
        return 1;
    }

    std::printf("开始读取 IMU (窗口内按 q 或 ESC 退出)\n");

    Viewer3D viewer;
    const std::string win = "IMU 姿态 (EKF)";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

    EKF ekf;
    ImuData data;
    bool first = true;
    double prev_ts = 0.0;

    while (imu.getImuData(data)) {
        // 用相邻两帧的主机时间戳算 dt
        double dt = 0.01;
        if (!first) dt = data.timestamp - prev_ts;
        prev_ts = data.timestamp;
        first = false;
        if (dt <= 0.0 || dt > 0.5) dt = 0.01;  // 异常 dt 兜底
        ekf.setDt(dt);

        ekf.predict(data.gyro.x(), data.gyro.y(), data.gyro.z());
        ekf.update(data.acc.x(), data.acc.y(), data.acc.z());

        // 取估计姿态, 手动转成 roll/pitch/yaw (deg) 便于观察
        Eigen::Vector4d qv = ekf.quaternion();
        double qw = qv(0), qx = qv(1), qy = qv(2), qz = qv(3);
        double roll = std::atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
        double pitch = std::asin(std::max(-1.0, std::min(1.0, 2 * (qw * qy - qz * qx))));
        double yaw = std::atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
        Eigen::Vector3d rpy(roll, pitch, yaw);
        rpy *= 180.0 / M_PI;

        Eigen::Vector2d b = ekf.bias();
        std::printf("\rEKF rpy[deg]: % 7.2f % 7.2f % 7.2f | bias[rad/s]: % 6.4f % 6.4f   ",
                    rpy(0), rpy(1), rpy(2), b(0), b(1));
        std::fflush(stdout);

        // 3D 可视化: 用估计姿态旋转立方体
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Matrix3d R = q.toRotationMatrix();

        cv::Mat canvas(720, 720, CV_8UC3, cv::Scalar(25, 25, 25));
        viewer.drawWorld(canvas);
        viewer.drawImu(canvas, R);
        cv::putText(canvas,
                    cv::format("roll %.1f  pitch %.1f  yaw %.1f (deg)", rpy(0), rpy(1), rpy(2)),
                    cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
        cv::putText(canvas, "X:red  Y:green  Z:blue", cv::Point(15, 55),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
        cv::imshow(win, canvas);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    cv::destroyAllWindows();
    std::printf("\n退出\n");
    return 0;
}
