#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <string>

// 一次 IMU 读数: 加速度(m/s^2) + 角速度(rad/s) + 时间戳(s)
struct ImuData {
    double timestamp = 0.0;             // 主机接收时刻, 单位秒
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();   // 加速度, m/s^2
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();  // 角速度, rad/s
};

// Hiwonder USB IMU (CH340 转串口, WitMotion 协议) 读取类。
//
// 数据包 11 字节: 0x55 帧头 + 类型 + 8 字节数据 + 校验和。
// 类型 0x51=加速度(±16g), 0x52=角速度(±2000 deg/s), 其它(角度/四元数)忽略。
// 默认串口 /dev/ttyUSB0, 波特率 9600。
class HiwonderImu {
public:
    explicit HiwonderImu(const std::string& dev = "/dev/ttyUSB0", int baud = 9600);
    ~HiwonderImu();

    HiwonderImu(const HiwonderImu&) = delete;
    HiwonderImu& operator=(const HiwonderImu&) = delete;

    // 是否成功打开串口
    bool isOpen() const { return fd_ >= 0; }

    // 阻塞读取, 直到凑齐一组新的 加速度+角速度, 写入 data。
    // 成功返回 true; 串口未打开或读失败返回 false。
    bool getImuData(ImuData& data);

private:
    // 从串口读满一个 11 字节数据包(带帧头对齐与校验), 成功返回 true
    bool readPacket(uint8_t* pkt);

    int fd_ = -1;

    // 缓存最近一次加速度/角速度, 便于凑成一组返回
    Eigen::Vector3d acc_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_ = Eigen::Vector3d::Zero();
    bool have_acc_ = false;
    bool have_gyro_ = false;
};
