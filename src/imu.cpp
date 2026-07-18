#include "vo/imu.h"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>

namespace {
constexpr double kG = 9.80665;                  // 标准重力, m/s^2
constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;

// data[off], data[off+1] 组成的有符号 16 位小端整数
int16_t toShort(const uint8_t* d, int off) {
    return static_cast<int16_t>((d[off + 1] << 8) | d[off]);
}

// 波特率整数 -> termios 常量
speed_t toSpeed(int baud) {
    switch (baud) {
        case 4800:   return B4800;
        case 9600:   return B9600;
        case 19200:  return B19200;
        case 38400:  return B38400;
        case 57600:  return B57600;
        case 115200: return B115200;
        default:     return B9600;
    }
}

double nowSeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace

HiwonderImu::HiwonderImu(const std::string& dev, int baud) {
    fd_ = ::open(dev.c_str(), O_RDONLY | O_NOCTTY);
    if (fd_ < 0) {
        std::fprintf(stderr, "[HiwonderImu] 无法打开 %s (%s)\n",
                     dev.c_str(), std::strerror(errno));
        return;
    }

    termios tio{};
    if (tcgetattr(fd_, &tio) != 0) {
        std::fprintf(stderr, "[HiwonderImu] tcgetattr 失败\n");
        ::close(fd_);
        fd_ = -1;
        return;
    }
    speed_t sp = toSpeed(baud);
    cfsetispeed(&tio, sp);
    cfsetospeed(&tio, sp);
    tio.c_cflag |= (CLOCAL | CREAD);
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |= CS8;       // 8 位
    tio.c_cflag &= ~PARENB;   // 无校验
    tio.c_cflag &= ~CSTOPB;   // 1 停止位
    tio.c_cflag &= ~CRTSCTS;  // 无硬件流控
    cfmakeraw(&tio);
    tio.c_cc[VMIN] = 1;
    tio.c_cc[VTIME] = 0;
    tcflush(fd_, TCIFLUSH);
    if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
        std::fprintf(stderr, "[HiwonderImu] tcsetattr 失败\n");
        ::close(fd_);
        fd_ = -1;
        return;
    }

    std::printf("[HiwonderImu] 已打开 %s @ %d\n", dev.c_str(), baud);
}

HiwonderImu::~HiwonderImu() {
    if (fd_ >= 0) ::close(fd_);
}

bool HiwonderImu::readPacket(uint8_t* pkt) {
    int idx = 0;
    uint8_t b = 0;
    while (true) {
        ssize_t n = ::read(fd_, &b, 1);
        if (n <= 0) return false;

        if (idx == 0) {
            if (b != 0x55) continue;  // 寻找帧头
            pkt[idx++] = b;
            continue;
        }
        pkt[idx++] = b;
        if (idx < 11) continue;

        // 校验和: 前 10 字节求和低 8 位
        uint8_t sum = 0;
        for (int i = 0; i < 10; ++i) sum += pkt[i];
        if (sum == pkt[10]) return true;

        idx = 0;  // 校验失败, 重新对齐
    }
}

bool HiwonderImu::getImuData(ImuData& data) {
    if (fd_ < 0) return false;

    uint8_t pkt[11];
    while (readPacket(pkt)) {
        const uint8_t* d = &pkt[2];
        switch (pkt[1]) {
            case 0x51:  // 加速度 ±16g -> m/s^2
                acc_.x() = toShort(d, 0) / 32768.0 * 16.0 * kG;
                acc_.y() = toShort(d, 2) / 32768.0 * 16.0 * kG;
                acc_.z() = toShort(d, 4) / 32768.0 * 16.0 * kG;
                have_acc_ = true;
                break;
            case 0x52:  // 角速度 ±2000 deg/s -> rad/s
                gyro_.x() = toShort(d, 0) / 32768.0 * 2000.0 * kDeg2Rad;
                gyro_.y() = toShort(d, 2) / 32768.0 * 2000.0 * kDeg2Rad;
                gyro_.z() = toShort(d, 4) / 32768.0 * 2000.0 * kDeg2Rad;
                have_gyro_ = true;
                break;
            default:
                break;  // 角度/四元数等忽略
        }

        // 凑齐一组(以角速度包为一帧结束)即返回
        if (have_acc_ && have_gyro_ && pkt[1] == 0x52) {
            data.acc = acc_;
            data.gyro = gyro_;
            data.timestamp = nowSeconds();
            return true;
        }
    }
    return false;
}
