cmake_minimum_required(VERSION 3.10)
project(vio_d435)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# librealsense2 随 ROS Humble 安装
list(APPEND CMAKE_PREFIX_PATH /opt/ros/humble)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# VO 库: include/vo 下的头文件 + src 下的实现
add_library(vo STATIC
    src/camera.cpp
    src/feature.cpp
    src/frame.cpp
    src/imu.cpp
    src/map.cpp
    src/odometry.cpp
    src/optimize.cpp
)
target_include_directories(vo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${realsense2_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(vo PUBLIC ${realsense2_LIBRARY} ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(capture_d435 src/capture_d435.cpp)
target_link_libraries(capture_d435 vo)

add_subdirectory(test)
