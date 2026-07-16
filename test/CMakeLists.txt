add_executable(test_camera test_camera.cpp)
target_link_libraries(test_camera vo)

add_executable(test_feature_tracker test_feature_tracker.cpp)
target_link_libraries(test_feature_tracker vo)

add_executable(test_frame_match test_frame_match.cpp)
target_link_libraries(test_frame_match vo)

add_executable(test_vo test_vo.cpp)
target_link_libraries(test_vo vo)

add_executable(test_optimize test_optimize.cpp)
target_link_libraries(test_optimize vo)
