add_executable(negative-filter-usm negative-filter-usm.cpp)
target_link_libraries(negative-filter-usm ${OpenCV_LIBS} fmt::fmt)
add_sycl_to_target(TARGET negative-filter-usm SOURCES negative-filter-usm.cpp)

add_executable(negative-filter-buffer negative-filter-buffer.cpp)
target_link_libraries(negative-filter-buffer ${OpenCV_LIBS} fmt::fmt)
add_sycl_to_target(TARGET negative-filter-buffer SOURCES negative-filter-buffer.cpp)

add_executable(negative-filter-host negative-filter-host.cpp)
target_link_libraries(negative-filter-host ${OpenCV_LIBS} fmt::fmt)
add_sycl_to_target(TARGET negative-filter-host SOURCES negative-filter-host.cpp)

add_executable(negative-filter-image negative-filter-image.cpp)
target_link_libraries(negative-filter-image ${OpenCV_LIBS} fmt::fmt)
add_sycl_to_target(TARGET negative-filter-image SOURCES negative-filter-image.cpp)
