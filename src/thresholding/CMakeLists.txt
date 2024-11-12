add_executable(thresholding-host thresholding-host.cpp)
target_link_libraries(thresholding-host PRIVATE ${OpenCV_LIBS} fmt::fmt)

add_executable(thresholding-usm thresholding-usm.cpp)
target_link_libraries(thresholding-usm PRIVATE ${OpenCV_LIBS} fmt::fmt)
add_sycl_to_target(TARGET thresholding-usm SOURCES thresholding-usm.cpp)

add_executable(thresholding-buffer thresholding-buffer.cpp)
target_link_libraries(thresholding-buffer PRIVATE ${OpenCV_LIBS} fmt::fmt)
add_sycl_to_target(TARGET thresholding-buffer SOURCES thresholding-buffer.cpp)
