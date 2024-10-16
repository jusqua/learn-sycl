add_executable(hello-world main.cpp)
add_sycl_to_target(TARGET hello-world SOURCES main.cpp)
