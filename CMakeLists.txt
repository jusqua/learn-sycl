if (CMAKE_HOST_WIN32)
    cmake_minimum_required(VERSION 3.25)
else()
    cmake_minimum_required(VERSION 3.23.5)
endif()

project(learnsycl LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra)
find_package(IntelSYCL CONFIG REQUIRED)

add_subdirectory(src)
