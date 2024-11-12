cmake_minimum_required(VERSION 3.25)

project(learnsycl LANGUAGES CXX)

find_package(IntelSYCL CONFIG REQUIRED)

find_package(OpenCV CONFIG REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(fmt REQUIRED)

add_subdirectory(src)
