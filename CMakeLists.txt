cmake_minimum_required(VERSION 3.25)

project(visionsycl LANGUAGES CXX)

find_package(IntelSYCL CONFIG REQUIRED)
include_directories(${ONEAPI_ROOT}/dev-utilities/latest/include)
include_directories(include/)

file(GLOB SOURCES include/**/*.hpp src/**/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})
add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${SOURCES})