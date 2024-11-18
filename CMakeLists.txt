cmake_minimum_required(VERSION 3.10)
project(folddotc)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -funroll-loops")

set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/csrc)

find_package(OpenMP)

add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(main OpenMP::OpenMP_CXX)
