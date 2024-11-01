cmake_minimum_required(VERSION 3.10)
project(folddotc)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/csrc)

add_executable(folddotc ${PROJECT_SOURCE_DIR}/main.cpp)
