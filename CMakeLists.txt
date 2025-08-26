cmake_minimum_required(VERSION 3.30)
project(heat_equation)

set(CMAKE_CXX_STANDARD 20)

include_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
link_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

add_executable(heat_equation heat2D.cpp)

target_link_libraries(heat_equation msmpi)
