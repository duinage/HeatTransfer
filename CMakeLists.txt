cmake_minimum_required(VERSION 3.30)
project(heat_equation)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

find_package(OpenMP REQUIRED)

add_executable(heat_equation heat2D.cpp)

target_link_libraries(heat_equation PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

target_compile_options(heat_equation PRIVATE /O2 /arch:AVX2)
