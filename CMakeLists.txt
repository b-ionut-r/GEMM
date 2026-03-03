cmake_minimum_required(VERSION 4.1)
project(GEMM CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(GEMM main.cu)

set_target_properties(GEMM PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
