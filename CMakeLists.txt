cmake_minimum_required(VERSION 4.1)
project(GEMM CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(GEMM main.cu step_1.cu step_2.cu step_3.cu step_4.cu
        step_5.cu
        step_6.cu)

set_target_properties(GEMM PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
