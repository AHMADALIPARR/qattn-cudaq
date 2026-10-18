cmake_minimum_required(VERSION 3.18)
project(qattn_cuda_arrayfire LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(ArrayFire REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(qattn_engine
    qattn_engine.cpp
    qattn_kernels.cu
)

target_link_libraries(qattn_engine
    ArrayFire::afcuda
    CUDA::cudart
)

target_compile_options(qattn_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>
)
