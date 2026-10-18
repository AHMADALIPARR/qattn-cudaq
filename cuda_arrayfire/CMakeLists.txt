cmake_minimum_required(VERSION 3.18)
project(qattn_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit QUIET)
find_package(ArrayFire QUIET)
find_package(CURL QUIET)

option(QATTN_ENABLE_CUDA "Build CUDA kernels" ON)
option(QATTN_ENABLE_ARRAYFIRE "Build ArrayFire backend" ON)
option(QATTN_ENABLE_CUDAQ "Build CUDA-Q module" ON)

set(QATTN_HAS_CUDA FALSE)
set(QATTN_HAS_ARRAYFIRE FALSE)
set(QATTN_HAS_CUDAQ FALSE)
set(QATTN_HAS_LIBCURL FALSE)

if(QATTN_ENABLE_CUDA AND CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(QATTN_HAS_CUDA TRUE)
endif()

if(QATTN_ENABLE_ARRAYFIRE AND ArrayFire_FOUND AND QATTN_HAS_CUDA)
    set(QATTN_HAS_ARRAYFIRE TRUE)
endif()

if(CURL_FOUND)
    set(QATTN_HAS_LIBCURL TRUE)
endif()

find_path(CUDAQ_INCLUDE cudaq.h)
if(QATTN_ENABLE_CUDAQ AND CUDAQ_INCLUDE)
    set(QATTN_HAS_CUDAQ TRUE)
endif()

add_library(qattn_core
    src/statevector.cpp
    src/attention.cpp
    src/entropy.cpp
)
target_include_directories(qattn_core PUBLIC include)
if(QATTN_HAS_LIBCURL)
    target_compile_definitions(qattn_core PRIVATE QATTN_HAS_LIBCURL)
    target_link_libraries(qattn_core PUBLIC CURL::libcurl)
endif()

add_executable(qattn_runtime src/main.cpp)
target_link_libraries(qattn_runtime PRIVATE qattn_core)

if(QATTN_HAS_CUDA)
    add_library(qattn_cuda cuda/kernels.cu)
    target_link_libraries(qattn_cuda PUBLIC CUDA::cudart)
    set_target_properties(qattn_cuda PROPERTIES CUDA_ARCHITECTURES "86")
endif()

if(QATTN_HAS_ARRAYFIRE)
    add_library(qattn_af arrayfire/af_backend.cpp)
    target_link_libraries(qattn_af PUBLIC ArrayFire::afcuda qattn_cuda)
endif()

if(QATTN_HAS_CUDAQ)
    add_library(qattn_cudaq cudaq/feature_circuit.cpp)
    target_include_directories(qattn_cudaq PUBLIC ${CUDAQ_INCLUDE})
    target_link_libraries(qattn_cudaq PUBLIC cudaq)
endif()

enable_testing()
add_executable(test_cpu_reference tests/test_cpu_reference.cpp)
target_link_libraries(test_cpu_reference PRIVATE qattn_core)
add_test(NAME cpu_reference COMMAND test_cpu_reference)

message(STATUS "QATTN CUDA:       ${QATTN_HAS_CUDA}")
message(STATUS "QATTN ArrayFire:  ${QATTN_HAS_ARRAYFIRE}")
message(STATUS "QATTN CUDA-Q:     ${QATTN_HAS_CUDAQ}")
message(STATUS "QATTN libcurl:    ${QATTN_HAS_LIBCURL}")
