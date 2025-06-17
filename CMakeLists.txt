cmake_minimum_required(VERSION 3.10)
project(JoMatch LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)
set(CMAKE_CUDA_LIBRARY_ARCHITECTURES 86) # A6000
# set(CMAKE_CUDA_LIBRARY_ARCHITECTURES 75) # 2080Ti
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --maxrregcount=32")

include_directories(/usr/local/cuda-12.1/include) # active at school
link_directories(/usr/local/cuda-12.1/lib64) # activate at school

# library
add_library(glob STATIC
    globals.cuh
    globals.cu
    CLI11.hpp
    cuda_helpers.h
    cuda_helpers.cu
    memManag.cuh
    memManag.cu

    structure.cuh
    structure.cu

    res_table.hpp

    defs.h
)

add_library(graph STATIC

    cpuGraph.h
    cpuGraph.cpp
    gpuGraph.h
    gpuGraph.cpp
)

add_library(preprocess STATIC
    decycle.h
    decycle.cpp

    io.cu
    io.cuh
)

add_library(tdfs STATIC
    gpu_match.cuh
    gpu_match.cu
    job_queue.cuh
    callstack.cuh
)

add_library(filter STATIC
    filter.cuh
    filter.cu
)

add_library(order STATIC
    order.h
    order.cpp
)

add_library(join STATIC
    join.cuh
    join.cu
)

set_target_properties(glob PROPERTIES
    CUDA_ARCHITECTURES "86" 
    # CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(glob PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

set_target_properties(graph PROPERTIES
    CUDA_ARCHITECTURES "86" 
    # CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(graph PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

set_target_properties(preprocess PROPERTIES
    CUDA_ARCHITECTURES "86" 
    # CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(preprocess PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

set_target_properties(tdfs PROPERTIES
    CUDA_ARCHITECTURES "86" 
    # CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(tdfs PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

set_target_properties(filter PROPERTIES
    CUDA_ARCHITECTURES "86" 
    # CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(filter PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

set_target_properties(order PROPERTIES
    CUDA_ARCHITECTURES "86" 
    # CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(order PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

set_target_properties(join PROPERTIES
    CUDA_ARCHITECTURES "86" 
    # CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(join PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

# executable
add_executable(JoMatch main.cu)

set_target_properties(JoMatch PROPERTIES
    CUDA_ARCHITECTURES "86"
    # CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# link
target_link_libraries(JoMatch PUBLIC glob graph preprocess tdfs filter order join cudart cuda)

target_compile_options(JoMatch PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)