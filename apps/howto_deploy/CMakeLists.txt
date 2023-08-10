cmake_minimum_required(VERSION 3.18)

#set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_ARCHITECTURES 61) #对应 Pascal 架构的 P40 GPU
project(GEMM LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -fPIC")


set(TVM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../")
set(DMLC_CORE "${TVM_ROOT}/3rdparty/dmlc-core")

add_library(tvm_rt_pk_cuda STATIC tvm_runtime_pack.cc)

target_include_directories(tvm_rt_pk_cuda PRIVATE 
    ${TVM_ROOT}/include
    ${DMLC_CORE}/include
    ${TVM_ROOT}/3rdparty/dlpack/include
    ${TVM_ROOT}/3rdparty/MPMCQueue/include)

target_link_libraries(tvm_rt_pk_cuda 
    CUDA::cudart
    #CUDA::cudart_static
    -ldl -pthread)
# target_link_libraries(tvm_rt_pk_cuda 
#     CUDA::cudart -ldl -pthread)

target_compile_definitions(tvm_rt_pk_cuda PRIVATE DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)



set_target_properties(tvm_rt_pk_cuda PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})