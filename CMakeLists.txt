cmake_minimum_required(VERSION 3.18)
project(spmm_vcsr LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 80 = A100, 90 = H100
if(NOT DEFINED SM_ARCH)
  set(SM_ARCH 90)
endif()
set(CMAKE_CUDA_ARCHITECTURES ${SM_ARCH})

add_executable(spmm_vcsr
  src/main.cpp
  src/kernels.cu
)
target_include_directories(spmm_vcsr PRIVATE include)

# ---- Keep flags minimal & portable ----
target_compile_options(spmm_vcsr PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xptxas=-v;--use_fast_math>
)

# ---- CUDA & cuSPARSE linking ----
option(USE_CUSPARSE "Build cuSPARSE baseline" ON)
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
  target_link_libraries(spmm_vcsr PRIVATE CUDA::cudart)
  if (USE_CUSPARSE)
    target_compile_definitions(spmm_vcsr PRIVATE HAVE_CUSPARSE=1)
    target_link_libraries(spmm_vcsr PRIVATE CUDA::cusparse)
  endif()
else()
  target_link_libraries(spmm_vcsr PRIVATE cudart)
  if (USE_CUSPARSE)
    target_compile_definitions(spmm_vcsr PRIVATE HAVE_CUSPARSE=1)
    target_link_libraries(spmm_vcsr PRIVATE cusparse)
  endif()
endif()
