cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(nipet LANGUAGES C CXX CUDA VERSION 1.0)

cmake_policy(PUSH)
cmake_policy(SET CMP0074 NEW)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(CUDA REQUIRED)
if(SKBUILD)
find_package(PythonExtensions REQUIRED)
endif()
cmake_policy(POP)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

cuda_select_nvcc_arch_flags(NVCC_ARCH_FLAGS)
list(APPEND CUDA_NVCC_FLAGS ${NVCC_ARCH_FLAGS})
message(STATUS "CUDA nvcc flags: ${CUDA_NVCC_FLAGS}")

add_subdirectory(niftypet/nipet)
