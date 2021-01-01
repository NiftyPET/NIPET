cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(nipet LANGUAGES C CXX CUDA VERSION 2.0)

cmake_policy(PUSH)
cmake_policy(SET CMP0074 NEW)  # <PackageName>_ROOT hints for find_package
cmake_policy(SET CMP0104 NEW)  # CMAKE_CUDA_ARCHITECTURES
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(SKBUILD)
find_package(PythonExtensions REQUIRED)
endif()
cmake_policy(POP)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_subdirectory(nipet)
