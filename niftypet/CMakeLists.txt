cmake_minimum_required(VERSION 3.2)

project(nipet)

set(PREINSTALL_NAME "_preinstall")

cmake_policy(PUSH)
cmake_policy(SET CMP0074 NEW)
find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
cmake_policy(POP)

add_subdirectory(${CMAKE_PROJECT_NAME})
add_subdirectory(${CMAKE_PROJECT_NAME}/lm)
add_subdirectory(${CMAKE_PROJECT_NAME}/prj)
add_subdirectory(${CMAKE_PROJECT_NAME}/sct)

set(CMAKE_BUILD_TYPE Release)
