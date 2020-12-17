cmake_minimum_required(VERSION 3.2)

project(nipet)

set(PREINSTALL_NAME "_preinstall")

cmake_policy(PUSH)
find_package(CUDA REQUIRED)
if(CMAKE_VERSION VERSION_EQUAL 3.12 OR CMAKE_VERSION VERSION_GREATER 3.12)
  cmake_policy(SET CMP0074 NEW)
  find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
else()
  find_package(FindPythonInterp 3 REQUIRED)
  set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}")
  find_package(FindPythonLibs 3 REQUIRED)
  set(Python3_INCLUDE_DIRS "${PYTHON_INCLUDE_DIRS}")
  set(Python3_LIBRARIES "${PYTHON_LIBRARIES}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE Python3_NumPy_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
cmake_policy(POP)

if(CMAKE_VERSION VERSION_EQUAL 3.7 OR CMAKE_VERSION VERSION_GREATER 3.7)
  cuda_select_nvcc_arch_flags(NVCC_ARCH_FLAGS)
else()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m miutil.cuinfo -f
    OUTPUT_VARIABLE NVCC_ARCH_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
list(APPEND CUDA_NVCC_FLAGS ${NVCC_ARCH_FLAGS})
message(STATUS "CUDA nvcc flags: ${CUDA_NVCC_FLAGS}")

add_subdirectory(${CMAKE_PROJECT_NAME})
add_subdirectory(${CMAKE_PROJECT_NAME}/lm)
add_subdirectory(${CMAKE_PROJECT_NAME}/prj)
add_subdirectory(${CMAKE_PROJECT_NAME}/sct)

set(CMAKE_BUILD_TYPE Release)
