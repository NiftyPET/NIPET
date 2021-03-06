cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
if("${NIPET_BUILD_VERSION}" STREQUAL "")
set(NIPET_BUILD_VERSION 2 CACHE STRING "version" FORCE)
endif()
project(nipet LANGUAGES C CXX CUDA VERSION "${NIPET_BUILD_VERSION}")

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

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/NiftyPET${CMAKE_PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION niftypet/${CMAKE_PROJECT_NAME}/cmake)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/NiftyPET${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
  VERSION "${CMAKE_PROJECT_VERSION}" COMPATIBILITY AnyNewerVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/NiftyPET${CMAKE_PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/NiftyPET${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION niftypet/${CMAKE_PROJECT_NAME}/cmake)
