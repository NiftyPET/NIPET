cmake_minimum_required(VERSION 3.2)

project(nipet)

set(PREINSTALL_NAME "_preinstall")

# find cuda, maybe set a minimum version
find_package(CUDA REQUIRED)

find_package(PythonInterp)
find_package(PythonLibs)

# find_library(PYTHON_LIBRARY
#               NAMES python27 libpython2.7.so
#               PATHS ${PYTHON_PREFIX_PATH}/libs/
#               ${PYTHON_PREFIX_PATH}/lib/
#             )
# set(PYTHON_LIBRARIES)
# set(PYTHON_INCLUDE_DIRS)
set(PYTHON_PREFIX_PATH)

add_subdirectory(${CMAKE_PROJECT_NAME})
add_subdirectory(${CMAKE_PROJECT_NAME}/dinf)
add_subdirectory(${CMAKE_PROJECT_NAME}/lm)
# add_subdirectory(${CMAKE_PROJECT_NAME}/lms)
add_subdirectory(${CMAKE_PROJECT_NAME}/prj)
add_subdirectory(${CMAKE_PROJECT_NAME}/sct)

set(CMAKE_BUILD_TYPE Release)


execute_process ( COMMAND python -c "import sys; print sys.prefix" OUTPUT_VARIABLE PYTHON_ENV_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX
    ${PYTHON_ENV_ROOT} CACHE PATH "Set the path of the anaconda environment to which you wish to install this package to" FORCE
    )
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
