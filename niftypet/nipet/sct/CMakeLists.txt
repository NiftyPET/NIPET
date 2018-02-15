cmake_minimum_required(VERSION 3.2)

# set project name as the module name of the .dll/.pyd 
project(petsct) 
# set name of folder, because it isnt set to the project name :(
set(CURR_FOLDER_NAME "sct")

# get the site packages location, then strip the trailing whitespace
execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

#get source files
file(GLOB_RECURSE SRC "src/*.cu")
file(GLOB_RECURSE HDR "src/*.h")

list(APPEND SRC ${HDR})

include_directories(include ${CMAKE_BINARY_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/${CMAKE_PROJECT_NAME})
include_directories(${CMAKE_SOURCE_DIR}/${CMAKE_PROJECT_NAME}/src)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_SITE_PACKAGES}/numpy/core/include/)

cuda_add_library(${PROJECT_NAME} SHARED ${SRC}) 

target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".pyd")
    ADD_CUSTOM_COMMAND(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_BINARY_DIR}/Release/${PROJECT_NAME}.pyd
        ${CMAKE_SOURCE_DIR}/${CMAKE_PROJECT_NAME}/${CURR_FOLDER_NAME}/${PROJECT_NAME}.pyd
    )
endif()

if(UNIX)
    set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
    ADD_CUSTOM_COMMAND(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.so
        ${CMAKE_SOURCE_DIR}/${CMAKE_PROJECT_NAME}/${CURR_FOLDER_NAME}/${PROJECT_NAME}.so
    )
endif()