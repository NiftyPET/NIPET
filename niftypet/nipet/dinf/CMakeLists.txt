cmake_minimum_required(VERSION 3.2)
 
project(dinf)

find_package(CUDA REQUIRED)

find_package(PythonInterp)
find_package(PythonLibs)

# find_library(PYTHON_LIBRARIES
#             NAMES python27 libpython2.7.so
#             PATHS ${PYTHON_PREFIX_PATH}/libs/
#             ${PYTHON_PREFIX_PATH}/lib/
#             )
# set(PYTHON_INCLUDE_DIRS)
# set(PYTHON_LIBRARIES)
set(PYTHON_PREFIX_PATH)


 
file(GLOB_RECURSE SRC "*.cu")
file(GLOB_RECURSE HDR "*.h")

include_directories(${PYTHON_INCLUDE_DIRS})
 
cuda_add_library(${PROJECT_NAME} SHARED ${SRC} ${HDR})
            
target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
 
if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".pyd")
    ADD_CUSTOM_COMMAND(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_BINARY_DIR}/Release/${PROJECT_NAME}.pyd
        ${CMAKE_SOURCE_DIR}/${CMAKE_PROJECT_NAME}/${PROJECT_NAME}/${PROJECT_NAME}.pyd
    )
endif()
 
if(UNIX)
    set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
    ADD_CUSTOM_COMMAND(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.so
        ${CMAKE_SOURCE_DIR}/${CMAKE_PROJECT_NAME}/${PROJECT_NAME}/${PROJECT_NAME}.so
    )
endif()
