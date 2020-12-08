cmake_minimum_required(VERSION 3.2)
# set project name as the module name of the .dll/.pyd
project(petprj)
# set name of folder, because it isnt set to the project name :(
set(CURR_FOLDER_NAME "prj")

file(GLOB SRC LIST_DIRECTORIES false "src/*.cu")

include_directories(${CMAKE_SOURCE_DIR}/${CMAKE_PROJECT_NAME}/include)
include_directories(src)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

cuda_add_library(${PROJECT_NAME} SHARED ${SRC})
target_link_libraries(${PROJECT_NAME} mmr_auxe ${Python3_LIBRARIES} ${CUDA_LIBRARIES})

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
