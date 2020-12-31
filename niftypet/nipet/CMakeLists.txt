cmake_minimum_required(VERSION 3.2)

project(mmr_auxe)

file(GLOB SRC LIST_DIRECTORIES false "src/*.cu")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

add_library(mmr_auxe SHARED ${SRC})
target_link_libraries(mmr_auxe ${Python3_LIBRARIES} ${CUDA_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")  # remove shared lib prefix to make importable
if(SKBUILD)
python_extension_module(mmr_auxe)
endif()
install(TARGETS mmr_auxe LIBRARY DESTINATION "niftypet/${CMAKE_PROJECT_NAME}")

add_subdirectory(lm)
add_subdirectory(prj)
add_subdirectory(sct)
