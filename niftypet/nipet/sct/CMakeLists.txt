cmake_minimum_required(VERSION 3.2)

project(nifty_scatter)

file(GLOB SRC LIST_DIRECTORIES false "src/*.cu")

include_directories(${CMAKE_CURRENT_LIST_DIR}/../include)
include_directories(src)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

add_library(nifty_scatter MODULE ${SRC})
target_link_libraries(nifty_scatter mmr_auxe ${Python3_LIBRARIES} ${CUDA_LIBRARIES})

if(SKBUILD)
python_extension_module(nifty_scatter)
endif()
install(TARGETS nifty_scatter LIBRARY DESTINATION "niftypet/${CMAKE_PROJECT_NAME}/sct")
