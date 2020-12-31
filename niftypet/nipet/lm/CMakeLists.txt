cmake_minimum_required(VERSION 3.2)

project(mmr_lmproc)

file(GLOB SRC LIST_DIRECTORIES false "src/*.cu")

include_directories(${CMAKE_CURRENT_LIST_DIR}/../include)
include_directories(src)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

add_library(mmr_lmproc MODULE ${SRC})
target_link_libraries(mmr_lmproc mmr_auxe ${Python3_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

if(SKBUILD)
python_extension_module(mmr_lmproc)
endif()
install(TARGETS mmr_lmproc LIBRARY DESTINATION "niftypet/${CMAKE_PROJECT_NAME}/lm")
