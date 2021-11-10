cmake_minimum_required(VERSION 3.0)

project(my-bpnn)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -g")

add_subdirectory(propagation)
add_subdirectory(autograd)
add_subdirectory(module)
add_subdirectory(model)
add_subdirectory(utils)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE model module data_generator)

# file(GLOB_RECURSE MODEL_SRC "model/*.cpp")
# add_library(model ${MODEL_SRC})
# 
# file(GLOB_RECURSE MODULE_SRC "module/*.cpp")
# add_library(module ${MODULE_SRC})
# target_link_libraries(module PRIVATE autograd)
# 
# add_library(data_generator "utils/data_generator.cpp")
# 
# file(GLOB_RECURSE AUTOGRAD_SRC "autograd/*.cpp")
# file(GLOB_RECURSE PROPAGATION_SRC "propagation/*.cpp")
# add_library(autograd ${AUTOGRAD_SRC} ${PROPAGATION_SRC})

