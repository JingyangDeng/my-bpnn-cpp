project(autograd)

file(GLOB_RECURSE SRC *.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC})

target_link_libraries(${PROJECT_NAME} PRIVATE propagation)
