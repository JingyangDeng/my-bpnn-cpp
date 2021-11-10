project(model)

file(GLOB_RECURSE SRC *.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC})
