add_library(a_${CMAKE_BUILD_TYPE} a.cpp)

add_dependencies(a_${CMAKE_BUILD_TYPE} reconf)