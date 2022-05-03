cmake_minimum_required(VERSION 3.9)

project(simpleMPI CXX)

find_package(UPCXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif(NOT CMAKE_BUILD_TYPE)

add_executable(simpleUPCXX simpleUPCXX.cpp)

target_include_directories(simpleUPCXX PRIVATE ${UPCXX_INCLUDE_DIRS})
target_compile_options(simpleUPCXX PUBLIC ${UPCXX_CXXFLAGS})
target_compile_definitions(simpleUPCXX PUBLIC ${UPCXX_DEFINITIONS})
target_link_libraries(simpleUPCXX PUBLIC UPCXX::upcxx)

