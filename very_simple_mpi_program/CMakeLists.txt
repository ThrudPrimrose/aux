cmake_minimum_required(VERSION 3.9)

project(simpleMPI CXX)

find_package(MPI)
find_package(OpenMP)

include_directories(${MPI_INCLUDE_PATH})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif(NOT CMAKE_BUILD_TYPE)

add_executable(simpleMPI simpleMPI.cpp)
target_link_libraries(simpleMPI PUBLIC ${MPI_LIBRARIES})
target_link_libraries(simpleMPI PUBLIC OpenMP::OpenMP_CXX)
