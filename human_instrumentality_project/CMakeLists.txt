cmake_minimum_required(VERSION 3.9)

project(HumanInstrumentalityProject CXX)

find_package(MPI)
find_package(OpenMP)

include_directories(${MPI_INCLUDE_PATH})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif(NOT CMAKE_BUILD_TYPE)

add_executable(hip human_instrumentality_project.cpp)
target_link_libraries(hip ${MPI_LIBRARIES})
target_link_libraries(hip PUBLIC OpenMP::OpenMP_CXX)
