cmake_minimum_required(VERSION 3.10)

project(MPIGPUHelloWorld LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(MPIGPUHelloWorld MPIGPUHelloWorld.cpp HelloWorldKernel.cu)

set_target_properties(MPIGPUHelloWorld PROPERTIES CUDA_ARCHITECTURES 86)

target_compile_options(MPIGPUHelloWorld PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>)
target_compile_options(MPIGPUHelloWorld PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fPIC -fPIE>)

target_link_libraries(MPIGPUHelloWorld PRIVATE ${CUDA_LIBRARIES})
target_include_directories(MPIGPUHelloWorld PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(MPIGPUHelloWorld PRIVATE MPI::MPI_CXX)