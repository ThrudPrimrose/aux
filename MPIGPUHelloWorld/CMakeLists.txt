cmake_minimum_required(VERSION 3.10)

project(MPIGPUHelloWorld LANGUAGES CXX CUDA)

# Find CUDA package
#find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable
add_executable(MPIGPUHelloWorld MPIGPUHelloWorld.cpp HelloWorldKernel.cu)

# Link CUDA runtime library
target_link_libraries(MPIGPUHelloWorld PRIVATE ${CUDA_LIBRARIES})
target_include_directories(MPIGPUHelloWorld PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
