project(kernel_launcher LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tarch_mimic tarch_mimic.cpp check.cu)
add_executable(main main.cpp)

target_link_libraries(main PUBLIC tarch_mimic)