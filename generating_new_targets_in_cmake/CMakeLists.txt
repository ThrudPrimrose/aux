cmake_minimum_required(VERSION 3.20.5)
project(generate_new_targets_in_cmake)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED TRUE)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)


execute_process(COMMAND cp -rsfn "${CMAKE_SOURCE_DIR}/src" "${CMAKE_BINARY_DIR}/src")

execute_process(COMMAND touch "${CMAKE_BINARY_DIR}/GeneratedSubdirectory.cmake")

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release" CACHE STRING "Enabled build types" FORCE)
if(NOT CMAKE_BUILD_TYPE OR NOT ${CMAKE_BUILD_TYPE} IN_LIST CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "${CMAKE_CONFIGURATION_TYPES}" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
  list(APPEND BUILD_MODES ${CMAKE_CONFIGURATION_TYPES})
else()
  list(APPEND BUILD_MODES ${CMAKE_BUILD_TYPE})
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_BINARY_DIR}/timestamp)

# If we change the Cache, we will update the time.stamp to trigger the garget for a reconfigure
# Generated applications depend on the reconf target
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/time.stamp
  COMMENT "Run custom command on CMakeCache change"
  DEPENDS ${CMAKE_BINARY_DIR}/CMakeCache.txt
)

# Need to create a custom target for custom command to work
add_custom_target(reconf ALL
  # Use absolute path for the DEPENDS file.
  # Relative paths are interpreted relative to the source directory.
  DEPENDS ${CMAKE_BINARY_DIR}/time.stamp ${CMAKE_BINARY_DIR}/CMakeCache.txt
)


add_subdirectory(src)
include(${CMAKE_BINARY_DIR}/GeneratedSubdirectory.cmake)