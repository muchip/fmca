cmake_minimum_required(VERSION 3.21)
project(FMCA CXX)

message(STATUS "
****************************************************************************
*                  ______   __    __    ______    ______                   * 
*                 |   ___| |  \\  /  |  /   ___|  /      \\                  *
*                 |   __   |   \\/   | |   /     |   /\\   |                 *
*                 |   __|  |        | |  |      |  |__|  |                 *
*                 |  |     |  |\\/|  | |  |      |   __   |                 *
*                 |  |     |  |  |  | |   \\___  |  |  |  |                 *
*                 |__|     |__|  |__|  \\______| |__|  |__|                 *
*                                                                          *
****************************************************************************
This is FMCA, the Fast Multiresolution Covariance Analysis package.

Copyright (c) 2022, Michael Multerer
All rights reserved.

This source code is subject to the GNU General Public License version 3
license and without any warranty, see <https://github.com/muchip/FMCA>
for further information.
")

#   Instructions for VisualStudio users (tested with VS 2019 Community)
#
#   Assuming you do not have some preixisting setup, you can compile 
#   tests and examples as follows: Uncomment the options
#
#   add_compile_options(-bigobj)
#   include_directories(eigen3)
#
#   in this file, and comment out the line  

#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
cmake_minimum_required (VERSION 3.0)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

include_directories(${EIGEN3_INCLUDE_DIR})
message("${CMAKE_MODULE_PATH}")
message("${EIGEN3_INCLUDE_DIR}")

set (FMCA_PATH ${PROJECT_SOURCE_DIR})

message("${PROJECT_SOURCE_DIR}")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" MARCHNATIVE)
CHECK_CXX_COMPILER_FLAG("-flto" FLTO)

set(CMAKE_CXX_FLAGS "-std=c++11 -I${FMCA_PATH}")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if (MARCHNATIVE)
message(STATUS "Found -march=native, adding option to builds of type >release< only")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native" CACHE STRING "compile flags" FORCE)
endif()
if (FLTO)
message(STATUS "Found -flto, adding option to builds of type >release< only")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto" CACHE STRING "compile flags" FORCE)
endif()

set( PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")

add_custom_target(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

add_custom_target(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

add_subdirectory(tests)
add_subdirectory(py)
enable_testing()
