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

This source code is subject to the GNU Affero General Public License v3.0
license and without any warranty, see <https://github.com/muchip/FMCA>
for further information.
")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(OpenMP)

include_directories(${EIGEN3_INCLUDE_DIR})
message("${CMAKE_MODULE_PATH}")
message("${EIGEN3_INCLUDE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set (FMCA_PATH ${PROJECT_SOURCE_DIR})
message("${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS "-I${FMCA_PATH}")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

set( PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")

add_subdirectory(tests)
add_subdirectory(py)
enable_testing()
