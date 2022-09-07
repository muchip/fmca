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
#find_package(pybind11 REQUIRED)

find_package(pybind11 2.7.0 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
else()
  find_package(Python 3.9 COMPONENTS Interpreter Development REQUIRED)
endif()

set(PYBIND11_CPP_STANDARD -std=c++11)


set(pybind11_INCLUDE_DIR ${PYBIND11_DIR}/include)
include_directories(${pybind11_INCLUDE_DIR})

include_directories(${EIGEN3_INCLUDE_DIR})
message("${CMAKE_MODULE_PATH}")
message("${EIGEN3_INCLUDE_DIR}")
message("${PYBIND11_INCLUDE_DIR}")

set (FMCA_PATH ${PROJECT_SOURCE_DIR})
message("${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS "-std=c++11 -I${FMCA_PATH}")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set( PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")

add_subdirectory(tests)
add_subdirectory(py)
enable_testing()
