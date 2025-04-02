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

# Add CMake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(OpenMP)

# Include Eigen headers
include_directories(${EIGEN3_INCLUDE_DIR})

# Set C++ standardc
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FMCA_PATH ${PROJECT_SOURCE_DIR})
set(MSI_PATH ${PROJECT_SOURCE_DIR}/MSI)

# Set compiler flags
set(CMAKE_CXX_FLAGS "-I${FMCA_PATH} -I${MSI_PATH}")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# Include project source directory
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/MSI")

# Check compiler and set OpenMP flags appropriately
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # For Clang, either remove OpenMP or use the correct flag
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  message(STATUS "Detected Clang compiler - disabling OpenMP")
elseif(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# Add tests and pybind11 subdirectories
add_subdirectory(tests)
add_subdirectory(py)

# Enable testing
enable_testing()

