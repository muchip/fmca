cmake_minimum_required(VERSION 3.21)
project(FMCA VERSION 0.1.0 LANGUAGES CXX)

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

# Language standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options (SKBUILD is defined by scikit-build-core; a pip build defaults to
# the python module only)
if(SKBUILD)
  set(_fmca_default_tests OFF)
else()
  set(_fmca_default_tests ON)
endif()
option(FMCA_BUILD_TESTS  "Build the C++ tests"       ${_fmca_default_tests})
option(FMCA_BUILD_PYTHON "Build the pybind11 module" ON)

# Eigen >= 5.0 (new SVD API): system package, otherwise fetch the sources and
# expose them as a header-only target (Eigen's own build system is
# deliberately not entered)
find_package(Eigen3 5.0 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
  message(STATUS "Eigen3 >= 5.0 not found, fetching Eigen 5.0.1")
  include(FetchContent)
  FetchContent_Declare(eigen
    URL      https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR cmake_disabled)
  FetchContent_MakeAvailable(eigen)
  add_library(eigen INTERFACE)
  add_library(Eigen3::Eigen ALIAS eigen)
  target_include_directories(eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")
endif()

# OpenMP: optional, consumed as a target rather than via global flags
find_package(OpenMP COMPONENTS CXX)
if(NOT OpenMP_CXX_FOUND)
  message(WARNING "OpenMP not found; FMCA will run single-threaded. "
    "Pass an OpenMP-capable compiler via -DCMAKE_CXX_COMPILER=... in a fresh build directory.")
endif()

# FMCA: header-only interface target carrying all usage requirements
add_library(fmca_headers INTERFACE)
add_library(FMCA::FMCA ALIAS fmca_headers)
target_include_directories(fmca_headers INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_compile_features(fmca_headers INTERFACE cxx_std_17)
target_link_libraries(fmca_headers INTERFACE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fmca_headers INTERFACE OpenMP::OpenMP_CXX)
endif()
target_compile_options(fmca_headers INTERFACE
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3>)

# Python module
if(FMCA_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  # standalone builds may only have pybind11 from pip: ask the interpreter
  # for its CMake config directory before find_package
  if(NOT pybind11_DIR)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _pybind11_rc ERROR_QUIET)
    if(_pybind11_rc EQUAL 0)
      set(pybind11_DIR "${_pybind11_dir}")
    endif()
  endif()
  find_package(pybind11 CONFIG REQUIRED)
  add_subdirectory(py)
endif()

# Tests
if(FMCA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
