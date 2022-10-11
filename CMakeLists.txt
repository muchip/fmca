cmake_minimum_required(VERSION 3.21)
project(FMCA CXX)
# Fetch dependency 
include(FetchContent)
 # More verbose Output for libraries using set_package_properties

include(FeatureSummary)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module Development.Embed REQUIRED)
else()
  find_package(Python 3.9 COMPONENTS Interpreter Development REQUIRED)
endif()
# Get include directory of Python
execute_process(COMMAND python3-config --includes
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE _PYTHON_HEADERS
		    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# for some reason the include dir is returned twice
# and with -I prepended which must be removed. 
separate_arguments(_INCLDIRS_LIST UNIX_COMMAND "${_PYTHON_HEADERS}")  # Convert to list
list(GET _INCLDIRS_LIST 0 PYTHON_HEADER_0)  # save the 1st element
string(REPLACE "-I" "" PYTHON_INCLUDE_DIRS ${PYTHON_HEADER_0})  # chop off leading -I, save result
 
message("python header are ${_PYTHON_HEADERS}")
message("python env variable for header ${PYTHON_INCLUDE_DIRS}")

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

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.0 
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    include_directories(${pybind11_SOURCE_DIR}/include)
else()
  find_package(pybind11  REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})
  set(pybind11_INCLUDE_DIR ${PYBIND11_DIR}/include)
  include_directories(${pybind11_INCLUDE_DIR})

endif()

if (NOT TARGET Eigen3::Eigen)


    FetchContent_Declare(
	  Eigen3
	  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
	  GIT_TAG        3.4.0
	  QUIET
	  CMAKE_ARGS
		-DEIGEN_TEST_CXX11:BOOL=OFF
		-DBUILD_TESTING:BOOL=OFF
		-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/_dep/Eigen_install
	 )
      
    FetchContent_GetProperties(Eigen3)
    if(NOT Eigen3_POPULATED)
  	FetchContent_Populate(Eigen3)
  	        
	  add_subdirectory(${eigen3_SOURCE_DIR} ${eigen3_BINARY_DIR})
	  message("SRC; ${Eigen3_SOURCE_DIR}") # Apparently empty?
	  message("BIN: ${Eigen3_BINARY_DIR}") # Apparently empty?
	  include_directories(${eigen3_SOURCE_DIR})
	  #FetchContent_MakeAvailable(eigen3)
	  message( "ls ${eigen3_SOURCE_DIR}")
	endif()
endif()


set(PYBIND11_CPP_STANDARD -std=c++11)
message("${CMAKE_MODULE_PATH}")
message("${eigen3_INCLUDE_DIR}")
message("${pybind11_INCLUDE_DIR}")

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
