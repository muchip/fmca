# Include these modules to handle the QUIETLY and REQUIRED arguments.
include(FindPackageHandleStandardArgs)
include(FetchContent)



FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.0 
	CMAKE_ARGS
	    -DPYBIND11_FINDPYTHON=ON
	    -DPYTHON_EXECUTABLE=$(python -c "import sys; print(sys.executable)")
	    
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    include_directories(${pybind11_SOURCE_DIR}/include)
else()
  find_package(pybind11  REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})


endif()



## Something from Python Include needed for CI 
	find_path(PYBIND11_INCLUDE_DIR
	NAMES pybind11/pybind11.h pybind11/eigen.h
	HINTS ${PYBIND11_DIR}/include
	)
	include_directories(${PYBIND11_DIR}/include)
	find_package_handle_standard_args(pybind11 DEFAULT_MSG PYBIND11_INCLUDE_DIR)



if(DEFINED ${PYTHONINCLUDEDIRS})
	include_directories(${PYTHONINCLUDEDIRS})
endif()