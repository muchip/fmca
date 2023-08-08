include(FetchContent)

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