find_package(Eigen3 REQUIRED)
find_package(Pybind11 REQUIRED)


link_directories(${CMAKE_SOURCE_DIR}/FMCA INTERFACE Eigen3)

include_directories(${CMAKE_SOURCE_DIR}/FMCA)


add_library(FMCA MODULE FMCA.cpp)

execute_process(OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND 
python3-config --extension-suffix OUTPUT_VARIABLE FMCA_SUFFIX)
message("Library suffix: ${FMCA_SUFFIX}")


  execute_process( COMMAND  python3-config "--includes"
                   OUTPUT_VARIABLE PYTHON_INCLUDE
                   RESULT_VARIABLE PYTHON_INCLUDE_RESULT
                   ERROR_QUIET    OUTPUT_STRIP_TRAILING_WHITESPACE
                   )

  if(NOT PYTHON_INCLUDE_RESULT)
  message(STATUS "Found python include at directory ${PYTHON_INCLUDE}")
  add_compile_options("${PYTHON_INCLUDE}")
  target_include_directories(FMCA PRIVATE ${PYTHON_INCLUDE})
  set_target_properties(FMCA PROPERTIES COMPILE_FLAGS ${PYTHON_INCLUDE} )
  else()
    message(STATUS "PYTHON Include could not be found by execute process.")
  endif()


set_target_properties(FMCA PROPERTIES SUFFIX "${FMCA_SUFFIX}")



add_custom_command(TARGET FMCA POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
${CMAKE_SOURCE_DIR}/py/FMCA_Samplets.ipynb
${CMAKE_CURRENT_BINARY_DIR}/FMCA_Samplets.ipynb)

add_custom_command(TARGET FMCA POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
${CMAKE_SOURCE_DIR}/py/FMCA_Cholesky.ipynb
${CMAKE_CURRENT_BINARY_DIR}/FMCA_Cholesky.ipynb)

add_custom_command(TARGET FMCA POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
${CMAKE_SOURCE_DIR}/py/FMCA_GP.ipynb
${CMAKE_CURRENT_BINARY_DIR}/FMCA_GP.ipynb)


add_custom_command(TARGET FMCA POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_SOURCE_DIR}/py/FMCA_H2Matrix.ipynb
${CMAKE_CURRENT_BINARY_DIR}/FMCA_H2Matrix.ipynb)

if( ${CMAKE_HOST_WIN32}) # 
set_target_properties(FMCA PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
SUFFIX "${PYTHON_MODULE_EXTENSION}")
target_link_libraries(FMCA PRIVATE  pybind11::module)
else() 
target_link_libraries(FMCA INTERFACE pybind11::module Eigen::Eigen3)
endif()
