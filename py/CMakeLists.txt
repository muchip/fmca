find_package(Eigen3 REQUIRED)

link_directories(${CMAKE_SOURCE_DIR}/FMCA INTERFACE Eigen3)

include_directories(${CMAKE_SOURCE_DIR}/FMCA)


add_library(FMCA MODULE FMCA.cpp)
execute_process(OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND 
python3-config --extension-suffix OUTPUT_VARIABLE FMCA_SUFFIX)
message("Library suffix: ${FMCA_SUFFIX}")
SET_TARGET_PROPERTIES(FMCA PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(FMCA PROPERTIES SUFFIX "${FMCA_SUFFIX}")

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
