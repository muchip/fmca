find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_library(FMCA MODULE FMCA.cpp)
target_link_libraries(FMCA PRIVATE Eigen3::Eigen pybind11::module)

# Python extension suffix
execute_process(
  COMMAND python3-config --extension-suffix
  OUTPUT_VARIABLE FMCA_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Library suffix: ${FMCA_SUFFIX}")

set_target_properties(FMCA PROPERTIES
  PREFIX ""
  SUFFIX "${FMCA_SUFFIX}"
)

# Copy notebooks next to the built module
foreach(nb
    FMCA_Samplets.ipynb
    FMCA_Cholesky.ipynb
    FMCA_GP.ipynb
    FMCA_H2Matrix.ipynb
    FMCA_Samplet_GP_Filtering.ipynb
    FMCA_Samplet_KRR.ipynb
    FMCA_SampletAdaptiveClustering.ipynb
)
  add_custom_command(TARGET FMCA POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/py/${nb}
            ${CMAKE_CURRENT_BINARY_DIR}/${nb}
  )
endforeach()



