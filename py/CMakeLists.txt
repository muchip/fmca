pybind11_add_module(FMCA MODULE FMCA.cpp)
target_link_libraries(FMCA PRIVATE FMCA::FMCA)

# used by pip (scikit-build-core); inert for a standalone build
install(TARGETS FMCA LIBRARY DESTINATION .)

# Standalone build: copy the example notebooks next to the built module
if(NOT SKBUILD)
  foreach(nb
      FMCA_Cholesky.ipynb
      FMCA_GP.ipynb
      FMCA_H2Matrix.ipynb
      FMCA_LowRankBenchmarks.ipynb
      FMCA_SampletAdaptiveClustering.ipynb
      FMCA_SampletCompression1D.ipynb
      FMCA_SampletDenoising1D.ipynb
      FMCA_SampletImageDenoising.ipynb
      FMCA_SampletKernelCompression.ipynb
      FMCA_Samplet_GP_Filtering.ipynb
      FMCA_Samplet_KRR.ipynb
      FMCA_Samplets.ipynb
  )
    add_custom_command(TARGET FMCA POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
              ${CMAKE_CURRENT_SOURCE_DIR}/${nb}
              ${CMAKE_CURRENT_BINARY_DIR}/${nb}
    )
  endforeach()
endif()
