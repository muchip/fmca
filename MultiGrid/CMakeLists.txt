# Multigrid/CMakeLists.txt
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add Multigrid library
add_library(MultigridLibrary
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Evaluation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Compression.cpp
)

# Include directories for Multigrid
target_include_directories(MultigridLibrary PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${FMCA_PATH}/FMCA
    ${EIGEN3_INCLUDE_DIR}
)

# Link Eigen (if needed)
target_link_libraries(MultigridLibrary Eigen3::Eigen)

# Link OpenMP (if enabled)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MultigridLibrary OpenMP::OpenMP_CXX)
endif()

# Optional: Add compiler warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(MultigridLibrary PRIVATE -Wall -Wextra -pedantic -fopenmp)
endif()
