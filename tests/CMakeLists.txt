find_package(Eigen3 REQUIRED)

link_directories(${CMAKE_SOURCE_DIR}/FMCA)

include_directories(${CMAKE_SOURCE_DIR}/FMCA)

add_executable(test_SampletBasis.out test_SampletBasis.cpp)
add_test(SampletBasis test_SampletBasis.out)

add_executable(test_ClusterTreeMetrics.out test_ClusterTreeMetrics.cpp)
add_test(ClusterTreeMetrics test_ClusterTreeMetrics.out)

enable_testing()
