find_package(Eigen3 REQUIRED)

link_directories(${CMAKE_SOURCE_DIR}/FMCA)

include_directories(${CMAKE_SOURCE_DIR}/FMCA)

add_executable(test_SampletBasis.out test_SampletBasis.cpp)
add_test(SampletBasis test_SampletBasis.out)

add_executable(test_ClusterTreeMetrics.out test_ClusterTreeMetrics.cpp)
add_test(ClusterTreeMetrics test_ClusterTreeMetrics.out)

add_executable(test_SampletTransformer.out test_SampletTransformer.cpp)
add_test(SampletTransformer test_SampletTransformer.out)

add_executable(test_TreeIterator.out test_TreeIterator.cpp)
add_test(TreeIterator test_TreeIterator.out)

add_executable(test_H2Matrix.out test_H2Matrix.cpp)
add_test(H2Matrix test_H2Matrix.out)

enable_testing()
