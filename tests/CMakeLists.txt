find_package(Eigen3 REQUIRED)

link_directories(${CMAKE_SOURCE_DIR}/FMCA)

include_directories(${CMAKE_SOURCE_DIR}/FMCA)

add_executable(test_ClusterTreeMetrics.out test_ClusterTreeMetrics.cpp)
add_test(ClusterTreeMetrics test_ClusterTreeMetrics.out)

add_executable(test_H2Matrix.out test_H2Matrix.cpp)
add_test(H2Matrix test_H2Matrix.out)

add_executable(test_H2SampletCompressor.out test_H2SampletCompressor.cpp)
add_test(H2SampletCompressor test_H2SampletCompressor.out)

add_executable(test_H2SampletCompressorUnsymmetric.out
  test_H2SampletCompressorUnsymmetric.cpp)
add_test(H2SampletCompressorUnsymmetric test_H2SampletCompressorUnsymmetric.out)

add_executable(test_RandomTreeAccessor.out test_RandomTreeAccessor.cpp)
add_test(RandomTreeAccessor test_RandomTreeAccessor.out)

add_executable(test_SampletBasis.out test_SampletBasis.cpp)
add_test(SampletBasis test_SampletBasis.out)

add_executable(test_SampletTransformer.out test_SampletTransformer.cpp)
add_test(SampletTransformer test_SampletTransformer.out)

add_executable(test_TreeIterator.out test_TreeIterator.cpp)
add_test(TreeIterator test_TreeIterator.out)

add_executable(test_kNN.out test_kNN.cpp)
add_test(kNN test_kNN.out)

enable_testing()
