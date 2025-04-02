find_package(Eigen3 REQUIRED)

link_directories(${CMAKE_SOURCE_DIR}/FMCA)
link_directories(${CMAKE_SOURCE_DIR}/MSI)

include_directories(${CMAKE_SOURCE_DIR}/FMCA)
include_directories(${CMAKE_SOURCE_DIR}/MSI)

add_executable(test_ClusterTreeMetrics.out test_ClusterTreeMetrics.cpp)
add_test(ClusterTreeMetrics test_ClusterTreeMetrics.out)

add_executable(test_H2DualClusterBases.out test_H2DualClusterBases.cpp)
add_test(H2DualClusterBases test_H2DualClusterBases.out)

add_executable(test_H2Matrix.out test_H2Matrix.cpp)
add_test(H2Matrix test_H2Matrix.out)

add_executable(test_H2SampletCompressor.out test_H2SampletCompressor.cpp)
add_test(H2SampletCompressor test_H2SampletCompressor.out)

add_executable(test_H2SampletCompressorUnsymmetric.out
  test_H2SampletCompressorUnsymmetric.cpp)
add_test(H2SampletCompressorUnsymmetric test_H2SampletCompressorUnsymmetric.out)

add_executable(test_H2kernelActor.out test_H2kernelActor.cpp)
add_test(H2kernelActor test_H2kernelActor.out)

add_executable(test_KMinList.out test_KMinList.cpp)
add_test(KMinList test_KMinList.out)

add_executable(test_MortonClusterTree.out test_MortonClusterTree.cpp)
add_test(MortonClusterTree test_MortonClusterTree.out)

add_executable(test_MortonCompare.out test_MortonCompare.cpp)
add_test(MortonCompare test_MortonCompare.out)

add_executable(test_RandomTreeAccessor.out test_RandomTreeAccessor.cpp)
add_test(RandomTreeAccessor test_RandomTreeAccessor.out)

add_executable(test_SampletBasis.out test_SampletBasis.cpp)
add_test(SampletBasis test_SampletBasis.out)

add_executable(test_SampletTransformer.out test_SampletTransformer.cpp)
add_test(SampletTransformer test_SampletTransformer.out)

add_executable(test_TreeIterator.out test_TreeIterator.cpp)
add_test(TreeIterator test_TreeIterator.out)

add_executable(test_AdaptiveTreeSearch.out test_AdaptiveTreeSearch.cpp)
add_test(AdaptiveTreeSearch test_AdaptiveTreeSearch.out)

add_executable(test_kNN.out test_kNN.cpp)
add_test(kNN test_kNN.out)

add_executable(test_FrankeFunction_MSI.out test_FrankeFunction_MSI.cpp)
add_test(FrankeFunction_MSI test_FrankeFunction_MSI.out)

enable_testing()
