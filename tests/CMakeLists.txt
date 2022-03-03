add_executable(test_H2matrix.out test_H2matrix.cpp)
target_link_libraries(test_H2matrix.out Eigen3::Eigen)
add_test(H2matrix test_H2matrix.out)

add_executable(test_SampletBasis.out test_SampletBasis.cpp)
target_link_libraries(test_SampletBasis.out Eigen3::Eigen)
add_test(SampletBasis test_SampletBasis.out)

add_executable(test_H2SampletCompressor.out test_H2SampletCompressor.cpp)
target_link_libraries(test_H2SampletCompressor.out Eigen3::Eigen)
add_test(H2SampletCompressor test_H2SampletCompressor.out)

enable_testing()
