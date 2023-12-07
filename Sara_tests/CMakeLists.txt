find_package(Eigen3 REQUIRED)

link_directories(${CMAKE_SOURCE_DIR}/FMCA)

include_directories(${CMAKE_SOURCE_DIR}/FMCA)

add_executable(test_GradKernel.out test_GradKernel.cpp)
add_test(test_GradKernel test_GradKernel.out)

add_executable(test_GradKernel_compressor.out test_GradKernel_compressor.cpp)
add_test(test_GradKernel_compressor test_GradKernel_compressor.out)

add_executable(test_Compression_time.out test_Compression_time.cpp)
add_test(test_Compression_time test_Compression_time.out)

add_executable(test_integration.out test_integration.cpp)
add_test(test_integration test_integration.out)

enable_testing()
