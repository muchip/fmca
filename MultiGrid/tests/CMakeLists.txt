cmake_minimum_required(VERSION 3.21)

# Find package dependencies
find_package(Eigen3 REQUIRED)

add_executable(test_FrankeFunction_MLI.out test_FrankeFunction_MLI.cpp)
add_test(test_FrankeFunction_MLI test_FrankeFunction_MLI.out)

enable_testing()