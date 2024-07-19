#include <Eigen/Sparse>
#include <iostream>
#include <chrono>
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"

using namespace Eigen;
using namespace std;


int main() {
    largeSparse mat1(3, 4);
    largeSparse mat2(4, 4);
    largeSparse mat3(4, 3);

    mat1.insert(0, 0) = 1;
    mat1.insert(0, 1) = 2;
    mat1.insert(0, 3) = 1;
    mat1.insert(1, 3) = 1;
    mat1.insert(2, 0) = 1;
    mat1.insert(2, 1) = 2;

    mat2.insert(0, 0) = 1;
    mat2.insert(0, 1) = 1;
    mat2.insert(1, 3) = 1;
    mat2.insert(2, 1) = 1;
    mat2.insert(2, 2) = 1;
    mat2.insert(3, 0) = 1;
    mat2.insert(3, 2) = 1;

    mat3.insert(0, 0) = 1;
    mat3.insert(0, 1) = 1;
    mat3.insert(1, 0) = 2;
    mat3.insert(1, 1) = 1;
    mat3.insert(2, 1) = 1;
    mat3.insert(3, 0) = 1;
    mat3.insert(3, 1) = 1;
    mat3.insert(3, 2) = 2;
    //mat3.setIdentity();

    largeSparse pattern(mat1.rows(),mat3.cols());
    std::vector<Eigen::Triplet<double>> triplets;
    for (long long int i = 0; i < mat1.rows(); ++i) {
        for (long long int j = 0; j < mat3.cols(); ++j) {
            triplets.push_back(Eigen::Triplet<double>(i, j, 1.0));
        }
    }
    pattern.setFromTriplets(triplets.begin(), triplets.end());
    pattern.makeCompressed();

    mat1.makeCompressed();
    mat2.makeCompressed();
    mat3.makeCompressed();
    formatted_sparse_multiplication_triple_product_largeSparse(pattern, mat1, mat2, mat3.transpose());

    std::cout << "mat1 = \n" << mat1 << std::endl;
    std::cout << "mat2 = \n" << mat2 << std::endl;
    std::cout << "mat3 = \n" << mat3 << std::endl;
    std::cout << "res = \n" << pattern.transpose() << std::endl;
    std::cout << "res eigen = \n" << mat1*mat2*mat3 << std::endl;
    return 0;
}



