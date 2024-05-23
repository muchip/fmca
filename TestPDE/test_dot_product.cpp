#include <Eigen/Sparse>
#include <iostream>
#include <chrono>
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"

using namespace Eigen;
using namespace std;
typedef Eigen::SparseMatrix<double> SpMat; 

// Function to compute the specific entry (i, j) in the product of three matrices
double computeEntry(const SpMat &mat1, const SpMat &mat2, const SpMat &mat3, int i, int j) {
    double sum = 0.0;
    for (Eigen::SparseMatrix<double>::InnerIterator it1(mat1, i); it1; ++it1) {
        int k = it1.col(); // column index of mat1, row index of mat2
        if (mat2.innerVector(k).nonZeros() > 0) {
            for (Eigen::SparseMatrix<double>::InnerIterator it2(mat2, k); it2; ++it2) {
                int m = it2.col(); // column index of mat2, row index of mat3
                if (mat3.coeff(m, j) != 0) {
                    sum += it1.value() * it2.value() * mat3.coeff(m, j);
                }
            }
        }
    }
    return sum;
}

// Main function to compute the triple product based on a known sparsity pattern
void computeTripleProduct(const SpMat &pattern, const SpMat &mat1, const SpMat &mat2, const SpMat &mat3, SpMat &result) {
    result = SpMat(pattern.rows(), pattern.cols());
    for (int k = 0; k < pattern.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(pattern, k); it; ++it) {
            int i = it.row();
            int j = it.col();
            double val = computeEntry(mat1, mat2, mat3, i, j);
            if (val != 0.0) {
                result.coeffRef(i, j) = val;
            }
        }
    }
    result.makeCompressed(); // Optional, to clean up the memory and ensure data is compressed
}



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

    largeSparse pattern23(mat2.rows(),mat3.cols());
    std::vector<Eigen::Triplet<double>> triplets23;
    for (long long int i = 0; i < mat3.rows(); ++i) {
        for (long long int j = 0; j < mat3.cols(); ++j) {
            triplets23.push_back(Eigen::Triplet<double>(i, j, 1.0));
        }
    }
    pattern23.setFromTriplets(triplets23.begin(), triplets23.end());
    pattern23.makeCompressed();

    mat1.makeCompressed();
    mat2.makeCompressed();
    mat3.makeCompressed();
    formatted_sparse_multiplication_triple_product(pattern, mat1, mat2, mat3.transpose());

    std::cout << "mat1 = " << mat1 << std::endl;
    std::cout << "mat2 = " << mat2 << std::endl;
    std::cout << "mat3 = " << mat3 << std::endl;
    std::cout << "res = " << pattern << std::endl;
    std::cout << "res eigen = " << mat1*mat2*mat3 << std::endl;
    return 0;
}


/*
int main() {
    Eigen::SparseMatrix<double> mat1(3, 4);
    Eigen::SparseMatrix<double> mat2(4, 4);
    Eigen::SparseMatrix<double> mat3(4, 3);
    Eigen::SparseMatrix<double> result;

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

    Eigen::SparseMatrix<double> pattern(mat1.rows(),mat3.cols());
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
    computeTripleProduct(pattern, mat1, mat2, mat3.transpose(), result);

    std::cout << "mat1 = " << mat1 << std::endl;
    std::cout << "mat2 = " << mat2 << std::endl;
    std::cout << "mat3 = " << mat3 << std::endl;
    std::cout << "res = " << result << std::endl;
    return 0;
}
*/


