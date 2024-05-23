#include <iostream>
#include <chrono>
#include <Eigen/Sparse>
#include "Uzawa.h"

Eigen::SparseMatrix<double> S(3,3);
Eigen::SparseMatrix<double> A(3,1);
FMCA::Vector f(3);
FMCA::Vector g(1);

int main() {
S.insert(0,0) = 4;
S.insert(0,1) = 1;
S.insert(1,0) = 1;
S.insert(1,1) = 4;
S.insert(2,2) = 1;

A.insert(0,0) = 1;

f(0) = 10;
f(1) = 9;
f(2) = 3;
g(0) = 1;

FMCA::Vector solution = UzawaAlgorithm(S,A,f,g);
std::cout << solution << std::endl;
}

