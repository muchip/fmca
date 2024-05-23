#include <cstdlib>
#include <iostream>
#include <Eigen/Sparse>
#include "../FMCA/GradKernel"


Eigen::SparseMatrix<double> NeumannNormalMultiplication(
    const Eigen::SparseMatrix<double> &grad, const FMCA::Vector &normal) {
  //     Eigen::SparseMatrix<double> result(grad.rows(), grad.cols());
  //     std::vector<Eigen::Triplet<double>> tripletList;
  //     tripletList.reserve(grad.nonZeros());

  //     for (int k = 0; k < grad.outerSize(); ++k) {
  //         for (Eigen::SparseMatrix<double>::InnerIterator it(grad, k); it;
  //         ++it) {
  //             tripletList.emplace_back(it.row(), it.col(), it.value() *
  //             normal(it.col()));
  //         }
  //     }

  //     result.setFromTriplets(tripletList.begin(), tripletList.end());
  //     return result;
  // }
  Eigen::SparseMatrix<double> diagNormal(grad.cols(), grad.cols());
  std::vector<Eigen::Triplet<double>> triplets;
  for (int i = 0; i < normal.size(); ++i) {
    triplets.emplace_back(i, i, normal(i));
  }
  diagNormal.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::SparseMatrix<double> result = grad * diagNormal;
  return result;
}
