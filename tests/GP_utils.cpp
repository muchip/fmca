#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/KernelInterpolation"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"

using namespace FMCA;

using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<SampletMoments, FMCA::CovarianceKernel>;

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// CONDITION NUMBER COMPUTATION

// Method 1: Lanczos-based estimation
Scalar estimateConditionNumberLanczos(const SparseMatrix& A, int max_iter = 50) {
  if (A.rows() != A.cols()) {
    return std::numeric_limits<Scalar>::infinity();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
  if (A.rows() > 5000) {
    // Use power iteration to estimate largest eigenvalue
    Vector v = Vector::Random(A.rows()).normalized();
    Scalar lambda_max = 0.0;
    for (int i = 0; i < max_iter; ++i) {
      Vector Av = A * v;
      lambda_max = v.dot(Av);
      v = Av.normalized();
    }
    // Use inverse power iteration for smallest eigenvalue
    Eigen::SparseLU<SparseMatrix> lu_solver;
    lu_solver.compute(A);
    if (lu_solver.info() != Eigen::Success) {
      return std::numeric_limits<Scalar>::infinity();
    }
    v = Vector::Random(A.rows()).normalized();
    Scalar lambda_min = std::numeric_limits<Scalar>::max();
    for (int i = 0; i < max_iter; ++i) {
      Vector x = lu_solver.solve(v);
      if (lu_solver.info() != Eigen::Success) {
        return std::numeric_limits<Scalar>::infinity();
      }
      Scalar rayleigh = v.dot(x);
      if (std::abs(rayleigh) > 1e-15) {
        lambda_min = std::min(lambda_min, 1.0 / rayleigh);
      }
      v = x.normalized();
    }
    return std::abs(lambda_max / lambda_min);
  } else {
    // For smaller matrices, convert to dense and use exact methods
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
    solver.compute(A_dense);
    if (solver.info() != Eigen::Success) {
      return std::numeric_limits<Scalar>::infinity();
    }
    auto eigenvals = solver.eigenvalues();
    Scalar lambda_max = eigenvals.maxCoeff();
    Scalar lambda_min = eigenvals.minCoeff();
    if (std::abs(lambda_min) < 1e-15) {
      return std::numeric_limits<Scalar>::infinity();
    }
    return std::abs(lambda_max / lambda_min);
  }
}

Scalar estimateConditionNumberSVD(const SparseMatrix& A, Scalar tol = 1e-12) {
  if (A.rows() != A.cols()) {
    return std::numeric_limits<Scalar>::infinity();
  }

  if (A.rows() > 2000) {
    std::cout << "Warning: SVD method not recommended for matrices larger than "
                 "2000x2000"
              << std::endl;
    return estimateConditionNumberLanczos(A);
  }
  // Convert to dense matrix for SVD
  Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A_dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto singular_values = svd.singularValues();
  Scalar sigma_max = singular_values(0);
  Scalar sigma_min = singular_values(singular_values.size() - 1);
  if (sigma_min < tol * sigma_max) {
    return std::numeric_limits<Scalar>::infinity();
  }
  return sigma_max / sigma_min;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
Scalar robustConditionNumber(const SparseMatrix& A) {
  std::cout << "Matrix size: " << A.rows() << "x" << A.cols()
            << ", nnz: " << A.nonZeros() << std::endl;

  if (A.rows() != A.cols()) {
    std::cout << "Matrix is not square!" << std::endl;
    return std::numeric_limits<Scalar>::infinity();
  }
  if (A.norm() < 1e-15) {
    std::cout << "Matrix is essentially zero!" << std::endl;
    return std::numeric_limits<Scalar>::infinity();
  }

  if (A.rows() <= 1000) {
    std::cout << "Using SVD method..." << std::endl;
    return estimateConditionNumberSVD(A);
  } else {
    std::cout << "Using Lanczos-based estimation..." << std::endl;
    return estimateConditionNumberLanczos(A);
  }
}