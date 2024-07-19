
/* This header contains functions useful for the resolution of the PDE using
Samplet compression. In particular, there are the compression commands for the
symmetric and unsymmetric cases. See SolvePoisson.h for an example of their
application. */

#include <cstdlib>
#include <iostream>
// ##############################
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluatorKernel =
    FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluatorKernel =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;


FMCA::Scalar computeTrace(const Eigen::SparseMatrix<FMCA::Scalar> &mat) {
  FMCA::Scalar trace = 0.0;
  for (FMCA::Index k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<FMCA::Scalar>::InnerIterator it(mat, k); it; ++it) {
      if (it.row() == it.col()) {
        trace += it.value();
      }
    }
  }
  return trace;
}

int countSmallerThan(const Eigen::SparseMatrix<FMCA::Scalar> &matrix,
                     FMCA::Scalar threshold) {
  int count = 0;
  for (FMCA::Index k = 0; k < matrix.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<FMCA::Scalar>::InnerIterator it(matrix, k); it;
         ++it) {
      if (abs(it.value()) < threshold && abs(it.value()) != 0) {
        count++;
      }
    }
  }
  return count / matrix.rows();
}

bool isSymmetric(const Eigen::SparseMatrix<FMCA::Scalar> &matrix,
                 FMCA::Scalar tol = 1e-10) {
  if (matrix.rows() != matrix.cols())
    return false;
  for (FMCA::Index k = 0; k < matrix.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<FMCA::Scalar>::InnerIterator it(matrix, k); it;
         ++it) {
      if (std::fabs(it.value() - matrix.coeff(it.col(), it.row())) > tol)
        return false;
    }
  }
  return true;
}

template <typename KernelType, typename EvaluatorType>
Eigen::SparseMatrix<FMCA::Scalar> createCompressedSparseMatrixSymmetric(
    const KernelType &kernel, const EvaluatorType &evaluator,
    const H2SampletTree &hst_sources, FMCA::Scalar &eta, FMCA::Scalar &threshold,
    FMCA::Matrix &P_rows) {
  int N_rows = P_rows.cols();
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> compressor;
  compressor.init(hst_sources, eta, threshold);
  compressor.compress(evaluator);
  const auto &triplets = compressor.triplets();
  // compression error
  FMCA::Vector x(N_rows), y1(N_rows), y2(N_rows);
  FMCA::Scalar err = 0;
  FMCA::Scalar nrm = 0;
  for (auto i = 0; i < 100; ++i) {
    FMCA::Index index = rand() % N_rows;
    x.setZero();
    x(index) = 1;
    FMCA::Vector col =
        kernel.eval(P_rows, P_rows.col(hst_sources.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst_sources.indices(),
                                             hst_sources.block_size()));
    x = hst_sources.sampletTransform(x);
    y2.setZero();
    for (const auto &i : triplets) {
      y2(i.row()) += i.value() * x(i.col());
    }
    y2 = hst_sources.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  std::cout << "compression error =          " << sqrt(err / nrm) << std::endl;;
  int anz = triplets.size() / N_rows;
  std::cout << "Anz:                             " << anz << std::endl;
  Eigen::SparseMatrix<FMCA::Scalar> sparseMatrix(N_rows, N_rows);
  sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
  sparseMatrix.makeCompressed();

  return sparseMatrix;
}

template <typename KernelType, typename EvaluatorType>
Eigen::SparseMatrix<FMCA::Scalar> createCompressedSparseMatrixUnSymmetric(
    KernelType &kernel,
    const EvaluatorType &evaluator, const H2SampletTree &hst_sources,
    const H2SampletTree &hst_quad, FMCA::Scalar eta, FMCA::Scalar threshold,
    FMCA::Matrix P_rows, FMCA::Matrix P_cols) {

  int N_rows = P_rows.cols();
  int N_cols = P_cols.cols();
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> compressor;
  compressor.init(hst_sources, hst_quad, eta, threshold);
  compressor.compress(evaluator);
  const auto &triplets = compressor.triplets();
  // compression error
  FMCA::Vector x(N_cols), y1(N_rows), y2(N_rows);
  FMCA::Scalar err = 0;
  FMCA::Scalar nrm = 0;
  for (auto i = 0; i < 100; ++i) {
    FMCA::Index index = rand() % N_cols;
    x.setZero();
    x(index) = 1;
    FMCA::Vector col =
        kernel.eval(P_rows, P_cols.col(hst_quad.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst_sources.indices(),
                                             hst_sources.block_size()));
    x = hst_quad.sampletTransform(x);
    y2.setZero();
    for (const auto &i : triplets) {
      y2(i.row()) += i.value() * x(i.col());
    }
    y2 = hst_sources.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  std::cout << "compression error =          " << sqrt(err / nrm) << std::endl;;
  int anz = triplets.size() / N_rows;
  std::cout << "Anz:                                  " << anz << std::endl;
  Eigen::SparseMatrix<FMCA::Scalar> sparseMatrix(N_rows, N_cols);
  sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
  sparseMatrix.makeCompressed();

  return sparseMatrix;
}


// the function createCompressedWeights is based on SparseMatrix Class 
Eigen::SparseMatrix<FMCA::Scalar> createCompressedWeights(
    const FMCA::SparseMatrixEvaluator &evaluator, const H2SampletTree &hst,
    FMCA::Scalar eta, FMCA::Scalar threshold, FMCA::Scalar N_rows) {
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> compressor;
  compressor.init(hst, eta, threshold);
  compressor.compress(evaluator);
  const auto &triplets = compressor.triplets();
  int anz = triplets.size() / N_rows;
  std::cout << "Anz:                         " << anz << std::endl;
  Eigen::SparseMatrix<FMCA::Scalar> sparseMatrix(N_rows, N_rows);
  sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
  sparseMatrix.makeCompressed();

  return sparseMatrix;
}

std::vector<Eigen::Triplet<FMCA::Scalar>> TripletsPatternSymmetric(
    const FMCA::Scalar &N, const H2SampletTree &hst_sources, FMCA::Scalar eta,
    FMCA::Scalar threshold) {
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> compressor;
  compressor.init(hst_sources, eta, threshold);
  std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets =
      compressor.a_priori_pattern_triplets();
  return a_priori_triplets;
}

typedef Eigen::Triplet<FMCA::Scalar> Triplet;
void applyPattern(
    Eigen::SparseMatrix<FMCA::Scalar> &M,
    const std::vector<Eigen::Triplet<FMCA::Scalar>> &patternTriplets) {
  // Create a mask eith ones in the pattern positions
  Eigen::SparseMatrix<FMCA::Scalar> mask(M.rows(), M.cols());
  std::vector<Eigen::Triplet<FMCA::Scalar>> maskTriplets;
  for (const auto &triplet : patternTriplets) {
    maskTriplets.emplace_back(triplet.row(), triplet.col(), 1.0);
  }
  mask.setFromTriplets(maskTriplets.begin(), maskTriplets.end());
  M = M.cwiseProduct(mask);
}

typedef Eigen::Triplet<FMCA::Scalar> Triplet;
void applyPatternAndFilter(
    Eigen::SparseMatrix<FMCA::Scalar> &M,
    const std::vector<Eigen::Triplet<FMCA::Scalar>> &patternTriplets,
    const FMCA::Scalar threshold) {
  Eigen::SparseMatrix<FMCA::Scalar> mask(M.rows(), M.cols());
  std::vector<Eigen::Triplet<FMCA::Scalar>> maskTriplets;
  for (const auto &triplet : patternTriplets) {
    maskTriplets.emplace_back(triplet.row(), triplet.col(), 1.0);
  }
  mask.setFromTriplets(maskTriplets.begin(), maskTriplets.end());
  M = M.cwiseProduct(mask);
  // Filter M based on the threshold
  std::vector<Triplet> filteredTriplets;
  for (FMCA::Index k = 0; k < M.outerSize(); ++k) {
    for (Eigen::SparseMatrix<FMCA::Scalar>::InnerIterator it(M, k); it; ++it) {
      if (it.row() == it.col()) {
        filteredTriplets.emplace_back(it.row(), it.col(), it.value());
      } else {
        if (std::abs(it.value()) >= threshold) {
          filteredTriplets.emplace_back(it.row(), it.col(), it.value());
        }
      }
    }
  }
  M.setZero();
  M.setFromTriplets(filteredTriplets.begin(), filteredTriplets.end());
}


Eigen::SparseMatrix<FMCA::Scalar> extractDiagonal(
    const Eigen::SparseMatrix<FMCA::Scalar> &S) {
  Eigen::SparseMatrix<FMCA::Scalar> S_diagonal(S.rows(), S.cols());
  for (FMCA::Index k = 0; k < S.outerSize(); ++k) {
    for (Eigen::SparseMatrix<FMCA::Scalar>::InnerIterator it(S, k); it; ++it) {
      if (it.row() == it.col()) {
        S_diagonal.insert(it.row(), it.col()) = it.value();
      }
    }
  }
  return S_diagonal;
}

Eigen::SparseMatrix<FMCA::Scalar> NeumannNormalMultiplication(
    const Eigen::SparseMatrix<FMCA::Scalar> &grad, const FMCA::Vector &normal) {
  Eigen::SparseMatrix<FMCA::Scalar> diagNormal(grad.cols(), grad.cols());
  std::vector<Eigen::Triplet<FMCA::Scalar>> triplets;
  for (FMCA::Index i = 0; i < normal.size(); ++i) {
    triplets.emplace_back(i, i, normal(i));
  }
  diagNormal.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::SparseMatrix<FMCA::Scalar> result = grad * diagNormal;
  return result;
}

