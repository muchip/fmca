#include <cstdlib>
#include <iostream>
//##############################
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"

// typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> Sparse;
// typedef Eigen::SparseMatrix<double> Sparse;


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
                          
using namespace std;

double computeTrace(const Eigen::SparseMatrix<double> &mat) {
  double trace = 0.0;
  // Iterate only over the diagonal elements
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
      if (it.row() == it.col()) {  // Check if it is a diagonal element
        trace += it.value();
      }
    }
  }
  return trace;
}

int countSmallerThan(const Eigen::SparseMatrix<double> &matrix,
                     FMCA::Scalar threshold) {
  int count = 0;
  // Iterate over all non-zero elements.
  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it;
         ++it) {
      if (abs(it.value()) < threshold && abs(it.value()) !=0) {
        count++;
      }
    }
  }
  return count / matrix.rows();
}

bool isSymmetric(const Eigen::SparseMatrix<double> &matrix,
                 FMCA::Scalar tol = 1e-10) {
  if (matrix.rows() != matrix.cols())
    return false;  // Non-square matrices are not symmetric

  // Iterate over the outer dimension
  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it;
         ++it) {
      if (std::fabs(it.value() - matrix.coeff(it.col(), it.row())) > tol)
        return false;
    }
  }
  return true;
}

template<typename KernelType, typename EvaluatorType>
Eigen::SparseMatrix<double>  createCompressedSparseMatrixSymmetric(
    const KernelType &kernel,
    const EvaluatorType &evaluator,
    const H2SampletTree &hst_sources,
    FMCA::Scalar eta,
    FMCA::Scalar threshold,
    FMCA::Scalar N_rows
) {
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> compressor;
    compressor.init(hst_sources, eta, threshold);
    compressor.compress(evaluator);
    const auto &triplets = compressor.triplets();
    int anz = triplets.size() / N_rows;
    std::cout << "Anz:                         " << anz << std::endl;

    Eigen::SparseMatrix<double> sparseMatrix(N_rows, N_rows);
    sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
    sparseMatrix.makeCompressed();

    return sparseMatrix;
}

template<typename KernelType, typename EvaluatorType>
Eigen::SparseMatrix<double>  createCompressedSparseMatrixUnSymmetric(
    const KernelType &kernel,
    const EvaluatorType &evaluator,
    const H2SampletTree &hst_sources,
    const H2SampletTree &hst_quad, // Add this if you need to handle unsymmetric cases
    FMCA::Scalar eta,
    FMCA::Scalar threshold,
    FMCA::Scalar N_rows,
    FMCA::Scalar N_cols
) {
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> compressor;
    compressor.init(hst_sources, hst_quad, eta, threshold);
    compressor.compress(evaluator);
    const auto &triplets = compressor.triplets();
    int anz = triplets.size() / N_rows;
    std::cout << "Anz:                         " << anz << std::endl;

    Eigen::SparseMatrix<double> sparseMatrix(N_rows, N_cols);
    sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
    sparseMatrix.makeCompressed();

    return sparseMatrix;
}

Eigen::SparseMatrix<double>  createCompressedWeights(
    const FMCA::SparseMatrixEvaluator &evaluator,
    const H2SampletTree &hst,
    FMCA::Scalar eta,
    FMCA::Scalar threshold,
    FMCA::Scalar N_rows
) {
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> compressor;
    compressor.init(hst, eta, threshold);
    compressor.compress(evaluator);
    const auto &triplets = compressor.triplets();
    int anz = triplets.size() / N_rows;
    std::cout << "Anz:                         " << anz << std::endl;

    Eigen::SparseMatrix<double> sparseMatrix(N_rows, N_rows);
    sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
    sparseMatrix.makeCompressed();

    return sparseMatrix;
}


std::vector<Eigen::Triplet<FMCA::Scalar>> TripletsPatternSymmetric (
    const FMCA::Scalar &N, 
    const H2SampletTree &hst_sources,
    FMCA::Scalar eta, 
    FMCA::Scalar threshold
){
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> compressor;
    compressor.init(hst_sources, eta, threshold);
    std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets = compressor.a_priori_pattern_triplets();
    // Sparse pattern(N,N);
    // pattern.setFromTriplets(a_priori_triplets.begin(), a_priori_triplets.end()); 
    // pattern.makeCompressed();
    return a_priori_triplets;
}



typedef Eigen::Triplet<double> Triplet;
void applyPattern(
    Eigen::SparseMatrix<double>  &M, 
    const std::vector<Eigen::Triplet<FMCA::Scalar>> &patternTriplets
) {
    // Create a sparse matrix from triplets, same size as M, with 1s in the pattern positions
    Eigen::SparseMatrix<double>  mask(M.rows(), M.cols());
    std::vector<Eigen::Triplet<FMCA::Scalar>> maskTriplets;
    for (const auto& triplet : patternTriplets) {
        maskTriplets.emplace_back(triplet.row(), triplet.col(), 1.0);
    }
    mask.setFromTriplets(maskTriplets.begin(), maskTriplets.end());
    M = M.cwiseProduct(mask);
}
