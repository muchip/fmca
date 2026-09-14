// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_UTIL_MACROS_H_
#define FMCA_UTIL_MACROS_H_
 
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <vector>
//
#include <Eigen/Dense>
#include <Eigen/Sparse>
//
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/QR>
#include <Eigen/SVD>
//
#ifdef CHOLMOD_SUPPORT
#include <Eigen/CholmodSupport>
#endif
 
#ifdef METIS_SUPPORT
#include <Eigen/MetisSupport>
#endif
//
namespace FMCA {
#ifndef M_PI
#define FMCA_PI 3.14159265358979323846264338327950288
#else
#define FMCA_PI M_PI
#endif
 
// define primitive types used throughout the toolbox
#define FMCA_INDEX unsigned int
#define FMCA_SCALAR double
#define FMCA_INF std::numeric_limits<FMCA_SCALAR>::infinity()
#define FMCA_ZERO_TOLERANCE std::numeric_limits<FMCA_SCALAR>::epsilon()
#define FMCA_BBOX_THREASHOLD FMCA_ZERO_TOLERANCE
#define FMCA_MAXINDEX UINT_MAX
 
#define FMCA_UNSAFE 0
 
typedef FMCA_INDEX Index;
 
typedef FMCA_SCALAR Scalar;
 
// matrix types
template <typename Derived>
using MatrixBase = Eigen::MatrixBase<Derived>;
 
template <typename Derived>
using Map = Eigen::Map<Derived>;
 
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
 
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1u>;
 
using iMatrix = Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic>;
 
using iVector = Eigen::Matrix<Index, Eigen::Dynamic, 1u>;
 
using Triplet = Eigen::Triplet<Scalar>;
 
using SparseMatrix = Eigen::SparseMatrix<Scalar>;
 
constexpr auto Upper = Eigen::Upper;
 
// matrix algorithms
using Cholesky = Eigen::LLT<Matrix>;
using HouseholderQR = Eigen::HouseholderQR<Matrix>;
using ColPivHouseholderQR = Eigen::ColPivHouseholderQR<Matrix>;
 
using FullPivHouseholderQR = Eigen::FullPivHouseholderQR<Matrix>;
 
constexpr auto ComputeThinUV = Eigen::ComputeThinU | Eigen::ComputeThinV;
constexpr auto ComputeFullUV = Eigen::ComputeFullU | Eigen::ComputeFullV;
 
#if EIGEN_VERSION_AT_LEAST(3, 4, 90)
// Eigen >= 3.4.90 (development branch): computation options are template
// parameters
using JacobiSVD = Eigen::JacobiSVD<Matrix, ComputeThinUV>;
using JacobiFullSVD = Eigen::JacobiSVD<Matrix, ComputeFullUV>;
using BDCSVD = Eigen::BDCSVD<Matrix, ComputeThinUV>;
using BDCFullSVD = Eigen::BDCSVD<Matrix, ComputeFullUV>;
#else
// Eigen <= 3.4.0: computation options are runtime arguments; bake them into
// thin wrappers so that FMCA::JacobiSVD svd(A) has the same meaning
namespace internal {
template <typename Base, unsigned int Options>
struct SVDWithOptions : public Base {
  SVDWithOptions() = default;
  SVDWithOptions(Eigen::Index rows, Eigen::Index cols)
      : Base(rows, cols, Options) {}
  explicit SVDWithOptions(const Matrix &m) : Base(m, Options) {}
  SVDWithOptions &compute(const Matrix &m) {
    Base::compute(m, Options);
    return *this;
  }
  using Base::compute;
};
}  // namespace internal
using JacobiSVD =
    internal::SVDWithOptions<Eigen::JacobiSVD<Matrix>, ComputeThinUV>;
using JacobiFullSVD =
    internal::SVDWithOptions<Eigen::JacobiSVD<Matrix>, ComputeFullUV>;
using BDCSVD = internal::SVDWithOptions<Eigen::BDCSVD<Matrix>, ComputeThinUV>;
using BDCFullSVD =
    internal::SVDWithOptions<Eigen::BDCSVD<Matrix>, ComputeFullUV>;
#endif
 
using SelfAdjointEigenSolver = Eigen::SelfAdjointEigenSolver<Matrix>;
 
constexpr auto Success = Eigen::Success;
constexpr auto ComputeEigenvectors = Eigen::ComputeEigenvectors;
 
#ifdef CHOLMOD_SUPPORT
using SparseCholesky = Eigen::CholmodSupernodalLLT<SparseMatrix, Eigen::Upper>;
#elif METIS_SUPPORT
using SparseCholesky = Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper,
                                             Eigen::MetisOrdering<int> >;
#else
using SparseCholesky = Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>;
#endif
 
using SparseCG = Eigen::ConjugateGradient<SparseMatrix, Eigen::Upper,
                                          Eigen::IdentityPreconditioner>;
 
using SparsePCG = Eigen::ConjugateGradient<SparseMatrix, Eigen::Upper>;
 
}  // namespace FMCA
 
#endif
