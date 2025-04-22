// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//

#ifndef FMCA_KERNELINTERPOLATION_SAMPLETKERNELSOLVER_H_
#define FMCA_KERNELINTERPOLATION_SAMPLETKERNELSOLVER_H_

namespace FMCA {
template <typename SparseMatrix = Eigen::SparseMatrix<FMCA::Scalar>>
class SampletKernelSolver {
 public:
  using Interpolator = TotalDegreeInterpolator;
  using SampletInterpolator = MonomialInterpolator;
  using Moments = NystromMoments<Interpolator>;
  using SampletMoments = NystromSampletMoments<SampletInterpolator>;
  using MatrixEvaluator = NystromEvaluator<Moments, FMCA::CovarianceKernel>;
  using SampletTree = H2SampletTree<ClusterTree>;
#ifdef CHOLMOD_SUPPORT
  using Cholesky = Eigen::CholmodSupernodalLLT<SparseMatrix, Eigen::Upper>;
#elif METIS_SUPPORT
  using Cholesky = Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper,
                                         Eigen::MetisOrdering<int>>;
#else
  using Cholesky = Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>;
#endif

  SampletKernelSolver() noexcept {}

  SampletKernelSolver(const SampletKernelSolver& other) = delete;

  SampletKernelSolver(SampletKernelSolver&& other) noexcept {
    // this is a dummy move constructor needed to be able to wrap the class into
    // std::vector
  }

  SampletKernelSolver(const CovarianceKernel& kernel, const Matrix& P,
                      Index dtilde, Scalar eta = 0., Scalar threshold = 0.,
                      Scalar ridgep = 0.) noexcept {
    init(kernel, P, dtilde, eta, threshold, ridgep);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void init(const CovarianceKernel& kernel, const Matrix& P, Index dtilde,
            Scalar eta = 0., Scalar threshold = 0., Scalar ridgep = 0.) {
    // set parameters
    kernel_ = kernel;
    dtilde_ = dtilde > 0 ? dtilde : 1;
    mpole_deg_ = dtilde_ > 1 ? (2 * (dtilde_ - 1)) : 1;
    eta_ = eta >= 0 ? eta : 0;
    threshold_ = threshold >= 0 ? threshold : 0;
    ridgep_ = ridgep >= 0 ? ridgep : 0;
    // init moments and samplet tree
    const Moments mom(P, mpole_deg_);
    const SampletMoments smom(P, dtilde_ - 1);
    hst_.init(mom, smom, 0, P);
    const Vector minvec = minDistanceVector(hst_, P);
    fill_distance_ = minvec.maxCoeff();
    separation_radius_ = minvec.minCoeff();
    // compress kernel

    return;
  }

  void compress(const Matrix& P) {
    const Moments mom(P, mpole_deg_);
    compressor_.init(hst_, eta_, FMCA_ZERO_TOLERANCE);
    const MatrixEvaluator mat_eval(mom, kernel_);
    compressor_.compress(mat_eval);
    compressor_.triplets();
    const auto& trips = compressor_.aposteriori_triplets_fast(threshold_);
    K_.resize(hst_.block_size(), hst_.block_size());
    K_.setFromTriplets(trips.begin(), trips.end());
    if (ridgep_ > 0) K_.diagonal() = K_.diagonal().array() + ridgep_;
    K_.makeCompressed();
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  Scalar compressionError(const Matrix& P) const {
    Vector x(K_.cols()), y1(K_.rows()), y2(K_.rows());
    Scalar err = 0;
    Scalar nrm = 0;
    for (auto i = 0; i < 10; ++i) {
      Index index = rand() % K_.cols();
      x.setZero();
      x(index) = 1;
      Vector col = kernel_.eval(P, P.col(hst_.indices()[index]));
      y1 = hst_.toClusterOrder(col);
      x = hst_.sampletTransform(x);
      y2 = K_.template selfadjointView<Eigen::Upper>() * x;
      y2 = hst_.inverseSampletTransform(y2);
      err += (y1 - y2).squaredNorm();
      nrm += y1.squaredNorm();
    }
    return sqrt(err / nrm);
  }

  //////////////////////////////////////////////////////////////////////////////
  void factorize() { llt_.compute(K_); }

  //////////////////////////////////////////////////////////////////////////////
  // getter
  const SparseMatrix& K() const { return K_; }
  const Scalar fill_distance() const { return fill_distance_; }
  const Scalar separation_radius() const { return separation_radius_; }
  //////////////////////////////////////////////////////////////////////////////
  Matrix solve(const Matrix& rhs) {
    Matrix sol = hst_.toClusterOrder(rhs);
    sol = hst_.sampletTransform(sol);
    sol = llt_.solve(sol);
    sol = hst_.inverseSampletTransform(sol);
    sol = hst_.toNaturalOrder(sol);
    return sol;
  }

 private:
  internal::SampletMatrixCompressor<SampletTree> compressor_;
  SampletTree hst_;
  CovarianceKernel kernel_;
  Cholesky llt_;
  SparseMatrix K_;
  Scalar dtilde_;
  Scalar mpole_deg_;
  Scalar eta_;
  Scalar threshold_;
  Scalar ridgep_;
  Scalar fill_distance_;
  Scalar separation_radius_;
};
}  // namespace FMCA

#endif