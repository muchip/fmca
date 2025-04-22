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

#ifndef FMCA_KERNELINTERPOLATION_MULTILEVELSAMPLETKERNELSOLVER_H_
#define FMCA_KERNELINTERPOLATION_MULTILEVELSAMPLETKERNELSOLVER_H_

#include <random>
#include <stdexcept>
#include <string>

#include "../util/Tictoc.h"

namespace FMCA {
//////////////////////////////////////////////////////////////////////////////
struct CompressionStats {
  Scalar time_planner = 0.0;       // Time for compression planning
  Scalar time_compressor = 0.0;    // Time for actual compression
  Scalar assembly_time = 0.0;      // Time for a posteriori compression
  size_t anz = 0;                  // Number of a posteriori triplets per row
  Scalar compression_error = 0.0;  // Relative compression error
};

//////////////////////////////////////////////////////////////////////////////
struct SolverStats {
  Scalar solver_time = 0.0;     // Total solver execution time
  int iterations = 0;           // Number of iterations used
  Scalar residual_error = 0.0;  // Final residual error
};

//////////////////////////////////////////////////////////////////////////////
template <typename SparseMatrix = Eigen::SparseMatrix<FMCA::Scalar>>
class MultilevelSampletKernelSolver {
 public:
  using Interpolator = TotalDegreeInterpolator;
  using SampletInterpolator = MonomialInterpolator;
  using Moments = NystromMoments<Interpolator>;
  using SampletMoments = NystromSampletMoments<SampletInterpolator>;
  using MatrixEvaluator = NystromEvaluator<Moments, FMCA::CovarianceKernel>;
  using SampletTree = H2SampletTree<ClusterTree>;

  MultilevelSampletKernelSolver() noexcept {}

  MultilevelSampletKernelSolver(const MultilevelSampletKernelSolver& other) =
      delete;

  MultilevelSampletKernelSolver(
      MultilevelSampletKernelSolver&& other) noexcept {
    // this is a dummy move constructor needed to be able to wrap the class into
    // std::vector
  }

  MultilevelSampletKernelSolver(const CovarianceKernel& kernel, const Matrix& P,
                                Index dtilde, Scalar eta = 0.,
                                Scalar threshold = 0.,
                                Scalar ridgep = 0.) noexcept {
    init(kernel, P, dtilde, eta, threshold, ridgep);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  Scalar computeFillDistance(const Matrix& P) const {
    // Create a temporary tree just for computing the fill distance
    const Moments tmp_mom(P, mpole_deg_);
    const SampletMoments tmp_smom(P, mpole_deg_ - 1);
    SampletTree tmp_tree;
    tmp_tree.init(tmp_mom, tmp_smom, 0, P);
    const Vector minvec = minDistanceVector(tmp_tree, P);
    return minvec.maxCoeff();
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
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  void compress(const Matrix& P) {
    Tictoc timer;
    const Moments mom(P, mpole_deg_);
    // init
    timer.tic();
    compressor_.init(hst_, eta_, FMCA_ZERO_TOLERANCE);
    compressor_stats_.time_planner = timer.toc();
    // compress
    const MatrixEvaluator mat_eval(mom, kernel_);
    timer.tic();
    compressor_.compress(mat_eval);
    compressor_stats_.time_compressor = timer.toc();
    // a priori compression
    compressor_.triplets();
    // a posteriori compression
    timer.tic();
    const auto& trips = compressor_.aposteriori_triplets_fast(threshold_);
    compressor_stats_.assembly_time = timer.toc();
    compressor_stats_.anz = std::round(trips.size() / double(P.cols()));
    // final compressed matrix
    K_.resize(hst_.block_size(), hst_.block_size());
    K_.setFromTriplets(trips.begin(), trips.end());
    // regularization
    if (ridgep_ > 0) {
      for (int i = 0; i < K_.rows(); ++i) {
        K_.coeffRef(i, i) += ridgep_;
      }
    }
    K_.makeCompressed();
    return;
  }

  ////////////////////////////////////////////////////////////////////////////////
  void compressionError(const Matrix& P) {
    Vector x(K_.cols()), y1(K_.rows()), y2(K_.rows());
    Scalar err = 0;
    Scalar nrm = 0;
    // check by selecting 10 random columns
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
    compressor_stats_.compression_error = sqrt(err / nrm);
  }

  //////////////////////////////////////////////////////////////////////////////
  void setCompressedKernel(const SparseMatrix& K) { K_ = K; }

  //////////////////////////////////////////////////////////////////////////////
  Vector solveIterative(const Vector& rhs,
                        const std::string& solverName = "ConjugateGradient",
                        Scalar threshold_CG = 1e-6) {
    Tictoc timer;
    Vector rhs_copy = rhs;
    rhs_copy = hst_.toClusterOrder(rhs_copy);
    rhs_copy = hst_.sampletTransform(rhs_copy);
    Vector sol;
    SparseMatrix K_sym = K_.template selfadjointView<Eigen::Upper>();

    if (solverName == "ConjugateGradient") {
      timer.tic();
      Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper,
                               Eigen::IdentityPreconditioner>
          solver;
      solver.setTolerance(threshold_CG);
      solver.compute(K_sym);
      if (solver.info() != Eigen::Success)
        throw std::runtime_error("CG: Matrix decomposition failed");
      sol = solver.solve(rhs_copy);
      CG_stats_.iterations = solver.iterations();
      CG_stats_.solver_time = timer.toc();
      CG_stats_.residual_error = (K_sym * sol - rhs_copy).norm();
    } else if (solverName == "ConjugateGradientwithPreconditioner") {
      timer.tic();
      Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper>
          solver;
      solver.setTolerance(threshold_CG);
      solver.compute(K_sym);
      if (solver.info() != Eigen::Success)
        throw std::runtime_error(
            "CG with Preconditioner: Decomposition failed");
      sol = solver.solve(rhs_copy);
      CG_stats_.iterations = solver.iterations();
      CG_stats_.solver_time = timer.toc();
      CG_stats_.residual_error = (K_sym * sol - rhs_copy).norm();
    } else {
      throw std::invalid_argument(
          "Invalid solver name. Options are: "
          "'ConjugateGradient' or "
          "'ConjugateGradientwithPreconditioner'");
    }

    sol = hst_.inverseSampletTransform(sol);
    sol = hst_.toNaturalOrder(sol);
    return sol;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Getter
  const SparseMatrix& K() const { return K_; }
  const SampletTree& sampletTree() const { return hst_; }
  const CompressionStats& getCompressorStats() const {
    return compressor_stats_;
  }
  const SolverStats& getCGStats() const { return CG_stats_; }

  //////////////////////////////////////////////////////////////////////////////
 private:
  CompressionStats compressor_stats_;
  SolverStats CG_stats_;
  internal::SampletMatrixCompressor<SampletTree> compressor_;
  SampletTree hst_;
  CovarianceKernel kernel_;
  SparseMatrix K_;
  Scalar dtilde_;
  Scalar mpole_deg_;
  Scalar eta_;
  Scalar threshold_;
  Scalar ridgep_;
};

}  // namespace FMCA

#endif  // FMCA_KERNELINTERPOLATION_SAMPLETKERNELSOLVER_H_