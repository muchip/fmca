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

namespace FMCA {

//////////////////////////////////////////////////////////////////////////////
struct CompressionStats {
  Scalar time_compressor = 0.0;    // Total compression time
  Scalar assembly_time = 0.0;      // Total assembly time
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
#ifdef CHOLMOD_SUPPORT
  using Cholesky = Eigen::CholmodSupernodalLLT<SparseMatrix, Eigen::Upper>;
#elif METIS_SUPPORT
  using Cholesky = Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper,
                                         Eigen::MetisOrdering<int>>;
#else
  using Cholesky = Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>;
#endif

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
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  void compress(const Matrix& P) {
    Tictoc timer;
    const Moments mom(P, mpole_deg_);
    // init
    compressor_.init(hst_, eta_, FMCA_ZERO_TOLERANCE);
    // compress
    const MatrixEvaluator mat_eval(mom, kernel_);
    timer.tic();
    compressor_.compress(mat_eval);
    compression_stats_.time_compressor = timer.toc();
    // a priori compression
    compressor_.triplets();
    // a posteriori compression
    timer.tic();
    const auto& trips = compressor_.aposteriori_triplets_fast(threshold_);
    compression_stats_.assembly_time = timer.toc();
    compression_stats_.anz = std::round(trips.size() / double(P.cols()));
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
    compression_stats_.compression_error = sqrt(err / nrm);
  }

#if defined(CHOLMOD_SUPPORT) || defined(METIS_SUPPORT)
  //////////////////////////////////////////////////////////////////////////////
  void factorize() { llt_.compute(K_); }

  //////////////////////////////////////////////////////////////////////////////
  Matrix solveCholesky(const Matrix& rhs) {
    Tictoc timer;
    Matrix sol = hst_.toClusterOrder(rhs);
    sol = hst_.sampletTransform(sol);
    timer.tic();
    sol = llt_.solve(sol);
    solver_stats_.solver_time = timer.toc();
    sol = hst_.inverseSampletTransform(sol);
    sol = hst_.toNaturalOrder(sol);
    solver_stats_.iterations = 1;        // Direct solver
    solver_stats_.residual_error = 0.0;  // Direct solver
    return sol;
  }
#endif

  //////////////////////////////////////////////////////////////////////////////
  Vector solveIterative(const Vector& rhs, bool CGwithPreconditioner = true,
                        Scalar threshold_CG = 1e-6) {
    Tictoc timer;
    Vector rhs_copy = rhs;
    rhs_copy = hst_.toClusterOrder(rhs_copy);
    rhs_copy = hst_.sampletTransform(rhs_copy);
    Vector sol;
    SparseMatrix K_sym = K_.template selfadjointView<Eigen::Upper>();

    if (!CGwithPreconditioner) {
      Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper,
                               Eigen::IdentityPreconditioner>
          solver;
      solver.setTolerance(threshold_CG);
      timer.tic();
      solver.compute(K_sym);
      sol = solver.solve(rhs_copy);
      solver_stats_.solver_time = timer.toc();
      solver_stats_.iterations = solver.iterations();
      solver_stats_.residual_error = (K_sym * sol - rhs_copy).norm();
    } else {
      Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper>
          solver;
      solver.setTolerance(threshold_CG);
      timer.tic();
      solver.compute(K_sym);
      sol = solver.solve(rhs_copy);
      solver_stats_.solver_time = timer.toc();
      solver_stats_.iterations = solver.iterations();
      solver_stats_.residual_error = (K_sym * sol - rhs_copy).norm();
    }
    sol = hst_.inverseSampletTransform(sol);
    sol = hst_.toNaturalOrder(sol);
    return sol;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  const SparseMatrix& K() const { return K_; }
  const SampletTree& getSampletTree() const { return hst_; }
  const Scalar fill_distance() const { return fill_distance_; }
  const Scalar separation_radius() const { return separation_radius_; }
  const CompressionStats& getCompressionStats() const {
    return compression_stats_;
  }
  const SolverStats& getSolverStats() const { return solver_stats_; }

 private:
  internal::SampletMatrixCompressor<SampletTree> compressor_;
  SampletTree hst_;
  CovarianceKernel kernel_;
#if defined(CHOLMOD_SUPPORT) || defined(METIS_SUPPORT)
  Cholesky llt_;
#endif
  SparseMatrix K_;
  Scalar dtilde_;
  Scalar mpole_deg_;
  Scalar eta_;
  Scalar nu_;
  Scalar threshold_;
  Scalar ridgep_;
  Scalar fill_distance_;
  Scalar separation_radius_;
  CompressionStats compression_stats_;
  SolverStats solver_stats_;

 public:
  void updateKernel(const CovarianceKernel& kernel) { kernel_ = kernel; }
};
}  // namespace FMCA

#endif