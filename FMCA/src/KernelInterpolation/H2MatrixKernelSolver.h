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

#ifndef FMCA_KERNELINTERPOLATION_H2MATRIXKERNELSOLVER_H_
#define FMCA_KERNELINTERPOLATION_H2MATRIXKERNELSOLVER_H_

template <typename H2Matrix>
class EigenWrapper;

template <typename H2Matrix, typename Rhs>
class EigenWrapper_ProductReturnType;

namespace Eigen {
namespace internal {

// Traits for the wrapper, inheriting from SparseMatrix traits for compatibility
template <typename H2Matrix>
struct traits<EigenWrapper<H2Matrix>>
    : Eigen::internal::traits<Eigen::SparseMatrix<double>> {};

// Traits for the product return type, templated on H2Matrix and Rhs
template <typename H2Matrix, typename Rhs>
struct traits<EigenWrapper_ProductReturnType<H2Matrix, Rhs>> {
  typedef Eigen::Matrix<typename Rhs::Scalar, Eigen::Dynamic,
                        Rhs::ColsAtCompileTime>
      ReturnType;
};

}  // namespace internal
}  // namespace Eigen

template <typename H2Matrix>
class EigenWrapper : public Eigen::EigenBase<EigenWrapper<H2Matrix>> {
 public:
  typedef double Scalar;
  typedef double RealScalar;
  typedef typename Eigen::Index Index;
  typedef int StorageIndex;

  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    RowsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    MaxRowsAtCompileTime = Eigen::Dynamic,
  };
  static constexpr bool IsRowMajor = false;

  explicit EigenWrapper(const H2Matrix& h2mat, const FMCA::Scalar ridge = 0.)
      : h2mat_(h2mat), ridge_(ridge >= 0. ? ridge : 0.) {}

  const H2Matrix& matrix() const { return h2mat_; }
  const FMCA::Scalar& ridgep() const { return ridge_; }
  Index rows() const { return h2mat_.rows(); }
  Index cols() const { return h2mat_.cols(); }

  template <typename Rhs>
  EigenWrapper_ProductReturnType<H2Matrix, Rhs> operator*(
      const Eigen::MatrixBase<Rhs>& x) const {
    return EigenWrapper_ProductReturnType<H2Matrix, Rhs>(*this, x.derived());
  }

 private:
  const H2Matrix& h2mat_;
  const FMCA::Scalar ridge_;
};

template <typename H2Matrix, typename Rhs>
class EigenWrapper_ProductReturnType
    : public Eigen::ReturnByValue<
          EigenWrapper_ProductReturnType<H2Matrix, Rhs>> {
 public:
  typedef typename EigenWrapper<H2Matrix>::Index Index;

  EigenWrapper_ProductReturnType(const EigenWrapper<H2Matrix>& matrix,
                                 const Rhs& rhs)
      : m_matrix(matrix), m_rhs(rhs) {}

  Index rows() const { return m_matrix.rows(); }
  Index cols() const { return m_rhs.cols(); }

  template <typename Dest>
  void evalTo(Dest& y) const {
    y = m_matrix.matrix() * m_rhs.eval();
    if (m_matrix.ridgep() > 0) y += m_matrix.ridgep() * m_rhs;
  }

 protected:
  const EigenWrapper<H2Matrix>& m_matrix;
  typename Rhs::Nested m_rhs;
};

namespace FMCA {
class H2MatrixKernelSolver {
 public:
  using Interpolator = TotalDegreeInterpolator;
  using Moments = NystromMoments<Interpolator>;
  using MatrixEvaluator = NystromEvaluator<Moments, CovarianceKernel>;
  using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
  using H2Matrix = FMCA::H2Matrix<H2ClusterTree, CompareCluster>;
  using CG = Eigen::ConjugateGradient<EigenWrapper<H2Matrix>,
                                      Eigen::Lower | Eigen::Upper,
                                      Eigen::IdentityPreconditioner>;
  using PreconditionedCG = Eigen::ConjugateGradient<EigenWrapper<H2Matrix>,
                                      Eigen::Lower | Eigen::Upper,
                                      Eigen::IdentityPreconditioner>;

  H2MatrixKernelSolver() noexcept {}

  H2MatrixKernelSolver(const H2MatrixKernelSolver& other) = delete;

  H2MatrixKernelSolver(H2MatrixKernelSolver&& other) noexcept {
    // this is a dummy move constructor needed to be able to wrap the class into
    // std::vector
  }

  H2MatrixKernelSolver(const Matrix& P,
                       Index dtilde, Scalar eta = 0., Scalar threshold = 0.,
                       Scalar ridgep = 0.) noexcept {
    init(P, dtilde, eta, threshold, ridgep);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void init(const Matrix& P, Index mpole_deg,
            Scalar eta = 0., Scalar threshold = 0., Scalar ridgep = 0.) {
    // set parameters
    mpole_deg_ = mpole_deg;
    eta_ = eta >= 0 ? eta : 0;
    threshold_ = threshold >= 0 ? threshold : 0;
    ridgep_ = ridgep >= 0 ? ridgep : 0;
    // init moments and samplet tree
    const Moments mom(P, mpole_deg_);
    hct_.init(mom, 0, P);
    const Vector minvec = minDistanceVector(hct_, P);
    fill_distance_ = minvec.maxCoeff();
    separation_radius_ = minvec.minCoeff();
    solver_iterations_ = 1;
    return;
  }

  void compress(const Matrix& P, const CovarianceKernel& kernel) {
    kernel_ = kernel;
    const Moments mom(P, mpole_deg_);
    const MatrixEvaluator mat_eval(mom, kernel_);
    K_.computeH2Matrix(hct_, hct_, mat_eval, eta_);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  Scalar compressionError(const Matrix& P, 
                          const CovarianceKernel& kernel) {
    Vector x(K_.cols()), y1(K_.rows()), y2(K_.rows());
    Scalar err = 0;
    Scalar nrm = 0;
    for (auto i = 0; i < 10; ++i) {
      Index index = rand() % K_.cols();
      x.setZero();
      x(index) = 1;
      Vector col = kernel_.eval(P, P.col(hct_.indices()[index]));
      y1 = hct_.toClusterOrder(col);
      y2 = K_ * x;
      err += (y1 - y2).squaredNorm();
      nrm += y1.squaredNorm();
    }
    return sqrt(err / nrm);
  }

  //////////////////////////////////////////////////////////////////////////////
  // getter
  const H2Matrix& K() const { return K_; }
  const Scalar fill_distance() const { return fill_distance_; }
  const Scalar separation_radius() const { return separation_radius_; }
  const Index solver_iterations() const { return solver_iterations_; }
  //////////////////////////////////////////////////////////////////////////////
  Vector solveIteratively(const Vector& rhs, bool CGwithPreconditioner = true,
                        Scalar threshold_CG = 1e-6) {
    Vector rhs_copy = rhs;
    rhs_copy = hct_.toClusterOrder(rhs_copy);
    Vector sol;

    if (!CGwithPreconditioner) {
      CG solver;
      EigenWrapper<H2Matrix> EigenH2(K_, ridgep_);
      solver.setTolerance(threshold_CG);
      solver.compute(EigenH2);
      sol = solver.solve(rhs_copy);
      solver_iterations_ = solver.iterations();
    } else {
      PreconditionedCG solver;
      EigenWrapper<H2Matrix> EigenH2(K_, ridgep_);
      solver.setTolerance(threshold_CG);
      solver.compute(EigenH2);
      sol = solver.solve(rhs_copy);
      solver_iterations_ = solver.iterations();
    }
    sol = hct_.toNaturalOrder(sol);
    return sol;
  }
  //////////////////////////////////////////////////////////////////////////////
  Matrix solveDirectly(const Matrix& rhs) {
    Matrix sol = hct_.toClusterOrder(rhs);
    sol = hct_.toNaturalOrder(sol);
    return sol;
  }

 private:
  H2Matrix K_;
  H2ClusterTree hct_;
  CovarianceKernel kernel_;
  Scalar mpole_deg_;
  Scalar eta_;
  Scalar threshold_;
  Scalar ridgep_;
  Scalar fill_distance_;
  Scalar separation_radius_;
  Index solver_iterations_;
};
}  // namespace FMCA

#endif
