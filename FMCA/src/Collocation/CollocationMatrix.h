// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer, Sara Avesani
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//

#ifndef FMCA_COLLOCATION_COLLOCATIONMATRIX_H_
#define FMCA_COLLOCATION_COLLOCATIONMATRIX_H_

namespace FMCA {

/**
 *  \ingroup Collocation
 *  \brief Samplet-compressed blocks of the unsymmetric (Kansa) collocation
 *         system for one point set and one kernel length scale sigma,
 *
 *      G = [ A  B ]     A = (Delta_x K)(X_I, X_I)   B = (Delta_x K)(X_I, X_B)
 *          [ C  D ]     C =           K (X_B, X_I)  D =           K (X_B, X_B).
 *
 **/
class CollocationMatrix {
 public:
  using Interpolator = TotalDegreeInterpolator;
  using SampletInterpolator = MonomialInterpolator;
  using Moments = NystromMoments<Interpolator>;
  using SampletMoments = NystromSampletMoments<SampletInterpolator>;
  using SampletTree = H2SampletTree<ClusterTree>;
  using KernelEvaluator =
      unsymmetricNystromEvaluator<Moments, CovarianceKernel>;
  using GradEvaluator = unsymmetricNystromEvaluator<Moments, GradKernel>;

  CollocationMatrix() noexcept {}

  CollocationMatrix(const CollocationMatrix& other) = delete;

  CollocationMatrix(CollocationMatrix&& other) noexcept {
    // dummy move constructor, needed to wrap the class into std::vector
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Sets the points, the kernel and the compression parameters, and
   *         builds the samplet trees.
   **/
  void init(const Matrix& PI, const Matrix& PB, const std::string& kernel_type,
            const std::string& laplace_kernel_type, Scalar sigma,
            Index dtilde = 4, Scalar eta = 0.5, Scalar threshold = 1e-6) {
    PI_ = PI;
    PB_ = PB;
    kernel_type_ = kernel_type;
    laplace_kernel_type_ = laplace_kernel_type;
    sigma_ = sigma > 0 ? sigma : 1.;
    dtilde_ = dtilde > 0 ? dtilde : 1;
    mpole_deg_ = dtilde_ > 1 ? (2 * (dtilde_ - 1)) : 1;
    eta_ = eta >= 0 ? eta : 0;
    threshold_ = threshold >= 0 ? threshold : 0;
    nI_ = PI_.cols();
    nB_ = PB_.cols();
    P_.resize(PI_.rows(), nI_ + nB_);
    P_.leftCols(nI_) = PI_;
    P_.rightCols(nB_) = PB_;
    initTree(TI_, PI_);
    initTree(TB_, PB_);
    initTree(T_, P_);
    anz_ = 0;
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Assembles and compresses the four blocks.
   **/
  void compute() {
    const Moments mom_I(PI_, mpole_deg_);
    const Moments mom_B(PB_, mpole_deg_);
    A_.resize(nI_, nI_);
    A_.setZero();
    B_.resize(nI_, nB_);
    B_.setZero();
    for (Index d = 0; d < PI_.rows(); ++d) {
      const GradKernel gk(laplace_kernel_type_, sigma_, 1, d);
      A_ += block(TI_, TI_, GradEvaluator(mom_I, mom_I, gk), nI_, nI_);
      B_ += block(TI_, TB_, GradEvaluator(mom_I, mom_B, gk), nI_, nB_);
    }
    const CovarianceKernel kernel(kernel_type_, sigma_);
    C_ = block(TB_, TI_, KernelEvaluator(mom_B, mom_I, kernel), nB_, nI_);
    D_ = block(TB_, TB_, KernelEvaluator(mom_B, mom_B, kernel), nB_, nB_);
    anz_ = A_.nonZeros() + B_.nonZeros() + C_.nonZeros() + D_.nonZeros();
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief The right hand side [f(X_I); g(X_B)] in samplet coordinates.
   **/
  Vector transformData(const Vector& fI, const Vector& gB) {
    Vector rhs(nI_ + nB_);
    rhs.head(nI_) = TI_.sampletTransform(TI_.toClusterOrder(fI));
    rhs.tail(nB_) = TB_.sampletTransform(TB_.toClusterOrder(gB));
    return rhs;
  }

  /**
   *  \brief Coefficients back from samplet coordinates to the ordering of
   *         points().
   **/
  Vector toNatural(const Vector& z) {
    Vector a(nI_ + nB_);
    a.head(nI_) =
        TI_.toNaturalOrder(TI_.inverseSampletTransform(Vector(z.head(nI_))));
    a.tail(nB_) =
        TB_.toNaturalOrder(TB_.inverseSampletTransform(Vector(z.tail(nB_))));
    return a;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Subtracts from rhs what a coarser level already represents, which
   *         is what turns this level into a correction of the coarser ones.
   *         a_coarse are the coefficients of that level in its natural order.
   *         The coupling blocks use the length scale of the COARSE level and
   *         are discarded again after use.
   **/
  void subtractCoarse(CollocationMatrix& coarse, const Vector& a_coarse,
                      Vector& rhs) {
    const Index nC = coarse.P_.cols();
    const Moments mom_I(PI_, mpole_deg_);
    const Moments mom_B(PB_, mpole_deg_);
    const Moments mom_C(coarse.P_, mpole_deg_);
    const Vector a =
        coarse.T_.sampletTransform(coarse.T_.toClusterOrder(a_coarse));
    for (Index d = 0; d < PI_.rows(); ++d) {
      const GradKernel gk(laplace_kernel_type_, coarse.sigma_, 1, d);
      rhs.head(nI_) -=
          block(TI_, coarse.T_, GradEvaluator(mom_I, mom_C, gk), nI_, nC) * a;
    }
    const CovarianceKernel kernel(kernel_type_, coarse.sigma_);
    rhs.tail(nB_) -=
        block(TB_, coarse.T_, KernelEvaluator(mom_B, mom_C, kernel), nB_, nC) *
        a;
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Releases the four blocks.  The trees and the points are kept, so
   *         the level can still serve as the coarse partner of a finer one.
   **/
  void releaseBlocks() {
    A_.resize(0, 0);
    B_.resize(0, 0);
    C_.resize(0, 0);
    D_.resize(0, 0);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  const SparseMatrix& A() const { return A_; }
  const SparseMatrix& B() const { return B_; }
  const SparseMatrix& C() const { return C_; }
  const SparseMatrix& D() const { return D_; }
  const Matrix& points() const { return P_; }
  const Scalar sigma() const { return sigma_; }
  const Index nI() const { return nI_; }
  const Index nB() const { return nB_; }
  const Scalar nonZerosPerRow() const { return anz_ / Scalar(nI_ + nB_); }

 private:
  //////////////////////////////////////////////////////////////////////////////
  void initTree(SampletTree& T, const Matrix& P) {
    const Moments mom(P, mpole_deg_);
    const SampletMoments smom(P, dtilde_ - 1);
    T.init(mom, smom, 0, P);
    return;
  }

  /**
   *  \brief A priori compression followed by a posteriori thresholding of a
   *         single block, the routine every block goes through.
   **/
  template <typename Evaluator>
  SparseMatrix block(SampletTree& row, SampletTree& col,
                     const Evaluator& mat_eval, Index rows, Index cols) {
    internal::SampletMatrixCompressorUnsymmetric<SampletTree> compressor;
    compressor.init(row, col, eta_, 100 * FMCA_ZERO_TOLERANCE);
    compressor.compress(mat_eval);
    compressor.triplets();
    const auto& trips = compressor.aposteriori_triplets_fast(threshold_);
    SparseMatrix M(rows, cols);
    M.setFromTriplets(trips.begin(), trips.end());
    M.makeCompressed();
    return M;
  }

  SampletTree TI_;
  SampletTree TB_;
  SampletTree T_;
  SparseMatrix A_;
  SparseMatrix B_;
  SparseMatrix C_;
  SparseMatrix D_;
  Matrix PI_;
  Matrix PB_;
  Matrix P_;
  std::string kernel_type_;
  std::string laplace_kernel_type_;
  Scalar sigma_;
  Index dtilde_;
  Index mpole_deg_;
  Scalar eta_;
  Scalar threshold_;
  Index nI_;
  Index nB_;
  size_t anz_;
};
}  // namespace FMCA

#endif
