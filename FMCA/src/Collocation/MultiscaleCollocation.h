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

#ifndef FMCA_COLLOCATION_MULTISCALECOLLOCATION_H_
#define FMCA_COLLOCATION_MULTISCALECOLLOCATION_H_

namespace FMCA {

/**
 *  \ingroup Collocation
 *  \brief Multiscale samplet-compressed kernel collocation for elliptic
 *         boundary value problems.
 *
 *
 *  The template parameter selects how a level is solved: SchurSolver for
 *  classical collocation, PIKLSolver for the regularised least-squares
 *  formulation.
 *
 **/
template <typename LevelSolver>
class MultiscaleCollocation {
 public:
  using Solver = LevelSolver;

  MultiscaleCollocation() noexcept : num_levels_(0), compute_condition_(true) {}

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Multiscale setup: every level gets sigma_l = nu * h_l, with h_l its
   *         fill distance.
   **/
  void init(const std::vector<Matrix>& PI_levels,
            const std::vector<Matrix>& PB_levels, Scalar nu,
            const std::string& kernel_type,
            const std::string& laplace_kernel_type, Index dtilde = 4,
            Scalar eta = 0.5, Scalar threshold = 1e-6) {
    std::vector<Scalar> sigmas(PI_levels.size());
    for (size_t l = 0; l < PI_levels.size(); ++l) {
      const Index nI = PI_levels[l].cols();
      Matrix P(PI_levels[l].rows(), nI + PB_levels[l].cols());
      P.leftCols(nI) = PI_levels[l];
      P.rightCols(PB_levels[l].cols()) = PB_levels[l];
      sigmas[l] = nu * fillDistance(P);
    }
    init(PI_levels, PB_levels, sigmas, kernel_type, laplace_kernel_type, dtilde,
         eta, threshold);
    return;
  }

  /**
   *  \brief Setup with prescribed length scales.  A single entry is the
   *         classical single-scale collocation with a fixed kernel width.
   **/
  void init(const std::vector<Matrix>& PI_levels,
            const std::vector<Matrix>& PB_levels,
            const std::vector<Scalar>& sigmas, const std::string& kernel_type,
            const std::string& laplace_kernel_type, Index dtilde = 4,
            Scalar eta = 0.5, Scalar threshold = 1e-6) {
    assert(PI_levels.size() == PB_levels.size() &&
           PI_levels.size() == sigmas.size() &&
           "one interior set, one boundary set and one sigma per level");
    num_levels_ = PI_levels.size();
    kernel_type_ = kernel_type;
    matrices_.clear();
    matrices_.reserve(num_levels_);
    for (Index l = 0; l < num_levels_; ++l) {
      matrices_.emplace_back();
      matrices_[l].init(PI_levels[l], PB_levels[l], kernel_type,
                        laplace_kernel_type, sigmas[l], dtilde, eta, threshold);
    }
    coefficients_.assign(num_levels_, Vector());
    iterations_.assign(num_levels_, 0);
    condition_.assign(num_levels_, 0);
    nnz_per_row_.assign(num_levels_, 0);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Solves one level against the residual of all coarser ones.
   **/
  void solveLevel(Index level, const Vector& fI, const Vector& gB) {
    CollocationMatrix& cm = matrices_[level];
    cm.compute();
    Vector rhs = cm.transformData(fI, gB);
    for (Index c = 0; c < level; ++c)
      cm.subtractCoarse(matrices_[c], coefficients_[c], rhs);
    solver_.compute(cm.A(), cm.B(), cm.C(), cm.D());
    const Vector z = solver_.solve(Vector(rhs.head(cm.nI())),
                                   Vector(rhs.tail(cm.nB())));
    iterations_[level] = solver_.iterations();
    nnz_per_row_[level] = cm.nonZerosPerRow();
    if (compute_condition_) condition_[level] = solver_.conditionEstimate();
    coefficients_[level] = cm.toNatural(z);
    cm.releaseBlocks();
    return;
  }

  /**
   *  \brief Contribution of one level to the solution on the background set of
   *         the evaluator.  The full solution is the sum over all levels.
   **/
  Vector evaluateLevel(MultiscaleEvaluator& evaluator, Index level) {
    return evaluator.evaluate(
        matrices_[level].points(), coefficients_[level],
        CovarianceKernel(kernel_type_, matrices_[level].sigma()));
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Switches the condition number estimate on or off.  It costs about
   *         one hundred products with the interior block, so it is worth
   *         turning off when it is not reported.  On by default.
   **/
  void setComputeCondition(bool flag) {
    compute_condition_ = flag;
    return;
  }

  /**
   *  \brief Fill distance of a point cloud, the largest nearest neighbour
   *         distance, used for the rescaling sigma_l = nu * h_l.
   **/
  static Scalar fillDistance(const Matrix& P) {
    const ClusterTree ct(P, 10);
    return minDistanceVector(ct, P).maxCoeff();
  }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  Solver& solver() { return solver_; }
  const Index numLevels() const { return num_levels_; }
  const Matrix& points(Index level) const { return matrices_[level].points(); }
  const Vector& coefficients(Index level) const { return coefficients_[level]; }
  const Scalar sigma(Index level) const { return matrices_[level].sigma(); }
  const Index nI(Index level) const { return matrices_[level].nI(); }
  const Index nB(Index level) const { return matrices_[level].nB(); }
  const Index iterations(Index level) const { return iterations_[level]; }
  const Scalar conditionEstimate(Index level) const { return condition_[level]; }
  const Scalar nonZerosPerRow(Index level) const { return nnz_per_row_[level]; }

 private:
  std::vector<CollocationMatrix> matrices_;
  std::vector<Vector> coefficients_;
  std::vector<Index> iterations_;
  std::vector<Scalar> condition_;
  std::vector<Scalar> nnz_per_row_;
  Solver solver_;
  std::string kernel_type_;
  Index num_levels_;
  bool compute_condition_;
};

//////////////////////////////////////////////////////////////////////////////
using MultiscaleCollocationSolver = MultiscaleCollocation<SchurSolver>;
using MultiscalePIKLSolver = MultiscaleCollocation<PIKLSolver>;

}  // namespace FMCA

#endif
