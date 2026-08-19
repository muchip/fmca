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

#ifndef FMCA_COLLOCATION_MULTISCALEEVALUATOR_H_
#define FMCA_COLLOCATION_MULTISCALEEVALUATOR_H_

namespace FMCA {

/**
 *  \ingroup Collocation
 *  \brief Evaluates kernel expansions on a fixed background point set by the
 *         fast multipole method.
 *
 *  A multiscale solution is a sum of expansions, one per level, each with its
 *  own points and its own length scale, but all evaluated on the SAME
 *  background set.  The cluster tree of that background set is therefore built
 *  once in init() and reused by every call to evaluate(), which matters as
 *  soon as the background set is large.
 **/
class MultiscaleEvaluator {
 public:
  using Interpolator = TotalDegreeInterpolator;
  using Moments = NystromMoments<Interpolator>;
  using H2ClusterT = H2ClusterTree<ClusterTree>;
  using H2Mat = H2Matrix<H2ClusterT, CompareCluster>;
  using MatrixEvaluator =
      unsymmetricNystromEvaluator<Moments, CovarianceKernel>;

  MultiscaleEvaluator() noexcept {}

  MultiscaleEvaluator(const Matrix& P_eval, Index mpole_deg = 6,
                      Scalar eta = 0.5) noexcept {
    init(P_eval, mpole_deg, eta);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Builds the cluster tree of the background set, once.
   **/
  void init(const Matrix& P_eval, Index mpole_deg = 6, Scalar eta = 0.5) {
    P_eval_ = P_eval;
    mpole_deg_ = mpole_deg;
    eta_ = eta >= 0 ? eta : 0;
    const Moments rmom(P_eval_, mpole_deg_);
    hct_eval_.init(rmom, 0, P_eval_);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief K(P_eval, P) * alpha, with alpha in the natural ordering of P.
   **/
  Vector evaluate(const Matrix& P, const Vector& alpha,
                  const CovarianceKernel& kernel) {
    const Moments rmom(P_eval_, mpole_deg_);
    const Moments cmom(P, mpole_deg_);
    H2ClusterT hct;
    hct.init(cmom, 0, P);
    const MatrixEvaluator mat_eval(rmom, cmom, kernel);
    H2Mat h2mat;
    h2mat.computePattern(hct_eval_, hct, eta_);
    return hct_eval_.toNaturalOrder(
        h2mat.action(mat_eval, hct.toClusterOrder(alpha)));
  }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  const Matrix& points() const { return P_eval_; }

 private:
  Matrix P_eval_;
  H2ClusterT hct_eval_;
  Index mpole_deg_;
  Scalar eta_;
};
}  // namespace FMCA

#endif
