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

#ifndef FMCA_KERNELINTERPOLATION_MULTIPOLEFUNCTIONEVALUATOR_H_
#define FMCA_KERNELINTERPOLATION_MULTIPOLEFUNCTIONEVALUATOR_H_

namespace FMCA {

class MultipoleFunctionEvaluator {
 public:
  using Interpolator = TotalDegreeInterpolator;
  using Moments = NystromMoments<Interpolator>;
  using H2CT = H2ClusterTree<ClusterTree>;
  using H2Mat = H2Matrix<H2CT, CompareCluster>;
  using MatEval = unsymmetricNystromEvaluator<Moments, CovarianceKernel>;

  MultipoleFunctionEvaluator() {}

  MultipoleFunctionEvaluator(const CovarianceKernel& kernel, const Matrix& P,
                             const Matrix& Peval, Scalar eta = 0.,
                             Index mpole_deg = 3) {
    init(kernel, P, Peval, eta, mpole_deg);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void init(const CovarianceKernel& kernel, const Matrix& P,
            const Matrix& Peval, Scalar eta = 0., Index mpole_deg = 3) {
    // set parameters
    eta_ = eta >= 0 ? eta : 0;
    mpole_deg_ = mpole_deg;
    kernel_ = kernel;
    const Moments rmom(Peval, mpole_deg_);
    const Moments cmom(P, mpole_deg_);
    hct_.init(cmom, 0, P);
    hct_eval_.init(rmom, 0, Peval);
    h2mat_.computePattern(hct_eval_, hct_, eta_);
    h2mat_.statistics();

    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  Matrix evaluate(const Matrix& P, const Matrix& Peval, const Matrix& C) {
    const Moments rmom(Peval, mpole_deg_);
    const Moments cmom(P, mpole_deg_);
    const MatEval mat_eval(rmom, cmom, kernel_);
    Matrix S = h2mat_.action(mat_eval, hct_.toClusterOrder(C));
    return hct_eval_.toNaturalOrder(S);
  }

 private:
  CovarianceKernel kernel_;
  H2CT hct_;
  H2CT hct_eval_;
  H2Mat h2mat_;
  Scalar mpole_deg_;
  Scalar eta_;
};
}  // namespace FMCA

#endif
