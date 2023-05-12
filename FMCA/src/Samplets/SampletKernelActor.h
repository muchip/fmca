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
#ifndef FMCA_H2MATRIX_SAMPLETKERNELACTOR_H_
#define FMCA_H2MATRIX_SAMPLETKERNELACTOR_H_

namespace FMCA {

template <typename MatrixEvaluator, typename H2ClusterTree,
          typename SampletTree>
class SampletKernelActor {
 public:
  SampletKernelActor(const MatrixEvaluator &mat_eval,
                     const SampletTree &hst_eval, const SampletTree &hst,
                     const Index mpole_deg, const Scalar eta)
      : hst_(hst),
        hst_eval_(hst_eval),
        hct_(mat_eval.c_mom_, hst),
        hct_eval_(mat_eval.r_mom_, hst_eval),
        mat_eval_(mat_eval),
        mpole_deg_(mpole_deg),
        eta_(eta) {
    h2mat_.computePattern(hct_eval_, hct_, eta_);
    scheduler_.resize(h2mat_.ncclusters());
    for (const auto &it : h2mat_)
      if (!it.nSons())
        scheduler_[it.ccluster()->block_id()].push_back(std::addressof(it));
    return;
  }

  Matrix action(const Matrix &rhs) const {
    Matrix lhs(h2mat_.rows(), rhs.cols());
    Matrix srhs = hst_.inverseSampletTransform(rhs);
    lhs.setZero();
    // forward transform righ hand side
    std::vector<Matrix> trhs = internal::forward_transform_impl(h2mat_, srhs);
    std::vector<Matrix> tlhs(h2mat_.nrclusters());
    for (const auto &it : *(h2mat_.rcluster())) {
      if (it.nSons())
        tlhs[it.block_id()].resize(it.Es()[0].rows(), srhs.cols());
      else
        tlhs[it.block_id()].resize(it.V().rows(), srhs.cols());
      tlhs[it.block_id()].setZero();
    }
    for (const auto &it2 : scheduler_) {
#pragma omp parallel for schedule(dynamic)
      for (Index k = 0; k < it2.size(); ++k) {
        Matrix S;
        const H2Matrix<H2ClusterTree> &it = *(it2[k]);
        const Index i = it.rcluster()->block_id();
        const Index j = it.ccluster()->block_id();
        const Index ii = (it.rcluster())->indices_begin();
        const Index jj = (it.ccluster())->indices_begin();
        if (it.is_low_rank()) {
          mat_eval_.interpolate_kernel(*(it.rcluster()), *(it.ccluster()), &S);
          tlhs[i] += S * trhs[j];
        } else {
          mat_eval_.compute_dense_block(*(it.rcluster()), *(it.ccluster()), &S);
          lhs.middleRows(ii, S.rows()) += S * srhs.middleRows(jj, S.cols());
        }
      }
    }
    // backward transform left hand side
    internal::backward_transform_recursion(*(h2mat_.rcluster()), &lhs, tlhs);
    // reverse ordering to the one given by Peval
    return hst_eval_.sampletTransform(lhs);
  }

 private:
  const MatrixEvaluator &mat_eval_;
  const H2ClusterTree hct_;
  const H2ClusterTree hct_eval_;
  const SampletTree &hst_;
  const SampletTree &hst_eval_;
  const Index mpole_deg_;
  const Scalar eta_;
  H2Matrix<H2ClusterTree> h2mat_;
  std::vector<std::vector<const H2Matrix<H2ClusterTree> *>> scheduler_;
};

}  // namespace FMCA
#endif
