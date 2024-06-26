// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2024, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_H2MATRIX_H2MATRIXPROJECTOR_H_
#define FMCA_H2MATRIX_H2MATRIXPROJECTOR_H_

namespace FMCA {

/**
 *  \ingroup H2Matrix
 *  \brief The H2MatrixProjector projects a given matrix into a given H2 format
 */
template <typename Derived>
class H2MatrixProjector {
 public:
  //////////////////////////////////////////////////////////////////////////////
  // constructor
  //////////////////////////////////////////////////////////////////////////////
  H2MatrixProjector(const H2MatrixBase<Derived> &h2mat)
      : h2mat_(h2mat.derived()) {
    assert(h2mat_.is_root() && "needs to be called from the root note atm");
    internal::compute_dual_cluster_bases_impl::compute(*(h2mat_.rcluster()),
                                                       &rRs_);
    internal::compute_dual_cluster_bases_impl::compute(*(h2mat_.ccluster()),
                                                       &cRs_);
  }
  Derived project(const Matrix &mat) {
    assert(mat.rows() == h2mat_.rows() && mat.cols() == h2mat_.cols() &&
           "incompatible matrix sizes");
    // assemble matrix pattern of the return value
    std::vector<std::vector<Matrix>> tmat;
    {
      std::vector<Matrix> trows(h2mat_.nrclusters());
      internal::forward_transform_recursion(*(h2mat_.rcluster()), &trows, mat);
      tmat.resize(trows.size());
      for (Index i = 0; i < tmat.size(); ++i) {
        tmat[i].resize(h2mat_.ncclusters());
        internal::forward_transform_recursion(*(h2mat_.ccluster()), &(tmat[i]),
                                              Matrix(trows[i].transpose()));
      }
    }
    // write matrix pattern
    Derived retval(h2mat_.copyPattern());
    // forward transform the entire matrix
    for (auto &&it : retval) {
      if (!it.nSons()) {
        const Index rid = it.rcluster()->block_id();
        const Index cid = it.ccluster()->block_id();
        const Matrix rinvR = rRs_[rid].rightCols(rRs_[rid].cols() / 2);
        const Matrix cinvR = cRs_[cid].rightCols(cRs_[cid].cols() / 2);
        if (it.is_low_rank())
          it.node().S_ = rinvR * (tmat[rid][cid] * rinvR).transpose() * cinvR *
                         cinvR.transpose();
        else
          it.node().S_ =
              mat.block(it.rcluster()->indices_begin(),
                        it.ccluster()->indices_begin(), it.rows(), it.cols());
      }
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
 private:
  const Derived &h2mat_;
  std::vector<Matrix> rRs_;
  std::vector<Matrix> cRs_;
};

}  // namespace FMCA
#endif
