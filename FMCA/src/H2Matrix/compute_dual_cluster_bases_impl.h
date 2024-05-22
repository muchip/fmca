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
#ifndef FMCA_H2MATRIX_COMPUTEDUALCLUSTERBASESIMPL_H_
#define FMCA_H2MATRIX_COMPUTEDUALCLUSTERBASESIMPL_H_

namespace FMCA {
namespace internal {

/**
 * \ingroup internal
 * \brief Removed the type checker here as this is in any case an internal
 *        method. This way, we can reuse it for the H2SampletTree. At the
 *        moment there is no more elegant solution to avoid the diamond
 *        when inheriting from H2ClusterTreeBase and SampletTreeBase
 **/
struct compute_dual_cluster_bases_impl {
  compute_dual_cluster_bases_impl() {}

  template <typename Derived>
  static void compute(const TreeBase<Derived> &CT, std::vector<Matrix> *VTVs) {
    Eigen::FullPivHouseholderQR<Matrix> qr;
    Eigen::JacobiSVD<Matrix> svd;
    const Derived &H2T = CT.derived();
    Matrix V;
    if (H2T.is_root()) VTVs->resize(std::distance(H2T.begin(), H2T.end()));
    if (H2T.nSons()) {
      for (auto i = 0; i < H2T.nSons(); ++i)
        if (H2T.sons(i).block_size()) {
          // compute sons VTV and VTV^-1
          compute(H2T.sons(i), VTVs);
          // store the part of V that goes to parent
          const Matrix &R = (*VTVs)[H2T.sons(i).block_id()];
          V.conservativeResize(H2T.Es()[i].rows(), V.cols() + R.cols() / 2);
          V.rightCols(R.cols() / 2) = H2T.Es()[i] * R.leftCols(R.cols() / 2);
        }
    } else
      V = H2T.V();
    if (V.size()) {
      const Index rows = V.rows() < V.cols() ? V.rows() : V.cols();
      qr.compute(V.transpose());
      const Matrix R =
          qr.matrixQR().topRows(rows).template triangularView<Eigen::Upper>();
      const Matrix RT = qr.colsPermutation() * R.transpose();
      // compute the Moore-Penrose inverse of RT using SVD
      svd.compute(RT, Eigen::ComputeThinU | Eigen::ComputeThinV);
      const Matrix invRT = svd.matrixV() *
                           svd.singularValues().cwiseInverse().asDiagonal() *
                           svd.matrixU().transpose();
      // store everything
      (*VTVs)[H2T.block_id()].resize(RT.rows(), 2 * RT.cols());
      (*VTVs)[H2T.block_id()].leftCols(RT.cols()) = RT;
      (*VTVs)[H2T.block_id()].rightCols(RT.cols()) = invRT.transpose();
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  static int check_dual_cluster_bases(TreeBase<Derived> &CT,
                                      const std::vector<Matrix> &VTVs) {
    for (const auto &it : CT.derived()) {
      if (it.block_size()) {
        const Index bsize = VTVs[it.block_id()].cols() / 2;
        const Matrix invRT = VTVs[it.block_id()].rightCols(bsize).transpose();

        const Matrix Vdag = invRT.transpose() * (invRT * it.V()).eval();
        Scalar err =
            (it.V().transpose() -
             it.V().transpose() * ((Vdag * it.V().transpose()).eval()).eval())
                .norm() /
            it.V().transpose().norm();
        if (err >= 5e-10)
          std::cout << "***" << it.nSons() << " " << err << std::endl;
        eigen_assert(err < 5e-10 && "the H2 dual cluster basis is faulty");
      }
    }
    return 0;
  }
};

}  // namespace internal

}  // namespace FMCA

#endif
