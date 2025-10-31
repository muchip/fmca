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
#ifndef FMCA_WEDGELETS_WEDGELETTREEHELPER_H_
#define FMCA_WEDGELETS_WEDGELETTREEHELPER_H_

namespace FMCA {

struct WedgeletTreeHelper {
  template <typename Indexset>
  static Matrix computeLeastSquaresFit(const Index *cluster_idcs,
                                       const Index block_size, const Matrix &P,
                                       const Matrix &F, const Indexset &midcs,
                                       Scalar *err = nullptr) {
    Matrix retval;
    Matrix VT(midcs.index_set().size(), block_size);
    Matrix rhs(block_size, F.cols());
    for (Index i = 0; i < block_size; ++i) {
      VT.col(i) = internal::evalPolynomials(midcs, P.col(cluster_idcs[i]));
      rhs.row(i) = F.row(cluster_idcs[i]);
    }
    Eigen::ColPivHouseholderQR<Matrix> qr;
    qr.compute(VT.transpose());
    retval = qr.solve(rhs);
    if (err != nullptr)
      *err = (rhs - VT.transpose() * retval).squaredNorm() / rhs.size();
    return retval;
  }
};
}  // namespace FMCA
#endif
