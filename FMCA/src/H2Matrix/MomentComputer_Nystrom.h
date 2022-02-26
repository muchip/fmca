// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_MOMENTS_MOMENTCOMPUTER_NYSTROM_H_
#define FMCA_MOMENTS_MOMENTCOMPUTER_NYSTROM_H_

namespace FMCA {
template <typename Derived> class MomentComputer_Nystrom {
public:
  MomentComputer_Nystrom(const Derived &P) : P_(P) {}

  IndexType dim() { return P_.rows(); }
  
  template <typename Interpolator, typename otherDerived>
  typename otherDerived::eigenMatrix moment_matrix(const Interpolator &interp,
                                                   TreeBase<otherDerived> &CT) {
    otherDerived &H2T = CT.derived();
    for (auto i = 0; i < H2T.indices().size(); ++i)
      H2T.V().col(i) = H2T.node().interp_->evalPolynomials(
          ((P_.col(H2T.indices()[i]) - H2T.bb().col(0)).array() /
           H2T.bb().col(2).array())
              .matrix());
  }

private:
  const Derived &P_;
};

} // namespace FMCA
#endif
