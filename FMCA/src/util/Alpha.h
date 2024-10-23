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
#ifndef FMCA_UTIL_ALPHA_H_
#define FMCA_UTIL_ALPHA_H_

#include "Macros.h"

namespace FMCA {
class Alpha {
 public:
  Alpha(){};
  Alpha(const iVector &nds) : nds_(nds) {
    n_ = 1;
    for (Index i = 0; i < nds_.size(); ++i) {
      assert(FMCA_MAXINDEX / n_ >= nds_[i] &&
             "Alpha: number of multi indices exceeds FMCA_MAXINDEX");
      n_ *= nds_(i);
    }
    base_ = computeBase(nds_);
  }

  template <typename T>
  Index toScalarIndex(const T &alpha) {
    Index retval = 0;
    for (Index i = 0; i < nds_.size(); ++i) retval += base_(i) * alpha[i];
    return retval;
  }

  template <typename T>
  T toMultiIndex(const Index ind) {
    T retval;
    retval.resize(nds_.size());
    Index remainder = ind;
    Index quotient = 0;
    for (Index i = 0; i < base_.size(); ++i) {
      quotient = remainder / base_(i);
      retval[i] = quotient;
      remainder -= base_(i) * quotient;
    }
    return retval;
  }

  Index n() const { return n_; };

 private:
  static iVector computeBase(const iVector &nds) {
    iVector base(nds.size());
    base.setOnes();
    for (Index i = 0; i < base.size(); ++i)
      base(i) = nds.tail(nds.size() - i - 1).prod();
    return base;
  }
  iVector nds_;
  iVector base_;
  Index n_;
};
}  // namespace FMCA
#endif
