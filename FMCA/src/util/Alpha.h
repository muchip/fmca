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
  Alpha(const iVector &nds) { init(nds); }

  void init(const iVector &nds) {
    nds_ = nds;
    n_ = 1;
    for (Index i = 0; i < nds_.size(); ++i) {
      assert(FMCA_MAXINDEX / n_ >= nds_[i] &&
             "Alpha: number of multi indices exceeds FMCA_MAXINDEX");
      n_ *= nds_(i);
    }
    base_ = computeBase(nds_);
    return;
  }

  template <typename T>
  Index toScalarIndex(const T &alpha) {
    Index retval = 0;
    for (Index d = 0; d < nds_.size(); ++d) retval += base_(d) * alpha[d];
    return retval;
  }

  template <typename T>
  T toMultiIndex(const Index ind) {
    T retval;
    retval.resize(nds_.size());
    Index remainder = ind;
    for (Index d = 0; d < base_.size(); ++d) {
      retval[d] = remainder / base_(d);
      remainder %= base_(d);
    }
    return retval;
  }

  Index matricize(const Index dim, const Index i, const Index j) const {
    Index retval = 0;
    Index remainder = j;

    for (Index d = 0; d < dim; ++d) {
      const Index base = base_(d) / nds_(dim);
      Index col = remainder / base;
      remainder %= base;
      retval += base_(d) * col;
    }

    retval += i * base_(dim);

    for (Index d = dim + 1; d < base_.size(); ++d) {
      Index col = remainder / base_(d);
      remainder %= base_(d);
      retval += base_(d) * col;
    }

    return retval;
  }

  const iVector &base() const { return base_; }

  const iVector &nds() const { return nds_; }

  Index n() const { return n_; }

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
