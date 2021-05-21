// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_SAMPLETS_BIVARIATECOMPRESSOR_H_
#define FMCA_SAMPLETS_BIVARIATECOMPRESSOR_H_

namespace FMCA {

template <typename SampletTree> class BivariateCompressor {
public:
  typedef SampletTree::value_type value_type;
  BivariateCompressor(){};
  template <typename Functor>
  BivariateCompressor(const SampletTree &ST, const Functor &fun,
                      value_type a_param = 1.1, value_type dp_param = 1.1,
                      value_type operator_order = 0) {
    init(ST, fun, a_param, dp_param, operator_order);
  }

  void init(const SampletTree &ST, const Functor &fun, value_type a_param = 1.1,
            value_type dp_param = 1.1, value_type operator_order = 0) {
    a_param_ = a_param;
    dp_param_ = dp_param;
    operator_order_ = operator_order;
    cut_const1_ = ST.tree_data_->max_wlevel * (dp_param_ - operator_order_) /
                  (ST.tree_data_->dtilde_ + operator_order_);
    cut_const2_ = (dp_param_ + ST.tree_data_->dtilde_) / 2 *
                  (ST.tree_data_->dtilde_ + operator_order_);
  }

  //////////////////////////////////////////////////////////////////////////////
  /// cutOff criterion
  //////////////////////////////////////////////////////////////////////////////
  value_type cutOffParameter(IndexType j, IndexType jp) {
    const value_type first = j < jp ? 1. / (1 << j) : 1. / (1 << jp);
    const value_type second =
        std::pow(2., cut_const1_ - (j + jp) * cut_const2_);
    return a_param_ * (first > second ? first : second);
  }
  //////////////////////////////////////////////////////////////////////////////
  bool cutOff(IndexType j, IndexType jp, value_type dist) {
    return cutOffParameter(j, jp) < dist;
  }

private:
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  value_type a_param_;
  value_type dp_param_;
  value_type dt_param_;
  value_type operator_order_;
  IndexType max_wlevel_;
  value_type cut_const1_;
  value_type cut_const2_;
}; // namespace FMCA
} // namespace FMCA
#endif
