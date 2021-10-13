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
#ifndef FMCA_UTIL_TREEBASE_H_
#define FMCA_UTIL_TREEBASE_H_

#include <memory>
#include <vector>

#include "Macros.h"
namespace FMCA {

template <typename Derived>
class TreeBase {
 public:
  // we assume polymorphic node data and allow up and
  // down casting while accessing them
  template <typename NodeDataType>
  const Derived &data() const {
    return *(dynamic_cast<NodeDataType *>(n_data_.get()));
  }
  template <typename NodeDataType>
  Derived &data() {
    return *(dynamic_cast<NodeDataType *>(n_data_.get()));
  }
  //////////////////////////////////////////////////////////////////////////////
  const std::vector<TreeBase> &sons() const { return sons_; }
  std::vector<TreeBase> &sons() { return sons_; }

  IndexType get_level() const { return level_; }

  IndexType get_id() const { return id_; }
  //////////////////////////////////////////////////////////////////////////////
 private:
  std::vector<TreeBase> sons_;
  std::unique_ptr<typename Derived::NodeData> n_data_;
  TreeBase *prev_;
  TreeBase *ndex_;
  IndexType level_;
  IndexType id_;
};

}  // namespace FMCA

#endif
