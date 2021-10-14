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

namespace FMCA {

namespace internal {
template <typename Derived>
struct traits {};

}  // namespace internal
template <typename Derived>
struct NodeBase {
  // return a reference to the derived object
  Derived &derived() { return *static_cast<Derived *>(this); }
  // return a const reference to the derived object */
};

/**
 *  \ingroup util
 *  \brief manages a generic tree providing tree topology and a node
 *         iterator
 */
template <typename Derived>
class TreeBase {
 public:
  typedef typename internal::traits<Derived>::node_type node_type;
  // when a tree is constructed, we add at least the memory for its node
  TreeBase() noexcept
      : dad_(nullptr), prev_(nullptr), next_(nullptr), level_(0) {
    node_ = std::unique_ptr<NodeBase<node_type>>(new node_type);
  }
  //////////////////////////////////////////////////////////////////////////////
  using iterator = GenericForwardIterator<TreeBase, false>;
  using const_iterator = GenericForwardIterator<TreeBase, true>;
  // return a reference to the derived object
  Derived &derived() { return *static_cast<Derived *>(this); }
  // return a const reference to the derived object */
  const Derived &derived() const { return *static_cast<const Derived *>(this); }
  // exposed the trees init routine
  template <typename... Ts>
  void init(Ts &&...ts) {
    derived().init(std::forward<Ts>(ts)...);
  }
  // we assume polymorphic node data and allow up and
  // down casting while accessing them
  node_type &node() { return node_->derived(); }

  const node_type &node() const { return node_->derived(); }

  //////////////////////////////////////////////////////////////////////////////
  Derived &sons(typename std::vector<TreeBase>::size_type i) {
    return sons_[i].derived();
  }
  const Derived &sons(typename std::vector<TreeBase>::size_type i) const {
    return sons_[i].derived();
  }

  typename std::vector<TreeBase>::size_type nSons() { return sons_.size(); }
  void appendSons(typename std::vector<TreeBase>::size_type n) {
    sons_.resize(n);
    for (TreeBase &s : sons_) {
      s.dad_ = this;
      s.level_ = level_ + 1;
    }
  }

  // we also expose the node_ ptr such that we may mutate it
  // std::unique_ptr<NodeBase<Derived>> &pnode() { return node_; }

  void updateNodeList() {}
  //////////////////////////////////////////////////////////////////////////////
 private:
  std::vector<TreeBase> sons_;
  std::unique_ptr<NodeBase<node_type>> node_;
  TreeBase *dad_;
  TreeBase *prev_;
  TreeBase *next_;
  IndexType level_;
};

}  // namespace FMCA

#endif
