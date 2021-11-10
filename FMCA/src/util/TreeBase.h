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
#ifndef FMCA_UTIL_TREEBASE_H_
#define FMCA_UTIL_TREEBASE_H_

#include <memory>
#include <vector>

#include "IDDFSForwardIterator.h"
#include "Macros.h"

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
  const Derived &derived() const { return *static_cast<const Derived *>(this); }
};

/**
 *  \ingroup util
 *  \brief manages a generic tree providing tree topology and a node
 *         iterator
 */
template <typename Derived>
class TreeBase {
 public:
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  typedef typename internal::traits<Derived>::node_type node_type;
  // when a tree is constructed, we add at least the memory for its node
  TreeBase() noexcept : dad_(nullptr), level_(0) {
    node_ = std::unique_ptr<NodeBase<node_type>>(new node_type);
  }
  //////////////////////////////////////////////////////////////////////////////
  using iterator = IDDFSForwardIterator<Derived, false>;
  using const_iterator = IDDFSForwardIterator<Derived, true>;
  friend iterator;
  friend const_iterator;
  //////////////////////////////////////////////////////////////////////////////
  // return a reference to the derived object
  Derived &derived() { return *static_cast<Derived *>(this); }
  // return a const reference to the derived object
  const Derived &derived() const { return *static_cast<const Derived *>(this); }
  //////////////////////////////////////////////////////////////////////////////
  // exposed the trees init routine
  template <typename... Ts>
  void init(Ts &&...ts) {
    derived().init(std::forward<Ts>(ts)...);
  }
  //////////////////////////////////////////////////////////////////////////////
  node_type &node() { return node_->derived(); }
  const node_type &node() const { return node_->derived(); }
  //////////////////////////////////////////////////////////////////////////////
  iterator begin() { return iterator(static_cast<Derived *>(this), 0); }
  iterator end() { return iterator(nullptr, 0); }
  const_iterator cbegin() const {
    return const_iterator(static_cast<const Derived *>(this), 0);
  }
  const_iterator cend() const { return const_iterator(nullptr, 0); }
  //////////////////////////////////////////////////////////////////////////////
  Derived &sons(typename std::vector<TreeBase>::size_type i) {
    return sons_[i].derived();
  }
  const Derived &sons(typename std::vector<TreeBase>::size_type i) const {
    return sons_[i].derived();
  }
  //////////////////////////////////////////////////////////////////////////////
  typename std::vector<TreeBase>::size_type nSons() { return sons_.size(); }
  const typename std::vector<TreeBase>::size_type nSons() const {
    return sons_.size();
  }
  //////////////////////////////////////////////////////////////////////////////
  void appendSons(typename std::vector<TreeBase>::size_type n) {
    sons_.resize(n);
    for (TreeBase &s : sons_) {
      s.dad_ = this;
      s.level_ = level_ + 1;
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  IndexType level() { return level_; };
  const IndexType level() const { return level_; };

  bool is_root() const { return dad_ == nullptr; }
  //////////////////////////////////////////////////////////////////////////////
  // provide a levelwise ordered output of the tree for debugging purposes only
  void exportTreeStructure(std::vector<std::vector<IndexType>> &tree) {
    if (level() >= tree.size()) tree.resize(level() + 1);
    tree[level()].push_back(node().indices_.size());
    for (auto i = 0; i < nSons(); ++i) sons(i).exportTreeStructure(tree);
  }
  //////////////////////////////////////////////////////////////////////////////
 private:
  std::vector<TreeBase> sons_;
  std::unique_ptr<NodeBase<node_type>> node_;
  TreeBase *dad_;
  IndexType level_;
};

}  // namespace FMCA

#endif
