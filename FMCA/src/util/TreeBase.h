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
  typedef typename internal::traits<Derived>::Node Node;
  // when a tree is constructed, we add at least the memory for its node
  TreeBase() noexcept : dad_(nullptr), level_(0) {
    node_ = std::unique_ptr<Node>(new Node);
  }

  //////////////////////////////////////////////////////////////////////////////
  // other constructors should follow in the future... -> big three
  TreeBase(TreeBase &&other) {
    // if a tree is copied, the current node is always considered as root;
    dad_ = nullptr;
    level_ = 0;
    // swap sons
    sons_.swap(other.sons_);
    // swap node
    node_.swap(other.node_);
    // fix topology of the tree (std guarantees that all addresses remain intact
    // in principle, however we need to fix the level in any case)
    std::vector<TreeBase *> stack;
    stack.push_back(this);
    while (stack.size()) {
      TreeBase *node = stack.back();
      stack.pop_back();
      for (TreeBase &s : sons_) {
        s.dad_ = node;
        s.level_ = node->level_ + 1;
        stack.push_back(std::addressof(s));
      }
    }
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
  Node &node() { return *node_; }
  const Node &node() const { return *node_; }
  //////////////////////////////////////////////////////////////////////////////
  iterator begin() { return iterator(static_cast<Derived *>(this), 0); }
  iterator end() { return iterator(nullptr, 0); }
  // the following guys are added to not break range based loop in const case
  const_iterator begin() const {
    return const_iterator(static_cast<const Derived *>(this), 0);
  }
  const_iterator end() const { return iterator(nullptr, 0); }
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
  Derived &dad() { return dad_->derived(); }
  const Derived &dad() const { return dad_->derived(); }
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
  Index level() { return level_; };
  const Index level() const { return level_; };

  bool is_root() const { return dad_ == nullptr; }
  //////////////////////////////////////////////////////////////////////////////
  // provide a levelwise ordered output of the tree for debugging purposes only
  void exportTreeStructure(std::vector<std::vector<Index>> &tree) {
    if (level() >= tree.size()) tree.resize(level() + 1);
    tree[level()].push_back(node().indices_.size());
    for (auto i = 0; i < nSons(); ++i) sons(i).exportTreeStructure(tree);
  }
  //////////////////////////////////////////////////////////////////////////////
 private:
  std::vector<TreeBase> sons_;
  std::unique_ptr<Node> node_;
  TreeBase *dad_;
  Index level_;
};

}  // namespace FMCA

#endif
