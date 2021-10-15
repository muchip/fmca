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
#ifndef FMCA_UTIL_IDDFSFORWARDFITERATOR_H_
#define FMCA_UTIL_IDDFSFORWARDFITERATOR_H_

namespace FMCA {

template <typename T, bool IS_CONST>
struct IDDFSForwardIterator {
  using value_type = typename std::conditional<IS_CONST, const T, T>::type;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  explicit IDDFSForwardIterator(pointer ptr, IndexType depth,
                                IndexType max_depth)
      : ptr_(ptr), depth_(depth), max_depth_(max_depth) {
    // always set the begin to the root of the tree
    if (ptr_ != nullptr)
      while (ptr_->dad_ != nullptr) ptr_ = ptr_->dad_;
    root_ = ptr_;
  }

  reference operator*() const { return *ptr_; }
  pointer operator->() const { return ptr_; }

  // Prefix increment
  IDDFSForwardIterator& operator++() {
    // store the current value of the iterator;
    pointer ptr_old = ptr_;

    // as our search terminated, we are at a leaf with the current allowed
    // depth. Check if there are more of them by going up
    for (pointer it_p = ptr_; it_p->dad_ != nullptr;) {
      // check which son is the current one
      auto i = 0;
      for (; i < it_p->dad_->sons_.size(); ++i)
        if (it_p == std::addressof(it_p->dad_->sons_[i])) {
          break;
        }
      // if there is another branch, lets follow it to a leaf
      if (i < it_p->dad_->sons_.size() - 1) {
        ptr_ = std::addressof(it_p->dad_->sons_[i + 1]);
        while (ptr_->sons_.size() && ptr_->level_ < depth_) {
          ptr_ = std::addressof(ptr_->sons_[0]);
        }
        // did we find a valid next node? if so return it
        if (ptr_->level_ == depth_) return *this;
        it_p = ptr_;
      } else {
        it_p = it_p->dad_;
      }
    }
    // if we ended up here, we are back to the root. Now, we either
    // have already traversed all the nodes or we need to increase the
    // depth
    if (ptr_old == ptr_) {
      // end of the tree
      if (depth_ > max_depth_) {
        ptr_ = nullptr;
        return *this;
      } else {
        ++depth_;
        ptr_ = root_;
        while (ptr_->sons_.size() && ptr_->level_ < depth_) {
          ptr_ = std::addressof(ptr_->sons_[0]);
        }
        // did we find a valid next node? if so return it
        if (ptr_->level_ == depth_) return *this;
      }
    }
    ++(*this);
    return *this;
  }

  // Postfix increment
  IDDFSForwardIterator operator++(int) {
    IDDFSForwardIterator<T, true> tmp = *this;
    ++(*this);
    return tmp;
  }
  bool operator==(IDDFSForwardIterator<T, true> a) const {
    return ptr_ == a.ptr_;
  }
  bool operator!=(IDDFSForwardIterator<T, true> a) const {
    return !(this->ptr_ == a.ptr_);
  }

  // provide implicit conversion from iterator to const_iterator
  operator IDDFSForwardIterator<T, true>() const {
    return IDDFSForwardIterator<T, true>(ptr_, depth_, max_depth_);
  }

 private:
  pointer root_;
  pointer ptr_;
  IndexType depth_;
  IndexType max_depth_;
  // give iterator access to const_iterator::m_ptr
  friend IDDFSForwardIterator<T, false>;
};
}  // namespace FMCA
#endif
