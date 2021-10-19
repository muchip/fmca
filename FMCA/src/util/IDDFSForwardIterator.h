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

template <typename T, bool IS_CONST> struct IDDFSForwardIterator {
  using value_type = typename std::conditional<IS_CONST, const T, T>::type;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;

  explicit IDDFSForwardIterator(pointer ptr, IndexType depth)
      : ptr_(ptr), depth_(depth), max_depth_(0) {}

  reference operator*() const { return *ptr_; }
  pointer operator->() const { return ptr_; }

  // Prefix increment
  IDDFSForwardIterator &operator++() {
    // as our search terminated, we are at a leaf with the current allowed
    // depth. Check if there are more of them by going up
    while (ptr_->dad_ != nullptr) {
      // if there is another branch, lets follow it to a leaf
      if (ptr_ != std::addressof(ptr_->dad_->sons_.back())) {
        ++ptr_;
        while (ptr_->sons_.size() && ptr_->level_ < depth_)
          ptr_ = std::addressof(ptr_->sons_[0]);
        // did we find a valid next node? if so return it
        max_depth_ = max_depth_ < ptr_->level_ ? ptr_->level_ : max_depth_;
        if (ptr_->level_ == depth_)
          return *this;
      } else
        ptr_ = ptr_->dad_;
    }
    if (depth_ <= max_depth_) {
      // increase depth and traverse the left branch to a leaf
      ++depth_;
      while (ptr_->sons_.size() && ptr_->level_ < depth_)
        ptr_ = std::addressof(ptr_->sons_[0]);
      max_depth_ = max_depth_ < ptr_->level_ ? ptr_->level_ : max_depth_;
      // did we find a valid next node? if so return it
      if (ptr_->level_ != depth_)
        ++(*this);
    } else
      ptr_ = nullptr;
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
    return IDDFSForwardIterator<T, true>(ptr_, depth_);
  }

private:
  pointer ptr_;
  IndexType depth_;
  IndexType max_depth_;
  // give iterator access to const_iterator::m_ptr
  friend IDDFSForwardIterator<T, false>;
};
} // namespace FMCA
#endif
