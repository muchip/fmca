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
#ifndef FMCA_UTIL_GENERICITERATOR_H_
#define FMCA_UTIL_GENERICITERATOR_H_

namespace FMCA {

template <typename T, bool IS_CONST>
struct GenericForwardIterator {
  using value_type = typename std::conditional<IS_CONST, const T, T>::type;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  explicit GenericForwardIterator(pointer ptr) : m_ptr(ptr) {}

  reference operator*() const { return *m_ptr; }
  pointer operator->() const { return m_ptr; }

  // Prefix increment
  GenericForwardIterator& operator++() {
    m_ptr = m_ptr->next_;
    return *this;
  }

  // Postfix increment
  GenericForwardIterator operator++(int) {
    GenericForwardIterator<T, true> tmp = *this;
    ++(*this);
    return tmp;
  }
  bool operator==(GenericForwardIterator<T, true> a) const {
    return m_ptr == a.m_ptr;
  }
  bool operator!=(GenericForwardIterator<T, true> a) const {
    return !(this->m_ptr == a.m_ptr);
  }

  // provide implicit conversion from iterator to const_iterator
  operator GenericForwardIterator<T, true>() const {
    return GenericForwardIterator<T, true>(m_ptr);
  }

 private:
  pointer m_ptr;
  // give iterator access to const_iterator::m_ptr
  friend GenericForwardIterator<T, false>;
};
}  // namespace FMCA
#endif
