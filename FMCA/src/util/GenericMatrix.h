// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_UTIL_GENERICMATRIX_H_
#define FMCA_UTIL_GENERICMATRIX_H_

#include <vector>

namespace FMCA {
template <typename T> class GenericMatrix {
public:
  typedef typename std::vector<T>::size_type colIndex;
  typedef typename std::vector<T>::size_type rowIndex;
  //////////////////////////////////////////////////////////////////////////////
  //  constructors
  //////////////////////////////////////////////////////////////////////////////
  GenericMatrix() : rows_(0), cols_(0){};

  GenericMatrix(rowIndex rows, colIndex cols) { resize(rows, cols); }

  GenericMatrix(const GenericMatrix &other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    m_data_ = other.m_data_;
  }

  GenericMatrix(GenericMatrix &&other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    m_data_ = std::move(other.m_data_);
  }
  //////////////////////////////////////////////////////////////////////////////
  //  methods
  //////////////////////////////////////////////////////////////////////////////
  void resize(rowIndex rows, colIndex cols) {
    m_data_.resize(rows * cols);
    rows_ = rows;
    cols_ = cols;
    return;
  }

  colIndex cols() const { return cols_; }

  rowIndex rows() const { return rows_; }

  rowIndex size() const { return rows_ * cols_; }
  const T &front() const { return m_data_.front(); }
  const T &back() const { return m_data_.back(); }
  T &back() { return m_data_.back(); }
  T &front() { return m_data_.front(); }
  typename std::vector<T>::const_iterator begin() const {
    return m_data_.cbegin();
  }
  typename std::vector<T>::const_iterator end() const { return m_data_.cend(); }
  //////////////////////////////////////////////////////////////////////////////
  //  operators
  //////////////////////////////////////////////////////////////////////////////
  const T &operator()(rowIndex row, colIndex col) const {
    return m_data_[col + cols_ * row];
  }

  T &operator()(rowIndex row, colIndex col) {
    return m_data_[col + cols_ * row];
  }

  GenericMatrix &operator=(GenericMatrix other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    std::swap(m_data_, other.m_data_);
    return *this;
  }
  //////////////////////////////////////////////////////////////////////////////
  //  private members
  //////////////////////////////////////////////////////////////////////////////
private:
  std::vector<T> m_data_;
  colIndex cols_;
  rowIndex rows_;
};
} // namespace FMCA
#endif
