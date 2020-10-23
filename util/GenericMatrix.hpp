#ifndef HIERARCHICALMATRIX_GENERICMATRIX_H_
#define HIERARCHICALMATRIX_GENERICMATRIX_H_

#include <iostream>
#include <vector>

template <typename T>
class GenericMatrix {
 public:
  typedef typename std::vector<std::vector<T> >::size_type colIndex;
  typedef typename std::vector<T>::size_type rowIndex;
  //////////////////////////////////////////////////////////////////////////////
  //  constructors
  //////////////////////////////////////////////////////////////////////////////
  GenericMatrix() : rows_(0), cols_(0){};

  GenericMatrix(rowIndex rows, colIndex cols) { resize(rows, cols); }

  GenericMatrix(const GenericMatrix<T>& other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    m_data_ = other.m_data_;
  }

  GenericMatrix(GenericMatrix<T>&& other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    m_data_ = std::move(other.m_data_);
  }
  //////////////////////////////////////////////////////////////////////////////
  //  methods
  //////////////////////////////////////////////////////////////////////////////
  void resize(rowIndex rows, colIndex cols) {
    m_data_.resize(cols);
    for (auto it = m_data_.begin(); it != m_data_.end(); ++it) it->resize(rows);
    rows_ = rows;
    cols_ = cols;
    return;
  }

  colIndex cols() const { return cols_; }

  rowIndex rows() const { return rows_; }
  //////////////////////////////////////////////////////////////////////////////
  //  operators
  //////////////////////////////////////////////////////////////////////////////
  const T& operator()(rowIndex row, colIndex col) const {
    return m_data_[col][row];
  }

  T& operator()(rowIndex row, colIndex col) { return m_data_[col][row]; }

  GenericMatrix<T>& operator=(GenericMatrix<T> other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    std::swap(m_data_, other.m_data_);
    return *this;
  }

  GenericMatrix<T>& operator+=(const GenericMatrix<T>& other) {
    assert(cols_ == other.cols_ && rows_ == other.rows_ &&
           "dimension mismatch");
    auto itc1 = m_data_.begin();
    auto itc2 = other.m_data_.begin();
    for (; itc1 != m_data_.end() && itc2 != other.m_data_.end();
         ++itc1, ++itc2) {
      auto itr1 = itc1->begin();
      auto itr2 = itc2->begin();
      for (; itr1 != itc1->end() && itr2 != itc2->end(); ++itr1, ++itr2)
        *itr1 += *itr2;
    }
    return *this;
  }

  GenericMatrix<T> operator+(const GenericMatrix<T>& other) {
    GenericMatrix<T> retval(other);
    return retval += *this;
  }
  //////////////////////////////////////////////////////////////////////////////
  //  private members
  //////////////////////////////////////////////////////////////////////////////
 private:
  std::vector<std::vector<T> > m_data_;
  colIndex cols_;
  rowIndex rows_;
};

#endif
