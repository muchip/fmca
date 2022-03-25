#ifndef FMCA_UTIL_SPARSEMATRIX_H_
#define FMCA_UTIL_SPARSEMATRIX_H_

#include <Eigen/Dense>
#include <vector>

#include "Macros.h"

namespace FMCA {
/*
 *  \brief Sparse matrix format which implements a special version of the
 *         compressed row storage (crs) format: Each row is stored in a
 *         sorted fashion with increasing indices. Since each row is
 *         represented by a standard vector, insertions and deletions are
 *         much faster than in the canonical csr format.
 *         The cost for reading/writing/inserting a particular entry A(i,j)
 *         is log_2(_Srows[i].size()).
 */

template <typename T> class SparseMatrix {
public:
  //////////////////////////////////////////////////////////////////////////////
  template <typename S> struct Cell {
    typedef S value_type;
    typedef typename std::vector<Cell<S>>::size_type index_type;
    Cell() = delete;
    Cell(const value_type &thevalue, index_type theindex)
        : value(thevalue), index(theindex) {}
    Cell(value_type &&thevalue, index_type theindex)
        : value(thevalue), index(theindex) {}

    value_type value;
    index_type index;
  };
  typedef T value_type;
  typedef typename std::vector<Cell<T>> SparseVector;
  typedef typename std::vector<SparseVector>::size_type size_type;
  //////////////////////////////////////////////////////////////////////////////
  /*
   *  class constructors
   */
  SparseMatrix() : m_(0), n_(0) {}

  SparseMatrix(size_type m, size_type n) : m_(m), n_(n) { S_.resize(m_); }

  template <typename Derived>
  SparseMatrix(const Eigen::MatrixBase<Derived> &M) {
    resize(M.rows(), M.cols());
    for (auto j = 0; j < M.cols(); ++j)
      for (auto i = 0; i < M.rows(); ++i)
        if (M(i, j))
          insert(i, j) = M(i, j);
  }
  // move constructor
  SparseMatrix(SparseMatrix<value_type> &&S) {
    m_ = S.m_;
    n_ = S.n_;
    S_.swap(S.S_);
  }
  // deep copy constructor, exploits deep copy of std::vector
  SparseMatrix(const SparseMatrix<value_type> &S) {
    m_ = S.m_;
    n_ = S.n_;
    S_ = S.S_;
  }
  // assignment operator based on copy and swap idiom
  SparseMatrix<value_type> &operator=(SparseMatrix<value_type> S) {
    std::swap(m_, S.m_);
    std::swap(n_, S.n_);
    S_.swap(S.S_);
    return *this;
  }
  /*
   *  return row of the sparse matrix col is not implemented since this would
   *  be really painful...
   */
  const SparseVector &row(size_type i) const { return S_[i]; }
  /*
   *  returns number of rows
   */
  size_type rows() const { return m_; }
  /*
   *  returns number of columns
   */
  size_type cols() const { return n_; }
  /*
   *  resize current sparse matrix, data is maintained due to the
   *  properties of the std::vector class;
   */
  SparseMatrix<value_type> &resize(size_type m, size_type n) {
    m_ = m;
    n_ = n;
    S_.resize(m_);
    if (n_ == 0) {
      setZero();
    } else {
      for (auto &&it : S_)
        if (it.size() && (it.back()).index >= n_)
          it.erase(binarySearch(it, n), it.end());
    }
    return *this;
  }

  SparseMatrix<value_type> &setZero() {
    for (auto &&it : S_)
      it.clear();
    return *this;
  }

  SparseMatrix<value_type> &reset() {
    S_.clear();
    m_ = 0;
    n_ = 0;
    return *this;
  }
  /*
   *  function for insertion returns reference to element (i,j),
   *  which is created if it does not already exist
   */
  value_type &insert(size_type i, size_type j) {
    assert(i < m_ && j < n_ && "insert out of bounds");
    auto itCol = binarySearch(S_[i], j);
    if (itCol == S_[i].end() || itCol->index != j)
      itCol = S_[i].insert(itCol, Cell<value_type>(value_type(0), j));
    return itCol->value;
  }
  /*
   *  function for insertion returns reference to element (i,j),
   *  which is created if it does not already exist
   */
  const value_type &getVal(size_type i, size_type j) const {
    assert(i < m_ && j < n_ && "index out of bounds");
    auto itCol = binarySearch(S_[i], j);
    if (itCol == S_[i].end() || itCol->index != j)
      return value_type(0);
    else
      return itCol->value;
  }

  /*
   *  dummy reference function to make it look like EigenSparse
   */
  value_type &coeffRef(size_type i, size_type j) { return insert(i, j); }
  const value_type &coeffRef(size_type i, size_type j) const {
    return getVal(i, j);
  }

  const value_type &operator()(size_type i, size_type j) const {
    return coeffRef(i, j);
  }

  void symmetrize() {
    for (auto i = 0; i < S_.size(); ++i)
      for (auto j = 0; j < S_[i].size(); ++j) {
        value_type val = 0.5 * (S_[i][j].value + coeffRef(S_[i][j].index, i));
        S_[i][j].value = val;
        coeffRef(S_[i][j].index, i) = val;
      }
    return;
  }

  value_type &operator()(size_type i, size_type j) { return coeffRef(i, j); }
  /*
   *  return full matrix
   */
  Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> full() const {
    Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> retVal;
    retVal.resize(m_, n_);
    retVal.setZero();
    for (auto i = 0; i < m_; ++i)
      for (auto j = 0; j < S_[i].size(); ++j) {
        retVal(i, S_[i][j].index) = S_[i][j].value;
      }
    return retVal;
  }

  size_type nnz() const {
    size_type retval = 0;
    for (auto &&it : S_)
      retval += it.size();
    return retval;
  }

  template <typename Derived>
  void setFromTriplets(const Derived &begin, const Derived &end) {
    S_.clear();
    S_.resize(m_);
    for (auto it = begin; it != end; ++it)
      coeffRef(it->row(), it->col()) = it->value();
    return;
  }

  std::vector<Eigen::Triplet<value_type>> toTriplets() const {
    std::vector<Eigen::Triplet<value_type>> triplets;
    for (auto i = 0; i < S_.size(); ++i)
      for (auto &&j : S_[i])
        triplets.push_back(Eigen::Triplet<value_type>(i, j.index, j.value));
    return triplets;
  }
  /*
   *  multiply sparse matrix with a dense matrix
   */
  Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>
  operator*(const Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> &M)
      const {
    eigen_assert(cols() == M.rows());
    Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> retVal;
    retVal.resize(rows(), M.cols());
    retVal.setZero();
#pragma omp parallel for
    for (auto j = 0; j < M.cols(); ++j)
      for (auto i = 0; i < m_; ++i)
        for (auto k = 0; k < S_[i].size(); ++k)
          retVal(i, j) += S_[i][k].value * M(S_[i][k].index, j);
    return retVal;
  }

  SparseMatrix<value_type> operator*(const SparseMatrix<value_type> &M) const {
    eigen_assert(cols() == M.rows());
    const size_type ssize = S_.size();
    const size_type msize = M.S_.size();
    SparseMatrix<value_type> retval(rows(), M.cols());
#pragma omp parallel for
    for (auto i = 0; i < ssize; ++i) {
      if (S_[i].size())
        for (auto j = 0; j < msize; ++j) {
          value_type entry = 0;
          if (M.S_[j].size()) {
            entry = dotProduct(S_[i], M.S_[j]);
            if (abs(entry) > 0)
              retval(i, j) = entry;
          }
        }
    }
    return retval;
  }

  // careful, this is a formated matrix product
  SparseMatrix<value_type> formatted_mult(const SparseMatrix<value_type> &M) {
    eigen_assert(cols() == M.rows());
    SparseMatrix<value_type> temp = *this;
    const size_type ssize = S_.size();
#pragma omp parallel for
    for (auto i = 0; i < ssize; ++i)
      for (auto j = 0; j < temp.S_[i].size(); ++j)
        temp.S_[i][j].value = dotProduct(S_[i], M.S_[S_[i][j].index]);
    return temp;
  }

  SparseMatrix<value_type> &operator+=(const SparseMatrix<value_type> &M) {
    eigen_assert(rows() == M.rows() && cols() == M.cols() &&
                 "dimension mismatch");
    const size_type msize = M.S_.size();
#pragma omp parallel for
    for (auto i = 0; i < msize; ++i)
      S_[i] = sparse_vector_addition(S_[i], M.S_[i]);
    return *this;
  }

  SparseMatrix<value_type> &operator-=(const SparseMatrix<value_type> &M) {
    eigen_assert(rows() == M.rows() && cols() == M.cols() &&
                 "dimension mismatch");
    const size_type msize = M.S_.size();
#pragma omp parallel for
    for (auto i = 0; i < msize; ++i)
      S_[i] = sparse_vector_subtraction(S_[i], M.S_[i]);
    return *this;
  }

  SparseMatrix<value_type> operator+(const SparseMatrix<value_type> &M) {
    eigen_assert(rows() == M.rows() && cols() == M.cols() &&
                 "dimension mismatch");
    const size_type msize = M.S_.size();
    SparseMatrix<value_type> temp = *this;
    temp += M;
    return temp;
  }

  SparseMatrix<value_type> operator-(const SparseMatrix<value_type> &M) {
    eigen_assert(rows() == M.rows() && cols() == M.cols() &&
                 "dimension mismatch");
    const size_type msize = M.S_.size();
    SparseMatrix<value_type> temp = *this;
    temp -= M;
    return temp;
  }

  static value_type dotProduct(const SparseVector &v1, const SparseVector &v2) {
    const size_type v1size = v1.size();
    const size_type v2size = v2.size();
    value_type retval = 0;
    if (v2.back().index < v1.front().index ||
        v1.back().index < v2.front().index)
      return retval;
    for (auto i = 0, j = 0; i < v1size && j < v2size;) {
      if (v1[i].index < v2[j].index)
        ++i;
      else if (v1[i].index > v2[j].index)
        ++j;
      else {
        retval += v1[i].value * v2[j].value;
        ++i;
        ++j;
      }
    }
    return retval;
  }

  static SparseVector sparse_vector_addition(const SparseVector &v1,
                                             const SparseVector &v2) {
    const size_type v1size = v1.size();
    const size_type v2size = v2.size();
    SparseVector retval;
    retval.reserve(v1size + v2size);
    size_type i = 0;
    size_type j = 0;
    while (i < v1size && j < v2size)
      if (v1[i].index < v2[j].index) {
        retval.push_back(Cell<value_type>(v1[i].value, v1[i].index));
        ++i;
      } else if (v1[i].index > v2[j].index) {
        retval.push_back(Cell<value_type>(v2[j].value, v2[j].index));
        ++j;
      } else {
        retval.push_back(Cell<value_type>(v1[i].value, v1[i].index));
        retval.back().value += v2[j].value;
        ++i;
        ++j;
      }
    if (i == v1size)
      retval.insert(retval.end(), v2.begin() + j, v2.end());
    else
      retval.insert(retval.end(), v1.begin() + i, v1.end());
    retval.shrink_to_fit();
    return retval;
  }

  static SparseVector sparse_vector_subtraction(const SparseVector &v1,
                                                const SparseVector &v2) {
    const size_type v1size = v1.size();
    const size_type v2size = v2.size();
    SparseVector retval;
    retval.reserve(v1size + v2size);
    size_type i = 0;
    size_type j = 0;
    while (i < v1size && j < v2size)
      if (v1[i].index < v2[j].index) {
        retval.push_back(Cell<value_type>(v1[i].value, v1[i].index));
        ++i;
      } else if (v1[i].index > v2[j].index) {
        retval.push_back(Cell<value_type>(-v2[j].value, v2[j].index));
        ++j;
      } else {
        retval.push_back(Cell<value_type>(v1[i].value, v1[i].index));
        retval.back().value -= v2[j].value;
        ++i;
        ++j;
      }
    if (i == v1size)
      while (j < v2size) {
        retval.push_back(Cell<value_type>(-v2[j].value, v2[j].index));
        ++j;
      }

    else
      while (i < v1size) {
        retval.push_back(Cell<value_type>(v1[i].value, v1[i].index));
        ++i;
      }
    retval.shrink_to_fit();
    return retval;
  }

private:
  /*
   *  performs a binary search for the ind array and returns iterators
   *  to the respective position j if the element is present or to j + 1 if
   *  it is not present (implementation is similar to std::lower_bound, cf.
   *  cppreference.com)
   */
  template <class S>
  typename S::const_iterator binarySearch(const S &row,
                                          typename S::size_type j) const {
    typename S::const_iterator itCol = row.begin();
    typename S::const_iterator it = itCol;
    auto dist = std::distance(row.begin(), row.end());
    auto step = dist;
    while (dist > 0) {
      it = itCol;
      step = dist / 2;
      std::advance(it, step);
      if (it->index < j) {
        itCol = ++it;
        dist -= step + 1;
      } else
        dist = step;
    }
    return itCol;
  }

  template <class S>
  typename S::iterator binarySearch(S &row, typename S::size_type j) {
    typename S::iterator itCol = row.begin();
    typename S::iterator it = itCol;
    auto dist = std::distance(row.begin(), row.end());
    auto step = dist;
    while (dist > 0) {
      it = itCol;
      step = dist / 2;
      std::advance(it, step);
      if (it->index < j) {
        itCol = ++it;
        dist -= step + 1;
      } else
        dist = step;
    }
    return itCol;
  }

  /*
   *  private member variables
   */
  std::vector<SparseVector> S_;
  size_type m_;
  size_type n_;
};

} // namespace FMCA
#endif
