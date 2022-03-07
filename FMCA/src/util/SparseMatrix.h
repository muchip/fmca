#ifndef FMCA_UTIL_SPARSEMATRIX_H_
#define FMCA_UTIL_SPARSEMATRIX_H_

#include <Eigen/Dense>
#include <vector>

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
    Cell(value_type &&thevalue, index_type theindex)
        : value(thevalue), index(theindex) {}

    value_type value;
    index_type index;
  };
  typedef T value_type;
  typedef typename std::vector<std::vector<Cell<T>>>::size_type size_type;
  //////////////////////////////////////////////////////////////////////////////
  /*
   *  class constructors
   */
  SparseMatrix() : m_(0), n_(0) {}
  SparseMatrix(size_type m, size_type n) : m_(m), n_(n) { S_.resize(m_); }
  SparseMatrix(
      const Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> &M) {
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
    S.swap(S.S_);
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
  const std::vector<Cell<value_type>> &row(size_type i) const { return S_[i]; }
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
    for (auto j = 0; j < M.cols(); ++j)
      for (auto i = 0; i < m_; ++i)
        for (auto k = 0; k < S_[i].size(); ++k)
          retVal(i, j) += S_[i][k].value * M(S_[i][k].index, j);
    return retVal;
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
  std::vector<std::vector<Cell<value_type>>> S_;
  size_type m_;
  size_type n_;
};

} // namespace FMCA
#endif
