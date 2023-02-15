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
#ifndef FMCA_UTIL_SPARSEMATRIX_H_
#define FMCA_UTIL_SPARSEMATRIX_H_

#include <algorithm>
#include <numeric>
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
class SparseMatrix {
 public:
  //////////////////////////////////////////////////////////////////////////////
  typedef typename std::vector<Scalar> value_vector;
  typedef typename std::vector<Index> index_vector;

  template <typename Triplet>
  static void sortTripletsInPlace(std::vector<Triplet> &trips) {
    struct customLess {
      bool operator()(const Triplet &a, const Triplet &b) const {
        if (a.row() == b.row())
          return a.col() < b.col();
        else
          return a.row() < b.row();
      }
    };
    std::sort(trips.begin(), trips.end(), customLess());
    return;
  }

  template <typename Derived, typename Triplet>
  static Matrix symTripletsTimesVector(const std::vector<Triplet> &trips,
                                       const Eigen::MatrixBase<Derived> &x) {
    Matrix y(x.rows(), x.cols());
    y.setZero();
    for (const auto &i : trips) {
      y.row(i.row()) += i.value() * x.row(i.col());
      if (i.row() != i.col()) y.row(i.col()) += i.value() * x.row(i.row());
    }
    return y;
  }

  //////////////////////////////////////////////////////////////////////////////
  /*
   *  class constructors
   */
  SparseMatrix() : m_(0), n_(0) {}

  SparseMatrix(Index m, Index n) : m_(m), n_(n) {
    val_.resize(m_);
    idx_.resize(m_);
  }

  template <typename Derived>
  SparseMatrix(const Eigen::MatrixBase<Derived> &M) {
    resize(M.rows(), M.cols());
    for (auto j = 0; j < M.cols(); ++j)
      for (auto i = 0; i < M.rows(); ++i)
        if (M(i, j)) insert(i, j) = M(i, j);
  }

  // move constructor
  SparseMatrix(SparseMatrix &&S) {
    m_ = S.m_;
    n_ = S.n_;
    val_.swap(S.val_);
    idx_.swap(S.idx_);
  }

  // deep copy constructor, exploits deep copy of std::vector
  SparseMatrix(const SparseMatrix &S) {
    m_ = S.m_;
    n_ = S.n_;
    val_ = S.val_;
    idx_ = S.idx_;
  }
  // assignment operator based on copy and swap idiom
  SparseMatrix &operator=(SparseMatrix S) {
    std::swap(m_, S.m_);
    std::swap(n_, S.n_);
    val_.swap(S.val_);
    idx_.swap(S.idx_);
    return *this;
  }
  //////////////////////////////////////////////////////////////////////////////
  Index rows() const { return m_; }

  Index cols() const { return n_; }

  const std::vector<value_vector> &val() const { return val_; };
  const std::vector<index_vector> &idx() const { return idx_; };

  std::vector<value_vector> &val() { return val_; };
  std::vector<index_vector> &idx() { return idx_; };
  /*
   *  resize current sparse matrix, data is maintained due to the
   *  properties of the std::vector class;
   */
  SparseMatrix &resize(Index m, Index n) {
    m_ = m;
    n_ = n;
    val_.resize(m_);
    idx_.resize(m_);
    if (n_ == 0) {
      setZero();
    } else {
      for (auto i = 0; i < m_; ++i)
        if (idx_[i].size() && idx_[i].back() >= n_) {
          const Index pos = binarySearch(idx_[i], n);
          idx_[i].erase(idx_[i].begin() + pos, idx_[i].end());
          val_[i].erase(val_[i].begin() + pos, val_[i].end());
        }
    }
    return *this;
  }

  SparseMatrix &setZero() {
    for (auto i = 0; i < m_; ++i) {
      val_[i].clear();
      idx_[i].clear();
    }
    return *this;
  }

  SparseMatrix &setIdentity() {
    setZero();
    const Index dlength = m_ > n_ ? n_ : m_;
    for (auto i = 0; i < dlength; ++i) coeffRef(i, i) = 1;
    return *this;
  }

  SparseMatrix &reset() {
    val_.clear();
    idx_.clear();
    m_ = 0;
    n_ = 0;
    return *this;
  }

  /*
   *  function for insertion returns reference to element (i,j),
   *  which is created if it does not already exist
   */
  Index find(Index i, Index j) {
    assert(i < m_ && j < n_ && "find out of bounds");
    const Index pos = binarySearch(idx_[i], j);
    if (pos < idx_[i].size() && idx_[i][pos] == j)
      return pos;
    else
      return idx_[i].size();
  }

  /*
   *  function for insertion returns reference to element (i,j),
   *  which is created if it does not already exist
   */
  Scalar &insert(Index i, Index j) {
    assert(i < m_ && j < n_ && "insert out of bounds");
    const Index pos = binarySearch(idx_[i], j);
    if (pos == idx_[i].size() || idx_[i][pos] != j) {
      val_[i].insert(val_[i].begin() + pos, Scalar());
      idx_[i].insert(idx_[i].begin() + pos, j);
    }
    return val_[i][pos];
  }
  /*
   *  function for read only returns value of element (i,j)
   */
  const Scalar getVal(Index i, Index j) const {
    assert(i < m_ && j < n_ && "index out of bounds");
    const Index pos = binarySearch(idx_[i], j);
    if (pos == idx_[i].size() || idx_[i][pos] != j)
      return Scalar(0);
    else
      return val_[i][pos];
  }

  /*
   *  reference function to make it look like EigenSparse
   */
  Scalar &coeffRef(Index i, Index j) { return insert(i, j); }
  const Scalar coeffRef(Index i, Index j) const { return getVal(i, j); }

  const Scalar operator()(Index i, Index j) const { return coeffRef(i, j); }

  Scalar &operator()(Index i, Index j) { return coeffRef(i, j); }

  /*
   *  return full matrix
   */
  Matrix full() const {
    Matrix retVal;
    retVal.resize(m_, n_);
    retVal.setZero();
    for (auto i = 0; i < m_; ++i)
      for (auto j = 0; j < idx_[i].size(); ++j)
        retVal(i, idx_[i][j]) = val_[i][j];

    return retVal;
  }

  Index nnz() const {
    Index retval = 0;
    for (auto &&it : idx_) retval += it.size();
    return retval;
  }

  /*
   * this method requires that the triplets are sorted using
   * SparseMatrix::sortTripletsInPlace
   */
  template <typename Derived>
  SparseMatrix &setFromTriplets(const Derived &begin, const Derived &end) {
    // first sort the triplets in row major manner lexicograpphically
    const Index n_triplets = std::distance(begin, end);
    // next get the row sizes and row begin/end like in crs
    std::vector<Index> rows(m_ + 1, 0);
    {
      rows[begin->row()] = 0;
      Index j = 0;
      for (auto i = begin->row() + 1; i <= m_; ++i) {
        while (j < n_triplets && i - 1 == (begin + j)->row()) ++j;
        rows[i] = j;
      }
    }
    val_.clear();
    idx_.clear();
    val_.resize(m_);
    idx_.resize(m_);
#pragma omp parallel for
    for (auto i = 0; i < m_; ++i) {
      const Index sze = rows[i + 1] - rows[i];
      val_[i].resize(sze);
      idx_[i].resize(sze);
      for (auto j = 0; j < sze; ++j) {
        idx_[i][j] = (begin + rows[i] + j)->col();
        val_[i][j] = (begin + rows[i] + j)->value();
      }
    }
    return *this;
  }

  template <typename ROWS, typename COLS, typename VALS>
  void setFromCRS(const ROWS &ia, const COLS &ja, const VALS &a, Index n) {
    m_ = n;
    n_ = n;
    val_.clear();
    idx_.clear();
    val_.resize(n);
    idx_.resize(n);
    for (auto i = 0; i < n; ++i) {
      val_[i].reserve(ia[i + 1] - ia[i]);
      idx_[i].reserve(ia[i + 1] - ia[i]);
      for (auto j = ia[i]; j < ia[i + 1]; ++j) {
        val_[i].push_back(a[j]);
        idx_[i].push_back(ja[j]);
      }
    }
    return;
  }

  template <typename Derived>
  SparseMatrix &setDiagonal(const Derived &diag) {
    SparseMatrix(diag.size(), diag.size());
    setZero();
    for (auto i = 0; i < m_; ++i) coeffRef(i, i) = diag[i];
    return *this;
  }

  template <typename Derived>
  SparseMatrix &setPermutation(const Derived &perm) {
    SparseMatrix(perm.size(), perm.size());
    setZero();
    for (auto i = 0; i < m_; ++i) coeffRef(i, perm[i]) = 1;
    return *this;
  }
  //////////////////////////////////////////////////////////////////////////////
  // parallel methods
  //////////////////////////////////////////////////////////////////////////////
  SparseMatrix &compress(Scalar threshold) {
    SparseMatrix temp = *this;
#pragma omp parallel for
    for (auto i = 0; i < m_; ++i) {
      Index k = 0;
      for (auto j = 0; j < idx_[i].size(); ++j)
        if (abs(val_[i][j]) > threshold) {
          temp.idx_[i][k] = idx_[i][j];
          temp.val_[i][k] = val_[i][j];
          ++k;
        }
      temp.idx_[i].resize(k);
      temp.val_[i].resize(k);
    }
    idx_.swap(temp.idx_);
    val_.swap(temp.val_);
    return *this;
  }

  /*
   *  multiply sparse matrix with a dense matrix
   */
  Matrix operator*(const Matrix &M) const {
    eigen_assert(cols() == M.rows());
    Matrix retVal(rows(), M.cols());
    retVal.setZero();
#pragma omp parallel for schedule( \
    dynamic,                       \
    (m_ + omp_get_num_threads() * 4 - 1) / (omp_get_num_threads() * 4))
    for (auto i = 0; i < m_; ++i)
      for (auto k = 0; k < idx_[i].size(); ++k)
        retVal.row(i) += val_[i][k] * M.row(idx_[i][k]);
    return retVal;
  }

  SparseMatrix &operator+=(const SparseMatrix &M) {
    eigen_assert(rows() == M.rows() && cols() == M.cols() &&
                 "dimension mismatch");
#pragma omp parallel for
    for (auto i = 0; i < m_; ++i)
      axpy(1, &(idx_[i]), &(val_[i]), M.idx_[i], M.val_[i]);
    return *this;
  }

  SparseMatrix &operator-=(const SparseMatrix &M) {
    eigen_assert(rows() == M.rows() && cols() == M.cols() &&
                 "dimension mismatch");
#pragma omp parallel for
    for (auto i = 0; i < m_; ++i)
      axpy(-1, &(idx_[i]), &(val_[i]), M.idx_[i], M.val_[i]);
    return *this;
  }

  SparseMatrix operator+(const SparseMatrix &M) {
    SparseMatrix retval = *this;
    retval += M;
    return retval;
  }

  static SparseMatrix gaxpy(Scalar a, const SparseMatrix &X,
                            const SparseMatrix &Y) {
    eigen_assert(X.rows() == Y.rows() && X.cols() == Y.cols() &&
                 "dimension mismatch");
    SparseMatrix retval = Y;
#pragma omp parallel for
    for (auto i = 0; i < retval.m_; ++i)
      axpy(a, &(retval.idx_[i]), &(retval.val_[i]), X.idx_[i], X.val_[i]);
    return retval;
  }

  SparseMatrix &scale(Scalar a) {
#pragma omp parallel for
    for (auto i = 0; i < m_; ++i)
      for (auto &&it : val_[i]) it *= a;
    return *this;
  }

  SparseMatrix operator-(const SparseMatrix &M) {
    SparseMatrix retval = *this;
    retval -= M;
    return retval;
  }

  Scalar norm() const {
    Scalar retval = 0;
    for (auto i = 0; i < m_; ++i)
      for (auto j = 0; j < idx_[i].size(); ++j)
        retval += val_[i][j] * val_[i][j];
    return sqrt(retval);
  }

  //////////////////////////////////////////////////////////////////////////////
  // low level linear algebra (serial)
  static Scalar dotProduct(const index_vector &iv1, const value_vector &vv1,
                           const index_vector &iv2, const value_vector &vv2) {
    const Index v1size = iv1.size();
    const Index v2size = iv2.size();
    Scalar retval = 0;
    if (!v1size || !v2size || iv2.back() < iv1.front() ||
        iv1.back() < iv2.front())
      return retval;
    auto j = 0;
    if (v1size < v2size)
      for (auto i = 0; i < v1size; ++i) {
        while (j < v2size && iv2[j] < iv1[i]) ++j;
        if (j >= v2size) break;
        if (iv2[j] == iv1[i]) retval += vv1[i] * vv2[j];
      }
    else
      for (auto i = 0; i < v2size; ++i) {
        while (j < v1size && iv1[j] < iv2[i]) ++j;
        if (j >= v1size) break;
        if (iv2[i] == iv1[j]) retval += vv1[j] * vv2[i];
      }

    return retval;
  }

  static void axpy(Scalar a, index_vector *iv1, value_vector *vv1,
                   const index_vector &iv2, const value_vector &vv2) {
    const Index v1size = iv1->size();
    const Index v2size = iv2.size();
    index_vector iretval(v1size + v2size);
    value_vector vretval(v1size + v2size);
    Index i = 0;
    Index j = 0;
    Index k = 0;
    while (i < v1size && j < v2size)
      if ((*iv1)[i] < iv2[j]) {
        iretval[k] = (*iv1)[i];
        vretval[k] = (*vv1)[i];
        ++i;
        ++k;
      } else if ((*iv1)[i] > iv2[j]) {
        iretval[k] = iv2[j];
        vretval[k] = a * vv2[j];
        ++j;
        ++k;
      } else {
        iretval[k] = (*iv1)[i];
        vretval[k] = (*vv1)[i] + a * vv2[j];
        ++i;
        ++j;
        ++k;
      }
    if (i == v1size) {
      for (; j < v2size; ++j) {
        iretval[k] = iv2[j];
        vretval[k] = a * vv2[j];
        ++k;
      }
    } else {
      for (; i < v1size; ++i) {
        iretval[k] = (*iv1)[i];
        vretval[k] = (*vv1)[i];
        ++k;
      }
    }
    iretval.resize(k);
    vretval.resize(k);
    iv1->swap(iretval);
    vv1->swap(vretval);
    return;
  }

  /*
   *  performs a binary search for the index array and returns the position
   *  j if the element is present or j + 1 if it is not present
   *  (implementation is similar to std::lower_bound, cf. cppreference.com)
   */
  static Index binarySearch(const index_vector &row, Index j) {
    Index pos = 0;
    Index i = 0;
    Index dist = row.size();
    Index step = dist;
    while (dist > 0) {
      i = pos;
      step = dist / 2;
      i += step;
      if (row[i] < j) {
        pos = ++i;
        dist -= step + 1;
      } else
        dist = step;
    }
    return pos;
  }

 private:
  /*
   *  private member variables
   */
  std::vector<value_vector> val_;
  std::vector<index_vector> idx_;
  Index m_;
  Index n_;
};

}  // namespace FMCA
#endif
