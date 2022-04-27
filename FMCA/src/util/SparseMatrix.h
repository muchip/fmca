// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
#ifndef FMCA_UTIL_SPARSEMATRIX_H_
#define FMCA_UTIL_SPARSEMATRIX_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <numeric>
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

template <typename T>
class SparseMatrix {
 public:
  //////////////////////////////////////////////////////////////////////////////
  typedef T value_type;
  typedef typename std::vector<value_type> value_vector;
  typedef typename value_vector::size_type size_type;
  typedef typename std::vector<size_type> index_vector;
  typedef typename Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>
      eigenMatrix;
  //////////////////////////////////////////////////////////////////////////////
  /*
   *  class constructors
   */
  SparseMatrix() : m_(0), n_(0) {}

  SparseMatrix(size_type m, size_type n) : m_(m), n_(n) {
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
  SparseMatrix(SparseMatrix<value_type> &&S) {
    m_ = S.m_;
    n_ = S.n_;
    val_.swap(S.val_);
    idx_.swap(S.idx_);
  }
  // deep copy constructor, exploits deep copy of std::vector
  SparseMatrix(const SparseMatrix<value_type> &S) {
    m_ = S.m_;
    n_ = S.n_;
    val_ = S.val_;
    idx_ = S.idx_;
  }
  // assignment operator based on copy and swap idiom
  SparseMatrix<value_type> &operator=(SparseMatrix<value_type> S) {
    std::swap(m_, S.m_);
    std::swap(n_, S.n_);
    val_.swap(S.val_);
    idx_.swap(S.idx_);
    return *this;
  }
  //////////////////////////////////////////////////////////////////////////////
  size_type rows() const { return m_; }

  size_type cols() const { return n_; }
  /*
   *  resize current sparse matrix, data is maintained due to the
   *  properties of the std::vector class;
   */
  SparseMatrix<value_type> &resize(size_type m, size_type n) {
    m_ = m;
    n_ = n;
    val_.resize(m_);
    idx_.resize(m_);
    if (n_ == 0) {
      setZero();
    } else {
      for (auto i = 0; i < m_; ++i)
        if (idx_[i].size() && idx_[i].back() >= n_) {
          const size_type pos = binarySearch(idx_[i], n);
          idx_[i].erase(idx_[i].begin() + pos, idx_[i].end());
          val_[i].erase(val_[i].begin() + pos, val_[i].end());
        }
    }
    return *this;
  }

  SparseMatrix<value_type> &setZero() {
    for (auto i = 0; i < m_; ++i) {
      val_[i].clear();
      idx_[i].clear();
    }
    return *this;
  }

  SparseMatrix<value_type> &setIdentity() {
    setZero();
    const size_type dlength = m_ > n_ ? n_ : m_;
    for (auto i = 0; i < dlength; ++i) coeffRef(i, i) = 1;
    return *this;
  }

  SparseMatrix<value_type> &reset() {
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
  value_type &insert(size_type i, size_type j) {
    assert(i < m_ && j < n_ && "insert out of bounds");
    const size_type pos = binarySearch(idx_[i], j);
    if (pos == idx_[i].size() || idx_[i][pos] != j) {
      val_[i].insert(val_[i].begin() + pos, 0);
      idx_[i].insert(idx_[i].begin() + pos, j);
    }
    return val_[i][pos];
  }
  /*
   *  function for read only returns value of element (i,j)
   */
  const value_type getVal(size_type i, size_type j) const {
    assert(i < m_ && j < n_ && "index out of bounds");
    const size_type pos = binarySearch(idx_[i], j);
    if (pos == idx_[i].size() || idx_[i][pos] != j)
      return value_type(0);
    else
      return val_[i][pos];
  }

  /*
   *  reference function to make it look like EigenSparse
   */
  value_type &coeffRef(size_type i, size_type j) { return insert(i, j); }
  const value_type coeffRef(size_type i, size_type j) const {
    return getVal(i, j);
  }

  const value_type operator()(size_type i, size_type j) const {
    return coeffRef(i, j);
  }

  value_type &operator()(size_type i, size_type j) { return coeffRef(i, j); }

  /*
   *  return full matrix
   */
  eigenMatrix full() const {
    eigenMatrix retVal;
    retVal.resize(m_, n_);
    retVal.setZero();
    for (auto i = 0; i < m_; ++i)
      for (auto j = 0; j < idx_[i].size(); ++j)
        retVal(i, idx_[i][j]) = val_[i][j];

    return retVal;
  }

  size_type nnz() const {
    size_type retval = 0;
    for (auto &&it : idx_) retval += it.size();
    return retval;
  }

  template <typename Derived>
  SparseMatrix<value_type> &setFromTriplets(const Derived &begin,
                                            const Derived &end) {
    // first sort the triplets in row major manner lexicograpphically
    const size_type n_triplets = std::distance(begin, end);
    std::vector<size_type> idcs(n_triplets);
    {
      struct customLess {
        customLess(const Derived &begin, const Derived &end)
            : begin_(begin), end_(end) {}
        bool operator()(size_type a, size_type b) const {
          if ((begin_ + a)->row() == (begin_ + b)->row())
            return ((begin_ + a)->col() < (begin_ + b)->col());
          else
            return ((begin_ + a)->row() < (begin_ + b)->row());
        }
        const Derived &begin_;
        const Derived &end_;
      };
      std::iota(idcs.begin(), idcs.end(), 0);
      std::sort(idcs.begin(), idcs.end(), customLess(begin, end));
    }
    // next get the row sizes and row begin/end like in crs
    std::vector<size_type> rows(m_ + 1, 0);
    {
      rows[(begin + idcs[0])->row()] = 0;
      size_type j = 0;
      for (auto i = (begin + idcs[0])->row() + 1; i <= m_; ++i) {
        while (j < n_triplets && i - 1 == (begin + idcs[j])->row()) ++j;
        rows[i] = j;
      }
    }
    val_.clear();
    idx_.clear();
    val_.resize(m_);
    idx_.resize(m_);
    for (auto i = 0; i < m_; ++i) {
      const size_type sze = rows[i + 1] - rows[i];
      val_[i].resize(sze);
      idx_[i].resize(sze);
      for (auto j = 0; j < sze; ++j) {
        idx_[i][j] = (begin + idcs[rows[i] + j])->col();
        val_[i][j] = (begin + idcs[rows[i] + j])->value();
      }
    }
    return *this;
  }

  std::vector<Eigen::Triplet<value_type>> toTriplets() const {
    std::vector<Eigen::Triplet<value_type>> triplets;
    for (auto i = 0; i < m_; ++i)
      for (auto j = 0; j < idx_[i].size(); ++j)
        triplets.push_back(
            Eigen::Triplet<value_type>(i, idx_[i][j], val_[i][j]));
    return triplets;
  }

  template <typename Derived>
  SparseMatrix<value_type> &setDiagonal(const Derived &diag) {
    SparseMatrix(diag.size(), diag.size());
    setZero();
    for (auto i = 0; i < m_; ++i) coeffRef(i, i) = diag[i];
    return *this;
  }

  template <typename Derived>
  SparseMatrix<value_type> &setPermutation(const Derived &perm) {
    SparseMatrix(perm.size(), perm.size());
    setZero();
    for (auto i = 0; i < m_; ++i) coeffRef(i, perm[i]) = 1;
    return *this;
  }
  //////////////////////////////////////////////////////////////////////////////
  // parallel methods
  SparseMatrix<value_type> &symmetrize() {
    // first sweep catches all entries below the diagonal
    for (auto i = 0; i < m_; ++i) {
      for (auto j = 0; j < idx_[i].size(); ++j) {
        if (idx_[i][j] < i) {
          {
            value_type &other_val = coeffRef(idx_[i][j], i);
            value_type val = 0.5 * (val_[i][j] + other_val);
            val_[i][j] = val;
            other_val = val;
          }
        } else
          break;
      }
    }
    // second sweep catches all entries above the diagonal
    // that have not been found so far
    for (auto i = 0; i < m_; ++i) {
      for (int j = idx_[i].size() - 1; j >= 0; --j) {
        if (idx_[i][j] > i) {
          const size_type pos = binarySearch(idx_[idx_[i][j]], i);
          if (pos == idx_[idx_[i][j]].size() || idx_[idx_[i][j]][pos] != i) {
            value_type &other_val = coeffRef(idx_[i][j], i);
            value_type val = 0.5 * (val_[i][j] + other_val);
            val_[i][j] = val;
            other_val = val;
          }
        } else
          break;
      }
    }
    return *this;
  }

  /**
   *  transposition symmetrizes the pattern, i.e. additional zeros are
   *  introduced if either a_ij or a_ji exists and the other does not.
   **/
  SparseMatrix<value_type> &transpose() {
    // first sweep catches all entries below the diagonal
    for (auto i = 0; i < m_; ++i)
      for (auto j = 0; j < idx_[i].size(); ++j) {
        if (idx_[i][j] < i) {
          value_type &other_val = coeffRef(idx_[i][j], i);
          std::swap(val_[i][j], other_val);
        } else
          break;
      }
    // second sweep catches all entries above the diagonal
    // that have not been found so far
    for (auto i = 0; i < m_; ++i) {
      for (int j = idx_[i].size() - 1; j >= 0; --j) {
        if (idx_[i][j] > i) {
          const size_type pos = binarySearch(idx_[idx_[i][j]], i);
          if (pos == idx_[idx_[i][j]].size() || idx_[idx_[i][j]][pos] != i) {
            value_type &other_val = coeffRef(idx_[i][j], i);
            std::swap(val_[i][j], other_val);
          }
        } else
          break;
      }
    }
    return *this;
  }

  SparseMatrix<value_type> &compress(value_type threshold) {
    SparseMatrix<value_type> temp = *this;
#pragma omp parallel for
    for (auto i = 0; i < m_; ++i) {
      size_type k = 0;
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
  eigenMatrix operator*(const eigenMatrix &M) const {
    eigen_assert(cols() == M.rows());
    Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> retVal;
    retVal.resize(rows(), M.cols());
    retVal.setZero();
#pragma omp parallel for
    for (auto j = 0; j < M.cols(); ++j)
      for (auto i = 0; i < m_; ++i)
        for (auto k = 0; k < idx_[i].size(); ++k)
          retVal(i, j) += val_[i][k] * M(idx_[i][k], j);
    return retVal;
  }

  SparseMatrix<value_type> operator*(const SparseMatrix<value_type> &M) const {
    eigen_assert(cols() == M.rows());
    SparseMatrix<value_type> retval(m_, M.n_);
#pragma omp parallel for
    for (auto i = 0; i < m_; ++i)
      for (auto j = 0; j < idx_[i].size(); ++j)
        axpy(val_[i][j], &(retval.idx_[i]), &(retval.val_[i]),
             M.idx_[idx_[i][j]], M.val_[idx_[i][j]]);
    return retval;
  }

  static SparseMatrix<value_type> formatted_ABT(
      const SparseMatrix<value_type> &P, const SparseMatrix<value_type> &M1,
      const SparseMatrix<value_type> &M2) {
    SparseMatrix<value_type> retval = P;
#pragma omp parallel for
    for (auto i = 0; i < retval.m_; ++i)
      for (auto j = 0; j < retval.idx_[i].size(); ++j)
        retval.val_[i][j] =
            dotProduct(M1.idx_[i], M1.val_[i], M2.idx_[retval.idx_[i][j]],
                       M2.val_[retval.idx_[i][j]]);
    return retval;
  }

  static SparseMatrix<value_type> formatted_BABT(
      const SparseMatrix<value_type> &P, const SparseMatrix<value_type> &A,
      const SparseMatrix<value_type> &B) {
    SparseMatrix<value_type> retval = P;
#pragma omp parallel for
    for (auto i = 0; i < retval.m_; ++i) {
      if (retval.idx_[i].size()) {
        index_vector temp_idx;
        value_vector temp_val;
        for (auto k = 0; k < B.idx_[i].size(); ++k)
          axpy(B.val_[i][k], &(temp_idx), &(temp_val), A.idx_[B.idx_[i][k]],
               A.val_[B.idx_[i][k]]);
        for (auto j = 0; j < retval.idx_[i].size(); ++j) {
          if (B.idx_[retval.idx_[i][j]].size()) {
            retval.val_[i][j] =
                dotProduct(temp_idx, temp_val, B.idx_[retval.idx_[i][j]],
                           B.val_[retval.idx_[i][j]]);
          } else
            retval.val_[i][j] = 0;
        }
      }
    }
    return retval;
  }

  static SparseMatrix<value_type> formatted_BABT_sym(
      const SparseMatrix<value_type> &P, const SparseMatrix<value_type> &A,
      const SparseMatrix<value_type> &B) {
    SparseMatrix<value_type> retval = P;
#pragma omp parallel for
    for (auto i = 0; i < retval.m_; ++i) {
      if (retval.idx_[i].size()) {
        index_vector temp_idx;
        value_vector temp_val;
        // if the i-th row has entries, we compute B(i,:) * A
        for (auto k = 0; k < B.idx_[i].size(); ++k)
          axpy(B.val_[i][k], &(temp_idx), &(temp_val), A.idx_[B.idx_[i][k]],
               A.val_[B.idx_[i][k]]);
        // next, we compute B(i,:) * A * B(j,:)' for j <= i
        for (auto j = 0; j < retval.idx_[i].size(); ++j) {
          if (retval.idx_[i][j] <= i) {
            if (B.idx_[retval.idx_[i][j]].size()) {
              retval.val_[i][j] =
                  dotProduct(temp_idx, temp_val, B.idx_[retval.idx_[i][j]],
                             B.val_[retval.idx_[i][j]]);
            } else
              retval.val_[i][j] = 0;
          }
        }
      }
    }
    // second sweep catches all entries above the diagonal
    // that have not been found so far
#pragma omp parallel for
    for (auto i = 0; i < retval.m_; ++i) {
      for (int j = retval.idx_[i].size() - 1; j >= 0; --j) {
        if (retval.idx_[i][j] > i)
          retval.val_[i][j] = retval(retval.idx_[i][j], i);
        else
          break;
      }
    }
    return retval;
  }

  SparseMatrix<value_type> &operator+=(const SparseMatrix<value_type> &M) {
    eigen_assert(rows() == M.rows() && cols() == M.cols() &&
                 "dimension mismatch");
#pragma omp parallel for
    for (auto i = 0; i < m_; ++i)
      axpy(1, &(idx_[i]), &(val_[i]), M.idx_[i], M.val_[i]);
    return *this;
  }

  SparseMatrix<value_type> &operator-=(const SparseMatrix<value_type> &M) {
    eigen_assert(rows() == M.rows() && cols() == M.cols() &&
                 "dimension mismatch");
#pragma omp parallel for
    for (auto i = 0; i < m_; ++i)
      axpy(-1, &(idx_[i]), &(val_[i]), M.idx_[i], M.val_[i]);
    return *this;
  }

  SparseMatrix<value_type> operator+(const SparseMatrix<value_type> &M) {
    SparseMatrix<value_type> retval = *this;
    retval += M;
    return retval;
  }

  static SparseMatrix<value_type> gaxpy(value_type a,
                                        const SparseMatrix<value_type> &X,
                                        const SparseMatrix<value_type> &Y) {
    eigen_assert(X.rows() == Y.rows() && X.cols() == Y.cols() &&
                 "dimension mismatch");
    SparseMatrix<value_type> retval = Y;
#pragma omp parallel for
    for (auto i = 0; i < retval.m_; ++i)
      axpy(a, &(retval.idx_[i]), &(retval.val_[i]), X.idx_[i], X.val_[i]);
    return retval;
  }

  SparseMatrix<value_type> &scale(value_type a) {
#pragma omp parallel for
    for (auto i = 0; i < m_; ++i)
      for (auto &&it : val_[i]) it *= a;
    return *this;
  }

  SparseMatrix<value_type> operator-(const SparseMatrix<value_type> &M) {
    SparseMatrix<value_type> retval = *this;
    retval -= M;
    return retval;
  }

  SparseMatrix<value_type> &setBlock(const SparseMatrix<value_type> &M,
                                     size_type row, size_type col,
                                     size_type nrows, size_type ncols) {
#pragma omp parallel for
    for (auto i = row; i < row + nrows; ++i)
      for (auto j = 0; j < M.idx_[i].size(); ++j)
        if (M.idx_[i][j] >= col) {
          if (M.idx_[i][j] < col + ncols)
            coeffRef(i, M.idx_[i][j]) = M.val_[i][j];
          else
            break;
        }
    return *this;
  }

  value_type norm() const {
    value_type retval = 0;
    for (auto i = 0; i < m_; ++i)
      for (auto j = 0; j < idx_[i].size(); ++j)
        retval += val_[i][j] * val_[i][j];
    return sqrt(retval);
  }
  template <typename Derived>
  SparseMatrix<value_type> &setSparseRow(size_type row, const Derived &vec) {
    for (auto i = 0; i < idx_[row].size(); ++i)
      val_[row][i] = vec[idx_[row][i]];
    return *this;
  }
  //////////////////////////////////////////////////////////////////////////////
  // low level linear algebra (serial)
  static value_type dotProduct(const index_vector &iv1, const value_vector &vv1,
                               const index_vector &iv2,
                               const value_vector &vv2) {
    const size_type v1size = iv1.size();
    const size_type v2size = iv2.size();
    value_type retval = 0;
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

  static void axpy(value_type a, index_vector *iv1, value_vector *vv1,
                   const index_vector &iv2, const value_vector &vv2) {
    const size_type v1size = iv1->size();
    const size_type v2size = iv2.size();
    index_vector iretval(v1size + v2size);
    value_vector vretval(v1size + v2size);
    size_type i = 0;
    size_type j = 0;
    size_type k = 0;
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
  static size_type binarySearch(const index_vector &row, size_type j) {
    size_type pos = 0;
    size_type i = 0;
    size_type dist = row.size();
    size_type step = dist;
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
  size_type m_;
  size_type n_;
};

}  // namespace FMCA
#endif
