#ifndef FMCA_FORMATTEDMULTIPLICATION_
#define FMCA_FORMATTEDMULTIPLICATION_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <unordered_map>

#include "../FMCA/src/util/Macros.h"

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, long long int> largeSparse;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor, long long int>
    largeSparse_col;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> Sparse;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor, int> Sparse_col;

// formatted_sparse_axpy performs an "axpy" operation, adding a scaled sparse
// vector to another sparse vector.
template <typename tIndex, typename tSize, typename tVal, typename sIndex,
          typename sSize, typename sVal>
inline void formatted_sparse_axpy(tIndex *tidx, const tSize tsze, tVal *target,
                                  const sIndex *sidx, const sSize ssze,
                                  const sVal *source, double a) {
  if (!tsze || !ssze || sidx[ssze - 1] < tidx[0] || tidx[tsze - 1] < sidx[0])
    return;
  double retval = 0;
  auto i = 0;
  auto j = 0;
  while (i < tsze && j < ssze) {
    if (sidx[j] == tidx[i]) {
      target[i] += a * source[j];
      ++i;
      ++j;
    } else if (sidx[j] < tidx[i]) {
      ++j;
    } else {
      ++i;
    }
  }
  return;
}

// sparse_dot_product2 calculates the dot product of two sparse vectors having
// non-zero values target and source at positions tidx and sidx
template <typename tIndex, typename tSize, typename tVal, typename sIndex,
          typename sSize, typename sVal>
inline double sparse_dot_product(tIndex *tidx, const tSize tsze, tVal *target,
                                 sIndex *sidx, const sSize ssze, sVal *source) {
  if (!tsze || !ssze || sidx[ssze - 1] < tidx[0] || tidx[tsze - 1] < sidx[0])
    return 0.;
  double retval = 0;
  auto i = 0;
  auto j = 0;
  while (i < tsze && j < ssze) {
    if (sidx[j] == tidx[i]) {
      retval += target[i] * source[j];
      ++i;
      ++j;
    } else if (sidx[j] < tidx[i]) {
      ++j;
    } else {
      ++i;
    }
  }
  return retval;
}

template <typename tIndex, typename tSize, typename tVal>
inline double sparse_dense_dot_product(tIndex *tidx, const tSize tsze,
                                       tVal *target,
                                       const Eigen::VectorXd &dense_vec) {
  double retval = 0;
  for (tSize i = 0; i < tsze; ++i) {
    const auto idx = tidx[i];
    retval += target[i] * dense_vec[idx];
  }
  return retval;
}

template <typename tIndex, typename tSize, typename tVal, typename sIndex,
          typename sSize, typename sVal>
inline double sparse_dot_product_omp(tIndex *tidx, const tSize tsze,
                                     tVal *target, sIndex *sidx,
                                     const sSize ssze, sVal *source) {
  if (!tsze || !ssze || sidx[ssze - 1] < tidx[0] || tidx[tsze - 1] < sidx[0])
    return 0.;
  double retval = 0.0;
  int initial_j = 0;
  int i = 0, j = 0;
// Parallel for loop with reduction clause
#pragma omp parallel for reduction(+ : retval) private(i, j)
  for (i = 0; i < tsze; ++i) {
    for (j = initial_j; j < ssze; ++j) {
      if (tidx[i] == sidx[j]) {
        retval += target[i] * source[j];
        initial_j = j + 1;
        break;  // We found the matching index, no need to continue inner loop
      }
    }
  }
  return retval;
}

// sparse_weighted_dot_product calculates the dot product of two sparse vectors
// having non-zero values target and source at positions tidx and sidx weighted
// by weights
template <typename tIndex, typename tSize, typename tVal, typename sIndex,
          typename sSize, typename sVal, typename wVal>
inline double sparse_weighted_dot_product(tIndex *tidx, const tSize tsze,
                                          tVal *target, sIndex *sidx,
                                          const sSize ssze, sVal *source,
                                          wVal *weights) {
  if (!tsze || !ssze || sidx[ssze - 1] < tidx[0] || tidx[tsze - 1] < sidx[0])
    return 0.;
  double retval = 0;
  auto i = 0;
  auto j = 0;
  while (i < tsze && j < ssze) {
    if (sidx[j] == tidx[i]) {
      retval += target[i] * source[j] * weights[sidx[j]];
      ++i;
      ++j;
    } else if (sidx[j] < tidx[i]) {
      ++j;
    } else {
      ++i;
    }
  }
  return retval;
}

template <typename tIndex, typename tSize, typename tVal, typename sIndex,
          typename sSize, typename sVal, typename uIndex, typename uSize,
          typename uVal>
double sparse_triple_product(tIndex *tidx, const tSize tsze, tVal *target,
                             sIndex *sidx, const sSize ssze, sVal *source,
                             uIndex *uidx, const uSize usize, uVal *util) {
  if (!tsze || !ssze || !usize) return 0.;
  double intermediate_result = 0.;
  double final_result = 0.;
  auto i = 0;
  auto j = 0;
  auto k = 0;
  // Compute sparse dot product for A*B
  while (i < tsze && j < ssze) {
    if (sidx[j] == tidx[i]) {
      intermediate_result += target[i] * source[j];
      ++i;
      ++j;
    } else if (sidx[j] < tidx[i]) {
      ++j;
    } else {
      ++i;
    }
  }
  // Multiply by corresponding element in C and accumulate
  while (k < usize) {
    final_result += intermediate_result * util[k];
    ++k;
  }
  return final_result;
}

#if 0
template <typename Index, typename Size>
  inline Size binarySearch(const Index *idx, const Size sze, const Size j) {
    Size pos = 0;
    Size i = 0;
    Size dist = sze;
    Size step = dist;
    while (dist > 0) {
      i = pos;
      step = dist / 2;
      i += step;
      if (idx[i] < j) {
        pos = ++i;
        dist -= step + 1;
      } else
        dist = step;
    }
    return pos;
  }
#else
template <typename Index, typename Size>
inline Size binarySearch(const Index *idx, const Size sze, const Size j) {
  Size low = 0;
  Size mid = 0;
  Size high = sze - 1;
  while (low < high) {
    mid = 0.5 * (low + high);
    if (idx[mid] < j)
      low = mid + 1;
    else if (idx[mid] > j)
      high = mid;
    else
      return mid;
  }
  return low;
}
#endif

template <typename tIndex, typename tSize, typename tVal, typename sIndex,
          typename sSize, typename sVal>
inline void formatted_sparse_axpy2(tIndex *tidx, const tSize tsze, tVal *target,
                                   const sIndex *sidx, const sSize ssze,
                                   const sVal *source, double a) {
  if (!tsze || !ssze || sidx[ssze - 1] < tidx[0] || tidx[tsze - 1] < sidx[0])
    return;
  auto j = 0;
  if (tsze < ssze) {
    if (ssze > 20 * tsze) {
      for (tSize i = 0; i < tsze; ++i) {
        j += binarySearch(sidx + j, ssze - j, tidx[i]);
        if (sidx[j] == tidx[i])
          target[i] += a * source[j];
        else if (sidx[j] < tidx[i])
          break;
      }
    } else {
      for (tSize i = 0; i < tsze; ++i) {
        while (j < ssze && sidx[j] < tidx[i]) ++j;
        if (j >= ssze)
          break;
        else if (sidx[j] == tidx[i])
          target[i] += a * source[j];
      }
    }
  } else {
    if (tsze > 20 * ssze) {
      for (sSize i = 0; i < ssze; ++i) {
        j += binarySearch(tidx + j, tsze - j, sidx[i]);
        if (sidx[i] == tidx[j])
          target[j] += a * source[i];
        else if (tidx[j] < sidx[i])
          break;
      }
    } else {
      for (sSize i = 0; i < ssze; ++i) {
        while (j < tsze && tidx[j] < sidx[i]) ++j;
        if (j >= tsze)
          break;
        else if (sidx[i] == tidx[j])
          target[j] += a * source[i];
      }
    }
  }
  return;
}

#if 1
// largeSparse &pattern: a reference to a largeSparse object, which will store
// the result of the multiplication const largeSparse &mat1: a constant
// reference to a largeSparse object, the first input matrix. const largeSparse
// &mat2: a constant reference to a largeSparse object, the second input matrix.
// double scal = 1: scaling factor with a default value of 1.
// row based matrix multiplication
void formatted_sparse_multiplication(largeSparse &pattern,
                                     const largeSparse &mat1,
                                     const largeSparse &mat2, double scal = 1) {
  eigen_assert(mat1.cols() == mat2.rows() &&
               "dimension mismatch");  // dimension check
  long long int n = pattern.rows();
  largeSparse::StorageIndex *ia = pattern.outerIndexPtr();
  largeSparse::StorageIndex *ja = pattern.innerIndexPtr();
  largeSparse::Scalar *a = pattern.valuePtr();
  for (int i = 0; i < pattern.nonZeros(); i++) {
    a[i] = 0;
  }
  const largeSparse::StorageIndex *ia2 = mat1.outerIndexPtr();
  const largeSparse::StorageIndex *ja2 = mat1.innerIndexPtr();
  const largeSparse::Scalar *a2 = mat1.valuePtr();
  const largeSparse::StorageIndex *ia3 = mat2.outerIndexPtr();
  const largeSparse::StorageIndex *ja3 = mat2.innerIndexPtr();
  const largeSparse::Scalar *a3 = mat2.valuePtr();

#pragma omp parallel for
  for (auto i = 0; i < n; ++i) {
    for (auto j = ia2[i]; j < ia2[i + 1]; ++j)
      if (std::abs(a2[j]) > 1e-15)
        formatted_sparse_axpy(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                              ja3 + ia3[ja2[j]], ia3[ja2[j] + 1] - ia3[ja2[j]],
                              a3 + ia3[ja2[j]], scal * a2[j]);
  }
  return;
}

// column based matrix multiplication
void formatted_sparse_multiplication_cols(largeSparse_col &pattern,
                                          const largeSparse_col &mat1,
                                          const largeSparse_col &mat2,
                                          double scal = 1) {
  eigen_assert(mat1.cols() == mat2.rows() &&
               "dimension mismatch");  // dimension check
  long long int n = pattern.cols();
  largeSparse::StorageIndex *ia = pattern.outerIndexPtr();
  largeSparse::StorageIndex *ja = pattern.innerIndexPtr();
  largeSparse::Scalar *a = pattern.valuePtr();
  for (int i = 0; i < pattern.nonZeros(); i++) {
    a[i] = 0;
  }
  const largeSparse::StorageIndex *ia2 = mat1.outerIndexPtr();
  const largeSparse::StorageIndex *ja2 = mat1.innerIndexPtr();
  const largeSparse::Scalar *a2 = mat1.valuePtr();
  const largeSparse::StorageIndex *ia3 = mat2.outerIndexPtr();
  const largeSparse::StorageIndex *ja3 = mat2.innerIndexPtr();
  const largeSparse::Scalar *a3 = mat2.valuePtr();

#pragma omp parallel for
  for (auto i = 0; i < n; ++i) {
    for (auto j = ia3[i]; j < ia3[i + 1]; ++j)
      if (std::abs(a2[j]) > 1e-15)
        formatted_sparse_axpy(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                              ja2 + ia2[ja3[j]], ia2[ja3[j] + 1] - ia2[ja3[j]],
                              a2 + ia2[ja3[j]], scal * a3[j]);
  }
  return;
}

// inner product base matrix multiplication
void formatted_sparse_multiplication_dotproduct(Sparse &pattern,
                                                const Sparse &mat1,
                                                const Sparse &mat2) {
  int n = pattern.rows();
  // eigen_assert(mat1.cols() == mat2.rows() && "dimension mismatch");
  Sparse::StorageIndex *ia = pattern.outerIndexPtr();
  Sparse::StorageIndex *ja = pattern.innerIndexPtr();
  Sparse::Scalar *a = pattern.valuePtr();
  for (int i = 0; i < pattern.nonZeros(); i++) {
    a[i] = 0;
  }
  const Sparse::StorageIndex *ia2 = mat1.outerIndexPtr();
  const Sparse::StorageIndex *ja2 = mat1.innerIndexPtr();
  const Sparse::Scalar *a2 = mat1.valuePtr();
  const Sparse_col::StorageIndex *ia3 = mat2.outerIndexPtr();
  const Sparse_col::StorageIndex *ja3 = mat2.innerIndexPtr();
  const Sparse_col::Scalar *a3 = mat2.valuePtr();
#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i < n; ++i) {
    for (auto j = ia[i]; j < ia[i + 1]; ++j)
      a[j] = sparse_dot_product(ja2 + ia2[i], ia2[i + 1] - ia2[i], a2 + ia2[i],
                                ja3 + ia3[ja[j]], ia3[ja[j] + 1] - ia3[ja[j]],
                                a3 + ia3[ja[j]]);
  }
  return;
}


void formatted_sparse_multiplication_dotproduct_largeSparse(largeSparse &pattern,
                                                const largeSparse &mat1,
                                                const largeSparse &mat2) {
  long long int n = pattern.rows();
  // eigen_assert(mat1.cols() == mat2.rows() && "dimension mismatch");
  largeSparse::StorageIndex *ia = pattern.outerIndexPtr();
  largeSparse::StorageIndex *ja = pattern.innerIndexPtr();
  largeSparse::Scalar *a = pattern.valuePtr();
  for (int i = 0; i < pattern.nonZeros(); i++) {
    a[i] = 0;
  }
  const largeSparse::StorageIndex *ia2 = mat1.outerIndexPtr();
  const largeSparse::StorageIndex *ja2 = mat1.innerIndexPtr();
  const largeSparse::Scalar *a2 = mat1.valuePtr();
  const largeSparse_col::StorageIndex *ia3 = mat2.outerIndexPtr();
  const largeSparse_col::StorageIndex *ja3 = mat2.innerIndexPtr();
  const largeSparse_col::Scalar *a3 = mat2.valuePtr();
#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i < n; ++i) {
    for (auto j = ia[i]; j < ia[i + 1]; ++j)
      a[j] = sparse_dot_product(ja2 + ia2[i], ia2[i + 1] - ia2[i], a2 + ia2[i],
                                ja3 + ia3[ja[j]], ia3[ja[j] + 1] - ia3[ja[j]],
                                a3 + ia3[ja[j]]);
  }
  return;
}


// This triple product was created to compute the integral of the stiffness matrix using kernels,
// so it aims to compute K * W * K^T. The result pattern will be symmetric. In case of
// generic triple product multiplication, pattern and mat3 has to be taken transpose --> col major)
void formatted_sparse_multiplication_triple_product(Sparse &pattern,
                                                    const Sparse &mat1,
                                                    const Sparse &mat2,
                                                    const Sparse &mat3) {
  Sparse::StorageIndex *ia = pattern.outerIndexPtr();
  Sparse::StorageIndex *ja = pattern.innerIndexPtr();
  Sparse::Scalar *a = pattern.valuePtr();

#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i < pattern.rows(); ++i) {
    Eigen::SparseVector<double> mat2_mat3_col = mat2 * mat3.row(i).transpose();
    for (auto j = ia[i]; j < ia[i + 1]; ++j) {
      a[j] = mat1.row(ja[j]).dot(mat2_mat3_col);
    }
  }
  return;
}

void formatted_sparse_multiplication_triple_product_largeSparse(largeSparse &pattern,
                                                    const largeSparse &mat1,
                                                    const largeSparse &mat2,
                                                    const largeSparse &mat3) {
  largeSparse::StorageIndex *ia = pattern.outerIndexPtr();
  largeSparse::StorageIndex *ja = pattern.innerIndexPtr();
  largeSparse::Scalar *a = pattern.valuePtr();

#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i < pattern.rows(); ++i) {
    Eigen::SparseVector<double> mat2_mat3_col = mat2 * mat3.row(i).transpose();
    for (auto j = ia[i]; j < ia[i + 1]; ++j) {
      a[j] = mat1.row(ja[j]).dot(mat2_mat3_col);
    }
  }
  return;
}



#else
void formatted_sparse_multiplication(largeSparse &pattern,
                                     const largeSparse &mat1,
                                     const largeSparse &mat2, double scal = 1) {
  eigen_assert(mat1.cols() == mat2.rows() && "dimension mismatch");
  long long int n = pattern.rows();
  largeSparse::StorageIndex *ia = pattern.outerIndexPtr();
  largeSparse::StorageIndex *ja = pattern.innerIndexPtr();
  largeSparse::Scalar *a = pattern.valuePtr();
  const largeSparse::StorageIndex *ia2 = mat1.outerIndexPtr();
  const largeSparse::StorageIndex *ja2 = mat1.innerIndexPtr();
  const largeSparse::Scalar *a2 = mat1.valuePtr();
  const largeSparse::StorageIndex *ia3 = mat2.outerIndexPtr();
  const largeSparse::StorageIndex *ja3 = mat2.innerIndexPtr();
  const largeSparse::Scalar *a3 = mat2.valuePtr();

  std::cout << "using dot\n";
#pragma omp parallel for
  for (auto i = 0; i < n; ++i) {
    for (auto j = ia[i]; j < ia[i + 1]; ++j)
      a[j] = sparse_dot_product(ja2 + ia2[i], ia2[i + 1] - ia2[i], a2 + ia2[i],
                                ja3 + ia3[ja[j]], ia3[ja[j] + 1] - ia3[ja[j]],
                                a3 + ia3[ja[j]]);
  }
  return;
}
#endif

#endif