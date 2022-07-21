#ifndef FMCA_FORMATTEDMULTIPLICATION_
#define FMCA_FORMATTEDMULTIPLICATION_

#include <Eigen/Sparse>
#include <iostream>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, long long int> largeSparse;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> Sparse;

template <typename tIndex, typename tSize, typename tVal, typename sIndex,
          typename sSize, typename sVal>
inline void formatted_sparse_axpy(tIndex *tidx, const tSize tsze, tVal *target,
                                  const sIndex *sidx, const sSize ssze,
                                  const sVal *source, double a) {
  if (!tsze || !ssze || sidx[ssze - 1] < tidx[0] || tidx[tsze - 1] < sidx[0])
    return;
  auto j = 0;
  if (tsze < ssze)
    for (tSize i = 0; i < tsze; ++i) {
      while (j < ssze && sidx[j] < tidx[i]) ++j;
      if (j >= ssze)
        break;
      else if (sidx[j] == tidx[i])
        target[i] += a * source[j];
    }
  else
    for (sSize i = 0; i < ssze; ++i) {
      while (j < tsze && tidx[j] < sidx[i]) ++j;
      if (j >= tsze)
        break;
      else if (sidx[i] == tidx[j])
        target[j] += a * source[i];
    }
  return;
}

template <typename tIndex, typename tSize, typename tVal, typename sIndex,
          typename sSize, typename sVal>
inline double sparse_dot_product(tIndex *tidx, const tSize tsze, tVal *target,
                                 sIndex *sidx, const sSize ssze, sVal *source) {
  if (!tsze || !ssze || sidx[ssze - 1] < tidx[0] || tidx[tsze - 1] < sidx[0])
    return 0.;
  double retval = 0;
  auto j = 0;
  if (tsze < ssze)
    for (tSize i = 0; i < tsze; ++i) {
      while (j < ssze && sidx[j] < tidx[i]) ++j;
      if (j >= ssze)
        break;
      else if (sidx[j] == tidx[i])
        retval += target[i] * source[j];
    }
  else
    for (sSize i = 0; i < ssze; ++i) {
      while (j < tsze && tidx[j] < sidx[i]) ++j;
      if (j >= tsze)
        break;
      else if (sidx[i] == tidx[j])
        retval += target[j] * source[i];
    }
  return retval;
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
  std::cout << "using axpy2\n";

#pragma omp parallel for
  for (auto i = 0; i < n; ++i) {
    for (auto j = ia2[i]; j < ia2[i + 1]; ++j)
      if (std::abs(a2[j]) > 1e-15)
        formatted_sparse_axpy2(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                              ja3 + ia3[ja2[j]], ia3[ja2[j] + 1] - ia3[ja2[j]],
                              a3 + ia3[ja2[j]], scal * a2[j]);
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
