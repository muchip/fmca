#ifndef FMCA_RECINV_
#define FMCA_RECINV_

#include <Eigen/Sparse>
#include <iostream>

#include "pardiso_interface.h"
#include "sampletMatrixGenerator.h"

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, long long int> largeSparse;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> Sparse;

template <typename tIndex, typename tSize, typename tVal, typename sIndex,
          typename sSize, typename sVal>
inline void formatted_set_sparse(tIndex *tidx, const tSize tsze, tVal *target,
                                 sIndex *sidx, const sSize ssze, sVal *source) {
  if (!tsze || !ssze || sidx[ssze - 1] < tidx[0] || tidx[tsze - 1] < sidx[0])
    return;
  auto j = 0;
  if (tsze < ssze)
    for (tSize i = 0; i < tsze; ++i) {
      while (j < ssze && sidx[j] < tidx[i]) ++j;
      if (j >= ssze)
        break;
      else if (sidx[j] == tidx[i])
        target[i] = source[j];
    }
  else
    for (sSize i = 0; i < ssze; ++i) {
      while (j < tsze && tidx[j] < sidx[i]) ++j;
      if (j >= tsze)
        break;
      else if (sidx[i] == tidx[j])
        target[j] = source[i];
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

void formatted_sparse_multiplicationGTG(largeSparse &pattern,
                                        const largeSparse &mat1,
                                        const largeSparse &mat2,
                                        double scal = 1) {
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
  for (auto i = 0; i < n; ++i) {
#pragma omp parallel for
    for (auto j = ia2[i]; j < ia2[i + 1]; ++j)
      if (std::abs(a2[j]) > 1e-15)
        formatted_sparse_axpy(ja + ia[ja2[j]], ia[ja2[j] + 1] - ia[ja2[j]],
                              a + ia[ja2[j]], ja3 + ia3[i], ia3[i + 1] - ia3[i],
                              a3 + ia3[i], scal * a2[j]);
  }
  return;
}

void formatted_sparse_multiplicationSG(largeSparse &pattern,
                                       const largeSparse &mat1,
                                       const largeSparse &mat2,
                                       double scal = 1) {
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
#pragma omp parallel for
  for (auto i = 0; i < n; ++i) {
    for (auto j = ia2[i]; j < ia2[i + 1]; ++j)
      if (std::abs(a2[j]) > 1e-15)
        formatted_sparse_axpy(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                              ja3 + ia3[ja2[j]], ia3[ja2[j] + 1] - ia3[ja2[j]],
                              a3 + ia3[ja2[j]], scal * a2[j]);
  }
  for (auto i = 0; i < n; ++i) {
#pragma omp parallel for
    for (auto j = ia2[i]; j < ia2[i + 1]; ++j)
      if (i < ja2[j] && std::abs(a2[j]) > 1e-15)
        formatted_sparse_axpy(ja + ia[ja2[j]], ia[ja2[j] + 1] - ia[ja2[j]],
                              a + ia[ja2[j]], ja3 + ia3[i], ia3[i + 1] - ia3[i],
                              a3 + ia3[i], scal * a2[j]);
  }
  return;
}

/**
 *  \brief Recursively inverts a matrix on a given pattern
 *         using block inversion
 **/
#if 1
void recInv(largeSparse &M, const int splitn) {
  if (M.rows() > splitn) {
    const long long int n = M.rows();
    const long long int n2 = M.rows() / 2;
    largeSparse invS;
    largeSparse R;
    largeSparse V;
    {
      largeSparse invM11 = M.topLeftCorner(n2, n2);
      recInv(invM11, splitn);
      {
        largeSparse T;
        {
          largeSparse M12 = M.topRightCorner(n2, n - n2);
          T = M12;
          memset(T.valuePtr(), 0, T.nonZeros() * sizeof(largeSparse::Scalar));
          formatted_sparse_multiplicationSG(T, invM11, M12);
          largeSparse M22 = M.bottomRightCorner(n - n2, n - n2);
          invS = M22;
          formatted_sparse_multiplicationGTG(invS, M12, T, -1);
        }
        recInv(invS, splitn);
        R = T;
        memset(R.valuePtr(), 0, R.nonZeros() * sizeof(largeSparse::Scalar));
        formatted_sparse_multiplication(R, T, invS, -1);
        formatted_sparse_multiplication(
            R, T, invS.triangularView<Eigen::StrictlyUpper>().transpose(), -1);
        V = invM11;
        formatted_sparse_multiplication(V, T, R.transpose(), -1);
      }
    }
    {
      V.makeCompressed();
      R.makeCompressed();
      invS.makeCompressed();
      largeSparse::StorageIndex *ia = M.outerIndexPtr();
      largeSparse::StorageIndex *ja = M.innerIndexPtr();
      largeSparse::Scalar *a = M.valuePtr();
      largeSparse::StorageIndex *ia2 = V.outerIndexPtr();
      largeSparse::StorageIndex *ja2 = V.innerIndexPtr();
      largeSparse::Scalar *a2 = V.valuePtr();
      largeSparse::StorageIndex *ia3 = R.outerIndexPtr();
      largeSparse::StorageIndex *ja3 = R.innerIndexPtr();
      largeSparse::Scalar *a3 = R.valuePtr();
      largeSparse::StorageIndex *ia4 = invS.outerIndexPtr();
      largeSparse::StorageIndex *ja4 = invS.innerIndexPtr();
      largeSparse::Scalar *a4 = invS.valuePtr();
      for (auto i = 0; i < ia[n]; ++i) a[i] = 0;
      for (auto i = 0; i < ia3[n2]; ++i) ja3[i] += n2;
      for (auto i = 0; i < ia4[n - n2]; ++i) ja4[i] += n2;
#pragma omp parallel for
      for (auto i = 0; i < n2; ++i) {
        formatted_set_sparse(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                             ja2 + ia2[i], ia2[i + 1] - ia2[i], a2 + ia2[i]);
      }
#pragma omp parallel for
      for (auto i = 0; i < n2; ++i) {
        formatted_set_sparse(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                             ja3 + ia3[i], ia3[i + 1] - ia3[i], a3 + ia3[i]);
      }
#pragma omp parallel for
      for (auto i = n2; i < n; ++i) {
        formatted_set_sparse(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                             ja4 + ia4[i - n2], ia4[i - n2 + 1] - ia4[i - n2],
                             a4 + ia4[i - n2]);
      }
    }
  } else {
    Sparse temp = M;
    pardiso_interface(temp.outerIndexPtr(), temp.innerIndexPtr(),
                      temp.valuePtr(), temp.rows());
    memcpy(M.valuePtr(), temp.valuePtr(),
           M.nonZeros() * sizeof(largeSparse::Scalar));
  }

  return;
}
#endif
#endif
