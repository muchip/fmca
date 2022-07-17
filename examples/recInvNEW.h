#ifndef FMCA_RECINV_
#define FMCA_RECINV_

#include <Eigen/Sparse>
#include <iostream>

#include "pardiso_interface.h"
#include "sampletMatrixGenerator.h"

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, long long int> largeSparse;
typedef largeSparse::StorageIndex lSIndex;
typedef largeSparse::Scalar lSScalar;
typedef Eigen::SparseMatrix<lSScalar, Eigen::RowMajor, int> Sparse;

template <typename tIndex, typename tSize, typename tVal, typename sIndex,
          typename sSize, typename sVal>
inline void formatted_set_sparse(const tIndex *tidx, const tSize tsze,
                                 tVal *target, const sIndex *sidx,
                                 const sSize ssze, const sVal *source) {
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
inline double sparse_dot_product(const tIndex *tidx, const tSize tsze,
                                 const tVal *target, const sIndex *sidx,
                                 const sSize ssze, const sVal *source) {
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
inline void formatted_sparse_axpy(const tIndex *tidx, const tSize tsze,
                                  tVal *target, const sIndex *sidx,
                                  const sSize ssze, const sVal *source,
                                  double a) {
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

void formatted_sparse_multiplication(const lSIndex *ia, const lSIndex *ja,
                                     lSScalar *a, const lSIndex *ia2,
                                     const lSIndex *ja2, const lSScalar *a2,
                                     const lSIndex *ia3, const lSIndex *ja3,
                                     const lSScalar *a3, lSIndex n) {
  memset(a, 0, ia[n] * sizeof(lSScalar));
#pragma omp parallel for
  for (lSIndex i = 0; i < n; ++i) {
    for (lSIndex j = ia2[i]; j < ia2[i + 1]; ++j)
      if (std::abs(a2[j]) > 1e-15)
        formatted_sparse_axpy(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                              ja3 + ia3[ja2[j]], ia3[ja2[j] + 1] - ia3[ja2[j]],
                              a3 + ia3[ja2[j]], a2[j]);
  }
  return;
}

void formatted_sparse_multiplicationGTG(const lSIndex *ia, const lSIndex *ja,
                                        lSScalar *a, const lSIndex *ia2,
                                        const lSIndex *ja2, const lSScalar *a2,
                                        const lSIndex *ia3, const lSIndex *ja3,
                                        const lSScalar *a3, lSIndex n) {
  memset(a, 0, ia[n] * sizeof(lSScalar));
  for (lSIndex i = 0; i < n; ++i) {
#pragma omp parallel for
    for (lSIndex j = ia2[i]; j < ia2[i + 1]; ++j)
      if (std::abs(a2[j]) > 1e-15)
        formatted_sparse_axpy(ja + ia[ja2[j]], ia[ja2[j] + 1] - ia[ja2[j]],
                              a + ia[ja2[j]], ja3 + ia3[i], ia3[i + 1] - ia3[i],
                              a3 + ia3[i], a2[j]);
  }
  return;
}

void formatted_sparse_multiplicationSG(const lSIndex *ia, const lSIndex *ja,
                                       lSScalar *a, const lSIndex *ia2,
                                       const lSIndex *ja2, const lSScalar *a2,
                                       const lSIndex *ia3, const lSIndex *ja3,
                                       const lSScalar *a3, lSIndex n) {
  memset(a, 0, ia[n] * sizeof(lSScalar));
#pragma omp parallel for
  for (lSIndex i = 0; i < n; ++i) {
    for (lSIndex j = ia2[i]; j < ia2[i + 1]; ++j)
      if (std::abs(a2[j]) > 1e-15)
        formatted_sparse_axpy(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                              ja3 + ia3[ja2[j]], ia3[ja2[j] + 1] - ia3[ja2[j]],
                              a3 + ia3[ja2[j]], a2[j]);
  }
  for (lSIndex i = 0; i < n; ++i) {
#pragma omp parallel for
    for (lSIndex j = ia2[i]; j < ia2[i + 1]; ++j)
      if (i < ja2[j] && std::abs(a2[j]) > 1e-15)
        formatted_sparse_axpy(ja + ia[ja2[j]], ia[ja2[j] + 1] - ia[ja2[j]],
                              a + ia[ja2[j]], ja3 + ia3[i], ia3[i + 1] - ia3[i],
                              a3 + ia3[i], a2[j]);
  }
  return;
}

/**
 *  \brief Recursively inverts a matrix on a given pattern
 *         using block inversion
 **/
void recInv(largeSparse &M, const int splitn) {
  largeSparse retval;
  if (M.rows() > splitn) {
    const lSIndex n = M.rows();
    const lSIndex n2 = M.rows() / 2;
    largeSparse invS;
    largeSparse R;
    largeSparse V;
    {
      largeSparse invM11 = M.topLeftCorner(n2, n2);
      recInv(invM11, splitn);
      {
        largeSparse T;
        {
          largeSparse M12 = -M.topRightCorner(n2, n - n2);
          T = M12;
          // T = invM11 * M12;
          formatted_sparse_multiplicationSG(
              T.outerIndexPtr(), T.innerIndexPtr(), T.valuePtr(),
              invM11.outerIndexPtr(), invM11.innerIndexPtr(), invM11.valuePtr(),
              M12.outerIndexPtr(), M12.innerIndexPtr(), M12.valuePtr(),
              T.rows());
          Eigen::MatrixXd bla = invM11.selfadjointView<Eigen::Upper>() *
                                Eigen::MatrixXd::Identity(n2, n2);
          Eigen::MatrixXd bla2 = M12;
          Eigen::MatrixXd bla3 = T;
          std::cout << "#### error first prod:" << (bla3 - bla * bla2).norm()
                    << std::endl;
          largeSparse M22 = M.bottomRightCorner(n - n2, n - n2);
          invS = M22;
          bla2 = bla2.transpose() * bla3;
          formatted_sparse_multiplicationGTG(
              invS.outerIndexPtr(), invS.innerIndexPtr(), invS.valuePtr(),
              M12.outerIndexPtr(), M12.innerIndexPtr(), M12.valuePtr(),
              T.outerIndexPtr(), T.innerIndexPtr(), T.valuePtr(), invS.rows());
          bla = invS.selfadjointView<Eigen::Upper>() *
                Eigen::MatrixXd::Identity(n - n2, n - n2);
          std::cout << "#### error second prod:" << (bla - bla2).norm()
                    << std::endl;
#pragma omp parallel for
          for (lSIndex i = 0; i < invS.nonZeros(); ++i)
            invS.valuePtr()[i] += M22.valuePtr()[i];
          bla = invS.selfadjointView<Eigen::Upper>() *
                Eigen::MatrixXd::Identity(n - n2, n - n2);
          bla3 = M22.selfadjointView<Eigen::Upper>() *
                 Eigen::MatrixXd::Identity(n - n2, n - n2);
          std::cout << "#### error addition:" << (bla - bla2 - bla3).norm()
                    << std::endl;
        }
        Eigen::MatrixXd bla = invS.selfadjointView<Eigen::Upper>() *
                              Eigen::MatrixXd::Identity(n - n2, n - n2);
        recInv(invS, splitn);
        Eigen::MatrixXd bla2 = invS.selfadjointView<Eigen::Upper>() *
                               Eigen::MatrixXd::Identity(n - n2, n - n2);
        std::cout << "#### error inversion: " << (bla2 - bla.inverse()).norm()
                  << std::endl;
        largeSparse M21 = M.topRightCorner(n2, n - n2).transpose();
        largeSparse TT = T.transpose();
        formatted_sparse_multiplicationSG(
            M21.outerIndexPtr(), M21.innerIndexPtr(), M21.valuePtr(),
            invS.outerIndexPtr(), invS.innerIndexPtr(), invS.valuePtr(),
            TT.outerIndexPtr(), TT.innerIndexPtr(), TT.valuePtr(), M21.rows());
        bla = T;
        Eigen::MatrixXd bla3 = M21;
        std::cout << "#### error top right block: "
                  << (bla3 - bla2 * bla.transpose()).norm() << std::endl;
        R = M21.transpose();

        V = invM11;
        formatted_sparse_multiplication(
            V.outerIndexPtr(), V.innerIndexPtr(), V.valuePtr(),
            T.outerIndexPtr(), T.innerIndexPtr(), T.valuePtr(),
            M21.outerIndexPtr(), M21.innerIndexPtr(), M21.valuePtr(), V.rows());
        bla2 = V.selfadjointView<Eigen::Upper>() *
               Eigen::MatrixXd::Identity(n2, n2);
        std::cout << "#### error top left block: " << (bla2 - bla * bla3).norm()
                  << std::endl;

        V += invM11;
      }
    }
    {
      Sparse temp = M;
      pardiso_interface(temp.outerIndexPtr(), temp.innerIndexPtr(),
                        temp.valuePtr(), temp.rows());
      std::cout << "&&&&&&&&&&&&&&&&&&&&&&&\n";
      std::cout << Eigen::MatrixXd(temp) << std::endl;
      V.makeCompressed();
      R.makeCompressed();
      invS.makeCompressed();
      const lSIndex *ia = M.outerIndexPtr();
      const lSIndex *ja = M.innerIndexPtr();
      lSScalar *a = M.valuePtr();
      memset(a, 0, M.nonZeros() * sizeof(lSScalar));
      lSIndex *ia2 = V.outerIndexPtr();
      lSIndex *ja2 = V.innerIndexPtr();
      lSScalar *a2 = V.valuePtr();
      lSIndex *ia3 = R.outerIndexPtr();
      lSIndex *ja3 = R.innerIndexPtr();
      lSScalar *a3 = R.valuePtr();
      lSIndex *ia4 = invS.outerIndexPtr();
      lSIndex *ja4 = invS.innerIndexPtr();
      lSScalar *a4 = invS.valuePtr();
      for (lSIndex i = 0; i < ia[n]; ++i) a[i] = 0;
      for (lSIndex i = 0; i < ia3[n2]; ++i) ja3[i] += n2;
      for (lSIndex i = 0; i < ia4[n - n2]; ++i) ja4[i] += n2;
#pragma omp parallel for
      for (lSIndex i = 0; i < n2; ++i) {
        formatted_set_sparse(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                             ja2 + ia2[i], ia2[i + 1] - ia2[i], a2 + ia2[i]);
      }
#pragma omp parallel for
      for (lSIndex i = 0; i < n2; ++i) {
        formatted_set_sparse(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                             ja3 + ia3[i], ia3[i + 1] - ia3[i], a3 + ia3[i]);
      }
#pragma omp parallel for
      for (lSIndex i = n2; i < n; ++i) {
        formatted_set_sparse(ja + ia[i], ia[i + 1] - ia[i], a + ia[i],
                             ja4 + ia4[i - n2], ia4[i - n2 + 1] - ia4[i - n2],
                             a4 + ia4[i - n2]);
      }
      std::cout << "&&&&&&&&&&&&&&&&&&&&&&&\n";

      std::cout << Eigen::MatrixXd(M) << std::endl;
      std::cout << "&&&&&&&&&&&&&&&&&&&&&&&\n";

      std::cout << Eigen::MatrixXd(temp) - Eigen::MatrixXd(M) << std::endl;
      std::cout << "&&&&&&&&&&&&&&&&&&&&&&&\n";
    }

  } else {
    Sparse temp = M;
    pardiso_interface(temp.outerIndexPtr(), temp.innerIndexPtr(),
                      temp.valuePtr(), temp.rows());
    memcpy(M.valuePtr(), temp.valuePtr(), M.nonZeros() * sizeof(lSScalar));
  }
  return;
}
#endif
