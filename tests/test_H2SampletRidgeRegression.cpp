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
// #define EIGEN_DONT_PARALLELIZE
#include <iostream>
//
#include <Eigen/Dense>
#include <Eigen/Sparse>
extern "C" {
#include <cholmod.h>
}
#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 100000
#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("MaternNu", 1., 1., 0.5);
  const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  const FMCA::Scalar threshold = 1e-5;
  const FMCA::Scalar eta = 0.5;
  const FMCA::Scalar dtilde = 4;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const Moments mom(P, mpole_deg);
  const MatrixEvaluator mat_eval(mom, function);
  std::cout << "dtilde:                       " << dtilde << std::endl;
  std::cout << "mpole_deg:                    " << mpole_deg << std::endl;
  std::cout << "eta:                          " << eta << std::endl;
  const SampletMoments samp_mom(P, dtilde - 1);
  const H2SampletTree hst(mom, samp_mom, 0, P);
  T.tic();
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Scomp;
  Scomp.init(hst, eta, threshold);
  T.toc("planner:                     ");
  T.tic();
  Scomp.compress(mat_eval);
  T.toc("compressor:                  ");
  T.tic();
  const auto &trips = Scomp.triplets();
  T.toc("triplets:                    ");
  std::cout << "anz:                          "
            << std::round(trips.size() / FMCA::Scalar(NPTS)) << std::endl;
  FMCA::Vector x(NPTS), y1(NPTS), y2(NPTS);
  FMCA::Scalar err = 0;
  FMCA::Scalar nrm = 0;
  for (auto i = 0; i < 10; ++i) {
    FMCA::Index index = rand() % P.cols();
    x.setZero();
    x(index) = 1;
    FMCA::Vector col = function.eval(P, P.col(hst.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst.indices(), hst.block_size()));
    x = hst.sampletTransform(x);
    y2.setZero();
    for (const auto &i : trips) {
      y2(i.row()) += i.value() * x(i.col());
      if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
    }
    y2 = hst.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  err = sqrt(err / nrm);
  std::cout << "compression error:            " << err << std::endl
            << std::flush;
  std::cout << std::string(60, '-') << std::endl;
  Eigen::SparseMatrix<FMCA::Scalar> K(NPTS, NPTS);
  K.setFromTriplets(trips.begin(), trips.end());
  Eigen::SparseMatrix<FMCA::Scalar> KT = K.transpose();
  K = KT;
  K.makeCompressed();
  T.tic();
  FMCA::Matrix rhs, sol;

  {
    cholmod_common c;
    cholmod_start(&c);
    cholmod_sparse *A =
        cholmod_allocate_sparse(K.rows(),      // Number of rows
                                K.cols(),      // Number of columns
                                K.nonZeros(),  // Number of non-zero entries
                                1,             // Sorted
                                1,             // Packed
                                0,  // Not a pattern matrix (values are present)
                                CHOLMOD_REAL,  // Matrix type (real or complex)
                                &c             // CHOLMOD common object
        );
    A->p = const_cast<int *>(K.outerIndexPtr());
    A->i = const_cast<int *>(K.innerIndexPtr());
    A->x = reinterpret_cast<void *>(K.valuePtr());
    A->stype = -1;
    cholmod_factor *L;
    cholmod_print_sparse(A, "A", &c);
    cholmod_nnz(A, &c);
    L = cholmod_analyze(A, &c);
    cholmod_factorize(A, L, &c);
    int nnzL = 0;
    {
      // Count nonzeros in the factorization
      const double *s_data = reinterpret_cast<const double *>(L->s);
      for (int i = 0; i < L->ssize; ++i)
        if (s_data[i] != 0.0) ++nnzL;
    }
    std::cout << "anz(L): " << nnzL / P.cols() << std::endl;
    cholmod_dense *x, *b;
    b = cholmod_ones(A->nrow, 1, A->xtype, &c);
    x = cholmod_solve(CHOLMOD_L, L, b, &c);
    std::cout << "solver worked" << std::endl;
      cholmod_finish(&c);
  }
  T.toc("CHOLMOD:                     ");
  std::cout << "err: " << (rhs - K.selfadjointView<Eigen::Lower>() * sol).norm()
            << rhs.norm() << std::endl;
  return 0;
}
