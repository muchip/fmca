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
#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"
#include "../FMCA/src/util/Tictoc.h"

using Cholesky = Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<FMCA::Scalar>,
                                             Eigen::Upper>;
template <typename T>
void PivotedCholesky(const T &K, FMCA::Matrix *L,
                     std::vector<FMCA::Index> *idcs, FMCA::Scalar tol = 1e-3,
                     FMCA::Index max_cols = 1000) {
  FMCA::Vector D = K.diagonal();
  FMCA::Index pivot = 0;
  FMCA::Scalar tr = 0;
  L->resize(K.rows(), max_cols);
  idcs->resize(max_cols);
  tr = D.sum();
  // we guarantee the error tr(K-LL^T)/tr(K) < tol
  tol *= tr;
  // perform pivoted Cholesky decomposition
  std::cout << "N: " << K.rows() << " max number of cols: " << max_cols
            << std::endl
            << "rel tol: " << tol << " initial trace: " << tr << std::endl;
  FMCA::Index step = 0;
  while ((step < max_cols) && (tol < tr)) {
    D.maxCoeff(&pivot);
    (*idcs)[step] = pivot;
    // get new column from K
    L->col(step) = K.col(pivot);
    // update column with the current matrix Lmatrix_
    L->col(step) -= L->leftCols(step) * L->row(pivot).head(step).transpose();
    if ((*L)(pivot, step) <= 0) {
      std::cout << "breaking with non positive pivot\n";
      break;
    }
    L->col(step) /= sqrt((*L)(pivot, step));
    // update the diagonal and the trace
    D.array() -= L->col(step).array().square();
    // compute the trace of the Schur complement
    tr = D.sum();
    ++step;
  }
  std::cout << "steps: " << step << " trace error: " << tr << std::endl;
  // crop L, indices to their actual size
  L->conservativeResize(L->rows(), step);
  idcs->resize(step);
  return;
}


#define NPTS 500000
#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("TPS2D", 1.);
  const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  const FMCA::Scalar threshold = 1e-9;
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 8;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const Moments mom(P, mpole_deg);
  const Moments Pmom(P, 1);
  const MatrixEvaluator mat_eval(mom, function);
  std::cout << "dtilde:                       " << dtilde << std::endl;
  std::cout << "mpole_deg:                    " << mpole_deg << std::endl;
  std::cout << "eta:                          " << eta << std::endl;
  const SampletMoments samp_mom(P, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, P);
  FMCA::clusterTreeStatistics(hst, P);

  // P
  const FMCA::Matrix Pol = Pmom.moment_matrix(hst);
  std::cout << Pol.leftCols(10).transpose() << std::endl
            << "--------" << std::endl;
  // T*P
  const FMCA::Matrix TPol = hst.sampletTransform(Pol.transpose());
  std::cout << TPol.topRows(10) << std::endl << "--------" << std::endl;

  FMCA::Index mq = Pmom.interp().idcs().index_set().size();
  std::cout << "mq= " << mq << std::endl;

  // TPol.topRows(mq) is the scaling part, the S block in [S;0]
  const FMCA::Matrix PTP = TPol.topRows(mq).transpose() * TPol.topRows(mq);
  std::cout << "error= " << (PTP - TPol.transpose() * TPol).norm() / TPol.norm()
            << std::endl;
  const FMCA::Matrix invPTP = PTP.inverse();
  std::cout << "inverse error= "
            << (invPTP * PTP - FMCA::Matrix::Identity(mq, mq)).norm() /
                   std::sqrt(mq)
            << std::endl;
  std::cout << TPol.topRows(mq) * invPTP * TPol.topRows(mq).transpose()
            << std::endl;
  
  // compression
  T.tic();
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Scomp;
  Scomp.init(hst, eta, 100 * FMCA_ZERO_TOLERANCE);
  T.toc("planner:                     ");
  T.tic();
  Scomp.compress(mat_eval);
  T.toc("compressor:                  ");
  T.tic();
  const auto &ap_trips = Scomp.triplets();
  std::cout << "anz (a-priori):               "
            << std::round(ap_trips.size() / FMCA::Scalar(NPTS)) << std::endl;
  T.toc("triplets:                    ");

  T.tic();
  const auto &trips = Scomp.aposteriori_triplets_fast(threshold);
  std::cout << "anz (a-posteriori):           "
            << std::round(trips.size() / FMCA::Scalar(NPTS)) << std::endl;

  T.toc("triplets:                    ");
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
  Eigen::SparseMatrix<FMCA::Scalar> S(NPTS, NPTS);
  S.setFromTriplets(trips.begin(), trips.end());
  // the first mq correspond to the polynomial subspace, I then just take the K_{psi,psi}
  Eigen::SparseMatrix<FMCA::Scalar> Sspd =
      S.block(mq, mq, NPTS - mq, NPTS - mq);
  Eigen::SparseMatrix<FMCA::Scalar> I(NPTS - mq, NPTS - mq);
  I.setIdentity();
#if 0
  FMCA::Matrix L;
  std::vector<FMCA::Index> idcs;
  Eigen::SparseMatrix<FMCA::Scalar> Ssym = Sspd.selfadjointView<Eigen::Upper>();
  PivotedCholesky(Ssym, &L, &idcs, 1e-3, 2000);
  std::cout << L.cols() << std::endl;
  FMCA::Matrix test(Ssym.rows(), 100);
  test.setRandom();
  FMCA::Matrix Y1 = Ssym * test;
  FMCA::Matrix Y2 = L * (L.transpose() * test).eval();
  std::cout << "error:" << (Y1 - Y2).norm() / test.norm() << std::endl;
#endif
  Sspd += I * 1e-8;
  Cholesky llt;
  llt.compute(Sspd);
#if 0
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper> cg;
  cg.setTolerance(1e-6);
  cg.setMaxIterations(1000);

  // Analyze and factorize pattern, then solve
  cg.compute(Sspd);
  Eigen::VectorXd sol = cg.solve(FMCA::Vector::Ones(Sspd.cols()));
  // Access info about iterations/residual/error
  int iters = cg.iterations();
  err = cg.error();
  std::cout << "  " << iters << " " << err << std::endl;
#endif
  FMCA::Matrix Sfull = S.block(0, 0, 10, 10);
  std::cout << Sfull << std::endl;
  Sfull = Sspd.block(0, 0, 10, 10);
  std::cout << Sfull << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  return 0;
}
