// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 40000
#define DIM 3

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

// ground truth: polynomial trend plus four localized Gaussian bumps
const FMCA::Scalar beta[4] = {2.0, 1.5, -2.0, 1.0};
const FMCA::Scalar bump_amp[4] = {0.8, -0.6, 0.5, -0.45};
const FMCA::Scalar bump_width[4] = {0.1, 0.12, 0.1, 0.12};

FMCA::Vector f_true(const FMCA::Matrix &P, const FMCA::Matrix &bump_centers) {
  FMCA::Vector f(P.cols());
  for (FMCA::Index i = 0; i < P.cols(); ++i) {
    FMCA::Scalar val =
        beta[0] + beta[1] * P(0, i) + beta[2] * P(1, i) + beta[3] * P(2, i);
    for (FMCA::Index m = 0; m < 4; ++m) {
      const FMCA::Scalar r2 = (P.col(i) - bump_centers.col(m)).squaredNorm();
      val += bump_amp[m] * std::exp(-r2 / (2 * bump_width[m] * bump_width[m]));
    }
    f(i) = val;
  }
  return f;
}

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel kernel("BIHARMONIC3D", 1.);
  const FMCA::Index dtilde = 4;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const FMCA::Scalar eta = 0.7;
  const FMCA::Scalar threshold = 1e-6;
  const FMCA::Scalar sigma_n = 0.05; // noise level
  const FMCA::Index mq = 1 + DIM;  // monomials {1, x, y, z}

  // data sites on a sphere of radius 0.5 centred in the unit cube
  FMCA::Matrix P_all(DIM, NPTS);
  for (FMCA::Index i = 0; i < NPTS; ++i) {
    FMCA::Vector v = FMCA::Vector::Random(DIM);
    while (v.norm() < 1e-2) v = FMCA::Vector::Random(DIM);
    P_all.col(i) = FMCA::Vector::Constant(DIM, 0.5) + 0.5 * v / v.norm();
  }

  // bump centres spread over the data set and ground truth values
  FMCA::Matrix bump_centers(DIM, 4);
  for (FMCA::Index m = 0; m < 4; ++m)
    bump_centers.col(m) = P_all.col(((m + 1) * NPTS) / 5);
  const FMCA::Vector f = f_true(P_all, bump_centers);

  // noisy observations
  std::srand(7);
  FMCA::Vector y_all(NPTS);
  for (FMCA::Index i = 0; i < NPTS; ++i) {
    const FMCA::Scalar u1 =
        (FMCA::Scalar(rand()) + 1.0) / (FMCA::Scalar(RAND_MAX) + 2.0);
    const FMCA::Scalar u2 =
        (FMCA::Scalar(rand()) + 1.0) / (FMCA::Scalar(RAND_MAX) + 2.0);
    y_all(i) = f(i) + sigma_n * std::sqrt(-2.0 * std::log(u1)) *
                          std::cos(2.0 * FMCA_PI * u2);
  }

  // split: hole (extrapolation), test (interpolation), train
  FMCA::Vector anchor(DIM);
  anchor << 1.0, 0.5, 0.5;
  FMCA::Index hole_center = 0;
  FMCA::Scalar best = FMCA_INF;
  for (FMCA::Index i = 0; i < NPTS; ++i) {
    const FMCA::Scalar d2 = (P_all.col(i) - anchor).squaredNorm();
    if (d2 < best) {
      best = d2;
      hole_center = i;
    }
  }
  std::vector<int> label(NPTS, 0);  // 0 = train, 1 = test, 2 = hole
  for (FMCA::Index i = 0; i < NPTS; ++i)
    if ((P_all.col(i) - P_all.col(hole_center)).norm() < 0.15) label[i] = 2;
  {
    FMCA::Index k = 0;
    for (FMCA::Index i = 0; i < NPTS; ++i)
      if (label[i] == 0 && (k++ % 20) == 19) label[i] = 1;
  }
  FMCA::Index N = 0;
  for (FMCA::Index i = 0; i < NPTS; ++i) N += (label[i] == 0);
  FMCA::Matrix P(DIM, N);
  FMCA::Vector y(N);
  {
    FMCA::Index k = 0;
    for (FMCA::Index i = 0; i < NPTS; ++i)
      if (label[i] == 0) {
        P.col(k) = P_all.col(i);
        y(k) = y_all(i);
        ++k;
      }
  }
  std::cout << "N_train:                      " << N << std::endl;
  std::cout << "sigma_n:                      " << sigma_n << std::endl;

  const Moments mom(P, mpole_deg);
  const MatrixEvaluator mat_eval(mom, kernel);
  const SampletMoments samp_mom(P, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, P);

  // polynomial block {1, x, y, z} -> [S; 0] after the samplet transform
  FMCA::Matrix Pol(mq, N);
  for (FMCA::Index i = 0; i < N; ++i) {
    Pol(0, i) = 1.0;
    Pol.block(1, i, DIM, 1) = P.col(i);
  }
  const FMCA::Matrix TPol =
      hst.sampletTransform(hst.toClusterOrder(Pol.transpose()));
  const FMCA::Matrix S_block = TPol.topRows(mq);
  const FMCA::Scalar dec_err = TPol.bottomRows(N - mq).norm() / TPol.norm();
  std::cout << "decoupling error TP=[S;0]:    " << dec_err << std::endl;
  assert(dec_err < 1e-10 && "polynomial block not annihilated");

  // compression and detail block K_PsiPsi + sigma_n^2 I
  T.tic();
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Scomp;
  Scomp.init(hst, eta, 1000 * FMCA_ZERO_TOLERANCE);
  Scomp.compress(mat_eval);
  Scomp.triplets();
  const auto &trips = Scomp.aposteriori_triplets_fast(threshold);
  T.toc("compression:                 ");
  std::cout << "anz (a-posteriori):           "
            << std::round(trips.size() / FMCA::Scalar(N)) << std::endl;
  Eigen::SparseMatrix<FMCA::Scalar> Smat(N, N);
  Smat.setFromTriplets(trips.begin(), trips.end());
  Eigen::SparseMatrix<FMCA::Scalar> Kpsi = Smat.block(mq, mq, N - mq, N - mq);
  {
    Eigen::SparseMatrix<FMCA::Scalar> Id(N - mq, N - mq);
    Id.setIdentity();
    Kpsi += Id * (sigma_n * sigma_n);
  }
  const Eigen::SparseMatrix<FMCA::Scalar> K_PPsi = Smat.block(0, mq, mq, N - mq);
  FMCA::Matrix K_PP =
      FMCA::Matrix(Smat.block(0, 0, mq, mq)).selfadjointView<Eigen::Upper>();
  K_PP.diagonal().array() += sigma_n * sigma_n;

  T.tic();
  Eigen::SimplicialLLT<Eigen::SparseMatrix<FMCA::Scalar>, Eigen::Upper> llt;
  llt.compute(Kpsi);
  T.toc("Cholesky (K_PsiPsi + s^2 I): ");
  assert(llt.info() == Eigen::Success && "Cholesky failed");

  // posterior mean coefficients by the null space approach
  const Eigen::PartialPivLU<FMCA::Matrix> Slu(S_block);
  const Eigen::PartialPivLU<FMCA::Matrix> SluT(S_block.transpose());
  const FMCA::Vector Uy = hst.sampletTransform(hst.toClusterOrder(y));
  const FMCA::Vector c_Psi = llt.solve(Uy.tail(N - mq));
  const FMCA::Vector d = Slu.solve(Uy.head(mq) - K_PPsi * c_Psi);
  FMCA::Vector c_samplet = FMCA::Vector::Zero(N);
  c_samplet.tail(N - mq) = c_Psi;
  const FMCA::Vector c =
      hst.toNaturalOrder(hst.inverseSampletTransform(c_samplet));
  const FMCA::Scalar side_err = (Pol * c).norm() / c.norm();
  std::cout << "side condition |P'c|:         " << side_err << std::endl;
  assert(side_err < 1e-10 && "side condition P'c=0 violated");

  // posterior mean at all data sites, K(x,y) = -|x-y|
  FMCA::Vector mu(NPTS);
  T.tic();
#pragma omp parallel for schedule(static)
  for (FMCA::Index i = 0; i < NPTS; ++i) {
    FMCA::Scalar s = 0;
    for (FMCA::Index j = 0; j < N; ++j)
      s -= c(j) * (P_all.col(i) - P.col(j)).norm();
    mu(i) = s + d(0) + d(1) * P_all(0, i) + d(2) * P_all(1, i) +
            d(3) * P_all(2, i);
  }
  T.toc("posterior mean (all points): ");
  FMCA::Scalar rmse[3] = {0, 0, 0};
  FMCA::Index count[3] = {0, 0, 0};
  for (FMCA::Index i = 0; i < NPTS; ++i) {
    rmse[label[i]] += std::pow(mu(i) - f(i), 2);
    ++count[label[i]];
  }
  for (FMCA::Index l = 0; l < 3; ++l) rmse[l] = std::sqrt(rmse[l] / count[l]);
  std::cout << "rmse train/test/hole:         " << rmse[0] << " / " << rmse[1]
            << " / " << rmse[2] << std::endl;
  assert(rmse[1] < sigma_n && "posterior mean does not filter the noise");

  // posterior standard deviation at a few test and hole sites
  auto std_at = [&](const FMCA::Vector &xstar) -> FMCA::Scalar {
    const FMCA::Matrix Kcol = kernel.eval(P, xstar);
    FMCA::Vector pstar(mq);
    pstar(0) = 1.0;
    pstar.segment(1, DIM) = xstar;
    const FMCA::Matrix ktil = hst.sampletTransform(hst.toClusterOrder(Kcol));
    const FMCA::Vector kP = ktil.col(0).head(mq);
    const FMCA::Vector kPsi = ktil.col(0).tail(N - mq);
    const FMCA::Vector lamP = SluT.solve(pstar);
    const FMCA::Vector lamPsi =
        llt.solve((kPsi - K_PPsi.transpose() * lamP).eval());
    const FMCA::Vector rmu = kP - K_PP * lamP - K_PPsi * lamPsi;
    const FMCA::Scalar var = -kP.dot(lamP) - kPsi.dot(lamPsi) - lamP.dot(rmu);
    return std::sqrt(std::max<FMCA::Scalar>(var, 0.0));
  };
  // at most 100 test and 100 hole sites, one back-substitution per site
  std::vector<FMCA::Index> var_idx;
  FMCA::Scalar sd_mean[3] = {0, 0, 0};
  FMCA::Index sd_count[3] = {0, 0, 0};
  T.tic();
  for (FMCA::Index i = 0; i < NPTS; ++i)
    if (label[i] > 0 && sd_count[label[i]] < 100) {
      var_idx.push_back(i);
      ++sd_count[label[i]];
    }
  FMCA::Vector sd(var_idx.size());
  for (FMCA::Index k = 0; k < FMCA::Index(var_idx.size()); ++k) {
    sd(k) = std_at(P_all.col(var_idx[k]));
    sd_mean[label[var_idx[k]]] += sd(k);
  }
  T.toc("posterior std (few points):  ");
  sd_mean[1] /= sd_count[1];
  sd_mean[2] /= sd_count[2];
  std::cout << "mean std test/hole:           " << sd_mean[1] << " / "
            << sd_mean[2] << std::endl;
  assert(sd_mean[2] > sd_mean[1] &&
         "no increased uncertainty inside the data hole");

  // export the results for visual inspection
  FMCA::IO::plotPointsColor("kriging_truth.vtk", P_all, f);
  FMCA::IO::plotPointsColor("kriging_mean.vtk", P_all, mu);
  FMCA::IO::plotPointsColor("kriging_error.vtk", P_all, (mu - f).cwiseAbs());
  FMCA::Matrix P_var(DIM, var_idx.size());
  for (FMCA::Index k = 0; k < FMCA::Index(var_idx.size()); ++k)
    P_var.col(k) = P_all.col(var_idx[k]);
  FMCA::IO::plotPointsColor("kriging_std.vtk", P_var, sd);
  return 0;
}
