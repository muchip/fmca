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
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/SSN.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 100000
#define DIM 2

namespace FMCA {
////////////////////////////////////////////////////////////////////////////////
Scalar powerIteration(const SparseMatrix& A, Index steps = 20) {
  Scalar norm = 0;
  Vector x = Vector::Random(A.rows());
  for (auto i = 0; i < steps; ++i) {
    x = A * x;
    x /= x.norm();
  }
  return x.dot(A * x);
}
////////////////////////////////////////////////////////////////////////////////
Vector conjugateGradient(const SparseMatrix& A, const Vector& b,
                         Index max_iter = 100, Scalar tol = 1e-10) {
  Vector x = Vector::Zero(A.rows());
  Vector r = b;
  Vector p = r;
  Scalar rs_old = r.squaredNorm();
  for (Index i = 0; i < max_iter; ++i) {
    Vector Ap = A * p;
    Scalar alpha = rs_old / p.dot(Ap);
    x += alpha * p;
    r -= alpha * Ap;
    Scalar rs_new = r.squaredNorm();
    if (std::sqrt(rs_new) < tol * b.norm()) break;
    p = r + (rs_new / rs_old) * p;
    rs_old = rs_new;
  }
  return x;
}
// Inverse Power Iteration con shift e CG
Scalar minEigenvalueCG(const SparseMatrix& A, Scalar shift = 1e-12,
                       Index steps = 30, Index cg_iter = 50) {
  const Index n = A.rows();
  Vector x = Vector::Random(n);
  x /= x.norm();
  // =A_shifted = A + shift*I
  std::vector<Eigen::Triplet<Scalar>> triplets;
  for (int k = 0; k < A.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
      triplets.push_back(
          Eigen::Triplet<Scalar>(it.row(), it.col(), it.value()));
    }
  }
  for (Index i = 0; i < n; ++i) {
    triplets.push_back(Eigen::Triplet<Scalar>(i, i, shift));
  }
  SparseMatrix A_shifted(n, n);
  A_shifted.setFromTriplets(triplets.begin(), triplets.end());
  for (auto i = 0; i < steps; ++i) {
    x = conjugateGradient(A_shifted, x, cg_iter);
    Scalar norm = x.norm();
    if (norm < 1e-15) break;
    x /= norm;
  }
  Scalar lambda = x.dot(A * x) / x.dot(x);
  return lambda;
}
////////////////////////////////////////////////////////////////////////////////
// Lanczos tridiagonalization
Scalar minEigenvalueLanczos(const SparseMatrix& A, Index k = 20) {
  const Index n = A.rows();
  Vector q = Vector::Random(n);
  q /= q.norm();
  Vector alpha(k);
  Vector beta(k);
  beta(0) = 0;
  Vector q_old = Vector::Zero(n);
  Vector q_cur = q;
  for (Index j = 0; j < k; ++j) {
    Vector v = A * q_cur;
    alpha(j) = q_cur.dot(v);
    if (j < k - 1) {
      v = v - alpha(j) * q_cur - beta(j) * q_old;
      beta(j + 1) = v.norm();
      if (beta(j + 1) < 1e-14) {
        alpha.conservativeResize(j + 1);
        beta.conservativeResize(j + 1);
        break;
      }
      q_old = q_cur;
      q_cur = v / beta(j + 1);
    }
  }
  Index m = alpha.size();
  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
  for (Index i = 0; i < m; ++i) {
    T(i, i) = alpha(i);
    if (i < m - 1) {
      T(i, i + 1) = beta(i + 1);
      T(i + 1, i) = beta(i + 1);
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
  return es.eigenvalues()(0);
}

}  // namespace FMCA
////////////////////////////////////////////////////////////////////////////////

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

FMCA::Matrix rhs(const H2SampletTree& hst, const FMCA::SparseMatrix& K) {
  srand(0);
  const FMCA::Index npts = K.cols();
  FMCA::Vector data(npts);
  FMCA::Vector x(npts);
  std::cout << "right hand side:              "
            << "sparse single-scale coeffs" << std::endl;
  x.setZero();
  for (FMCA::Index i = 0; i < 10; ++i) {
    FMCA::Index rdm = rand() % npts;
    while (x(rdm)) rdm = rand() % npts;
    x(rdm) = 1;
  }
  data = npts * hst.inverseSampletTransform(K.selfadjointView<Eigen::Upper>() *
                                            hst.sampletTransform(x));
  data /= 1 * data.cwiseAbs().maxCoeff();
  return data;
}

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("Matern32", 0.25 / sqrt(DIM));
  const FMCA::Matrix P = 0.5 * FMCA::Matrix::Random(DIM, NPTS).array();
  const FMCA::Scalar threshold = 1e-7;
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 5;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const Moments mom(P, mpole_deg);
  const MatrixEvaluator mat_eval(mom, function);
  std::cout << "dtilde:                       " << dtilde << std::endl;
  std::cout << "mpole_deg:                    " << mpole_deg << std::endl;
  std::cout << "eta:                          " << eta << std::endl;
  const SampletMoments samp_mom(P, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.tic();
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Scomp;
  Scomp.init(hst, eta, 100 * FMCA_ZERO_TOLERANCE);
  T.toc("planner:                     ");
  T.tic();
  Scomp.compress(mat_eval);
  T.toc("compressor:                  ");
  T.tic();
  const auto& ap_trips = Scomp.triplets();
  std::cout << "anz (a-priori):               "
            << std::round(ap_trips.size() / FMCA::Scalar(NPTS)) << std::endl;
  T.toc("triplets:                    ");

  T.tic();
  const auto& trips = Scomp.aposteriori_triplets_fast(threshold);
  std::cout << "anz (a-posteriori):           "
            << std::round(trips.size() / FMCA::Scalar(NPTS)) << std::endl;

  T.toc("triplets:                    ");
  {
    FMCA::Vector x(NPTS), y1(NPTS), y2(NPTS);
    FMCA::Scalar err = 0;
    FMCA::Scalar nrm = 0;
    for (auto i = 0; i < 10; ++i) {
      FMCA::Index index = rand() % P.cols();
      x.setZero();
      x(index) = 1;
      FMCA::Vector col = function.eval(P, P.col(hst.indices()[index]));
      y1 =
          col(Eigen::Map<const FMCA::iVector>(hst.indices(), hst.block_size()));
      x = hst.sampletTransform(x);
      y2.setZero();
      for (const auto& i : trips) {
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
  }
  Eigen::SparseMatrix<FMCA::Scalar> S(NPTS, NPTS);
  S.setFromTriplets(trips.begin(), trips.end());
  Eigen::SparseMatrix<FMCA::Scalar> Ssym = S.selfadjointView<Eigen::Upper>();
  Ssym *= 1. / FMCA::Scalar(NPTS);
  std::cout << "max eval: " << FMCA::powerIteration(Ssym) << std::endl;

  FMCA::Matrix data = rhs(hst, Ssym);
  //   FMCA::Vector data(NPTS);
  //   for (FMCA::Index i = 0; i < NPTS; ++i)
  //     data(i) = std::cos(10 * FMCA_PI * P.col(i).norm()) *
  //               std::exp(-4 * P.col(i).norm());
  // data = hst.toClusterOrder(data);
  FMCA::Matrix P3(3, P.cols());
  P3.topRows(2) = P;
  P3.bottomRows(1) = hst.toNaturalOrder(data).transpose();
  FMCA::IO::plotPointsColor("data.vtk", P3, hst.toNaturalOrder(data));
  FMCA::Matrix Tdata =
      1. / std::sqrt(FMCA::Scalar(NPTS)) * hst.sampletTransform(data);

  FMCA::Vector w(NPTS);
  FMCA::Vector x0(NPTS);
  x0.setZero();
  w.setOnes();
  w *= 1. / std::sqrt(FMCA::Scalar(NPTS));
  FMCA::ActiveSetManager asmgr;
  x0(0) = 1.;
  FMCA::Scalar ramp = 1;  // << 12;

  FMCA::Scalar L = FMCA::powerIteration(Ssym);
  FMCA::Scalar lambda = 2. / L;
  std::cout << "lambda:                      " << lambda << std::endl;

  FMCA::Scalar eta1 = 1e-4;
  FMCA::Scalar eta2 = 1e-1;
  FMCA::Scalar tau = 2 * 0.05 / (L * L * lambda * lambda + 2);
  FMCA::Scalar nu =
      0.5 *
      std::min(tau, 0.05 * (1 - tau / 2 * (L * L * lambda * lambda * 0.5 + 1)));
  std::cout << "tau:                         " << tau << std::endl;
  std::cout << "nu:                          " << nu << std::endl;
  FMCA::Index max_it = 50;
  FMCA::Scalar tol = 1e-8;

  FMCA::Vector x = x0;
  while (ramp > 1e-4) {
    std::cout << "------------------------------" << std::endl;
    std::cout << "ramp:                         " << ramp << std::endl;
    // x = FMCA::SSN(Ssym, Tdata, ramp * 1e0 * w, x, asmgr, max_it, tol);
    x = TRSSN(Ssym, Tdata, ramp * 1e0 * w, x, asmgr, lambda, eta1, eta2, tau, nu, max_it, tol);
    ramp *= 0.1;
  }
  x *= 1. / std::sqrt(FMCA::Scalar(NPTS));

  //////////////////// reconstruction
  FMCA::Vector Trec = hst.inverseSampletTransform(NPTS * Ssym * x);
  Trec = hst.toNaturalOrder(Trec);

  FMCA::Vector err = hst.toNaturalOrder(data) - Trec;
  FMCA::Scalar rel_err = err.norm() / data.norm();
  FMCA::Scalar rel_err_2 = err.squaredNorm() / data.squaredNorm();

  std::cout << " relative inf error = " << rel_err << std::endl;
  std::cout << " relative 2 error =   " << rel_err_2 << std::endl;

  std::cout << "------------------------" << std::endl;

  P3.bottomRows(1) = Trec.transpose();
  FMCA::IO::plotPointsColor("rec.vtk", P3, Trec);

  FMCA::Vector err_abs = err;
  for (FMCA::Index i = 0; i < err.size(); ++i) {
    err_abs(i) = std::abs(err(i));
  }
  P3.bottomRows(1) = err_abs.transpose();
  FMCA::IO::plotPointsColor("err.vtk", P3, err_abs);
  //   }

  return 0;
}
