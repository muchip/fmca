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

#define MSIDE 128    // landmarks per side, N = MSIDE^2
#define RENDER 512  // evaluation grid per side
#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

// ----------------------------- settings ------------------------------------
const FMCA::CovarianceKernel kernel("TPS2D", 1.);
const FMCA::Index dtilde = 4;
const FMCA::Index mpole_deg = 2 * (dtilde - 1);
const FMCA::Scalar eta = 0.5;
const FMCA::Scalar threshold = 1e-8;
const FMCA::Scalar lambda = 1e-6;  // numerical ridge
// ---------------------------------------------------------------------------

// ground truth deformation: swirl about the centre plus two Gaussian bumps
void g_true(FMCA::Scalar x, FMCA::Scalar y, FMCA::Scalar &gx,
                   FMCA::Scalar &gy) {
  const FMCA::Scalar dx = x - 0.5, dy = y - 0.5;
  const FMCA::Scalar s = std::sqrt(dx * dx + dy * dy) / 0.4;
  const FMCA::Scalar w = s < 1.0 ? (1.0 - s * s) * (1.0 - s * s) : 0.0;
  const FMCA::Scalar ca = std::cos(w), sa = std::sin(w);
  gx = 0.5 + ca * dx - sa * dy;
  gy = 0.5 + sa * dx + ca * dy;
  const FMCA::Scalar bx[2] = {0.30, 0.70}, by[2] = {0.30, 0.65};
  const FMCA::Scalar ax[2] = {0.060, -0.050}, ay[2] = {0.040, 0.050};
  const FMCA::Scalar bw[2] = {0.12, 0.10};
  for (FMCA::Index m = 0; m < 2; ++m) {
    const FMCA::Scalar e =
        std::exp(-(std::pow(x - bx[m], 2) + std::pow(y - by[m], 2)) /
                 (2 * bw[m] * bw[m]));
    gx += ax[m] * e;
    gy += ay[m] * e;
  }
}

// synthetic image, evaluated analytically instead of loading pixel data
FMCA::Scalar image(FMCA::Scalar x, FMCA::Scalar y) {
  return std::sin(8 * FMCA_PI * x) * std::sin(8 * FMCA_PI * y);
}

// fits both components of a TPS on the centres C, one Cholesky, two solves
void fit_tps(const FMCA::Matrix &C, const FMCA::Vector &fx,
                    const FMCA::Vector &fy, FMCA::Vector &cx, FMCA::Vector &cy,
                    FMCA::Vector &dx, FMCA::Vector &dy) {
  const FMCA::Index N = C.cols(), mq = 1 + DIM;  // monomials {1, x, y}
  const Moments mom(C, mpole_deg);
  const MatrixEvaluator mat_eval(mom, kernel);
  const SampletMoments samp_mom(C, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, C);
  FMCA::Matrix Pol(mq, N);
  for (FMCA::Index i = 0; i < N; ++i) {
    Pol(0, i) = 1.0;
    Pol(1, i) = C(0, i);
    Pol(2, i) = C(1, i);
  }
  const FMCA::Matrix TPol =
      hst.sampletTransform(hst.toClusterOrder(Pol.transpose()));
  const FMCA::Matrix S_block = TPol.topRows(mq);
  assert(TPol.bottomRows(N - mq).norm() / TPol.norm() < 1e-10 &&
         "polynomial block not annihilated");
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Scomp;
  Scomp.init(hst, eta, FMCA_ZERO_TOLERANCE);
  Scomp.compress(mat_eval);
  Scomp.triplets();
  const auto &trips = Scomp.aposteriori_triplets_fast(threshold);
  Eigen::SparseMatrix<FMCA::Scalar> Smat(N, N);
  Smat.setFromTriplets(trips.begin(), trips.end());
  Eigen::SparseMatrix<FMCA::Scalar> Kpsi = Smat.block(mq, mq, N - mq, N - mq);
  {
    Eigen::SparseMatrix<FMCA::Scalar> Id(N - mq, N - mq);
    Id.setIdentity();
    Kpsi += Id * lambda;
  }
  const Eigen::SparseMatrix<FMCA::Scalar> K_PPsi = Smat.block(0, mq, mq, N - mq);
  Eigen::SimplicialLLT<Eigen::SparseMatrix<FMCA::Scalar>, Eigen::Upper> llt;
  llt.compute(Kpsi);
  assert(llt.info() == Eigen::Success && "Cholesky failed");
  const Eigen::PartialPivLU<FMCA::Matrix> Slu(S_block);
  const FMCA::Vector Ufx = hst.sampletTransform(hst.toClusterOrder(fx));
  const FMCA::Vector Ufy = hst.sampletTransform(hst.toClusterOrder(fy));
  const FMCA::Vector cx_Psi = llt.solve(Ufx.tail(N - mq));
  const FMCA::Vector cy_Psi = llt.solve(Ufy.tail(N - mq));
  dx = Slu.solve(Ufx.head(mq) - K_PPsi * cx_Psi);
  dy = Slu.solve(Ufy.head(mq) - K_PPsi * cy_Psi);
  FMCA::Vector cs = FMCA::Vector::Zero(N);
  cs.tail(N - mq) = cx_Psi;
  cx = hst.toNaturalOrder(hst.inverseSampletTransform(cs));
  cs.setZero();
  cs.tail(N - mq) = cy_Psi;
  cy = hst.toNaturalOrder(hst.inverseSampletTransform(cs));
}

// evaluates the fitted warp at the points Peval, returns 2 x M
FMCA::Matrix eval_warp(const FMCA::Matrix &C, const FMCA::Vector &cx,
                              const FMCA::Vector &cy, const FMCA::Vector &dx,
                              const FMCA::Vector &dy,
                              const FMCA::Matrix &Peval) {
  const FMCA::Index N = C.cols(), M = Peval.cols();
  FMCA::Matrix out(2, M);
#pragma omp parallel for schedule(static)
  for (FMCA::Index p = 0; p < M; ++p) {
    FMCA::Scalar sx = 0, sy = 0;
    for (FMCA::Index j = 0; j < N; ++j) {
      const FMCA::Scalar r2 = (Peval.col(p) - C.col(j)).squaredNorm();
      const FMCA::Scalar phi = r2 > 0 ? 0.5 * r2 * std::log(r2) : 0;
      sx += cx(j) * phi;
      sy += cy(j) * phi;
    }
    out(0, p) = sx + dx(0) + dx(1) * Peval(0, p) + dx(2) * Peval(1, p);
    out(1, p) = sy + dy(0) + dy(1) * Peval(0, p) + dy(2) * Peval(1, p);
  }
  return out;
}

int main() {
  FMCA::Tictoc T;
  const FMCA::Index N = MSIDE * MSIDE;

  // landmarks X on a regular grid and targets Y = g(X)
  FMCA::Matrix X(DIM, N), Y(DIM, N);
  for (FMCA::Index i = 0, k = 0; i < MSIDE; ++i)
    for (FMCA::Index j = 0; j < MSIDE; ++j, ++k) {
      X(0, k) = 0.05 + 0.90 * i / (MSIDE - 1);
      X(1, k) = 0.05 + 0.90 * j / (MSIDE - 1);
      g_true(X(0, k), X(1, k), Y(0, k), Y(1, k));
    }
  std::cout << "landmarks:                    " << N << std::endl;
  std::cout << "dtilde:                       " << dtilde << std::endl;
  std::cout << "eta:                          " << eta << std::endl;
  std::cout << "ridge:                        " << lambda << std::endl;

  // forward warp Phi (X -> Y) and inverse warp Psi (Y -> X)
  FMCA::Vector cxf, cyf, dxf, dyf, cxi, cyi, dxi, dyi;
  T.tic();
  fit_tps(X, Y.row(0).transpose(), Y.row(1).transpose(), cxf, cyf, dxf, dyf);
  fit_tps(Y, X.row(0).transpose(), X.row(1).transpose(), cxi, cyi, dxi, dyi);
  T.toc("fit Phi + Psi:               ");

  // landmark residual
  const FMCA::Matrix PhiX = eval_warp(X, cxf, cyf, dxf, dyf, X);
  FMCA::Scalar rms = 0;
  for (FMCA::Index i = 0; i < N; ++i)
    rms += (PhiX.col(i) - Y.col(i)).squaredNorm();
  rms = std::sqrt(rms / N);
  std::cout << "landmark rms:                 " << rms << std::endl;
  assert(rms < 1e-4 && "landmark residual too large");

  // field recovery and Jacobian on the evaluation grid
  const FMCA::Index Ng = RENDER * RENDER;
  FMCA::Matrix G(DIM, Ng);
  for (FMCA::Index i = 0, k = 0; i < RENDER; ++i)
    for (FMCA::Index j = 0; j < RENDER; ++j, ++k) {
      G(0, k) = FMCA::Scalar(i) / (RENDER - 1);
      G(1, k) = FMCA::Scalar(j) / (RENDER - 1);
    }
  T.tic();
  const FMCA::Matrix PhiG = eval_warp(X, cxf, cyf, dxf, dyf, G);
  const FMCA::Matrix PsiG = eval_warp(Y, cxi, cyi, dxi, dyi, G);
  T.toc("warp evaluation:             ");
  FMCA::Scalar num = 0, den = 0;
  for (FMCA::Index k = 0; k < Ng; ++k) {
    FMCA::Scalar gx, gy;
    g_true(G(0, k), G(1, k), gx, gy);
    num += std::pow(PhiG(0, k) - gx, 2) + std::pow(PhiG(1, k) - gy, 2);
    den += gx * gx + gy * gy;
  }
  const FMCA::Scalar field_err = std::sqrt(num / den);
  std::cout << "field recovery |Phi-g|/|g|:   " << field_err << std::endl;
  assert(field_err < 1e-2 && "field recovery error too large");
  const FMCA::Scalar h = 1.0 / (RENDER - 1);
  FMCA::Vector det = FMCA::Vector::Ones(Ng);
  FMCA::Scalar detmin = FMCA_INF;
  for (FMCA::Index i = 1; i + 1 < RENDER; ++i)
    for (FMCA::Index j = 1; j + 1 < RENDER; ++j) {
      const FMCA::Index k = i * RENDER + j;
      const FMCA::Scalar ax = (PhiG(0, k + RENDER) - PhiG(0, k - RENDER)) / (2 * h);
      const FMCA::Scalar ay = (PhiG(0, k + 1) - PhiG(0, k - 1)) / (2 * h);
      const FMCA::Scalar bx = (PhiG(1, k + RENDER) - PhiG(1, k - RENDER)) / (2 * h);
      const FMCA::Scalar by = (PhiG(1, k + 1) - PhiG(1, k - 1)) / (2 * h);
      det(k) = ax * by - ay * bx;
      detmin = std::min(detmin, det(k));
    }
  std::cout << "min det(grad Phi):            " << detmin << std::endl;
  assert(detmin > 0 && "the recovered deformation is folding");

  // backward warping of the analytic image and export for visual inspection
  FMCA::Vector I0(Ng), Iwarp(Ng), disp(Ng);
  for (FMCA::Index k = 0; k < Ng; ++k) {
    I0(k) = image(G(0, k), G(1, k));
    Iwarp(k) = image(PsiG(0, k), PsiG(1, k));
    disp(k) = (PhiG.col(k) - G.col(k)).norm();
  }
  FMCA::Matrix G3 = FMCA::Matrix::Zero(3, Ng);
  G3.topRows(DIM) = G;
  FMCA::IO::plotPointsColor("registration_original.vtk", G3, I0);
  FMCA::IO::plotPointsColor("registration_warped.vtk", G3, Iwarp);
  FMCA::IO::plotPointsColor("registration_displacement.vtk", G3, disp);
  FMCA::IO::plotPointsColor("registration_jacobian.vtk", G3, det);
  std::cout << "wrote registration_{original,warped,displacement,jacobian}.vtk"
            << std::endl;
  return 0;
}
