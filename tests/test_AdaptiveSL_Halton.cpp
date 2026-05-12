// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer, Sara Avesani
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//

// ---------------------------------------------------------------------------
// Adaptive semi-Lagrangian Burgers on a 2D Halton point cloud.
// At each time step the detector is run on the *current* u and the SL
// step combines kernel (high-order) and k-NN (low-order) reconstructions
// per flag.

#include <cstdio>
#include <iostream>

#include "../FMCA/SemiLagrangian"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;

namespace {

// Gaussian bump centred at (x0, y0), peak amplitude A, width sigma.
// Burgers self-advection u_t + u u_x = 0 transports the top of the
// bump to the right at speed = u. The right side compresses into a
// shock at t ~ sigma / A.
FMCA::Scalar initialCondition(FMCA::Scalar x, FMCA::Scalar y) {
  constexpr FMCA::Scalar x0 = -0.5;
  constexpr FMCA::Scalar y0 = 0.5;
  constexpr FMCA::Scalar A = 1.0;
  constexpr FMCA::Scalar sigma = 0.15;
  const FMCA::Scalar r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
  return A * std::exp(-r2 / (sigma * sigma));
}

FMCA::Scalar haltonOne(FMCA::Index n, FMCA::Index base) {
  FMCA::Scalar f = 1.0;
  FMCA::Scalar r = 0.0;
  while (n > 0) {
    f /= base;
    r += f * (n % base);
    n /= base;
  }
  return r;
}

FMCA::Matrix haltonPoints2D(FMCA::Index N, FMCA::Scalar xmin, FMCA::Scalar xmax,
                            FMCA::Scalar ymin, FMCA::Scalar ymax) {
  using namespace FMCA;
  Matrix P(2, N);
  for (Index i = 0; i < N; ++i) {
    P(0, i) = xmin + (xmax - xmin) * haltonOne(i + 1, 2);
    P(1, i) = ymin + (ymax - ymin) * haltonOne(i + 1, 3);
  }
  return P;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////////////
int main() {
  using namespace FMCA;
  Tictoc T;

  // ---------------------------------------------------------------- points
  constexpr Index N = 20000;
  const Scalar xmin = -1.21, xmax = 0.42, ymin = 0.01, ymax = 1.0;
  Matrix P = haltonPoints2D(N, xmin, xmax, ymin, ymax);

  Vector u(N);
  for (Index i = 0; i < N; ++i) u(i) = initialCondition(P(0, i), P(1, i));
  const Scalar u0_min = u.minCoeff();
  const Scalar u0_max = u.maxCoeff();

  // ---------------------------------------------------------------- param
  const Index dtilde = 6;
  const Scalar eta = 0.5;
  const Scalar threshold = 1e-8;
  const Scalar ridgep = 1e-10;  
  const Index k_low = 4;
  const Scalar alpha_thr = 2.5; 
  const Scalar smooth_thr = 1e-6;

  const Moments mom(P, 2 * (dtilde - 1));
  const SampletMoments samp_mom(P, dtilde - 1);
  const FMCA::H2SampletTree<FMCA::ClusterTree> hst(mom, samp_mom, 0, P);
  const FMCA::Vector minvec = minDistanceVector(hst, P);
  FMCA::Scalar fill_distance = minvec.maxCoeff();
  std::cout << "Fill Distance: " << fill_distance << std::endl;
  const CovarianceKernel kernel("MATERN52", 4 * fill_distance); 

  AdaptiveSemiLagrangian2D solver;
  T.tic();
  solver.init(P, kernel, dtilde, ridgep, k_low, eta, threshold);
  std::cout << "compression error:        " << solver.compressionError()
            << "\n";
  std::cout << "interp. residual on u0:   " << solver.interpolationResidual(u)
            << "\n";
  T.toc("solver init time:            ");
  solver.setValueClamp(u0_min, u0_max);

  // ---------------------------------------------------------------- detector
  {
    const Scalar x_jump = 0.015;
    Vector u_test(N);
    for (Index i = 0; i < N; ++i) u_test(i) = (P(0, i) > x_jump) ? 1.0 : 0.0;
    const auto test_flags = solver.detectFlags(u_test, alpha_thr, smooth_thr);
    Index n_flagged_test = 0;
    Scalar x_flag_min = std::numeric_limits<Scalar>::infinity();
    Scalar x_flag_max = -std::numeric_limits<Scalar>::infinity();
    for (Index i = 0; i < N; ++i)
      if (test_flags[i]) {
        n_flagged_test++;
        x_flag_min = std::min(x_flag_min, P(0, i));
        x_flag_max = std::max(x_flag_max, P(0, i));
      }
    std::cout << "detected " << n_flagged_test << "/" << N;
    if (n_flagged_test) {
      std::cout << "  band = [" << x_flag_min << ", " << x_flag_max << "]";
    }
    std::cout << "\n";
  }

  // ---------------------------------------------------------------- time loop
  const Scalar dt = 0.01;
  const Scalar Tf = 0.3;
  const Index nsteps = static_cast<Index>(std::round(Tf / dt));
  const auto rk = AdaptiveSemiLagrangian2D::RKMethod::RK4;

  Matrix P3(3, N);
  P3.topRows(2) = P;
  P3.row(2).setZero();

  auto dump = [&](Index step, const Vector& u, const std::vector<int>& flags) {
    char fname[256];
    Vector flagv(N);
    for (Index i = 0; i < N; ++i) flagv(i) = flags[i];
    std::snprintf(fname, sizeof(fname), "halton_u_%04u.vtk",
                  static_cast<unsigned>(step));
    IO::plotPointsColor(fname, P3, u);
    std::snprintf(fname, sizeof(fname), "halton_flag_%04u.vtk",
                  static_cast<unsigned>(step));
    IO::plotPointsColor(fname, P3, flagv);
    Matrix P3u(3, N);
    P3u.topRows(2) = P;
    P3u.row(2) = u;
    std::snprintf(fname, sizeof(fname), "halton_u3d_%04u.vtk",
                  static_cast<unsigned>(step));
    IO::plotPointsColor(fname, P3u, u);
  };

  std::vector<int> flags(N, 0);
  T.tic();
  for (Index n = 0; n <= nsteps; ++n) {
    flags = solver.detectFlags(u, alpha_thr, smooth_thr);
    Index n_flagged = 0;
    Scalar x_flag_min = std::numeric_limits<Scalar>::infinity();
    Scalar x_flag_max = -std::numeric_limits<Scalar>::infinity();
    for (Index i = 0; i < N; ++i)
      if (flags[i]) {
        n_flagged++;
        x_flag_min = std::min(x_flag_min, P(0, i));
        x_flag_max = std::max(x_flag_max, P(0, i));
      }
    std::cout << "step " << n << "  t = " << n * dt
              << "  flagged = " << n_flagged << "/" << N;
    if (n_flagged)
      std::cout << "  band = [" << x_flag_min << ", " << x_flag_max << "]";
    dump(n, u, flags);
    if (n < nsteps) {
      Vector u_prev = u;
      u = solver.burgersStep(u, dt, rk, flags);
      const Scalar du_inf = (u - u_prev).cwiseAbs().maxCoeff();
      const Scalar du_l2 = (u - u_prev).norm() / std::max(u_prev.norm(), 1.0);
      std::cout << "  ||du||_inf = " << du_inf << "  ||du||_l2 = " << du_l2;
    }
    std::cout << "\n";
  }
  T.toc("total time-stepping:         ");
  return 0;
}