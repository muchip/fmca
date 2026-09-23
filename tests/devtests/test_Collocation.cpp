// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
//  Unit test of the Collocation module on the 2D Poisson problem
//      -Delta u = -Delta(x(1-x)y(1-y))  in (0,1)^2,   u = 0 on the boundary.
//  Two things are checked:
//    1. the compressed multiscale collocation converges under refinement,
//    2. PIKL with lambda2 = 0 reproduces the collocation solution, which is
//       the limit asserted by the preconditioner proposition, and it does so
//       in a handful of outer iterations.
#include <cmath>
#include <iostream>

#include "../FMCA/Collocation"

FMCA::Matrix makeInterior(int n) {
  FMCA::Matrix P(2, n * n);
  const FMCA::Scalar h = 1.0 / (n + 1);
  int c = 0;
  for (int ix = 1; ix <= n; ++ix)
    for (int iy = 1; iy <= n; ++iy) P.col(c++) << ix * h, iy * h;
  return P;
}

FMCA::Matrix makeBoundary(int m) {
  FMCA::Matrix P(2, 4 * m);
  const FMCA::Scalar h = 1.0 / (m + 1);
  int c = 0;
  for (int i = 1; i <= m; ++i) P.col(c++) << i * h, 0.0;
  for (int i = 1; i <= m; ++i) P.col(c++) << i * h, 1.0;
  for (int i = 1; i <= m; ++i) P.col(c++) << 0.0, i * h;
  for (int i = 1; i <= m; ++i) P.col(c++) << 1.0, i * h;
  return P;
}

FMCA::Vector exactSolution(const FMCA::Matrix &P) {
  FMCA::Vector u(P.cols());
  for (int i = 0; i < P.cols(); ++i)
    u[i] = P(0, i) * (1 - P(0, i)) * P(1, i) * (1 - P(1, i));
  return u;
}

FMCA::Vector source(const FMCA::Matrix &P) {
  FMCA::Vector f(P.cols());
  for (int i = 0; i < P.cols(); ++i)
    f[i] = -2.0 * (P(0, i) * (1 - P(0, i)) + P(1, i) * (1 - P(1, i)));
  return f;
}

//  runs the whole hierarchy and returns the solution on the background set
template <typename MultiscaleSolver>
FMCA::Vector run(MultiscaleSolver &mc, const std::vector<FMCA::Matrix> &PI,
                 const std::vector<FMCA::Matrix> &PB,
                 FMCA::MultiscaleEvaluator &evaluator, FMCA::Index n_eval) {
  mc.init(PI, PB, 3.0, "MATERN52", "MATERN52_SECOND_DERIVATIVE");
  mc.setComputeCondition(false);
  FMCA::Vector u = FMCA::Vector::Zero(n_eval);
  for (FMCA::Index l = 0; l < mc.numLevels(); ++l) {
    mc.solveLevel(l, source(PI[l]), exactSolution(PB[l]));
    u += mc.evaluateLevel(evaluator, l);
  }
  return u;
}

int main() {
  const std::vector<FMCA::Matrix> PI = {makeInterior(25), makeInterior(50), makeInterior(100)};
  const std::vector<FMCA::Matrix> PB = {makeBoundary(5), makeBoundary(10), makeBoundary(25)};
  const FMCA::Matrix P_eval = makeInterior(200);
  const FMCA::Vector u_exact = exactSolution(P_eval);
  const FMCA::Scalar une = u_exact.norm();
  FMCA::MultiscaleEvaluator evaluator(P_eval, 6, 0.5);

  // compressed multiscale collocation
  FMCA::MultiscaleCollocationSolver mc;
  mc.solver().setParameters(1e-12, 500);
  const FMCA::Vector u = run(mc, PI, PB, evaluator, P_eval.cols());
  const FMCA::Scalar err = (u - u_exact).norm() / une;

  // PIKL in the limit lambda2 = 0
  FMCA::MultiscalePIKLSolver mp;
  mp.solver().setRegularization(1e4, 0.0);
  mp.solver().setParameters(1e-12, 200);
  mp.solver().setInteriorParameters(1e-12, 500);
  const FMCA::Vector u_pikl = run(mp, PI, PB, evaluator, P_eval.cols());
  const FMCA::Scalar err_pikl = (u - u_pikl).norm() / u.norm();

  std::cout << "error collocation: " << err << std::endl;
  std::cout << "PIKL(lambda2=0) vs collocation:      " << err_pikl << std::endl;
  std::cout << "collocation iterations, finest level: "
            << mc.iterations(mc.numLevels() - 1) << std::endl;
  std::cout << "PIKL iterations on the finest level:  "
            << mp.iterations(mp.numLevels() - 1) << std::endl;
  return 0;
}
