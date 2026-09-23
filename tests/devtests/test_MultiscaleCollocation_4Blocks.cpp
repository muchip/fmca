#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

//////////
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/UmfPackSupport>
#include <unsupported/Eigen/IterativeSolvers>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/KernelInterpolation"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2

using namespace FMCA;

using Interpolator = TotalDegreeInterpolator;
using SampletInterpolator = MonomialInterpolator;
using Moments = NystromMoments<Interpolator>;
using SampletMoments = NystromSampletMoments<SampletInterpolator>;
using Eval = unsymmetricNystromEvaluator<Moments, CovarianceKernel>;
using EvalDeriv = unsymmetricNystromEvaluator<Moments, GradKernel>;
using H2ST = H2SampletTree<ClusterTree>;
using Trip = Eigen::Triplet<double>;

// Interior grid on (0,1)^2, boundary excluded
Matrix makeInteriorGrid(int N) {
  int n = (int)std::sqrt((double)N);
  Scalar h = 1.0 / (n + 1);
  Matrix P(DIM, n * n);
  int idx = 0;
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) {
      P(0, idx) = i * h;
      P(1, idx) = j * h;
      ++idx;
    }
  return P;
}

// Boundary of [0,1]^2, corners excluded, N must be divisible by 4
Matrix makeBorderGrid(int N) {
  assert(N % 4 == 0);
  int s = N / 4;
  Scalar h = 1.0 / (s + 1);
  Matrix P(DIM, N);
  int col = 0;
  for (int i = 1; i <= s; ++i) {
    P(0, col) = i * h;
    P(1, col) = 0.0;
    ++col;
  }
  for (int i = 1; i <= s; ++i) {
    P(0, col) = i * h;
    P(1, col) = 1.0;
    ++col;
  }
  for (int i = 1; i <= s; ++i) {
    P(0, col) = 0.0;
    P(1, col) = i * h;
    ++col;
  }
  for (int i = 1; i <= s; ++i) {
    P(0, col) = 1.0;
    P(1, col) = i * h;
    ++col;
  }
  return P;
}

int main() {
  // -----------------------------------------------------------------
  // Point sets per level (coarse to fine)
  // -----------------------------------------------------------------
  const std::vector<Matrix> P_int_levels = {
      makeInteriorGrid(256),   makeInteriorGrid(1024),
      makeInteriorGrid(4096),  makeInteriorGrid(16384)};
  const std::vector<Matrix> P_bdr_levels = {
      makeBorderGrid(16),  makeBorderGrid(32),  makeBorderGrid(64),
      makeBorderGrid(128)};

  // PDE: -Delta u = f on interior, u = g on boundary
  // Exact solution: u(x,y) = sin(pi x) sin(pi y)
  auto f_rhs = [](Scalar x, Scalar y) {
    return -2. * FMCA_PI * FMCA_PI * std::sin(FMCA_PI * x) *
           std::sin(FMCA_PI * y);
  };
  auto u_bc = [](Scalar, Scalar) { return 0.0; };

  const Scalar nu = 5.0;
  const Scalar eta = 1.0 / DIM;
  const Index dtilde = 4;
  const Scalar thr_kernel = 1e-6;
  const Index mpole_deg = 2 * (dtilde - 1);
  const std::string ktype = "MATERN52";
  const std::string ktype_lap = "MATERN52_SECOND_DERIVATIVE";
  const int NEVAL = 100000;
  Matrix P_eval = (Matrix::Random(DIM, NEVAL).array() + 1.0) / 2.0;

  const int L = (int)P_int_levels.size();
  std::vector<Matrix> P(L);
  std::vector<int> N_int(L), N_bdr(L), N(L);
  std::vector<Scalar> sigma(L);

  std::vector<std::unique_ptr<Moments>> mom_int(L), mom_bdr(L);
  std::vector<std::unique_ptr<SampletMoments>> smom_int(L), smom_bdr(L);
  std::vector<std::unique_ptr<H2ST>> hst_int(L), hst_bdr(L);

  for (int l = 0; l < L; ++l) {
    N_int[l] = (int)P_int_levels[l].cols();
    N_bdr[l] = (int)P_bdr_levels[l].cols();
    N[l] = N_int[l] + N_bdr[l];

    P[l].resize(DIM, N[l]);
    P[l].leftCols(N_int[l]) = P_int_levels[l];
    P[l].rightCols(N_bdr[l]) = P_bdr_levels[l];

    mom_int[l] = std::make_unique<Moments>(P_int_levels[l], mpole_deg);
    smom_int[l] = std::make_unique<SampletMoments>(P_int_levels[l], dtilde - 1);
    hst_int[l] =
        std::make_unique<H2ST>(*mom_int[l], *smom_int[l], 0, P_int_levels[l]);

    mom_bdr[l] = std::make_unique<Moments>(P_bdr_levels[l], mpole_deg);
    smom_bdr[l] = std::make_unique<SampletMoments>(P_bdr_levels[l], dtilde - 1);
    hst_bdr[l] =
        std::make_unique<H2ST>(*mom_bdr[l], *smom_bdr[l], 0, P_bdr_levels[l]);

    ClusterTree CT(P[l], 0);
    sigma[l] = nu * minDistanceVector(CT, P[l]).maxCoeff();
  }

  // -----------------------------------------------------------------
  std::vector<Vector> residuals(L);
  for (int l = 0; l < L; ++l) {
    Vector f_l(N_int[l]);
    for (int i = 0; i < N_int[l]; ++i)
      f_l[i] = f_rhs(P_int_levels[l](0, i), P_int_levels[l](1, i));

    Vector g_l(N_bdr[l]);
    for (int i = 0; i < N_bdr[l]; ++i)
      g_l[i] = u_bc(P_bdr_levels[l](0, i), P_bdr_levels[l](1, i));

    residuals[l].resize(N[l]);
    residuals[l] << hst_int[l]->sampletTransform(
        hst_int[l]->toClusterOrder(f_l)),
        hst_bdr[l]->sampletTransform(hst_bdr[l]->toClusterOrder(g_l));
  }

  // -----------------------------------------------------------------
  // Multiscale forward substitution.
  // -----------------------------------------------------------------
  Tictoc T;
  std::vector<Vector> alpha(L);
  std::vector<Scalar> t_compress(L), t_factor(L), t_solve(L);
  std::vector<int> anz_diag(L);

  for (int l = 0; l < L; ++l) {
    std::cout << "------ Level " << l + 1 << "  (N = " << N[l] << ") ------\n";
    // ===== Cross blocks: residuals[l] -= A_lj * alpha[j] =====
    for (int j = 0; j < l; ++j) {
      const Scalar sig = sigma[j];
      std::vector<Trip> trips;
      // (I,I): Delta_x K(P_int_l, P_int_j)
      for (int d = 0; d < DIM; ++d) {
        GradKernel gk(ktype_lap, sig, 1, d);
        EvalDeriv ev(*mom_int[l], *mom_int[j], gk);
        internal::SampletMatrixCompressorUnsymmetric<H2ST> sc;
        sc.init(*hst_int[l], *hst_int[j], eta, thr_kernel);
        sc.compress(ev);
        for (const auto& t : sc.triplets())
          trips.emplace_back(t.row(), t.col(), t.value());
      }
      // (I,B): Delta_x K(P_int_l, P_bdr_j)  -- col offset N_int[j]
      for (int d = 0; d < DIM; ++d) {
        GradKernel gk(ktype_lap, sig, 1, d);
        EvalDeriv ev(*mom_int[l], *mom_bdr[j], gk);
        internal::SampletMatrixCompressorUnsymmetric<H2ST> sc;
        sc.init(*hst_int[l], *hst_bdr[j], eta, thr_kernel);
        sc.compress(ev);
        for (const auto& t : sc.triplets())
          trips.emplace_back(t.row(), t.col() + N_int[j], t.value());
      }
      // (B,I): K(P_bdr_l, P_int_j)  -- row offset N_int[l]
      {
        CovarianceKernel kernel(ktype, sig);
        Eval ev(*mom_bdr[l], *mom_int[j], kernel);
        internal::SampletMatrixCompressorUnsymmetric<H2ST> sc;
        sc.init(*hst_bdr[l], *hst_int[j], eta, thr_kernel);
        sc.compress(ev);
        for (const auto& t : sc.triplets())
          trips.emplace_back(t.row() + N_int[l], t.col(), t.value());
      }
      // (B,B): K(P_bdr_l, P_bdr_j)  -- row/col offsets N_int[l]/N_int[j]
      {
        CovarianceKernel kernel(ktype, sig);
        Eval ev(*mom_bdr[l], *mom_bdr[j], kernel);
        internal::SampletMatrixCompressorUnsymmetric<H2ST> sc;
        sc.init(*hst_bdr[l], *hst_bdr[j], eta, thr_kernel);
        sc.compress(ev);
        for (const auto& t : sc.triplets())
          trips.emplace_back(t.row() + N_int[l], t.col() + N_int[j], t.value());
      }
      SparseMatrix A_lj(N[l], N[j]);
      A_lj.setFromTriplets(trips.begin(), trips.end());
      residuals[l] -= A_lj * alpha[j];
    }
    // ===== Diagonal block A_ll =====
    T.tic();
    const Scalar sig = sigma[l];
    std::vector<Trip> trips;
    // (I,I) SYMMETRIC: same tree both sides; mirror upper -> lower.
    for (int d = 0; d < DIM; ++d) {
      GradKernel gk(ktype_lap, sig, 1, d);
      EvalDeriv ev(*mom_int[l], *mom_int[l], gk);
      internal::SampletMatrixCompressor<H2ST> sc;
      sc.init(*hst_int[l], eta, thr_kernel);
      sc.compress(ev);
      for (const auto& t : sc.triplets()) {
        trips.emplace_back(t.row(), t.col(), t.value());
        if (t.row() != t.col()) trips.emplace_back(t.col(), t.row(), t.value());
      }
    }
    // (I,B) unsymmetric: rows hst_int[l], cols hst_bdr[l] -- col offset
    // N_int[l]
    for (int d = 0; d < DIM; ++d) {
      GradKernel gk(ktype_lap, sig, 1, d);
      EvalDeriv ev(*mom_int[l], *mom_bdr[l], gk);
      internal::SampletMatrixCompressorUnsymmetric<H2ST> sc;
      sc.init(*hst_int[l], *hst_bdr[l], eta, thr_kernel);
      sc.compress(ev);
      for (const auto& t : sc.triplets())
        trips.emplace_back(t.row(), t.col() + N_int[l], t.value());
    }
    // (B,I) unsymmetric: rows hst_bdr[l], cols hst_int[l] -- row offset
    // N_int[l]
    {
      CovarianceKernel kernel(ktype, sig);
      Eval ev(*mom_bdr[l], *mom_int[l], kernel);
      internal::SampletMatrixCompressorUnsymmetric<H2ST> sc;
      sc.init(*hst_bdr[l], *hst_int[l], eta, thr_kernel);
      sc.compress(ev);
      for (const auto& t : sc.triplets())
        trips.emplace_back(t.row() + N_int[l], t.col(), t.value());
    }
    // (B,B) SYMMETRIC: same tree both sides; mirror upper -> lower.
    {
      CovarianceKernel kernel(ktype, sig);
      Eval ev(*mom_bdr[l], *mom_bdr[l], kernel);
      internal::SampletMatrixCompressor<H2ST> sc;
      sc.init(*hst_bdr[l], eta, thr_kernel);
      sc.compress(ev);
      for (const auto& t : sc.triplets()) {
        const int r = t.row() + N_int[l];
        const int c = t.col() + N_int[l];
        trips.emplace_back(r, c, t.value());
        if (t.row() != t.col()) trips.emplace_back(c, r, t.value());
      }
    }

    SparseMatrix A_ll(N[l], N[l]);
    A_ll.setFromTriplets(trips.begin(), trips.end());
    A_ll.makeCompressed();
    t_compress[l] = T.toc();
    anz_diag[l] = (int)(A_ll.nonZeros() / N[l]);

    {
      Vector diag = A_ll.diagonal();
      std::cerr << "  |diag|min=" << std::scientific << std::setprecision(2)
                << diag.array().abs().minCoeff()
                << "  |diag|max=" << diag.array().abs().maxCoeff() << "\n";
    }

    // =====================================================================
    // Solvers tried.
    // =====================================================================

    // ---- SparseLU + Metis (DIRECT, currently active) ----
    // T.tic();
    // Eigen::SparseLU<SparseMatrix, Eigen::MetisOrdering<int>> solver;
    // solver.analyzePattern(A_ll);
    // solver.factorize(A_ll);
    // t_factor[l] = T.toc();
    // if (solver.info() != Eigen::Success)
    //   throw std::runtime_error("SparseLU failed at level " + std::to_string(l
    //   + 1));
    // T.tic();
    // alpha[l] = solver.solve(residuals[l]);
    // t_solve[l] = T.toc();

    // ---- UMFPACK (DIRECT) ---- \\ failed level 5 but pretty fast until level
    // 5 T.tic(); Eigen::UmfPackLU<SparseMatrix> solver;
    // solver.umfpackControl()[UMFPACK_PIVOT_TOLERANCE] = 0.0;
    // solver.compute(A_ll);
    // t_factor[l] = T.toc();
    // if (solver.info() != Eigen::Success)
    //   throw std::runtime_error("UMFPACK failed at level " +
    //                            std::to_string(l + 1));
    // T.tic();
    // alpha[l] = solver.solve(residuals[l]);
    // t_solve[l] = T.toc();

    // ---- BiCGSTAB + ILUT (ITERATIVE) ---- \\ residual explodes from level 2 + slow; 
    // T.tic(); Eigen::BiCGSTAB<SparseMatrix,
    // Eigen::IncompleteLUT<Scalar>> solver;
    // solver.preconditioner().setDroptol(1e-3);
    // solver.preconditioner().setFillfactor(40);
    // solver.setMaxIterations(2000);
    // solver.setTolerance(1e-10);
    // solver.compute(A_ll);
    // t_factor[l] = T.toc();
    // if (solver.info() != Eigen::Success)
    //   throw std::runtime_error("BiCGSTAB setup failed at level " +
    //                            std::to_string(l + 1));
    // T.tic();
    // alpha[l] = solver.solve(residuals[l]);
    // t_solve[l] = T.toc();
    // std::cout << "  BiCGSTAB it=" << solver.iterations()
    //           << "  err=" << std::scientific << solver.error() << "\n";

    // ---- GMRES + ILUT (ITERATIVE) ---- \\ residual explodes from level 2 + slow;
    // T.tic(); Eigen::GMRES<SparseMatrix, Eigen::IncompleteLUT<Scalar>>
    // solver; solver.preconditioner().setDroptol(1e-3);
    // solver.preconditioner().setFillfactor(20);
    // solver.set_restart(50);
    // solver.setMaxIterations(2000);
    // solver.setTolerance(1e-10);
    // solver.compute(A_ll);
    // t_factor[l] = T.toc();
    // if (solver.info() != Eigen::Success)
    //   throw std::runtime_error("GMRES setup failed at level " +
    //   std::to_string(l + 1));
    // T.tic();
    // alpha[l] = solver.solve(residuals[l]);
    // t_solve[l] = T.toc();
    // std::cout << "  GMRES it=" << solver.iterations()
    //           << "  err=" << std::scientific << solver.error() << "\n";

    // ---- Schur complement (DIRECT) ----
    // Split A_ll into 4 blocks, factorize the large interior block A_II once,
    // then reduce to a tiny dense (N_bdr x N_bdr) Schur system on the boundary
    // unknowns and back-substitute.
    //
    //   [A_II  A_IB] [a_I]   [b_I]
    //   [A_BI  A_BB] [a_B] = [b_B]
    //
    //   Y       = A_II^-1 A_IB           (N_int x N_bdr,  N_bdr back-solves)
    //   y_I     = A_II^-1 b_I            (one back-solve)
    //   S       = A_BB - A_BI * Y        (dense N_bdr x N_bdr)
    //   a_B     = S^-1 (b_B - A_BI y_I)  (dense LU on a tiny matrix)
    //   a_I     = y_I - Y * a_B          (reuses Y, no extra solve)
    T.tic();
    SparseMatrix A_II(N_int[l], N_int[l]);
    SparseMatrix A_IB(N_int[l], N_bdr[l]);
    SparseMatrix A_BI(N_bdr[l], N_int[l]);
    SparseMatrix A_BB(N_bdr[l], N_bdr[l]);
    {
      std::vector<Trip> tII, tIB, tBI, tBB;
      for (int k = 0; k < A_ll.outerSize(); ++k)
        for (SparseMatrix::InnerIterator it(A_ll, k); it; ++it) {
          const int r = it.row(), c = it.col();
          const Scalar v = it.value();
          if (r < N_int[l] && c < N_int[l]) tII.emplace_back(r, c, v);
          else if (r < N_int[l])            tIB.emplace_back(r, c - N_int[l], v);
          else if (c < N_int[l])            tBI.emplace_back(r - N_int[l], c, v);
          else                              tBB.emplace_back(r - N_int[l], c - N_int[l], v);
        }
      A_II.setFromTriplets(tII.begin(), tII.end()); A_II.makeCompressed();
      A_IB.setFromTriplets(tIB.begin(), tIB.end()); A_IB.makeCompressed();
      A_BI.setFromTriplets(tBI.begin(), tBI.end()); A_BI.makeCompressed();
      A_BB.setFromTriplets(tBB.begin(), tBB.end()); A_BB.makeCompressed();
    }

    Eigen::SparseLU<SparseMatrix, Eigen::MetisOrdering<int>> luII;
    luII.analyzePattern(A_II);
    luII.factorize(A_II);
    if (luII.info() != Eigen::Success)
      throw std::runtime_error("A_II factorize failed at level " +
                               std::to_string(l + 1));
    t_factor[l] = T.toc();

    T.tic();
    const Vector b_I = residuals[l].head(N_int[l]);
    const Vector b_B = residuals[l].tail(N_bdr[l]);

    Matrix Y = luII.solve(Matrix(A_IB));        // N_int x N_bdr (dense)
    Vector y_I = luII.solve(b_I);                // N_int
    Matrix S = Matrix(A_BB) - A_BI * Y;          // N_bdr x N_bdr (dense)
    Vector rhs_B = b_B - A_BI * y_I;             // N_bdr
    Vector alpha_B = S.partialPivLu().solve(rhs_B);
    Vector alpha_I = y_I - Y * alpha_B;          // reuse Y, no extra solve

    alpha[l].resize(N[l]);
    alpha[l].head(N_int[l]) = alpha_I;
    alpha[l].tail(N_bdr[l]) = alpha_B;
    t_solve[l] = T.toc();

    std::cout << "  Schur:  N_int=" << N_int[l] << "  N_bdr=" << N_bdr[l]
              << "  fact(A_II)=" << std::fixed << std::setprecision(3)
              << t_factor[l] << "s  solve=" << t_solve[l] << "s\n";

  }

  // -----------------------------------------------------------------
  // Evaluation
  // -----------------------------------------------------------------
  Vector u_exact(NEVAL);
  for (int i = 0; i < NEVAL; ++i)
    u_exact[i] =
        std::sin(FMCA_PI * P_eval(0, i)) * std::sin(FMCA_PI * P_eval(1, i));

  std::cout << "\n"
            << std::left << std::setw(7) << "Level" << std::setw(10) << "N"
            << std::setw(10) << "Comp(s)" << std::setw(10) << "Fact(s)"
            << std::setw(10) << "Solve(s)" << std::setw(8) << "anz"
            << std::setw(10) << "Eval(s)" << "L2 error\n"
            << std::string(75, '-') << "\n";

  Vector acc = Vector::Zero(NEVAL);
  for (int l = 0; l < L; ++l) {
    Vector a_int_nat = hst_int[l]->toNaturalOrder(
        hst_int[l]->inverseSampletTransform(alpha[l].head(N_int[l])));
    Vector a_bdr_nat = hst_bdr[l]->toNaturalOrder(
        hst_bdr[l]->inverseSampletTransform(alpha[l].tail(N_bdr[l])));
    Vector alpha_nat(N[l]);
    alpha_nat << a_int_nat, a_bdr_nat;

    CovarianceKernel kernel_l(ktype, sigma[l]);
    T.tic();
    MultipoleFunctionEvaluator mfe;
    mfe.init(kernel_l, P[l], P_eval);
    acc += mfe.evaluate(P[l], P_eval, alpha_nat);
    Scalar t_eval = T.toc();

    Scalar l2_err = (acc - u_exact).norm() / u_exact.norm();

    std::cout << std::left << std::setw(7) << l + 1 << std::setw(10) << N[l]
              << std::setw(10) << std::fixed << std::setprecision(3)
              << t_compress[l] << std::setw(10) << std::fixed
              << std::setprecision(3) << t_factor[l] << std::setw(10)
              << std::fixed << std::setprecision(3) << t_solve[l]
              << std::setw(8) << anz_diag[l] << std::setw(10) << std::fixed
              << std::setprecision(3) << t_eval << std::scientific
              << std::setprecision(3) << l2_err << "\n";
  }

  return 0;
}
