#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cmath>
#include <functional>
#include <cassert>

#include <Eigen/Sparse>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include "../FMCA/CovarianceKernel"
#include "../FMCA/KernelInterpolation"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2

using namespace FMCA;

using Interpolator       = TotalDegreeInterpolator;
using SampletInterpolator= MonomialInterpolator;
using Moments            = NystromMoments<Interpolator>;
using SampletMoments     = NystromSampletMoments<SampletInterpolator>;
using usEval             = unsymmetricNystromEvaluator<Moments, CovarianceKernel>;
using usEvalDeriv        = unsymmetricNystromEvaluator<Moments, GradKernel>;
using H2ST               = H2SampletTree<ClusterTree>;
using H2CT               = H2ClusterTree<ClusterTree>;


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
    for (int i = 1; i <= s; ++i) { P(0,col)=i*h; P(1,col)=0.0; ++col; }
    for (int i = 1; i <= s; ++i) { P(0,col)=i*h; P(1,col)=1.0; ++col; }
    for (int i = 1; i <= s; ++i) { P(0,col)=0.0; P(1,col)=i*h; ++col; }
    for (int i = 1; i <= s; ++i) { P(0,col)=1.0; P(1,col)=i*h; ++col; }
    return P;
}

////////////////////////////////////////////////////////////////////////////////
// Build the collocation block A(row_level, col_level):
////////////////////////////////////////////////////////////////////////////////
SparseMatrix buildBlock(
    // row-level objects
    H2ST& hst_int_row,  H2ST& hst_bdr_row,
    Moments& mom_int_row, Moments& mom_bdr_row,
    int N_int_row, int N_bdr_row,
    Scalar thr_lap_row,
    // col-level objects
    H2ST& hst_col, Moments& mom_col,
    int N_col,
    // kernel
    const CovarianceKernel& kernel_col,
    Scalar sigma_col,
    // shared params
    Scalar eta, Scalar thr_kernel,
    const std::string& kernel_type_lap)
{
    const int N_row = N_int_row + N_bdr_row;

    // --- Border rows: K(P_bdr_row, P_col) ---
    SparseMatrix K_bdr(N_bdr_row, N_col);
    {
        usEval ev(mom_bdr_row, mom_col, kernel_col);
        internal::SampletMatrixCompressorUnsymmetric<H2ST> sc;
        sc.init(hst_bdr_row, hst_col, eta, thr_kernel);
        sc.compress(ev);
        K_bdr.setFromTriplets(sc.triplets().begin(), sc.triplets().end());
        K_bdr.makeCompressed();
    }

    // --- Interior rows: sum_d d^2/dx_d^2 K(P_int_row, P_col) ---
    SparseMatrix L_int(N_int_row, N_col);
    L_int.setZero();
    for (int d = 0; d < DIM; ++d) {
        GradKernel gk(kernel_type_lap, sigma_col, 1, d);
        usEvalDeriv ev(mom_int_row, mom_col, gk);
        internal::SampletMatrixCompressorUnsymmetric<H2ST> sc;
        sc.init(hst_int_row, hst_col, eta, thr_lap_row);
        sc.compress(ev);
        SparseMatrix Ld(N_int_row, N_col);
        Ld.setFromTriplets(sc.triplets().begin(), sc.triplets().end());
        Ld.makeCompressed();
        L_int += Ld;
    }

    // --- Stack: [L_int ; K_bdr] ---
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(L_int.nonZeros() + K_bdr.nonZeros());
    for (int k = 0; k < L_int.outerSize(); ++k)
        for (SparseMatrix::InnerIterator it(L_int, k); it; ++it)
            trips.emplace_back(it.row(), it.col(), it.value());
    for (int k = 0; k < K_bdr.outerSize(); ++k)
        for (SparseMatrix::InnerIterator it(K_bdr, k); it; ++it)
            trips.emplace_back(it.row() + N_int_row, it.col(), it.value());

    SparseMatrix A(N_row, N_col);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();
    return A;
}

////////////////////////////////////////////////////////////////////////////////
int main() {
    // Point sets per level (coarse to fine)
    const std::vector<Matrix> P_int_levels = {
        makeInteriorGrid(256),
        makeInteriorGrid(1024),
        makeInteriorGrid(4096),
        makeInteriorGrid(16384),
    };
    const std::vector<Matrix> P_bdr_levels = {
        makeBorderGrid(16),
        makeBorderGrid(32),
        makeBorderGrid(64),
        makeBorderGrid(128),
    };

    const Scalar nu = 10.0;

    // PDE: -Delta u = f on interior,  u = g on boundary
    // Exact solution: sin(pi x) sin(pi y)
    auto f_rhs = [](Scalar x, Scalar y){ return -2.*FMCA_PI*FMCA_PI * std::sin(FMCA_PI*x) * std::sin(FMCA_PI*y); };
    auto u_bc  = [](Scalar /*x*/, Scalar /*y*/){ return 0.0; };

    // Evaluation points
    const int NEVAL = 100000;
    Matrix P_eval(DIM, NEVAL);
    P_eval = (Matrix::Random(DIM, NEVAL).array() + 1.0) / 2.0;

    // FMCA parameters
    const Scalar      eta          = 1.0 / DIM;
    const Index       dtilde       = 4;
    const Scalar      thr_kernel   = 1e-8;
    const Index       mpole_deg    = 2 * (dtilde - 1);
    const std::string ktype        = "MATERN52";
    const std::string ktype_lap    = "MATERN52_SECOND_DERIVATIVE";

    // =========================================================================
    // build trees
    // =========================================================================
    const int L = (int)P_int_levels.size();

    // Per-level combined point sets and geometry
    std::vector<Matrix> P(L);
    std::vector<int>    N_int(L), N_bdr(L), N(L);
    std::vector<Scalar> fill_dist(L), sigma(L), thr_lap(L);

    // Trees stored in parallel unique_ptr vectors (stable heap addresses)
    std::vector<std::unique_ptr<Moments>>       mom_int(L), mom_bdr(L), mom(L);
    std::vector<std::unique_ptr<SampletMoments>>smom_int(L), smom_bdr(L), smom(L);
    std::vector<std::unique_ptr<H2ST>>          hst_int(L), hst_bdr(L), hst(L);

    for (int l = 0; l < L; ++l) {
        N_int[l] = (int)P_int_levels[l].cols();
        N_bdr[l] = (int)P_bdr_levels[l].cols();
        N[l]     = N_int[l] + N_bdr[l];

        P[l].resize(DIM, N[l]);
        P[l].leftCols(N_int[l])  = P_int_levels[l];
        P[l].rightCols(N_bdr[l]) = P_bdr_levels[l];

        mom_int[l]  = std::make_unique<Moments>(P_int_levels[l], mpole_deg);
        smom_int[l] = std::make_unique<SampletMoments>(P_int_levels[l], dtilde-1);
        hst_int[l]  = std::make_unique<H2ST>(*mom_int[l], *smom_int[l], 0, P_int_levels[l]);

        mom_bdr[l]  = std::make_unique<Moments>(P_bdr_levels[l], mpole_deg);
        smom_bdr[l] = std::make_unique<SampletMoments>(P_bdr_levels[l], dtilde-1);
        hst_bdr[l]  = std::make_unique<H2ST>(*mom_bdr[l], *smom_bdr[l], 0, P_bdr_levels[l]);

        mom[l]  = std::make_unique<Moments>(P[l], mpole_deg);
        smom[l] = std::make_unique<SampletMoments>(P[l], dtilde-1);
        hst[l]  = std::make_unique<H2ST>(*mom[l], *smom[l], 0, P[l]);

        Vector minDist = minDistanceVector(*hst[l], P[l]);
        fill_dist[l]   = minDist.maxCoeff();
        sigma[l]       = nu * fill_dist[l];
        thr_lap[l]     = thr_kernel * minDist.mean();
    }

    // =========================================================================
    // Initial residual b_l = [f(P_int_l); u_bc(P_bdr_l)] in samplet order
    // =========================================================================
    std::vector<Vector> residuals(L);
    for (int l = 0; l < L; ++l) {
        Vector f_l(N_int[l]);
        for (int i = 0; i < N_int[l]; ++i)
            f_l[i] = f_rhs(P_int_levels[l](0,i), P_int_levels[l](1,i));

        Vector g_l(N_bdr[l]);
        for (int i = 0; i < N_bdr[l]; ++i)
            g_l[i] = u_bc(P_bdr_levels[l](0,i), P_bdr_levels[l](1,i));

        residuals[l].resize(N[l]);
        residuals[l] << hst_int[l]->sampletTransform(hst_int[l]->toClusterOrder(f_l)),
                        hst_bdr[l]->sampletTransform(hst_bdr[l]->toClusterOrder(g_l));
    }

    // =========================================================================
    // Multiscale forward substitution
    // =========================================================================
    Tictoc T;
    std::vector<Vector> alpha(L);

    std::vector<Scalar> t_compress(L), t_solve(L);
    std::vector<int>    anz_diag(L);

    for (int l = 0; l < L; ++l) {
        std::cout << std::string(60, '-') << "\nLevel " << l + 1 << "\n";

        // --- Cross blocks: subtract A_lj * alpha[j] from residuals[l] ---
        for (int j = 0; j < l; ++j) {
            CovarianceKernel kernel_j(ktype, sigma[j]);
            SparseMatrix A_lj = buildBlock(
                *hst_int[l], *hst_bdr[l], *mom_int[l], *mom_bdr[l],
                N_int[l], N_bdr[l], thr_lap[l],
                *hst[j], *mom[j], N[j],
                kernel_j, sigma[j],
                eta, thr_kernel, ktype_lap);
            residuals[l] -= A_lj * alpha[j];
        }  // A_lj, kernel_j destroyed here

        // --- Diagonal block and solve ---
        {
            CovarianceKernel kernel_l(ktype, sigma[l]);

            T.tic();
            SparseMatrix A_ll = buildBlock(
                *hst_int[l], *hst_bdr[l], *mom_int[l], *mom_bdr[l],
                N_int[l], N_bdr[l], thr_lap[l],
                *hst[l], *mom[l], N[l],
                kernel_l, sigma[l],
                eta, thr_kernel, ktype_lap);
            t_compress[l] = T.toc("  compress");
            anz_diag[l] = (int)(A_ll.nonZeros() / N[l]);

            T.tic();
            Eigen::SparseLU<SparseMatrix, Eigen::MetisOrdering<int>> solver;
            solver.analyzePattern(A_ll);
            solver.factorize(A_ll);
            if (solver.info() != Eigen::Success)
                throw std::runtime_error("LU failed at level " + std::to_string(l+1));
            alpha[l] = solver.solve(residuals[l]);
            t_solve[l] = T.toc("  solve");

            std::cout << "  residual: " << (A_ll * alpha[l] - residuals[l]).norm() << "\n";
        }  // out of scope
    }

    // =========================================================================
    // Evaluation
    // =========================================================================
    const Moments  rmom(P_eval, mpole_deg);
    const H2CT     hct_eval(rmom, 0, P_eval);
    Vector rec = Vector::Zero(NEVAL);

    for (int l = 0; l < L; ++l) {
        Vector alpha_nat = hst[l]->inverseSampletTransform(alpha[l]);
        alpha_nat        = hst[l]->toNaturalOrder(alpha_nat);

        Moments cmom_l(P[l], mpole_deg);
        H2CT    hct_l(cmom_l, 0, P[l]);
        CovarianceKernel kernel_l(ktype, sigma[l]);

        usEval mat_eval(rmom, cmom_l, kernel_l);
        H2Matrix<H2CT, CompareCluster> hmat;
        hmat.computePattern(hct_eval, hct_l, eta);
        rec += hct_eval.toNaturalOrder(hmat.action(mat_eval, hct_l.toClusterOrder(alpha_nat)));
    }

    Vector u_exact(NEVAL);
    for (int i = 0; i < NEVAL; ++i)
        u_exact[i] = std::sin(FMCA_PI * P_eval(0,i)) * std::sin(FMCA_PI * P_eval(1,i));

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << std::left
              << std::setw(8)  << "Level"
              << std::setw(8)  << "N"
              << std::setw(14) << "CompressTime"
              << std::setw(12) << "SolveTime"
              << std::setw(8)  << "anz"
              << std::setw(14) << "L2 error"
              << "\n" << std::string(60, '-') << "\n";

    Vector acc = Vector::Zero(NEVAL);
    for (int l = 0; l < L; ++l) {
        Vector alpha_nat = hst[l]->inverseSampletTransform(alpha[l]);
        alpha_nat        = hst[l]->toNaturalOrder(alpha_nat);
        Moments cmom_l(P[l], mpole_deg);
        H2CT hct_l(cmom_l, 0, P[l]);
        CovarianceKernel kernel_l(ktype, sigma[l]);
        usEval mat_eval(rmom, cmom_l, kernel_l);
        H2Matrix<H2CT, CompareCluster> hmat;
        hmat.computePattern(hct_eval, hct_l, eta);
        acc += hct_eval.toNaturalOrder(hmat.action(mat_eval, hct_l.toClusterOrder(alpha_nat)));

        Scalar l2_err = (acc - u_exact).norm() / u_exact.norm();
        std::cout << std::left
                  << std::setw(8)  << l+1
                  << std::setw(8)  << N[l]
                  << std::setw(14) << std::fixed << std::setprecision(4) << t_compress[l]
                  << std::setw(12) << std::fixed << std::setprecision(4) << t_solve[l]
                  << std::setw(8)  << anz_diag[l]
                  << std::setw(14) << std::scientific << std::setprecision(4) << l2_err
                  << "\n";
    }

    return 0;
}