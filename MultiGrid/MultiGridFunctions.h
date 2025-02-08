#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
// ##############################
#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Plotter.h"
#include "../FMCA/src/util/Plotter3D.h"
#include "../FMCA/src/util/Tictoc.h"

// using Interpolator = FMCA::TotalDegreeInterpolator;
// using SampletInterpolator = FMCA::MonomialInterpolator;
// using Moments = FMCA::NystromMoments<Interpolator>;
// using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
// using MatrixEvaluatorKernel =
//     FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
// using usMatrixEvaluatorKernel =
//     FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
// using EigenCholesky =
//     Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
//                           Eigen::MetisOrdering<int>>;

using namespace FMCA;

Eigen::SparseMatrix<Scalar> UnsymmetricCompressor(
    const NystromMoments<TotalDegreeInterpolator>& mom_rows,
    const NystromSampletMoments<MonomialInterpolator>& samp_mom_rows,
    const H2SampletTree<ClusterTree>& hst_rows,
    const CovarianceKernel& function, const Scalar& eta,
    const Scalar& threshold_kernel, const Scalar& threshold_aPost,
    const Scalar& mpole_deg,
    const Scalar& dtilde, const Matrix& P_rows, const Matrix& P_cols) {
  ///////////////////////////////// Samplet quantities cols
  const NystromMoments<TotalDegreeInterpolator> mom_cols(P_cols, mpole_deg);
  const NystromSampletMoments<MonomialInterpolator> samp_mom_cols(P_cols,
                                                                  dtilde - 1);
  const H2SampletTree<ClusterTree> hst_cols(mom_cols, samp_mom_cols, 0, P_cols);

  //////////////////////////////// Compression
  int n_pts_rows = P_rows.cols();
  int n_pts_cols = P_cols.cols();

  const unsymmetricNystromEvaluator<NystromMoments<TotalDegreeInterpolator>,
                                    FMCA::CovarianceKernel>
      mat_eval(mom_rows, mom_cols, function);
  FMCA::Tictoc T;
  T.tic();
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree<ClusterTree>>
      K_comp;
  if (P_cols.cols() < 100) {
    K_comp.init(hst_rows, hst_cols, eta, 0);
  } else {
    K_comp.init(hst_rows, hst_cols, eta, threshold_kernel);
  }
  T.toc("planner:                     ");

  T.tic();
  K_comp.compress(mat_eval);
  T.toc("compressor:                  ");
 
  T.tic();
  const auto& triplets_aPriori = K_comp.triplets();
  FMCA::Scalar trips_time_aPriori = T.toc();

  T.tic();
  const auto& triplets_aPost = K_comp.aposteriori_triplets(threshold_aPost);
  FMCA::Scalar trips_time_aPost = T.toc();

  std::cout << "" << std::endl;
  std::cout << "anz (a-priori):               "
            << std::round(triplets_aPriori.size() / FMCA::Scalar(n_pts_rows))
            << std::endl;
  std::cout << "anz (a-posteriori):           "
            << std::round(triplets_aPost.size() / FMCA::Scalar(n_pts_rows))
            << std::endl;
  std::cout << "" << std::endl;
  std::cout << "time (a-priori):              " << trips_time_aPriori << "sec." << std::endl;
  std::cout << "time (a-posteriori):          " << trips_time_aPost << "sec." << std::endl;
  std::cout << "" << std::endl;

  //////////////////////////////// Compression error
  Vector x(n_pts_cols), y1(n_pts_rows), y2(n_pts_rows);
  Scalar err = 0;
  Scalar nrm = 0;

  for (auto i = 0; i < 100; ++i) {
    Index index = rand() % n_pts_cols;
    x.setZero();
    x(index) = 1;

    Vector col = function.eval(P_rows, P_cols.col(hst_cols.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst_rows.indices(),
                                             hst_rows.block_size()));
    x = hst_cols.sampletTransform(x);
    y2.setZero();

    for (const auto& triplet : triplets_aPost) {
      y2(triplet.row()) += triplet.value() * x(triplet.col());
    }

    y2 = hst_rows.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }

  std::cout << "Compression error unsymmetric      " << sqrt(err / nrm)
            << std::endl;

  //////////////////////////////// Create and return the sparse matrix
  Eigen::SparseMatrix<Scalar> Kcomp_Sparse(n_pts_rows, n_pts_cols);
  Kcomp_Sparse.setFromTriplets(triplets_aPost.begin(), triplets_aPost.end());
  Kcomp_Sparse.makeCompressed();

  return Kcomp_Sparse;
}

Eigen::SparseMatrix<Scalar> SymmetricCompressor(
    const NystromMoments<TotalDegreeInterpolator>& mom,
    const NystromSampletMoments<MonomialInterpolator>& samp_mom,
    const H2SampletTree<ClusterTree>& hst, const CovarianceKernel& function, 
    const Scalar& eta,
    const Scalar& threshold_kernel, const Scalar& threshold_aPost,
    const Scalar& mpole_deg,
    const Scalar& dtilde, const Matrix& P) {
  //////////////////////////////// Compression
  int n_pts = P.cols();

  const NystromEvaluator<NystromMoments<TotalDegreeInterpolator>,
                         CovarianceKernel>
      mat_eval(mom, function);

  FMCA::Tictoc T;
  T.tic();
  internal::SampletMatrixCompressor<H2SampletTree<ClusterTree>> K_comp;
  K_comp.init(hst, eta, threshold_kernel);
  T.toc("planner:                     ");

  T.tic();
  K_comp.compress(mat_eval);
  T.toc("compressor:                  ");

  T.tic();
  const auto& triplets_aPriori = K_comp.triplets();
  FMCA::Scalar trips_time_aPriori = T.toc();

  T.tic();
  const auto& triplets_aPost = K_comp.aposteriori_triplets(threshold_aPost);
  FMCA::Scalar trips_time_aPost = T.toc();

  std::cout << "" << std::endl;
  std::cout << "anz (a-priori):               "
            << std::round(triplets_aPriori.size() / FMCA::Scalar(n_pts))
            << std::endl;
  std::cout << "anz (a-posteriori):           "
            << std::round(triplets_aPost.size() / FMCA::Scalar(n_pts))
            << std::endl;
  std::cout << "" << std::endl;
  std::cout << "time (a-priori):              " << trips_time_aPriori << "sec." << std::endl;
  std::cout << "time (a-posteriori):          " << trips_time_aPost << "sec." << std::endl;
  std::cout << "" << std::endl;

  //////////////////////////////// Compression error
  Vector x(n_pts), y1(n_pts), y2(n_pts);
  Scalar err = 0;
  Scalar nrm = 0;
  for (auto i = 0; i < 100; ++i) {
    Index index = rand() % n_pts;
    x.setZero();
    x(index) = 1;
    Vector col = function.eval(P.leftCols(n_pts),
                               P.leftCols(n_pts).col(hst.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst.indices(), hst.block_size()));
    x = hst.sampletTransform(x);
    y2.setZero();
    for (const auto& i : triplets_aPost) {
      y2(i.row()) += i.value() * x(i.col());
      if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
    }
    y2 = hst.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  std::cout << "Compression error symmetric        " << sqrt(err / nrm)
            << std::endl;
  int anz = triplets_aPost.size() / n_pts;
  std::cout << "Anz:                               " << anz << std::endl;

  //////////////////////////////// Create and return the sparse matrix
  Eigen::SparseMatrix<Scalar> Kcomp_Sparse(n_pts, n_pts);
  Kcomp_Sparse.setFromTriplets(triplets_aPost.begin(), triplets_aPost.end());
  Kcomp_Sparse.makeCompressed();

  return Kcomp_Sparse;
}

Vector Evaluate(
    const NystromMoments<TotalDegreeInterpolator>& mom_eval,
    const NystromSampletMoments<MonomialInterpolator>& samp_mom_eval,
    const H2SampletTree<ClusterTree>& hst_eval, const std::string& kernel_type,
    const std::vector<Matrix>& P_Matrices, const Matrix& Peval,
    const std::vector<Vector>& ALPHA, const Vector& fill_distances,
    const Scalar& max_level, const Scalar& nu, const Scalar& eta,
    const Scalar& threshold_kernel, const Scalar& threshold_aPost,
    const Scalar& mpole_deg,
    const Scalar& dtilde, const Vector& exact_sol,
    const H2SampletTree<ClusterTree>& hst, const std::string& base_filename) {
  Vector solution = Eigen::VectorXd::Zero(Peval.cols());
  for (Index i = 0; i < max_level; ++i) {
    std::cout << "-------- Evaluation Level " << i + 1 << " --------"
              << std::endl;
    Scalar sigma = nu * fill_distances[i];
    const CovarianceKernel function(kernel_type, sigma);
    Eigen::SparseMatrix<double> K_eval = UnsymmetricCompressor(
        mom_eval, samp_mom_eval, hst_eval, function, eta, threshold_kernel,
        threshold_aPost,
        mpole_deg, dtilde, Peval, P_Matrices[i]);
    solution += K_eval * ALPHA[i];
    Vector solution_natural_basis = hst.inverseSampletTransform(solution);
    solution_natural_basis = hst.toNaturalOrder(solution_natural_basis);

    if (base_filename != "" and Peval.rows() == 2) {
      // Create the filename for the current level
      std::ostringstream oss;
      oss << base_filename << "_level_" << i << ".vtk";
      std::string filename_solution = oss.str();
      Plotter2D plotter;
      plotter.plotFunction2D(filename_solution, Peval, solution_natural_basis);
    }

    if (base_filename != "" and Peval.rows() == 3) {
      // Create the filename for the current level
      std::ostringstream oss;
      oss << base_filename << "_level_" << i << ".vtk";
      std::string filename_solution = oss.str();
      Plotter3D plotter;
      plotter.plotFunction(filename_solution, Peval, solution_natural_basis);
    }

    Vector diff_abs(solution_natural_basis.rows());
    for (Index i = 0; i < solution_natural_basis.rows(); ++i) {
      diff_abs[i] = abs(solution_natural_basis[i] - exact_sol[i]);
    }

    std::cout << "Error l2                           "
              << (solution_natural_basis - exact_sol).norm() / exact_sol.norm()
              << std::endl;
    std::cout << "Error l_inf                        " << diff_abs.maxCoeff()
              << std::endl;

    K_eval.resize(0, 0);
  }
  return solution;
}

Vector solveSystem(const Eigen::SparseMatrix<double>& A_comp, const Vector& rhs,
                   const std::string& solverName, Scalar threshold_CG = 1e-6) {
  Eigen::VectorXd alpha;
  Scalar solver_time;
  Tictoc T;
  Eigen::SparseMatrix<double> A_comp_Symmetric =
      A_comp.selfadjointView<Eigen::Upper>();
  T.tic();
  if (solverName == "SimplicialLLT") {
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                         Eigen::MetisOrdering<int>>
        solver;
    solver.compute(A_comp);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomposition failed");
    }
    alpha = solver.solve(rhs);
  } else if (solverName == "SimplicialLDLT") {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>
        solver;
    solver.compute(A_comp);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomposition failed");
    }
    alpha = solver.solve(rhs);
  } else if (solverName == "ConjugateGradient") {
    T.tic();
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                             Eigen::Lower | Eigen::Upper,
                             Eigen::IdentityPreconditioner>
        solver;
    solver.setTolerance(threshold_CG);
    solver.compute(A_comp_Symmetric);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomposition failed");
    }
    alpha = solver.solve(rhs);
    T.toc("solver:                  ");
    std::cout << "#iterations:     " << solver.iterations() << std::endl;
  } else if (solverName == "ConjugateGradientwithPreconditioner") {
    T.tic();
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                             Eigen::Lower | Eigen::Upper>
        solver;
    solver.setTolerance(threshold_CG);
    solver.compute(A_comp_Symmetric);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomposition failed");
    }
    alpha = solver.solve(rhs);
    T.toc("solver:                  ");
    std::cout << "#iterations:     " << solver.iterations() << std::endl;
  } else {
    throw std::invalid_argument("Unknown solver name");
  }
  solver_time = T.toc();

  std::cout << "Solver: " << solverName << std::endl;
  std::cout << "Solver time: " << solver_time << std::endl;
  std::cout << "Residual error: " << (A_comp_Symmetric * alpha - rhs).norm()
            << std::endl;

  return alpha;
}

