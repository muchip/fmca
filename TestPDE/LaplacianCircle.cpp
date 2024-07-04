/* We solve here the Poisson problem on the unit circle
- laplacian(u) = 4
u = 0 on \partial u
Using Matern32 kernel and penalty method to impose the boundary conditions.
We rely on the FMCA library by M.Multerer.
 */

#include "SolvePoisson.h"
///////////////////////////////////
#include "../FMCA/src/util/Plotter.h"
#include "read_files_txt.h"

#define DIM 2

FMCA::Vector RandomPointInterior() {
  FMCA::Vector point_interior;
  do {
    point_interior = Eigen::Vector2d::Random();
  } while (point_interior.squaredNorm() >= 1.0);
  return point_interior;
}

FMCA::Vector RandomPointBoundary() {
  FMCA::Vector point_bnd = RandomPointInterior();
  return point_bnd / point_bnd.norm();
}

FMCA::Matrix MonteCarloPointsInterior(int N) {
  FMCA::Matrix P_interior(DIM, N);
  for (int i = 0; i < N; ++i) {
    FMCA::Vector x = RandomPointInterior();
    P_interior.col(i) = x;
  }
  return P_interior;
}

FMCA::Matrix MonteCarloPointsBoundary(int N) {
  FMCA::Matrix P_bnd(DIM, N);
  for (int i = 0; i < N; ++i) {
    FMCA::Vector x = RandomPointBoundary();
    P_bnd.col(i) = x;
  }
  return P_bnd;
}

FMCA::Scalar WeightInterior(int N, FMCA::Scalar r) {
  return FMCA_PI * r * r *
         (1. / N);  // Monte Carlo integration weights: Volume*1/N
}

FMCA::Scalar WeightBnd(int N, FMCA::Scalar r) {
  return 2. * FMCA_PI * r *
         (1. / N);  // Monte Carlo integration weights: Surface*1/N
}

// Function to save matrix to a text file
void saveMatrixToFile(const FMCA::Matrix& matrix, const std::string& filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    for (int i = 0; i < matrix.rows(); ++i) {
      for (int j = 0; j < matrix.cols(); ++j) {
        file << matrix(i, j) << " ";
      }
      file << "\n";
    }

    std::cout << "Matrix saved to " << filename << std::endl;
    file.close();  // Close the file
  } else {
    std::cerr << "Unable to open file " << filename << std::endl;
  }
}

int main() {
  // DATA
  FMCA::Tictoc T;
  FMCA::Matrix P;
  FMCA::Matrix P_interior_full;;
  FMCA::Matrix P_bnd_full;
  /*
    int N_interior = 500000;
    FMCA::Matrix P_interior = MonteCarloPointsInterior(N_interior);
    saveMatrixToFile(P_interior.transpose(), "MC_Interior_Circle.txt");
    int N_bnd = 200000;
    FMCA::Matrix P_bnd = MonteCarloPointsBoundary(N_bnd);
    saveMatrixToFile(P_bnd.transpose(), "MC_Bnd_Circle.txt");
  */

  readTXT("data/MC_Interior_Circle.txt", P_interior_full, DIM);
  readTXT("data/MC_Bnd_Circle.txt", P_bnd_full, DIM);

  int N_interior = 100000;
  int N_bnd = 50000;
  FMCA::Matrix P_interior = P_interior_full.leftCols(N_interior);
  FMCA::Matrix P_bnd = P_bnd_full.leftCols(N_bnd);

  FMCA::Matrix Normals = P_bnd;

  readTXT("data/Points_Circle_Uniform.txt", P, DIM);
  //   FMCA::IO::plotPoints("P_circle.vtk", P);
  std::cout << "P done" << std::endl;
  //   FMCA::Scalar factor = 0.2;
  //   int N_interior_sources = factor * N_interior;
  //   int N_bnd_sources = factor * N_bnd;
  //   FMCA::Matrix P_interior_sources =
  //       MonteCarloPointsInterior(N_interior_sources);
  //   FMCA::Matrix P_bnd_sources = MonteCarloPointsBoundary(N_bnd_sources);

  //   FMCA::Matrix P(DIM, N_interior_sources + N_bnd_sources);
  //   P << P_interior_sources, P_bnd_sources;

  //   FMCA::Matrix Normals;
  FMCA::Scalar w = WeightInterior(N_interior, 1.0);
  std::cout << "w_vec:                      " << w << std::endl;
  FMCA::Vector w_vec = Eigen::VectorXd::Ones(N_interior);
  w_vec *= w;

  FMCA::Scalar w_border = WeightBnd(N_bnd, 1.0);
  std::cout << "w_vec_border:               " << w_border << std::endl;
  FMCA::Vector w_vec_border = Eigen::VectorXd::Ones(N_bnd);
  w_vec_border *= w_border;

  // U_BC vector
  FMCA::Vector u_bc(P_bnd.cols());
  for (FMCA::Index i = 0; i < P_bnd.cols(); ++i) {
    u_bc[i] = 0;
  }
  // Right hand side
  FMCA::Vector f(P_interior.cols());
  for (FMCA::Index i = 0; i < P_interior.cols(); ++i) {
    f[i] = 4;
  }
  // Analytical sol of the problem
  FMCA::Vector analytical_sol(P.cols());
  for (int i = 0; i < P.cols(); ++i) {
    FMCA::Scalar x = P(0, i);
    FMCA::Scalar y = P(1, i);
    analytical_sol[i] = 1 - x * x - y * y;
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar threshold_kernel = 1e0 / N_interior;
  const FMCA::Scalar threshold_gradKernel = 1e2 / N_interior;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 4;
  const FMCA::Scalar beta = 10000;
  const std::string kernel_type = "MATERN32";

  const Moments mom_sources(P, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P);
  FMCA::Vector minDistance = minDistanceVector(hst_sources, P);

  // fill distance
  auto maxElementIterator =
      std::max_element(minDistance.begin(), minDistance.end());
  FMCA::Scalar sigma_h = *maxElementIterator;
  std::cout << "fill distance:                      " << sigma_h << std::endl;

  std::vector<double> sigmas = {0.8, 1, 1.5, 1.8, 2};

  for (FMCA::Scalar sigma_factor : sigmas) {
    std::cout << "sigma factor =                         " << sigma_factor
              << std::endl;
    FMCA::Scalar sigma = sigma_factor * sigma_h;

    // FMCA::Vector u = SolvePoisson_MonteCarlo(
    //     DIM, P, P_interior, w, P_bnd, w_border, Normals, u_bc, f, sigma,
    //     eta, dtilde, threshold_kernel, threshold_gradKernel,
    //     threshold_weights, MPOLE_DEG, beta, kernel_type);
    FMCA::Vector u = SolvePoisson_constantWeights(
        DIM, P, P_interior, w_vec, P_bnd, w_vec_border, Normals, u_bc, f, sigma,
        eta, dtilde, threshold_kernel, threshold_gradKernel, threshold_weights,
        MPOLE_DEG, beta, kernel_type);

    FMCA::Vector u_grid = hst_sources.inverseSampletTransform(u);
    u_grid = hst_sources.toNaturalOrder(u_grid);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Cumpute the solution K*u with the compression of the kernel matrix K
    const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
    const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources,
                                                   kernel_funtion_ss);
    std::cout << "Kernel Source-Source" << std::endl;
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
    Kcomp_ss.init(hst_sources, eta, threshold_kernel);
    std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets =
        Kcomp_ss.a_priori_pattern_triplets();
    Kcomp_ss.compress(mat_eval_kernel_ss);
    const auto& triplets = Kcomp_ss.triplets();
    int anz = triplets.size() / P.cols();
    std::cout << "Anz:                         " << anz << std::endl;

    Eigen::SparseMatrix<double> Kcomp_ss_Sparse(P.cols(), P.cols());
    Kcomp_ss_Sparse.setFromTriplets(triplets.begin(), triplets.end());
    Kcomp_ss_Sparse.makeCompressed();

    Eigen::SparseMatrix<double> Kcomp_ss_Sparse_symm =
        Kcomp_ss_Sparse.selfadjointView<Eigen::Upper>();
    FMCA::Vector KU = Kcomp_ss_Sparse_symm * u;
    KU = hst_sources.inverseSampletTransform(KU);
    // Numerical solution
    KU = hst_sources.toNaturalOrder(KU);

    // Error
    FMCA::Scalar error = (KU - analytical_sol).norm() / analytical_sol.norm();
    std::cout << "KU 1:                                          " << KU(0)
              << std::endl;
    std::cout << "analytical_sol 1:                                "
              << analytical_sol(0) << std::endl;
    std::cout << "Error:                                            " << error
              << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // plot solution
    // {
    //   const Moments cmom(P_sources, MPOLE_DEG);
    //   const Moments rmom(Peval, MPOLE_DEG);
    //   const H2ClusterTree hct(cmom, 0, P_sources);
    //   const H2ClusterTree hct_eval(rmom, 0, Peval);
    //   const FMCA::Vector sx = hct.toClusterOrder(u_grid);
    //   const usMatrixEvaluatorKernel mat_eval(rmom, cmom, kernel_funtion_ss);
    //   FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster> hmat;
    //   hmat.computePattern(hct_eval, hct, eta);
    //   hmat.statistics();
    //   FMCA::Vector srec = hmat.action(mat_eval, sx);
    //   FMCA::Vector rec = hct_eval.toNaturalOrder(srec);
    //   FMCA::Plotter2D plotter;
    //   plotter.plotFunction(outputNamePlot, Peval, rec);

    //   // Analytical sol of the problem
    //   FMCA::Vector analytical_sol(Peval.cols());
    //   for (int i = 0; i < Peval.cols(); ++i) {
    //     double x = Peval(0, i);
    //     double y = Peval(1, i);
    //     analytical_sol[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
    //   }

    //   FMCA::Scalar error =
    //       (rec - analytical_sol).norm() / analytical_sol.norm();
    //   std::cout << "Error:                                            " <<
    //   error
    //             << std::endl;

    //   FMCA::Vector absolute_error(Peval.cols());
    //   for (int i = 0; i < Peval.cols(); ++i) {
    //     absolute_error[i] = abs(rec[i] - analytical_sol[i]);
    //   }

    //   // plot error
    //   {
    //     FMCA::Plotter2D plotter_error;
    //     plotter_error.plotFunction(outputNameErrorPlot, Peval,
    //     absolute_error);
    //   }
    // }
  }

  return 0;
}
