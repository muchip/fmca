/* We solve here the Poisson problem on the square [-1,1]x[-1,1]
laplacian(u) = 2*pi^2*sin(pi*x)*sin(pi*y)
u = 0 on \partial u
Using Matern32 kernel and penalty method to impose the boundary conditions.
We rely on the FMCA library by M.Multerer.
 */


#include "SolvePoisson.h"
///////////////////////////////////
#include "../FMCA/src/util/Plotter.h"
#include "read_files_txt.h"

#define DIM 2

int main() {
  // DATA
  FMCA::Tictoc T;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_quad_border;
  FMCA::Matrix Peval;
  FMCA::Matrix Normals;
  FMCA::Vector w_vec;
  FMCA::Vector w_vec_border;

  std::string outputNamePlot = "weakHalton_400.vtk";
  std::string outputNameErrorPlot = "error_weakHalton_400.vtk";

    readTXT("data/InteriorAndBnd_square409600.txt", P_sources, 2);
    // readTXT("data/quadrature3_points_square90000.txt", P_quad, 2);
    // readTXT("data/quadrature3_weights_square90000.txt", w_vec);
    // readTXT("data/quadrature_border90000.txt", P_quad_border, 2);
    // readTXT("data/weights_border90000.txt", w_vec_border);
    // readTXT("data/normals90000.txt", Normals, 2);
  readTXT("data/uniform_quadratures250k.txt", P_quad, 2);
  readTXT("data/uniform_weights250k.txt", w_vec);
  readTXT("data/quadrature5_border90k.txt", P_quad_border, 2);
  readTXT("data/weights5_border90k.txt", w_vec_border);
  readTXT("data/5normals90k.txt", Normals, 2);
   readTXT("data/uniform_vertices_square10k.txt", Peval, 2);

  // U_BC vector
  FMCA::Vector u_bc(P_quad_border.cols());
  for (FMCA::Index i = 0; i < P_quad_border.cols(); ++i) {
    FMCA::Scalar x = P_quad_border(0, i);
    FMCA::Scalar y = P_quad_border(1, i);
    u_bc[i] = 0;
  }
  // Right hand side
  FMCA::Vector f(P_quad.cols());
  for (FMCA::Index i = 0; i < P_quad.cols(); ++i) {
    FMCA::Scalar x = P_quad(0, i);
    FMCA::Scalar y = P_quad(1, i);
    f[i] = 2 * FMCA_PI * FMCA_PI * sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }
  // Analytical sol of the problem
  FMCA::Vector analytical_sol(P_sources.cols());
  for (int i = 0; i < P_sources.cols(); ++i) {
    double x = P_sources(0, i);
    double y = P_sources(1, i);
    analytical_sol[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar threshold_kernel = 1e-6;
  //   const FMCA::Scalar threshold_gradKernel = 1e-1;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 4;
  const FMCA::Scalar beta = 10000;
  const std::string kernel_type = "MATERN32";

  const Moments mom_sources(P_sources, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  FMCA::Vector minDistance = minDistanceVector(hst_sources, P_sources);

  // fill distance
  auto maxElementIterator =
      std::max_element(minDistance.begin(), minDistance.end());
  FMCA::Scalar sigma_h = *maxElementIterator;
  std::cout << "fill distance:                      " << sigma_h << std::endl;
  std::vector<double> sigmas = {1.5, 1.6, 1.8, 2, 2.2, 2.4, 2.5, 2.7, 3};
  // 1.5, 1.6, 1.8, 2, 2.2, 2.4, 2.5, 2.7, 3
  for (FMCA::Scalar sigma_factor : sigmas) {
    std::cout << "sigma factor =                         " << sigma_factor <<
std::endl; FMCA::Scalar sigma = sigma_factor * sigma_h; FMCA::Vector u =
SolvePoisson_constantWeights( 2, P_sources, P_quad, w_vec, P_quad_border,
w_vec_border, Normals, u_bc, f, sigma, eta, dtilde, threshold_kernel, 1e-3,
threshold_weights, MPOLE_DEG, beta, kernel_type);

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
    const auto &triplets = Kcomp_ss.triplets();
    int anz = triplets.size() / P_sources.cols();
    std::cout << "Anz:                         " << anz << std::endl;

    Eigen::SparseMatrix<double> Kcomp_ss_Sparse(P_sources.cols(),
                                                P_sources.cols());
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
    std::cout << "Error:                                            " << error
              << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // plot solution
    {
      const Moments cmom(P_sources, MPOLE_DEG);
      const Moments rmom(Peval, MPOLE_DEG);
      const H2ClusterTree hct(cmom, 0, P_sources);
      const H2ClusterTree hct_eval(rmom, 0, Peval);
      const FMCA::Vector sx = hct.toClusterOrder(u_grid);
      const usMatrixEvaluatorKernel mat_eval(rmom, cmom, kernel_funtion_ss);
      FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster> hmat;
      hmat.computePattern(hct_eval, hct, eta);
      hmat.statistics();
      FMCA::Vector srec = hmat.action(mat_eval, sx);
      FMCA::Vector rec = hct_eval.toNaturalOrder(srec);
      FMCA::Plotter2D plotter;
      plotter.plotFunction(outputNamePlot, Peval, rec);

      // Analytical sol of the problem
      FMCA::Vector analytical_sol(Peval.cols());
      for (int i = 0; i < Peval.cols(); ++i) {
        double x = Peval(0, i);
        double y = Peval(1, i);
        analytical_sol[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
      }

      FMCA::Scalar error =
          (rec - analytical_sol).norm() / analytical_sol.norm();
      std::cout << "Error:                                            " << error
                << std::endl;

      FMCA::Vector absolute_error(Peval.cols());
      for (int i = 0; i < Peval.cols(); ++i) {
        absolute_error[i] = abs(rec[i] - analytical_sol[i]);
      }

      // plot error
      {
        FMCA::Plotter2D plotter_error;
        plotter_error.plotFunction(outputNameErrorPlot, Peval, absolute_error);
      }
    }
  }
  return 0;
}


// #include "SolvePoisson.h"
// ///////////////////////////////////
// #include <fstream>
// #include <limits>  // For numeric_limits

// #include "../FMCA/src/util/Plotter.h"
// #include "read_files_txt.h"

// #define DIM 2

// int main() {
//   // Output file to save results
//   std::ofstream outputFile("results_CSCS1_Galerkin.txt");

//   if (!outputFile.is_open()) {
//     std::cerr << "Failed to open the output file." << std::endl;
//     return -1;
//   }

//   // Factors to loop through
//   std::vector<FMCA::Scalar> factors = {0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1};

//   FMCA::Matrix P_sources_interior;
//   FMCA::Matrix P_sources_boundary;
//   FMCA::Matrix P_quad;
//   FMCA::Matrix P_quad_border;
//   FMCA::Matrix Peval;
//   FMCA::Matrix Normals;
//   FMCA::Vector w_vec;
//   FMCA::Vector w_vec_border;

//   readTXT("data/Halton250kInterior.txt", P_sources_interior, 2);
//   readTXT("data/Halton250kBnd.txt", P_sources_boundary, 2);
//   std::string outputNamePlot = "weakHalton_400.vtk";
//   std::string outputNameErrorPlot = "error_weakHalton_400.vtk";
//   readTXT("data/uniform_quadratures250k.txt", P_quad, 2);
//   readTXT("data/uniform_weights250k.txt", w_vec);
//   readTXT("data/quadrature5_border90k.txt", P_quad_border, 2);
//   readTXT("data/weights5_border90k.txt", w_vec_border);
//   readTXT("data/5normals90k.txt", Normals, 2);
//   readTXT("data/uniform_vertices_square10k.txt", Peval, 2);

//   // U_BC vector
//   FMCA::Vector u_bc(P_quad_border.cols());
//   for (FMCA::Index i = 0; i < P_quad_border.cols(); ++i) {
//     u_bc[i] = 0;
//   }
//   // Right hand side
//   FMCA::Vector f(P_quad.cols());
//   for (FMCA::Index i = 0; i < P_quad.cols(); ++i) {
//     FMCA::Scalar x = P_quad(0, i);
//     FMCA::Scalar y = P_quad(1, i);
//     f[i] = 2 * FMCA_PI * FMCA_PI * sin(FMCA_PI * x) * sin(FMCA_PI * y);
//   }

//   for (FMCA::Scalar factor : factors) {
//     // DATA
//     FMCA::Tictoc T;
//     FMCA::Matrix P_sources;

//     int total_columns_interior = P_sources_interior.cols();
//     int N = static_cast<int>(std::floor(factor * total_columns_interior));
//     int total_columns_boundary = P_sources_boundary.cols();
//     int M = static_cast<int>(std::floor(sqrt(factor) * total_columns_boundary));

//     // Concatenate matrices to form a new matrix
//     P_sources.resize(DIM, N + M);
//     for (int i = 0; i < DIM; ++i) {
//       for (int j = 0; j < N; ++j) {
//         P_sources(i, j) = P_sources_interior(i, j);
//       }
//       for (int j = 0; j < M; ++j) {
//         P_sources(i, N + j) = P_sources_boundary(i, j);
//       }
//     }

//     // Analytical sol of the problem
//     FMCA::Vector analytical_sol(P_sources.cols());
//     for (int i = 0; i < P_sources.cols(); ++i) {
//       double x = P_sources(0, i);
//       double y = P_sources(1, i);
//       analytical_sol[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
//     }

//     // Parameters
//     const FMCA::Scalar eta = 0.5;
//     const FMCA::Index dtilde = 3;
//     const FMCA::Scalar threshold_kernel = 1e-5;
//     const FMCA::Scalar threshold_weights = 0;
//     const FMCA::Scalar MPOLE_DEG = 4;
//     const FMCA::Scalar beta = 10000;
//     const std::string kernel_type = "MATERN32";

//     const Moments mom_sources(P_sources, MPOLE_DEG);
//     const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
//     H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
//     FMCA::Vector minDistance = minDistanceVector(hst_sources, P_sources);

//     // fill distance
//     auto maxElementIterator =
//         std::max_element(minDistance.begin(), minDistance.end());
//     FMCA::Scalar sigma_h = *maxElementIterator;

//     std::vector<double> sigmas = {1, 1.2, 1.4, 1.5, 1.8, 2, 2.2, 2.5};

//     double min_error = std::numeric_limits<double>::max();
//     double best_sigma_factor = 0;

//     for (FMCA::Scalar sigma_factor : sigmas) {
//       std::cout<< "sigma factor             : " << sigma_factor << std::endl; 
//       FMCA::Scalar sigma = sigma_factor * sigma_h;
//       FMCA::Vector u = SolvePoisson_constantWeights(
//           2, P_sources, P_quad, w_vec, P_quad_border, w_vec_border, Normals,
//           u_bc, f, sigma, eta, dtilde, threshold_kernel, sigma_h/10,
//           threshold_weights, MPOLE_DEG, beta, kernel_type);
//       std::cout<< "threshold grad kernel        : " << sigma_h/10 << std::endl; 

//       FMCA::Vector u_grid = hst_sources.inverseSampletTransform(u);
//       u_grid = hst_sources.toNaturalOrder(u_grid);

//       // Cumpute the solution K*u with the compression of the kernel matrix K
//       const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
//       const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources,
//                                                      kernel_funtion_ss);
//       FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
//       Kcomp_ss.init(hst_sources, eta, threshold_kernel);
//       Kcomp_ss.compress(mat_eval_kernel_ss);
//       const auto& triplets = Kcomp_ss.triplets();

//       Eigen::SparseMatrix<double> Kcomp_ss_Sparse(P_sources.cols(),
//                                                   P_sources.cols());
//       Kcomp_ss_Sparse.setFromTriplets(triplets.begin(), triplets.end());
//       Kcomp_ss_Sparse.makeCompressed();

//       Eigen::SparseMatrix<double> Kcomp_ss_Sparse_symm =
//           Kcomp_ss_Sparse.selfadjointView<Eigen::Upper>();
//       FMCA::Vector KU = Kcomp_ss_Sparse_symm * u;
//       KU = hst_sources.inverseSampletTransform(KU);
//       KU = hst_sources.toNaturalOrder(KU);

//       // Error
//       FMCA::Scalar error = (KU - analytical_sol).norm() / analytical_sol.norm();
//       std::cout << "error             : " << error << std::endl;

//       if (error < min_error) {
//         min_error = error;
//         best_sigma_factor = sigma_factor;
//       }
//     }

//     // Save the results for this factor
//     outputFile << "Factor: " << factor << ", "
//                << "Number of source points: " << (N + M) << ", "
//                << "Best sigma factor: " << best_sigma_factor << ", "
//                << "Minimum error: " << min_error << std::endl;
//   }

//   outputFile.close();
//   return 0;
// }
