/* We solve here the Poisson problem on the square [-1,1]x[-1,1]
laplacian(u) = 2*pi^2*sin(pi*x)*sin(pi*y)
u = 0 on \partial u
Using Matern32 kernel and penalty method to impose the boundary conditions.
We rely on the FMCA library by M.Multerer.
 */

#include "SolvePoisson.h"
#include "read_files_txt.h"

#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluatorKernel =
    FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluatorKernel =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using EigenCholesky =
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>;

int main() {
  // POINTS
  FMCA::Tictoc T;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_quad_border;
  FMCA::Matrix Normals;
  FMCA::Vector w_vec;
  FMCA::Vector w_vec_border;
  // pointers
  readTXT("data/vertices_square_Halton40000.txt", P_sources, 2);
  readTXT("data/quadrature3_points_square40k.txt", P_quad, 2);
  readTXT("data/quadrature3_weights_square40k.txt", w_vec);
  readTXT("data/quadrature_border40k.txt", P_quad_border, 2);
  readTXT("data/weights_border40k.txt", w_vec_border);
  readTXT("data/normals40k.txt", Normals, 2);

  std::string outputNamePlot = "square_Halton40k.vtk";
  std::string outputNameErrorPlot = "square_error_Halton40k.vtk";
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
    // f[i] = -2;
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 4;
  const FMCA::Scalar threshold_kernel = 1e-6;
  const FMCA::Scalar threshold_gradKernel = 1e-2;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 6;
  const FMCA::Scalar beta = 20000;
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
  std::vector<FMCA::Scalar> sigmas = {2 * sigma_h};
  FMCA::Scalar sigma = 2 * sigma_h;

  FMCA::Vector u = SolvePoisson(
      2, P_sources, P_quad, w_vec, P_quad_border, w_vec_border, Normals, u_bc,
      f, sigma, eta, dtilde, threshold_kernel, threshold_gradKernel,
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
  KU = hst_sources.toNaturalOrder(KU);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // plot solution
  {
    Eigen::Vector2d min = {-1, -1};
    Eigen::Vector2d max = {1, 1};
    FMCA::Grid2D grid(min, max, 100, 100);
    const Moments cmom(P_sources, MPOLE_DEG);
    const FMCA::Matrix Peval = grid.P().topRows(2);
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
    grid.plotFunction(outputNamePlot, rec);

    // Analytical sol of the problem
    FMCA::Vector analytical_sol(Peval.cols());
    for (int i = 0; i < Peval.cols(); ++i) {
      double x = Peval(0, i);
      double y = Peval(1, i);
      analytical_sol[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
    }

    FMCA::Scalar error = (rec - analytical_sol).norm() / analytical_sol.norm();
    std::cout << "Error:                                            " << error
              << std::endl;

      FMCA::Vector absolute_error(Peval.cols());
      for (int i = 0; i < Peval.cols(); ++i) {
        absolute_error[i] = abs(rec[i] - analytical_sol[i]);
      }

      // plot error
      {
        Eigen::Vector2d min = {-1, -1};
        Eigen::Vector2d max = {1, 1};
        FMCA::Grid2D grid(min, max, 100, 100);
        const Moments cmom(P_sources, MPOLE_DEG);
        const FMCA::Matrix Peval = grid.P().topRows(2);
        const Moments rmom(Peval, MPOLE_DEG);
        const H2ClusterTree hct(cmom, 0, P_sources);
        const H2ClusterTree hct_eval(rmom, 0, Peval);
        const FMCA::Vector sx = hct.toClusterOrder(absolute_error);
        const usMatrixEvaluatorKernel mat_eval(rmom, cmom,
        kernel_funtion_ss); FMCA::H2Matrix<H2ClusterTree,
        FMCA::CompareCluster> hmat; hmat.computePattern(hct_eval, hct, eta);
        hmat.statistics();
        FMCA::Vector srec = hmat.action(mat_eval, sx);
        FMCA::Vector rec = hct_eval.toNaturalOrder(srec);
        grid.plotFunction(outputNameErrorPlot, rec);
      }
  }
  // std::vector<const H2SampletTree *> adaptive_tree =
  //     adaptiveTreeSearch(hst_sources, u, 1e-4 * u.squaredNorm());
  // const FMCA::Index nclusters =
  //     std::distance(hst_sources.begin(), hst_sources.end());

  // FMCA::Vector thres_tdata = u;
  // thres_tdata.setZero();
  // FMCA::Index nnz = 0;
  // for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
  //   if (adaptive_tree[i] != nullptr) {
  //     const H2SampletTree &node = *(adaptive_tree[i]);
  //     const FMCA::Index ndist =
  //         node.is_root() ? node.Q().cols() : node.nsamplets();
  //     thres_tdata.segment(node.start_index(), ndist) =
  //         u.segment(node.start_index(), ndist);
  //     nnz += ndist;
  //   }
  // }
  // std::cout << "active coefficients: " << nnz << " / " <<
  // P_sources.cols()
  //           << std::endl;
  // std::cout << "tree error: " << (thres_tdata - u).norm() / u.norm()
  //           << std::endl;

  // std::vector<FMCA::Matrix> bbvec_active;
  // for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
  //   if (adaptive_tree[i] != nullptr) {
  //     const H2SampletTree &node = *(adaptive_tree[i]);
  //     bbvec_active.push_back(node.bb());
  //   }
  // }

  // FMCA::IO::plotBoxes2D("active_boxes_Poisson_square.vtk", bbvec_active);
return 0;
}
