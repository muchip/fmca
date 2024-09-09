/* We solve here the Poisson problem on the L shape [0,0], [1,0], [1,1], [-1,1],
[-1,-1], [-1,0] laplacian(u) = 1 u = 0 on \partial u Using Matern32 kernel and
penalty method to impose the boundary conditions. We rely on the FMCA library by
M.Multerer.
 */

#include "../FMCA/src/util/Plotter.h"
#include "SolvePoisson.h"
#include "read_files_txt.h"
#include "../FMCA/src/Samplets/adaptiveTreeSearch.h"

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
  FMCA::Vector w_vec_border;

  readTXT("data/int_and_bnd_L_25k.txt", P_sources, 2);
  readTXT("data/inside_points_L.txt", P_quad, 2);
  readTXT("data/quadrature_border_L.txt", P_quad_border, 2);
  readTXT("data/weights_border_L.txt", w_vec_border);
  readTXT("data/normals_L.txt", Normals, 2);

  std::string outputNamePlot = "L_25k.vtk";
  std::string outputNameExactSol = "L_25k_exact_sol.vtk";
  std::string outputNameErrorPlot = "L_error_25k.vtk";
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // U_BC vector
  int counter_u_bc_x = 0;
  int counter_u_bc_y = 0;
  FMCA::Vector u_bc(P_quad_border.cols());
  for (FMCA::Index i = 0; i < P_quad_border.cols(); ++i) {
    FMCA::Scalar x = P_quad_border(0, i);
    FMCA::Scalar y = P_quad_border(1, i);
    if (abs(y - 0.) < FMCA_ZERO_TOLERANCE){u_bc[i] = 10 * (1 - pow(x,4)); ++counter_u_bc_x;}
    else if (abs(x - 0.) < FMCA_ZERO_TOLERANCE){u_bc[i] = 10 * (1 - pow(y,4)); ++counter_u_bc_y;}
    else {u_bc[i] = 0;}
  }
  std::cout << "number of non hom boundary conditions in x =   " << counter_u_bc_x << std::endl;
  std::cout << "number of non hom boundary conditions in y =   " << counter_u_bc_y << std::endl;
  
  // Right hand side
  FMCA::Vector f(P_quad.cols());
  for (FMCA::Index i = 0; i < P_quad.cols(); ++i) {
    FMCA::Scalar x = P_quad(0, i);
    FMCA::Scalar y = P_quad(1, i);
    f[i] = 100;
  }
  // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 4;
  const FMCA::Scalar threshold_kernel = 1e-4;
  const FMCA::Scalar threshold_gradKernel = 1e-2;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 6;
  const FMCA::Scalar beta = 10000;
  const std::string kernel_type = "MATERN32";

  const Moments mom_sources(P_sources, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  FMCA::Vector minDistance = minDistanceVector(hst_sources, P_sources);

  FMCA::Scalar sigma_h = minDistance.maxCoeff();
  std::cout << "fill distance:                      " << sigma_h << std::endl;

  FMCA::Scalar sigma_mean = minDistance.mean();
  std::cout << "fill dist mean:                     " << sigma_mean
            << std::endl;

  std::cout << "fill distance:                      " << sigma_h << std::endl;
  std::vector<FMCA::Scalar> sigmas = {2 * sigma_h};
  FMCA::Scalar sigma = 2 * sigma_h;

  FMCA::Scalar w_interior = 3;
  FMCA::Vector u = SolvePoisson_constantWeighs(
      2, P_sources, P_quad, w_interior, P_quad_border, w_vec_border(0), Normals,
      u_bc, f, sigma, eta, dtilde, threshold_kernel, threshold_gradKernel,
      threshold_weights, MPOLE_DEG, beta, kernel_type);
  std::cout << "Solution Found." << std::endl;

  FMCA::Vector u_grid = hst_sources.inverseSampletTransform(u);
  u_grid = hst_sources.toNaturalOrder(u_grid);
  FMCA::Plotter2D plotter;
  plotter.plotFunction("U_L.vtk", P_sources, u_grid);

  //////////////////////////////////////////////////////////////////////////////
  std::vector<const H2SampletTree*> adaptive_tree =
      adaptiveTreeSearch(hst_sources, u, 1e-3 * u.squaredNorm());
  const FMCA::Index nclusters = std::distance(hst_sources.begin(), hst_sources.end());

  FMCA::Vector thres_tdata = u;
  thres_tdata.setZero();
  FMCA::Index nnz = 0;
  for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    if (adaptive_tree[i] != nullptr) {
      const H2SampletTree& node = *(adaptive_tree[i]);
      const FMCA::Index ndist =
          node.is_root() ? node.Q().cols() : node.nsamplets();
      thres_tdata.segment(node.start_index(), ndist) =
          u.segment(node.start_index(), ndist);
      nnz += ndist;
    }
  }
  std::cout << "active coefficients: " << nnz << " / " << P_sources.cols() << std::endl;
  std::cout << "tree error: " << (thres_tdata - u).norm() / u.norm()
            << std::endl;

  std::vector<FMCA::Matrix> bbvec_active;
  for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    if (adaptive_tree[i] != nullptr) {
      const H2SampletTree& node = *(adaptive_tree[i]);
      bbvec_active.push_back(node.bb());
    }
  }

  FMCA::IO::plotBoxes2D("active_boxes_LShape_25k.vtk", bbvec_active);


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // plot solution
    {
      // plot solution
      const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
      const Moments cmom(P_sources, MPOLE_DEG);
      const Moments rmom(P_sources, MPOLE_DEG);
      const H2ClusterTree hct(cmom, 0, P_sources);
      const H2ClusterTree hct_eval(rmom, 0, P_sources);
      const FMCA::Vector sx = hct.toClusterOrder(u_grid);
      const usMatrixEvaluatorKernel mat_eval(rmom, cmom, kernel_funtion_ss);
      FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster> hmat;
      hmat.computePattern(hct_eval, hct, eta);
      hmat.statistics();
      FMCA::Vector srec = hmat.action(mat_eval, sx);
      FMCA::Vector rec = hct_eval.toNaturalOrder(srec);
      FMCA::Plotter2D plotter;
      plotter.plotFunction("U_L_SampletTransform.vtk", P_sources, rec);
  }
  return 0;
}
