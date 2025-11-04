#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

#include "../FMCA/Clustering"
#include "../FMCA/CovarianceKernel"
#include "../FMCA/KernelInterpolation"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeInitializer_KDTree.h"
#include "../FMCA/src/Clustering/ClusterTreeSplitter.h"
#include "../FMCA/src/Clustering/KDTree.h"
#include "../FMCA/src/util/ForwardDeclarations.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2

using namespace FMCA;

////////////////////////////////////////////////////////////////////////////////////////////////
Scalar Function(Scalar x, Scalar y) {
  // return (0.5 * std::tanh(2000 * (x + y - 1)));
  Scalar term1 =
      (3.0 / 4.0) * std::exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0) / 4.0) -
                             ((9.0 * y - 2.0) * (9.0 * y - 2.0) / 4.0));
  Scalar term2 =
      (3.0 / 4.0) * std::exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0) / 49.0) -
                             ((9.0 * y - 2.0) * (9.0 * y - 2.0) / 10.0));
  Scalar term3 =
      (1.0 / 2.0) * std::exp(-((9.0 * x - 7.0) * (9.0 * x - 7.0) / 4.0) -
                             ((9.0 * y - 3.0) * (9.0 * y - 3.0) / 4.0));
  Scalar term4 = (1.0 / 5.0) * std::exp(-((9.0 * x - 4.0) * (9.0 * x - 4.0))
  -
                                        ((9.0 * y - 7.0) * (9.0 * y - 7.0)));
  return term1 + term2 + term3 - term4;
}

Vector evalFunction(const Matrix& Points) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = Function(Points(0, i), Points(1, i));
  }
  return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////
Matrix generateUniformGrid(int n) {
  int gridSize = std::sqrt(n);
  Matrix P(DIM, gridSize * gridSize);
  Scalar h = 1.0 / (gridSize - 1);
  int idx = 0;
  for (int i = 0; i < gridSize; ++i) {
    for (int j = 0; j < gridSize; ++j) {
      P(0, idx) = i * h;
      P(1, idx) = j * h;
      idx++;
    }
  }
  return P;
}

////////////////////////////////////////////////////////////////////////////////////////////////
void plotPoints(const std::string& filename, const Matrix& P,
                const Vector& data) {
  Matrix P3D(3, P.cols());
  P3D.setZero();
  P3D.topRows(2) = P;
  P3D.bottomRows(1) = data.transpose();
  FMCA::IO::plotPointsColor(filename, P3D, data);
}

////////////////////////////////////////////////////////////////////////////////////////////////
template <typename TreeType>
Index computeMaxLevel(const TreeType& tree) {
  Index max_level = 0;
  for (const auto& node : tree) {
    max_level = std::max(max_level, node.level());
  }
  return max_level;
}

////////////////////////////////////////////////////////////////////////////////////////////////
Index selectRepresentativeCenter(const Matrix& points_cluster,
                                 const Vector& f_cluster,
                                 const CovarianceKernel& kernel) {
  int m = points_cluster.cols();
  if (m == 0) return -1;
  if (m == 1) return 0;

  Scalar e_min = FMCA_INF;
  Index idx_min = -1;

  for (Index j = 0; j < m; ++j) {  // loop in the centers of the cluster
    Vector candidate_point = points_cluster.col(j);
    Scalar f_candidate_point = f_cluster(j);
    Scalar K_cc = kernel.eval(candidate_point, candidate_point)(0, 0);
    Scalar alpha_scalar = f_candidate_point / K_cc;
    Matrix alpha(1, 1);
    alpha(0, 0) = alpha_scalar;

    // Evaluate interpolant at all cluster points
    MultipoleFunctionEvaluator evaluator;
    evaluator.init(kernel, candidate_point, points_cluster);
    Vector eval_c = evaluator.evaluate(candidate_point, points_cluster, alpha);

    // Compute error
    Scalar e = (f_cluster - eval_c).squaredNorm();
    if (e < e_min) {
      e_min = e;
      idx_min = j;
    }
  }
  return idx_min;
}

////////////////////////////////////////////////////////////////////////////////////////////////
Index selectRepresentativeCenterRandom(const Matrix& points_cluster,
                                       const Vector& f_cluster,
                                       const CovarianceKernel& kernel) {
  int m = points_cluster.cols();
  if (m == 0) return -1;
  if (m == 1) return 0;

  Scalar e_min = FMCA_INF;
  Index idx_min = -1;

  // Instead of testing all points, test a random sample
  // int sample_size = std::min(100, m);
  // sample size = log(m) if m > 100, else sample_size = m
  int sample_size = log10(m);
  std::vector<Index> candidate_indices(m);
  std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(candidate_indices.begin(), candidate_indices.end(), g);

  for (Index j = 0; j < sample_size;
       ++j) {  // loop in the centers of the cluster
    Index idx = candidate_indices[j];
    Vector candidate_point = points_cluster.col(idx);
    Scalar f_candidate_point = f_cluster(idx);

    Scalar K_cc = kernel.eval(candidate_point, candidate_point)(0, 0);
    Scalar alpha_scalar = f_candidate_point / K_cc;
    Matrix alpha(1, 1);
    alpha(0, 0) = alpha_scalar;

    // Evaluate interpolant at all cluster points
    MultipoleFunctionEvaluator evaluator;
    evaluator.init(kernel, candidate_point, points_cluster);
    Vector eval_c = evaluator.evaluate(candidate_point, points_cluster, alpha);

    // Compute error
    Scalar e = (f_cluster - eval_c).squaredNorm();
    if (e < e_min) {
      e_min = e;
      idx_min = idx;
    }
  }
  return idx_min;
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

template<typename KernelSolver>
void runMultigridTest(Scalar nu) {
  Matrix P_finest = generateUniformGrid(262145);
  Matrix Peval = generateUniformGrid(1000000);
  Vector data = evalFunction(P_finest);
  Scalar computational_time_points_selection = 0;
  std::string boxes_name = "boxes_level_random_";
  std::string points_name = "centers_level_random_";

  std::cout << "Total points: " << P_finest.cols() << std::endl;

  //////////////////////////////// Storage
  FMCA::KDTree KDTree(P_finest, 10);
  std::vector<Matrix> P_Active;
  std::vector<Vector> F_Active;

  // Tree
  Index max_level = computeMaxLevel(KDTree);
  std::cout << "Maximum tree level: " << max_level << std::endl;
  std::map<Index, int> nodes_per_level;
  for (const auto& node : KDTree) {
    nodes_per_level[node.level()]++;
  }
  std::cout << "\nNodes per level:" << std::endl;
  for (const auto& pair : nodes_per_level) {
    std::cout << "  Level " << pair.first << ": " << pair.second << " nodes"
              << std::endl;
  }

  // is the kernel always the same?
  Scalar sigma = nu * pow(P_finest.cols(), -1. / DIM);

  for (Index l = 0; l <= max_level; ++l) {
    // Scalar sigma = nu * pow(P_finest.cols(), -1. / DIM) * pow(2, -l);
    CovarianceKernel kernel("matern32", sigma);
    
    std::cout << "\n=== Processing Level " << l << " ===" << std::endl;
    Tictoc time_level;
    time_level.tic();

    std::vector<Index> selected_indices;
    std::vector<Scalar> selected_values;
    std::vector<Matrix> bounding_boxes;
    std::vector<Scalar> box_colors;
    Vector point_colors(P_finest.cols());
    point_colors.setZero();

    int num_empty_nodes = 0;
    int num_non_empty_nodes = 0;

    // STEP 5: Loop through all nodes at this level
    for (const auto& node : KDTree) {
      if (node.level() != l) continue;       // Skip nodes not at current level
      int cluster_size = node.block_size();  // Number of points in this cluster

      if (cluster_size == 0) {
        num_empty_nodes++;
        continue;  // Skip empty nodes
      }

      num_non_empty_nodes++;

      // Random colors for bounding boxes
      const Scalar rdm_color = std::rand() % 256;
      Matrix cluster_points(DIM, cluster_size);
      Vector cluster_values(cluster_size);

      for (int i = 0; i < cluster_size; ++i) {
        Index global_idx = node.indices()[i];

        if (global_idx < 0 || global_idx >= P_finest.cols()) {
          std::cerr << "ERROR: Invalid index " << global_idx << " (max is "
                    << P_finest.cols() - 1 << ")" << std::endl;
          continue;
        }
        // Extract point coordinates and function value
        cluster_points.col(i) = P_finest.col(global_idx);
        cluster_values(i) = data(global_idx);
        point_colors(global_idx) = rdm_color;
      }
      bounding_boxes.push_back(node.bb());
      box_colors.push_back(rdm_color);

      
      Index local_best_idx = selectRepresentativeCenterRandom(
          cluster_points, cluster_values, kernel);

      if (local_best_idx >= 0 && local_best_idx < cluster_size) {
        Index global_best_idx = node.indices()[local_best_idx];
        selected_indices.push_back(global_best_idx);
        selected_values.push_back(data(global_best_idx));
      }
    }  // End of node loop
    Scalar time = time_level.toc();
    computational_time_points_selection += time;

    std::cout << "Empty nodes: " << num_empty_nodes << std::endl;
    std::cout << "Non-empty nodes: " << num_non_empty_nodes << std::endl;
    std::cout << "Selected centers: " << selected_indices.size() << std::endl;

    // Store results
    if (selected_indices.size() > 0) {
      Matrix P_level(DIM, selected_indices.size());
      Vector F_level(selected_indices.size());
      for (size_t i = 0; i < selected_indices.size(); ++i) {
        P_level.col(i) = P_finest.col(selected_indices[i]);
        F_level(i) = selected_values[i];
      }
      P_Active.push_back(P_level);
      F_Active.push_back(F_level);

      std::cout << "Stored " << selected_indices.size() << " centers for level "
                << l << std::endl;
    } else {
      std::cout << "WARNING: No centers selected at level " << l << std::endl;
      P_Active.push_back(Matrix(DIM, 0));
      F_Active.push_back(Vector(0));
    }
    if (bounding_boxes.size() > 0) {
      Matrix P3D(3, P_finest.cols());
      P3D.setZero();
      P3D.topRows(2) = P_finest;
      std::stringstream boxes_filename;
      boxes_filename << boxes_name << l << ".vtk";
      FMCA::IO::plotBoxes2D(boxes_filename.str(), bounding_boxes, box_colors);
      std::cout << "Saved boxes to: " << boxes_filename.str() << " for level "
                << l << std::endl;
    }

  }  // End of level loop

  // Plot all selected centers
  std::cout << "\n=== Generating final center plots ===" << std::endl;
  for (Index l = 0; l <= max_level; ++l) {
    if (l < static_cast<Index>(P_Active.size()) && P_Active[l].cols() > 0) {
      std::stringstream centers_filename;
      centers_filename << points_name << l << ".vtk";
      plotPoints(centers_filename.str(), P_Active[l], F_Active[l]);
      std::cout << "Saved centers to: " << centers_filename.str() << std::endl;
    }
  }
  std::cout << "Computational time: " << computational_time_points_selection
            << std::endl;

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////// now we normally solve the multilevel interpolation problem
  ////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar threshold = 1e-8;
  const Scalar mpole_deg = 2 * (dtilde - 1);
  const std::string kernel_type = "matern52";
  const Scalar ridgep = 0;
  const bool preconditioner = true;
  Scalar cg_threshold = 1e-6;

  std::cout << "Parameters:" << std::endl;
  std::cout << "- eta:              " << eta << std::endl;
  std::cout << "- dtilde:           " << dtilde << std::endl;
  std::cout << "- threshold a post: " << threshold << std::endl;
  std::cout << "- mpole_deg:        " << mpole_deg << std::endl;
  std::cout << "- kernel_type:      " << kernel_type << std::endl;
  std::cout << "- nu:               " << nu << std::endl;

  ////////////////////////////// Multiscale Interpolator
  MultiscaleInterpolator<KernelSolver> MSI;
  MSI.init(P_Active, dtilde, eta, threshold, ridgep, nu, DIM);
  const Index num_levels = MSI.numLevels();

  ////////////////////////////// Residuals
  std::vector<Vector> residuals(num_levels);
  for (int l = 0; l < num_levels; ++l) {
    residuals[l] = evalFunction(MSI.points(l));
  }

  ////////////////////////////// Vector of coefficients (solution at each level)
  std::vector<Vector> ALPHA(num_levels);
  std::vector<Scalar> fill_distances(num_levels);

  ////////////////////////////// Summary results
  std::vector<int> levels;
  std::vector<int> N;
  std::vector<Scalar> compression_time;
  std::vector<Scalar> cg_time;
  std::vector<int> iterationsCG;
  std::vector<size_t> anz;

  ////////////////////////////// Diagonal Loop
  Tictoc timer;
  for (int l = 0; l < num_levels; ++l) {
    std::cout << std::endl;
    std::cout << "-------- LEVEL " << (l + 1) << " --------" << std::endl;
    std::cout << "Number of points: " << MSI.points(l).cols() << std::endl;

    ////////////////////////////// Extra-Diagonal Loop
    for (int j = 0; j < l; ++j) {
      Scalar sigma_B = nu * pow(P_Active[0].cols(), -1. / DIM) * pow(2, -j);
      // Scalar sigma_B = nu * MSI.fillDistance(j);
      CovarianceKernel kernel_B(kernel_type, sigma_B);
      MultipoleFunctionEvaluator evaluator;
      evaluator.init(kernel_B, MSI.points(j), MSI.points(l));
      Matrix correction =
          evaluator.evaluate(MSI.points(j), MSI.points(l), ALPHA[j]);
      correction *= std::pow(sigma_B, -DIM);
      residuals[l] -= correction;
    }

    ////////////////////////////// Compress the diagonal block
    Scalar sigma_l = nu * pow(P_Active[0].cols(), -1. / DIM) * pow(2, -l);
    // Scalar sigma_l = nu * MSI.fillDistance(l);
    CovarianceKernel kernel_l(kernel_type, sigma_l);

    timer.tic();
    MSI.solver(l).compress(MSI.points(l), kernel_l);
    Scalar compress_time = timer.toc();

    ////////////////////////////// Statistics
    std::cout << "\nCompression Stats:" << std::endl;
    std::cout << "- Compression time: " << compress_time << " s" << std::endl;
    std::cout << "- ANZ (non-zeros per row): " << MSI.solver(l).anz()
              << std::endl;

    // Calcola compression error se necessario
    Scalar comp_error = MSI.solver(l).compressionError(MSI.points(l), kernel_l);
    std::cout << "- Compression error: " << comp_error << std::endl;

    ////////////////////////////// Solver
    timer.tic();
    Vector solution = MSI.solver(l).solveIteratively(
        residuals[l], preconditioner, cg_threshold);
    Scalar solver_time = timer.toc();

    solution /= std::pow(sigma_l, -DIM);
    ALPHA[l] = solution;

    //// Stats
    std::cout << "\nSolver Stats:" << std::endl;
    std::cout << "- Iterations: " << MSI.solver(l).solver_iterations()
              << std::endl;
    std::cout << "- Solver time: " << solver_time << " s" << std::endl;

    levels.push_back(l + 1);
    N.push_back(MSI.points(l).cols());
    compression_time.push_back(compress_time);
    cg_time.push_back(solver_time);
    iterationsCG.push_back(MSI.solver(l).solver_iterations());
    anz.push_back(MSI.solver(l).anz());
  }

  ////////////////////////////// Evaluation
  std::cout << "\nEvaluating solution on " << Peval.cols() << " points..."
            << std::endl;
  Vector exact_sol = evalFunction(Peval);
  Vector final_res = Vector::Zero(Peval.cols());
  std::vector<Scalar> l2_errors;
  std::vector<Scalar> linf_errors;
  for (int l = 0; l < num_levels; ++l) {
    Scalar sigma = nu * pow(P_Active[0].cols(), -1. / DIM) * pow(2, -l);
    // Scalar sigma = nu * MSI.fillDistance(l);
    CovarianceKernel kernel(kernel_type, sigma);
    MultipoleFunctionEvaluator evaluator;
    evaluator.init(kernel, MSI.points(l), Peval);
    Vector eval = evaluator.evaluate(MSI.points(l), Peval, ALPHA[l]);
    eval *= std::pow(sigma, -DIM);
    final_res += eval;
    // Errors
    Scalar l2_err = (final_res - exact_sol).norm() / exact_sol.norm();
    Scalar linf_err = (final_res - exact_sol).cwiseAbs().maxCoeff();
    l2_errors.push_back(l2_err);
    linf_errors.push_back(linf_err);
    std::cout << "======= Evaluation Level " << (l + 1)
              << " =======" << std::endl;
    std::cout << "- L2 error: " << l2_err << std::endl;
    std::cout << "- Linf error: " << linf_err << std::endl;
    std::cout << std::endl;
  }

  ////////////////////////////// Results Summary
  std::cout << "\n======== Results Summary ========" << std::endl;
  std::cout << std::left << std::setw(10) << "Level" << std::setw(10) << "N"
            << std::setw(15) << "CompressTime" << std::setw(15) << "CGTime"
            << std::setw(10) << "IterCG" << std::setw(10) << "ANZ"
            << std::setw(15) << "L2 Error" << std::setw(15) << "Linf Error"
            << std::endl;
  for (size_t i = 0; i < levels.size(); ++i) {
    std::cout << std::left << std::setw(10) << levels[i] << std::setw(10)
              << N[i] << std::setw(15) << std::fixed << std::setprecision(6)
              << compression_time[i] << std::setw(15) << std::fixed
              << std::setprecision(6) << cg_time[i] << std::setw(10)
              << iterationsCG[i] << std::setw(10) << static_cast<int>(anz[i])
              << std::setw(15) << std::scientific << std::setprecision(6)
              << l2_errors[i] << std::setw(15) << std::scientific
              << std::setprecision(6) << linf_errors[i] << std::endl;
  }
}

////////////////////////////// MAIN
int main() {
  std::srand(42);  // Fixed seed for reproducible colors
  Scalar nu = 1.0;
  runMultigridTest<SampletKernelSolver<>>(nu);
  return 0;
}
