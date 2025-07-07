#include "GP_utils.cpp"

#define DIM 1

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// Test Functions
Scalar SinFunction1D(Scalar x) {
  return std::sin(2 * M_PI * x) + 0.5 * std::sin(6 * M_PI * x) +
         0.2 * std::sin(12 * M_PI * x);
}

Scalar SinFunction1D_Noisy(Scalar x) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::normal_distribution<Scalar> noise(0.0, 0.1);  // mean=0, std=0.1

  Scalar clean_value = std::sin(2 * M_PI * x) + 0.5 * std::sin(6 * M_PI * x) +
                       0.2 * std::sin(12 * M_PI * x);
  return clean_value + noise(gen);
}

Vector addNoise(const Vector& clean_data, Scalar noise_std,
                unsigned seed = 123) {
  std::mt19937 gen(seed);
  std::normal_distribution<Scalar> noise_dist(0.0, noise_std);

  Vector noisy_data = clean_data;
  for (Index i = 0; i < clean_data.size(); ++i) {
    noisy_data(i) += noise_dist(gen);
  }
  return noisy_data;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
/// Evaluate Functions
Vector evalFunction1D(const Matrix& Points, const std::string& func_name) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    if (func_name == "sin") {
      f(i) = SinFunction1D(Points(0, i));
    } else if (func_name == "sin_noisy") {
      f(i) = SinFunction1D_Noisy(Points(0, i));
    }
  }
  return f;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// Points generators
Matrix generateUniformPoints1D(int n) {
  Matrix P(DIM, n);
  for (int i = 0; i < n; ++i) {
    P(0, i) = static_cast<Scalar>(i) / (n - 1);
  }
  return P;
}
/////////////////////////////////////////////////////////////////////////
Matrix generateRandomPoints1D(int n, unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
  Matrix P(DIM, n);
  for (int i = 0; i < n; ++i) {
    P(0, i) = dist(gen);
  }
  // Sort the points
  std::vector<Scalar> points;
  for (int i = 0; i < n; ++i) {
    points.push_back(P(0, i));
  }
  std::sort(points.begin(), points.end());
  for (int i = 0; i < n; ++i) {
    P(0, i) = points[i];
  }
  return P;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// Points sets generators
std::vector<Matrix> generateHierarchicalPoints1D(
    const std::vector<int>& sizes, const std::string& distribution,
    unsigned seed = 42) {
  std::vector<Matrix> P_Matrices;

  if (distribution == "uniform") {
    for (int size : sizes) {
      P_Matrices.push_back(generateUniformPoints1D(size));
    }
  } else {
    // For random points, ensure nested structure
    std::mt19937 gen(seed);
    std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
    std::vector<Scalar> all_points;
    // Generate points for the finest level
    int max_size = *std::max_element(sizes.begin(), sizes.end());
    for (int i = 0; i < max_size; ++i) {
      all_points.push_back(dist(gen));
    }
    std::sort(all_points.begin(), all_points.end());
    // Create nested subsets
    for (int size : sizes) {
      Matrix P(DIM, size);
      int step = max_size / size;
      for (int i = 0; i < size; ++i) {
        int idx = std::min(i * step, max_size - 1);
        P(0, i) = all_points[idx];
      }
      P_Matrices.push_back(P);
    }
  }
  return P_Matrices;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
/// Main function

void runHierarchicalGPTest1D(Scalar nu) {
  std::cout << "======== Running 1D Hierarchical GP Test with nu = " << nu
            << " ========" << std::endl;
  //////////////////////////////
  std::string test_function = "sin";
  std::string point_distribution = "uniform";
  std::vector<int> level_sizes = {65,   129,  257,   513};
                                  // ,  1025, 2049, 8193, 16385, 32769};
  std::vector<Scalar> noise_levels = {
      0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0.};  // {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                            // 0.1};
  std::string kernel_type = "matern52";
  Index dtilde = 5;
  Scalar mpole_deg = 2 * (dtilde - 1);
  Scalar eta = 1. / 2;
  Scalar threshold = 1e-8;
  Scalar ridgep = 0;
  const bool preconditioner = true;
  Scalar cg_threshold = 1e-6;
  int eval_points_count = 1000;
  unsigned random_seed = 42;

  std::cout << "Parameters:" << std::endl;
  std::cout << "- eta:              " << eta << std::endl;
  std::cout << "- dtilde:           " << dtilde << std::endl;
  std::cout << "- threshold:        " << threshold << std::endl;
  std::cout << "- kernel_type:      " << kernel_type << std::endl;
  std::cout << "- nu:               " << nu << std::endl;

  ////////////////////////////// Points
  int max_level = level_sizes.size();
  std::vector<Matrix> P_Matrices = generateHierarchicalPoints1D(
      level_sizes, point_distribution, random_seed);
  Matrix Peval = generateUniformPoints1D(eval_points_count);

  ////////////////////////////// Residuals
  std::vector<Vector> residuals;
  for (int i = 0; i < max_level; ++i) {
    Vector clean_obs = evalFunction1D(P_Matrices[i], test_function);
    residuals.push_back(clean_obs);
  }

  ////////////////////////////// Vector of coefficients
  std::vector<Vector> ALPHA;
  ALPHA.reserve(max_level);
  std::vector<Scalar> fill_distances;
  Vector posterior_var = Vector::Zero(eval_points_count);
  Vector posterior_std = Vector::Zero(eval_points_count);

  ////////////////////////////// Summary results
  std::vector<int> levels;
  std::vector<int> N;
  std::vector<Scalar> assembly_time;
  std::vector<Scalar> cg_time;
  std::vector<int> iterationsCG;
  std::vector<Scalar> anz;

  ////////////////////////////// Diagonal Loop
  for (int l = 0; l < max_level; ++l) {
    std::cout << std::endl;
    std::cout << "-------- LEVEL " << (l + 1) << " --------" << std::endl;
    MultilevelSampletKernelSolver<> solver;
    CovarianceKernel kernel(kernel_type, 1);

    // Add ridge regression for noise
    Scalar ridgep = noise_levels[l] * noise_levels[l];

    solver.init(kernel, P_Matrices[l], dtilde, eta, threshold, ridgep);
    Scalar h = solver.fill_distance();
    fill_distances.push_back(h);
    int n_pts = P_Matrices[l].cols();
    std::cout << "Fill distance: " << h << std::endl;
    std::cout << "Number of points: " << n_pts << std::endl;
    Scalar noise_std = (l < noise_levels.size()) ? noise_levels[l] : 0.01;
    std::cout << "Noise level: " << noise_std << std::endl;

    ////////////////////////////// Extra-Diagonal Loop
    for (int j = 0; j < l; ++j) {
      Scalar sigma_B = nu * fill_distances[j];
      CovarianceKernel kernel_B(kernel_type, sigma_B);
      MultipoleFunctionEvaluator evaluator;
      evaluator.init(kernel_B, P_Matrices[j], P_Matrices[l]);
      Matrix correction =
          evaluator.evaluate(P_Matrices[j], P_Matrices[l], ALPHA[j]);
      correction *= std::pow(sigma_B, -DIM);
      residuals[l] -= correction;
    }

    ////////////////////////////// Compress the diagonal block
    Scalar sigma_l = nu * fill_distances[l];
    CovarianceKernel kernel_l(kernel_type, sigma_l);

    solver.updateKernel(kernel_l);
    solver.compress(P_Matrices[l]);
    solver.compressionError(P_Matrices[l]);

    ////////////////////////////// Statistics
    const auto& compressor_stats = solver.getCompressionStats();
    std::cout << "\nCompression Stats:" << std::endl;
    std::cout << "- Compression time: " << compressor_stats.time_compressor
              << " s" << std::endl;
    std::cout << "- ANZ (non-zeros per row): " << compressor_stats.anz
              << std::endl;
    std::cout << "- Compression error: " << compressor_stats.compression_error
              << std::endl;

    ////////////////////////////// Solver
    Vector solution =
        solver.solveIterative(residuals[l], preconditioner, cg_threshold);
    solution /= std::pow(sigma_l, -DIM);

    const auto& cg_stats = solver.getSolverStats();
    std::cout << "Solver Stats:" << std::endl;
    std::cout << "- Iterations: " << cg_stats.iterations << std::endl;
    std::cout << "- Residual: " << cg_stats.residual_error << std::endl;
    std::cout << "- Solver time: " << cg_stats.solver_time << " s" << std::endl;

    levels.push_back(l + 1);
    N.push_back(n_pts);
    assembly_time.push_back(compressor_stats.assembly_time);
    cg_time.push_back(cg_stats.solver_time);
    iterationsCG.push_back(cg_stats.iterations);
    anz.push_back(compressor_stats.anz);

    ////////////////////////////// Posterior variance
    // if (l == max_level - 1) {
    std::cout << "\nComputing posterior variance..." << std::endl;

    /////////////////////////// K (P_L P_L)
    MultilevelSampletKernelSolver<> solver_cov;
    solver_cov.init(kernel_l, P_Matrices[max_level-1], dtilde, eta, threshold, ridgep);
    solver_cov.compress(P_Matrices[max_level-1]);
    SparseMatrix K_ell = solver_cov.K();

    /////////////////////////// K (Peval, Peval)
    MultilevelSampletKernelSolver<> solver_eval;
    solver_eval.init(kernel_l, Peval, dtilde, eta, threshold, ridgep);
    solver_eval.compress(Peval);
    SparseMatrix K_eval = solver_eval.K();

    /////////////////////////// K (Peval, P_ell)
    const Moments mom_rows(Peval, mpole_deg);
    const Moments mom_cols(P_Matrices[max_level-1], mpole_deg);
    const usMatrixEvaluator mat_eval(mom_rows, mom_cols, kernel_l);
    const SampletMoments samp_mom_rows(Peval, dtilde - 1);
    const SampletMoments samp_mom_cols(P_Matrices[max_level-1], dtilde - 1);
    FMCA::H2SampletTree<FMCA::ClusterTree> hst_rows(mom_rows, samp_mom_rows, 0, Peval);
    FMCA::H2SampletTree<FMCA::ClusterTree> hst_cols(mom_cols, samp_mom_cols, 0, P_Matrices[max_level-1]);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<
        FMCA::H2SampletTree<FMCA::ClusterTree>>
        K_eval_ell_comp;
    K_eval_ell_comp.init(hst_rows, hst_cols, eta, 1e-15);
    K_eval_ell_comp.compress(mat_eval);
    K_eval_ell_comp.triplets();
    const auto& trips = K_eval_ell_comp.aposteriori_triplets(1e-8);
    SparseMatrix K_eval_ell(Peval.cols(), P_Matrices[max_level-1].cols());
    K_eval_ell.setFromTriplets(trips.begin(), trips.end());
    K_eval_ell.makeCompressed();

    /////////////////////////// Fully Sparse Posterior Variance Computation
    Vector K_eval_diag = Vector::Zero(eval_points_count);
    for (int i = 0; i < eval_points_count; ++i) {
      K_eval_diag(i) = K_eval.coeff(i, i);
    }

    // Process evaluation points in chunks to show progress
    const int chunk_size = 10000;
    const int num_chunks = (eval_points_count + chunk_size - 1) / chunk_size;

#pragma omp parallel for schedule(dynamic)
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
      int start_idx = chunk * chunk_size;
      int end_idx = std::min(start_idx + chunk_size, eval_points_count);

      for (int i = start_idx; i < end_idx; ++i) {
        // Extract i-th row of K_eval_ell as a sparse vector
        Vector k_eval_ell_i = Vector::Zero(P_Matrices[max_level-1].cols());

        // Only iterate over non-zero elements in row i
        for (SparseMatrix::InnerIterator it(K_eval_ell, i); it; ++it) {
          k_eval_ell_i(it.col()) = it.value();
        }

        // Skip if row is empty (no covariance)
        if (k_eval_ell_i.norm() < 1e-15) {
          posterior_var(i) = K_eval_diag(i);
          continue;
        }

        // Solve K_ell * a_i = k_eval_ell_i
        Vector a_i =
            solver_cov.solveIterative(k_eval_ell_i, preconditioner, cg_threshold);

        // Compute variance: Var_i = K_eval[i,i] - k_eval_ell_i^T * a_i
        Scalar quadratic_form = k_eval_ell_i.dot(a_i);
        Scalar variance = K_eval_diag(i) - quadratic_form;

        // Ensure non-negative variance (numerical stability)
        posterior_var(i) = std::max(0.0, variance);
      }

// Progress reporting (thread-safe)
#pragma omp critical
      {
        if (chunk % 10 == 0) {
          std::cout << "Processed chunks: " << chunk << "/" << num_chunks
                    << " (" << (chunk * 100 / num_chunks) << "%)" << std::endl;
        }
      }
    }
    // posterior_var *= std::pow(sigma_l, -DIM);
    posterior_var = hst_rows.inverseSampletTransform(posterior_var);
    posterior_var = hst_rows.toNaturalOrder(posterior_var);
    // }

    std::cout << "Posterior variance computation completed." << std::endl;
    if (l == 0) {
      posterior_std = posterior_var;
    } else {
      posterior_std += posterior_var;
    }

    ////////////////////////////// Solution update
    ALPHA.push_back(solution);
  }

  ////////////////////////////// Evaluation
  std::cout << "\nEvaluating solution on " << Peval.cols() << " points..."
            << std::endl;
  Vector exact_sol = evalFunction1D(Peval, test_function);
  Vector final_res = Vector::Zero(Peval.cols());
  std::vector<Scalar> l2_errors;
  std::vector<Scalar> linf_errors;

  for (int l = 0; l < max_level; ++l) {
    Scalar sigma = nu * fill_distances[l];
    CovarianceKernel kernel(kernel_type, sigma);
    MultipoleFunctionEvaluator evaluator;
    evaluator.init(kernel, P_Matrices[l], Peval);
    Vector eval = evaluator.evaluate(P_Matrices[l], Peval, ALPHA[l]);
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

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Results Summary
  ////////////////////////////// Export to Python
  Vector noisy_data = evalFunction1D(Peval, "sin_noisy");

  exportResults1DToPython(P_Matrices, Peval, exact_sol, noisy_data, final_res,
                          posterior_std, ALPHA, fill_distances, levels, N,
                          assembly_time, cg_time, iterationsCG, anz, l2_errors,
                          linf_errors, nu, kernel_type, test_function);

  ////////////////////////////// Results Summary
  std::cout << "\n======== Results Summary ========" << std::endl;
  std::cout << std::left << std::setw(8) << "Level" << std::setw(8) << "N"
            << std::setw(12) << "AssemblyT" << std::setw(10) << "CGTime"
            << std::setw(8) << "CGIter" << std::setw(10) << "ANZ"
            << std::setw(12) << "L2Error" << std::setw(12) << "L∞Error"
            << std::endl;

  for (size_t i = 0; i < levels.size(); ++i) {
    std::cout << std::left << std::setw(8) << levels[i] << std::setw(8) << N[i]
              << std::setw(12) << std::fixed << std::setprecision(6)
              << assembly_time[i] << std::setw(10) << std::fixed
              << std::setprecision(6) << cg_time[i] << std::setw(8)
              << iterationsCG[i] << std::setw(10) << static_cast<int>(anz[i])
              << std::setw(12) << std::scientific << std::setprecision(3)
              << l2_errors[i] << std::setw(12) << std::scientific
              << std::setprecision(3) << linf_errors[i] << std::endl;
  }
}

////////////////////////////// MAIN
int main() {
  // Test different nu values
  std::vector<Scalar> nus = {2.0};

  for (Scalar nu : nus) {
    runHierarchicalGPTest1D(nu);
    std::cout << "\n\n";
  }

  return 0;
}