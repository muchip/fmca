#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/KernelInterpolation"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"

using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// CONDITION NUMBER COMPUTATION

// Method 1: Lanczos-based estimation
Scalar estimateConditionNumberLanczos(const SparseMatrix& A,
                                      int max_iter = 50) {
  if (A.rows() != A.cols()) {
    return std::numeric_limits<Scalar>::infinity();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
  if (A.rows() > 5000) {
    // Use power iteration to estimate largest eigenvalue
    Vector v = Vector::Random(A.rows()).normalized();
    Scalar lambda_max = 0.0;
    for (int i = 0; i < max_iter; ++i) {
      Vector Av = A * v;
      lambda_max = v.dot(Av);
      v = Av.normalized();
    }
    // Use inverse power iteration for smallest eigenvalue
    Eigen::SparseLU<SparseMatrix> lu_solver;
    lu_solver.compute(A);
    if (lu_solver.info() != Eigen::Success) {
      return std::numeric_limits<Scalar>::infinity();
    }
    v = Vector::Random(A.rows()).normalized();
    Scalar lambda_min = std::numeric_limits<Scalar>::max();
    for (int i = 0; i < max_iter; ++i) {
      Vector x = lu_solver.solve(v);
      if (lu_solver.info() != Eigen::Success) {
        return std::numeric_limits<Scalar>::infinity();
      }
      Scalar rayleigh = v.dot(x);
      if (std::abs(rayleigh) > 1e-15) {
        lambda_min = std::min(lambda_min, 1.0 / rayleigh);
      }
      v = x.normalized();
    }
    return std::abs(lambda_max / lambda_min);
  } else {
    // For smaller matrices, convert to dense and use exact methods
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
    solver.compute(A_dense);
    if (solver.info() != Eigen::Success) {
      return std::numeric_limits<Scalar>::infinity();
    }
    auto eigenvals = solver.eigenvalues();
    Scalar lambda_max = eigenvals.maxCoeff();
    Scalar lambda_min = eigenvals.minCoeff();
    if (std::abs(lambda_min) < 1e-15) {
      return std::numeric_limits<Scalar>::infinity();
    }
    return std::abs(lambda_max / lambda_min);
  }
}

Scalar estimateConditionNumberSVD(const SparseMatrix& A, Scalar tol = 1e-12) {
  if (A.rows() != A.cols()) {
    return std::numeric_limits<Scalar>::infinity();
  }

  if (A.rows() > 2000) {
    std::cout << "Warning: SVD method not recommended for matrices larger than "
                 "2000x2000"
              << std::endl;
    return estimateConditionNumberLanczos(A);
  }
  // Convert to dense matrix for SVD
  Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A_dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto singular_values = svd.singularValues();
  Scalar sigma_max = singular_values(0);
  Scalar sigma_min = singular_values(singular_values.size() - 1);
  if (sigma_min < tol * sigma_max) {
    return std::numeric_limits<Scalar>::infinity();
  }
  return sigma_max / sigma_min;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
Scalar robustConditionNumber(const SparseMatrix& A) {
  std::cout << "Matrix size: " << A.rows() << "x" << A.cols()
            << ", nnz: " << A.nonZeros() << std::endl;

  if (A.rows() != A.cols()) {
    std::cout << "Matrix is not square!" << std::endl;
    return std::numeric_limits<Scalar>::infinity();
  }
  if (A.norm() < 1e-15) {
    std::cout << "Matrix is essentially zero!" << std::endl;
    return std::numeric_limits<Scalar>::infinity();
  }

  if (A.rows() <= 1000) {
    std::cout << "Using SVD method..." << std::endl;
    return estimateConditionNumberSVD(A);
  } else {
    std::cout << "Using Lanczos-based estimation..." << std::endl;
    return estimateConditionNumberLanczos(A);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
void exportResults1DToPython(
    const std::vector<Matrix>& P_Matrices, const Matrix& Peval,
    const Vector& exact_sol, const Vector& noisy_data, const Vector& final_res,
    const Vector& posterior_std, const std::vector<Vector>& ALPHA,
    const std::vector<Scalar>& fill_distances, const std::vector<int>& levels,
    const std::vector<int>& N, const std::vector<Scalar>& assembly_time,
    const std::vector<Scalar>& cg_time, const std::vector<int>& iterationsCG,
    const std::vector<Scalar>& anz, const std::vector<Scalar>& l2_errors,
    const std::vector<Scalar>& linf_errors, Scalar nu,
    const std::string& kernel_type, const std::string& test_function,
    const std::string& filename =
        "/Users/saraavesani/Desktop/Archive/hierarchical_gp_1d_results.py") {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }

  file << "import numpy as np\nimport matplotlib.pyplot as plt\nfrom "
          "matplotlib.patches import Rectangle\nimport seaborn as sns\n\n";
  file << "# Set "
          "style\nplt.style.use('seaborn-v0_8')\nsns.set_palette('husl')\n\n";

  // Export basic parameters
  file << "# Parameters\n";
  file << "nu = " << nu << "\n";
  file << "kernel_type = '" << kernel_type << "'\n";
  file << "test_function = '" << test_function << "'\n";
  file << "num_levels = " << levels.size() << "\n\n";

  // Export basic vectors
  auto exportVector = [&file](const std::string& name, const auto& vec) {
    file << name << " = np.array([";
    for (size_t i = 0; i < vec.size(); ++i) {
      file << vec[i];
      if (i + 1 < vec.size()) file << ", ";
    }
    file << "])\n";
  };

  file << "# Statistics\n";
  exportVector("levels", levels);
  exportVector("N", N);
  exportVector("fill_distances", fill_distances);
  exportVector("assembly_time", assembly_time);
  exportVector("cg_time", cg_time);
  exportVector("cg_iterations", iterationsCG);
  exportVector("anz", anz);
  exportVector("l2_errors", l2_errors);
  exportVector("linf_errors", linf_errors);

  // Export evaluation data
  file << "\n# Evaluation data\n";
  file << "eval_x = np.array([";
  for (Index i = 0; i < Peval.cols(); ++i) {
    file << Peval(0, i);
    if (i + 1 < Peval.cols()) file << ", ";
  }
  file << "])\n";

  file << "true_values = np.array([";
  for (Index i = 0; i < exact_sol.size(); ++i) {
    file << exact_sol(i);
    if (i + 1 < exact_sol.size()) file << ", ";
  }
  file << "])\n";

  file << "noisy_values = np.array([";
  for (Index i = 0; i < noisy_data.size(); ++i) {
    file << noisy_data(i);
    if (i + 1 < noisy_data.size()) file << ", ";
  }
  file << "])\n";

  file << "predicted_values = np.array([";
  for (Index i = 0; i < final_res.size(); ++i) {
    file << final_res(i);
    if (i + 1 < final_res.size()) file << ", ";
  }
  file << "])\n";

  file << "posterior_std = np.array([";
  for (Index i = 0; i < posterior_std.size(); ++i) {
    file << posterior_std(i);
    if (i + 1 < posterior_std.size()) file << ", ";
  }
  file << "])\n";

  // Export training points for each level
  file << "\n# Training points for each level\n";
  file << "training_points = []\n";
  for (size_t l = 0; l < P_Matrices.size(); ++l) {
    file << "training_points.append(np.array([";
    for (Index i = 0; i < P_Matrices[l].cols(); ++i) {
      file << P_Matrices[l](0, i);
      if (i + 1 < P_Matrices[l].cols()) file << ", ";
    }
    file << "]))\n";
  }

  // Add final Python print statement and close file (OUTSIDE the loop)
  file << "\nprint('Data exported successfully to Python!')\n";
  file.close();

  std::cout << "Results exported to " << filename << std::endl;
}