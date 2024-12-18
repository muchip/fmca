#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#include "MultigridSolver.h"

#define DIM 2
using namespace FMCA;

struct GridKey {
  int x, y;
  bool operator==(const GridKey &other) const {
    return x == other.x && y == other.y;
  }
};

namespace std {
template <>
struct hash<GridKey> {
  std::size_t operator()(const GridKey &k) const {
    return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1);
  }
};
}  // namespace std

GridKey pointToGridKey(Scalar x, Scalar y, Scalar cell_size) {
  return {static_cast<int>(std::floor(x / cell_size)),
          static_cast<int>(std::floor(y / cell_size))};
}

Matrix generateGreedyPointsHash(Scalar x_min, Scalar x_max, Scalar y_min,
                                Scalar y_max, Scalar target_fill_distance,
                                Matrix bb_points, Scalar eps) {
  Scalar cell_size = target_fill_distance;
  std::unordered_map<GridKey, std::vector<Eigen::Vector2d>> grid;

  int num_points = bb_points.cols() * 4;
  Matrix points(2, bb_points.cols() + num_points);
  points.block(0, 0, 2, bb_points.cols()) = bb_points;

  for (int i = 0; i < bb_points.cols(); ++i) {
    Eigen::Vector2d point = bb_points.col(i);
    grid[pointToGridKey(point.x(), point.y(), cell_size)].push_back(point);
  }

  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<Scalar> x_dist(x_min, x_max);
  std::uniform_real_distribution<Scalar> y_dist(y_min, y_max);

  int count = bb_points.cols();
  while (count < bb_points.cols() + num_points) {
    Scalar x = x_dist(rng);
    Scalar y = y_dist(rng);

    Eigen::Vector2d candidate(x, y);
    GridKey candidate_key = pointToGridKey(x, y, cell_size);

    bool is_valid = true;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        GridKey neighbor_key = {candidate_key.x + dx, candidate_key.y + dy};
        if (grid.find(neighbor_key) != grid.end()) {
          for (const auto &neighbor_point : grid[neighbor_key]) {
            Scalar dist = (neighbor_point - candidate).norm();
            if (dist < target_fill_distance - eps ||
                dist > target_fill_distance + eps) {
              is_valid = false;
              break;
            }
          }
        }
        if (!is_valid) break;
      }
      if (!is_valid) break;
    }

    if (is_valid) {
      points.col(count) = candidate;
      grid[candidate_key].push_back(candidate);
      ++count;
    }
  }
  return points.block(0, bb_points.cols(), 2, count - bb_points.cols());
}

Matrix generateInitialGridPoints(Scalar x_min, Scalar x_max, Scalar y_min,
                                 Scalar y_max, Scalar target_fill_distance) {
  int num_x = static_cast<int>((x_max - x_min) / target_fill_distance);
  int num_y = static_cast<int>((y_max - y_min) / target_fill_distance);

  Matrix points(2, num_x * num_y);
  int index = 0;
  for (int i = 0; i < num_x; ++i) {
    for (int j = 0; j < num_y; ++j) {
      points(0, index) = x_min + i * target_fill_distance;
      points(1, index) = y_min + j * target_fill_distance;
      ++index;
    }
  }
  points.conservativeResize(2, index);
  return points;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// int main() {
//     Scalar x_min = -1.0, x_max = 1.0, y_min = -1.0, y_max = 1.0;
//     Scalar target_fill_distance = 0.1;
//     Scalar eps = 0.001;

//     // Generate initial quasi-uniform grid points for the bounding box
//     Matrix bb_points = generateInitialGridPoints(x_min, x_max, y_min, y_max,
//     target_fill_distance);

//     std::cout << "Testing generateGreedyPoints function...\n";
//     auto start = std::chrono::high_resolution_clock::now();
//     Matrix generated_points = generateGreedyPoints(x_min, x_max, y_min,
//     y_max, target_fill_distance/4, bb_points, eps); auto end =
//     std::chrono::high_resolution_clock::now(); std::chrono::duration<double>
//     duration = end - start; std::cout << "generateGreedyPoints generated " <<
//     generated_points.cols()
//               << " points in " << duration.count() << " seconds.\n\n";
//   FMCA::IO::plotPoints2D("Plots/testGreedyOld.vtk", bb_points);
//   FMCA::IO::plotPoints2D("Plots/testGreedyNew.vtk", generated_points);
//   return 0;
// }

Scalar AnalyticalSolution(Scalar x, Scalar y) {
  Scalar r = sqrt(x * x + y * y);
  Scalar phi = atan2(y, x);
  if (phi <= 0) {
    phi += 2 * FMCA_PI;  // Adjust phi to be in [Pi/2,2*Pi]
  }
  if (r == 0) {
    return 0;
  }
  return -pow(r, 2.0 / 3.0) * sin((2 * phi - M_PI) / 3);
}

Vector evalAnalyticalSolution(Matrix Points) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = AnalyticalSolution(Points(0, i), Points(1, i));
  }
  return f;
}

int main() {
  Tictoc T;
  ///////////////////////////////// Inputs: points + maximum level
  Matrix P1 = generateInitialGridPoints(-0.5, 0.5, -0.5, 0.5, 0.25);
  readTXT("data/L_shape_uniform_grid_level1.txt", P1, DIM);
  filterPointsInDomain(P1);

  Matrix Peval;
  readTXT("data/L_shape_uniform_grid_level8.txt", Peval, DIM);
  std::cout << "Cardianlity Peval   " << Peval.cols() << std::endl;

  std::vector<Matrix> P_Matrices = {P1};

  ///////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar threshold_kernel = 1e-4;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = 2 * (dtilde - 1);
  const std::string kernel_type = "exponential";
  const Scalar nu = 1;
  const std::string base_filename = "Plots/newPointsGreedy";
  const std::string base_filename_bb = "Plots/bbGreedy";

  std::vector<Vector> evaluated_rhs;
  for (Index i = 0; i < P_Matrices.size(); ++i) {
    Vector rhs = evalAnalyticalSolution(P_Matrices[i]);
    evaluated_rhs.push_back(rhs);
  }

  ///////////////////////////////// Solve the first level
  MultigridSolver(P_Matrices, eta, dtilde, threshold_kernel, threshold_weights,
                  kernel_type, nu, evaluated_rhs, 0, "ConjugateGradient",
                  "alpha_Lshape", "residuals_Lshape", "fill_distances_Lshape");

  std::vector<Vector> ALPHA;
  std::vector<Vector> RESIDUALS;
  Vector fill_distances;

  ALPHA = readBinaryVectors("alpha_Lshape");
  RESIDUALS = readBinaryVectors("residuals_Lshape");
  readTXT("fill_distances_Lshape", fill_distances);

  /////////////////////////////////  Evaluation of the first level
  const Moments mom(Peval, mpole_deg);
  const SampletMoments samp_mom(Peval, dtilde - 1);
  const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, Peval);
  Vector exact_sol = evalAnalyticalSolution(Peval);

  ///////////////////////////////// Plot old points and the refined points
  std::cout << "-----------------------------------" << std::endl;
  IO::plotPoints2D("Plots/oldPointsGreedy.vtk",
                   P_Matrices[P_Matrices.size() - 1]);
  {
    std::vector<FMCA::Matrix> bb_active = MarkActiveLeavesSampletCoefficients(
        P_Matrices, RESIDUALS, P_Matrices.size() - 1, mpole_deg, dtilde, 0.01);
    std::ostringstream oss_bb;
    oss_bb << base_filename_bb << "_level_" << 1 << ".vtk";
    std::string filename_bb = oss_bb.str();

    IO::plotBoxes2D(filename_bb, bb_active);

    int counter = 0;
    int total_points = 0;
    Matrix newMatrix(2, 0);
    for (const auto &matrix : bb_active) {
      counter++;
      Scalar x_min = matrix(0, 0);
      Scalar x_max = matrix(0, 1);
      Scalar y_min = matrix(1, 0);
      Scalar y_max = matrix(1, 1);

      Matrix newPoints = generateGreedyPointsHash(
          x_min, x_max, y_min, y_max,
          fill_distances[fill_distances.size() - 1] / 4,
          P_Matrices[P_Matrices.size() - 1],
          1. / 10 * fill_distances[fill_distances.size() - 1]);

      Matrix newPointsFilitered = filterPointsInDomain(newPoints);

      // Update total_points and resize newMatrix
      int filtered_cols = newPointsFilitered.cols();
      newMatrix.conservativeResize(2, total_points + filtered_cols);

      // Add newPointsFiltered to newMatrix
      newMatrix.block(0, total_points, 2, filtered_cols) = newPointsFilitered;

      // Update total_points
      total_points += filtered_cols;
    }

    std::ostringstream oss;
    oss << base_filename << "_level_" << 1 << ".vtk";
    std::string filename_solution = oss.str();

    IO::plotPoints2D(filename_solution, newMatrix);

    Matrix nextP_Matrix(
        DIM, P_Matrices[P_Matrices.size() - 1].cols() + newMatrix.cols());
    nextP_Matrix << P_Matrices[P_Matrices.size() - 1], newMatrix;
    P_Matrices.emplace_back(nextP_Matrix);
  }
  std::cout << "Iteration 1 done" << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int max_iter = 3;
  for (int i = 0; i < max_iter; ++i) {
    MultigridSolverWithPreviousSolutions(
        P_Matrices, eta, dtilde, threshold_kernel, threshold_weights,
        kernel_type, nu,
        evalAnalyticalSolution(P_Matrices[P_Matrices.size() - 1]), 0,
        "ConjugateGradient", ALPHA, RESIDUALS, fill_distances, "alpha_Lshape",
        "residuals_Lshape", "fill_distances_Lshape");

    ALPHA = readBinaryVectors("alpha_Lshape");
    RESIDUALS = readBinaryVectors("residuals_Lshape");
    readTXT("fill_distances_Lshape", fill_distances);

    if (i == max_iter - 1) {
      Vector solution =
          Evaluate(mom, samp_mom, hst, kernel_type, P_Matrices, Peval, ALPHA,
                   fill_distances, P_Matrices.size(), nu, eta, threshold_kernel,
                   mpole_deg, dtilde, exact_sol, hst, "");
    }

    std::cout << "-----------------------------------" << std::endl;
    if (i < max_iter - 1)
    {
      std::vector<FMCA::Matrix> bb_active = MarkActiveLeavesSampletCoefficients(
          P_Matrices, RESIDUALS, P_Matrices.size() - 1, mpole_deg, dtilde,
          0.01);
      std::ostringstream oss_bb;
      oss_bb << base_filename_bb << "_level_" << i + 2 << ".vtk";
      std::string filename_bb = oss_bb.str();

      IO::plotBoxes2D(filename_bb, bb_active);
      std::cout << "BB level" << i + 2 << "ploetted." << std::endl;
      std::cout << " " << std::endl;

      int counter = 0;
      int total_points = 0;
      Matrix newMatrix(2, 0);
      for (const auto &matrix : bb_active) {
        counter++;
        Scalar x_min = matrix(0, 0);
        Scalar x_max = matrix(0, 1);
        Scalar y_min = matrix(1, 0);
        Scalar y_max = matrix(1, 1);
        Matrix CheckPoints = PointsInsideBB2D(
            x_min, x_max, y_min, y_max, P_Matrices[P_Matrices.size() - 1]);

        Matrix newPoints = generateGreedyPointsHash(
            x_min, x_max, y_min, y_max,
            fill_distances[fill_distances.size() - 1] / 8, CheckPoints,
            0);  // 1. / 10 * fill_distances[fill_distances.size() - 1]

        Matrix newPointsFilitered = filterPointsInDomain(newPoints);

        // Update total_points and resize newMatrix
        int filtered_cols = newPointsFilitered.cols();
        newMatrix.conservativeResize(2, total_points + filtered_cols);

        // Add newPointsFiltered to newMatrix
        newMatrix.block(0, total_points, 2, filtered_cols) = newPointsFilitered;

        // Update total_points
        total_points += filtered_cols;
      }

      std::ostringstream oss;
      oss << base_filename << "_level_" << i + 2 << ".vtk";
      std::string filename_solution = oss.str();

      IO::plotPoints2D(filename_solution, newMatrix);

      Matrix nextP_Matrix(
          DIM, P_Matrices[P_Matrices.size() - 1].cols() + newMatrix.cols());
      nextP_Matrix << P_Matrices[P_Matrices.size() - 1], newMatrix;
      P_Matrices.emplace_back(nextP_Matrix);
    }
    std::cout << "Iteration " << i + 2 << " done" << std::endl;
    std::cout << "-----------------------------------------------------------"
              << std::endl;
    std::cout << "-----------------------------------------------------------"
              << std::endl;
  }
  ////////////////////////////////////////////////////////////////////////////

  //   std::cout << "number bb_active = " <<  counter << std::endl;

  //   VisualizeActiveLeavesSampletCoefficients(bb_active, "Plots/bb_Lshape",
  //                                            "Plots/points_Lshape");

  //   VisualizePointsError(mom, samp_mom, hst, kernel_type, P_Matrices,
  //   Peval,
  //                        ALPHA, fill_distances, max_level, nu, eta,
  //                        threshold_kernel, mpole_deg, dtilde, exact_sol,
  //                        "Plots/error_Lshape");

  return 0;
}
