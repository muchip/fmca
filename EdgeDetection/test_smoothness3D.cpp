#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
// ########################
#include "SmoothnessDetection.h"

#define DIM 3
using namespace FMCA;

//////////////////////////////////////////////////////////////// VOLUME
// Generate uniform points inside the volume of a unit cube
Matrix generateCubeVolumePoints(int num_points) {
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<Scalar> dist(0.0, 1.0);

  Matrix points(DIM, num_points);
  for (int i = 0; i < num_points; ++i) {
    points(0, i) = dist(rng);
    points(1, i) = dist(rng);
    points(2, i) = dist(rng);
  }
  return points;
}

// Assign values to points based on the region in the unit cube
Vector assignRegionValues(const Matrix& points) {
  Vector values(points.cols());
  for (int i = 0; i < points.cols(); ++i) {
    Scalar x = points(0, i), y = points(1, i), z = points(2, i);
    int regionIndex = (x < 0.55 ? 0 : 1) + (y < 0.55 ? 0 : 2) + (z < 0.55 ? 0 : 4);
    values[i] = regionIndex + 1;  // Region values from 1 to 8
  }
  return values;
}

// // Apply a smooth perturbation to points
// Matrix perturbPoints(const Matrix& points, Scalar epsilon) {
//   Matrix perturbed = points;
//   for (int i = 0; i < points.cols(); ++i) {
//     perturbed(0, i) += epsilon * std::sin(2 * FMCA_PI * points(1, i));
//     perturbed(1, i) += epsilon * std::cos(2 * FMCA_PI * points(2, i));
//     perturbed(2, i) += epsilon * std::sin(2 * FMCA_PI * points(0, i));
//   }
//   return perturbed;
// }

Matrix perturbPoints(const Matrix& points, Scalar epsilon) {
  Matrix perturbed = points;
  for (int i = 0; i < points.cols(); ++i) {
      perturbed(0, i) += epsilon * std::sin(2 * FMCA_PI * points(2,i)); 
      perturbed(1, i) = points(1,i);
      perturbed(2, i) = points(2,i); 
  }
  return perturbed;
}


///////////////////////////////////////////////////////// SURFACE
Matrix generateCubeSurfacePoints(int num_points) {
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
  std::uniform_int_distribution<int> face_dist(0, 5);

  Matrix points(DIM, num_points);
  for (int i = 0; i < num_points; ++i) {
    // Randomly assign points to one of the 6 faces of the cube
    int face = face_dist(rng);

    Scalar x = dist(rng), y = dist(rng);

    switch (face) {
      case 0: points.col(i) << 0.0, x, y; break;  // Face at x = 0
      case 1: points.col(i) << 1.0, x, y; break;  // Face at x = 1
      case 2: points.col(i) << x, 0.0, y; break;  // Face at y = 0
      case 3: points.col(i) << x, 1.0, y; break;  // Face at y = 1
      case 4: points.col(i) << x, y, 0.0; break;  // Face at z = 0
      case 5: points.col(i) << x, y, 1.0; break;  // Face at z = 1
    }
  }
  return points;
}

Matrix perturbCubeSurfacePointsTransversal(const Matrix& points, Scalar epsilon) {
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<Scalar> rand_dist(-1.0, 1.0);

  Matrix perturbed = points;
  for (int i = 0; i < points.cols(); ++i) {
      perturbed(0, i) += epsilon * std::sin(2 * FMCA_PI * points(2,i)); // x-direction
      perturbed(1, i) = points(1,i);
      perturbed(2, i) = points(2,i);
  }
  return perturbed;
}



//////////////////////////////////////////////////////////////////////////////////////////
int main() {
  Tictoc T;
  Scalar threshold_active_leaves = 1e-10;
  Scalar epsilon = 1e-1;
  int num_points = 100000;

  // Generate points and assign region values
  Matrix P = generateCubeVolumePoints(num_points);
  Vector f = assignRegionValues(P);
  // Apply smooth perturbation
  Matrix P_perturbed = perturbPoints(P, epsilon);
  /////////////////////////////////
  const Scalar eta = 1. / DIM;
  const Index dtilde = 1;
  const Scalar threshold_kernel = 0;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "threshold_kernel    " << threshold_kernel << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;

  const Moments mom(P_perturbed, mpole_deg);
  const SampletMoments samp_mom(P_perturbed, dtilde - 1);
  const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_perturbed);

  Vector f_ordered = hst.toClusterOrder(f);
  Vector tdata = hst.sampletTransform(f_ordered);
  ///////////////////////////////////////////////// Compute the decay of the

  Index max_level = computeMaxLevel(hst);
  std::cout << "Maximum level of the tree: " << max_level << std::endl;
  const Index nclusters = std::distance(hst.begin(), hst.end());
  std::cout << "Total number of clusters: " << nclusters << std::endl;

  // Coeff decay
  printMaxCoefficientsPerLevel(hst, tdata);
  // Singularity localization
  std::vector<const H2SampletTree<ClusterTree>*> adaptive_tree =
      computeAdaptiveTree(hst, tdata, threshold_active_leaves);
  std::vector<const H2SampletTree<ClusterTree>*> nodes;
  std::vector<FMCA::Matrix> bbvec_active;
  computeNodeAndBBActive(adaptive_tree, nodes, bbvec_active);

  IO::plotBoxes("Plots/Cube_EdgeDetection.vtk", bbvec_active);
  IO::plotPointsColor("Plots/Cube_EdgeDetectionPoints.vtk", P, f);
  IO::plotPointsColor("Plots/Cube_EdgeDetectionPointsPerturbed.vtk", P_perturbed, f);

  return 0;
}
