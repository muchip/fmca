#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
// ########################
#include "InputOutput.h"
#include "Regression.h"
#include "SmoothnessDetection.h"

#define DIM 3

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using H2ST = FMCA::H2SampletTree<FMCA::ClusterTree>;

using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////
int main() {
  Scalar threshold_active_leaves = 1e-6;

  /////////////////////////////////
  Matrix P;
  readTXT("data/coordinates_bunny.txt", P, DIM);
  std::cout << "Dimension P = " << P.rows() << " x " << P.cols() << std::endl;

  Matrix P_bunny;
  readTXT("data/coordinates_bunny_inside.txt", P_bunny, DIM);
  std::cout << "Dimension P = " << P_bunny.rows() << " x " << P_bunny.cols() << std::endl;

  Vector f;
  readTXT("data/values_2functions_bunny.txt", f);
  std::cout << "Dimension f = " << f.rows() << " x " << f.cols() << std::endl;
//   for (int i = 0; i < f.rows(); ++i){
//     f(i) = std::abs(f(i));
//   }

  /////////////////////////////////
  const Scalar eta = 1. / DIM;
  const Index dtilde = 2;
  const Scalar threshold_kernel = 0;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "threshold_kernel    " << threshold_kernel << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;

  const Moments mom(P, mpole_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  const H2ST hst(mom, samp_mom, 0, P);

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
  std::vector<const H2ST*> adaptive_tree =
      computeAdaptiveTree(hst, tdata, threshold_active_leaves);
  std::vector<const H2ST*> nodes;
  std::vector<Matrix> bbvec_active;
  computeNodeAndBBActive(adaptive_tree, nodes, bbvec_active);

  IO::plotBoxes("Plots/Bunny_EdgeDetection_2functions.vtk", bbvec_active);
  IO::plotPointsColor("Plots/Bunny_EdgeDetectionPoints_2functions.vtk", P, f);
  IO::plotPoints("Plots/Bunny_EdgeDetectionPointsInside_2functions.vtk", P_bunny);

  return 0;
}
