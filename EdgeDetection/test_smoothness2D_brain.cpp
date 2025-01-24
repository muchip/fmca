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

#define DIM 2

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using H2ST = FMCA::H2SampletTree<FMCA::ClusterTree>;

using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////

Scalar f_elaborated(Scalar x, Scalar y) {
  if ((x < -0.4) & (y < -0.4)) {
    return 5;
  } else if ((x >= -0.4) & (x < -0.2) & (y >= -0.4) & (y < -0.2)) {
    return 0.1 * std::abs(-20 * x - 9) + 5;
  } else if ((x >= -0.2) & (x < 0.2) & (y >= -0.2) & (y < 0.2)) {
    return 6 + sin(10 * FMCA_PI * x) * cos(10 * FMCA_PI * y);
  } else if ((x >= 0.2) & (y >= 0.2)) {
    return 0.5 * sin(6 * FMCA_PI * x) * cos(6 * FMCA_PI * y);
  }
  return 0;
}


Vector eval_f_elaborated(Matrix Points) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = f_elaborated(Points(0, i), Points(1, i));
  }
  return f;
}

//////////////////////////////////////////////////////////////////////////////////////////

int main() {
  Scalar threshold_active_leaves = 1e-3;

  /////////////////////////////////
  Matrix P;
  readTXT("data/coordinates_brain1.txt", P, DIM);
  std::cout << "Dimension P = " << P.rows() << " x " << P.cols() << std::endl;

  Vector f_brain;
  readTXT("data/values_brain1.txt", f_brain);
  std::cout << "Dimension f = " << f_brain.rows() << " x " << f_brain.cols() << std::endl;

  /////////////////////////////////
  const std::string outputFile_boxes =
      "/Users/saraavesani/Desktop/Archive/output_boxes_brain.py";
  const std::string outputFile_step =
      "/Users/saraavesani/Desktop/Archive/output_step_brain.py";

  const std::string function_type = "brain";  // "f_elaborated"
  const Scalar eta = 1. / DIM;
  const Index dtilde = 4;
  const Scalar threshold_kernel = 0;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "threshold_kernel    " << threshold_kernel << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;

  const Moments mom(P, mpole_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  const ST hst(samp_mom, 0, P);
  // const H2ST hst(mom, samp_mom, 0, P);
  std::cout << "Tree done." << std::endl;

  Vector f_brain_ordered = hst.toClusterOrder(f_brain);
  Vector f_brain_Samplets = hst.sampletTransform(f_brain_ordered);
  ///////////////////////////////////////////////// Compute the decay of the
  /// coeffcients
  Vector tdata;
  tdata = f_brain_Samplets;

  Index max_level = computeMaxLevel(hst);
  std::cout << "Maximum level of the tree: " << max_level << std::endl;
  const Index nclusters = std::distance(hst.begin(), hst.end());
  std::cout << "Total number of clusters: " << nclusters << std::endl;

  // std::map<const ST*, std::vector<Scalar>> leafCoefficients;
  // traverseAndStackCoefficients(hst, tdata, leafCoefficients);

  // const std::string outputFile_slopes = "relativeSlopes2D.csv";
  // auto slopes = computeRelativeSlopes2D<ST, Scalar>(leafCoefficients, dtilde,
  //                                                   1e-4, outputFile_slopes);

  // saveBoxesWithSlopesToFile(slopes, outputFile_step);

  // Coeff decay
  printMaxCoefficientsPerLevel(hst, tdata);

  // detect singularities
  // Singularity localization
  std::vector<const ST*> adaptive_tree =
      computeAdaptiveTree(hst, tdata, threshold_active_leaves);
  std::vector<const ST*> nodes;
  std::vector<FMCA::Matrix> bbvec_active;
  computeNodeAndBBActive(adaptive_tree, nodes, bbvec_active);
  saveBoxesToFile(bbvec_active, outputFile_boxes);

  return 0;
}