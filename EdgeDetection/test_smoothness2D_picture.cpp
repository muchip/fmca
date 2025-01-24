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
int main() {
  Scalar threshold_active_leaves = 1e-3;

  /////////////////////////////////
  Matrix P;
  readTXT("data/coordinates_picture.txt", P, DIM);
  std::cout << "Dimension P = " << P.rows() << " x " << P.cols() << std::endl;

  Vector f_phantom;
  readTXT("data/values_picture.txt", f_phantom);
  std::cout << "Dimension f = " << f_phantom.rows() << " x " << f_phantom.cols() << std::endl;

  /////////////////////////////////
  const std::string outputFile_boxes =
      "/Users/saraavesani/Desktop/Archive/output_boxes_picture.py";

  const Scalar eta = 1. / DIM;
  const Index dtilde = 10;
  const Scalar mpole_deg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;

  const Moments mom(P, mpole_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  const ST hst(samp_mom, 0, P);
  // const H2ST hst(mom, samp_mom, 0, P);
  std::cout << "Tree done." << std::endl;

  Vector f_phantom_ordered = hst.toClusterOrder(f_phantom);
  Vector f_phantom_Samplets = hst.sampletTransform(f_phantom_ordered);

  ///////////////////////////////////////////////// Compute the decay of the
  /// coeffcients
  Vector tdata;
  tdata = f_phantom_Samplets;

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