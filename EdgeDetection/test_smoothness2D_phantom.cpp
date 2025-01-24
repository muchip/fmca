#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
// ########################
#include "../FMCA/src/util/Plotter.h"
#include "InputOutput.h"
#include "Regression.h"
#include "SmoothnessDetection.h"

#define DIM 2

using ST = FMCA::SampletTree<FMCA::RandomProjectionTree>;
// using H2ST = FMCA::H2SampletTree<FMCA::ClusterTree>;

using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////
int main() {
  // Scalar threshold_active_leaves = 1e-10;
  /////////////////////////////////
  Matrix P;
  readTXT("data/coordinates_phantom.txt", P, DIM);
  std::cout << "Dimension P = " << P.rows() << " x " << P.cols() << std::endl;

  Vector f_phantom;
  readTXT("data/values_phantom.txt", f_phantom);
  std::cout << "Dimension f = " << f_phantom.rows() << " x " << f_phantom.cols() << std::endl;

  /////////////////////////////////
  const std::string outputFile_boxes =
      "/Users/saraavesani/Desktop/Archive/output_boxes_phantom.py";
    const std::string outputFile_boxes_total =
      "/Users/saraavesani/Desktop/Archive/output_boxes_total_phantom.py";
  const std::string outputFile_slopes_py =
      "/Users/saraavesani/Desktop/Archive/output_slopes_phantom.py";
  const std::string outputFile_points_slope =
      "/Users/saraavesani/Desktop/Archive/output_points_slope.txt";

  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar mpole_deg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;

  const Moments mom(P, mpole_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  const ST hst(samp_mom, 0, P);
  const RandomProjectionTree rp(P, 10);

  clusterTreeStatistics(rp, P);
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

  std::map<const ST*, std::vector<Scalar>> leafCoefficients;
  traverseAndStackCoefficients(hst, tdata, leafCoefficients);

  const std::string outputFile_slopes = "relativeSlopes2D.csv";
  auto slopes = computeRelativeSlopes2D<ST, Scalar>(leafCoefficients, dtilde,
                                                    1e-4, outputFile_slopes);

// Initialize color vector for each column in P
  Vector colr(P.cols());
  for (const auto& [leaf, slopeValue] : slopes) {
      for (int j = 0; j < leaf->block_size(); ++j) {
          colr(leaf->indices()[j]) = slopeValue; // Assign slopeValue as the color
      }
  }

  Matrix P3(3, P.cols());
  P3.topRows(2) = P;
  P3.row(2).setZero();
  FMCA::IO::plotPointsColor("Plots/Points_slope_phantom.vtk", P3, colr);
  //plotter.plotFunction2D("Plots/Points_slope_phantom.vtk", P, colr);


  // Vector PointsSlope = PointsWithSlopes(slopes, tdata); 
  // PointsSlope = hst.toNaturalOrder(PointsSlope); 
  // saveVectorToFile(PointsSlope, outputFile_points_slope);                                              

  // saveBoxesWithSlopesToFile(slopes, outputFile_boxes_total, outputFile_slopes_py);

/*
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
*/

  return 0;
}