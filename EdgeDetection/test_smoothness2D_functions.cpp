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
#include "FitDtilde.h"

#define DIM 2

using ST = FMCA::SampletTree<FMCA::RandomProjectionTree>;
// using H2ST = FMCA::H2SampletTree<FMCA::ClusterTree>;

using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////
int main() {
  // Scalar threshold_active_leaves = 1e-10;
  /////////////////////////////////
  Matrix P;
  readTXT("data/coordinates_picture.txt", P, DIM);
  std::cout << "Dimension P = " << P.rows() << " x " << P.cols() << std::endl;

  Vector f;
  readTXT("data/values_picture.txt", f);
  std::cout << "Dimension f = " << f.rows() << " x " << f.cols() << std::endl;

  /////////////////////////////////

  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar mpole_deg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;

  const Moments mom(P, mpole_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  const ST hst(samp_mom, 0, P);

  clusterTreeStatistics(hst, P);

  // const H2ST hst(mom, samp_mom, 0, P);
  std::cout << "Tree done." << std::endl;

  Vector f_ordered = hst.toClusterOrder(f);
  Vector f_Samplets = hst.sampletTransform(f_ordered);

  ///////////////////////////////////////////////// Compute the decay of the
  /// coeffcients
  Vector tdata;
  tdata = f_Samplets;

  Index max_level = computeMaxLevel(hst);
  std::cout << "Maximum level of the tree: " << max_level << std::endl;
  const Index nclusters = std::distance(hst.begin(), hst.end());
  std::cout << "Total number of clusters: " << nclusters << std::endl;

  std::map<const ST*, std::pair<std::vector<Scalar>, std::vector<Scalar>>> leafData;
  traverseAndStackCoefficientsAndDiameters(hst, tdata, leafData);

  auto results = FitDtilde(leafData, dtilde, 1e-8);


    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

/*
    int count = 0;
    for (const auto& [leaf, dataPair] : leafData) {
        // if (count >= 20) break;
        auto coefficients = dataPair.first;  // First vector = coefficients
        const auto& diameters = dataPair.second;     // Second vector = diameters
        if (leaf->bb()(0, 0) > 8.91e-01 && leaf->bb()(0, 0) < 8.93e-01 && leaf->bb()(0, 1) > 9.03e-01
        && leaf->bb()(0, 1) < 9.05-01) {
            
        std::cout << "Leaf bb: " << leaf->bb() << "\n";
        std::cout << "-----------------------------------------" << std::endl;
        size_t n = coefficients.size();

        Scalar accum = 0.;
        for (int i = 0; i < n; ++i) {
            accum += coefficients[i] * coefficients[i];
        }
        Scalar norm = sqrt(accum);
        if (norm != 0) {
          for (auto& coeff : coefficients) {
              coeff /= norm;
          }
        }

        std::cout << "  Coefficients: ";
        int zero_count = 0;
        for (const auto& coeff : coefficients) {
            std::cout <<  coeff << " ";
            if (coeff == 0) {
                zero_count++;
            }
        }
        // std::cout << "\n";
        // std::cout << "number of 0 coeffs = " << zero_count << "\n";
        std::cout << "\n";
        std::cout << "-----------------------------------------" << std::endl;
        
        std::cout << "\n  Diameters: ";
        for (const auto& diam : diameters) {
            std::cout << diam << " ";
        }
        std::cout << "\n";
        std::cout << "-------------------------" << std::endl;
             }
        count++;
    }

    int count2 = 0;
    for (const auto& [leaf, slope] : slopes) {
        if (slope > 2) { // break;
            std::cout << "Leaf bb: " << leaf->bb() << "\n";
            std::cout << "-------------------------" << std::endl;

            std::cout << "Slope: " << slope << "\n";
            std::cout << "-------------------------" << std::endl;
            std::cout << "-------------------------" << std::endl;
        }
        count2++;
    }
*/

// Initialize color vector for each column in P
  Vector colr(P.cols());
  for (const auto& [leaf, res] : results) {
      for (int j = 0; j < leaf->block_size(); ++j) {
          colr(leaf->indices()[j]) = res.get_dtilde(); // Assign slopeValue as the color
      }
  }

  FMCA::Matrix P3D(3, P.cols());
  P3D.setZero();
  P3D.topRows(2) = P;
  FMCA::IO::plotPointsColor("Plots/Points_slope_2d.vtk", P3D, colr);

  return 0;
}