#include "SampletCoefficientsAnalyzer.h"
#include "SlopeFitter.h"
#include "read_files_txt.h"
#include "../FMCA/src/util/IO.h"

#define DIM 2

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using H2ST = FMCA::H2SampletTree<FMCA::ClusterTree>;

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::MinNystromSampletMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using MapCoeffDiam =
    std::map<const ST*,
             std::pair<std::vector<FMCA::Scalar>, std::vector<FMCA::Scalar>>>;

using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////
int main() {
  // Scalar threshold_active_leaves = 1e-10;
  /////////////////////////////////
  Matrix P;
  readTXT("data/coordinates_grid.txt", P, DIM);
  std::cout << "Dimension P = " << P.rows() << " x " << P.cols() << std::endl;

  Vector f;
  readTXT("data/values_grid.txt", f);
  f = f / f.maxCoeff();
  std::cout << "Dimension f = " << f.rows() << " x " << f.cols() << std::endl;

  /////////////////////////////////

  const Scalar eta = 1. / DIM;
  const Index dtilde = 3;
  const Scalar mpole_deg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;

  const Moments mom(P, mpole_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  const ST hst(samp_mom, 0, P);
  // const H2ST hst(mom, samp_mom, 0, P);

  // clusterTreeStatistics(hst, P);

  std::cout << "Tree done." << std::endl;

  std::cout << "Norm of f = " << f.norm() << std::endl;
  Vector f_ordered = hst.toClusterOrder(f);
  Vector f_Samplets = hst.sampletTransform(f_ordered);
  std::cout << "Norm of f Samplets = " << f_Samplets.norm() << std::endl;

  ///////////////////////////////////////////////// Compute the decay of the
  ///coeffcients
  Vector tdata;
  tdata = f_Samplets;

  SampletCoefficientsAnalyzer<ST> coeff_analyzer;
  coeff_analyzer.init(hst, tdata);

  Index max_level = coeff_analyzer.computeMaxLevel(hst);
  std::cout << "Max level = " << max_level << std::endl;

  auto max_coeff_per_level =
      coeff_analyzer.getMaxCoefficientsPerLevel(hst, tdata);
  for (auto [lvl, coeff] : max_coeff_per_level) {
    std::cout << "Level " << lvl << ": max coeff = " << coeff << "\n";
  }

  MapCoeffDiam leafData;
  coeff_analyzer.traverseAndStackCoefficientsAndDiameters(hst, tdata, leafData);

  SlopeFitter<ST> fitter;
  fitter.init(leafData, dtilde, 1e-8);
  auto results = fitter.fitSlopeRegression(true);
  std::cout << "--------------------------------------------------------"
            << std::endl;

  ///////////////////////////////////////////////////////////////////////////////////
  /*
  int count = 0;
  for (const auto& [leaf, res] : results) {
    if (res.get_slope() > dtilde) {
      std::cout << "Leaf bb: " << leaf->bb() << "\n";
      std::cout << "-------------------------" << std::endl;

      std::cout << "Slope: " << res.get_slope() << "\n";
      std::cout << "-------------------------" << std::endl;

      // Check if leaf exists in leafData
      auto it = leafData.find(leaf);
      if (it != leafData.end()) {
        const auto& coefficients =
            it->second.first;  // First vector = coefficients
        const auto& diameters = it->second.second;  // Second vector =diameters

        std::cout << "  Coefficients: ";
        for (const auto& coeff : coefficients) {
          std::cout << coeff << " ";
        }
        std::cout << "\n";
        std::cout << "-----------------------------------------" << std::endl;

        std::cout << "\n  Diameters: ";
        for (const auto& diam : diameters) {
          std::cout << diam << " ";
        }
        std::cout << "\n";
        std::cout << "-------------------------" << std::endl;
      }
      std::cout << "=========================" << std::endl;
    }
    count++;
  }

 */

  for (const auto& [leaf, dataPair] : leafData) {
    auto coefficients = dataPair.first;
    auto diameters = dataPair.second;
    for (int j = 0; j < leaf->block_size(); ++j) {
      if (leaf->indices()[j] == 486702) {
        std::cout << "leaf bb " << leaf->bb() << std::endl;
        // coefficients is likely a vector or array, need to print specific
        // elements
        std::cout << "coeffs ";
        for (const auto& coeff : coefficients) {
          std::cout << coeff << " ";
        }
        std::cout << std::endl;
        std::cout << "diams ";
        for (const auto& diam : diameters) {
          std::cout << diam << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  for (const auto& [leaf, res] : results) {
    for (int j = 0; j < leaf->block_size(); ++j) {
      if (leaf->indices()[j] == 486702) {
        std::cout << res.get_slope() << std::endl;
      }
    }
  }


///////////////////////////////////////////////////////////////////////////////////

  // Initialize color vector for each column in P
  Vector colr(P.cols());
  for (const auto& [leaf, res] : results) {
    for (int j = 0; j < leaf->block_size(); ++j) {
      Scalar slope = res.get_slope();
      // save the slope just if slope < 2, otherwirse save dtilde
      Scalar slope_filtered = (slope <= 1.5) ? slope : dtilde;
      Scalar dtilde_binned = std::abs(std::floor((slope + 0.25) / 0.5) * 0.5);
      colr(leaf->indices()[j]) = slope;
    }
  }

  Matrix P3(3, P.cols());
  P3.topRows(2) = P;
  P3.row(2).setZero();
  // P3.row(2) = f;
  FMCA::IO::plotPointsColor("Slope_picture.vtk", P3, colr);
  FMCA::IO::plotPointsColor("Function_picture.vtk", P3, f);

  // Print the min value of colr
  std::cout << "Min value of colr: " << colr.minCoeff() << std::endl;

  return 0;
}