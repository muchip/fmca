#include "SampletCoefficientsAnalyzer.h"
#include "SlopeFitter.h"
#include "read_files_txt.h"
#include "../FMCA/src/util/IO.h"

#define DIM 3

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using H2ST = FMCA::H2SampletTree<FMCA::ClusterTree>;

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::MinNystromSampletMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using MapCoeffDiam =
    std::map<const ST*, std::pair<std::vector<FMCA::Scalar>, std::vector<FMCA::Scalar>>>;

using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////
int main() {
  // Scalar threshold_active_leaves = 1e-10;
  /////////////////////////////////
  Matrix P;
  readTXT("local_tests/data/coordinates_grid_3d.txt", P, DIM);
  // readTXT("local_tests/data/coordinates_bunny.txt", P, DIM);
  std::cout << "Dimension P = " << P.rows() << " x " << P.cols() << std::endl;

  Matrix P_bunny;
  readTXT("local_tests/data/coordinates_bunny_inside.txt", P_bunny, DIM);
  std::cout << "Dimension Inside Points = " << P_bunny.rows() << " x "
            << P_bunny.cols() << std::endl;
  // FMCA::IO::plotPoints("Plots/Points_bunny_3d.vtk", P_bunny);

  Vector f;
  readTXT("local_tests/data/values_grid_3d.txt", f);
  // readTXT("local_tests/data/values_2functions_bunny.txt", f);
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

  Vector f_ordered = hst.toClusterOrder(f);
  Vector f_Samplets = hst.sampletTransform(f_ordered);

  ///////////////////////////////////////////////// Compute the decay of the
  /// coeffcients
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
  auto results = fitter.fitSlope();
  std::cout << "--------------------------------------------------------"
            << std::endl;

  ///////////////////////////////////////////////////////////////////////////////////
  /*
  int count = 0;
  for (const auto& [leaf, res] : results) {
    if (res.get_slope() < 0.9) {
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
    

  for (const auto& [leaf, dataPair] : leafData) {
    auto coefficients = dataPair.first;
    auto diameters = dataPair.second;
    for (int j = 0; j < leaf->block_size(); ++j) {
      if (leaf->indices()[j] == 3.98271e+06) {
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
      if (leaf->indices()[j] == 3.98271e+06) {
        std::cout << res.get_slope() << std::endl;
      }
    }
  }
  */
  ///////////////////////////////////////////////////////////////////////////////////


  // Initialize color vector for each column in P
  Vector colr(P.cols());
  for (const auto& [leaf, res] : results) {
    for (int j = 0; j < leaf->block_size(); ++j) {
      Scalar slope = res.get_slope();
      Scalar dtilde_binned = std::abs(std::floor((slope + 0.25) / 0.5) * 0.5);
      colr(leaf->indices()[j]) = slope;
    }
  }

  FMCA::IO::plotPointsColor("Slopes_bunny.vtk", P, colr);
  FMCA::IO::plotPointsColor("Points_bunny.vtk", P, f);
  std::cout << "Min value of colr: " << colr.minCoeff() << std::endl;

  return 0;
}