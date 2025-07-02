#include <iostream>

#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"
#include "SampletCoefficientsAnalyzer.h"
#include "SlopeFitter.h"
#include "read_files_txt.h"

#define DIM 2

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MapCoeffDiam =
    std::map<const ST*,
             std::pair<std::vector<FMCA::Scalar>, std::vector<FMCA::Scalar>>>;
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
  const Index dtilde = 3;
  const Scalar mpole_deg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;
  
  Tictoc t;
  t.tic();
  const SampletMoments samp_mom(P, dtilde - 1);
  const ST st(samp_mom, 0, P);
  t.toc("Tree creation: ");

  const RandomProjectionTree rp(P, 10);

  Vector f_phantom_ordered = st.toClusterOrder(f_phantom);
  Vector f_phantom_Samplets = st.sampletTransform(f_phantom_ordered);

  ///////////////////////////////////////////////// Compute the decay of the
  /// coeffcients
  Vector scoeffs;
  scoeffs = f_phantom_Samplets;

   ///////////////////////////////////////////////// Compute the global decay of
  /// the
  /// coeffcients
  SampletCoefficientsAnalyzer<ST> coeff_analyzer;
  coeff_analyzer.init(st, scoeffs);

  Index max_level = coeff_analyzer.computeMaxLevel(st);
  std::cout << "Max level = " << max_level << std::endl;

  auto squared_coeff_per_level =
      coeff_analyzer.getSumSquaredPerLevel(st, scoeffs);
  std::cout << "--------------------------------------------------------"
            << std::endl;

  for (FMCA::Index i = 0; i < squared_coeff_per_level.size(); ++i)
    std::cout << std::sqrt(squared_coeff_per_level[i]) << "," << std::endl;

  std::cout << "--------------------------------------------------------"
            << std::endl;
  for (FMCA::Index i = 1; i < squared_coeff_per_level.size(); ++i)
    std::cout << "c=" << std::sqrt(squared_coeff_per_level[i]) / std::sqrt(scoeffs.rows())
              << " alpha="
              << std::log(std::sqrt(squared_coeff_per_level[i - 1]) /
                          std::sqrt(squared_coeff_per_level[i])) /
                     (std::log(2))
              << std::endl;
  //////////////////////////////////////////////////////////////
  Tictoc t2;
  t2.tic();
  MapCoeffDiam leafData;
  coeff_analyzer.traverseAndStackCoefficientsAndDiametersL2Norm(st, scoeffs,
                                                                leafData);

  SlopeFitter<ST> fitter;
  fitter.init(leafData, dtilde, 1e-12);
  auto results = fitter.fitSlope();
  t2.toc("Result traverse and fit: ");
  std::cout << "--------------------------------------------------------"
            << std::endl;

  // Initialize color vector for each column in P
  Vector colr(P.cols());
  for (const auto& [leaf, res] : results) {
    for (int j = 0; j < leaf->block_size(); ++j) {
      Scalar slope = res.get_slope();
      // save the slope just if slope < 2, otherwirse save dtilde
      Scalar slope_filtered = (slope < 1.8) ? slope : dtilde;
      Scalar dtilde_binned = std::abs(std::floor((slope + 0.3) / 0.5) * 0.5);
      colr(leaf->indices()[j]) = slope_filtered;
    }
  }
  // Print the min value of colr
  std::cout << "Min value of colr: " << colr.minCoeff() << std::endl;

  Matrix P3(3, P.cols());
  P3.topRows(2) = P;
  P3.row(2).setZero();
  // P3.row(2) = f;
  FMCA::IO::plotPointsColor("Slope_picture.vtk", P3, colr);
  FMCA::IO::plotPointsColor("Function_picture.vtk", P3, f_phantom);

  return 0;
}