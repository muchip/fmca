#include <iostream>

#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"
#include "SampletCoefficientsAnalyzer.h"
#include "SlopeFitter.h"
#include "read_files_txt.h"

#define DIM 3

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MapCoeffDiam =
    std::map<const ST*,
             std::pair<std::vector<FMCA::Scalar>, std::vector<FMCA::Scalar>>>;
using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////
int main() {
  Scalar threshold_active_leaves = 1e-6;

  /////////////////////////////////
  Matrix P;
  readTXT("local_tests/data/coordinates_bunny.txt", P, DIM);
  std::cout << "Dimension P = " << P.rows() << " x " << P.cols() << std::endl;

  Matrix P_bunny;
  readTXT("local_tests/data/coordinates_bunny_inside.txt", P_bunny, DIM);
  std::cout << "Dimension P = " << P_bunny.rows() << " x " << P_bunny.cols() << std::endl;

  Vector data = Vector::Zero(P.cols());

  // coordinated of top left corner
  Scalar x_min = P.row(0).minCoeff();
  Scalar y_min = P.row(1).minCoeff();
  Scalar z_min = P.row(2).minCoeff();

  Vector gaussian;
  readTXT("local_tests/data/values_gaussian_bunny.txt", gaussian);

  // data = signed_dist if gausian == 0, otherwise data = gaussian
  for (Index i = 0; i < P.cols(); ++i) {
    if (gaussian(i) == 0) {
      data(i) = std::sqrt((P(0,i) - x_min) * (P(0,i) - x_min) + (P(1,i) - y_min) * (P(1,i) - y_min) + (P(2,i) - z_min) * (P(2,i) - z_min));
    } else {
      data(i) = gaussian(i);
    }
  }

  std::cout << "Dimension f = " << data.rows() << " x " << data.cols() << std::endl;

/////////////////////////////////
  constexpr FMCA::Index dtilde = 3;
  FMCA::Tictoc T;
  T.tic();
  const SampletMoments samp_mom(P, dtilde - 1);
  ST st(samp_mom, 0, P);
  T.toc("samplet tree generation:");
  const FMCA::Vector scoeffs = st.sampletTransform(st.toClusterOrder(data));

  ///////////////////////////////////////////////// Compute the global decay of
  /// the
  /// coeffcients
  FMCA::Tictoc T1;
  T1.tic();
  SampletCoefficientsAnalyzer<ST> coeff_analyzer;
  coeff_analyzer.init(st, scoeffs);

  Index max_level = coeff_analyzer.computeMaxLevel(st);
  std::cout << "Max level = " << max_level << std::endl;

  auto max_coeff_per_level =
      coeff_analyzer.getMaxCoefficientsPerLevel(st, scoeffs);

  for (FMCA::Index i = 1; i < max_coeff_per_level.size(); ++i)
    std::cout << "c=" << max_coeff_per_level[i] / std::sqrt(data.rows()) << " alpha="
              << std::log(max_coeff_per_level[i - 1] / max_coeff_per_level[i]) /
                     (std::log(2))
              << std::endl;

  std::cout << "--------------------------------------------------------"
            << std::endl;

  auto squared_coeff_per_level =
      coeff_analyzer.getSumSquaredPerLevel(st, scoeffs);
  std::cout << "--------------------------------------------------------"
            << std::endl;

  for (FMCA::Index i = 1; i < squared_coeff_per_level.size(); ++i)
    std::cout << "c=" << std::sqrt(squared_coeff_per_level[i]) << std::endl;

  std::cout << "--------------------------------------------------------"
            << std::endl;
  for (FMCA::Index i = 1; i < squared_coeff_per_level.size(); ++i)
    std::cout << "c=" << std::sqrt(squared_coeff_per_level[i]) / std::sqrt(data.rows())
              << " alpha="
              << std::log(std::sqrt(squared_coeff_per_level[i - 1]) /
                          std::sqrt(squared_coeff_per_level[i])) /
                     (std::log(2))
              << std::endl;

  //////////////////////////////////////////////////////////////

  MapCoeffDiam leafData;
  coeff_analyzer.traverseAndStackCoefficientsAndDiametersL2Norm(st, scoeffs,
                                                                leafData);

  SlopeFitter<ST> fitter;
  fitter.init(leafData, dtilde, 1e-12);
  auto results = fitter.fitSlope();
  std::cout << "--------------------------------------------------------"
            << std::endl;

  T.toc("samplet tree generation:");

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
  // Print the min value of colr
  std::cout << "Min value of colr: " << colr.minCoeff() << std::endl;

  FMCA::IO::plotPoints("PointsBunny.vtk", P_bunny);
  FMCA::IO::plotPointsColor("SlopeBunny.vtk", P, colr);
  FMCA::IO::plotPointsColor("FunctionBunny.vtk", P, data);

  return 0;
}