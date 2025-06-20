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
  constexpr FMCA::Index l = 11;
  constexpr FMCA::Index d = 2;
  constexpr FMCA::Index N = 1 << l;
  constexpr FMCA::Scalar h = 1. / N;
  FMCA::Index Nd = std::pow(N, d);
  constexpr FMCA::Index dtilde = 3;
  FMCA::Tictoc T;
  FMCA::Matrix P(d, Nd);
  FMCA::Vector data(Nd);
  T.tic();
  {
    // generate a uniform grid
    FMCA::Vector pt(d);
    pt.setZero();
    FMCA::Index p = 0;
    FMCA::Index i = 0;
    while (pt(d - 1) < N) {
      if (pt(p) >= N) {
        pt(p) = 0;
        ++p;
      } else {
        P.col(i++) = h * (pt.array() + 0.5).matrix();
        p = 0;
      }
      pt(p) += 1;
    }
  }
  {
    FMCA::Vector pt(d);
    // create a non axis aligned jump
    pt.setOnes();
    pt /= std::sqrt(d);
    for (FMCA::Index i = 0; i < Nd; ++i) {
      Scalar signed_dist = P.col(i).dot(pt) - 0.5 * sqrt(d);
      data(i) = std::abs(signed_dist);
      //data(i) = P.col(i).dot(pt) > 0.5 * sqrt(d);
    }
  }
  T.toc("data generation: ");
  std::cout << "Nd=" << Nd << std::endl;
  /////////////////////////////////
  T.tic();
  const SampletMoments samp_mom(P, dtilde - 1);
  ST st(samp_mom, 0, P);
  T.toc("samplet tree generation:");
  const FMCA::Vector scoeffs = st.sampletTransform(st.toClusterOrder(data));

  ///////////////////////////////////////////////// Compute the global decay of
  /// the
  /// coeffcients

  SampletCoefficientsAnalyzer<ST> coeff_analyzer;
  coeff_analyzer.init(st, scoeffs);

  Index max_level = coeff_analyzer.computeMaxLevel(st);
  std::cout << "Max level = " << max_level << std::endl;

  // auto max_coeff_per_level =
  //     coeff_analyzer.getMaxCoefficientsPerLevel(st, scoeffs);

  // for (FMCA::Index i = 1; i < max_coeff_per_level.size(); ++i)
  //   std::cout << "c=" << max_coeff_per_level[i] / std::sqrt(Nd) << " alpha="
  //             << std::log(max_coeff_per_level[i - 1] /
  //             max_coeff_per_level[i]) /
  //                    (std::log(2))
  //             << std::endl;

  // std::cout << "--------------------------------------------------------"
  //           << std::endl;

  auto squared_coeff_per_level =
      coeff_analyzer.getSumSquaredPerLevel(st, scoeffs);
  std::cout << "--------------------------------------------------------"
            << std::endl;

  for (FMCA::Index i = 0; i < squared_coeff_per_level.size(); ++i)
    std::cout << "c=" << std::sqrt(squared_coeff_per_level[i]) << std::endl;

  std::cout << "--------------------------------------------------------"
            << std::endl;
  for (FMCA::Index i = 1; i < squared_coeff_per_level.size(); ++i)
    std::cout << "c=" << std::sqrt(squared_coeff_per_level[i]) / std::sqrt(Nd)
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

  // for (auto &node : st){
  //   std::cout << "Level: " << node.level() << std::endl;
  //   std::cout << "Size: " << node.block_size() << std::endl;

  // }

  for (const auto& [leaf, dataPair] : leafData) {
    auto coefficients = dataPair.first;
    auto diameters = dataPair.second;
    for (int j = 0; j < leaf->block_size(); ++j) {
      if (leaf->indices()[j] == 461373) {
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

  Matrix P3(3, P.cols());
  P3.topRows(2) = P;
  P3.row(2).setZero();
  // P3.row(2) = f;
  FMCA::IO::plotPointsColor("Slope2D.vtk", P3, colr);
  FMCA::IO::plotPointsColor("Function2D.vtk", P3, data);

  Matrix P3_values(3, P.cols());
  P3_values.topRows(2) = P;
  P3_values.row(2) = data;
  FMCA::IO::plotPointsColor("Function2D_3D.vtk", P3_values, data);

  return 0;
}