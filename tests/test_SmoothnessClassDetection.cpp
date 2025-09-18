#include "../FMCA/SmoothnessClassDetection"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MapCoeffDiam =
    std::map<const ST*,
             std::pair<std::vector<FMCA::Scalar>, std::vector<FMCA::Scalar>>>;
using namespace FMCA;

Scalar combined_function(Scalar x, Scalar y) {
    Scalar sum_xy = x + y;
    // C1 not C2
    if (sum_xy < 1.0) {
        return (sum_xy - 0.5) * std::abs(sum_xy - 0.5);
    }
    // Jump
    else if (sum_xy < 1.5) {
        return 0.0;
    }
    // Corner
    else {
        return std::abs(sum_xy - 1.5);
    }
}

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
    pt.setOnes();
    pt /= std::sqrt(d);
    for (FMCA::Index i = 0; i < Nd; ++i) {
      data(i) = combined_function(P(0, i), P(1, i));
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

  ///////////////////////////////////////////////// 
  SampletCoefficientsAnalyzer<ST> coeff_analyzer;
  coeff_analyzer.init(st, scoeffs);
  Index max_level = coeff_analyzer.computeMaxLevel(st);
  std::cout << "Max level = " << max_level << std::endl;

  auto squared_coeff_per_level =
      coeff_analyzer.getSumSquaredPerLevel(st, scoeffs);
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
  T.tic();
  MapCoeffDiam leafData;
  coeff_analyzer.traverseAndStackCoefficientsAndDiametersL2Norm(st, scoeffs,
                                                                leafData);

  SlopeFitter<ST> fitter;
  fitter.init(leafData, dtilde, 1e-12);
  auto results = fitter.fitSlope();
  std::cout << "--------------------------------------------------------"
            << std::endl;
  T.toc("fitting: ");

  //////////////////////////////////////////////////////////////
  Vector colr(P.cols());
  for (const auto& [leaf, res] : results) {
    for (int j = 0; j < leaf->block_size(); ++j) {
      Scalar slope = res.get_slope();
      //// round the slope for seak of better visualization
      // Scalar dtilde_binned = std::abs(std::floor((slope + 0.25) / 0.5) * 0.5);
      colr(leaf->indices()[j]) = slope;
    }
  }

  Matrix P3(3, P.cols());
  P3.topRows(2) = P;
  P3.row(2).setZero();
  FMCA::IO::plotPointsColor("Slope2D.vtk", P3, colr);
  FMCA::IO::plotPointsColor("Function2D.vtk", P3, data);

  Matrix P3_values(3, P.cols());
  P3_values.topRows(2) = P;
  P3_values.row(2) = data;
  FMCA::IO::plotPointsColor("Function2D_3D.vtk", P3_values, data);

  return 0;
}