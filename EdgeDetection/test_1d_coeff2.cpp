#include <iostream>

#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"
#include "SampletCoefficientsAnalyzer.h"
#include "SlopeFitter.h"
#include "read_files_txt.h"

#define DIM 1

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////
int main() {
  constexpr FMCA::Index l = 20;
  constexpr FMCA::Index d = 1;
  constexpr FMCA::Index N = 1 << l;
  constexpr FMCA::Scalar h = 1. / N;
  FMCA::Index Nd = N;  // In 1D: Nd = N^1 = N
  constexpr FMCA::Index dtilde = 4;
  FMCA::Tictoc T;
  FMCA::Matrix P(d, Nd);
  FMCA::Vector data(Nd);
  T.tic();
  {
    // generate a uniform grid in 1D
    for (FMCA::Index i = 0; i < N; ++i) {
      P(0, i) = h * (i + 0.5);  // P is now 1xN matrix
    }
  }
//   {
//     // create 1D corner function |x - 0.5|
//     for (FMCA::Index i = 0; i < Nd; ++i) {
//       Scalar x = P(0, i);
//       data(i) = std::abs(x - 0.5);  
//     }
//   }
  {
    FMCA::Vector pt(d);
    // create a non axis aligned jump
    pt.setOnes();
    pt /= std::sqrt(d);
    for (FMCA::Index i = 0; i < Nd; ++i) {
      data(i) =  10*(P.col(i).dot(pt) > 0.3 * sqrt(d));
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

  ///////////////////////////////////////////////// Compute the global decay of the
  /// coeffcients

  SampletCoefficientsAnalyzer<ST> coeff_analyzer;
  coeff_analyzer.init(st, scoeffs);

//   Index max_level = coeff_analyzer.computeMaxLevel(st);
//   std::cout << "Max level = " << max_level << std::endl;

//   auto max_coeff_per_level =
//       coeff_analyzer.getMaxCoefficientsPerLevel(st, scoeffs);
//   for (FMCA::Index i = 1; i < max_coeff_per_level.size(); ++i)
//     std::cout << "c=" << std::sqrt(max_coeff_per_level[i])
//               << std::endl;

//   for (FMCA::Index i = 1; i < max_coeff_per_level.size(); ++i)
//     std::cout << "c=" << max_coeff_per_level[i] / std::sqrt(Nd) << " alpha="
//               << std::log(max_coeff_per_level[i - 1] / max_coeff_per_level[i]) /
//                      (std::log(2))
//               << std::endl;

//   std::cout << "--------------------------------------------------------"
//             << std::endl;

  auto squared_coeff_per_level =
      coeff_analyzer.getSumSquaredPerLevel(st, scoeffs);
  std::cout << "--------------------------------------------------------"
            << std::endl;

  for (FMCA::Index i = 1; i < squared_coeff_per_level.size(); ++i)
    std::cout << "c=" << std::sqrt(squared_coeff_per_level[i])
              << std::endl;

  std::cout << "--------------------------------------------------------"
            << std::endl;
  for (FMCA::Index i = 1; i < squared_coeff_per_level.size(); ++i)
    std::cout << "c=" << std::sqrt(squared_coeff_per_level[i]) / std::sqrt(Nd)
              << " alpha="
              << std::log(std::sqrt(squared_coeff_per_level[i - 1]) /
                          std::sqrt(squared_coeff_per_level[i])) /
                     (std::log(2))
              << std::endl;

  return 0;
}