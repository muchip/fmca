#include <iostream>
#include <random>

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
  constexpr FMCA::Index l = 12;
  constexpr FMCA::Index d = 3;
  constexpr FMCA::Index N = 1 << l;
  FMCA::Index Nd = N * N; // For sphere, we use N^2 points instead of N^3
  constexpr FMCA::Index dtilde = 3;
  FMCA::Tictoc T;
  FMCA::Matrix P(d, Nd);
  FMCA::Vector data(Nd);
  
  // Random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<FMCA::Scalar> uniform(0.0, 1.0);
  
  T.tic();
  {
    // Generate random points on unit sphere using Marsaglia method
    FMCA::Index i = 0;
    while (i < Nd) {
      // Generate two uniform random numbers
      FMCA::Scalar u1 = 2.0 * uniform(gen) - 1.0; // [-1, 1]
      FMCA::Scalar u2 = 2.0 * uniform(gen) - 1.0; // [-1, 1]
      
      FMCA::Scalar s = u1 * u1 + u2 * u2;
      
      // Accept only if inside unit circle
      if (s < 1.0 && s > 0.0) {
        FMCA::Scalar factor = 2.0 * std::sqrt(1.0 - s);
        P(0, i) = u1 * factor;           // x
        P(1, i) = u2 * factor;           // y  
        P(2, i) = 1.0 - 2.0 * s;         // z
        i++;
      }
    }
  }
  
  {
    // Create a non-trivial HARD jump function on the sphere
    // Using a combination of spherical harmonics-like patterns
    for (FMCA::Index i = 0; i < Nd; ++i) {
      FMCA::Scalar x = P(0, i);
      FMCA::Scalar y = P(1, i);
      FMCA::Scalar z = P(2, i);
      
      // Convert to spherical coordinates
      FMCA::Scalar theta = std::atan2(y, x);        // azimuthal angle
      FMCA::Scalar phi = std::acos(z);              // polar angle
      
      // Create a complex jump pattern using multiple harmonics
      // This creates a flower-like pattern with 6 petals and additional complexity
      FMCA::Scalar pattern1 = std::sin(3.0 * theta) * std::sin(2.0 * phi);
      FMCA::Scalar pattern2 = std::cos(2.0 * theta) * std::cos(phi);
      FMCA::Scalar pattern3 = std::sin(4.0 * theta) * std::sin(phi) * std::sin(phi);
      
      // Combine patterns with different weights
      FMCA::Scalar combined_pattern = 0.5 * pattern1 + 0.3 * pattern2 + 0.2 * pattern3;
      
      // Add some radial variation based on position
      FMCA::Scalar radial_mod = 0.1 * std::sin(5.0 * (x + y + z));
      
      // Create a HARD jump: discontinuous step function
      FMCA::Scalar threshold = 0.1 + radial_mod;
      
      // Hard jump: either 0 or 1, no smoothing
      if (combined_pattern > threshold) {
        data(i) = 1.0;
      } else {
        data(i) = 0.0;
      }
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
  T.tic();
  SampletCoefficientsAnalyzer<ST> coeff_analyzer;
  coeff_analyzer.init(st, scoeffs);

  Index max_level = coeff_analyzer.computeMaxLevel(st);
  std::cout << "Max level = " << max_level << std::endl;

  auto max_coeff_per_level =
      coeff_analyzer.getMaxCoefficientsPerLevel(st, scoeffs);

  for (FMCA::Index i = 1; i < max_coeff_per_level.size(); ++i)
    std::cout << "c=" << max_coeff_per_level[i] / std::sqrt(Nd) << " alpha="
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
  T.toc("Slope Fitting: ");

  // // Initialize color vector for each column in P
  // Vector colr(P.cols());
  // for (const auto& [leaf, res] : results) {
  //   for (int j = 0; j < leaf->block_size(); ++j) {
  //     Scalar slope = res.get_slope();
  //     // save the slope just if slope < 2, otherwirse save dtilde
  //     Scalar slope_filtered = (slope <= 1.5) ? slope : dtilde;
  //     Scalar dtilde_binned = std::abs(std::floor((slope + 0.25) / 0.5) * 0.5);
  //     colr(leaf->indices()[j]) = slope;
  //   }
  // }
  // // Print the min value of colr
  // std::cout << "Min value of colr: " << colr.minCoeff() << std::endl;

  // FMCA::IO::plotPointsColor("SlopeSphere.vtk", P, colr);
  // FMCA::IO::plotPointsColor("FunctionSphere.vtk", P, data);

  return 0;
}