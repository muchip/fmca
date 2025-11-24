#include <random>

#include "../FMCA/SmoothnessClassDetection"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 1

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MapCoeffDiam =
    std::map<const ST*,
             std::pair<std::vector<FMCA::Scalar>, std::vector<FMCA::Scalar>>>;
using namespace FMCA;

// Structure to hold all information about a singularity
struct SingularityInfo {
  Scalar slope = 0.0;
  std::vector<Scalar> coeffs;
  std::vector<Scalar> diams;
};

Scalar combined_function(Scalar x) {
  if (x < -0.4) {
    return 6;
  } else if (x >= -0.4 && x < -0.35) {
    return 0.1 * std::abs(20 * x + 9) + 6;
  } else if (x >= -0.35 && x < -0.15) {
    return 0.1 * std::abs(20 * x + 5) + 6;
  } else if (x >= -0.15 && x < -0.05) {
    return 0.1 * std::abs(20 * x + 1) + 6;
  } else if (x >= -0.05 && x < 0.55) {
    return 6 + std::sin(20 * FMCA_PI * x);
  } else if (x >= 0.55 && x <= 1) {
    return 4 - 20 * std::abs(x - 0.7) * (x - 0.7);
  }
  return 0;
  // return std::sqrt(x);
}

FMCA::Index findClosestPoint(const FMCA::Matrix& P, Scalar target_x) {
  FMCA::Index closest_idx = 0;
  Scalar min_dist = std::numeric_limits<Scalar>::max();

  for (FMCA::Index i = 0; i < P.cols(); ++i) {
    Scalar dist = std::pow(P(0, i) - target_x, 2);
    if (dist < min_dist) {
      min_dist = dist;
      closest_idx = i;
    }
  }
  return closest_idx;
}

//////////////////////////////////////////////////////////////////////////////////////////
std::vector<SingularityInfo> runN(const FMCA::Index l) {
  std::vector<SingularityInfo> res(3);  // For 3 singularities
  const FMCA::Index d = 1;
  const FMCA::Index N = 1 << l;
  const FMCA::Scalar h = 1. / N;
  FMCA::Index Nd = std::pow(N, d);
  constexpr FMCA::Index dtilde = 5;
  FMCA::Tictoc T;
  FMCA::Matrix P(d, Nd);
  FMCA::Vector data(Nd);
  T.tic();
  {
    // generate a uniform grid from -1.001 to 1
    const Scalar x_min = 0;  // -1.000097;
    const Scalar x_max = 0.999983;
    const Scalar range = x_max - x_min;
    const Scalar h_new = range / N;

    FMCA::Vector pt(d);
    pt.setZero();
    FMCA::Index p = 0;
    FMCA::Index i = 0;
    while (pt(d - 1) < N) {
      if (pt(p) >= N) {
        pt(p) = 0;
        ++p;
      } else {
        P.col(i++) = (h_new * (pt.array() + 0.5) + x_min).matrix();
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
      data(i) = combined_function(P(0, i));
    }
  }
  T.toc("data generation: ");
  std::cout << "Nd=" << Nd << std::endl;

  // Find the indices of the closest points to each singularity
  Index idx_singularity1 = findClosestPoint(P, -0.15);
  Index idx_singularity2 = findClosestPoint(P, 0.7);
  Index idx_singularity3 = findClosestPoint(P, 0.55);

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
  //////////////////////////////////////////////////////////////
  T.tic();
  MapCoeffDiam leafData;
  coeff_analyzer.traverseAndStackCoefficientsAndDiametersL2Norm(st, scoeffs,
                                                                leafData);

  SlopeFitter<ST> fitter;
  fitter.init(leafData, dtilde, 1e-16);
  auto results = fitter.fitSlope();
  std::cout << "--------------------------------------------------------"
            << std::endl;
  T.toc("fitting: ");

  //////////////////////////////////////////////////////////////
  Vector colr(P.cols());
  for (const auto& [leaf, fit_result] : results) {
    for (int j = 0; j < leaf->block_size(); ++j) {
      Scalar slope = fit_result.get_slope();
      Index current_idx = leaf->indices()[j];

      // Check if this index matches any of our singularity points
      if (current_idx == idx_singularity1) {
        res[0].slope = slope;
        // Get coefficients and diameters from leafData
        if (leafData.find(leaf) != leafData.end()) {
          res[0].coeffs = leafData[leaf].first;
          res[0].diams = leafData[leaf].second;
        }
      }
      if (current_idx == idx_singularity2) {
        res[1].slope = slope;
        if (leafData.find(leaf) != leafData.end()) {
          res[1].coeffs = leafData[leaf].first;
          res[1].diams = leafData[leaf].second;
        }
      }
      if (current_idx == idx_singularity3) {
        res[2].slope = slope;
        if (leafData.find(leaf) != leafData.end()) {
          res[2].coeffs = leafData[leaf].first;
          res[2].diams = leafData[leaf].second;
        }
      }
      colr(current_idx) = slope;
    }
  }

  return res;
}

int main() {
  // Test different nu values
  std::vector<std::vector<SingularityInfo>> all_results;

  std::vector<Index> ls = {20};
                           // 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
  for (Index l : ls) {
    std::vector<SingularityInfo> res = runN(l);
    all_results.push_back(res);
  }

  // Print as a nice table with slopes
  std::cout << "\n========================================" << std::endl;
  std::cout << "SLOPES AT SINGULARITIES" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::setw(8) << "l" << std::setw(15) << "Sing1" << std::setw(15)
            << "Sing2" << std::setw(15) << "Sing3" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  for (size_t i = 0; i < ls.size(); ++i) {
    std::cout << std::setw(8) << static_cast<int>(std::pow(2, ls[i]))
              << std::setw(15) << std::fixed << std::setprecision(6)
              << all_results[i][0].slope << std::setw(15) << std::fixed
              << std::setprecision(6) << all_results[i][1].slope
              << std::setw(15) << std::fixed << std::setprecision(6)
              << all_results[i][2].slope << std::endl;
  }
  std::cout << "========================================" << std::endl;

  // Print detailed information for each singularity
  const char* sing_names[] = {"Singularity 1", "Singularity 2",
                              "Singularity 3"};

  for (int sing_idx = 0; sing_idx < 3; ++sing_idx) {
    std::cout << "\n========================================" << std::endl;
    std::cout << sing_names[sing_idx] << std::endl;
    std::cout << "========================================" << std::endl;

    for (size_t i = 0; i < ls.size(); ++i) {
      const auto& info = all_results[i][sing_idx];
      std::cout << "\nl = " << static_cast<int>(std::pow(2, ls[i]))
                << ", slope = " << std::fixed << std::setprecision(6)
                << info.slope << std::endl;

      if (!info.coeffs.empty() && !info.diams.empty()) {
        std::cout << "  Level" << std::setw(15) << "Coeff" << std::setw(15)
                  << "Diam" << std::endl;
        std::cout << "  " << std::string(44, '-') << std::endl;

        for (size_t j = 0; j < info.coeffs.size(); ++j) {
          std::cout << "  " << std::setw(5) << j << std::setw(15)
                    << std::scientific << std::setprecision(6) << info.coeffs[j]
                    << std::setw(15) << std::scientific << std::setprecision(6)
                    << info.diams[j] << std::endl;
        }
      }
    }
  }

  return 0;
}