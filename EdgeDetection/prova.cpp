#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/Samplets"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;

int main() {
  constexpr FMCA::Index l = 13;
  constexpr FMCA::Index d = 1;
  constexpr FMCA::Index N = 1 << l;
  constexpr FMCA::Scalar h = 1. / N;
  FMCA::Index Nd = std::pow(N, d);
  constexpr FMCA::Index dtilde = 4;
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
    for (FMCA::Index i = 0; i < Nd; ++i)
      data(i) = P.col(i).dot(pt) > 0.5 * sqrt(d);
  }
  T.toc("data generation: ");
  std::cout << "Nd=" << Nd << std::endl;
  T.tic();
  const SampletMoments samp_mom(P, dtilde - 1);
  SampletTree st(samp_mom, 0, P);
  T.toc("samplet tree: ");
  FMCA::Index max_level = 0;
  for (auto &&node : st) {
    max_level = node.level() < max_level ? max_level : node.level();
  }
  std::cout << max_level << std::endl;
/*
  const FMCA::Vector scoeffs = st.sampletTransform(st.toClusterOrder(data));
  std::vector<FMCA::Index> levels = FMCA::internal::sampletLevelMapper(st);
  FMCA::Index max_l = 0;
  for (const auto &it : levels) max_l = max_l < it ? it : max_l;
  std::vector<FMCA::Scalar> max_c(max_l + 1);
  for (FMCA::Index i = 0; i < scoeffs.size(); ++i)
    max_c[levels[i]] = max_c[levels[i]] < std::abs(scoeffs[i])
                           ? std::abs(scoeffs[i])
                           : max_c[levels[i]];
  for (FMCA::Index i = 1; i < max_c.size(); ++i)
    std::cout << "c=" << max_c[i] / std::sqrt(Nd)
              << " alpha=" << std::log(max_c[i - 1] / max_c[i]) / (std::log(2))
              << std::endl;
              */
  return 0;
}
