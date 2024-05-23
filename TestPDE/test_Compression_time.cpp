#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS_SOURCE 10000
#define NPTS_QUAD 500000
#define DIM 2
#define MPOLE_DEG 6

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
// const std::vector<int> NPTS = {10000, 50000, 100000, 150000, 200000, 250000, 300000};
// for (int n : NPTS) {
  FMCA::Tictoc T;
  const FMCA::GradKernel function("GAUSSIAN", 1, 1, 0);
  const FMCA::Matrix P_sources = 0.5 * (FMCA::Matrix::Random(DIM, NPTS_SOURCE).array() + 1);
  const FMCA::Matrix P_quad = 0.5 * (FMCA::Matrix::Random(DIM, NPTS_QUAD).array() + 1);
  const FMCA::Scalar threshold = 1e-10;
  const FMCA::Index dtilde = 4;
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const Moments mom_quad(P_quad, MPOLE_DEG);
  const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);

    double eta = 2.0;
    // H2 samplets tree
    const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
    const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
    H2SampletTree hst_sources(mom_sources, samp_mom_sources , 0, P_sources);
    H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
    Scomp.init(hst_sources, hst_quad, eta, threshold);
    T.tic();
    Scomp.compress(mat_eval);
    double compressor_time = T.toc();
    // std::cout << NPTS_SOURCE << "," << NPTS_QUAD << "," << compressor_time << std::endl;
    std::cout << "NPTS_SOURCE =           " << NPTS_SOURCE << std::endl;
    std::cout << "NPTS_QUAD =             " << NPTS_QUAD << std::endl;
    std::cout << "Compressor time =       " << compressor_time << std::endl;

// }
  return 0;
}
