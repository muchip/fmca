// This code computes the integral of gradK * gradK.transpose() using some weights W.
// step 1: initilaize the gradient of the kernel, the matrices of source and quadrature points and the weights
// step 2: create the matrices T_s and T_q of samplets transformation
// step 3: compress gradK in samplet basis, obtaining gradK_s = T_s * gradK * T_q.transpose()
// step 4: compute fast matrix multiplication

#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/Macros.h"

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, long long int> largeSparse;

#define NPTS_SOURCE 1000
#define NPTS_QUAD 10000
#define DIM 3
#define MPOLE_DEG 6
#define GRADCOMPONENT 0

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
    FMCA::Tictoc T;
    // step 1 
    const FMCA::GradKernel function("GAUSSIAN", 1, 1, GRADCOMPONENT);
    const FMCA::Matrix P_sources = 0.5 * (FMCA::Matrix::Random(DIM, NPTS_SOURCE).array() + 1);
    const FMCA::Matrix P_quad = 0.5 * (FMCA::Matrix::Random(DIM, NPTS_QUAD).array() + 1);
    const Eigen::VectorXd w = (Eigen::VectorXd::Random(NPTS_QUAD).array() + 1.0) * 0.5;

    const FMCA::Scalar threshold = 1e-10;
    const FMCA::Index dtilde = 4;
    const Moments mom_sources(P_sources, MPOLE_DEG);
    const Moments mom_quad(P_quad, MPOLE_DEG);
    const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
    double eta = 0.8;

    // step 2
    const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
    const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
    H2SampletTree hst_sources(mom_sources, samp_mom_sources , 0, P_sources);
    H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
    for (auto &&it : hst_quad) {
      if(it.is_root()) {
        for (int i = 0; i < it.node().Q_.cols(); ++i)
          it.node().Q_.col(i) *= sqrt(w(it.start_index() + i));
      } else {
          for (int i = 0; i < it.nsamplets(); ++i)
            it.node().Q_.col(i+it.nscalfs()) *= sqrt(w(it.start_index() + i));
      }
    }

    // Eigen::MatrixXd T_s =
    //     hst_sources.inverseSampletTransform(Eigen::MatrixXd::Identity(P_sources.cols(), P_sources.cols()));
    // Eigen::MatrixXd T_q =
    //     hst_quad.inverseSampletTransform(Eigen::MatrixXd::Identity(P_quad.cols(), P_quad.cols()));

    // // check the correctness of T_s and T_q
    // Eigen::MatrixXd W = Eigen::MatrixXd(w.asDiagonal());
    // std::cout << std::string(60, '-') << std::endl;
    // std::cout << "check for the sources tranformer, basis orthogonality error:    "
    //         << (T_s.transpose() * T_s - Eigen::MatrixXd::Identity(P_sources.cols(), P_sources.cols()))
    //                     .norm() /
    //                 sqrt(P_sources.cols())
    //         << std::endl;
    // std::cout << std::string(80, '-') << std::endl;

    // Eigen::MatrixXd T_qW = T_q.transpose() * T_q;
    // for (int i = 0; i < NPTS_QUAD; ++i){
    //     T_qW(i,i) -= w[i];
    // }
    // std::cout << "check for the quad tranformer, basis orthogonality error:    "
    //         << T_qW.norm() /
    //                 sqrt(P_quad.cols())
    //         << std::endl;
    // std::cout << std::string(80, '-') << std::endl;

    // step 3
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
    Scomp.init(hst_sources, hst_quad, eta, threshold);
    T.tic();
    Scomp.compress(mat_eval);
    double compressor_time = T.toc();
    std::cout << "compression time:      " << compressor_time << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // step 4: gradK row major 
    const auto &trips = Scomp.triplets();
    largeSparse Scomp_largeSparse(NPTS_SOURCE, NPTS_QUAD);
    Scomp_largeSparse.setFromTriplets(trips.begin(), trips.end());
    Scomp_largeSparse.makeCompressed();
    // std::cout << "Scomp_largeSparse:      " << Scomp_largeSparse << std::endl;
    // std::cout << std::string(80, '-') << std::endl;
    
    largeSparse pattern(NPTS_SOURCE, NPTS_SOURCE);
    pattern.setIdentity();
    pattern.makeCompressed();
    formatted_sparse_multiplication_dotproduct(pattern, Scomp_largeSparse, Scomp_largeSparse);
    // std::cout << "res:      " << pattern << std::endl;
    // std::cout << std::string(80, '-') << std::endl;
    return 0;
}
