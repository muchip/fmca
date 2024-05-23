// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "read_files_txt.h"

#define DIM 2
#define MPOLE_DEG 8
#define GRADCOMPONENT 0

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, long long int> largeSparse;

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
    // Initialize the matrices of source points, quadrature points, weights
    FMCA::Tictoc T;
    int NPTS_SOURCE = 5000;
    int NPTS_QUAD = 500000;
    int N_WEIGHTS = 500000;
    FMCA::Matrix P_sources;
    FMCA::Matrix P_quad;
    FMCA::Vector w_vec;

    P_sources = (FMCA::Matrix::Random(DIM, NPTS_SOURCE).array());
    P_quad = (FMCA::Matrix::Random(DIM, NPTS_QUAD).array());
    w_vec = 0.02 * FMCA::Vector::Random(N_WEIGHTS).array() + 1;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
    std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
    std::cout << "Number of weights = " << N_WEIGHTS << std::endl;
    //////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
    const FMCA::Scalar eta = .5;
    const FMCA::Index dtilde = 6;
    const FMCA::Scalar threshold = 1e-4;
    const FMCA::Scalar sigma = 0.125;
    const Moments mom_sources(P_sources, MPOLE_DEG);
    const Moments mom_quad(P_quad, MPOLE_DEG);
//////////////////////////////////////////////////////////////////////////////
    // Create the H2 samplet trees and change the basis of T_quad such that the
    // norm square is equal to the weights
    const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
    const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
    H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
    H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
    //FMCA::clusterTreeStatistics(hst_quad, P_quad);
    //FMCA::clusterTreeStatistics(hst_sources, P_sources);
    FMCA::Vector w_perm = w_vec(permutationVector(hst_quad));
//////////////////////////////////////////////////////////////////////////////
    // reweight the samplet tree to accommodate the quadrature weights
    for (auto &&it : hst_quad) {
      FMCA::Matrix &Q = it.node().Q_;
      if (!it.nSons()) {
        Q = w_perm.segment(it.indices_begin(), Q.rows())
                .array()
                .sqrt()
                .matrix()
                .asDiagonal() *
            Q;
      }
    }
    //////////////////////////////////////////////////////////////////////////////
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
    std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
    std::cout << "Number of weights = " << N_WEIGHTS << std::endl;
    std::cout << "minimum element:                     " <<  *std::min_element(w_vec.begin(), w_vec.end()) << std::endl;
    std::cout << "maximum element:                     " << *std::max_element(w_vec.begin(), w_vec.end()) << std::endl;

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "sigma = " << sigma << std::endl;
    std::cout << "eta = " << eta << std::endl;
    std::cout << "dtilde = " << dtilde << std::endl;
    std::cout << "threshold = " << threshold << std::endl;
    std::cout << "MPOLE_DEG = " << MPOLE_DEG << std::endl;
    //////////////////////////////////////////////////////////////////////////////
    // initialize the pattern and the final result matrix
    largeSparse pattern(NPTS_SOURCE, NPTS_SOURCE);
    largeSparse res = pattern;
    res.setZero();    
    //////////////////////////////////////////////////////////////////////////////
    // iterate over the gradient components
    for (int i = 0; i < DIM; ++i) {
      std::cout << std::string(80, '-') << std::endl;
      const FMCA::GradKernel function("GAUSSIAN", sigma, 1, i);
      const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);

    //////////////////////////////////////////////////////////////////////////////
    //   // S_full for the compression error
    //   FMCA::Matrix K_full;
    //   mat_eval.compute_dense_block(hst_sources, hst_quad, &K_full);
    //   FMCA::Matrix S_full = K_full * w_perm.asDiagonal() * K_full.transpose();
    //   // S2 for the multiplication
    //   K_full = hst_sources.sampletTransform(K_full);
    //   K_full = hst_quad.sampletTransform(K_full.transpose()).transpose();
    //   FMCA::Matrix S2 = K_full * K_full.transpose();
    //   // S_full_transf for the compression error
    //   FMCA::Matrix S_full_transf = S_full;
    //   hst_sources.sampletTransformMatrix(S_full_transf);
    //////////////////////////////////////////////////////////////////////////////

      //  gradKernel compression
      FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
      Scomp.init(hst_sources, hst_quad, eta, threshold);
      T.tic();
      Scomp.compress(mat_eval);
      double compressor_time = T.toc();
      std::cout << "compression time component " << i << ":         " <<
      compressor_time << std::endl;
      const auto &trips = Scomp.triplets();
      std::cout << "anz:                                      " << trips.size() / NPTS_SOURCE << std::endl;
      largeSparse Scomp_largeSparse(NPTS_SOURCE, NPTS_QUAD);
      Scomp_largeSparse.setFromTriplets(trips.begin(), trips.end());
      Scomp_largeSparse.makeCompressed();

      //////////////////////////////////////////////////////////////////////////// 
      //Scomp_large - K_full
      // std::cout << "error Scomp_large - K_full:           "
      //           << (FMCA::Matrix(Scomp_largeSparse) - K_full).norm() /
      //                  (K_full.norm())
      //           << std::endl;
      //////////////////////////////////////////////////////////////////////////// 

      // set the pattern as a dense matrix --> BAD, to be modified
      std::vector<Eigen::Triplet<double>> triplets;
      for (long long int i = 0; i < NPTS_SOURCE; ++i) {
        for (long long int j = 0; j < NPTS_SOURCE; ++j) {
          triplets.push_back(Eigen::Triplet<double>(i, j, 1.0));
        }
      }
      pattern.setFromTriplets(triplets.begin(), triplets.end());
      pattern.makeCompressed();
      T.tic();
      formatted_sparse_multiplication_dotproduct(pattern, Scomp_largeSparse,Scomp_largeSparse);
      double mult_time = T.toc();
      std::cout << "multiplication time component " << i << ":      " <<
      mult_time << std::endl;
      res += pattern;
      // std::cout << "multiplication error:                 "
      //           << (FMCA::Matrix(pattern) - S2).norm() / (S2.norm()) << std::endl;
      std::cout << std::string(80, '-') << std::endl;

      //////////////////////////////////////////////////////////////////////////// 
      // compression error
      const auto Perms = FMCA::permutationMatrix(hst_sources);
      srand(time(NULL));
      FMCA::Vector ek(NPTS_SOURCE), ej(NPTS_SOURCE);
      FMCA::Scalar err_m = 0;
      FMCA::Scalar nrm_m = 0;
      for (auto n = 0; n < 100; ++n) {
        FMCA::Index k = rand() % NPTS_SOURCE;
        FMCA::Index j = rand() % NPTS_SOURCE;
        ek.setZero();
        ek(k) = 1;
        ej.setZero();
        ej(j) = 1;

        FMCA::Matrix P_sources_k = P_sources.col(hst_sources.indices()[k]);
        FMCA::Matrix gradK_row = function.eval(P_sources_k, P_quad);
        FMCA::Matrix P_sources_j = P_sources.col(hst_sources.indices()[j]);
        FMCA::Matrix gradK_col = function.eval(P_sources_j, P_quad);
        FMCA::Matrix y_original = gradK_row * w_vec.asDiagonal() * gradK_col.transpose();
        // std::cout << S_full(k,j) - y_original(0,0) << std::endl; // this works
        
        FMCA::Vector ek_transf;
        ek_transf = hst_sources.sampletTransform(ek);
        FMCA::Vector ej_transf;
        ej_transf = hst_sources.sampletTransform(ej);
        FMCA::Matrix y_reconstructed =  ek_transf.transpose() * (pattern * ej_transf).eval();
        // std::cout << S_full(k,j) - y_reconstructed(0,0) << std::endl; // this works

        err_m += (y_original - y_reconstructed).squaredNorm();
        nrm_m += (y_original).squaredNorm();
      }
      err_m = sqrt(err_m / nrm_m);
      std::cout << "compression error:                    " << err_m << std::endl
                << std::flush;      
    
    }
  return 0;
}
