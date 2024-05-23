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
#include <cstdlib>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "read_files_txt.h"

#define DIM 2
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
    int NPTS_SOURCE = 1000;
    int NPTS_QUAD = 20000;
    int N_WEIGHTS = 20000;
    FMCA::Matrix P_sources;
    FMCA::Matrix P_quad;
    FMCA::Vector w_vec;

    P_sources = (FMCA::Matrix::Random(DIM, NPTS_SOURCE).array());
    P_quad = (FMCA::Matrix::Random(DIM, NPTS_QUAD).array());
    w_vec = 0.0 * FMCA::Vector::Random(N_WEIGHTS).array() + 1;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
    std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
    std::cout << "Number of weights = " << N_WEIGHTS << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  std::vector<FMCA::Scalar> eta_values = {0.4, 0.8, 1.2};
  std::vector<FMCA::Index> dtilde_values = {2, 4, 6 };
  std::vector<FMCA::Scalar> threshold_values = {1e-2, 1e-6, 1e-10};
  std::vector<FMCA::Scalar> sigma_values = {0.2, 0.6, 1};
  std::vector<FMCA::Scalar> mpoledeg_values = {4, 7, 10};

  // Open CSV file for writing results
  std::ofstream resultsFile("results.csv");
  resultsFile << "eta,dtilde,threshold,sigma,mpole_deg,compression_time,anz,multiplication_time,compression_error\n";

  for (FMCA::Scalar eta : eta_values) {
    for (FMCA::Index dtilde : dtilde_values) {
      for (FMCA::Scalar threshold : threshold_values) {
        for (FMCA::Scalar MPOLE_DEG : mpoledeg_values) {
          for (FMCA::Scalar sigma : sigma_values) {
          const Moments mom_sources(P_sources, MPOLE_DEG);
          const Moments mom_quad(P_quad, MPOLE_DEG);
          //////////////////////////////////////////////////////////////////////////////
          // Create the H2 samplet trees and change the basis of T_quad such that the
          // norm square is equal to the weights
          const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
          const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
          if (mom_sources.interp().Xi().cols() < samp_mom_sources.mdtilde()){
            continue;
          }

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
          // initialize the pattern and the final result matrix
          largeSparse pattern(NPTS_SOURCE, NPTS_SOURCE);
          largeSparse res = pattern;
          res.setZero();
          
          //////////////////////////////////////////////////////////////////////////////
          // iterate over the gradient components
          for (int i = 0; i < 1; ++i) {
            const FMCA::GradKernel function("MATERN32", sigma, 1, i);
            const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
          //////////////////////////////////////////////////////////////////////////////

            // K full
            // FMCA::Matrix K_full;
            // mat_eval.compute_dense_block(hst_sources, hst_quad, &K_full);
            // FMCA::Matrix S_full = K_full * w_perm.asDiagonal() * K_full.transpose();
            // K_full = hst_sources.sampletTransform(K_full);
            // K_full = hst_quad.sampletTransform(K_full.transpose()).transpose();
            // FMCA::Matrix S2 = K_full * K_full.transpose();
            // hst_sources.sampletTransformMatrix(S_full);

            //  gradKernel compression
            FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
            Scomp.init(hst_sources, hst_quad, eta, threshold);
            T.tic();
            Scomp.compress(mat_eval);
            double compressor_time = T.toc();
            const auto &trips = Scomp.triplets();
            largeSparse Scomp_largeSparse(NPTS_SOURCE, NPTS_QUAD);
            Scomp_largeSparse.setFromTriplets(trips.begin(), trips.end());
            Scomp_largeSparse.makeCompressed();

            ////////////////////////////////////////////////////////////////////////////// 
            // Scomp_large - K_full
            // std::cout << "error Scomp_large - K_full:           "
            //           << (FMCA::Matrix(Scomp_largeSparse) - K_full).norm() /
            //                  (K_full.norm())
            //           << std::endl;

            // std::cout<<"makeCompressed for i = "<<i<<" DONE"<<std::endl;

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
            formatted_sparse_multiplication_dotproduct(pattern, Scomp_largeSparse,
                                                      Scomp_largeSparse);
            double mult_time = T.toc();
            res += pattern;
            // std::cout << "multiplication error:                 "
            //           << (FMCA::Matrix(pattern) - S2).norm() / (S2.norm()) << std::endl;

            // compression error
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

              FMCA::Vector ek_transf;
              ek_transf = hst_sources.sampletTransform(ek);
              FMCA::Vector ej_transf;
              ej_transf = hst_sources.sampletTransform(ej);
              FMCA::Matrix y_reconstructed = ek_transf.transpose() * (pattern * ej_transf).eval();

              err_m += (y_original - y_reconstructed).squaredNorm();
              nrm_m += (y_original).squaredNorm();
            }
            err_m = sqrt(err_m / nrm_m);            
            // Save results to CSV
            resultsFile << std::scientific << std::setprecision(4);
            resultsFile << eta << "," << dtilde << "," << threshold << "," << sigma << "," << MPOLE_DEG << ","
                        << compressor_time << "," << trips.size() / NPTS_SOURCE << ","
                        << mult_time << ","
                        << err_m << "\n";
            }    
          }
        }
      }
    }
  }
  resultsFile.close();
  return 0;
}