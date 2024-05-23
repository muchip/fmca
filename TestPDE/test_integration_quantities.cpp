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
#define MPOLE_DEG 6
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
  int NPTS_SOURCE = 5;
  int NPTS_QUAD = 0;
  int N_WEIGHTS = 0;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Vector w_vec;
  //////////////////////////////////////////////////////////////////////////////
  // Read the txt files containing the points coords and fill the matrices
  //P_sources = 0.5 * (FMCA::Matrix::Random(DIM, NPTS_SOURCE).array() + 1);
  readTXT("data/grid_points.txt", P_sources, NPTS_SOURCE, 2);
  readTXT("data/baricenters_circle.txt", P_quad, NPTS_QUAD, 2);
  readTXT("data/triangles_volumes_circle.txt", w_vec, N_WEIGHTS);
  double minElement = *std::min_element(w_vec.begin(), w_vec.end());
  for (double& element : w_vec) {
            element /= minElement;
  }
  // w_vec.setRandom();
  // w_vec = 0.5 * w_vec.array() + 1;
  //////////////////////////////////////////////////////////////////////////////
  const FMCA::Scalar eta = 0.8;
  const FMCA::Index dtilde = 6;
  const FMCA::Scalar threshold = 1e-6;
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const Moments mom_quad(P_quad, MPOLE_DEG);
  //////////////////////////////////////////////////////////////////////////////
  // Create the H2 samplet trees and change the basis of T_quad such that the
  // norm square is equal to the weights
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
  FMCA::clusterTreeStatistics(hst_quad, P_quad);
  FMCA::clusterTreeStatistics(hst_sources, P_sources);
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

  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
  std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
  std::cout << "Number of weights = " << N_WEIGHTS << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "eta = " << eta << std::endl;
  std::cout << "dtilde = " << dtilde << std::endl;
  std::cout << "threshold = " << threshold << std::endl;
  std::cout << "MPOLE_DEG = " << MPOLE_DEG << std::endl;
  {
    std::vector<Eigen::Triplet<FMCA::Scalar>> trips =
        hst_quad.transformationMatrixTriplets();
    largeSparse T(hst_quad.block_size(), hst_quad.block_size());
    T.setFromTriplets(trips.begin(), trips.end());
    std::cout << "T_q^T * T_q - W:               " <<
     (FMCA::Matrix(T.transpose() * T).diagonal() - w_perm).norm()
              << std::endl;
  }
  // initialize the pattern and the final result matrix
  largeSparse pattern(NPTS_SOURCE, NPTS_SOURCE);
  largeSparse res = pattern;
  res.setZero();

  // iterate over the gradient components
  for (int i = 0; i < 2; ++i) {
    std::cout << std::string(80, '-') << std::endl;
    // std::cout << "enter in the for cicle for i = " << i << "        DONE"
    //           << std::endl;
    const FMCA::GradKernel function("GAUSSIAN", 0.5, 1, i);
    // std::cout << "created the gradient kernel for i = " << i << "   DONE"
    //           << std::endl;
    const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);

    //////////////////////////////////////////////////////////////////////////////  TESTS
    // test the permutations
    {
      FMCA::Matrix K = function.eval(P_sources, P_quad);
      FMCA::Matrix Kmat_eval;
      mat_eval.compute_dense_block(hst_sources, hst_quad, &Kmat_eval);
      const auto Perms = FMCA::permutationMatrix(hst_sources);
      const auto Permq = FMCA::permutationMatrix(hst_quad);
      const FMCA::Scalar err =
          (Perms.transpose() * K * Permq - Kmat_eval).norm() / Kmat_eval.norm();
      assert(err < 2 * FMCA_ZERO_TOLERANCE && "this has to be small");
    }
    //////////////////////////////////////////////////////////////////////////////
    // generate the stiffness matrix in the permuted single scale basis
        FMCA::Matrix Sdense;
        {
            const auto Perms = FMCA::permutationMatrix(hst_sources);
            const FMCA::Matrix gradK_eval = function.eval(P_sources, P_quad);
            Sdense = gradK_eval * w_vec.asDiagonal() * gradK_eval.transpose();
            Sdense = Perms.transpose() * Sdense * Perms;
            FMCA::Matrix Kmat_eval;
            mat_eval.compute_dense_block(hst_sources, hst_quad, &Kmat_eval);
            FMCA::Matrix Sdensetest =
                Kmat_eval * w_perm.asDiagonal() * Kmat_eval.transpose();
            const FMCA::Scalar err = (Sdensetest - Sdense).norm() / Sdense.norm();
            std::cout << "Sdensetest - Sdense:                  " << err << std::endl;
            //assert(err < 2 * FMCA_ZERO_TOLERANCE && "this has to be small");
        }
    //////////////////////////////////////////////////////////////////////////////
        FMCA::Matrix SSigmadense;
        {
            FMCA::Matrix K;
            mat_eval.compute_dense_block(hst_sources, hst_quad, &K);
            K = hst_sources.sampletTransform(K);
            K = hst_quad.sampletTransform(K.transpose()).transpose();
            SSigmadense = K * K.transpose();
            FMCA::Matrix testSSigma = Sdense;
            hst_sources.sampletTransformMatrix(testSSigma);
            const FMCA::Scalar err =
                (SSigmadense - testSSigma).norm() / testSSigma.norm();
            std::cout << "SSigmadense - testSSigma              " << err << std::endl;
        }
    //////////////////////////////////////////////////////////////////////////////

    // K full
    FMCA::Matrix K_full;
    mat_eval.compute_dense_block(hst_sources, hst_quad, &K_full);
    FMCA::Matrix S_full = K_full * w_perm.asDiagonal() * K_full.transpose();
    K_full = hst_sources.sampletTransform(K_full);
    K_full = hst_quad.sampletTransform(K_full.transpose()).transpose();
    FMCA::Matrix S2 = K_full * K_full.transpose();
    hst_sources.sampletTransformMatrix(S_full);

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

    ////////////////////////////////////////////////////////////////////////////// 

    // Scomp_large - K_full
    std::cout << "error Scomp_large - K_full:           "
              << (FMCA::Matrix(Scomp_largeSparse) - K_full).norm() /
                     (K_full.norm())
              << std::endl;

    // set the pattern as a dense matrix --> BAD, to be modified
    std::vector<Eigen::Triplet<double>> triplets;
    for (long long int i = 0; i < NPTS_SOURCE; ++i) {
      for (long long int j = 0; j < NPTS_SOURCE; ++j) {
        triplets.push_back(Eigen::Triplet<double>(i, j, 1.0));
      }
    }
  }

  return 0;
}
