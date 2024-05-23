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

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "read_files_txt.h"
#include "read_files.h"

#define DIM 3
#define MPOLE_DEG 6
#define GRADCOMPONENT 0

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, long long int> largeSparse;

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
    // Initialize the matrices of source points, quadrature points, weights
    FMCA::Tictoc T;
    int NPTS_SOURCE = 1000;
    int NPTS_QUAD;
    int N_WEIGHTS;
    FMCA::Matrix P_sources;
    FMCA::Matrix P_quad;
    FMCA::Vector w_vec;
    // Read the txt files containing the points coords and fill the matrices
    P_sources = FMCA::Matrix::Random(DIM, NPTS_SOURCE).array();
    // readTXT("grid_points_bunny_1000.txt", P_sources, NPTS_SOURCE, 3);
    readTXT("data/tetrahedra_volumes.txt", w_vec, N_WEIGHTS);
    // w_vec.setOnes();
    readCSV("data/barycenters.csv", P_quad, NPTS_QUAD, 3);
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
    std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
    std::cout << "Number of weights = " << N_WEIGHTS << std::endl;
    //////////////////////////////////////////////////////////////////////////////
    const FMCA::Scalar eta = 0.5;
    const FMCA::Index dtilde = 4;
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
    // {
    // std::vector<Eigen::Triplet<FMCA::Scalar>> trips =
    //     hst_quad.transformationMatrixTriplets();
    // largeSparse T(hst_quad.block_size(), hst_quad.block_size());
    // T.setFromTriplets(trips.begin(), trips.end());
    // std::cout << " T_q.transpose() * T_q).diagonal() - w_perm = "
    //             << (FMCA::Matrix(T.transpose() * T).diagonal() - w_perm).norm()
    //             << std::endl;
    // }
    //////////////////////////////////////////////////////////////////////////////
    // initialize the pattern and the final result matrix
    largeSparse pattern(NPTS_SOURCE, NPTS_SOURCE);
    largeSparse res = pattern;
    res.setZero();
    //////////////////////////////////////////////////////////////////////////////
    // iterate over the gradient components
        std::cout << std::string(80, '-') << std::endl;
        const FMCA::CovarianceKernel function("GAUSSIAN", 1 , 1);
        const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);

    //     // test the permutations
    //     {
    //         FMCA::Matrix K = function.eval(P_sources, P_quad);
    //         FMCA::Matrix Kmat_eval;
    //         mat_eval.compute_dense_block(hst_sources, hst_quad, &Kmat_eval);
    //         const auto Perms = FMCA::permutationMatrix(hst_sources);
    //         const auto Permq = FMCA::permutationMatrix(hst_quad);
    //         const FMCA::Scalar err =
    //             (Perms.transpose() * K * Permq - Kmat_eval).norm() / Kmat_eval.norm();
    //   std::cout << "Perms.transpose() * K * Permq - Kmat_eval                      " << err << std::endl;
    //         assert(err < 2 * FMCA_ZERO_TOLERANCE && "this has to be small");
    //     }

        // generate the stiffness matrix in the permuted single scale basis
        FMCA::Matrix Sdense;
        // {
        //     const auto Perms = FMCA::permutationMatrix(hst_sources);
        //     const FMCA::Matrix gradK_eval = function.eval(P_sources, P_quad);
        //     Sdense = gradK_eval * w_vec.asDiagonal() * gradK_eval.transpose();
        //     Sdense = Perms.transpose() * Sdense * Perms;
        //     FMCA::Matrix Kmat_eval;
        //     mat_eval.compute_dense_block(hst_sources, hst_quad, &Kmat_eval);
        //     FMCA::Matrix Sdensetest =
        //         Kmat_eval * w_perm.asDiagonal() * Kmat_eval.transpose();
        //     const FMCA::Scalar err = (Sdensetest - Sdense).norm() / Sdense.norm();
        //     std::cout << "Sdensetest - Sdense:                  " << err << std::endl;
        //     //assert(err < 2 * FMCA_ZERO_TOLERANCE && "this has to be small");
        // }
        // FMCA::Matrix SSigmadense;
        // {
        //     FMCA::Matrix K;
        //     mat_eval.compute_dense_block(hst_sources, hst_quad, &K);
        //     K = hst_sources.sampletTransform(K);
        //     K = hst_quad.sampletTransform(K.transpose()).transpose();
        //     SSigmadense = K * K.transpose();
        //     FMCA::Matrix testSSigma = Sdense;
        //     hst_sources.sampletTransformMatrix(testSSigma);
        //     const FMCA::Scalar err =
        //         (SSigmadense - testSSigma).norm() / testSSigma.norm();
        //     std::cout << "SSigmadense - testSSigma              " << err << std::endl;
        // }

        FMCA::Matrix K_full;
        mat_eval.compute_dense_block(hst_sources, hst_quad, &K_full);
        FMCA::Matrix S_full = K_full * w_perm.asDiagonal() * K_full.transpose();

        K_full = hst_sources.sampletTransform(K_full);
        K_full = hst_quad.sampletTransform(K_full.transpose()).transpose();
        FMCA::Matrix S2 = K_full * K_full.transpose();
        hst_sources.sampletTransformMatrix(S_full);
        std::cout << "S_full - S2:                        " << (S_full - S2).norm() / S_full.norm() << std::endl; 
        //////////////////////////////////////////////////////////////////////////////    
        std::cout << std::string(80, '-') << std::endl;
        FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
        Scomp.init(hst_sources, hst_quad, eta, threshold);
        T.tic();
        Scomp.compress(mat_eval);
        double compressor_time = T.toc();
        std::cout << "compression time         " << ":         " <<
        compressor_time << std::endl;

        const auto &trips = Scomp.triplets();
        std::cout << "anz:                                " << trips.size() / NPTS_SOURCE << std::endl;
        largeSparse Scomp_largeSparse(NPTS_SOURCE, NPTS_QUAD);
        Scomp_largeSparse.setFromTriplets(trips.begin(), trips.end());
        Scomp_largeSparse.makeCompressed();
        FMCA::Matrix myS2 = Scomp_largeSparse * Scomp_largeSparse.transpose();
        //////////////////////////////////////////////////////////////////////////////
        // // Scomp_large - K_full
        // std::cout << "error Scomp_large - K_full:           "
        //             << (FMCA::Matrix(Scomp_largeSparse) - K_full).norm() /
        //                     (K_full.norm())
        //             << std::endl;
        //////////////////////////////////////////////////////////////////////////////

        // set the pattern as a dense matrix --> BAD, to be modified
        std::vector<Eigen::Triplet<double>> triplets;
        for (long long int i = 0; i < NPTS_SOURCE; ++i) {
            for (long long int j = 0; j < NPTS_SOURCE; ++j) {
                triplets.push_back(Eigen::Triplet<double>(i, j, 1.0));
            }
        }
        pattern.setFromTriplets(triplets.begin(), triplets.end());
        pattern.makeCompressed();

        //////////////////////////////////////////////////////////////////////////////
        //  multiplication gradKernal * gradKernel.transpose()
        T.tic();
        formatted_sparse_multiplication_dotproduct(pattern, Scomp_largeSparse,
                                                    Scomp_largeSparse);
        double mult_time = T.toc();
        std::cout << "multiplication time          " << ":      " <<
        mult_time << std::endl;
        res += pattern;
        //////////////////////////////////////////////////////////////////////////////
        std::cout << "FMCA::Matrix(pattern) - S2 :          "
                    << (S2 - FMCA::Matrix(pattern)).norm() / (S2.norm()) << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        //////////////////////////////////////////////////////////////////////////////

    return 0;
}
