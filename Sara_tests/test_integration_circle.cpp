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
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/Macros.h"
#include "read_files_txt.h"

#define DIM 2
#define MPOLE_DEG 10
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
    int NPTS_QUAD;
    int N_WEIGHTS;
    FMCA::Matrix P_sources;
    FMCA::Matrix P_quad;
    FMCA::Matrix w;
    // Read the txt files containing the points coords and fill the matrices
    P_sources = 0.5 * (FMCA::Matrix::Random(DIM, NPTS_SOURCE).array() + 1);
    //readTXT("grid_points.txt", P_sources, NPTS_SOURCE, 2);
    readTXT("baricenters_circle.txt", P_quad, NPTS_QUAD, 2);
    readTXT("triangles_volumes_circle.txt", w, N_WEIGHTS, 1);
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
    std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
    std::cout << "Number of weights = " << N_WEIGHTS << std::endl;

    const FMCA::Scalar threshold = 1e-10;
    const FMCA::Index dtilde = 8;
    const FMCA::Index eta = 0.5;
    const Moments mom_sources(P_sources, MPOLE_DEG);
    const Moments mom_quad(P_quad, MPOLE_DEG);

    // Create the H2 samplet trees and change the basis of T_quad such that the norm square is equal to the weights
    const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
    const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
    H2SampletTree hst_sources(mom_sources, samp_mom_sources , 0, P_sources);
    H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);

    FMCA::Vector w_vec = w.transpose();
    FMCA::Vector w_vec_reordered = w_vec;
    FMCA::Vector w_vec_inv = w_vec;
    // for (FMCA::Index i = 0; i < w_vec_inv.size() ; ++i)
    //     w_vec_inv(hst_quad.indices()[i]) = w_vec(i);
    for (FMCA::Index i = 0; i < w_vec_reordered.size() ; ++i)
        w_vec_reordered(i) = w_vec(0, hst_quad.indices()[i]);
    // w_vec.setOnes();

    for (auto &&it : hst_quad) {
    if(it.is_root()) {
        for (int i = 0; i < it.node().Q_.cols(); ++i)
        it.node().Q_.col(i) *= sqrt(w_vec_reordered(it.start_index() + i));
    } else {
        for (int i = 0; i < it.nsamplets(); ++i)
            it.node().Q_.col(i+it.nscalfs()) *= sqrt(w_vec_reordered(it.start_index() + i));
    }
    }

    // initialize the pattern and the final result matrix
    largeSparse pattern(NPTS_SOURCE, NPTS_SOURCE);
    largeSparse res = pattern;
    res.setZero();

    // iterate over the gradient components
    for (int i=0; i < 2; ++i){
        std::cout << std::string(80, '-') << std::endl;
        std::cout<<"enter in the for cicle for i = "<<i<<"        DONE"<<std::endl;
        const FMCA::GradKernel function("GAUSSIAN", 1, 1, i);
        std::cout<<"created the gradient kernel for i = "<<i<<"   DONE"<<std::endl;
        const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);

        // K full
        // FMCA::Matrix gradK_eval = function.eval(P_sources, P_quad);
        // FMCA::Matrix S_full_grad = gradK_eval*w.asDiagonal()*gradK_eval.transpose();
        // std::cout << "S_full_grad \n" << S_full_grad << std::endl;
        FMCA::Matrix K_full;
        mat_eval.compute_dense_block(hst_sources, hst_quad, &K_full);
        FMCA::Matrix S_full = K_full*w_vec_reordered.asDiagonal()*K_full.transpose();

        // FMCA::Matrix TsTT = S_full;
        // hst_sources.sampletTransformMatrix(TsTT); 
        K_full = hst_sources.sampletTransform(K_full);
        K_full = hst_quad.sampletTransform(K_full.transpose()).transpose();
        FMCA::Matrix S2 = K_full*K_full.transpose();
        //std::cout << "S2 norm before:                             "<< S2.norm()<< std::endl;
        // FMCA::Matrix S3 = S2;
        hst_sources.sampletTransformMatrix(S_full);
        std::cout << "S_full \n" << S_full << std::endl;
        std::cout << "S2 \n" << S2 << std::endl;
        //std::cout << "S2 norm:                                    "<< S2.norm()<< std::endl;

        // std::cout<< "error TsTT - S2:                             "<< (TsTT - S2).norm()/(S2.norm()) << std::endl;
        //std::cout<< "error S_full_gras - S2:                           "<< (S_full_grad - S2).norm()/(S2.norm()) << std::endl;

        //std::cout<<"evaluate the matrix for i = "<<i<<"           DONE"<<std::endl;
        // gradKernel compression
        FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
        Scomp.init(hst_sources, hst_quad, eta, threshold);
        T.tic();
        Scomp.compress(mat_eval);
        //std::cout<<"compress the matrix for i = "<<i<<"           DONE"<<std::endl;
        double compressor_time = T.toc();
        //std::cout << "compression time component " << i << ":         " << compressor_time << std::endl;

        const auto &trips = Scomp.triplets();
        //std::cout<<"triplets for i = "<<i<<"                      DONE"<<std::endl;
        largeSparse Scomp_largeSparse(NPTS_SOURCE, NPTS_QUAD);
        Scomp_largeSparse.setFromTriplets(trips.begin(), trips.end());
        //std::cout<<"setFromTriplets for i = "<<i<<"               DONE"<<std::endl;
        Scomp_largeSparse.makeCompressed();

        // Scomp_large - K_full
        std::cout<< "error Scomp_large - K_full:          "<< (FMCA::Matrix(Scomp_largeSparse) - K_full).norm()/(K_full.norm()) << std::endl;

        //std::cout<<"makeCompressed for i = "<<i<<"                DONE"<<std::endl;

        // set the pattern as a dense matrix --> BAD, to be modified 
        std::vector<Eigen::Triplet<double>> triplets;
        for (long long int i = 0; i < NPTS_SOURCE; ++i) {
            for (long long int j = 0; j < NPTS_SOURCE; ++j){
                triplets.push_back(Eigen::Triplet<double>(i, j, 1.0));
            }
        }
        pattern.setFromTriplets(triplets.begin(), triplets.end());
        pattern.makeCompressed();
        //std::cout<<"pattern makeCompressed for i = "<<i<<"        DONE"<<std::endl;
        // multiplication gradKernal * gradKernel.transpose()
        T.tic();
        formatted_sparse_multiplication_dotproduct(pattern, Scomp_largeSparse, Scomp_largeSparse);
        //std::cout<<"multiplication for i = "<<i<<"                DONE"<<std::endl;
        double mult_time = T.toc();
        //std::cout << "multiplication time component " << i << ":      " << mult_time << std::endl;
        res += pattern;
        std::cout<< "error:                                "<< (FMCA::Matrix(pattern) - S2).norm()/(S2.norm()) << std::endl;
        std::cout << std::string(80, '-') << std::endl;
    }

  return 0;
}
