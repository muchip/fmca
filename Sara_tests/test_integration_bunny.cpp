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
#include "read_files.h"
#include "read_files_txt.h"

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
    // Initialize the matrices of source points, quadrature points, weights
    FMCA::Tictoc T;
    int NPTS_SOURCE;
    int NPTS_QUAD;
    int N_WEIGHTS;
    FMCA::Matrix P_sources;
    FMCA::Matrix P_quad;
    FMCA::Matrix w;
    // Read the txt files containing the points coords and fill the matrices
    readTXT("grid_points_bunny.txt", P_sources, NPTS_SOURCE, 3);
    readTXT("tetrahedra_volumes.txt", w, N_WEIGHTS, 1);
    readCSV("barycenters.csv", P_quad, NPTS_QUAD, 3);
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
    std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
    std::cout << "Number of weights = " << N_WEIGHTS << std::endl;

    const FMCA::Scalar threshold = 1e-10;
    const FMCA::Index dtilde = 4;
    const FMCA::Index eta = 0.8;
    const Moments mom_sources(P_sources, MPOLE_DEG);
    const Moments mom_quad(P_quad, MPOLE_DEG);

    // Create the H2 samplet trees and change the basis of T_quad such that the norm square is equal to the weights
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
    // initialize the pattern and the final result matrix
    largeSparse pattern(NPTS_SOURCE, NPTS_SOURCE);
    largeSparse res = pattern;
    res.setZero();
    const double ridgep = 0;
    std::vector<double> data_p;

    // iterate over the gradient components
    for (int i=0; i < 2; ++i){
        std::cout << std::string(80, '-') << std::endl;
        std::cout<<"enter in the for cicle for i = "<<i<<"        DONE"<<std::endl;
        // Gradient of the Kernel, component i
        const FMCA::GradKernel function("GAUSSIAN", 1, 1, i);
        std::cout<<"created the gradient kernel for i = "<<i<<"   DONE"<<std::endl;
        // evaluate the gradKernel
        const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
        std::cout<<"evaluate the matrix for i = "<<i<<"           DONE"<<std::endl;
        // gradKernel compression
        FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
        Scomp.init(hst_sources, hst_quad, eta, threshold);
        T.tic();
        Scomp.compress(mat_eval);
        std::cout<<"compress the matrix for i = "<<i<<"           DONE"<<std::endl;
        double compressor_time = T.toc();
        std::cout << "compression time component " << i << ":         " << compressor_time << std::endl;

        const auto &trips = Scomp.triplets();
        std::cout<<"triplets for i = "<<i<<"                      DONE"<<std::endl;
        largeSparse Scomp_largeSparse(NPTS_SOURCE, NPTS_QUAD);
        Scomp_largeSparse.setFromTriplets(trips.begin(), trips.end());
        std::cout<<"setFromTriplets for i = "<<i<<"               DONE"<<std::endl;
        Scomp_largeSparse.makeCompressed();
        std::cout<<"makeCompressed for i = "<<i<<"                DONE"<<std::endl;

        // set pattern as Samplet matrix generator
        largeSparse S = sampletMatrixGenerator(P_sources, MPOLE_DEG, dtilde, eta, threshold,ridgep, &data_p);
        S.makeCompressed();
        largeSparse A = S.selfadjointView<Eigen::Upper>();
        S = A;
        largeSparse pattern = S;
        pattern.makeCompressed();
        std::cout<<"pattern makeCompressed for i = "<<i<<"        DONE"<<std::endl;

        // multiplication gradKernal * gradKernel.transpose()
        T.tic();
        formatted_sparse_multiplication_dotproduct(pattern, Scomp_largeSparse, Scomp_largeSparse);
        std::cout<<"multiplication for i = "<<i<<"                DONE"<<std::endl;
        double mult_time = T.toc();
        std::cout << "multiplication time component " << i << ":      " << mult_time << std::endl;
        res += pattern;
        std::cout << std::string(80, '-') << std::endl;
    }

  return 0;
}
