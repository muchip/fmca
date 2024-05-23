#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/MatrixEvaluator"
#include "../FMCA/src/SparseCompressor/sparse_compressor_impl.h"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "read_files_txt.h"


#define DIM 2
#define GRADCOMPONENT 

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, long long int> largeSparse;

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using SparseMatrixEvaluator = FMCA::SparseMatrixEvaluator;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
    FMCA::Tictoc T;
    int NPTS_SOURCE = 1000;
    int NPTS_QUAD = 10000;
    int N_WEIGHTS = 10000;
    FMCA::Matrix P_sources;
    FMCA::Matrix P_quad;
    FMCA::Vector w_vec;

    P_sources = (FMCA::Matrix::Random(DIM, NPTS_SOURCE).array());
    P_quad = (FMCA::Matrix::Random(DIM, NPTS_QUAD).array());
    w_vec = 0.9 * FMCA::Vector::Random(NPTS_QUAD).array() + 1;   
//////////////////////////////////////////////////////////////////////////////

   std::ofstream resultsFile("compression_results.csv");
    resultsFile << "eta, sigma, dtilde, threshold, MPOLE_DEG, compression_error" << std::endl;

    // Loop over eta values
    for (const FMCA::Scalar eta : {0.5, 0.8, 1.0}) {
        // Loop over sigma values
        for (const FMCA::Scalar sigma : {0.2, 0.5, 1.0}) {
            // Loop over dtilde values
            for (const FMCA::Index dtilde : {2, 5, 8}) {
                // Loop over threshold values
                for (const FMCA::Scalar threshold : {0.0, 1e-8, 1e-4}) {
                    // Loop over MPOLE_DEG values
                    for (const FMCA::Scalar MPOLE_DEG : {4, 6, 8}) {

                        const Moments mom_sources(P_sources, MPOLE_DEG);
                        const Moments mom_quad(P_quad, MPOLE_DEG);
                    //////////////////////////////////////////////////////////////////////////////
                        const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
                        const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
                        H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
                        H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
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
                        FMCA::Vector w_perm = w_vec(permutationVector(hst_quad));   
                        FMCA::Matrix W = w_perm.asDiagonal();
                        T.tic();
                        hst_quad.sampletTransformMatrix(W);
                        double samplet_time = T.toc();
                        std::cout << "samplet transform weights time"<< ":         " <<  samplet_time << std::endl;
                        // Convert W to largeSparse format
                        largeSparse W_largeSparse(N_WEIGHTS, N_WEIGHTS);
                        std::vector<Eigen::Triplet<double>> triplets_weights;
                        for (long long int i = 0; i < N_WEIGHTS; ++i) {
                            for (long long int j = 0; j < N_WEIGHTS; ++j) {
                            triplets_weights.push_back(Eigen::Triplet<double>(i, j, W(i,j)));
                            }
                        }
                        W_largeSparse.setFromTriplets(triplets_weights.begin(), triplets_weights.end());
                        W_largeSparse.makeCompressed();
                    //////////////////////////////////////////////////////////////////////////////  
                        for (int i = 0; i < 1; ++i) {
                        std::cout << std::string(80, '-') << std::endl;
                        const FMCA::GradKernel function("GAUSSIAN", sigma, 1, i);
                        const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
                    //////////////////////////////////////////////////////////////////////////////  
                        //gradKernel compression
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
                        // eigen multiplication
                        T.tic();
                        FMCA::Matrix res_eigen  = Scomp_largeSparse * W_largeSparse * Scomp_largeSparse.transpose();
                        double mult_time_eigen = T.toc();
                        std::cout << "eigen multiplication time component " << i << ":      " << mult_time_eigen << std::endl;
                    //////////////////////////////////////////////////////////////////////////////  
                        // error
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
                            FMCA::Matrix y_reconstructed =  ek_transf.transpose() * (res_eigen * ej_transf).eval();

                            err_m += (y_original - y_reconstructed).squaredNorm();
                            nrm_m += (y_original).squaredNorm();
                        }
                        err_m = sqrt(err_m / nrm_m);
                        std::cout << "compression error:                    " << err_m << std::endl
                                    << std::flush;      
                        
                        
                        // Save results to CSV file
                        resultsFile << eta << "," << sigma << "," << dtilde << ","
                                    << threshold << "," << MPOLE_DEG << ","
                                    << err_m << std::endl;
                        }
                    }
                }
            }
        }
    }

    resultsFile.close();

    return 0;
}