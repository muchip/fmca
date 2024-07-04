#include <iostream>
#include <fstream>
#include "SolvePoisson.h"
#include "read_files_txt.h"

#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluatorKernel = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluatorKernel = FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator = FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using EigenCholesky = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper, Eigen::MetisOrdering<int>>;

void runSimulation(const std::string& P_source_file, FMCA::Scalar sigma_factor, FMCA::Scalar beta, std::ofstream& results) {
    FMCA::Matrix P_sources;
    FMCA::Matrix P_quad;
    FMCA::Matrix P_quad_border;
    FMCA::Matrix Peval;
    FMCA::Matrix Normals;
    FMCA::Vector w_vec;
    FMCA::Vector w_vec_border;

    readTXT(P_source_file, P_sources, 2);
    readTXT("data/quadrature3_points_square40k.txt", P_quad, 2);
    readTXT("data/quadrature3_weights_square40k.txt", w_vec);
    readTXT("data/quadrature3_border40k.txt", P_quad_border, 2);
    readTXT("data/weights3_border40k.txt", w_vec_border);
    readTXT("data/3normals40k.txt", Normals, 2);
    readTXT("data/uniform_vertices_square10k.txt", Peval, 2);

    FMCA::Vector u_bc(P_quad_border.cols());
    for (FMCA::Index i = 0; i < P_quad_border.cols(); ++i) {
        u_bc[i] = 0;
    }

    FMCA::Vector f(P_quad.cols());
    for (FMCA::Index i = 0; i < P_quad.cols(); ++i) {
        FMCA::Scalar x = P_quad(0, i);
        FMCA::Scalar y = P_quad(1, i);
        f[i] = 2 * FMCA_PI * FMCA_PI * sin(FMCA_PI * x) * sin(FMCA_PI * y);
    }

    const FMCA::Scalar eta = 0.5;
    const FMCA::Index dtilde = 3;
    const FMCA::Scalar threshold_kernel = 1e-6;
    const FMCA::Scalar threshold_gradKernel = 1e-1;
    const FMCA::Scalar threshold_weights = 0;
    const FMCA::Scalar MPOLE_DEG = 4;
    const std::string kernel_type = "MATERN32";

    const Moments mom_sources(P_sources, MPOLE_DEG);
    const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
    H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
    FMCA::Vector minDistance = minDistanceVector(hst_sources, P_sources);

    auto maxElementIterator = std::max_element(minDistance.begin(), minDistance.end());
    FMCA::Scalar sigma_h = *maxElementIterator;
    FMCA::Scalar sigma = sigma_factor * sigma_h;

    FMCA::Vector u = SolvePoisson_constantWeights(
        2, P_sources, P_quad, w_vec, P_quad_border, w_vec_border, Normals,
        u_bc, f, sigma, eta, dtilde, threshold_kernel, threshold_gradKernel,
        threshold_weights, MPOLE_DEG, beta, kernel_type);
    FMCA::Vector u_grid = hst_sources.inverseSampletTransform(u);
    u_grid = hst_sources.toNaturalOrder(u_grid);

    const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
    const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources, kernel_funtion_ss);
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
    Kcomp_ss.init(hst_sources, eta, threshold_kernel);
    Kcomp_ss.compress(mat_eval_kernel_ss);
    const auto &triplets = Kcomp_ss.triplets();
    Eigen::SparseMatrix<double> Kcomp_ss_Sparse(P_sources.cols(), P_sources.cols());
    Kcomp_ss_Sparse.setFromTriplets(triplets.begin(), triplets.end());
    Kcomp_ss_Sparse.makeCompressed();

    Eigen::SparseMatrix<double> Kcomp_ss_Sparse_symm = Kcomp_ss_Sparse.selfadjointView<Eigen::Upper>();
    FMCA::Vector KU = Kcomp_ss_Sparse_symm * u;
    KU = hst_sources.inverseSampletTransform(KU);
    KU = hst_sources.toNaturalOrder(KU);

    const Moments cmom(P_sources, MPOLE_DEG);
    const Moments rmom(Peval, MPOLE_DEG);
    const H2ClusterTree hct(cmom, 0, P_sources);
    const H2ClusterTree hct_eval(rmom, 0, Peval);
    const FMCA::Vector sx = hct.toClusterOrder(u_grid);
    const usMatrixEvaluatorKernel mat_eval(rmom, cmom, kernel_funtion_ss);
    FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster> hmat;
    hmat.computePattern(hct_eval, hct, eta);
    FMCA::Vector srec = hmat.action(mat_eval, sx);
    FMCA::Vector rec = hct_eval.toNaturalOrder(srec);

    FMCA::Vector analytical_sol(Peval.cols());
    for (int i = 0; i < Peval.cols(); ++i) {
        double x = Peval(0, i);
        double y = Peval(1, i);
        analytical_sol[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
    }
    FMCA::Scalar error = (rec - analytical_sol).norm() / analytical_sol.norm();
    results << P_source_file << " " << sigma << " " << beta << " " << sigma_h << " " << error << std::endl;
}

int main() {
    std::vector<std::string> Psource_files = {
        "data/InteriorAndBnd_square100.txt",
        "data/vertices_square_Halton100.txt",

        "data/InteriorAndBnd_square1000.txt",
        "data/vertices_square_Halton1000.txt",

        "data/uniform_vertices_square10k.txt",
        "data/vertices_square_Halton10000.txt"

        // // "data/InteriorAndBnd_square100k.txt",
        // "data/vertices_square_Halton100000.txt"
    };

    std::vector<double> sigma_factors = {1.5, 1.8, 2.0};
    std::vector<double> betas = {10000, 100000};

    std::ofstream results("simulation_results_weak_formulation_l2.txt");
    results << "file_name sigma beta fill_distance error" << std::endl;

    for (const auto& Psource_file : Psource_files) {
        for (const auto& sigma_factor : sigma_factors) {
            for (const auto& beta : betas) {
                runSimulation(Psource_file, sigma_factor, beta, results);
            }
        }
    }

    results.close();
    return 0;
}
