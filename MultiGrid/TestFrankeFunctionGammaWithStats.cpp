#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Plotter.h"
#include "../FMCA/src/util/Tictoc.h"
#include "read_files_txt.h"
#include "MultiGridFunctions.h"
#include "TimeStats.h"

#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluatorKernel =
    FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluatorKernel =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using EigenCholesky =
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>;

// 9) Franke Function, as in your code
Scalar FrankeFunction(Scalar x, Scalar y) {
    Scalar term1 = (3.0 / 4.0) 
                 * std::exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0) / 4.0)
                            -((9.0 * y - 2.0) * (9.0 * y - 2.0) / 4.0));
    Scalar term2 = (3.0 / 4.0)
                 * std::exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0) / 49.0)
                            -((9.0 * y - 2.0) * (9.0 * y - 2.0) / 10.0));
    Scalar term3 = (1.0 / 2.0)
                 * std::exp(-((9.0 * x - 7.0) * (9.0 * x - 7.0) / 4.0)
                            -((9.0 * y - 3.0) * (9.0 * y - 3.0) / 4.0));
    Scalar term4 = (1.0 / 5.0)
                 * std::exp(-((9.0 * x - 4.0) * (9.0 * x - 4.0))
                            -((9.0 * y - 7.0) * (9.0 * y - 7.0)));
    return term1 + term2 + term3 - term4;
}
Vector evalFrankeFunction(Matrix Points) {
    Vector f(Points.cols());
    for (Index i = 0; i < Points.cols(); ++i) {
        f(i) = FrankeFunction(Points(0, i), Points(1, i));
    }
    return f;
}

//---------------------------------------------------------
// 10) The main function that runs one "nu" and logs everything
void runForMu(Scalar nu) 
{
    // We want to produce "levels, N, symmetric_time, unsymmetric_time,
    // solver_time, anz, iterationsCG" in a single file.

    // Create output filename
    std::ostringstream filename;
    filename << "ResultsFranke_nu" << nu << "_exponential_stats1.txt";
    std::ofstream outFile(filename.str());
    if (!outFile.is_open()) {
        std::cerr << "Could not open file " << filename.str() << std::endl;
        return;
    }

    // We will still redirect standard output to file, but keep track
    // that we are printing important info. If you prefer, you can
    // print to std::cout and also to outFile. For simplicity, we do 
    // as you had: redirect to outFile:
    std::streambuf* coutbuf = std::cout.rdbuf(); // save old buf
    std::cout.rdbuf(outFile.rdbuf());            // redirect

    // 1) read data
    Matrix P1, P2, P3, P4, P5, P6, P7, P8, Peval; // P9, P10,
    readTXT("data/square01_uniform_grid_level1.txt",  P1, DIM);
    readTXT("data/square01_uniform_grid_level2.txt",  P2, DIM);
    readTXT("data/square01_uniform_grid_level3.txt",  P3, DIM);
    readTXT("data/square01_uniform_grid_level4.txt",  P4, DIM);
    readTXT("data/square01_uniform_grid_level5.txt",  P5, DIM);
    readTXT("data/square01_uniform_grid_level6.txt",  P6, DIM);
    readTXT("data/square01_uniform_grid_level7.txt",  P7, DIM);
    readTXT("data/square01_uniform_grid_level8.txt",  P8, DIM);
    // readTXT("data/square01_uniform_grid_level9.txt",  P9, DIM);
    // readTXT("data/square01_uniform_grid_level10.txt", P10, DIM);
    readTXT("data/uniform_vertices_UnitSquare_40k.txt", Peval, DIM);

    std::vector<Matrix> P_Matrices = {P1, P2, P3, P4, P5, P6, P7, P8}; // P9, P10
    int max_level = (int) P_Matrices.size();

    // 2) define parameters
    const Scalar eta            = 1. / DIM;
    const Index dtilde          = 5;
    const Scalar threshold_kernel = 1e-5;
    const Scalar threshold_aPost  = -1;
    const Scalar threshold_weights= 0;
    const Scalar mpole_deg       = 2 * (dtilde - 1);
    const std::string kernel_type= "exponential";

    std::cout << "eta:               " << eta << std::endl;
    std::cout << "dtilde:            " << dtilde << std::endl;
    std::cout << "threshold_kernel:  " << threshold_kernel << std::endl;
    std::cout << "mpole_deg:         " << mpole_deg << std::endl;
    std::cout << "kernel_type:       " << kernel_type << std::endl;
    std::cout << "nu:                " << nu << std::endl;

    // 3) fill-distances and residual initialization
    std::vector<Vector> residuals;
    Vector fill_distances(max_level);

    for (int i = 0; i < max_level; ++i) {
        Moments mom(P_Matrices[i], mpole_deg);
        SampletMoments samp_mom(P_Matrices[i], dtilde - 1);
        H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[i]);
        Vector minDistance = minDistanceVector(hst, P_Matrices[i]);
        Scalar h = minDistance.maxCoeff();
        fill_distances[i] = h;

        Vector resid = evalFrankeFunction(P_Matrices[i]);
        resid = hst.toClusterOrder(resid);
        resid = hst.sampletTransform(resid);
        residuals.push_back(resid);
    }

    // 4) Vectors for solution at each level
    std::vector<Vector> ALPHA;
    ALPHA.reserve(max_level);

    // 5) For the final output: we collect stats in these arrays
    std::vector<int>    levels;
    std::vector<int>    N;
    std::vector<double> symmetric_time;
    std::vector<double> unsymmetric_time;
    std::vector<double> solver_time;
    std::vector<double> anz;            // #triplets_apriori / n_points for the symmetric block
    std::vector<int>    iterationsCG;   // iteration count

    levels.reserve(max_level);
    N.reserve(max_level);
    symmetric_time.reserve(max_level);
    unsymmetric_time.reserve(max_level);
    solver_time.reserve(max_level);
    anz.reserve(max_level);
    iterationsCG.reserve(max_level);

    //-----------------------------------------------------
    // 6) solve level by level
    for (int l = 0; l < max_level; ++l) {
        std::cout << "---------------- LEVEL " << (l + 1) << " ----------------\n";
        Scalar h = fill_distances[l];
        int n_pts = P_Matrices[l].cols();
        std::cout << "Fill distance:        " << h << std::endl;
        std::cout << "Number of points:     " << n_pts << std::endl;

        // Prepare the cluster/moments for this level
        Moments mom(P_Matrices[l], mpole_deg);
        SampletMoments samp_mom(P_Matrices[l], dtilde - 1);
        H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[l]);

        // We accumulate the unsymmetric time across blocks [0..l-1]
        double total_unsymm_time = 0.0;

        // 6a) substract the unsymmetric blocks from the residual
        for (int j = 0; j < l; ++j) {
            Scalar sigma_B = nu * fill_distances[j];
            CovarianceKernel kernel_B(kernel_type, sigma_B);

            // We'll measure the compression for B block
            CompressionResult crB = UnsymmetricCompressorWithStats(
                mom, samp_mom, hst, kernel_B, eta, threshold_kernel,
                threshold_aPost, mpole_deg, dtilde, P_Matrices[l], P_Matrices[j]);

            // sum the times for unsymmetric
            double sumBlock = crB.stats.time_planner 
                            + crB.stats.time_compressor 
                            + crB.stats.time_apriori;
            total_unsymm_time += sumBlock;

            // update residual
            Tictoc localT; 
            localT.tic();
            residuals[l] -= crB.matrix * ALPHA[j];
            localT.toc("time residual update (mat-vec) = ");
        }

        // 6b) now compress the symmetric block at level l
        Scalar sigma_l = nu * fill_distances[l];
        CovarianceKernel kernel_l(kernel_type, sigma_l);

        CompressionResult crA = SymmetricCompressorWithStats(
            mom, samp_mom, hst, kernel_l, eta, threshold_kernel,
            threshold_aPost, mpole_deg, dtilde, P_Matrices[l]);

        double sumSymm = crA.stats.time_planner
                       + crA.stats.time_compressor
                       + crA.stats.time_apriori;

        // store #triplets/n_pts for the *symmetric* block
        double anz_symmetric = double(crA.stats.triplets_apriori)/double(n_pts);

        // 6c) Solve the system
        Vector rhs = residuals[l];
        SolverResult sres = solveSystemWithStats(crA.matrix, rhs, 
                                                 "ConjugateGradient", 1e-6);
        // store alpha
        ALPHA.push_back(sres.solution);

        // fill stats in our vectors
        levels.push_back(l+1);
        N.push_back(n_pts);
        symmetric_time.push_back(sumSymm);
        unsymmetric_time.push_back(total_unsymm_time);
        solver_time.push_back(sres.stats.solver_time);
        anz.push_back(std::round(anz_symmetric));
        iterationsCG.push_back(sres.stats.iterations);
    }

    //-----------------------------------------------------
    // Evaluate on Peval
    // Moments mom_eval(Peval, mpole_deg);
    // SampletMoments samp_mom_eval(Peval, dtilde - 1);
    // H2SampletTree<ClusterTree> hst_eval(mom_eval, samp_mom_eval, 0, Peval);
    // Vector exact_sol = evalFrankeFunction(Peval);

    // EvaluateWithStats(mom_eval, samp_mom_eval, hst_eval, kernel_type, 
    //          P_Matrices, Peval, ALPHA, fill_distances, max_level, 
    //          nu, eta, threshold_kernel, threshold_aPost, 
    //          mpole_deg, dtilde, exact_sol, hst_eval, "");



/////////////////////////// Save results
    auto printVector = [&](const std::string& name, const auto& vec) {
        std::cout << name << " = [";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i + 1 < vec.size()) std::cout << ", ";
        }
        std::cout << "];\n";
    };

    printVector("levels", levels);
    printVector("N", N);
    printVector("symmetric_time", symmetric_time);
    printVector("unsymmetric_time", unsymmetric_time);
    printVector("solver_time", solver_time);
    printVector("anz", anz);
    printVector("iterationsCG", iterationsCG);

    // restore std::cout
    std::cout.rdbuf(coutbuf);
}

//---------------------------------------------------------
// 11) Main
int main() {
    // For demonstration, we run for a few 'nu' values:
    std::vector<Scalar> nus = {1};
    for (Scalar nu : nus) {
        runForMu(nu);
    }
    return 0;
}