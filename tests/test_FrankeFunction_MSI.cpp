#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>

#include <Eigen/Sparse>

#include "../MSI/src/Evaluation.h"
#include "../MSI/src/Solver.h"
#include "../MSI/src/Compression.h"

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;

using namespace FMCA;

// Franke Function
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

Vector evalFrankeFunction(const Matrix& Points) {
    Vector f(Points.cols());
    for (Index i = 0; i < Points.cols(); ++i) {
        f(i) = FrankeFunction(Points(0, i), Points(1, i));
    }
    return f;
}

// Function to generate uniform grid points on unit square
Matrix generateUniformGrid(int n) {
    int gridSize = std::sqrt(n);
    Matrix P(DIM, gridSize * gridSize);
    
    Scalar h = 1.0 / (gridSize - 1);
    int idx = 0;
    
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            P(0, idx) = i * h;
            P(1, idx) = j * h;
            idx++;
        }
    }
    
    return P;
}

// Function to calculate minimum distance between points
Vector minDistanceVector(const H2SampletTree<ClusterTree>& hst, const Matrix& P) {
    // Simple implementation for unit test - for a uniform grid, 
    // the min distance is just the grid spacing
    int n = P.cols();
    int gridSize = std::sqrt(n);
    Scalar h = 1.0 / (gridSize - 1);
    
    Vector minDistance(n);
    minDistance.setConstant(h);
    
    return minDistance;
}

void runMultigridTest(Scalar nu) {
    std::cout << "======== Running Multigrid Test with nu = " << nu << " ========" << std::endl;
    
    // 1) Generate uniform grid points instead of reading them
    std::vector<int> gridSizes = {9, 25, 81, 289, 1089, 4225, 16641, 66049};
    int max_level = gridSizes.size();
    
    std::vector<Matrix> P_Matrices;
    for (int size : gridSizes) {
        P_Matrices.push_back(generateUniformGrid(size));
    }
    
    // Generate evaluation points - 200x200 grid (40,000 points)
    Matrix Peval = generateUniformGrid(40000);
    
    // 2) Define parameters
    const Scalar eta = 1. / DIM;
    const Index dtilde = 5;
    const Scalar threshold_kernel = 1e-12;
    const Scalar threshold_aPost = 1e-8;
    const Scalar mpole_deg = 2 * (dtilde - 1);
    const std::string kernel_type = "matern52";
    
    std::cout << "Parameters:" << std::endl;
    std::cout << "- eta: " << eta << std::endl;
    std::cout << "- dtilde: " << dtilde << std::endl;
    std::cout << "- threshold_kernel: " << threshold_kernel << std::endl;
    std::cout << "- mpole_deg: " << mpole_deg << std::endl;
    std::cout << "- kernel_type: " << kernel_type << std::endl;
    std::cout << "- nu: " << nu << std::endl;
    
    // 3) Calculate fill-distances and residuals
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
    
    // 5) For the final output: collect stats
    std::vector<int> levels;
    std::vector<int> N;
    std::vector<Scalar> symmetric_time;
    std::vector<Scalar> unsymmetric_time;
    std::vector<Scalar> solver_time;
    std::vector<Scalar> anz;
    std::vector<int> iterationsCG;
    
    levels.reserve(max_level);
    N.reserve(max_level);
    symmetric_time.reserve(max_level);
    unsymmetric_time.reserve(max_level);
    solver_time.reserve(max_level);
    anz.reserve(max_level);
    iterationsCG.reserve(max_level);
    
    // Solve level by level
    for (int l = 0; l < max_level; ++l) {
        std::cout << "-------- LEVEL " << (l + 1) << " --------" << std::endl;
        Scalar h = fill_distances[l];
        int n_pts = P_Matrices[l].cols();
        std::cout << "Fill distance: " << h << std::endl;
        std::cout << "Number of points: " << n_pts << std::endl;
        
        // Prepare the cluster/moments for this level
        Moments mom(P_Matrices[l], mpole_deg);
        SampletMoments samp_mom(P_Matrices[l], dtilde - 1);
        H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[l]);
        
        // Track unsymmetric time
        Scalar total_unsymm_time = 0.0;
        
        // 6a) Subtract the unsymmetric blocks from the residual
        for (int j = 0; j < l; ++j) {
            Scalar sigma_B = nu * fill_distances[j];
            CovarianceKernel kernel_B(kernel_type, sigma_B);
            
            // Use your function from Compression.h
            CompressionResult crB = UnsymmetricCompressorWithStats(
                mom, samp_mom, hst, kernel_B, eta, threshold_kernel,
                threshold_aPost, mpole_deg, dtilde, P_Matrices[l], P_Matrices[j]);
            crB.matrix *= std::pow(sigma_B, -DIM);
            
            // Sum the times for unsymmetric
            Scalar sumBlock = crB.stats.time_planner 
                            + crB.stats.time_compressor 
                            + crB.stats.time_apost;
            total_unsymm_time += sumBlock;
            
            // Update residual
            residuals[l] -= crB.matrix * ALPHA[j];
        }
        
        // Compress the symmetric block at level l
        Scalar sigma_l = nu * fill_distances[l];
        CovarianceKernel kernel_l(kernel_type, sigma_l);
        
        // Use your function from Compression.h
        CompressionResult crA = SymmetricCompressorWithStats(
            mom, samp_mom, hst, kernel_l, eta, threshold_kernel,
            threshold_aPost, mpole_deg, dtilde, P_Matrices[l]);
        crA.matrix *= std::pow(sigma_l, -DIM);
        
        Scalar sumSymm = crA.stats.time_planner
                       + crA.stats.time_compressor
                       + crA.stats.time_apost;
        
        // Store #triplets/n_pts for the symmetric block
        Scalar anz_symmetric = Scalar(crA.stats.triplets_apriori) / Scalar(n_pts);
        
        // 6c) Solve the system using your function from Solver.h
        Vector rhs = residuals[l];
        SolverResult sres = solveSystemWithStats(
            crA.matrix, rhs, "ConjugateGradient", 1e-6);
        
        // Store solution
        ALPHA.push_back(sres.solution);
        
        // Fill stats in vectors
        levels.push_back(l + 1);
        N.push_back(n_pts);
        symmetric_time.push_back(sumSymm);
        unsymmetric_time.push_back(total_unsymm_time);
        solver_time.push_back(sres.stats.solver_time);
        anz.push_back(anz_symmetric);
        iterationsCG.push_back(sres.stats.iterations);
        
        // Print some verification info for this level
        std::cout << "Level " << (l + 1) << " complete:" << std::endl;
        std::cout << "- Symmetric compression time: " << sumSymm << " s" << std::endl;
        std::cout << "- Unsymmetric compression time: " << total_unsymm_time << " s" << std::endl;
        std::cout << "- Solver time: " << sres.stats.solver_time << " s" << std::endl;
        std::cout << "- CG iterations: " << sres.stats.iterations << std::endl;
        std::cout << "- ANZ: " << anz_symmetric << std::endl;
    }
    
    // Evaluate solution on evaluation points
    std::cout << "\nEvaluating solution on " << Peval.cols() << " points..." << std::endl;
    
    // Setup for evaluation (assuming you have an Evaluation.h with EvaluateWithStats)
    Moments mom_eval(Peval, mpole_deg);
    SampletMoments samp_mom_eval(Peval, dtilde - 1);
    H2SampletTree<ClusterTree> hst_eval(mom_eval, samp_mom_eval, 0, Peval);
    Vector exact_sol = evalFrankeFunction(Peval);
    
    // Evaluate the solution using Evaluation.h
    EvaluationResult eval_res = EvaluateWithStats(
        mom_eval, samp_mom_eval, hst_eval, kernel_type, 
        P_Matrices, Peval, ALPHA, fill_distances, max_level,
        nu, eta, threshold_kernel, threshold_aPost,
        mpole_deg, dtilde, exact_sol, "", 2);
    
    // Report error
    std::cout << "Evaluation complete:" << std::endl;
    std::cout << "- L2 Error: " << eval_res.l2_error << std::endl;
    std::cout << "- Linf Error: " << eval_res.linf_error << std::endl;
    std::cout << "- Evaluation time: " << eval_res.stats.eval_time << " s" << std::endl;
    
    std::cout << "\n======== Results Summary ========" << std::endl;
    std::cout << std::left 
              << std::setw(10) << "Level" 
              << std::setw(10) << "N" 
              << std::setw(15) << "SymTime" 
              << std::setw(15) << "UnsymTime" 
              << std::setw(15) << "SolverTime" 
              << std::setw(15) << "ANZ" 
              << std::setw(10) << "IterCG" 
              << std::endl;
    
    for (size_t i = 0; i < levels.size(); ++i) {
        std::cout << std::left
                  << std::setw(10) << levels[i]
                  << std::setw(10) << N[i]
                  << std::setw(15) << std::fixed << std::setprecision(6) << symmetric_time[i]
                  << std::setw(15) << std::fixed << std::setprecision(6) << unsymmetric_time[i]
                  << std::setw(15) << std::fixed << std::setprecision(6) << solver_time[i]
                  << std::setw(15) << anz[i]
                  << std::setw(10) << iterationsCG[i]
                  << std::endl;
    }
    
    // Print MATLAB-compatible output
    std::cout << "\nVector Results:" << std::endl;
    
    auto printVector = [](const std::string& name, const auto& vec) {
        std::cout << name << " = [";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i + 1 < vec.size()) std::cout << ", ";
        }
        std::cout << "];" << std::endl;
    };
    
    printVector("levels", levels);
    printVector("N", N);
    printVector("symmetric_time", symmetric_time);
    printVector("unsymmetric_time", unsymmetric_time);
    printVector("solver_time", solver_time);
    printVector("anz", anz);
    printVector("iterationsCG", iterationsCG);
}

int main() {
    // Test different nu values
    std::vector<Scalar> nus = {1.0};
    
    for (Scalar nu : nus) {
        runMultigridTest(nu);
        std::cout << "\n\n";
    }
    
    return 0;
}