#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <cmath>  // Include cmath for sqrt function
#include <vector> // Include vector for std::vector
#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
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
#include "../FMCA/src/util/Plotter3D.h"
#include "../FMCA/src/util/Tictoc.h"
#include "read_files_txt.h"
#include "MultiGridFunctions.h"

#define DIM 3

// Define types based on Eigen library
using Scalar = double;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluatorKernel = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluatorKernel = FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using EigenCholesky = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper, Eigen::MetisOrdering<int>>;

using namespace FMCA;

Scalar DistanceCenterofMass(Scalar x, Scalar y, Scalar z, Scalar x_m, Scalar y_m, Scalar z_m) {
    return std::sqrt((x - x_m) * (x - x_m) + (y - y_m) * (y - y_m) + (z - z_m) * (z - z_m));
}

Vector evalFunction(const Matrix& Points, Scalar x_m, Scalar y_m, Scalar z_m) {
    Vector f(Points.cols());
    for (Eigen::Index i = 0; i < Points.cols(); ++i) {
        f(i) = DistanceCenterofMass(Points(0, i), Points(1, i), Points(2, i), x_m, y_m, z_m);
    }
    return f;
}

Scalar averageCoordinates(const Eigen::VectorXd& Points) {
    return Points.mean();
}

int main() {
    Tictoc T;
    ///////////////////////////////// Inputs: points + maximum level
    Matrix P1;
    readTXT("data/bunny_level0.txt", P1, DIM);
    std::cout << "Cardinality P1      " << P1.cols() << std::endl;
    Matrix P2;
    readTXT("data/bunny_level1.txt", P2, DIM);
    std::cout << "Cardinality P2      " << P2.cols() << std::endl;
    Matrix P3;
    readTXT("data/bunny_level2.txt", P3, DIM);
    std::cout << "Cardinality P3      " << P3.cols() << std::endl;
    Matrix P4;
    readTXT("data/bunny_level3.txt", P4, DIM);
    std::cout << "Cardinality P4      " << P4.cols() << std::endl;
    Matrix P5;
    readTXT("data/bunny_level4.txt", P5, DIM);
    std::cout << "Cardinality P5      " << P5.cols() << std::endl;
    Matrix P6;
    readTXT("data/bunny_level5.txt", P6, DIM);
    std::cout << "Cardinality P6      " << P6.cols() << std::endl;

    Scalar x_m = averageCoordinates(P6.row(0));
    Scalar y_m = averageCoordinates(P6.row(1));
    Scalar z_m = averageCoordinates(P6.row(2));
    std::cout << "Center of mass = (" << x_m << ", " << y_m << ", " << z_m << ")" << std::endl;

    Matrix Peval;
    readTXT("data/bunny_eval.txt", Peval, DIM);
    std::cout << "Cardinality Peval   " << Peval.cols() << std::endl;

    ///////////////////////////////// Nested cardinality of points
    std::vector<Matrix> P_Matrices = {P1, P2, P3, P4, P5, P6};
    int max_level = P_Matrices.size();

    ///////////////////////////////// Parameters
    const Scalar eta = 1. / DIM;
    const Eigen::Index dtilde = 3;
    const Scalar threshold_kernel = 1e-4;
    const Scalar threshold_weights = 0;
    const Scalar mpole_deg = 2 * (dtilde - 1);
    const std::string kernel_type = "matern32";
    const Scalar nu = 0.6;

    std::cout << "eta                 " << eta << std::endl;
    std::cout << "dtilde              " << dtilde << std::endl;
    std::cout << "threshold_kernel    " << threshold_kernel << std::endl;
    std::cout << "mpole_deg           " << mpole_deg << std::endl;
    std::cout << "kernel_type         " << kernel_type << std::endl;
    std::cout << "nu                  " << nu << std::endl;

    ///////////////////////////////// Rhs
    std::vector<Vector> residuals;

    ///////////////////////////////// Fill Distances and Residuals
    Vector fill_distances(max_level);
    for (Eigen::Index i = 0; i < max_level; ++i) {
        const Moments mom(P_Matrices[i], mpole_deg);
        const SampletMoments samp_mom(P_Matrices[i], dtilde - 1);
        const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[i]);
        Vector minDistance = minDistanceVector(hst, P_Matrices[i]);
        Scalar h = minDistance.maxCoeff();
        fill_distances[i] = h;

        Vector residual = evalFunction(P_Matrices[i], x_m, y_m, z_m);
        residual = hst.toClusterOrder(residual);
        residual = hst.sampletTransform(residual);
        residuals.push_back(residual);
    }
    std::cout << fill_distances << std::endl;

    ///////////////////////////////// Coeffs Initialization
    std::vector<Vector> ALPHA;
    std::string base_filename_residuals = "Plots/ResidualsBunny";

    ///////////////////////////////// Resolution --> Scheme = Matricial form
    for (Eigen::Index l = 0; l < max_level; ++l) {
        std::cout << "-------- Level " << l + 1 << " --------" << std::endl;
        std::cout << "Fill distance                      " << fill_distances[l] << std::endl;
        int n_pts = P_Matrices[l].cols();
        std::cout << "Number of points                   " << n_pts << std::endl;

        const Moments mom(P_Matrices[l], mpole_deg);
        const SampletMoments samp_mom(P_Matrices[l], dtilde - 1);
        const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[l]);

        for (Eigen::Index j = 0; j < l; ++j) {
            int n_pts_B = P_Matrices[j].cols();
            FMCA::Scalar sigma_B = nu * fill_distances[j];
            const CovarianceKernel kernel_funtion_B(kernel_type, sigma_B);
            T.tic();
            // Scope block for B_comp
            {
                Eigen::SparseMatrix<double> B_comp = UnsymmetricCompressor(
                    mom, samp_mom, hst, kernel_funtion_B, eta, threshold_kernel,
                    mpole_deg, dtilde, P_Matrices[l], P_Matrices[j]);
                Scalar compression_time_unsymmetric = T.toc();
                std::cout << "Compression time                   " << compression_time_unsymmetric << std::endl;
                residuals[l] -= B_comp * ALPHA[j];
                std::cout << "Residuals updated" << std::endl;
            }  // B_comp goes out of scope and is destroyed here
        }

        ///////////////////////////////// Plot the residuals
        {
            Vector residual_natural_basis = hst.inverseSampletTransform(residuals[l]);
            residual_natural_basis = hst.toNaturalOrder(residual_natural_basis);
            Vector abs_residual(residual_natural_basis.size());
            for (Eigen::Index i = 0; i < residual_natural_basis.size(); ++i) {
                abs_residual[i] = std::log(std::abs(residual_natural_basis[i]));
            }
            // Create the filename for the residual
            std::ostringstream oss;
            oss << base_filename_residuals << "_level_" << l << ".vtk";
            std::string filename_residuals = oss.str();
            Plotter3D plotter;
            plotter.plotFunction(filename_residuals, P_Matrices[l], abs_residual);
        }

        Scalar sigma = nu * fill_distances[l];
        const CovarianceKernel kernel_funtion(kernel_type, sigma);
        T.tic();
        // Scope block for A_comp
        {
            Eigen::SparseMatrix<double> A_comp = SymmetricCompressor(
                mom, samp_mom, hst, kernel_funtion, eta, threshold_kernel, mpole_deg,
                dtilde, P_Matrices[l]);
            Scalar compression_time = T.toc();
            std::cout << "Compression time                   " << compression_time << std::endl;

            ///////////////////////////////// Solve the linear system
            Vector rhs = residuals[l];
            Vector alpha = solveSystem(A_comp, rhs, "ConjugateGradientwithPreconditioner", 1e-4);
            ALPHA.push_back(alpha);
        }  // A_comp goes out of scope and is destroyed here
    }

    ///////////////////////////////// Final Evaluation
    const Moments mom(Peval, mpole_deg);
    const SampletMoments samp_mom(Peval, dtilde - 1);
    const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, Peval);
    Vector exact_sol = evalFunction(Peval, x_m, y_m, z_m);
    Vector solution = Evaluate(
        mom, samp_mom, hst, kernel_type, P_Matrices, Peval, ALPHA, fill_distances,
        max_level, nu, eta, threshold_kernel, mpole_deg, dtilde, exact_sol, hst,
        "Plots/SolutionBunny"); // Plots/SolutionBunny

    return 0;
}
