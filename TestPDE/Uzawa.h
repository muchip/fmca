#include <iostream>
#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include </opt/homebrew/Cellar/metis/5.1.0/include/metis.h>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/MetisSupport>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/Macros.h"

using EigenCholesky =
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                       Eigen::MetisOrdering<int>>;

// Solver
FMCA::Vector solveSPD(const Eigen::SparseMatrix<double>& S, const FMCA::Vector& b){   
    EigenCholesky choleskySolver;
    choleskySolver.compute(S);
    if (choleskySolver.info() != Eigen::Success) {
        throw std::runtime_error("Decomposition failed");
    }    
    return choleskySolver.solve(b);
}        

FMCA::Vector UzawaAlgorithm(const Eigen::SparseMatrix<double>& S, const Eigen::SparseMatrix<double>& A, const FMCA::Vector& F, const FMCA::Vector& G) {
    FMCA::Vector lambda = FMCA::Vector::Zero(G.size()); // Initialize lambda with zeros
    FMCA::Vector U = solveSPD(S, F - A * lambda);
    FMCA::Vector r_lambda = A.transpose() * U - G;
    
    int maxIterations = 1000;
    double tolerance = 1e-6;

    for (int k = 0; k < maxIterations; ++k) {
        FMCA::Vector p_U = solveSPD(S, A * r_lambda);
        double num = r_lambda.dot(r_lambda);
        double denom = (A * r_lambda).dot(p_U);
        if (std::abs(denom) < 1e-10) { // Prevent division by very small number
            std::cerr << "Denominator too small. Possible numerical instability." << std::endl;
            break;
        }
        double alpha = num / denom;
        lambda += alpha * r_lambda;
        U -= alpha * p_U;
        FMCA::Vector new_r_lambda = A.transpose() * U - G;
        if (new_r_lambda.norm() < tolerance) {
            std::cout << "Converged in " << k << " iterations." << std::endl;
            break;
        }
        if (std::abs(num) < 1e-10) { // Prevent division by very small number
            std::cerr << "Denominator too small. Possible numerical instability." << std::endl;
            break;
        }
        double beta = new_r_lambda.dot(new_r_lambda) / num;
        r_lambda = new_r_lambda + beta * r_lambda;
    }
    return U;
}


// FMCA::Vector UzawaAlgorithm(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, const FMCA::Vector& b1, const FMCA::Vector& b2) {
//     Eigen::SparseMatrix<double> Bt = B.transpose();
//     Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cgA;
//     cgA.compute(A);

//     // Initial guess for x2, can be zero or any other appropriate value
//     FMCA::Vector x2 = FMCA::Vector::Zero(B.cols());
//     FMCA::Vector x1 = cgA.solve(b1 - B * x2); // Compute x1 = A^-1(b1 - Bx2)

//     // Compute initial residual
//     FMCA::Vector r2 = Bt * x1 - b2;
//     FMCA::Vector p2 = r2; // Initial search direction

//     double tolerance = 1e-8; // Tolerance for convergence
//     int maxIterations = 1000; // Maximum number of iterations
//     int iteration = 0;

//     while (r2.norm() > tolerance && iteration < maxIterations) {
//         FMCA::Vector p1 = cgA.solve(B * p2); // Intermediate result
//         FMCA::Vector a2 = Bt * p1; // Compute a2 = B^*A^-1Bp2 = B^*p1

//         double alpha = p2.dot(a2) / p2.dot(r2); // Compute scaling factor

//         // Update x2, r2, and x1
//         x2 += alpha * p2;
//         r2 -= alpha * a2;
//         x1 -= alpha * p1;

//         double beta = r2.dot(a2) / p2.dot(a2); // Compute Gram-Schmidt coefficient
//         p2 = r2 - beta * p2; // Update search direction

//         iteration++;
//     }

//     // Depending on your needs, you might want to combine x1 and x2 into a single vector
//     // or return them separately. This example just returns x2 for simplicity.
//     return x1;
// }

