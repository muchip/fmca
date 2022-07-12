// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/time.h>
////////////////////////////////////////////////////////////////////////////////
#include "../FMCA/src/util/SparseMatrix.h"
#include "pardiso_interface.h"
#include "sampletMatrixGenerator.h"
////////////////////////////////////////////////////////////////////////////////
class Tictoc {
public:
  void tic(void) { gettimeofday(&start, NULL); }
  double toc(void) {
    gettimeofday(&stop, NULL);
    double dtime =
        stop.tv_sec - start.tv_sec + 1e-6 * (stop.tv_usec - start.tv_usec);
    return dtime;
  }
  double toc(const std::string &message) {
    gettimeofday(&stop, NULL);
    double dtime =
        stop.tv_sec - start.tv_sec + 1e-6 * (stop.tv_usec - start.tv_usec);
    std::cout << message << " " << dtime << "sec.\n";
    return dtime;
  }

private:
  struct timeval start; /* variables for timing */
  struct timeval stop;
};

int main(int argc, char *argv[]) {
  const unsigned int dtilde = 4;
  const double eta = 0.8;
  const unsigned int mp_deg = 6;
  const unsigned int dim = 2;
  const double threshold = 0;
  Tictoc T;
  std::fstream output_file;

  output_file.open("output_Inversion_" + std::to_string(dim) + "D.txt",
                   std::ios::out);
  output_file << "     npts  "
              << "    mpdeg  "
              << "   dtilde  "
              << "      eta  "
              << "   ridgep  "
              << " comp_err  "
              << "     pnnz  "
              << "comp_time  "
              << "unif_cnst  "
              << "  inv_err  "
              << "pardiso_t  " << std::endl;
  for (double ridgep : {1e-8, 1e-6, 1e-4, 1e-2, 1.})
    for (unsigned int n : {1e3, 5e3, 1e4}) {
      Eigen::MatrixXd P = 0.5 * (Eigen::MatrixXd::Random(dim, n).array() + 1);
      FMCA::SparseMatrix<double> S(n, n);
      FMCA::SparseMatrix<double> X(n, n);
      FMCA::SparseMatrix<double> Xold(n, n);
      FMCA::SparseMatrix<double> I2(n, n);
      FMCA::SparseMatrix<double> ImXS(n, n);
      Eigen::MatrixXd randFilter = Eigen::MatrixXd::Random(S.rows(), 20);
      T.tic();
      {
        SampletCRS S2 =
            sampletMatrixGenerator(P, mp_deg, dtilde, eta, threshold, ridgep);
        auto trips = S2.toTriplets();
        S.setFromTriplets(trips.begin(), trips.end());
        S.mirrorUpper();
      }
      T.toc("Wall time for generating matrix: ");
      double lambda_max = 0;
      {
        Eigen::MatrixXd x = Eigen::VectorXd::Random(S.cols());
        x /= x.norm();
        for (auto i = 0; i < 20; ++i) {
          x = S * x;
          lambda_max = x.norm();
          x /= lambda_max;
        }
        std::cout << "lambda_max (est by 20its of power it): " << lambda_max
                  << std::endl;
      }
      double err = 10;
      double err_old = 10;
      double alpha = 1. / lambda_max;
      X.setIdentity();
      X.scale(alpha);
      std::cout << "initial guess: "
                << ((X * (S * randFilter)) - randFilter).norm() /
                       randFilter.norm()
                << std::endl;
      I2.setIdentity().scale(2);
      T.tic();
      for (auto inner_iter = 0; inner_iter < 100; ++inner_iter) {
        Xold = X;
        //FMCA::SparseMatrix<double>::formatted_ABT(X, S, Xold);
        //ImXS = I2 - X;
        //FMCA::SparseMatrix<double>::formatted_ABT(X, S, ImXS);
        X = I2 * X - FMCA::SparseMatrix<double>::formatted_BABT(S, S, X);
        //    X = X * (I2 - (S  * X));
        X.compress(1e-8/n);
        X.symmetrize();
        err_old = err;
        err = ((X * (S * randFilter)) - randFilter).norm() / randFilter.norm();
        std::cout << "anz: " << X.nnz() / S.rows() << " err: " << err
                  << std::endl;
        if (err > err_old) {
          X = Xold;
          break;
        }
      }
      std::cout << err_old << std::endl;
      T.toc("time inner: ");
    }
  output_file.close();
  return 0;
}
