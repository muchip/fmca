// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
#include <sys/time.h>

#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include "sampletMatrixGenerator.h"
////////////////////////////////////////////////////////////////////////////////
#include "formatted_multiplication.h"
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
  const double threshold = 1e-5;
  const double ridgep = 0;
  const unsigned int n = atoi(argv[1]);
  std::vector<double> data_p;
  Tictoc T;
  std::fstream output_file;
  output_file.open("output_Multiplication_FINAL_" + std::to_string(dim) +
                       "D.txt",
                   std::ios::out | std::ios::app);
  if (output_file.tellg() < 1) {
    output_file << "     npts  "
                << "    mpdeg  "
                << "   dtilde  "
                << "      eta  "
                << "   ridgep  "
                << " comp_err  "
                << "     pnnz  "
                << "comp_time  "
                << "unif_cnst  "
                << " mult_err  "
                << "   opnorm  "
                << " mult_tim  " << std::endl;
  }
  Eigen::MatrixXd P = 0.5 * (Eigen::MatrixXd::Random(dim, n).array() + 1);
  largeSparse S = sampletMatrixGenerator(P, mp_deg, dtilde, eta, threshold,
                                         ridgep, &data_p);
  output_file << std::scientific << std::setprecision(2);
  output_file << std::setw(9) << n << "  ";
  output_file << std::setw(9) << mp_deg << "  ";
  output_file << std::setw(9) << dtilde << "  ";
  output_file << std::setw(9) << eta << "  ";
  output_file << std::setw(9) << ridgep << "  ";
  output_file << std::setw(9) << data_p[3] << "  ";
  output_file << std::setw(9) << data_p[2] << "  ";
  output_file << std::setw(9) << data_p[1] << "  ";
  output_file << std::setw(9) << data_p[0] << "  ";
  output_file << std::flush;
  S.makeCompressed();
  largeSparse A = S.selfadjointView<Eigen::Upper>();
  S = A;
  largeSparse S2 = S;
  for (auto i = 0; i < A.nonZeros(); ++i)
    A.valuePtr()[i] *= (1. + (0.1 * rand()) / double(RAND_MAX));
  double opnorm = 0;
  {
    Eigen::VectorXd x = Eigen::VectorXd::Random(S.cols());
    x /= x.norm();
    for (auto i = 0; i < 100; ++i) {
      x = S * x;
      opnorm = x.norm();
      x /= opnorm;
    }
    std::cout << "op. norm (100its of power it): " << opnorm << std::endl;
  }
  memset(S2.valuePtr(), 0, S2.nonZeros() * sizeof(double));
  T.tic();
  formatted_sparse_multiplication(S2, S, A);
  const double mtime = T.toc("multiplication time: ");
  Eigen::MatrixXd rdm = Eigen::MatrixXd::Random(n, 10);
  double err = 0;
  Eigen::MatrixXd y = S2 * rdm;
  Eigen::MatrixXd y_ref = S * (A * rdm).eval();
  err = (y - y_ref).norm() / y_ref.norm();
  std::cout << "multiplication error: " << err << std::endl;
  output_file << std::setw(9) << err << "  ";
  output_file << std::setw(9) << opnorm << "  ";
  output_file << std::setw(9) << mtime << "  ";
  output_file << std::endl;
  output_file.close();
  return 0;
}
