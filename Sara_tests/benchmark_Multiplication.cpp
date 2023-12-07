// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
#include <sys/time.h>

#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <../FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
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
  Eigen::MatrixXd P = 0.5 * (Eigen::MatrixXd::Random(dim, n).array() + 1);
  largeSparse S = sampletMatrixGenerator(P, mp_deg, dtilde, eta, threshold,
                                         ridgep, &data_p);
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
  return 0;
}
