// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/time.h>
////////////////////////////////////////////////////////////////////////////////
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
    for (unsigned int n : {1e3, 5e3, 1e4, 5e4, 1e5}) {
      Eigen::MatrixXd P = 0.5 * (Eigen::MatrixXd::Random(dim, n).array() + 1);
      SampletCRS S =
          sampletMatrixGenerator(P, mp_deg, dtilde, eta, threshold, ridgep);
      SampletCRS invS = S;
      output_file << std::scientific << std::setprecision(2);
      output_file << std::setw(9) << n << "  ";
      output_file << std::setw(9) << mp_deg << "  ";
      output_file << std::setw(9) << dtilde << "  ";
      output_file << std::setw(9) << eta << "  ";
      output_file << std::setw(9) << ridgep << "  ";
      output_file << std::setw(9) << S.comp_err << "  ";
      output_file << std::setw(9) << S.pnnz() << "  ";
      output_file << std::setw(9) << S.comp_time << "  ";
      output_file << std::setw(9) << S.unif_const << "  ";
      output_file << std::flush;
      //////////////////////////////////////////////////////////////////////////////
      // PARDISO BLOCK
      //////////////////////////////////////////////////////////////////////////////
      std::cout << "\n\nentering pardiso block\n" << std::flush;
      std::printf("ia=%p ja=%p a=%p n=%i nnz=%i\n", invS.ia.data(),
                  invS.ja.data(), invS.a.data(), n, invS.ia[n]);
      std::cout << std::flush;
      T.tic();
      pardiso_interface(invS.ia.data(), invS.ja.data(), invS.a.data(), n);
      double tpardiso = T.toc("wall time pardiso: ");
      std::cout << std::string(75, '=') << std::endl;
      //////////////////////////////////////////////////////////////////////////////
      // error checking
      //////////////////////////////////////////////////////////////////////////////
      Eigen::MatrixXd x(n, 10), y(n, 10), z(n, 10);
      x.setRandom();
      Eigen::VectorXd nrms = x.colwise().norm();
      for (auto i = 0; i < x.cols(); ++i)
        x.col(i) /= nrms(i);
      y.setZero();
      z.setZero();
      // compute y = S * x
      for (auto i = 0; i < n; ++i)
        for (auto j = S.ia[i]; j < S.ia[i + 1]; ++j) {
          y.row(i) += S.a[j] * x.row(S.ja[j]);
          if (i != S.ja[j])
            y.row(S.ja[j]) += S.a[j] * x.row(i);
        }
      // compute z = invS * y;
      for (auto i = 0; i < n; ++i)
        for (auto j = invS.ia[i]; j < invS.ia[i + 1]; ++j) {
          z.row(i) += invS.a[j] * y.row(invS.ja[j]);
          if (i != invS.ja[j])
            z.row(invS.ja[j]) += invS.a[j] * y.row(i);
        }
      double err = (z - x).norm() / x.norm();
      std::cout << "inverse error:               " << err
                << std::endl;
      output_file << std::setw(9) << err << "  ";
      output_file << std::setw(9) << tpardiso << "  ";
      output_file << std::endl;
    }
  output_file.close();
  return 0;
}
