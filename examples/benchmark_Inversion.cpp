// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
#include <sys/time.h>

#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include "../FMCA/src/util/print2file.h"
#include "pardiso_interface.h"
#include "recInv.h"
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
  const double ridgep = atof(argv[2]);
  const unsigned int n = atoi(argv[1]);
  std::vector<double> data_p;
  Tictoc T;
  std::fstream output_file;

  output_file.open("output_Inversion_" + std::to_string(dim) + "D.txt",
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
                << "  inv_err  "
                << "pardiso_t  " << std::endl;
  }
  Eigen::MatrixXd P = 0.5 * (Eigen::MatrixXd::Random(dim, n).array() + 1);
  largeSparse S = sampletMatrixGenerator(P, mp_deg, dtilde, eta, threshold,
                                         ridgep, &data_p);
  Eigen::SparseMatrix<double, Eigen::RowMajor> invS = S;
  invS.makeCompressed();
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
  //////////////////////////////////////////////////////////////////////////////
  // PARDISO BLOCK
  //////////////////////////////////////////////////////////////////////////////
  std::cout << "\n\nentering pardiso block\n" << std::flush;
  std::printf("ia=%p ja=%p a=%p n=%i nnz=%i\n", invS.outerIndexPtr(),
              invS.innerIndexPtr(), invS.valuePtr(), n,
              invS.outerIndexPtr()[n]);
  std::cout << std::flush;
  T.tic();
#if 0
  pardiso_interface(invS.outerIndexPtr(), invS.innerIndexPtr(), invS.valuePtr(),
                    n);
#else
  invS = recInv(S, n / 4);
#endif
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
  y = S.selfadjointView<Eigen::Upper>() * x;
  z = invS.selfadjointView<Eigen::Upper>() * y;
  double err = (z - x).norm() / x.norm();
  std::cout << "inverse error:               " << err << std::endl;
  output_file << std::setw(9) << err << "  ";
  output_file << std::setw(9) << tpardiso << "  ";
  output_file << std::endl;
  output_file.close();
  return 0;
}
