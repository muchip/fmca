#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <functional>
#include <iomanip>

#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/H2Matrix/ChebyshevInterpolation.h"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
#include "FMCA/src/util/print2file.hpp"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"
////////////////////////////////////////////////////////////////////////////////
#define NPTS 8192
#define DIM 3
#define MPOLE_DEG 4
#define DTILDE 4
#define LEAFSIZE 4

struct exponentialKernel {
  double operator()(const Eigen::Matrix<double, DIM, 1> &x,
                    const Eigen::Matrix<double, DIM, 1> &y) const {
    return exp(-4 * (x - y).norm());
  }
};

using ClusterT = FMCA::ClusterTree<double, DIM, LEAFSIZE, MPOLE_DEG>;

int main() {
#if 0
  std::cout << "loading data: ";
  Eigen::MatrixXd B = readMatrix("defiant.txt");
  std::cout << "data size: ";
  std::cout << B.rows() << " " << B.cols() << std::endl;
  std::cout << "----------------------------------------------------\n";
  Eigen::MatrixXd P = B.transpose();
#else
#if 0
  srand(0);
  Eigen::Matrix3d rot;
  rot << 0.8047379, -0.3106172, 0.5058793, 0.5058793, 0.8047379, -0.3106172,
      -0.3106172, 0.5058793, 0.8047379;
  Eigen::MatrixXd P1 =
      2 * FMCA_PI * (Eigen::MatrixXd::Random(1, NPTS).array() + 1);
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  P.row(0).array() = P1.array().cos();
  P.row(1).array() = P1.array().sin();
  P.row(2) = 0.5 * P1;
  P = rot * P;
#endif
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  P.row(2) *= 0;
  Eigen::VectorXd nrms = P.colwise().norm();
  for (auto i = 0; i < P.cols(); ++i)
    P.col(i) *= 1 / nrms(i);
#endif
  tictoc T;
  T.tic();
  ClusterT CT(P);
  T.toc("set up cluster tree: ");
  T.tic();
  FMCA::SampletTree<ClusterT> ST(P, CT, DTILDE);
  T.toc("set up samplet tree: ");
  Eigen::VectorXd data(P.cols());
  auto fun = exponentialKernel();
  for (auto j = 0; j < P.cols(); ++j)
    data(j) = fun(P.col(0), P.col(CT.get_indices()[j]));
  Eigen::VectorXd Tdata = ST.sampletTransform(data);
  ST.visualizeCoefficients(Tdata, "coeff.vtk");
  FMCA::IO::plotPoints("points.vtk", CT, P, data);
  return 0;
}
