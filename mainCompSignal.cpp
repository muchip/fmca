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
#define DTILDE 1
#define LEAFSIZE 4

struct exponentialKernel {
  double operator()(const Eigen::Matrix<double, DIM, 1> &x,
                    const Eigen::Matrix<double, DIM, 1> &y) const {
    return exp(-4 * (x - y).norm());
  }
};

using ClusterT = FMCA::ClusterTree<double, DIM, LEAFSIZE, MPOLE_DEG>;

int main() {
  Eigen::MatrixXd Grayscale = readMatrix("imgCompression/GrayBicycle.txt");
  std::cout << "img data: " << Grayscale.rows() << "x" << Grayscale.cols()
            << std::endl;
  Eigen::MatrixXd P;
  P.resize(3, Grayscale.rows() * Grayscale.cols());
  auto k = 0;
  for (auto i = 0; i < Grayscale.cols(); ++i)
    for (auto j = 0; j < Grayscale.rows(); ++j) {
      P(0, k) = i;
      P(1, k) = j;
      P(2, k) = 0;
      ++k;
    }
#if 0 
  std::cout << "loading data: ";
  Eigen::MatrixXd B = readMatrix("./Points/bunnyVolume.txt");
  std::cout << "data size: ";
  std::cout << B.rows() << " " << B.cols() << std::endl;
  std::cout << "----------------------------------------------------\n";
  Eigen::MatrixXd P = B.transpose();
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

  auto indices = CT.get_indices();
  Grayscale /= 255;
  Eigen::VectorXd graydata(indices.size());
  for (auto i = 0; i < indices.size(); ++i) {
    graydata(i) = Grayscale(indices[i]);
  }
  Eigen::VectorXd rcompf = ST.sampletTransform(graydata);
#if 0
  Eigen::VectorXd data(P.cols());
  auto fun = exponentialKernel();
  for (auto j = 0; j < P.cols(); ++j)
    data(j) = fun(P.col(0), P.col(CT.get_indices()[j]));
  Eigen::VectorXd Tdata = ST.sampletTransform(data);
#endif
  ST.visualizeCoefficients(rcompf, "coeff.vtk",
                           1e-3 * rcompf.cwiseAbs().maxCoeff());
  FMCA::IO::plotPoints("points.vtk", CT, P, graydata);
  return 0;
}
