#define USE_QR_CONSTRUCTION_
////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/Sparse>
////////////////////////////////////////////////////////////////////////////////
#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/H2Matrix/ChebyshevInterpolation.h"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
#include "FMCA/src/util/print2file.hpp"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"
////////////////////////////////////////////////////////////////////////////////
#define DIM 2
#define MPOLE_DEG 5
#define DTILDE 3
#define LEAFSIZE 4

using ClusterT = FMCA::ClusterTree<double, DIM, LEAFSIZE, MPOLE_DEG>;

int main() {
  const double threshold = 1e-4;
  tictoc T;
  Eigen::MatrixXd Rchan = readMatrix("imgCompression/Rchan.txt");
  Eigen::MatrixXd Gchan = readMatrix("imgCompression/Gchan.txt");
  Eigen::MatrixXd Bchan = readMatrix("imgCompression/Bchan.txt");
  std::cout << "img data: " << Rchan.rows() << "x" << Rchan.cols() << std::endl;
  Eigen::MatrixXd P;
  Eigen::MatrixXd Q;
  P.resize(2, Rchan.rows() * Rchan.cols());
  auto k = 0;
  for (auto i = 0; i < Rchan.cols(); ++i)
    for (auto j = 0; j < Rchan.rows(); ++j) {
      P(0, k) = i;
      P(1, k) = j;
      ++k;
    }
  Q.resize(3, P.cols());
  Q.setZero();
  Q.topRows(2) = P;
  T.tic();
  ClusterT CT(P);
  T.toc("set up cluster tree: ");
  T.tic();
  FMCA::SampletTree<ClusterT> ST(P, CT, DTILDE);
  T.toc("set up samplet tree: ");

  auto indices = CT.get_indices();
  std::vector<FMCA::IndexType> rev_indices;
  rev_indices.resize(indices.size());
  for (auto i = 0; i < indices.size(); ++i) rev_indices[indices[i]] = i;
  Eigen::VectorXd reddata(indices.size());
  Eigen::VectorXd greendata(indices.size());
  Eigen::VectorXd bluedata(indices.size());

  for (auto i = 0; i < indices.size(); ++i) {
    reddata(i) = Rchan(indices[i]);
    greendata(i) = Gchan(indices[i]);
    bluedata(i) = Bchan(indices[i]);
  }
  Eigen::VectorXd Treddata = ST.sampletTransform(reddata);
  Eigen::VectorXd Tgreendata = ST.sampletTransform(greendata);
  Eigen::VectorXd Tbluedata = ST.sampletTransform(bluedata);
  double Rthres = threshold * Treddata.cwiseAbs().maxCoeff();
  double Gthres = threshold * Tgreendata.cwiseAbs().maxCoeff();
  double Bthres = threshold * Tbluedata.cwiseAbs().maxCoeff();
  // plot relevant samplet coefficients
  ST.visualizeCoefficients(Treddata, "coeffR.vtk", Rthres);
  ST.visualizeCoefficients(Tgreendata, "coeffG.vtk", Gthres);
  ST.visualizeCoefficients(Tbluedata, "coeffB.vtk", Bthres);
  FMCA::IO::plotPoints("pointsR.vtk", CT, Q, reddata);
  FMCA::IO::plotPoints("pointsG.vtk", CT, Q, greendata);
  FMCA::IO::plotPoints("pointsB.vtk", CT, Q, bluedata);

  unsigned int rcoeff = 0;
  unsigned int gcoeff = 0;
  unsigned int bcoeff = 0;
  for (auto i = 0; i < Treddata.size(); ++i) {
    if (abs(Treddata(i)) < Rthres)
      Treddata(i) = 0;
    else
      ++rcoeff;
    if (abs(Tgreendata(i)) < Gthres)
      Tgreendata(i) = 0;
    else
      ++gcoeff;
    if (abs(Tbluedata(i)) < Bthres)
      Tbluedata(i) = 0;
    else
      ++bcoeff;
  }
  std::cout << "original coeffs: " << Treddata.size() << " remaining coeffs: ("
            << rcoeff << "|" << gcoeff << "|" << bcoeff << ") compression: ("
            << 1 - double(rcoeff) / Treddata.size() << "|"
            << 1 - double(gcoeff) / Treddata.size() << "|"
            << 1 - double(bcoeff) / Treddata.size() << ")" << std::endl;

  //Treddata = ST.inverseSampletTransform(Treddata);
  //Tgreendata = ST.inverseSampletTransform(Tgreendata);
  //Tbluedata = ST.inverseSampletTransform(Tbluedata);
  Eigen::VectorXd compTR = Treddata;
  Eigen::VectorXd compTG = Tgreendata;
  Eigen::VectorXd compTB = Tbluedata;
  for (auto i = 0; i < indices.size(); ++i) {
    compTR(i) = Treddata(rev_indices[i]);
    compTG(i) = Tgreendata(rev_indices[i]);
    compTB(i) = Tbluedata(rev_indices[i]);
  }
  Eigen::Map<Eigen::MatrixXd> Rmap(compTR.data(), Rchan.rows(), Rchan.cols());
  Eigen::Map<Eigen::MatrixXd> Gmap(compTG.data(), Gchan.rows(), Gchan.cols());
  Eigen::Map<Eigen::MatrixXd> Bmap(compTB.data(), Bchan.rows(), Bchan.cols());

  Bembel::IO::print2m("Rcompress.m", "R", Rmap, "w");
  Bembel::IO::print2m("Gcompress.m", "G", Gmap, "w");
  Bembel::IO::print2m("Bcompress.m", "B", Bmap, "w");
  return 0;
}
