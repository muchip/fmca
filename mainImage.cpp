#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <functional>
#include <iomanip>

#include "FMCA/BlockClusterTree"
#include "FMCA/Samplets"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
#include "imgCompression/matrixReader.h"
#include "print2file.hpp"
#include "util/tictoc.hpp"
#define DIM 3

using ClusterT = FMCA::ClusterTree<double, DIM, 3>;

int main() {
  Eigen::MatrixXd Rchan = readMatrix("imgCompression/Rchan.txt");
  Eigen::MatrixXd Gchan = readMatrix("imgCompression/Gchan.txt");
  Eigen::MatrixXd Bchan = readMatrix("imgCompression/Bchan.txt");
  std::cout << "img data: " << Rchan.rows() << "x" << Rchan.cols() << std::endl;
  Eigen::MatrixXd P;
  P.resize(3, Rchan.rows() * Rchan.cols());
  auto k = 0;
  for (auto i = 0; i < Rchan.cols(); ++i)
    for (auto j = 0; j < Rchan.rows(); ++j) {
      P(0, k) = i;
      P(1, k) = j;
      P(2, k) = 0;
      ++k;
    }
  ClusterT CT(P);
  FMCA::SampletTree<ClusterT> ST(P, CT, 0);
  auto indices = CT.get_indices();

  Eigen::VectorXd rfdata(indices.size());
  Eigen::VectorXd gfdata(indices.size());
  Eigen::VectorXd bfdata(indices.size());
  for (auto i = 0; i < indices.size(); ++i) {
    rfdata(i) = Rchan(indices[i]);
    gfdata(i) = Gchan(indices[i]);
    bfdata(i) = Bchan(indices[i]);
  }
  Eigen::VectorXd rcompf = ST.sampletTransform(rfdata);
  Eigen::VectorXd gcompf = ST.sampletTransform(gfdata);
  Eigen::VectorXd bcompf = ST.sampletTransform(bfdata);

  std::cout << "transform done!\n";
  Eigen::VectorXd inv_compf = ST.inverseSampletTransform(rcompf);
  std::cout << "???????????? " << (inv_compf - rfdata).norm() / rfdata.norm()
            << std::endl;

  double rnorm = rcompf.norm();
  double gnorm = gcompf.norm();
  double bnorm = bcompf.norm();
  std::cout << rnorm << " " << gnorm << " " << bnorm << std::endl;
  for (auto i = 0; i < rcompf.size(); ++i) {
    rcompf(i) = abs(rcompf(i)) > 1e-3 * rnorm ? rcompf(i) : 0;
    gcompf(i) = abs(gcompf(i)) > 1e-3 * gnorm ? gcompf(i) : 0;
    bcompf(i) = abs(bcompf(i)) > 1e-3 * bnorm ? bcompf(i) : 0;
  }
  rcompf = ST.inverseSampletTransform(rcompf);
  gcompf = ST.inverseSampletTransform(gcompf);
  bcompf = ST.inverseSampletTransform(bcompf);

  std::vector<FMCA::IndexType> rev_indices;
  rev_indices.resize(indices.size());
  for (auto i = 0; i < indices.size(); ++i)
    rev_indices[indices[i]] = i;

  std::cout << "get here\n";
  for (auto i = 0; i < indices.size(); ++i) {
    rfdata(i) = rcompf(rev_indices[i]);
    gfdata(i) = gcompf(rev_indices[i]);
    bfdata(i) = bcompf(rev_indices[i]);
  }

  std::cout << "get here\n";
  FMCA::IO::plotPoints<ClusterT>("rComp.vtk", CT, P, rcompf);
  FMCA::IO::plotPoints<ClusterT>("gComp.vtk", CT, P, gcompf);
  FMCA::IO::plotPoints<ClusterT>("bComp.vtk", CT, P, bcompf);

  std::cout << "get here\n";
  Eigen::Map<Eigen::MatrixXd> rmap(rfdata.data(), Rchan.rows(), Rchan.cols());
  Eigen::Map<Eigen::MatrixXd> gmap(gfdata.data(), Gchan.rows(), Gchan.cols());
  Eigen::Map<Eigen::MatrixXd> bmap(bfdata.data(), Bchan.rows(), Bchan.cols());
  Bembel::IO::print2m("Rchannel.m", "R", rmap, "w");
  Bembel::IO::print2m("Gchannel.m", "G", gmap, "w");
  Bembel::IO::print2m("Bchannel.m", "B", bmap, "w");

  return 0;
}
