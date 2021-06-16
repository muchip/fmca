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

#define DIM 3
#define MPOLE_DEG 10
#define DTILDE 2
#define LEAFSIZE 4

struct exponentialKernel {
  double operator()(const Eigen::Matrix<double, DIM, 1> &x,
                    const Eigen::Matrix<double, DIM, 1> &y) const {
    return exp(-4 * (x - y).norm());
  }
};

template <typename Functor>
Eigen::VectorXd matrixMultiplier(const Eigen::MatrixXd &P,
                                 const std::vector<FMCA::IndexType> &idcs,
                                 const Functor &fun, const Eigen::VectorXd &x) {
  Eigen::VectorXd retval(x.size());
  retval.setZero();
  for (auto i = 0; i < x.size(); ++i)
    for (auto j = 0; j < x.size(); ++j)
      retval(i) += fun(P.col(idcs[i]), P.col(idcs[j])) * x(j);
  return retval;
}
using ClusterT = FMCA::ClusterTree<double, DIM, LEAFSIZE, MPOLE_DEG>;
using H2ClusterT = FMCA::H2ClusterTree<ClusterT, MPOLE_DEG>;

int main(int argc, char *argv[]) {
  const int npts = atoi(argv[1]);
  const double eta = 0.8;
  const double svd_threshold = 1e-8;
  const double aposteriori_threshold = 1e-10;
  tictoc T;
  std::cout << std::string(60, '-') << std::endl;
  std::cout << "dim:       " << DIM << std::endl;
  std::cout << "leaf size: " << LEAFSIZE << std::endl;
  std::cout << "mpole deg: " << MPOLE_DEG << std::endl;
  std::cout << "dtilde:    " << DTILDE << std::endl;
  std::cout << "npts:      " << npts << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  srand(0);
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, npts);
  //////////////////////////////////////////////////////////////////////////////
  // set up cluster tree
  T.tic();
  ClusterT CT(P);
  T.toc("set up cluster tree: ");
  //////////////////////////////////////////////////////////////////////////////
  // set up H2 cluster tree
  T.tic();
  H2ClusterT H2CT(P, CT);
  T.toc("set up H2-cluster tree: ");
  {
    std::vector<std::vector<FMCA::IndexType>> tree;
    CT.exportTreeStructure(tree);
    std::cout << "cluster structure: " << std::endl;
    std::cout << "l)\t#pts\ttotal#pts" << std::endl;
    for (auto i = 0; i < tree.size(); ++i) {
      int numInd = 0;
      for (auto j = 0; j < tree[i].size(); ++j) numInd += tree[i][j];
      std::cout << i << ")\t" << tree[i].size() << "\t" << numInd << "\n";
    }
    std::cout << std::string(60, '-') << std::endl;
  }
  //////////////////////////////////////////////////////////////////////////////
  // set up samplet tree
  T.tic();
  FMCA::SampletTree<ClusterT> ST(P, CT, DTILDE, svd_threshold);
  T.toc("set up samplet tree: ");
  T.tic();
#ifdef USE_QR_CONSTRUCTION_
  std::cout << "using QR construction for samplet basis\n";
#else
  std::cout << "using SVD construction for samplet basis\n";
  std::cout << "SVD orthogonality threshold: " << svd_threshold << std::endl;
#endif
  ST.computeMultiscaleClusterBases(H2CT);
  T.toc("set up time multiscale cluster bases");
  std::cout << std::string(60, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // set up bivariate compressor
  T.tic();
  FMCA::BivariateCompressorH2<FMCA::SampletTree<ClusterT>> BC;
  BC.set_eta(eta);
  BC.set_threshold(aposteriori_threshold);
  std::cout << "bivariate compressor: \n";
  std::cout << "eta:       " << eta << std::endl;
  std::cout << "threshold: " << aposteriori_threshold << std::endl;
  BC.init(P, ST, exponentialKernel());
  std::cout << std::string(60, '-') << std::endl;
  T.toc("set up Samplet compressed matrix: ");
  {
    auto trips = BC.get_Pattern_triplets();
    std::cout << "nz per row:     " << trips.size() / P.cols() << std::endl;
    std::cout << "storage sparse: "
              << double(sizeof(double) * trips.size() * 3) / double(1e9)
              << "GB\n";
    std::cout << "storage full:   "
              << double(sizeof(double) * P.cols() * P.cols()) / double(1e9)
              << "GB" << std::endl;
  }
  std::cout << std::string(60, '-') << std::endl;
  {
    std::cout << "testing vanishing moments:\n";
    // compute the multi indices for the monomials used for vanishing moments
    FMCA::MultiIndexSet<DIM> idcs(DTILDE - 1);
    // evaluate all these monomials at the points
    Eigen::MatrixXd Pol = FMCA::momentComputer(P, CT, idcs);
    double err = 0;
    for (auto i = 0; i < Pol.rows(); ++i) {
      Pol.row(i) = ST.sampletTransform(Pol.row(i));
      err += Pol.row(i).tail(Pol.cols() - ST.get_nscalfs()).norm();
    }
    std::cout << "orthogonality error: " << err / Pol.rows() << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }
  {
    T.tic();
    auto trips = BC.get_Pattern_triplets();
    Eigen::SparseMatrix<double> W(P.cols(), P.cols());
    W.setFromTriplets(trips.begin(), trips.end());
    double err = 0;
    Eigen::VectorXd y1(P.cols());
    Eigen::VectorXd y2(P.cols());
    Eigen::VectorXd x(P.cols());
    for (auto i = 0; i < 10; ++i) {
      x.setRandom();
      y1 = matrixMultiplier(P, CT.get_indices(), exponentialKernel(), x);
      y2 = ST.inverseSampletTransform(W * ST.sampletTransform(x));
      err += (y1 - y2).norm() / y1.norm();
    }
    T.toc("time error computation: ");
    std::cout << "error: " << err / 10 << std::endl;
  }
  return 0;
}
