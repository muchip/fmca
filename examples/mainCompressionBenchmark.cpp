#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#define USE_QR_CONSTRUCTION_
//#define FMCA_CLUSTERSET_
#define DIM 2
#define MPOLE_DEG 3
#define DTILDE 3
#define LEAFSIZE 16
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <functional>
#include <iomanip>
////////////////////////////////////////////////////////////////////////////////
#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename Derived>
double get2norm(const Eigen::SparseMatrix<Derived> &A) {
  double retval = 0;
  Eigen::VectorXd vec = Eigen::VectorXd::Random(A.cols());
  vec /= vec.norm();
  for (auto i = 0; i < 100; ++i) {
    vec = A.template selfadjointView<Eigen::Lower>() * vec;
    retval = vec.norm();
    vec *= 1. / retval;
  }
  return retval;
}
////////////////////////////////////////////////////////////////////////////////
struct exponentialKernel {
  double operator()(const Eigen::Matrix<double, DIM, 1> &x,
                    const Eigen::Matrix<double, DIM, 1> &y) const {
    return exp(-20 * (x - y).norm() / sqrt(DIM));
  }
};
////////////////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////////////////
template <typename Functor>
Eigen::VectorXd matrixColumnGetter(const Eigen::MatrixXd &P,
                                   const std::vector<FMCA::IndexType> &idcs,
                                   const Functor &fun, Eigen::Index colID) {
  Eigen::VectorXd retval(P.cols());
  retval.setZero();
  for (auto i = 0; i < retval.size(); ++i)
    retval(i) = fun(P.col(idcs[i]), P.col(idcs[colID]));
  return retval;
}
////////////////////////////////////////////////////////////////////////////////
using ClusterT = FMCA::ClusterTree<double, DIM, LEAFSIZE, MPOLE_DEG>;
using H2ClusterT = FMCA::H2ClusterTree<ClusterT, MPOLE_DEG>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const double eta = 0.8;
  const double aposteriori_threshold = 1e-6;
  const std::string logger =
      "loggerCompressionBenchmark_" + std::to_string(DIM) + ".txt";
  tictoc T;
  {
    std::ifstream file;
    file.open(logger);
    if (!file.good()) {
      file.close();
      std::fstream newfile;
      newfile.open(logger, std::fstream::out);
      newfile << std::setw(10) << "Npts" << std ::setw(5) << "dim"
              << std::setw(8) << "mpdeg" << std::setw(8) << "dtilde"
              << std::setw(6) << "eta" << std::setw(8) << "apost"
              << std::setw(9) << "nza" << std::setw(9) << "nzp" << std::setw(14)
              << "Comp_err" << std::setw(14) << "2norm" << std::setw(12)
              << "ctime" << std::endl;
      newfile.close();
    }
  }
  std::cout << std::string(60, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  for (auto i = 4; i <= 20; ++i) {
    const unsigned int npts = 1 << i;
    Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, npts);
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "dim:       " << DIM << std::endl;
    std::cout << "leaf size: " << LEAFSIZE << std::endl;
    std::cout << "mpole deg: " << MPOLE_DEG << std::endl;
    std::cout << "dtilde:    " << DTILDE << std::endl;
    std::cout << "npts:      " << npts << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    ////////////////////////////////////////////////////////////////////////////
    // set up cluster tree
    T.tic();
    ClusterT CT(P);
    T.toc("set up cluster tree: ");
    ////////////////////////////////////////////////////////////////////////////
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
    FMCA::SampletTree<ClusterT> ST(P, CT, DTILDE);
    T.toc("set up samplet tree: ");
    T.tic();
#ifdef USE_QR_CONSTRUCTION_
    std::cout << "using QR construction for samplet basis\n";
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
    BC.init(P, ST, exponentialKernel());
    std::cout << std::string(60, '-') << std::endl;
    double ctime = T.toc("set up Samplet compressed matrix: ");
    std::cout << std::string(60, '-') << std::endl;
    double nz = 0;
    double nza = 0;
    unsigned int nzL = 0;
    Eigen::SparseMatrix<double> W(P.cols(), P.cols());
    {
      const std::vector<Eigen::Triplet<double>> &trips =
          BC.get_Pattern_triplets();
      nz = 2 * std::ceil(double(trips.size()) / double(P.cols())) - 1;
      nza = std::ceil(double(BC.get_storage_size()) / double(P.cols()));
      std::cout << "nz per row:     " << nza << " / " << nz << std::endl;
      std::cout << "storage sparse: "
                << double(sizeof(double) * trips.size() * 3) / double(1e9)
                << "GB\n";
      std::cout << "storage full:   "
                << double(sizeof(double) * P.cols() * P.cols()) / double(1e9)
                << "GB" << std::endl;
      std::cout << "beginning set from triplets ...\n" << std::flush;
      W.setFromTriplets(trips.begin(), trips.end());
      std::cout << "done.\n" << std::flush;
    }
    std::cout << std::string(60, '-') << std::endl;
    ////////////////////////////////////////////////////////////////////////////
    // perform error computation
    double err = 0;
    T.tic();
    Eigen::VectorXd y1(P.cols());
    Eigen::VectorXd y2(P.cols());
    Eigen::VectorXd x(P.cols());
    FMCA::ProgressBar PB;
    auto fun = exponentialKernel();
    const unsigned int errSamples = 20;
    PB.reset(errSamples);
    for (auto i = 0; i < errSamples; ++i) {
      Eigen::Index colID = rand() % P.cols();
      y1 = matrixColumnGetter(P, CT.get_indices(), fun, colID);
      x.setZero();
      x(colID) = 1;
      y2 = ST.inverseSampletTransform(W.selfadjointView<Eigen::Lower>() *
                                      ST.sampletTransform(x));
      err += (y1 - y2).norm() / y1.norm();
      PB.next();
    }
    err /= errSamples;
    double nrm = get2norm(W);
    std::cout << std::endl;
    T.toc("time compression error computation: ");
    std::cout << "compression error: " << err << std::endl;
    std::cout << "nrm2: " << nrm << std::endl;
    std::fstream newfile;
    newfile.open(logger, std::fstream::app);
    newfile << std::setw(10) << npts << std ::setw(5) << DIM << std::setw(8)
            << MPOLE_DEG << std::setw(8) << DTILDE << std::setw(6) << eta
            << std::setw(8) << aposteriori_threshold << std::setw(9) << nza
            << std::setw(9) << nz << std::setw(14) << err << std::setw(14)
            << nrm << std::setw(12) << ctime << std::endl;
    newfile.close();
  }
  return 0;
}
