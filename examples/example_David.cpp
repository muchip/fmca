// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
#include <sys/time.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
////////////////////////////////////////////////////////////////////////////////
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/print2file.h"
#include "../Points/matrixReader.h"
#include "formatted_multiplication.h"
#include "pardiso_interface.h"
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/Samplets/omp_samplet_compressor.h>
#include <FMCA/src/Samplets/omp_samplet_compressor_unsymmetric.h>
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/IO.h>
////////////////////////////////////////////////////////////////////////////////
void plotGrid(const std::string &fileName, const Eigen::MatrixXd &P, int nx,
              int ny, int nz, const Eigen::VectorXd &color) {
  std::ofstream myfile;
  myfile.open(fileName);
  myfile << "# vtk DataFile Version 3.1\n";
  myfile << "this file hopefully represents my surface now\n";
  myfile << "ASCII\n";
  myfile << "DATASET STRUCTURED_GRID\n";
  myfile << "DIMENSIONS " << nx << " " << ny << " " << nz << std::endl;
  myfile << "POINTS " << P.cols() << " FLOAT" << std::endl;
  for (auto i = 0; i < P.cols(); ++i)
    myfile << P(0, i) << " " << P(1, i) << " " << P(2, i) << std::endl;
  myfile << "POINT_DATA " << color.size() << "\n";
  myfile << "SCALARS value FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < color.size(); ++i) myfile << color(i) << std::endl;
  myfile.close();
}
////////////////////////////////////////////////////////////////////////////////
struct thinPCov {
  thinPCov(const double R) : R_(R), R3_(R * R * R) {}
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    const double r = (x - y).norm();
    return r * r * (2 * r - 3 * R_) + R3_;
  }
  const double R_;
  const double R3_;
};
////////////////////////////////////////////////////////////////////////////////
using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, thinPCov>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using MatrixEvaluatorUS =
    FMCA::unsymmetricNystromMatrixEvaluator<Moments, thinPCov>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const unsigned int dtilde = 4;
  const double eta = 0.8;
  const unsigned int mp_deg = 6;
  const double ridgep = atof(argv[4]);
  double threshold = 1e-5;
  FMCA::Tictoc T;
  std::fstream output_file;
  Eigen::MatrixXd P;
  Eigen::MatrixXd Peval;
  const unsigned int nx = 100;
  const unsigned int ny = 100;
  const unsigned int nz = 60;
  Eigen::VectorXd y;
  //////////////////////////////////////////////////////////////////////////////
  // init data points and evaluation points
  //////////////////////////////////////////////////////////////////////////////
  {
    const unsigned int ninner = atoi(argv[1]);
    const unsigned int nouter = atoi(argv[2]);
    const unsigned int nbdry = atoi(argv[3]);
    Eigen::MatrixXd Pts = readMatrix("../Points/bunnySurface.txt");
    Eigen::MatrixXd pts_min = Pts.colwise().minCoeff();
    Eigen::MatrixXd pts_max = Pts.colwise().maxCoeff();
    pts_min *= 1.1;
    pts_max *= 1.1;
    std::vector<int> subsample(Pts.rows());
    std::iota(subsample.begin(), subsample.end(), 0);
    std::random_shuffle(subsample.begin(), subsample.end());
    P.resize(Pts.cols(), ninner + nouter + nbdry);
    y.resize(P.cols());
    for (auto i = 0; i < ninner; ++i) {
      Eigen::Vector3d rdm;
      rdm.setRandom();
      P.col(i) = 0.1 * rdm / rdm.norm();
      // P(2, i) += 0.2;
      y(i) = 1;
    }
    for (auto i = ninner; i < ninner + nbdry; ++i) {
      P.col(i) = Pts.row(subsample[i]).transpose();
      y(i) = 0;
    }
    Peval.resize(P.rows(), nx * ny * nz);
    double hx = (pts_max(0) - pts_min(0)) / (nx - 1);
    double hy = (pts_max(1) - pts_min(1)) / (ny - 1);
    double hz = (pts_max(2) - pts_min(2)) / (nz - 1);
    for (auto i = ninner + nbdry; i < P.cols(); ++i) {
      Eigen::Vector3d rdm;
      rdm.setRandom();
      rdm /= rdm.cwiseAbs().maxCoeff();
      rdm = 0.5 * (rdm.array() + 1);
      P.col(i) = (pts_max - pts_min).transpose().array() * rdm.array() +
                 pts_min.transpose().array();
      y(i) = -1;
    }
    auto l = 0;
    for (auto k = 0; k < nz; ++k)
      for (auto j = 0; j < ny; ++j)
        for (auto i = 0; i < nx; ++i, ++l)
          Peval.col(l) << pts_min(0) + hx * i, pts_min(1) + hy * j,
              pts_min(2) + hz * k;

    FMCA::IO::plotPointsColor("initalSetup.vtk", P, y);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init samplet matrix at data sites
  //////////////////////////////////////////////////////////////////////////////
  const unsigned int npts = P.cols();
  const unsigned int dim = P.rows();
  const auto function = thinPCov(2);
  std::cout << std::string(75, '=') << std::endl;
  std::cout << "N:                           " << npts << std::endl
            << "dim:                         " << dim << std::endl
            << "eta:                         " << eta << std::endl
            << "multipole degree:            " << mp_deg << std::endl
            << "vanishing moments:           " << dtilde << std::endl
            << "aposteriori threshold:       " << threshold << std::endl
            << "ridge parameter:             " << ridgep << std::endl
            << "number of evaluation points: " << nx * ny * nz << std::endl;
  const Moments mom(P, mp_deg);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("tree setup:                 ");
  FMCA::Vector min_dist = minDistanceVector(hst, P);
  FMCA::Scalar min_min_dist = min_dist.minCoeff();
  FMCA::Scalar max_min_dist = min_dist.maxCoeff();
  std::cout << "fill distance:               " << max_min_dist << std::endl;
  std::cout << "separation distance:         " << min_min_dist << std::endl;
  T.tic();
  FMCA::ompSampletCompressor<H2SampletTree> comp;
  comp.init(hst, eta, 0);
  comp.compress(hst, mat_eval);
  double comp_time = T.toc("cummulative compressor:     ");
  std::vector<Eigen::Triplet<double>> trips = comp.triplets();
  std::cout << "anz:                         " << (double)(trips.size()) / npts
            << std::endl;
  FMCA::SparseMatrix<double>::sortTripletsInPlace(trips);
  double comperr =
      FMCA::errorEstimatorSymmetricCompressor(trips, function, hst, P);
  std::cout << "compression error:           " << comperr << std::endl;
  Eigen::SparseMatrix<double> K(npts, npts);
  K.setFromTriplets(trips.begin(), trips.end());
  if (ridgep > 0) {
    T.tic();
    for (auto i = 0; i < K.rows(); ++i)
      K.coeffRef(i, i) = K.coeffRef(i, i) + ridgep;
    T.toc("added regularization:       ");
  }
  K.makeCompressed();
  //////////////////////////////////////////////////////////////////////////////
  // invert samplet matrix
  //////////////////////////////////////////////////////////////////////////////
  Eigen::SparseMatrix<double, Eigen::RowMajor> invK = K;
  T.tic();
  pardiso_interface(invK.outerIndexPtr(), invK.innerIndexPtr(), invK.valuePtr(),
                    invK.rows());
  T.toc("matrix inversion:           ");
  double err = 0;
  {
    Eigen::MatrixXd x(npts, 10), y(npts, 10), z(npts, 10);
    x.setRandom();
    Eigen::VectorXd nrms = x.colwise().norm();
    for (auto i = 0; i < x.cols(); ++i) x.col(i) /= nrms(i);
    y.setZero();
    z.setZero();
    y = K.selfadjointView<Eigen::Upper>() * x;
    z = invK.selfadjointView<Eigen::Upper>() * y;
    err = (z - x).norm() / x.norm();
  }
  std::cout << "inverse error:               " << err << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // evaluate posterior mean
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(75, '=') << std::endl;
  std::cout << "evaluation phase" << std::endl;
  const Moments mom2(Peval, mp_deg);
  const MatrixEvaluator mat_eval2(mom2, function);
  const SampletMoments samp_mom2(Peval, dtilde - 1);
  T.tic();
  H2SampletTree hst_eval(mom2, samp_mom2, 0, Peval);
  T.toc("evaluation tree setup:      ");
  T.tic();
  FMCA::ompSampletCompressorUnsymmetric<H2SampletTree> comp_unsym;
  comp_unsym.init(hst_eval, hst, eta, threshold);
  T.toc("omp initializer unsymmetric:");
  const MatrixEvaluatorUS mat_eval_us(mom2, mom, function);
  comp_unsym.compress(hst_eval, hst, mat_eval_us);
  comp_time = T.toc("cummulative compressor usym:");
  std::vector<Eigen::Triplet<double>> trips_eval = comp_unsym.triplets();
  std::cout << "anz:                         "
            << (double)(trips_eval.size()) / npts << std::endl;
  auto J = hst.indices();
  auto I = hst_eval.indices();
  Eigen::SparseMatrix<double> Keval(I.size(), J.size());
  Keval.setFromTriplets(trips_eval.begin(), trips_eval.end());
  Eigen::VectorXd yJ = y;
  Eigen::MatrixXd PJ = P;
  Eigen::MatrixXd PevalI = Peval;
  for (auto i = 0; i < J.size(); ++i) yJ(i) = y(J[i]);
  for (auto i = 0; i < J.size(); ++i) PJ.col(i) = P.col(J[i]);
  for (auto i = 0; i < I.size(); ++i) PevalI.col(i) = Peval.col(I[i]);
  Eigen::VectorXd TyJ = hst.sampletTransform(yJ);
  Eigen::VectorXd Tmu = invK.selfadjointView<Eigen::Upper>() * TyJ;
  err = (K.selfadjointView<Eigen::Upper>() * Tmu - TyJ).norm() / TyJ.norm();
  std::cout << err << std::endl;
  for (auto i = 0; i < 20; ++i) {
    Eigen::VectorXd res = TyJ - K.selfadjointView<Eigen::Upper>() * Tmu;
    Tmu += 0.9 * (invK.selfadjointView<Eigen::Upper>() * res).eval();
    err = (K.selfadjointView<Eigen::Upper>() * Tmu - TyJ).norm() / TyJ.norm();
    std::cout << err << std::endl;
  }
  Eigen::VectorXd mu = hst.inverseSampletTransform(Tmu);
  Eigen::VectorXd predI = hst_eval.inverseSampletTransform(Keval * Tmu);
  Eigen::VectorXd pred = predI;
  for (auto i = 0; i < pred.size(); ++i) pred(I[i]) = predI(i);
  plotGrid("prediction.vtk", Peval, nx, ny, nz, pred);
  return 0;
}
