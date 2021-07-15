#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#define USE_QR_CONSTRUCTION_
#define FMCA_CLUSTERSET_
#define DIM 3
#define MPOLE_DEG 3
#define DTILDE 2
#define LEAFSIZE 4
////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <functional>
#include <iomanip>

#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"
#include "matrix.h"
#include "mex.h"
////////////////////////////////////////////////////////////////////////////////
struct exponentialKernel {
  double operator()(const Eigen::Matrix<double, DIM, 1> &x,
                    const Eigen::Matrix<double, DIM, 1> &y) const {
    return exp(-10 * (x - y).norm() / sqrt(DIM));
  }
};

using ClusterT = FMCA::ClusterTree<double, DIM, 10, MPOLE_DEG>;
using H2ClusterT = FMCA::H2ClusterTree<ClusterT, MPOLE_DEG>;

/** nlhs Number of expected output mxArrays
 *   plhs Array of pointers to the expected output mxArrays
 *   nrhs Number of input mxArrays
 *   prhs Array of pointers to the input mxArrays.
 *        Do not modify any prhs values in your MEX file.
 *        Changing the data in these read-only mxArrays can
 *        produce undesired side effects.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 1) {
    mexErrMsgIdAndTxt("MATLAB:MXmain:nargin", "MXmain requires one arguments.");
  }
  const FMCA::IndexType lvl = std::round(*(mxGetPr(prhs[0])));
  const double eta = 0.8;
  const double svd_threshold = 1e-4;
  const double aposteriori_threshold = 1e-4;
  const double ridge_param = 1e-1;
  const FMCA::IndexType npts = 1 << lvl;
  const std::string logger =
      "loggerTimeBenchmark" + std::to_string(DIM) + "DQR.txt";
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
              << std::setw(8) << "svd" << std::setw(9) << "nza" << std::setw(9)
              << "nzp" << std::setw(9) << "nzL" << std::setw(14) << "Comp_err"
              << std::setw(14) << "Chol_err" << std::setw(14) << "cond"
              << std::setw(12) << "ctime" << std::endl;
      newfile.close();
    }
  }
  tictoc T;
  srand(0);
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, npts);
  T.tic();
  ClusterT CT(P);
  T.tic();
  H2ClusterT H2CT(P, CT);
  T.toc("set up H2-cluster tree: ");
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
  T.tic();
  FMCA::BivariateCompressorH2<FMCA::SampletTree<ClusterT>> BC;
  BC.set_eta(eta);
  BC.set_threshold(aposteriori_threshold);
  std::cout << "bivariate compressor: \n";
  BC.init(P, ST, exponentialKernel());
  const double ctime = T.toc("set up Samplet compressed matrix: ");
  const std::vector<Eigen::Triplet<double>> &pattern_triplets =
      BC.get_Pattern_triplets();
  plhs[0] = mxCreateDoubleMatrix(pattern_triplets.size(), 3, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  *mxGetPr(plhs[1]) = ctime;
  Eigen::Map<Eigen::MatrixXd> retU(mxGetPr(plhs[0]), pattern_triplets.size(),
                                   3);
  for (auto i = 0; i < pattern_triplets.size(); ++i) {
    eigen_assert(pattern_triplets[i].row() >= 0 && pattern_triplets[i].col() >= 0
    && "triplet negative index");
    retU.row(i) << pattern_triplets[i].row() + 1, pattern_triplets[i].col() + 1,
        pattern_triplets[i].value();
  }

  return;
}
