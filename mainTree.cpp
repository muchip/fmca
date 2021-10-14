#include <Eigen/Dense>

#include "FMCA/ClusterTree"
#include "FMCA/src/util/tictoc.hpp"

  using ClusterT = FMCA::ClusterTree<double>;
 // using ClusterT = FMCA::ClusterTree<double, 3, 100, 3>;

int main() {
  std::cout << "using random points\n";
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(3, 10000000);
  std::cout << P.rows() << " " << P.cols() << std::endl;
  // P.row(2) *= 0;
  // Eigen::VectorXd nrms = P.colwise().norm();
  // for (auto i = 0; i < P.cols(); ++i) P.col(i) *= 1 / nrms(i);
  tictoc T;

  T.tic();
  //ClusterT CT(P, 100);
  ClusterT CT();
  T.toc("tree setup: ");

  return 0;
}
