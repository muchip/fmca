#include <FMCA/src/util/print2file.h>

#include <Eigen/Dense>

int main() {
  size_t dim = 6;//size_t(786729241);
  Eigen::MatrixXd data = Eigen::MatrixXd::Random(dim, 3);
  Eigen::MatrixXd read_data;
  FMCA::IO::print2bin("matrix.dat", data);
  FMCA::IO::bin2Mat("matrix.dat", &read_data);
  std::cout << (data-read_data).colwise().maxCoeff() << std::endl;
  return 0;
}
