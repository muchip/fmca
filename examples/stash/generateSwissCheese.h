
#include <Eigen/Dense>
#include <vector>

Eigen::MatrixXd generateSwissCheese(unsigned int dim, unsigned int npts) {
  unsigned int nholes = 0;
  double min_rad = 0;
  double max_rad = 0;
  std::cout << "using swiss cheese\n";
  switch (dim) {
    case 1:
      nholes = 100;
      min_rad = 0.004;
      max_rad = 0.006;
      break;
    case 2:
      nholes = 3000;
      min_rad = 0.009;
      max_rad = 0.01;
      break;
    case 3:
      nholes = 10000;
      min_rad = 0.03;
      max_rad = 0.034;
      break;
  }

  srand(0);
  Eigen::MatrixXd retval(dim, npts);
  retval.setZero();
  // generate holes
  std::vector<std::pair<Eigen::VectorXd, double>> holes;
  for (auto i = 0; i < nholes; ++i) {
    holes.push_back(std::make_pair(
        1.1 * 0.5 * (Eigen::VectorXd::Random(dim).array() + 1),
        min_rad + (min_rad - max_rad) * log(1. - 1. * rand() / RAND_MAX)));
  }
  for (auto i = 0; i < npts; ++i) {
    Eigen::VectorXd cur_pt;
    bool found_pt = false;
    while (!found_pt) {
      cur_pt = 0.5 * (Eigen::VectorXd::Random(dim).array() + 1);
      bool hit_hole = false;
      for (auto j = 0; j < nholes; ++j)
        if ((cur_pt - holes[j].first).norm() < holes[j].second) {
          hit_hole = true;
          break;
        }
      if (!hit_hole) found_pt = true;
    }
    retval.col(i) = cur_pt;
  }

  return retval;
}
