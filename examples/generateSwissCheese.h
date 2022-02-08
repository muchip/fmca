
#include <Eigen/Dense>
#include <vector>

Eigen::MatrixXd generateSwissCheese(unsigned int dim, unsigned int npts,
                                    unsigned int nholes) {
  constexpr double min_rad = 0.008;
  constexpr double max_rad = 0.014;
  srand(666);
  Eigen::MatrixXd retval(dim, npts);
  retval.setZero();
  // generate holes
  std::vector<std::pair<Eigen::VectorXd, double>> holes;
  for (auto i = 0; i < nholes; ++i) {
    holes.push_back(std::make_pair(
        1.1 * Eigen::VectorXd::Random(dim),
        min_rad + (min_rad - max_rad) * log(1. - 1. * rand() / RAND_MAX)));
  }
  for (auto i = 0; i < npts; ++i) {
    Eigen::VectorXd cur_pt;
    bool found_pt = false;
    while (!found_pt) {
      cur_pt = Eigen::VectorXd::Random(dim);
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
