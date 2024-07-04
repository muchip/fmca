#ifndef FMCA_UTIL_PLOTTER2D_H_
#define FMCA_UTIL_PLOTTER2D_H_

#include <fstream>

namespace FMCA {
class Plotter2D {
  public:
      void plotFunction(const std::string &fileName, const Eigen::MatrixXd &points, const Eigen::VectorXd &f) const {
          if (points.rows() != 2 || points.cols() != f.size()) {
              throw std::invalid_argument("Points matrix should have 2 rows and the same number of columns as the size of the function vector.");
          }

          std::ofstream myfile;
          myfile.open(fileName);
          myfile << "# vtk DataFile Version 3.1\n";
          myfile << "2D function\n";
          myfile << "ASCII\n";
          myfile << "DATASET UNSTRUCTURED_GRID\n";
          myfile << "POINTS " << points.cols() << " FLOAT" << std::endl;

          for (auto i = 0; i < points.cols(); ++i) {
              myfile << points(0, i) << " " << points(1, i) << " " << f(i) <<  std::endl; // 0.0 for the z-coordinate
          }

          myfile << "POINT_DATA " << f.size() << "\n";
          myfile << "SCALARS value FLOAT\n";
          myfile << "LOOKUP_TABLE default\n";

          for (auto i = 0; i < f.size(); ++i) {
              myfile << f(i) << std::endl;
          }

          myfile.close();
      }
  };
}  // namespace FMCA

#endif
