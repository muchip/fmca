/* 
In this code, we just test the function read_files_txt.h. The idea is to read the coordinates of the points and store them row major.
Our data point, in 2d for instance, look like:
x_0 x_1 x_2 ... x_{N-2} x_{N-1} x_{N}
y_0 y_1 y_2 ... y_{N-2} y_{N-1} y_{N}
We rely on the FMCA library by M.Multerer.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "read_files_txt.h"

int main() {
  int NPTS_SOURCE;
  int NPTS_QUAD;
  int N_WEIGHTS;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Vector w;

  readTXT("data/grid_points.txt", P_sources, NPTS_SOURCE, 2);
  std::cout << "P_sources dimensions: " << P_sources.rows() << "x" << P_sources.cols() << std::endl;

  readTXT("data/baricenters_circle.txt", P_quad, NPTS_QUAD, 2);
  std::cout << "P_quad dimensions: " << P_quad.rows() << "x" << P_quad.cols() << std::endl;

  readTXT("data/triangles_volumes_circle.txt", w, N_WEIGHTS);
  std::cout << "w dimensions: " << w.rows() << "x" << w.cols() << std::endl;

  return 0;
}
