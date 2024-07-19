#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

Vector3d RandomPointInterior(){
    Vector3d  point_interior;
    do {
        point_interior = Vector3d::Random();
    } while (point_interior.squaredNorm() >= 1.0);
    return point_interior;
}

Vector3d RandomPointBoundary(){
    Vector3d point_bnd = RandomPointInterior();
    return point_bnd/point_bnd.norm();
}

int main()
{
 Vector3d interior = RandomPointInterior();
 Vector3d bnd = RandomPointBoundary();

 Vector3d interior1 = RandomPointInterior();
 Vector3d bnd1 = RandomPointBoundary();
 
 std::cout << interior << std::endl;
 std::cout << "norm   " << interior.norm() << std::endl;

 std::cout << bnd << std::endl;
 std::cout << "bnd    " << bnd.norm() << std::endl;

 std::cout << interior1 << std::endl;
 std::cout << "norm1  " << interior1.norm() << std::endl;

 std::cout << bnd1 << std::endl;
 std::cout << "bnd1   " << bnd1.norm() << std::endl;

}