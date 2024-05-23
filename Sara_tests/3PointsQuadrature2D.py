import sys
import gmsh
import igl
import numpy as np
import os

#quadrature points for the new mesh
def quadrature_3points_2d(a,b,c):
    m00 = b[0] - a[0]
    m01 = c[0] - a[0]
    m10 = b[1] - a[1]
    m11 = c[1] - a[1]
    J = [[m00, m01],
        [m10, m11]]
    detJ = (m00*m11-m01*m10); 

    X = np.array([
     [1/6, 2/3],
     [1/6, 1/6],
     [2/3, 1/6]
    ])

    W_unit = np.array([
     1/6,
     1/6,
     1/6,
    ])
    
    Y = np.dot(J, np.transpose(X))
    for i in range(len(X)):
        Y[:,i] += a
        
    W = np.array(W_unit) * np.abs(detJ)
    
    return Y, W

def main():
    if len(sys.argv) < 4:
        print("Usage: python script_name.py inputfile.mesh output_weights_file.txt output_points_file.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_weights_file = sys.argv[2]
    output_points_file = sys.argv[3]

    output_directory = "/Users/saraavesani/fmca/Sara_tests/data/"
    full_path_weights = os.path.join(output_directory, output_weights_file)
    full_path_points = os.path.join(output_directory, output_points_file)

    v01, t, f = igl.read_mesh(input_file)
    v = 2 * v01 - 1
    x_quadrature_points = []
    y_quadrature_points = []
    weights = []
    for i in range(len(f)):
        Y,W = quadrature_3points_2d(  [v[f[i,0],0],v[f[i,0],1]],  [v[f[i,1],0],v[f[i,1],1]]  ,  [v[f[i,2],0],v[f[i,2],1]])
        x_quadrature_points.append(Y[0][:])
        y_quadrature_points.append(Y[1][:])
        weights.append(W)

    quadrature_weights = np.hstack(weights)
    np.savetxt(full_path_weights, quadrature_weights, fmt="%.16f")

    quadrature_points = np.transpose([np.hstack(x_quadrature_points), np.hstack(y_quadrature_points)])
    np.savetxt(full_path_points, quadrature_points, fmt="%.16f")

if __name__ == "__main__":
    main()
