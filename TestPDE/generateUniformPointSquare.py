""" It generates set of Uniform Points in the Square [-1,1] x [1,1] """
import sys
import numpy as np
from scipy.stats.qmc import Halton
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

def generate_uniform_grid_interior(num_points_per_dimension):
    # Calculate the step size for each dimension
    step = 2 / (num_points_per_dimension - 1)
    points = []
    for i in range(1,num_points_per_dimension-1):
        for j in range(1,num_points_per_dimension-1):
            x = -1 + i * step
            y = -1 + j * step
            points.append((x, y))
            
    return points

def generate_uniform_grid_bnd(num_points_per_dimension):
    # Calculate the step size for each dimension
    step = 2 / (num_points_per_dimension - 1)
    points = []
    for i in range(0,num_points_per_dimension):
            x = -1 + i * step
            points.append((x,-1))
            points.append((x,1))
    for i in range(0,num_points_per_dimension):
            y = -1 + i * step
            points.append((-1,y))
            points.append((1,y))   
    points = list(OrderedDict.fromkeys(points))
    return points

# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <number_of_points_per_dimension>")
        sys.exit(1)

    N = int(sys.argv[1])
    output_filename1 = f"data/interior_square{N*N}.txt"
    output_filename2 = f"data/boundary_square{N*N}.txt"
    output_filename3 = f"data/int_and_bnd_square{N*N}.txt"

    interior_points = generate_uniform_grid_interior(N)
    boundary_points = generate_uniform_grid_bnd(N)
    int_and_bnd = interior_points + boundary_points

    np.savetxt(output_filename1, interior_points)
    np.savetxt(output_filename2, boundary_points)
    np.savetxt(output_filename3, int_and_bnd)
    print(f"Saved interior points to {output_filename1}")
    print(f"Saved boundary points to {output_filename2}")
    print(f"Saved int_and_bnd points to {output_filename3}")
    print(f"Len interior points = ", len(interior_points))
    print(f"Len boundary points = ", len(boundary_points))
