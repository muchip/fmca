""" It generates set of Halton Points in the Square [-1,1] x [1,1] """
import sys
import numpy as np
from scipy.stats.qmc import Halton
from collections import OrderedDict

# Function to generate 2D Halton points scaled to [-1, 1]
def halton_points_2d(num_points, dimensions):
    halton_gen = Halton(dimensions)
    halton_points = halton_gen.random(num_points)
    return 2 * halton_points - 1  # Scale to [-1, 1]

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

# Function to generate boundary Halton points
def generate_boundary_points(num_points):
    sides = []
    # Left side (x = -1)
    left_points = halton_points_2d(num_points, 2)
    left_points[:, 0] = -1
    sides.append(left_points)

    # Right side (x = 1)
    right_points = halton_points_2d(num_points, 2)
    right_points[:, 0] = 1
    sides.append(right_points)

    # Bottom side (y = -1)
    bottom_points = halton_points_2d(num_points, 2)
    bottom_points[:, 1] = -1
    sides.append(bottom_points)

    # Top side (y = 1)
    top_points = halton_points_2d(num_points, 2)
    top_points[:, 1] = 1
    sides.append(top_points)
    # corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    all_points = np.concatenate(sides)
    return all_points

# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <number_of_points>")
        sys.exit(1)

    N = int(sys.argv[1])
    output_filename1 = f"data/interior_square_Halton{N*N}.txt"
    output_filename2 = f"data/boundary_square_Halton{N*N}.txt"
    output_filename3 = f"data/int_and_bnd_square_Halton{N*N}.txt"

    N_boundary = N
    N_interior = N*N - 4*N_boundary

    interior_points = halton_points_2d(N_interior, 2)
    boundary_points = generate_uniform_grid_bnd(N_boundary)
    points_total = np.concatenate([interior_points, boundary_points])
    
    np.savetxt(output_filename1, interior_points)
    np.savetxt(output_filename2, boundary_points)
    np.savetxt(output_filename3, points_total)
    print(f"Saved interior points to {output_filename1}")
    print(f"Saved boundary points to {output_filename2}")
    print(f"Saved int_and_bnd points to {output_filename3}")
    print(f"Len interior points = ", len(interior_points))
    print(f"Len boundary points = ", len(boundary_points))
