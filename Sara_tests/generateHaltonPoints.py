import sys
import numpy as np
from scipy.stats.qmc import Halton

# Function to generate 2D Halton points scaled to [-1, 1]
def halton_points_2d(num_points, dimensions):
    halton_gen = Halton(dimensions, scramble=True, optimization="lloyd")
    halton_points = halton_gen.random(num_points)
    return 2 * halton_points - 1  # Scale to [-1, 1]

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
    output_filename = f"/Users/saraavesani/fmca/Sara_tests/data/vertices_square_Halton{N}.txt"

    N_interior = N
    N_boundary = int(np.sqrt(N_interior))

    interior_points = halton_points_2d(N_interior, 2)
    boundary_points = generate_boundary_points(N_boundary)
    points_total = np.concatenate([interior_points, boundary_points])

    np.savetxt(output_filename, points_total, fmt="%.16f")
    print(f"Saved Halton points to {output_filename}")
