""" This python code uses igl to read a file .mesh and to create the qudarature points, the weights and the normals of the boundary segments 
in the mesh. This works for 2D meshes whose boundary are 1D."""

import sys
import igl
import numpy as np
import os

# Function to reorder boundary vertices 
def reorder_boundary_vertices(V_2d, boundary_vertices):
    starting_vertex_index = boundary_vertices[np.argmin(V_2d[boundary_vertices, 0])]
    ordered_indices = np.roll(boundary_vertices, -np.where(boundary_vertices == starting_vertex_index)[0][0])
    return V_2d[ordered_indices]

# Function to compute outer normals for the ordered boundary vertices
def compute_outer_normals(ordered_boundary_vertices):
    vectors = np.diff(ordered_boundary_vertices, axis=0, append=ordered_boundary_vertices[[0]])
    normals = np.array([-vectors[:, 1], vectors[:, 0]]).T
    # normalize
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return normals

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def gaussian_quadrature(n):
    # This function returns the nodes and weights for the n-point Gaussian quadrature.
    if n == 1:
        nodes = np.array([0.0])
        weights = np.array([2.0])
    elif n == 2:
        nodes = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        weights = np.array([1.0, 1.0])
    elif n == 3:
        nodes = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        weights = np.array([5/9, 8/9, 5/9])
    elif n == 4:
        nodes = np.array([-np.sqrt((3/7) + 2/7 * np.sqrt(6/5)), np.sqrt((3/7) + 2/7 * np.sqrt(6/5)), -np.sqrt((3/7) - 2/7 * np.sqrt(6/5)), np.sqrt((3/7) - 2/7 * np.sqrt(6/5))])
        weights = np.array([(18 - np.sqrt(30))/36, (18 - np.sqrt(30))/36, (18 + np.sqrt(30))/36, (18 + np.sqrt(30))/36])
    elif n == 5:
        nodes = np.array([0, -1/3 * np.sqrt(5 - 2 * np.sqrt(10/7)), 1/3 * np.sqrt(5 - 2 * np.sqrt(10/7)), -1/3 * np.sqrt(5 + 2 * np.sqrt(10/7)), 1/3 * np.sqrt(5 + 2 * np.sqrt(10/7))])
        weights = np.array([128/225, (322 + 13 * np.sqrt(70))/900, (322 + 13 * np.sqrt(70))/900, (322 - 13 * np.sqrt(70))/900, (322 - 13 * np.sqrt(70))/900])
    else:
        raise NotImplementedError("Gaussian quadrature with more than 5 points is not implemented here.")
    return nodes, weights

def compute_intermediate_points_and_normals_gaussian(ordered_boundary_vertices, normals, points_per_segment):
    intermediate_points = []
    segment_normals = []
    weights = []
    
    for i in range(len(ordered_boundary_vertices) - 1):
        # Calculate start and end of the current segment
        start_point = ordered_boundary_vertices[i]
        end_point = ordered_boundary_vertices[(i + 1) % len(ordered_boundary_vertices)]
        
        # Gaussian nodes and weights for the current segment
        nodes, segment_weights = gaussian_quadrature(points_per_segment)
        
        # Transform Gaussian nodes from [-1, 1] to [start_point, end_point]
        for node, weight in zip(nodes, segment_weights):
            t = 0.5 * (node + 1)  # Transform from [-1, 1] to [0, 1]
            intermediate_point = (1 - t) * start_point + t * end_point
            intermediate_points.append(intermediate_point)
            segment_normals.append(normalize(normals[i]))
            weights.append(weight * np.linalg.norm(end_point - start_point) / 2)  # Scaling weight by the segment length / 2
    
    return np.array(intermediate_points), np.array(segment_normals), np.array(weights)


def main():
    if len(sys.argv) != 6:
        print("Usage: python script.py file.mesh points_per_segment namefileQuadraturePoints namefileQuadratureWeights namefileNormals")
        sys.exit(1)
    
    input_mesh = sys.argv[1]
    points_per_segment = int(sys.argv[2])
    output_file_points = sys.argv[3]
    output_file_weights = sys.argv[4]
    output_file_normals = sys.argv[5]
    
    output_directory = "/Users/saraavesani/fmca/Sara_tests/data/"
    full_path_points = os.path.join(output_directory, output_file_points)
    full_path_weights = os.path.join(output_directory, output_file_weights)
    full_path_normals = os.path.join(output_directory, output_file_normals)


    # Read mesh
    v01, t, f = igl.read_mesh(input_mesh)
    v = 2 * v01 - 1
    V_2d = v[:, :2]
    boundary_vertices = igl.boundary_loop(f)

    ordered_boundary_vertices = reorder_boundary_vertices(V_2d, boundary_vertices)
    normals = compute_outer_normals(ordered_boundary_vertices)
    intermediate_points, segment_normals, weights = compute_intermediate_points_and_normals_gaussian(ordered_boundary_vertices, normals, points_per_segment)

    # Save outputs
    np.savetxt(full_path_points, intermediate_points, fmt="%.16f")
    np.savetxt(full_path_weights, weights, fmt="%.16f")
    np.savetxt(full_path_normals, segment_normals, fmt="%.16f")

if __name__ == "__main__":
    main()
