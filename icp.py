
import open3d as o3d 

import plotly.graph_objects as go
import numpy as np

# Load the Bunny mesh
bunny = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(bunny.path)

#compute vertex normals
mesh.compute_vertex_normals()

# Sample points from the mesh
pcd = mesh.sample_points_poisson_disk(number_of_points=1000)

#visualize pc
#o3d.visualization.draw_geometries([pcd])

# Convert Open3D point cloud to NumPy array
xyz = np.asarray(pcd.points)

# Create a 3D scatter plot
scatter = go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers', marker=dict(size=1))
fig = go.Figure(data=[scatter])
fig.show()

# Apply an arbitrary rotation to the original point cloud
R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 4, np.pi / 4, np.pi / 4))
rotated_pcd = pcd.rotate(R, center=(0, 0, 0))

# Convert Open3D point cloud to NumPy array
xyz_rot = np.asarray(rotated_pcd.points)

# Create a 3D scatter plot
scatter = go.Scatter3d(x=xyz_rot[:, 0], y=xyz_rot[:, 1], z=xyz[:, 2], mode='markers', marker=dict(size=1.0))
fig = go.Figure(data=[scatter])
fig.show()

# Use ICP to find the rotation
threshold = 0.02  # Distance threshold
trans_init = np.identity(4)  # Initial guess (identity matrix)
trans_init[:3, :3] = R  # We set the initial rotation to the known rotation
reg_p2p = o3d.pipelines.registration.registration_icp(
    source=rotated_pcd, target=pcd, max_correspondence_distance=threshold,
    init=trans_init
)

# Extract the rotation matrix from the transformation matrix
estimated_rotation_matrix = reg_p2p.transformation[:3, :3]
rotation_matrix = reg_p2p.transformation[:3, :3]
print(estimated_rotation_matrix)
print("Estimated rotation matrix:")
print(rotation_matrix)

# Apply the inverse of the estimated rotation to the rotated point cloud
inverse_rotation_matrix = np.linalg.inv(estimated_rotation_matrix)
rotated_back_pcd = rotated_pcd.rotate(inverse_rotation_matrix, center=(0, 0, 0))

# Compare the original point cloud to the one rotated back to its original state
# We can use the mean squared error (MSE) between corresponding points as a metric
original_points = np.asarray(pcd.points)
rotated_back_points = np.asarray(rotated_back_pcd.points)

print("rotated_back_points",rotated_back_points.shape)
mse = np.mean(np.linalg.norm(original_points - rotated_back_points, axis=1) ** 2)

# Check if the MSE is below a certain tolerance
tolerance = 1e-6
if mse < tolerance:
    print(f"Test passed: MSE = {mse}")
else:
    print(f"Test failed: MSE = {mse}")
