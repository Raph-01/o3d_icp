import open3d as o3d
import numpy as np

def read_point_cloud_from_numpy():
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    
    return point_cloud

# Example Usage:
# Reading the point cloud from the saved .npy file
result_pc_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/TestFolder/ransac_test_results_pc_400.npy"
model_point_cloud = o3d.geometry.PointCloud()
target_point_cloud = o3d.geometry.PointCloud()
vertices = np.load(result_pc_path, allow_pickle=True)
excel_rows = [10]
origin = np.array([0, 0, 0])
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=origin)  # Size=0.3m (30cm)
    
    # X-axis -> Red (points along the X direction)
    # Y-axis -> Green (points along the Y direction)
    # Z-axis -> Blue (points along the Z direction)

for row in excel_rows:
    model_point_cloud.points = o3d.utility.Vector3dVector(vertices[row-1,0])
    target_point_cloud.points = o3d.utility.Vector3dVector(vertices[row-1,1])
    model_point_cloud.paint_uniform_color([0, 0, 1]) # Blue
    target_point_cloud.paint_uniform_color([1, 0, 0]) # Red
    o3d.visualization.draw_geometries([model_point_cloud,target_point_cloud, axis])