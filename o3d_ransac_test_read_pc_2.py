import open3d as o3d
import numpy as np

def read_point_cloud_from_numpy():
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    
    return point_cloud

# Example Usage:
# Reading the point cloud from the saved .npy file
result_pc_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/Data/test_backup/"
model_point_cloud = o3d.geometry.PointCloud()
target_point_cloud = o3d.geometry.PointCloud()

# frames selection
""" INPUT """
test_num = 9
start = 371
end = 372
""" END INPUT"""

frames = np.linspace(start,end,end-start+1, dtype=int)
origin = np.array([0, 0, 0])
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=origin)  # Size=0.3m (30cm)
    
    # X-axis -> Red (points along the X direction)
    # Y-axis -> Green (points along the Y direction)
    # Z-axis -> Blue (points along the Z direction)

for sample in range(25):
    for frame in frames:
        print(frame)
        vertices = np.load(f"{result_pc_path}test_{test_num}/transform_result_{sample}_{frame}.npy", allow_pickle=True)
        model_point_cloud.points = o3d.utility.Vector3dVector(vertices[0])
        target_point_cloud.points = o3d.utility.Vector3dVector(vertices[1])
        model_point_cloud.paint_uniform_color([0, 0, 1]) # Blue
        target_point_cloud.paint_uniform_color([1, 0, 0]) # Red
        o3d.visualization.draw_geometries([model_point_cloud,target_point_cloud, axis])