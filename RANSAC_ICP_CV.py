# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:49:36 2024

@author: Raphael
https://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html
"""

# Import Statements
import pyrealsense2 as rs
import numpy as np
import time
import math
import threading
import sys
import open3d as o3d
import cv2
import pickle
import os

def read_recorded_data():
    my_frame_data = savedrecording[recording_frameNUM] # ["Depth Verts", "Time Stamp"]
    recording_frameNUM += 1
    
    if recording_frameNUM > len(savedrecording)-1:
        try:
            recording_segNUM += 1
            savedrecording = np.load(read_video_path+str(recording_segNUM)+".npy", allow_pickle=True)
            print(recording_segNUM)
            recording_frameNUM = 1
            end_recording = False
        except Exception: # next segment doesnt exist -> end of recording reached
            end_recording = True
    return my_frame_data, not(end_recording)

def read_live_data():
    # Wait for a coherent pair of frames: depth and color       
    frames = pipeline.wait_for_frames()
    
    frame_timestamp = frames.get_timestamp()
    depth_frame = frames.get_depth_frame()
    depth_frame = decimate.process(depth_frame) # The Decimate reduces resolution by 2 folds

    points = pc.calculate(depth_frame)

    # Pointcloud data to array
    v = points.get_vertices()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    return [verts, frame_timestamp]

def record_data(data):
    recording.append(data)
    
    if len(recording) >=10:
        recording = np.asarray(recording, dtype=object)
        np.save(record_video_path+str(recording_segNUM)+".npy", recording, allow_pickle=True)
        recording_segNUM += 1
        recording = [recording_header]

"""
USER INPUT START
"""
Select_Mode = "Live" # "Read" "Live" "Record Live"
control_fps = False  # [BOOL] Relevant if "Read" 
platform = "Orin" # "Orin" "Raph" "Xavier"
"""
USER INPUT END
"""

# Set script states
read_recording, record_enabled = (True, False) if Select_Mode == "Read" else (False, True) if Select_Mode == "Record Live" else (False, False)

# Paths
if platform == "Orin":
    ## Video: "/home/spot-vision/Documents/o3d_icp/recording/myrecording"
    ## Video Intrinsics: "/home/spot-vision/Documents/o3d_icp/recording/myrecordingintrinsics.npy"
    read_video_path = "/home/spot-vision/Documents/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
    read_video_intrinsics_path = "/home/spot-vision/Documents/o3d_icp/recording/myrecordingintrinsics.npy" # Path String with file name
    record_video_path = "/home/spot-vision/Documents/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
    record_video_intrinsics_path = "/home/spot-vision/Documents/o3d_icp/recording/myrecordingintrinsics.npy" # Path String with file name
    pose_estimation_path = "/home/spot-vision/Documents/o3d_icp/pose_estimation/pose_estimation.csv" # Path String with file name with extension
    model_path = "/home/spot-vision/Documents/o3d_icp/Target_v10.stl"
    model_pc_path = "/home/spot-vision/Documents/o3d_icp/Target_v10_pc.npy"
elif platform == "Raph":
    read_video_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
    read_video_intrinsics_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/recording/myrecordingintrinsics.npy" # Path String with file name
    record_video_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
    record_video_intrinsics_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/recording/myrecordingintrinsics.npy" # Path String with file name
    pose_estimation_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/pose_estimation/pose_estimation.csv" # Path String with file name with extension
    model_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/Target_v10.stl"
    model_pc_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/Target_v10_pc.npy"
elif platform == "Xavier":
    print("Not set for Xavier")
    exit(0)

# RANSAC and ICP parameters
voxel_size = 0.05  # (m) Change depending on point cloud size
radius_normal = voxel_size * 2
radius_feature = voxel_size * 10
distance_threshold_RANSAC = voxel_size * 3 # 1.5
distance_threshold_ICP = voxel_size * 2 # 0.4

# Model Initialization
## Convert Model Mesh to point cloud
try:
    model_pcd = np.load(model_pc_path) # Scaled
except:
    print("Could not find saved model point cloud. Proceeding with loading STL")
    origin = np.array([0, 0, 0])
    model = o3d.io.read_triangle_mesh(model_path)
    model_pcd = model.sample_points_poisson_disk(number_of_points=5000)
    model_pcd.scale(0.001, origin)
    np.save(model_pc_path, model_pcd, allow_pickle=True)
finally:
    model_down = model_pcd.voxel_down_sample(voxel_size) # Downsample and estimate normals for point cloud
    model_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # Compute FPFH features
    model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        model_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# Camera frame to SPOT frame transform
RotZ90 = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
RotX90 = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
transform_camera = np.dot(RotZ90,RotX90)

# Configure depth stream
if not read_recording:    
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # Realsense Objects Initialization
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** 1)

    if len(device) == 0:
        print("Connect L515 Camera")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

# Script
target_pcd = o3d.geometry.PointCloud()
recording_header = ["Depth Verts", "Time Stamp"]
recording = [recording_header]
recording_frameNUM = 1 # If recording frame iterator to replace camera stream. Skip 0 and 1 (column title + 1st frame)
recording_segNUM = 1 # Video segment number, used for both read and write recording
transform = None
transform_array = []

try:
    run_loop = True
    while run_loop:       
        # Live Camera
        if not read_recording:
            print("Loop Live")
            frame_data = read_live_data()
            if record_enabled:
                record_data()

        # Reading a Recorded Video
        elif read_recording:
            print("Loop Read")
            frame_data, run_loop = read_recorded_data()

        verts = frame_data[0]

        # Pointcloud Vertices (verts) processing
        # TO CHECK IF BETTER LOGIC
        indicesZ = np.where(verts[:,2] > 2)[0] # find where z > 1.7 m
        verts = np.delete(verts, indicesZ, axis=0)
        
        target_pcd.points = o3d.utility.Vector3dVector(verts)
        
        # TO CHECK IF NEEDED
        target_pcd = target_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
        target_down = target_pcd.voxel_down_sample(voxel_size) # Downsample and estimate normals for point cloud
        
        try:
            target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))  
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        except:
            next
            
        # RANSAC
        transform_RANSAC = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            model_down, target_down, model_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold_RANSAC,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_RANSAC),
            ], criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),)

        # ICP
    #    result = o3d.pipelines.registration.registration_icp(
    #       model, target, distance_threshold, transform.transformation,
    #        o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transform_ICP = o3d.pipelines.registration.registration_icp(
                    model_pcd, target_pcd, distance_threshold_ICP, transform.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

        if transform is None:
            transform = execute_global_registration(model_down, target_down, model_fpfh, target_fpfh, voxel_size)
            print("RANSAC")

        # ICP REFINEMENT
        transform_fitness = transform.fitness
        transform_rmse = transform.inlier_rmse

        # Check if a valid transformation was found
        if best_transformation is not None:
            #print(transform.transformation)
            transform_relative = np.dot(transform_camera,transform.transformation) # Transform from camera frame to inertial standard
            #print(transform_relative)
            transform_package = [transform_relative[0,3],transform_relative[1,3], transform_relative[1,1]] # X, Y, rotation of X-axis about Z (rad) [relative chaser-target]
            print(transform_package)
            transform_array.append([frame_timestamp, transform_package[0], transform_package[1],transform_package[2], transform_fitness, transform_rmse]) # SEND TRANSFORM ARRAY TO GNC
            print("ICP SUCCESS")
        else:
            transform = None # TO REVIEW LOGIC, DO WE WANT TO JUMP STRAIGHT BACK TO RANSAC OR TRY ICP AGAIN. Do we send data to GNC anyway? With fitness score?
            print("ICP FAIL")

        if read_recording and control_fps:
            time.sleep(delta_T/1000)

finally:
    np.savetxt(pose_estimation_path, transform_array, delimiter=',')
    if not read_recording:
        profile.stop()
        pipeline.stop()
        profile = pipeline = None

    if record_enabled:
        recording = np.asarray(recording, dtype=object)
        np.save(record_video_path+str(recording_segNUM)+".npy", recording, allow_pickle=True)
